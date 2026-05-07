"""
FOMC Statement Scraper + Loughran-McDonald Sentiment Scorer  v2
===============================================================
Fixes from v1:
  - Correctly scrapes 2016-2024 statements from the Fed's calendar page
  - Drops PDF links (those are policy documents, not statements)
  - Better text extraction that strips nav/header boilerplate
  - Covers 1994-2024

Requirements:
    pip install requests beautifulsoup4 pandas
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import time
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

BASE_URL   = "https://www.federalreserve.gov"
OUTPUT_PATH = Path("fomc_statements.csv")
CACHE_DIR   = Path("fomc_cache")
CACHE_DIR.mkdir(exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0 (academic research)"}

# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_date_from_url(href: str) -> str | None:
    match = re.search(r"(\d{8})", href)
    if match:
        raw = match.group(1)
        return f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}"
    return None


def is_pdf(url: str) -> bool:
    return url.lower().endswith(".pdf")


def get(url: str) -> requests.Response | None:
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        return r if r.status_code == 200 else None
    except Exception as e:
        print(f"  Request failed: {e}")
        return None

# ── Part 1: Link Collection ───────────────────────────────────────────────────

def collect_links() -> list[dict]:
    links = []

    # ── Recent statements: 2016–present ──────────────────────────────────────
    # The Fed's calendar page lists upcoming + past meetings for recent years
    print("Scraping recent statements (2016–2024)...")
    calendar_url = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
    resp = get(calendar_url)
    if resp:
        soup = BeautifulSoup(resp.text, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            text = a.get_text(strip=True).lower()
            # Statement links look like /newsevents/pressreleases/monetary20240131a.htm
            if (
                "monetary" in href
                and re.search(r"\d{8}", href)
                and "pressreleases" in href
                and not is_pdf(href)
                and ("statement" in text or href.endswith("a.htm"))
            ):
                full_url = BASE_URL + href if href.startswith("/") else href
                date = parse_date_from_url(href)
                if date and date >= "2016-01-01":
                    links.append({"date": date, "url": full_url})

    # ── Historical statements: 1994–2015 ─────────────────────────────────────
    print("Scraping historical statements (1994–2015)...")
    for year in range(1994, 2016):
        url = f"{BASE_URL}/monetarypolicy/fomchistorical{year}.htm"
        resp = get(url)
        if not resp:
            print(f"  {year}: no response, skipping")
            time.sleep(0.5)
            continue

        soup = BeautifulSoup(resp.text, "html.parser")
        year_links = []

        for a in soup.find_all("a", href=True):
            href = a["href"]
            text = a.get_text(strip=True).lower()

            if is_pdf(href):
                continue  # skip PDFs entirely

            is_statement_text = "statement" in text or "press release" in text
            is_monetary_url   = "monetary" in href or "boarddocs" in href or "press/general" in href
            has_date          = bool(re.search(r"\d{8}", href))

            if is_statement_text and is_monetary_url and has_date:
                full_url = BASE_URL + href if href.startswith("/") else href
                date = parse_date_from_url(href)
                if date:
                    year_links.append({"date": date, "url": full_url})

        print(f"  {year}: {len(year_links)} links")
        links += year_links
        time.sleep(0.4)

    return links


def deduplicate(links: list[dict]) -> list[dict]:
    """Keep one link per date; prefer pressreleases URLs over boarddocs."""
    by_date: dict[str, dict] = {}
    for item in links:
        d = item["date"]
        if d not in by_date:
            by_date[d] = item
        else:
            # Prefer the cleaner pressreleases URL
            if "pressreleases" in item["url"] and "pressreleases" not in by_date[d]["url"]:
                by_date[d] = item
    return sorted(by_date.values(), key=lambda x: x["date"])

# ── Part 2: Text Extraction ───────────────────────────────────────────────────

def fetch_text(url: str) -> str:
    cache_file = CACHE_DIR / re.sub(r"[^\w]", "_", url)[:200]

    if cache_file.exists():
        return cache_file.read_text(encoding="utf-8")

    resp = get(url)
    if not resp:
        return ""

    soup = BeautifulSoup(resp.text, "html.parser")

    # Remove boilerplate elements
    for tag in soup(["script", "style", "nav", "footer", "header",
                     "noscript", ".skip-nav"]):
        tag.decompose()

    # For newer Fed pages, the statement is inside article or main
    main = soup.find("div", {"id": "article"}) or \
           soup.find("div", {"class": "col-xs-12"}) or \
           soup.find("article") or \
           soup.find("main") or \
           soup.body

    text = main.get_text(separator=" ", strip=True) if main else ""
    text = re.sub(r"\s+", " ", text).strip()

    cache_file.write_text(text, encoding="utf-8")
    time.sleep(0.3)
    return text

# ── Part 3: Scoring ───────────────────────────────────────────────────────────

def load_lm_dictionary() -> dict[str, set[str]]:
    lm_cache = Path("lm_dictionary.csv")

    if not lm_cache.exists():
        print("Downloading Loughran-McDonald dictionary...")
        url = (
            "https://raw.githubusercontent.com/Loughran-McDonald/"
            "Master-Dictionary/master/Loughran-McDonald_MasterDictionary_1993-2021.csv"
        )
        resp = requests.get(url, timeout=30)
        lm_cache.write_bytes(resp.content)

    df = pd.read_csv(lm_cache, low_memory=False)
    df.columns = [c.strip().upper() for c in df.columns]

    lm = {}
    for key, col in [("negative", "NEGATIVE"), ("positive", "POSITIVE"),
                     ("uncertainty", "UNCERTAINTY")]:
        if col in df.columns:
            lm[key] = set(df.loc[df[col] != 0, "WORD"].str.upper())
        else:
            lm[key] = set()

    print(f"LM dictionary: {sum(len(v) for v in lm.values())} entries")
    return lm


# Domain-specific hawkish/dovish vocabulary for Fed communications
HAWKISH = {
    "tighten", "tightening", "restrictive", "restriction", "hawkish",
    "inflationary", "overheating", "elevated", "persistent", "vigilant",
    "increase", "increases", "increased", "raise", "raises", "raised",
    "hike", "hikes", "hiking", "further", "additional", "concerns",
    "upside", "risks", "firming",
}

DOVISH = {
    "accommodative", "supportive", "stimulus", "stimulative",
    "easing", "ease", "cut", "cuts", "cutting", "reduce", "reduction",
    "pause", "patient", "gradual", "below", "weak", "weakness", "slack",
    "transitory", "temporary", "stable", "anchored", "downside",
    "employment", "maximum", "shortfall",
}


def score(text: str, lm: dict[str, set[str]]) -> dict:
    words = re.findall(r"\b[a-z]+\b", text.lower())
    n = len(words)
    if n == 0:
        return dict(word_count=0, lm_negative=0, lm_positive=0,
                    lm_uncertainty=0, lm_net=0,
                    hawkish_count=0, dovish_count=0, hawkish_net=0)

    wu = [w.upper() for w in words]
    lm_neg  = sum(1 for w in wu if w in lm["negative"])
    lm_pos  = sum(1 for w in wu if w in lm["positive"])
    lm_unc  = sum(1 for w in wu if w in lm["uncertainty"])
    hawk    = sum(1 for w in words if w in HAWKISH)
    dove    = sum(1 for w in words if w in DOVISH)

    return {
        "word_count":    n,
        "lm_negative":   lm_neg / n,
        "lm_positive":   lm_pos / n,
        "lm_uncertainty": lm_unc / n,
        "lm_net":        (lm_pos - lm_neg) / n,
        "hawkish_count": hawk,
        "dovish_count":  dove,
        # PRIMARY SIGNAL: positive = hawkish, negative = dovish
        "hawkish_net":   (hawk - dove) / n * 100,
    }

# ── Part 4: Main ─────────────────────────────────────────────────────────────

def run():
    # 1. Collect and deduplicate links
    raw_links = collect_links()
    links = deduplicate(raw_links)
    print(f"\n{len(links)} unique FOMC statements found ({links[0]['date']} to {links[-1]['date']})\n")

    # 2. Load LM dictionary
    lm = load_lm_dictionary()

    # 3. Fetch and score
    records = []
    for i, item in enumerate(links):
        print(f"[{i+1:3d}/{len(links)}] {item['date']}  {item['url'][-60:]}")
        text = fetch_text(item["url"])
        if not text:
            continue
        s = score(text, lm)
        records.append({
            "date":         item["date"],
            "url":          item["url"],
            "text_snippet": text[:300],
            **s,
        })

    # 4. Save
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n✓ Saved {len(df)} statements → {OUTPUT_PATH}")

    # 5. Sanity check
    print("\n── Sanity Check ──────────────────────────────────────────")
    checks = {
        "2020-03": "COVID emergency cut       → expect very dovish (negative)",
        "2022-03": "First 2022 hike           → expect hawkish (positive)",
        "2022-06": "75bp hike                 → expect very hawkish",
        "2015-12": "Liftoff from ZLB          → expect hawkish",
        "2008-12": "ZLB adoption              → expect very dovish",
        "2019-07": "Insurance cut             → expect dovish",
    }
    for ym, label in checks.items():
        mask = df["date"].dt.to_period("M").astype(str) == ym
        row = df[mask]
        if not row.empty:
            r = row.iloc[0]
            direction = "✓ hawkish" if r["hawkish_net"] > 0 else "✓ dovish" if r["hawkish_net"] < 0 else "~ neutral"
            print(f"  {ym}  {r['hawkish_net']:+.3f}  {direction}  ← {label}")
        else:
            print(f"  {ym}  NOT FOUND")

    return df


if __name__ == "__main__":
    df = run()
