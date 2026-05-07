"""
FOMC Gap Filler — scrapes missing 2016-2020 statements
Run this from your Fomc_Research folder, then it merges into fomc_statements.csv
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import time
from pathlib import Path

BASE_URL  = "https://www.federalreserve.gov"
CACHE_DIR = Path("fomc_cache")
CACHE_DIR.mkdir(exist_ok=True)
HEADERS   = {"User-Agent": "Mozilla/5.0 (academic research)"}

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


def get(url):
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        return r if r.status_code == 200 else None
    except Exception as e:
        print(f"  Error: {e}")
        return None


def fetch_text(url):
    cache_file = CACHE_DIR / re.sub(r"[^\w]", "_", url)[:200]
    if cache_file.exists():
        return cache_file.read_text(encoding="utf-8")
    resp = get(url)
    if not resp:
        return ""
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()
    main = (soup.find("div", {"id": "article"}) or
            soup.find("div", {"class": "col-xs-12"}) or
            soup.find("article") or soup.find("main") or soup.body)
    text = main.get_text(separator=" ", strip=True) if main else ""
    text = re.sub(r"\s+", " ", text).strip()
    cache_file.write_text(text, encoding="utf-8")
    time.sleep(0.3)
    return text


def load_lm():
    df = pd.read_csv("lm_dictionary.csv", low_memory=False)
    df.columns = [c.strip().upper() for c in df.columns]
    lm = {}
    for key, col in [("negative","NEGATIVE"),("positive","POSITIVE"),("uncertainty","UNCERTAINTY")]:
        lm[key] = set(df.loc[df[col]!=0,"WORD"].str.upper()) if col in df.columns else set()
    return lm


def score(text, lm):
    words = re.findall(r"\b[a-z]+\b", text.lower())
    n = len(words)
    if n == 0:
        return dict(word_count=0, lm_negative=0, lm_positive=0,
                    lm_uncertainty=0, lm_net=0,
                    hawkish_count=0, dovish_count=0, hawkish_net=0)
    wu = [w.upper() for w in words]
    lm_neg = sum(1 for w in wu if w in lm["negative"])
    lm_pos = sum(1 for w in wu if w in lm["positive"])
    lm_unc = sum(1 for w in wu if w in lm["uncertainty"])
    hawk   = sum(1 for w in words if w in HAWKISH)
    dove   = sum(1 for w in words if w in DOVISH)
    return dict(word_count=n, lm_negative=lm_neg/n, lm_positive=lm_pos/n,
                lm_uncertainty=lm_unc/n, lm_net=(lm_pos-lm_neg)/n,
                hawkish_count=hawk, dovish_count=dove,
                hawkish_net=(hawk-dove)/n*100)


def get_missing_links():
    """
    For 2016-2020 the Fed uses year-specific historical pages,
    same format as earlier years.
    """
    links = []
    for year in range(2016, 2021):
        url = f"{BASE_URL}/monetarypolicy/fomchistorical{year}.htm"
        print(f"Fetching {year} index... ", end="")
        resp = get(url)
        if not resp:
            print("failed")
            time.sleep(0.5)
            continue

        soup = BeautifulSoup(resp.text, "html.parser")
        year_links = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            text = a.get_text(strip=True).lower()
            if href.lower().endswith(".pdf"):
                continue
            if ("statement" in text) and re.search(r"\d{8}", href):
                full_url = BASE_URL + href if href.startswith("/") else href
                match = re.search(r"(\d{8})", href)
                if match:
                    raw = match.group(1)
                    date = f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}"
                    year_links.append({"date": date, "url": full_url})

        print(f"{len(year_links)} found")
        links += year_links
        time.sleep(0.5)

    return links


def run():
    # Load existing data
    existing = pd.read_csv("fomc_statements.csv")
    existing["date"] = pd.to_datetime(existing["date"])
    existing_dates = set(existing["date"].dt.strftime("%Y-%m-%d"))
    print(f"Existing: {len(existing)} statements, latest: {existing['date'].max().date()}")

    # Find missing 2016-2020 links
    print("\nLooking for missing 2016-2020 statements...")
    links = get_missing_links()

    # Filter to only truly missing dates
    new_links = [l for l in links if l["date"] not in existing_dates]
    print(f"\n{len(new_links)} new statements to fetch")

    if not new_links:
        print("Nothing missing — you're good!")
        return

    # Load LM dict and score new statements
    lm = load_lm()
    new_records = []
    for i, item in enumerate(sorted(new_links, key=lambda x: x["date"])):
        print(f"[{i+1}/{len(new_links)}] {item['date']}")
        text = fetch_text(item["url"])
        if not text:
            continue
        s = score(text, lm)
        new_records.append({"date": item["date"], "url": item["url"],
                            "text_snippet": text[:300], **s})

    # Merge and save
    new_df = pd.DataFrame(new_records)
    new_df["date"] = pd.to_datetime(new_df["date"])
    combined = pd.concat([existing, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)
    combined.to_csv("fomc_statements.csv", index=False)

    print(f"\n✓ Done. Total statements: {len(combined)}")
    print(f"  Date range: {combined['date'].min().date()} to {combined['date'].max().date()}")

    # Quick check on newly added years
    print("\nStatements per year (new data):")
    new_years = combined[combined["date"].dt.year.between(2016, 2020)]
    print(new_years.groupby(new_years["date"].dt.year).size().to_string())


if __name__ == "__main__":
    run()
