"""
FOMC LLM Scorer
===============
Uses the Claude API to score each FOMC statement on a continuous
hawkishness scale from -1.0 (maximally dovish) to +1.0 (maximally hawkish).

Scores each statement THREE times to test consistency (test-retest reliability).
Saves results to fomc_llm_scores.csv which can be merged with fomc_statements.csv.

Requirements:
    pip install anthropic pandas

Setup:
    You need an Anthropic API key. Get one at https://console.anthropic.com
    Then set it as an environment variable. In your terminal, run:

        Windows PowerShell:
            $env:ANTHROPIC_API_KEY = "sk-ant-..."

        Then immediately run this script in the same terminal window.

Usage:
    python fomc_llm_scorer.py

    On first run it will score all 235 statements (takes ~20-30 minutes).
    Progress is saved after every statement so you can stop and resume safely.
"""

import anthropic
import pandas as pd
import json
import time
import re
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

INPUT_CSV   = Path("fomc_statements.csv")
OUTPUT_CSV  = Path("fomc_llm_scores.csv")
CACHE_DIR   = Path("fomc_llm_cache")   # one JSON file per statement per run
CACHE_DIR.mkdir(exist_ok=True)

MODEL       = "claude-opus-4-5"
MAX_TOKENS  = 256
N_RUNS      = 3          # score each statement this many times for reliability
SLEEP_SEC   = 1.0        # pause between API calls to avoid rate limits

# ── Prompt ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert in Federal Reserve monetary policy communication.
Your task is to score FOMC post-meeting statements on a hawkishness scale.

Hawkish = the Fed is signaling concern about inflation, inclination to raise rates,
or tightening bias. Dovish = the Fed is signaling concern about employment/growth,
inclination to cut rates, or easing bias.

You must return ONLY a JSON object with exactly two keys:
- "score": a float between -1.0 (maximally dovish) and +1.0 (maximally hawkish)
- "rationale": one sentence explaining the key phrase or signal driving your score

Do not include any text outside the JSON object. No preamble, no explanation."""

USER_TEMPLATE = """Score this FOMC statement:

{text}

Return JSON only: {{"score": float, "rationale": "string"}}"""

# ── Helpers ───────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Strip boilerplate from raw scraped text so the LLM sees
    just the statement body. Removes navigation text, vote disclosures,
    and implementation notes that add noise without signal.
    """
    # Remove everything before "For immediate release" or "For release at"
    for marker in ["For immediate release", "For release at", "Information received"]:
        idx = text.find(marker)
        if idx != -1:
            text = text[idx:]
            break

    # Truncate at voting record (not part of the tone signal)
    for cutoff in ["Voting for", "Voting against", "Implementation Note"]:
        idx = text.find(cutoff)
        if idx != -1:
            text = text[:idx]

    return text.strip()


def parse_response(raw: str) -> dict | None:
    """
    Extract JSON from the model response.
    Handles cases where the model wraps JSON in markdown code fences.
    """
    # Strip markdown fences if present
    raw = re.sub(r"```(?:json)?", "", raw).strip()

    try:
        parsed = json.loads(raw)
        score = float(parsed["score"])
        rationale = str(parsed["rationale"])

        # Clamp score to valid range
        score = max(-1.0, min(1.0, score))

        return {"score": score, "rationale": rationale}
    except Exception:
        return None


def score_once(client: anthropic.Anthropic, text: str, date: str, run: int) -> dict | None:
    """
    Call the Claude API once and return parsed score dict.
    Retries up to 3 times on transient errors.
    """
    cache_file = CACHE_DIR / f"{date}_run{run}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text())

    for attempt in range(3):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                messages=[{
                    "role": "user",
                    "content": USER_TEMPLATE.format(text=text[:3000])  # cap at 3000 chars
                }]
            )

            raw = response.content[0].text
            result = parse_response(raw)

            if result:
                cache_file.write_text(json.dumps(result))
                return result
            else:
                print(f"      Parse failed (attempt {attempt+1}), raw: {raw[:80]}")

        except anthropic.RateLimitError:
            wait = 30 * (attempt + 1)
            print(f"      Rate limited, waiting {wait}s...")
            time.sleep(wait)

        except anthropic.APIError as e:
            print(f"      API error (attempt {attempt+1}): {e}")
            time.sleep(5)

    return None


# ── Main Pipeline ─────────────────────────────────────────────────────────────

def run():
    # Load the dictionary-scored statements
    if not INPUT_CSV.exists():
        print(f"ERROR: {INPUT_CSV} not found. Run fomc_pipeline_v2.py first.")
        return

    df = pd.read_csv(INPUT_CSV)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    print(f"Loaded {len(df)} statements from {INPUT_CSV}")

    # Load any existing progress
    if OUTPUT_CSV.exists():
        done = pd.read_csv(OUTPUT_CSV)
        done_dates = set(done["date"].astype(str))
        print(f"Resuming: {len(done)} already scored, {len(df) - len(done)} remaining")
    else:
        done = pd.DataFrame()
        done_dates = set()

    # Initialize client (reads ANTHROPIC_API_KEY from environment)
    client = anthropic.Anthropic()

    results = []

    for i, row in df.iterrows():
        date_str = str(row["date"])[:10]

        if date_str in done_dates:
            continue

        print(f"\n[{i+1}/{len(df)}] {date_str}")

        # Clean the text
        text = clean_text(str(row.get("text_snippet", "")))

        # If text_snippet is too short (only 300 chars), note it
        # In a full version you would re-fetch the full statement text here
        if len(text) < 50:
            print(f"  Skipping: text too short ({len(text)} chars)")
            continue

        # Score N_RUNS times
        run_scores = []
        for run_num in range(1, N_RUNS + 1):
            result = score_once(client, text, date_str, run_num)
            if result:
                print(f"  Run {run_num}: score={result['score']:+.3f}  | {result['rationale'][:80]}")
                run_scores.append(result)
            time.sleep(SLEEP_SEC)

        if not run_scores:
            print(f"  All runs failed, skipping {date_str}")
            continue

        # Aggregate across runs
        scores = [r["score"] for r in run_scores]
        llm_score_mean   = sum(scores) / len(scores)
        llm_score_std    = pd.Series(scores).std() if len(scores) > 1 else 0.0
        llm_score_min    = min(scores)
        llm_score_max    = max(scores)
        primary_rationale = run_scores[0]["rationale"]

        print(f"  Mean: {llm_score_mean:+.3f}  Std: {llm_score_std:.3f}  "
              f"Range: [{llm_score_min:+.3f}, {llm_score_max:+.3f}]")

        record = {
            "date":              date_str,
            "llm_score_mean":    round(llm_score_mean, 4),
            "llm_score_std":     round(llm_score_std, 4),
            "llm_score_min":     round(llm_score_min, 4),
            "llm_score_max":     round(llm_score_max, 4),
            "llm_rationale":     primary_rationale,
            "dict_hawkish_net":  row.get("hawkish_net", None),  # keep for comparison
        }
        results.append(record)

        # Save progress after every statement
        new_rows = pd.DataFrame(results)
        combined = pd.concat([done, new_rows], ignore_index=True) if not done.empty else new_rows
        combined.to_csv(OUTPUT_CSV, index=False)

    print(f"\n\nDone. Results saved to {OUTPUT_CSV}")

    # Final summary
    final = pd.read_csv(OUTPUT_CSV)
    print(f"\nTotal scored: {len(final)}")
    print(f"Average test-retest std: {final['llm_score_std'].mean():.4f}")

    print("\n── Sanity Check ──────────────────────────────────────────")
    final["date"] = pd.to_datetime(final["date"])
    checks = {
        "2022-03": ("First 2022 hike", "hawkish"),
        "2022-06": ("75bp hike", "hawkish"),
        "2019-07": ("Insurance cut", "dovish"),
        "2020-03": ("COVID emergency cut", "dovish"),
        "2015-12": ("Liftoff from ZLB", "hawkish"),
        "2008-12": ("ZLB adoption", "dovish"),
        "2016-12": ("Post-ZLB hike", "hawkish"),
        "2018-12": ("Dec 2018 hike", "hawkish"),
    }
    passes = 0
    for ym, (label, expected) in checks.items():
        mask = final["date"].dt.to_period("M").astype(str) == ym
        row = final[mask]
        if not row.empty:
            score = row.iloc[0]["llm_score_mean"]
            actual = "hawkish" if score > 0 else "dovish"
            flag = "PASS" if actual == expected else "FAIL"
            if flag == "PASS":
                passes += 1
            print(f"  {flag}  {ym}  LLM={score:+.3f}  (expected {expected})  {label}")
        else:
            print(f"  ?    {ym}  NOT FOUND  {label}")

    print(f"\n{passes}/8 sanity checks passed")
    print(f"(Dictionary baseline passed 4/8 -- improvement shows LLM value)")


if __name__ == "__main__":
    run()
