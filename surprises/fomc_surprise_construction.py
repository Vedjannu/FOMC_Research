"""
FOMC Phase 3 — Surprise Construction
======================================
Builds four hawkishness surprise measures from the scored FOMC statements
and pre-meeting market data. These surprises are the independent variables
for the predictive regressions in fomc_regressions.py.

Surprise measures constructed:
  1. dict_surprise_ar1     — AR(1) residual, dictionary score
  2. dict_surprise_market  — Market-implied residual, dictionary score
  3. llm_surprise_ar1      — AR(1) residual, LLM score
  4. llm_surprise_market   — Market-implied residual, LLM score

Training window:  1997-03-25 to 2015-12-31  (in-sample fit)
Test window:      2016-01-01 to 2026-04-29  (out-of-sample)

Models are fit on training window ONLY. Surprises for ALL 235 meetings
are then computed using those fixed coefficients. This prevents any
look-ahead bias in the regression stage.

Output:
  fomc_surprises.csv  — one row per FOMC meeting, all four surprise
                        series plus dependent variables (market returns)

Author: Ved Jannu
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression

# ── CONFIG ────────────────────────────────────────────────────────────────────

MARKET_CSV   = Path("fomc_market_data.csv")
FOMC_CSV     = Path("fomc_statements.csv")
LLM_CSV      = Path("fomc_llm_scores.csv")
OUTPUT_CSV   = Path("fomc_surprises.csv")

TRAIN_END    = "2015-12-31"   # last date of training window
TEST_START   = "2016-01-01"   # first date of test window

# ── LOAD DATA ─────────────────────────────────────────────────────────────────

def load_data():
    """Load and merge all inputs into a single analysis DataFrame."""

    if not MARKET_CSV.exists():
        raise FileNotFoundError(
            f"Cannot find {MARKET_CSV}. Run fomc_market_data.py first."
        )

    # Everything is already merged in fomc_market_data.csv
    df = pd.read_csv(MARKET_CSV, parse_dates=["fomc_date"])
    df = df.sort_values("fomc_date").reset_index(drop=True)

    # Rename llm_score to llm_score_mean for consistency with rest of script
    if "llm_score" in df.columns and "llm_score_mean" not in df.columns:
        df = df.rename(columns={"llm_score": "llm_score_mean"})

    print(f"Loaded {len(df)} rows")
    print(f"Columns: {list(df.columns)}")
    print(f"Dictionary scores available: {df['hawkish_net'].notna().sum()}")
    print(f"LLM scores available: {df['llm_score_mean'].notna().sum()}")

    return df

# ── SURPRISE CONSTRUCTION ─────────────────────────────────────────────────────

def build_ar1_surprise(df: pd.DataFrame, score_col: str,
                        prefix: str) -> pd.DataFrame:
    """
    AR(1) surprise: fit score_t = a + b * score_{t-1} on training window.
    Surprise = actual - predicted for all meetings.

    Parameters
    ----------
    df         : full DataFrame sorted by fomc_date
    score_col  : column name of the raw hawkishness score
    prefix     : prefix for output column names (e.g. 'dict', 'llm')

    Returns
    -------
    df with new columns: {prefix}_lag1, {prefix}_surprise_ar1
    """
    df = df.copy()
    lag_col     = f"{prefix}_lag1"
    surprise_col = f"{prefix}_surprise_ar1"
    fitted_col  = f"{prefix}_ar1_fitted"

    # Create lagged score
    df[lag_col] = df[score_col].shift(1)

    # Training mask — fit model only on in-sample data
    # Require both score and its lag to be non-missing
    train_mask = (
        (df["fomc_date"] <= TRAIN_END) &
        df[score_col].notna() &
        df[lag_col].notna()
    )

    X_train = df.loc[train_mask, [lag_col]].values
    y_train = df.loc[train_mask, score_col].values

    model = LinearRegression()
    model.fit(X_train, y_train)

    intercept = model.intercept_
    slope     = model.coef_[0]
    r2_train  = model.score(X_train, y_train)

    print(f"\n  AR(1) model ({prefix}):")
    print(f"    score_t = {intercept:.4f} + {slope:.4f} * score_{{t-1}}")
    print(f"    In-sample R² = {r2_train:.4f}")
    print(f"    Training observations: {train_mask.sum()}")

    # Apply model to ALL rows (in-sample and out-of-sample)
    all_mask = df[score_col].notna() & df[lag_col].notna()
    df.loc[all_mask, fitted_col]   = intercept + slope * df.loc[all_mask, lag_col]
    df.loc[all_mask, surprise_col] = df.loc[all_mask, score_col] - df.loc[all_mask, fitted_col]

    return df


def build_market_surprise(df: pd.DataFrame, score_col: str,
                           prefix: str) -> pd.DataFrame:
    """
    Market-implied surprise: fit score_t = a + b * ff_pre_t on training window.
    ff_pre is the pre-meeting effective Fed Funds rate, which proxies for
    market expectations of the policy stance going into the meeting.
    Surprise = actual score - fitted score.

    Parameters
    ----------
    df         : full DataFrame sorted by fomc_date
    score_col  : column name of the raw hawkishness score
    prefix     : prefix for output column names

    Returns
    -------
    df with new column: {prefix}_surprise_market
    """
    df = df.copy()
    surprise_col = f"{prefix}_surprise_market"
    fitted_col   = f"{prefix}_market_fitted"

    # The pre-meeting Fed Funds rate is our proxy for market expectations.
    # We use ff_pre (the level the day before the meeting).
    ff_col = "ff_pre"

    if ff_col not in df.columns:
        print(f"\n  Market surprise ({prefix}): ff_pre column not found, skipping.")
        df[surprise_col] = np.nan
        return df

    train_mask = (
        (df["fomc_date"] <= TRAIN_END) &
        df[score_col].notna() &
        df[ff_col].notna()
    )

    X_train = df.loc[train_mask, [ff_col]].values
    y_train = df.loc[train_mask, score_col].values

    model = LinearRegression()
    model.fit(X_train, y_train)

    intercept = model.intercept_
    slope     = model.coef_[0]
    r2_train  = model.score(X_train, y_train)

    print(f"\n  Market-implied model ({prefix}):")
    print(f"    score_t = {intercept:.4f} + {slope:.4f} * ff_pre_t")
    print(f"    In-sample R² = {r2_train:.4f}")
    print(f"    Training observations: {train_mask.sum()}")

    # Apply to all rows
    all_mask = df[score_col].notna() & df[ff_col].notna()
    df.loc[all_mask, fitted_col]   = intercept + slope * df.loc[all_mask, ff_col]
    df.loc[all_mask, surprise_col] = df.loc[all_mask, score_col] - df.loc[all_mask, fitted_col]

    return df


# ── DIAGNOSTICS ───────────────────────────────────────────────────────────────

def print_correlation_table(df: pd.DataFrame):
    """
    Print correlations between the four surprise measures and
    the key dependent variables (1-day yield changes).
    Gives a quick sanity check that surprises have the right signs.
    """
    surprise_cols = [
        "dict_surprise_ar1", "dict_surprise_market",
        "llm_surprise_ar1",  "llm_surprise_market",
    ]
    dep_vars = ["dgs2_ret1d", "dgs10_ret1d", "sp500_ret1d"]

    available_surprises = [c for c in surprise_cols if c in df.columns and df[c].notna().sum() > 10]
    available_deps      = [c for c in dep_vars      if c in df.columns and df[c].notna().sum() > 10]

    if not available_surprises or not available_deps:
        print("\nNot enough data for correlation table.")
        return

    print("\n── Correlation Table: Surprises vs Market Returns ────────────────")
    print("(Positive surprise = more hawkish than expected)")
    print("(Expected sign: positive with yield changes, negative with SP500)\n")

    header = f"{'':30s}" + "".join(f"{c:>14s}" for c in available_deps)
    print(header)
    print("-" * (30 + 14 * len(available_deps)))

    for sc in available_surprises:
        row_str = f"{sc:<30s}"
        for dc in available_deps:
            valid = df[[sc, dc]].dropna()
            if len(valid) > 5:
                corr = valid[sc].corr(valid[dc])
                row_str += f"{corr:>+14.3f}"
            else:
                row_str += f"{'N/A':>14s}"
        print(row_str)

    print()


def print_sample_surprises(df: pd.DataFrame):
    """Print surprise values for the eight key validation episodes."""
    print("\n── Surprise Values: Key Episodes ─────────────────────────────────")
    episodes = {
        "2022-03-16": "First 2022 hike    (expect +)",
        "2022-06-15": "75bp hike          (expect +)",
        "2020-03-15": "COVID cut          (expect -)",
        "2019-07-31": "Insurance cut      (expect -)",
        "2015-12-16": "Liftoff from ZLB   (expect +)",
        "2008-12-16": "ZLB adoption       (expect -)",
        "2016-12-14": "Post-ZLB hike      (expect +)",
        "2018-12-19": "Dec 2018 tightening(expect +)",
    }

    cols = ["dict_surprise_ar1", "llm_surprise_ar1",
            "dict_surprise_market", "llm_surprise_market"]
    available = [c for c in cols if c in df.columns]

    header = f"{'Date':<12} {'Episode':<28}" + "".join(f"{c:>22s}" for c in available)
    print(header)
    print("-" * (40 + 22 * len(available)))

    df["fomc_date_str"] = df["fomc_date"].astype(str).str[:10]
    for date_str, label in episodes.items():
        row = df[df["fomc_date_str"] == date_str]
        if not row.empty:
            r = row.iloc[0]
            val_str = f"{date_str:<12} {label:<28}"
            for c in available:
                v = r.get(c, np.nan)
                val_str += f"{v:>+22.4f}" if pd.notna(v) else f"{'nan':>22s}"
            print(val_str)
        else:
            print(f"{date_str:<12} {'NOT FOUND':<28}")


def print_window_summary(df: pd.DataFrame):
    """Print mean surprise by training vs test window."""
    print("\n── Window Summary ─────────────────────────────────────────────────")
    df["window"] = np.where(df["fomc_date"] <= TRAIN_END, "Train (1997-2015)", "Test  (2016-2026)")

    surprise_cols = [c for c in df.columns if "surprise" in c and df[c].notna().sum() > 5]

    print(f"{'':30s} {'Train mean':>12s} {'Test mean':>12s} {'Train N':>9s} {'Test N':>9s}")
    print("-" * 75)
    for sc in surprise_cols:
        train_vals = df.loc[df["window"] == "Train (1997-2015)", sc].dropna()
        test_vals  = df.loc[df["window"] == "Test  (2016-2026)", sc].dropna()
        print(
            f"{sc:<30s} "
            f"{train_vals.mean():>+12.4f} "
            f"{test_vals.mean():>+12.4f} "
            f"{len(train_vals):>9d} "
            f"{len(test_vals):>9d}"
        )


# ── MAIN ──────────────────────────────────────────────────────────────────────

def run():
    print("=" * 65)
    print("FOMC Phase 3 — Surprise Construction")
    print("=" * 65)

    # 1. Load
    df = load_data()
    print(f"\nDate range: {df['fomc_date'].min().date()} to {df['fomc_date'].max().date()}")
    print(f"Training window ends:  {TRAIN_END}")
    print(f"Test window starts:    {TEST_START}")
    n_train = (df["fomc_date"] <= TRAIN_END).sum()
    n_test  = (df["fomc_date"] >  TRAIN_END).sum()
    print(f"Training meetings: {n_train} | Test meetings: {n_test}")

    # 2. Build surprises
    print("\n── Fitting Surprise Models ────────────────────────────────────────")

    # AR(1) surprises
    df = build_ar1_surprise(df, score_col="hawkish_net",    prefix="dict")
    df = build_ar1_surprise(df, score_col="llm_score_mean", prefix="llm")

    # Market-implied surprises
    df = build_market_surprise(df, score_col="hawkish_net",    prefix="dict")
    df = build_market_surprise(df, score_col="llm_score_mean", prefix="llm")

    # 3. Diagnostics
    print_sample_surprises(df)
    print_correlation_table(df)
    print_window_summary(df)

    # 4. Save
    # Keep only the columns needed for regressions
    keep_cols = [
        "fomc_date",
        # Raw scores
        "hawkish_net", "llm_score_mean", "llm_score_std",
        # Surprise measures
        "dict_surprise_ar1", "dict_surprise_market",
        "llm_surprise_ar1",  "llm_surprise_market",
        # Pre-meeting level (for market surprise construction verification)
        "ff_pre",
        # Dependent variables — 1-day and 1-week returns
        "dgs2_ret1d",  "dgs2_ret1w",
        "dgs10_ret1d", "dgs10_ret1w",
        "sp500_ret1d", "sp500_ret1w",
        # Window label
        "window",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    result = df[keep_cols].copy()

    result.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved {len(result)} rows to {OUTPUT_CSV}")
    print(f"Columns: {list(result.columns)}")
    print("\nDone. Next step: run fomc_regressions.py")

    return result


if __name__ == "__main__":
    run()
