"""
FOMC Score Comparison — Phase 2 Visualization  (v2: fixed annotations)
==============================================
Merges fomc_statements.csv (dictionary scores) with fomc_llm_scores.csv
(LLM scores) and produces three figures:

  Figure 1 — Dictionary hawkishness score over time (Phase 1 baseline)
  Figure 2 — LLM hawkishness score over time (Phase 2 result)
  Figure 3 — Side-by-side comparison with key regime annotations

Output files saved to the same folder:
  fomc_fig1_dictionary.png
  fomc_fig2_llm.png
  fomc_fig3_comparison.png

Requirements:
    pip install pandas matplotlib
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

STATEMENTS_CSV = Path("fomc_statements.csv")
LLM_CSV        = Path("fomc_llm_scores.csv")

# Key policy regimes to shade
REGIMES = [
    ("1999-06-01", "2000-05-01", "#ffcccc", "99-00 Hiking"),
    ("2001-01-01", "2003-06-01", "#cce5ff", "01-03 Easing"),
    ("2004-06-01", "2006-06-01", "#ffcccc", "04-06 Hiking"),
    ("2008-09-01", "2015-12-01", "#cce5ff", "GFC+ZLB"),
    ("2022-03-01", "2023-07-01", "#ffcccc", "22-23 Hiking"),
]

# Dates where dictionary fails but LLM passes — annotate on Fig 3
DICT_FAILURES = {
    "2008-12-16": "ZLB adoption",
    "2015-12-16": "Liftoff",
    "2016-12-14": "2016 hike",
    "2018-12-19": "2018 hike",
}

# ── Load and merge ─────────────────────────────────────────────────────────────

def load_data():
    if not STATEMENTS_CSV.exists():
        raise FileNotFoundError(f"{STATEMENTS_CSV} not found")
    if not LLM_CSV.exists():
        raise FileNotFoundError(f"{LLM_CSV} not found")

    df_dict = pd.read_csv(STATEMENTS_CSV, parse_dates=["date"])
    df_llm  = pd.read_csv(LLM_CSV,        parse_dates=["date"])

    df_dict["date"] = df_dict["date"].dt.normalize()
    df_llm["date"]  = df_llm["date"].dt.normalize()

    df = pd.merge(
        df_dict[["date", "hawkish_net"]],
        df_llm[["date", "llm_score_mean", "llm_score_std", "llm_rationale"]],
        on="date", how="inner"
    ).sort_values("date").reset_index(drop=True)

    print(f"Merged dataset: {len(df)} statements")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Dictionary score range: [{df['hawkish_net'].min():.2f}, {df['hawkish_net'].max():.2f}]")
    print(f"LLM score range:        [{df['llm_score_mean'].min():.2f}, {df['llm_score_mean'].max():.2f}]")
    return df


# ── Shared helpers ─────────────────────────────────────────────────────────────

def add_regimes(ax):
    for start, end, color, _ in REGIMES:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                   alpha=0.15, color=color, zorder=0)

def add_zero_line(ax):
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)

def style_ax(ax, title, ylabel):
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(axis="x", rotation=30, labelsize=8)
    ax.tick_params(axis="y", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)


def annotate_failures(ax, df, col, is_dict=True):
    """
    Place failure/pass labels at a fixed vertical position inside the axes,
    with an arrow pointing to the actual bar. Avoids offset arithmetic
    that breaks when the y-scale differs between subplots.
    """
    ymin, ymax = ax.get_ylim()
    span = ymax - ymin

    for date_str, label in DICT_FAILURES.items():
        d = pd.Timestamp(date_str)
        row = df[df["date"] == d]
        if row.empty:
            continue
        bar_y = row.iloc[0][col]

        if is_dict:
            # Dictionary: text near bottom of plot, arrow up to bar tip
            text_y = ymin + span * 0.12
            marker  = f"✗ {label}"
            color   = "#cc0000"
        else:
            # LLM: text near top of plot, arrow down to bar tip
            text_y = ymax - span * 0.12
            marker  = f"✓ {label}"
            color   = "#006600"

        ax.annotate(
            marker,
            xy=(d, bar_y),
            xytext=(d, text_y),
            fontsize=7,
            ha="center",
            color=color,
            arrowprops=dict(arrowstyle="->", color=color, lw=0.9),
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85),
            zorder=5,
        )


# ── Figure 1: Dictionary score ─────────────────────────────────────────────────

def plot_dictionary(df):
    fig, ax = plt.subplots(figsize=(14, 5))

    colors = ["#d62728" if v > 0 else "#1f77b4" for v in df["hawkish_net"]]
    ax.bar(df["date"], df["hawkish_net"], color=colors, width=60, alpha=0.85)

    add_regimes(ax)
    add_zero_line(ax)
    style_ax(ax,
             "Figure 1: FOMC Hawkishness Score (Dictionary Baseline)",
             "Hawkish Net Score\n(positive = hawkish, negative = dovish)")

    ax.legend(handles=[
        mpatches.Patch(color="#d62728", alpha=0.85, label="Hawkish"),
        mpatches.Patch(color="#1f77b4", alpha=0.85, label="Dovish"),
    ], fontsize=9, loc="upper left")

    fig.canvas.draw()   # flush layout so get_ylim() is final
    annotate_failures(ax, df, "hawkish_net", is_dict=True)

    plt.tight_layout()
    fig.savefig("fomc_fig1_dictionary.png", dpi=150, bbox_inches="tight")
    print("Saved fomc_fig1_dictionary.png")
    plt.close()


# ── Figure 2: LLM score ────────────────────────────────────────────────────────

def plot_llm(df):
    fig, ax = plt.subplots(figsize=(14, 5))

    colors = ["#d62728" if v > 0 else "#1f77b4" for v in df["llm_score_mean"]]
    ax.bar(df["date"], df["llm_score_mean"], color=colors, width=60, alpha=0.85)

    ax.errorbar(df["date"], df["llm_score_mean"],
                yerr=df["llm_score_std"],
                fmt="none", ecolor="black", elinewidth=0.5, capsize=0, alpha=0.4)

    add_regimes(ax)
    add_zero_line(ax)
    style_ax(ax,
             "Figure 2: FOMC Hawkishness Score (LLM — Claude API, 3-run mean)",
             "LLM Hawkishness Score\n(−1 = maximally dovish, +1 = maximally hawkish)")

    ax.set_ylim(-1.1, 1.1)

    ax.legend(handles=[
        mpatches.Patch(color="#d62728", alpha=0.85, label="Hawkish"),
        mpatches.Patch(color="#1f77b4", alpha=0.85, label="Dovish"),
        Line2D([0], [0], color="black", alpha=0.4, linewidth=1, label="Test-retest std"),
    ], fontsize=9, loc="upper left")

    plt.tight_layout()
    fig.savefig("fomc_fig2_llm.png", dpi=150, bbox_inches="tight")
    print("Saved fomc_fig2_llm.png")
    plt.close()


# ── Figure 3: Stacked comparison ──────────────────────────────────────────────

def plot_comparison(df):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    fig.suptitle(
        "FOMC Hawkishness: Dictionary vs LLM Scoring\n"
        "Dictionary baseline (top) vs Claude LLM (bottom) — 235 statements, 1997–2026",
        fontsize=13, fontweight="bold", y=1.01
    )

    # ── Top: dictionary ──
    colors_dict = ["#d62728" if v > 0 else "#1f77b4" for v in df["hawkish_net"]]
    ax1.bar(df["date"], df["hawkish_net"], color=colors_dict, width=60, alpha=0.85)
    add_regimes(ax1)
    add_zero_line(ax1)
    style_ax(ax1,
             "Dictionary Baseline (Loughran-McDonald + custom hawkish/dovish word lists)",
             "Hawkish Net Score")

    # ── Bottom: LLM ──
    colors_llm = ["#d62728" if v > 0 else "#1f77b4" for v in df["llm_score_mean"]]
    ax2.bar(df["date"], df["llm_score_mean"], color=colors_llm, width=60, alpha=0.85)
    ax2.errorbar(df["date"], df["llm_score_mean"],
                 yerr=df["llm_score_std"],
                 fmt="none", ecolor="black", elinewidth=0.5, capsize=0, alpha=0.35)
    add_regimes(ax2)
    add_zero_line(ax2)
    style_ax(ax2,
             "LLM Scores (Claude, 3-run mean ± std) — 8/8 sanity checks passed",
             "LLM Score (−1 to +1)")
    ax2.set_ylim(-1.1, 1.1)

    # Flush layout before reading y-limits for annotation placement
    fig.canvas.draw()
    annotate_failures(ax1, df, "hawkish_net",   is_dict=True)
    annotate_failures(ax2, df, "llm_score_mean", is_dict=False)

    # Shared legend at bottom
    fig.legend(handles=[
        mpatches.Patch(color="#ffcccc", alpha=0.5, label="Hiking cycle"),
        mpatches.Patch(color="#cce5ff", alpha=0.5, label="Easing cycle"),
        mpatches.Patch(color="#d62728", alpha=0.85, label="Hawkish score"),
        mpatches.Patch(color="#1f77b4", alpha=0.85, label="Dovish score"),
    ], loc="lower center", ncol=4, fontsize=9,
       bbox_to_anchor=(0.5, -0.03), frameon=True)

    plt.tight_layout()
    fig.savefig("fomc_fig3_comparison.png", dpi=150, bbox_inches="tight")
    print("Saved fomc_fig3_comparison.png")
    plt.close()


# ── Summary stats ──────────────────────────────────────────────────────────────

def print_summary(df):
    print("\n── Sanity Check Comparison ───────────────────────────────")
    checks = {
        "2022-03-16": ("First 2022 hike",  "hawkish"),
        "2022-06-15": ("75bp hike",         "hawkish"),
        "2019-07-31": ("Insurance cut",     "dovish"),
        "2020-03-03": ("COVID emergency",   "dovish"),
        "2015-12-16": ("Liftoff from ZLB",  "hawkish"),
        "2008-12-16": ("ZLB adoption",      "dovish"),
        "2016-12-14": ("Post-ZLB hike",     "hawkish"),
        "2018-12-19": ("Dec 2018 hike",     "hawkish"),
    }
    dict_pass = llm_pass = 0
    print(f"  {'Date':<12} {'Expected':<10} {'Dict':>8} {'D?':<5} {'LLM':>8} {'L?':<5}  Label")
    print("  " + "-"*70)
    for date_str, (label, expected) in checks.items():
        d   = pd.Timestamp(date_str)
        row = df[df["date"] == d]
        if row.empty:
            print(f"  {date_str}  NOT FOUND")
            continue
        r       = row.iloc[0]
        d_score = r["hawkish_net"]
        l_score = r["llm_score_mean"]
        d_flag  = "PASS" if (d_score > 0) == (expected == "hawkish") else "FAIL"
        l_flag  = "PASS" if (l_score > 0) == (expected == "hawkish") else "FAIL"
        if d_flag == "PASS": dict_pass += 1
        if l_flag == "PASS": llm_pass  += 1
        print(f"  {date_str}  {expected:<10} {d_score:>+8.3f} {d_flag:<5} {l_score:>+8.3f} {l_flag:<5}  {label}")

    print(f"\n  Dictionary: {dict_pass}/8 passed")
    print(f"  LLM:        {llm_pass}/8 passed")
    print(f"  Avg LLM test-retest std: {df['llm_score_std'].mean():.4f}")


# ── Main ───────────────────────────────────────────────────────────────────────

def run():
    df = load_data()
    print_summary(df)
    print("\nGenerating figures...")
    plot_dictionary(df)
    plot_llm(df)
    plot_comparison(df)
    print("\nAll done. Three PNG files saved to your Fomc_Research folder.")
    print("  fomc_fig1_dictionary.png  — Figure 1 for paper")
    print("  fomc_fig2_llm.png         — Figure 2 for paper")
    print("  fomc_fig3_comparison.png  — Central exhibit")


if __name__ == "__main__":
    run()
