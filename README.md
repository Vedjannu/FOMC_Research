# FOMC Hawkishness Scoring & Market Reaction Research

Research project studying whether the tone of FOMC post-meeting statements
contains information about subsequent moves in rate-sensitive futures
markets that is not already priced in at the time of release.

Mentored by Jusvin Dhillon, Tudor Investment Corp.

## Phases

1. **Scraping + dictionary baseline** — scraped all FOMC statements
   (1997-2026) from federalreserve.gov, scored with the Loughran-McDonald
   financial dictionary plus a custom hawkish/dovish word list
2. **LLM scoring** — scored each statement with the Claude API
   (3 runs per statement for test-retest reliability), compared against
   the dictionary baseline
3. **Surprise construction + regressions** — built AR(1) and
   market-implied hawkishness "surprise" measures (fit on a 1997-2015
   training window to avoid look-ahead bias), ran predictive regressions
   against 2yr/10yr Treasury and S&P 500 returns

## Repo structure

- `scraping/` — Phase 1 scraper and dictionary scoring
- `scoring/` — Phase 2 Claude API scorer + cached raw outputs
- `surprises/` — Phase 3 surprise construction and comparison script
- `figures/` — Figures 1-3
- `docs/` — Phase 1 technical documentation
- `paper/` — current paper draft
