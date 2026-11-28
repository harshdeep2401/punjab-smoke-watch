# Forward test on the 2026 season: plan (written 2026-11-28, before any 2026 result was computed)

## Why
Every result so far is a backtest on 2024 and 2025, and the variant used by the planner was picked on those same
seasons. 2026 was never used for any design choice. Scoring the unchanged method on it is the closest thing this project
can have to a prospective test.

## What is frozen
- Code: the fire-risk model, features, baselines and allocator as last changed in commit `f42a1e5` (2026-11-27).
  Nothing in `response/risk.py`, `response/harvest.py`, `response/allocate.py` or `models/train.py` changes for this.
- Recipe: train on every season before 2026 (2022 to 2025), predict 2026, the same call as each backtest fold.
- Variant used by the planner: `model` (from `risk_metrics.json`, chosen on 2024 and 2025). All three are reported.
- Allocation settings: 500 units, k = 20, at most 25% per district (the decision backtest's settings).
- No setting is changed after looking at 2026. If something has to change to make the code run, it is logged.

## The data problem, and how it is handled
Checked on 2026-11-28 with the FIRMS data-availability endpoint:
- **S-NPP near-real-time data stops on 2026-11-02.** NOAA-20, NOAA-21 and MODIS run to 2026-11-27.
- Standard (archive) products end on 2026-08-31, so 2026 is near-real-time (NRT) data only. Training used archive data.
- MCD64A1 burned area ends 2026-09-01, so there is **no 2026 gap range** and no audit of 2026 yet.
- The weather archive lags by 1 to 2 days.

So the test has two arms:

| Arm | Fire data for 2026 | Issue dates | Role |
|---|---|---|---|
| **A: S-NPP** | S-NPP NRT, the sensor the model was trained on | 15 Sep to 3 days before the last full S-NPP day | **Primary.** A clean test, but it ends before the usual peak. |
| **B: NOAA-20 stand-in** | NOAA-20 NRT x a fixed ratio (S-NPP / NOAA-20 in Punjab, 2022 to 2025 archive, no 2026 data used) | 15 Sep to 3 days before the data cutoff | **Secondary, labelled.** Covers the peak, but swaps the input sensor. |

Arm B is also scored on arm A's dates, so the cost of the sensor swap is measured, not assumed.

Rules fixed now:
- The last S-NPP day (2026-11-02) may be partial, so arm A uses data up to 2026-11-01.
- The data cutoff for arm B is the last day the weather archive covers, at most 2026-11-26.
- 2026 counts and weather are cached under names that include the end date, so a later run with the full season never
  reads a partial file.
- Low-confidence VIIRS detections are dropped, as in training.

## Success criteria (fixed before running)
Backtest reference for the `model` variant: top-5 hit 69.4% (2024) and 57.6% (2025); Spearman 0.77 and 0.69;
allocation 95.5% and 95.0% of the oracle, against 90.9% and 91.5% for the naive split by last season's fires.

On arm A (and repeated on arm B):
1. **Ranking holds:** the model's top-5 hit rate is at least 52.6% (the weaker backtest season minus 5 points) and its
   Spearman at least 0.64 (0.69 minus 0.05).
2. **Not worse than persistence:** the model's top-5 hit rate is at least persistence's. Reported with the same 7-day
   block-bootstrap interval as the backtest.
3. **The allocation still beats the naive split:** model allocation's % of the oracle is above the split by last
   season's fires.
4. **The band is honest:** 80% band coverage between 70% and 90%.

Each criterion is reported as met / not met with its number, whatever the result. Arm A has fewer days than a full
season, so a miss there may be noise; the interval is reported alongside.

## Out of scope (and why)
- The next-day PM2.5 forecast: its fire features did not help in the backtest, and its fire inputs need S-NPP after
  2026-11-02 too. Not the decision the app supports.
- A 2026 gap range: needs October and November burned area, not published yet. Revisit around February 2027.
- The LLM agents: this tests the numbers they explain, not the text.

## Steps
1. `ingest/forward.py`: fetch 2026 NRT S-NPP and NOAA-20 up to a given end date (5-day chunks, end-dated cache), and
   2026 district weather from the archive up to the same end date.
2. `response/forward.py` (pure): the NOAA-20 scaling ratio; 2026 district-day counts truncated at an end date; run the
   frozen recipe for one arm; score it (the backtest's metrics, ranking and allocation); compare with the criteria.
3. `pipeline forward-test`: writes `data/processed/forward_2026/` (2026 detections, forecasts per arm, `forward_test.json`).
4. Tests on synthetic data, offline: truncation never invents zeros after the end date, the ratio uses no 2026 data,
   the model is trained on seasons before 2026 only, the criteria verdicts follow the numbers.
5. Evidence page, "Forecast and allocation" tab: a forward-test section with the table beside the backtest and the four
   verdicts. One line on the Overview's trust card.
6. PROGRESS.md (results, deviations), AGENTS.md decision D25, README results.
