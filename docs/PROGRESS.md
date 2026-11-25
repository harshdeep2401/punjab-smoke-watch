# Progress

Update this file at the end of every phase. Newest notes go at the top of each section.

**Current phase:** 3 (Phase 2 done)

## Phase checklist
- [x] Phase 0: Setup and feasibility
- [x] Phase 1: Ingestion with caching
- [x] Phase 2: Districts, daily tables, Replay map
- [ ] Phase 3: Features, quantile model, backtest, ablation
- [ ] Phase 4: RAG knowledge base
- [ ] Phase 5: Agents (tools, Sentinel, Analyst, Verifier)
- [ ] Phase 6: Forecast, Watch & Briefing, Evidence pages
- [ ] Phase 7: Demo cache, README, rehearsal
- [ ] Stretch

## Results (fill in during the phases)
- Ingest (Phase 1): fires 139,419 rows after dropping 6,707 low-confidence (2022: 59,162; 2023: 41,962; 2024: 21,812; 2025: 16,483). Daily PM2.5 and weather: 1,472 rows each (4 cities × 92 days × 4 seasons), 0 NaN. First run about 3 min, cached rerun about 1 s.
- Peak-window (Oct 25 to Nov 15) top districts: 2022 Sangrur 5014, Bathinda 3819, Ferozepur 2805, Moga 2689; 2025 Tarn Taran 437, Sangrur 434, Ferozepur 233. This matches public reporting (Sangrur, Bathinda, Ferozepur, Moga, Tarn Taran). Fires fall sharply from 2022 to 2025, which also matches reporting. 88,487 of 139,419 fires fall inside the 22 districts.
- Usable seasons: 2022, 2023, 2024, 2025 (2026 has FIRMS only to about Nov 2). Test on 2024 and 2025.
- Backtest (variant × MAE × pinball × coverage):
- Briefing verifier pass rate (n, agent / agent_revised / template):

## Deviations from PLAN.md
- D10 fallback applied: 22 districts, no Malerkotla (neither source has it). Source is geoBoundaries (ODbL), not udit-001, because the latter has no licence file.
- Weather archive lag measured at 1 day, not ~5 as PLAN.md assumed.
- `SEASONS = [2022..2025]`, not including 2026, because FIRMS NRT for 2026 stops around Nov 2.
- FIRMS max `DAY_RANGE` is 5 (not the 10 PLAN.md hoped for), so Phase 1 must chunk requests by 5 days.
- `libomp` was missing, so I ran `brew install libomp` (system change outside the repo).
- Phase 1: `fetch_season` falls back to NRT only when an SP chunk is empty. For 2022 to 2025 all 16 chunks per season came from SP (no NRT used).
- Phase 1: added `psw/ingest/http.py` (shared retry/backoff helper that scrubs the FIRMS key from errors) and `tests/conftest.py` (temp cache dir, no sleeping). Not in PLAN.md, but small.
- Phase 1: `weather.fetch_city_weather` uses the forecast API for days newer than a 2-day archive lag. It is untested against the real API, because no season in `SEASONS` needs it.
- Phase 2: `process` also writes `pm25_daily.parquet` and `weather_daily.parquet` (needed later in Phase 3, and the app must run from committed data).
- Phase 2: `fire_count_7d_mean` uses `min_periods=1`, so the first 6 days of a season average over the days available.
- Phase 2: fires exactly on a district edge get `district = NaN` (predicate `within`). Fires in the bbox but outside Punjab are kept with `in_punjab = False`, since upwind fires matter.
- Phase 2: pipeline subcommands `train | kb | brief | demo-cache` exist but exit with "Not implemented yet (Phase N)". `all` currently runs `ingest` and `process`.
- Phase 2: the map was only tested with Streamlit's `AppTest` (no exceptions, play loop runs to Nov 30 and stops) and a headless server health check (HTTP 200). **I could not look at the rendered map.**

## Open questions / user actions needed
- User (Phase 2): run `uv run streamlit run src/psw/app/main.py`, open Replay Map, and check that it looks right and Play works.
- User: confirm the LLM key works when Phase 5 starts (not tested yet).
- User: the 2026 FIRMS feed looks stale after about Nov 2. Decide if that matters (it only affects the stretch Live mode).
- User: review `kb/curated/*.md` for factual accuracy (after Phase 4).

## Review notes (Reviewer: GPT 6 Luna)
- Phase 2: `src/psw/app/pages/1_Replay_Map.py:54` builds the fire layer with a Python loop over every detection for the selected day. This conflicts with the Phase 2 review checklist requirement to avoid Python loops over fire rows; construct the layer data with vectorized columns instead.
  - **Resolved:** the fire layer is now built from vectorised pandas columns (`radius`, `tip`) and passed as a DataFrame. The only remaining loop is over the 22 district polygons. Both layers now use one top-level `tip` field for the tooltip (`{tip}`). Re-run the app and recheck the tooltips.
