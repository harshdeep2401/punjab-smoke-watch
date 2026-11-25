# Progress

Update this file at the end of every phase. Newest notes go at the top of each section.

**Current phase:** 2 (Phase 1 done)

## Phase checklist
- [x] Phase 0: Setup and feasibility
- [x] Phase 1: Ingestion with caching
- [ ] Phase 2: Districts, daily tables, Replay map
- [ ] Phase 3: Features, quantile model, backtest, ablation
- [ ] Phase 4: RAG knowledge base
- [ ] Phase 5: Agents (tools, Sentinel, Analyst, Verifier)
- [ ] Phase 6: Forecast, Watch & Briefing, Evidence pages
- [ ] Phase 7: Demo cache, README, rehearsal
- [ ] Stretch

## Results (fill in during the phases)
- Ingest (Phase 1): fires 139,419 rows after dropping 6,707 low-confidence (2022: 59,162; 2023: 41,962; 2024: 21,812; 2025: 16,483). Daily PM2.5 and weather: 1,472 rows each (4 cities × 92 days × 4 seasons), 0 NaN. First run about 3 min, cached rerun about 1 s.
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

## Open questions / user actions needed
- User: confirm the LLM key works when Phase 5 starts (not tested yet).
- User: the 2026 FIRMS feed looks stale after about Nov 2. Decide if that matters (it only affects the stretch Live mode).
- User: review `kb/curated/*.md` for factual accuracy (after Phase 4).

## Review notes (Reviewer: GPT 6 Luna)
(none yet)
