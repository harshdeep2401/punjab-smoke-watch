# Progress

Update this file at the end of every phase. Newest notes go at the top of each section.

**Current phase:** 5 (Phase 4 done)

## Phase checklist
- [x] Phase 0: Setup and feasibility
- [x] Phase 1: Ingestion with caching
- [x] Phase 2: Districts, daily tables, Replay map
- [x] Phase 3: Features, quantile model, backtest, ablation
- [x] Phase 4: RAG knowledge base
- [ ] Phase 5: Agents (tools, Sentinel, Analyst, Verifier)
- [ ] Phase 6: Forecast, Watch & Briefing, Evidence pages
- [ ] Phase 7: Demo cache, README, rehearsal
- [ ] Stretch

## Results (fill in during the phases)
- Ingest (Phase 1): fires 139,419 rows after dropping 6,707 low-confidence (2022: 59,162; 2023: 41,962; 2024: 21,812; 2025: 16,483). Daily PM2.5 and weather: 1,472 rows each (4 cities × 92 days × 4 seasons), 0 NaN. First run about 3 min, cached rerun about 1 s.
- Peak-window (Oct 25 to Nov 15) top districts: 2022 Sangrur 5014, Bathinda 3819, Ferozepur 2805, Moga 2689; 2025 Tarn Taran 437, Sangrur 434, Ferozepur 233. This matches public reporting (Sangrur, Bathinda, Ferozepur, Moga, Tarn Taran). Fires fall sharply from 2022 to 2025, which also matches reporting. 88,487 of 139,419 fires fall inside the 22 districts.
- Usable seasons: 2022, 2023, 2024, 2025 (2026 has FIRMS only to about Nov 2). Test on 2024 and 2025.
- Backtest (Phase 3; train on seasons < S, test on 2024 and 2025 pooled, 4 cities, n = 608 each; 80% band):

  | variant | MAE (µg/m³) | pinball | coverage | band width |
  |---|---|---|---|---|
  | full | 11.94 | 4.05 | 84.4% | 41.6 |
  | no_fire | 11.88 | 3.97 | 83.8% | 40.5 |
  | persistence | 11.20 | 3.83 | 76.9% | 33.9 |

  **Honest reading: the fire features do not help, and persistence is the best on MAE and pinball.** `full` is no better than `no_fire` (MAE 11.94 vs 11.88). I did not tune on the test seasons. Likely reasons: (1) CAMS PM2.5 already assimilates fire emissions (GFAS), so today's PM2.5 carries the fire signal; (2) only 2 to 3 training seasons (about 600 to 900 rows) and fires fall sharply from 2022 to 2025; (3) a daily mean is a coarse target. Top gain features of the final model: pm25_today, day_of_season, temp_mean, wind_speed_mean, pm25_3d_mean, punjab_fire_count_3d. Persistence is also the narrowest band but undercovers (76.9%).
  Conformal (CQR) was applied in every fold: the raw 80% band covered only 41 to 51% of the last training season, so it was widened by 8 to 10 µg/m³. After this, `full` covers 84% on the test seasons.
- RAG (Phase 4): 8 curated notes + 105 generated docs (88 season_{year}_{district}, 16 city_season, 1 model_card) = 113 docs, 137 chunks. Index 0.24 MB (vectors 211 KB, chunks 27 KB). Tests: 41 pass, offline (fake embedder).
  Acceptance checks: "why do satellites miss fires" returns firms_detection chunks first (score 0.81). `search("Sangrur last season", before=2025-11-05, district="Sangrur")` never returns 2025, and returns the 2022, 2024 and 2023 Sangrur docs in that order (scores 0.787, 0.771, 0.765), so 2024 is in the top 4 but **not first**. Embeddings cannot read "last season", so the Phase 5 `search_knowledge` tool / Analyst prompt should put the year in the query (e.g. "Sangrur 2024 season").
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

- Phase 3: `train` also writes `features.parquet` (feature table, 1,232 rows). `pipeline all` now runs ingest, process, train.
- Phase 3: `psw/models/__init__.py` and `fit_with_conformal` / `predict_with_correction` / `cqr_correction` are extra helpers in `train.py`. The conformal step runs for both `full` and `no_fire` (so the ablation is fair), not for `persistence`.
- Phase 3: wind direction for the upwind feature is the vector mean of (u_mean, v_mean) via `dir_from_uv`, not the `wind_dir_mean` column (which is a plain average and wrong across the 0/360 wrap).
- Phase 3: lagged fire features on the first days of a season treat days before Sep 15 as zero fires (FIRMS only covers the season window). `punjab_fire_count_3d` is a 3-day sum.
- Phase 3: `upwind_contributors` groups fires outside the 22 districts as "Outside Punjab".
- Phase 3: the saved `model_full_{alpha}.txt` boosters have no conformal correction (the backtest forecasts do).
- Phase 3 (fix after review): `seasons_train` wrongly listed the test seasons. It is now the seasons before the first test season ([2022, 2023]), and a new `folds` key gives the exact training seasons per test season (2024: [2022, 2023]; 2025: [2022, 2023, 2024]). Metrics are unchanged. The final model in `model_full_*.txt` trains on all seasons.
- Phase 3: `metrics.json` has an extra `conformal` key with per-season inner coverage and corrections, and `n` per score.

- Phase 4: `rag/index.py` also holds the `Doc` dataclass, `load_docs` (a tiny frontmatter parser, so no PyYAML dependency) and `FastEmbedder`. Embedders are injected objects with `embed_passages(list)` and `embed_query(str)`. Queries use the bge query prefix, and the doc title is embedded together with each chunk.
- Phase 4: chunk ids are `K000...` in order of sorted doc id, so they are stable for the same corpus. They change when docs are added or edited.
- Phase 4: curated notes get `as_of: 2022-01-01` (timeless, visible for every briefing date). The index also stores `source_url`, a column not in PLAN.md.
- Phase 4: `source_url` of our own notes (method, limitations, how_to_read_forecast) is a repo path (`docs/PLAN.md#phase-3`, `Punjab_Smoke_Watch.md#...`), since there is no external source.
- Phase 4: `pipeline kb` regenerates `kb/generated/` (deleting old files) and rebuilds the index. `pipeline all` now runs ingest, process, train, kb.
- Phase 4: the model card is generated from `metrics.json` with conditional wording ("did not improve" / "did not beat"), so it states the negative result as it is. Its "possible reasons" are marked "not tested here".
- Phase 4: `data/processed/kb_*` should be committed. `kb/generated/` is gitignored (rebuilt by `pipeline kb`).

## Open questions / user actions needed
- User (Phase 3): decide whether the negative result is acceptable for the demo. Plan: the Evidence page shows it as is. Say if you want an extra experiment (not tuned on test seasons) before Phase 4.
- User (Phase 2): run `uv run streamlit run src/psw/app/main.py`, open Replay Map, and check that it looks right and Play works.
- User: confirm the LLM key works when Phase 5 starts (not tested yet).
- User: the 2026 FIRMS feed looks stale after about Nov 2. Decide if that matters (it only affects the stretch Live mode).
- User (Phase 4): **review `kb/curated/*.md` for factual accuracy.** I wrote them from memory and did not browse. Check especially: the S-NPP overpass times (about 01:30 / 13:30), the CAMS note (coarse grid, GFAS fire emissions), NAAQS 60 and WHO 15 µg/m³ (24 h) and the annual values, the GRAP stage names (Poor / Very Poor / Severe / Severe+) and that GRAP covers Delhi-NCR, and the paddy harvest / wheat sowing dates. The `source_url` for `grap.md` (caqm.nic.in) and `crop_residue_context.md` (agri.punjab.gov.in) are site roots, not exact pages. Replace them with the exact page.
- User (Phase 4): the first `pipeline kb` run downloaded the bge-small model (about 130 MB) to the local cache. Done on this machine already.

## Review notes (Reviewer: GPT 6 Luna)
- Phase 2: `src/psw/app/pages/1_Replay_Map.py:54` builds the fire layer with a Python loop over every detection for the selected day. This conflicts with the Phase 2 review checklist requirement to avoid Python loops over fire rows; construct the layer data with vectorized columns instead.
  - **Resolved:** the fire layer is now built from vectorised pandas columns (`radius`, `tip`) and passed as a DataFrame. The only remaining loop is over the 22 district polygons. Both layers now use one top-level `tip` field for the tooltip (`{tip}`). Re-run the app and recheck the tooltips.
