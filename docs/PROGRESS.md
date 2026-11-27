# Progress

Update this file at the end of every phase. Newest notes go at the top of each section.

**Current phase:** Part 1 (0–7) done, plus Part 1 stretch 5 (hybrid retrieval) and 6 (Planner). **Part 2 (Audit + Response) planned on 2026-11-26. Phase 8 done on 2026-11-26, Phase 9 on 2026-11-27. Opus review of the gap formula done (D23). Next: Phase 10.** See AGENTS.md §1b and PLAN.md Part 2.

**Part 2 motivating finding** (Opus, from `fires.parquet`, S-NPP, Indian Punjab):

| Season | Day detections | Night detections | Night share |
|---|---|---|---|
| 2022 | 45,305 | 82 | 0.2% |
| 2023 | 30,300 | 90 | 0.3% |
| 2024 | 8,530 | 142 | 1.6% |
| 2025 | 3,942 | 96 | 2.4% |

Afternoon-pass (IST 12–14 h) detections fell about 91%, but night detections didn't. The night counts are small, so this is a lead, not proof. Phase 10 tests it.

## Phase checklist
- [x] Phase 0: Setup and feasibility
- [x] Phase 1: Ingestion with caching
- [x] Phase 2: Districts, daily tables, Replay map
- [x] Phase 3: Features, quantile model, backtest, ablation
- [x] Phase 4: RAG knowledge base
- [x] Phase 5: Agents (tools, Sentinel, Analyst, Verifier)
- [x] Phase 6: Forecast, Watch & Briefing, Evidence pages
- [x] Phase 7: Demo cache, README, rehearsal (screenshots and the clean-clone check are yours, see below)
- [x] Part 1 stretch 5 (hybrid retrieval) and 6 (Planner). Item 7 (Critic) moved into Phase 10 (D21). Item 8 (Comparison) superseded by the Auditor. Old item 3 (detection-gap panel) superseded by Phases 9–10.
- [x] Phase 8: Feasibility for the new data (**user decision point open**, see below)
- [x] Phase 9: Audit data (multi-sensor FIRMS, burned area, manual CSVs, district rain)
- [ ] Phase 10: Audit analytics, gap ranges, Auditor agent (+ optional Planner, Critic on by default for audits)
- [ ] Phase 11: Harvest signal and 3-day fire-risk forecast
- [ ] Phase 12: Allocator, 5 km grid, Dispatcher agent
- [ ] Phase 13: Audit and Response Planner pages, Evidence, docs, demo
- [ ] Part 2 stretch

## Results (fill in during the phases)
- Phase 8 (full table in `docs/feasibility.md` Part 2). **Go:** NOAA-20 (2022–25), NOAA-21 (2023–25), MODIS Aqua, MODIS Terra (weak), MCD64A1 burned area via the University of Maryland SFTP server (all months 2022–25), district rain. Also **go** (found 2026-11-27 in National Green Tribunal filings): official district counts for 2022–25 and district CRM machine counts. **No-go:** MCD64A1 via Planetary Computer (Sep 2023 and Sep–Oct 2025 missing), Sentinel-2 dNBR (measures harvest: 86.6% of Sangrur has dNBR > 0.27), mandi arrivals (HTTP 403). Tests: 178 pass (no source code changed).
- Phase 8, state totals side by side (inputs for Phase 10, **not an audit result**):

  | Season | S-NPP detections | MCD64A1 burned km² | PPCB official count |
  |---|---|---|---|
  | 2022 | 45,387 | 16,537 | 49,922 |
  | 2023 | 30,390 | 14,998 | 36,663 |
  | 2024 | 8,672 | 6,347 | 10,909 |
  | 2025 | 4,038 | 9,102 | 5,114 |

  Burned area fell less than detections (2025 vs 2022: −45% against −91%) and **rose** from 2024 to 2025 while detections halved. Official counts fell about as much as detections (−90%). MCD64A1 is low confidence in cropland, so this is a lead, not proof.
- Phase 9, multi-sensor fires (step 1 of 4): `fires_multisensor.parquet`, 408,056 rows (3.1 MB), 253,800 inside Punjab. MODIS rows below confidence 30 and VIIRS 'l' rows are dropped. S-NPP rows match `fires.parquet` one for one, districts included.

  | Inside Punjab | 2022 | 2023 | 2024 | 2025 |
  |---|---|---|---|---|
  | S-NPP | 45,387 | 30,390 | 8,672 | 4,038 |
  | NOAA-20 | 48,365 | 32,778 | 9,958 | 4,153 |
  | NOAA-21 | none | 31,244 | 8,949 | 4,050 |
  | MODIS | 10,565 | 8,347 | 3,853 | 3,051 |

  Whole bounding box: S-NPP 59,162 / 41,962 / 21,812 / 16,483; NOAA-20 62,689 / 44,914 / 24,763 / 16,992; NOAA-21 – / 43,611 / 23,473 / 16,254; MODIS 13,156 / 10,940 / 6,398 / 5,447.

- Phase 9, burned area (step 2 of 4): `burned_area.parquet`, 88 district-seasons (22 x 4). Punjab burned area for burn dates 15 Sep to 30 Nov: 16,532 / 14,980 / 6,343 / 9,102 km2 (2022 to 2025). That is down 45% from 2022 to 2025, against 91% for S-NPP detections. A lead for Phase 10, not a finding: MCD64A1 is low confidence in cropland. Some district values look implausibly low (Jalandhar 2024: 1.2 km2).
- Phase 9, official counts (step 3 of 4): `data/manual/official_counts.csv`, 92 rows (23 districts x 4 seasons), generated from the three report PDFs, not retyped; each season adds up to the report's stated total (49,922 / 36,663 / 10,909 / 5,114). Official counts fell 90% from 2022 to 2025, close to S-NPP's 91%. Both are satellite-based, so this agreement is not independent confirmation.
- Phase 9, district weather (step 4 of 4): `district_weather_daily.parquet`, 8,272 rows (22 districts, 1 Sep to 3 Dec, 2022 to 2025), no gaps. Median district rain for 15 Sep to 30 Nov: 122 / 94 / 16 / 51 mm, the same as Phase 8.
- Phase 9 acceptance: 210 tests pass. `audit-ingest` rerun with the network blocked gives identical parquet files. Neither the FIRMS nor the LLM key appears in `src`, `tests`, `docs`, `data/manual` or `data/processed`.
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
- Agents (Phase 5): 130 tests pass offline (FakeLLM, no network). Live check with `gemini-3.5-flash`, daily watch for 2025-10-31 (busiest day of 2025; 15 of 22 districts flagged): the top 3 (Tarn Taran, Bathinda, Moga) were all written by the agent, mode `agent`, verified (22 to 24 numbers checked each, 5 to 6 tool calls).
- Briefing verifier pass rate (n = 12 sampled flagged district-days from the 2024 and 2025 test seasons, seed 42; 779 flagged district-days in total): **12 verified of 12. agent 12 / agent_revised 0 / template 0.** The target was at least 10. Models: 9 briefings by `gemini-3.5-flash-lite`, 3 by `gemini-3.5-flash` (reused from an earlier partial run that hit the free-tier quota; the first resumable rerun kept them). Word counts 108 to 147 (mean 128). Rerun with `uv run python -m psw.pipeline brief --sample 12`. Token use of the 9 flash-lite briefings: 18 calls, 35,862 prompt and 3,887 completion tokens.
  Reading two of them as a reader: every number is real and the text is clear, but (1) both were **zero-fire days flagged only as `smog_risk`** (the nearest city's p90 is above 120, so every district near that city is flagged on a smoggy day), and (2) one cites a 2023 season-summary chunk after "ranking 7 in Punjab", which that chunk does not support. **The Verifier proves a cited chunk was retrieved, not that it supports the sentence.** Phase 6 should say this on the Evidence page, and the Daily Watch should list fire-heavy flags first (it already sorts by priority score, which includes the fire-count rank).

- App (Phase 6): 145 tests pass offline (15 new in `tests/test_app.py`: helpers on synthetic data, plus a Streamlit `AppTest` run of all 5 pages and of the Watch page's no-LLM and off-season paths). Acceptance run with `.env` moved to `.env.bak`: all 5 pages load with 0 exceptions, and the headless server health check returns 200. **I could not look at the rendered pages** (no browser), so check the layout by eye.
- Evidence page, read from the real `metrics.json` (all text is computed, none typed in): 616 forecasts per variant (4 cities × 154 days; the earlier "n = 608" above was a typo), lowest MAE is persistence (11.20), fire features did not help (11.94 vs 11.88), the model's 80% band covers 84.4% and persistence's 76.9%. Agent reliability table: demo set 4 of 4 verified (3 agent, 1 template), sampled run 12 of 12 verified (12 agent).

- Demo cache (Phase 7): demo dates chosen by rule from the 2025 season: peak 2025-10-31 (273 detections in Punjab), moderate 2025-10-26 (closest to half the peak total among days with a fire-driven flag), early 2025-10-18 (the first day with a fire-driven flag). `demo-cache` wrote 8 briefings in `briefings/` and 3 watch files, all by the agent and all verified. Peak day: Tarn Taran, Bathinda, Moga, Ludhiana, Sangrur. One run of the final `demo-cache` made 8 LLM calls on `gemini-3.5-flash-lite` (about 36 s).
- Verifier change (Phase 7): reading the demo briefings found "the district ranked rank_in_punjab" (a raw tool field name in the text, no number, so it passed). The Verifier now flags snake_case field names, and the Analyst prompt says not to write them. Scanning every saved briefing found 2 offenders (this one and `sample_run/.../2025-11-27_Jalandhar.json`, which said `punjab_total`). I deleted both and regenerated them; both pass now. The sampled run is again **12 of 12 verified, 12 by the agent**, but it needed this fix, so the earlier "12/12" was true only under the weaker Verifier.
- Definition of done: backtest table has all three variants on 2 test seasons (yes); 12 sampled briefings pass (yes, 12/12, agent 12 / template 0); RAG `as_of` test exists (`tests/test_rag.py`); README credits sources and lists limitations (yes); secret scan of `git log -p` for `key=`-style assignments with a long value printed nothing, and `.env` is not tracked (yes); **clean-clone run: not done by me, see below**.

- Hybrid retrieval (stretch 5), 17-query eval, k = 3, real bge model: hit@3 hybrid 17/17, embeddings 17/17, BM25 16/17. hit@1 16 / 15 / 15. MRR 0.971 / 0.931 / 0.912. Per query, hybrid ranked the right doc higher than embeddings on 2 (a rare exact term, "Happy Seeder", 3 to 1; "Bathinda PM2.5 season 2022", 2 to 1) and lower on 1 ("why do satellites miss some fires", 1 to 2, because BM25 has no stemming and found nothing for "satellites" vs "satellite"). **At hit@3 this set cannot separate the methods (embeddings are already perfect), and 17 queries written by us after reading the corpus is a sanity check, not a benchmark.** I did not tune queries or the tokenizer after seeing the results.
- Live check with hybrid on (`gemini-3.5-flash-lite`, Ludhiana 2025-10-31, written to a scratch folder): `agent_revised`, verified. The revision was triggered by the new raw-field-name check (`share_pct`), not by retrieval.

- Planner (stretch 6), live checks on `gemini-3.5-flash-lite`, scratch folders only (nothing in `data/processed/` changed): 4 briefings with the planner on, all verified, 3 agent and 1 agent_revised (a 13-word headline, not the plan). For Bathinda 2025-11-20, flagged only for `smog_risk` (4 detections), the plan put the forecast and the wind first and the briefing led with them, as the prompt asks. **I did not measure whether plans improve the briefings** (no A/B on quality, only 4 runs), and the Verifier pass rate was not re-measured with the planner on. It adds one LLM call per briefing, about 1.1k prompt and 0.3k completion tokens.
- Planner: the first real run produced no usable plan once (reason not recorded, because the error text was vague). Two later calls with the same input gave good plans. I made the cleaner allow the Verifier's constants (7-day, 80%, p10/p50/p90, 24-hour), since the tool descriptions in the prompt contain "7-day" and the model repeats it, and I made the failure reason specific (`planner_error` now says why). A plan with no usable steps is still possible and is simply skipped.

## Deviations from PLAN.md
- Phase 9: district weather covers 1 Sep to 3 Dec, so Phase 11 has trailing and next-3-day rain at the season's edges. Wind speed is km/h, as in `weather_daily.parquet`. It is ERA5 reanalysis at one centroid per district, not rain gauges.
- Phase 9: the CSV keeps the sources' Malerkotla rows. `audit.official.load_official` adds them to Sangrur (D10) and returns 22 districts. District names in the CSV are the boundary file's spellings. No parquet is written; `audit-ingest` validates the CSV and stops on a bad row.
- Phase 9: burned area comes from the University of Maryland server (D22, replaces D17). `s2_dnbr_district` was not written (optional; Phase 8 no-go).
- Phase 9: in `burned_area.parquet`, `n_pixels` is the number of burned pixels. A pixel that burned in two months counts once, so state totals are up to 18 km2 below the Phase 8 figures (which added the months).
- Phase 9: the SFTP download does not check the server's host key (public data, public password).
- Phase 9: `pass_label` uses whole IST hours: morning 10–11, afternoon 12–15, night 0–3, else "other" (only Terra's evening passes, 6.6% of Terra rows).
- Phase 9: `acq_date` stays the UTC date, as in `fires.parquet`, so a night pass (about 01:30 IST) carries the previous day's date.
- Phase 9: `ingest.firms.fetch_season` takes a `product` argument (default S-NPP). `load_multisensor` raises if a sensor has no rows for a season where it should (NOAA-21 starts 2023).
- Phase 8: **D17 does not work as written.** Planetary Computer is missing Sep 2023 and Sep–Oct 2025, so the proposed burned-area path is the monthly `burndate` GeoTIFFs from the University of Maryland SFTP server (window 18). Phase 9 would add `rasterio` and `paramiko` instead of `pystac-client` + `planetary-computer` + `rasterio`. Approved on 2026-11-26 (D22); the two packages were added in Phase 9.
- Phase 8: the checks are in a new `scripts/feasibility_part2.py` (`firms | burned | burned_umd | s2 | rain`). The extra packages are passed with `uv run --with ...`. The official-count, mandi, machinery and licence checks were done by hand and are only recorded in `docs/feasibility.md`.
- Phase 8: the SFTP host, login and password are in the script. They are the shared ones printed in the MCD64A1 user guide §4.1, not a secret of ours.
- Phase 8: downloaded rasters are cached in `data/raw/burned/mcd64_win18/` and FIRMS windows in `data/raw/firms/` (both gitignored).
- Phase 8: the plan expected the VIIRS satellites to add different pass times. They do not (all about 13:30 IST). Phase 9 `pass_label` windows need to be wide (Aqua is at about 15:15 in 2025).
- Stretch 6: new `agents/planner.py` (`build_planner`, `make_plan`, `clean_plan`, `plan_text`, `flag_kinds`). PLAN says the plan shows at the top of the Agent trace tab: done, as a block above the table when `plan` is in the record. The Planner also appears as the first row of the trace (agent `planner`, tool `plan`). `BriefingRecord` has two new fields, `plan` and `planner_error`; old saved briefings lack them and the page handles that. `run_briefing`, `run_daily_watch` and `build_demo_cache` take `use_planner` (default False); the CLI flag is `--plan` on `brief` and `demo-cache`; the app's Run agent / Regenerate button has a "Plan first" checkbox. `analyst.analyse` and `user_message` take an optional `plan_text`. `reuse=True` returns a saved briefing even if it has no plan, so use `--force` to add plans to the demo set.
- Planner input: it gets the **kinds** of Sentinel flag (spike, high_count, smog_risk), not their text, so no number can enter the plan from there. PLAN did not say what the Planner sees.
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
- Stretch 5: new dependency `rank-bm25` (in `pyproject.toml` and `uv.lock`). `KnowledgeBase.search` got `mode` ("hybrid" default, "embedding", "bm25"); `score` means cosine, BM25 or fused RRF score by mode. BM25 uses the same text as the embeddings (title + chunk), a plain `[a-z0-9]+` tokenizer, no stemming, and returns nothing for chunks sharing no word with the query. The district boost applies to the embedding side only. The BM25 index is built in memory on first use, so no new committed file besides the eval report.
- Stretch 5: the default retrieval changed from embeddings to hybrid, so the Analyst's `search_knowledge` now uses it. The saved briefings were written with embeddings and are unchanged. `search_knowledge` for the LLM has no `mode` argument.
- Stretch 5: new `rag/evaluate.py`, `tests/fixtures/retrieval_eval.json` (17 queries: 9 meaning, 7 keyword, 1 with the `before` and `district` filters), `pipeline kb-eval` (writes `data/processed/retrieval_eval.json`), and an Evidence-page section whose wording is computed from the report. The old district-boost test was pinned to `mode="embedding"`, because it checks cosine arithmetic.
- Phase 7: new `agents/demo.py` (`pick_demo_dates`, `build_demo_cache`) and `data/processed/demo.json`. `run_briefing` / `run_daily_watch` got `reuse=True`, which returns a saved briefing only if the agent wrote it and it verified (templates are redone). `demo-cache` uses it, `--force` turns it off.
- Phase 7: the plan says the early-season demo day is "early-season", but before about 2025-10-15 the Sentinel flags nothing, so a daily watch would brief nobody. "Early" is the first day with a fire-driven flag (2025-10-18), and each of the moderate and early days has only 1 flagged district. So those two days show a one-row watchlist.
- Phase 7: the Watch page opens on `demo.json`'s peak date and Ludhiana (falls back to the computed peak day if the file is missing). The Forecast page already defaulted to Ludhiana. The sidebar lists the three demo dates.
- Phase 7: **no screenshots.** Headless Chrome only captured Streamlit's loading skeleton (the page renders over a websocket), so I deleted the blank images. `docs/screenshots/` does not exist.
- Phase 7: brief §12 (demo script) was rewritten. The old step 4 said to show that "the model beats persistence and removing fire features makes it worse". That is the opposite of the result, so the new step 4 presents the negative result. The README states it in the same way.
- Phase 7: the app (all 5 pages) was run with `lightgbm`, `fastembed`, `sklearn` and `openai` blocked from importing, and with `.env` absent: 0 exceptions. So a clean clone needs none of them to run the demo (`libomp` is only for `train`).
- Phase 6: new `psw/app/helpers.py` holds the pure logic (no Streamlit): peak day, finding saved briefings, badge, the Verifier number report, backtest tables, `interpret(metrics)` (the Evidence text, which changes wording with the result), and the reliability table. `common.py` gained cached loaders (`get_store`, `get_kb`, `load_forecasts`, `load_metrics`, `scan_watchlist`).
- Phase 6: the Watch page always runs the Sentinel live (offline, fast) and does **not** read `watch/{date}.json`. It marks the districts that have a saved briefing. Any date in the data can be picked. Outside the test seasons there is no forecast, so the page says so.
- Phase 6: the page looks for a saved briefing in `briefings/` first, then `sample_run/briefings/`. The Evidence reliability table covers both folders separately ("demo briefings", "sampled run"), since the 12-briefing sample lives in `sample_run/`, not `briefings/` as PLAN.md assumed.
- Phase 6: the Verifier report maps each number to the evidence value and the tool that supplied it by re-applying `Evidence.matches` to each tool result alone. The Verifier itself is unchanged.
- Phase 6: the "Run agent" / "Regenerate" button calls `run_briefing` and **writes into `data/processed/briefings/`**. Without an LLM key it only shows a message. The Forecast "upwind contributors" are for the issue date (the wind and fires the forecast was made from), via `get_wind_and_upwind`.
- Phase 6: used `width="stretch"` instead of the deprecated `use_container_width=True`.
- Phase 5: `run_agent` has one extra optional argument `history` and `AgentResult` has an extra `messages` field. The orchestrator uses them to continue the same conversation for the one revision ("Fix these: ..."), so the revision costs one LLM call and not a second full tool run.
- Phase 5: the orchestrator **injects** `date` into the three date tools, `city` (the district's nearest city) into the city tools, and `district` into the snapshot, in addition to `before` and `district` for `search_knowledge`. PLAN.md only injects the retrieval filters. Without this the model could ask for a future date and read tomorrow's data. The tool schemas still list these arguments, and the trace shows the effective values.
- Phase 5: the tools are built per run by `build_tools(store, kb)` (closures over a `DataStore` and the knowledge base), so there is no module-level `ANALYST` constant. `analyst.build_analyst()` returns the `AgentSpec`. `TOOL_SCHEMAS` is module-level as planned.
- Phase 5: `get_city_forecast` returns `"p90 below 60"` (not just "below") for the third hint value. It never returns the forecast table's `actual` column (that would leak the future). `get_district_snapshot` also returns `nearest_city`.
- Phase 5: `get_model_reliability` also returns `test_seasons` and `n_forecasts`, so the briefing may quote "2024 and 2025".
- Phase 5: Sentinel `priority_score` = rank(fire_count) + rank(ratio to 7-day mean) + rank(nearest-city p90). Missing values (no forecast outside the test seasons) rank last. `run_daily_watch` briefs only **flagged** districts, so a quiet day produces a watch file with no briefings.
- Phase 5: the Verifier compares magnitudes (signs are ignored, so "fell 68%" matches -68.3). It cannot tell "rose" from "fell", or catch a sentence that uses a real number wrongly. The first live run showed this ("ranking 3 among Punjab districts out of a total of 273 detections"). I fixed it in the Analyst prompt, not the verifier.
- Phase 5: numbers glued to letters (PM2.5, PM10, p10, N95, K012) are not extracted. ISO dates in the text are checked part by part (year, month, day) against the dates in the evidence.
- Phase 5: the template needs a citation, so the orchestrator fetches the `limitations` chunk directly from the index (`limitations_chunk`, no embedding), and it counts as retrieved evidence. The template therefore works even if fastembed is unavailable.
- Phase 5: `BriefingRecord` has extra fields: `nearest_city`, `first_attempt_errors`, `llm_error`, `label`. The watch file also stores the whole 22-district watchlist, `n_flagged` and `top_n`.
- Phase 5: `llm.py` echoes back `extra_content` on tool calls. **Gemini 3.x requires its `thought_signature`**, or the next request fails with a 400. This is not in PLAN.md.
- Phase 5: `brief --sample N` samples from the district-days the Sentinel flags in the two test seasons (seed 42). Output goes to `data/processed/sample_run/` and `data/processed/brief_sample_report.json`, apart from the demo briefings. `brief` also takes `--top-n` and `--no-llm`.
- Phase 5: `LLM_MODEL=gemini-2.5-flash` (the default in PLAN and AGENTS.md) returns a 404 ("no longer available to new users"). `gemini-3.8-flash` returned 503 "high demand" for every request. All live runs used `gemini-3.5-flash`, set per command with the `LLM_MODEL` environment variable. `.env.example` was changed by the user to `gemini-3.8-flash`.

## Open questions / user actions needed
- **Decided 2026-11-26 (decision 1):** the user approved the University of Maryland server for MCD64A1, with new deps `rasterio` and `paramiko`. Add them and log the decision in AGENTS.md in Phase 9. **Done (D22).**
- **Decision 2 resolved (2026-11-27): district counts found for all four seasons.** 2022 and 2023: [PPCB action taken report, 19 Mar 2024](https://www.greentribunal.gov.in/sites/default/files/news_updates/Action%20Taken%20Report%20on%20behalf%20of%20Punjab%20Pollution%20Control%20Board%20in%20OA%20No.%20632%20of%202023%20%28In%20re%20%20News%20item%20appearing%20in%20Hindu%20dated%2006.10.2023%20....pdf), PDF page 57. 2024 and 2025: the PPCB reports in `docs/feasibility.md` §3. **Entered in `official_counts.csv` in Phase 9.**
- **Settled 2026-11-27 (Opus, D24):** district CRM machine counts go in `data/manual/crm_machines.csv` in Phase 12, as context only (Dispatcher tool and page), never in the allocation maths. Fleet size stays a user input (D19).
- **Settled 2026-11-27 (Opus, D23): gap method revised** (PLAN.md Phase 10). Burned area is the only reference; S-NPP and official counts are what gets measured against it; night share and MODIS-vs-S-NPP only set confidence; NOAA-20/21 are a robustness check. Range over count series × base season; no clipping; small-number rule; confidence never "high". Design check on Phase 9 data (Punjab, not a result): 2023 0.19–0.26, 2024 0.30–0.50, 2025 0.77–0.84. At district level, 15 of 22 districts pass the small-number rule in 2024 and 16 in 2025 (threshold applied to the season only); the 50 km² rule on both seasons will drop a few more. The range excludes MCD64A1's own error, so it must always be shown with that caveat.
- **Closed 2026-11-27 (Opus):** official counts. The CSV is generated from the three report PDFs, every season matches its stated total, and the 2022–23 page was checked against the scanned image. No licence is stated; each row cites its address.
- **Settled (Phase 10):** official counts are used as published. The 2022 and 2023 tables cover 15 Sep to 30 Nov; the 2024 and 2025 tables give no start date. `method_note` carries this, and any gap `methods` entry using official counts must say so. No adjustment.
- **Closed 2026-11-27 (Opus):** Agmarknet arrivals not pursued. Phase 11 uses the proxies (`docs/feasibility.md` §4). Reopen only if you download arrivals by hand.
- **Closed 2026-11-27 (Opus):** Open-Meteo attribution. The app credits line (every page), the Evidence sources table and the README now link to open-meteo.com and CC BY 4.0, as https://open-meteo.com/en/licence asks.
- **Settled (Phase 10 task):** open the ISRO *Current Science* paper before writing `evening_burning_reports.md`. If it can't be opened, the note cites only the news reports that were opened, attributes the claim to them, and says the paper itself was not read.
- **Closed 2026-11-27 (Opus):** the negative forecast result stays as it is on the Evidence page. No extra experiment.
- **Closed 2026-11-27 (Opus):** Replay map checked in headless Chrome: it renders, and Play advances the date (2025-11-05 to 2025-11-13 in 6 s). Fixed a stale second "Play" button that showed while playing (the button now has a stable key).
- **Closed 2026-11-27 (Opus):** the demo stays without the Planner. There is no evidence plans improve briefings, and each costs an extra LLM call.
- **Closed 2026-11-27 (Opus):** screenshots taken from the clean copy (no `.env`) and saved to `docs/screenshots/` (home, Replay map, Forecast, watchlist, Ludhiana briefing, Evidence). Four are shown in the README.
- **Closed 2026-11-27 (Opus):** clean-copy check. AI tools may not run `git clone` (D12), so the files that will be committed (`git ls-files -co --exclude-standard`) were copied to an empty folder, then `uv sync`, `pytest` (210 pass) and the app with no `.env` and no keys in the environment. All five pages load without errors, and the Ludhiana briefing shows with its verified badge, trace and sources. After your commit, the repo has exactly these files.
- **Closed 2026-11-27 (Opus):** read the Ludhiana and both Tarn Taran briefings as a reader. Figures are right (e.g. Ludhiana's "rise" is 11 detections against a 7-day mean of 4.3), and every cited chunk contains the lower-bound caveat it supports. The "rank 1 ... out of a total state count of 42" wording stays.
- **Closed 2026-11-27 (Opus):** kept. It is a real, verified agent run (revised once), so it belongs in the reliability table.
- **Closed 2026-11-27 (Opus):** kept as they are. Each briefing records its model, and the app shows it.
- **Closed 2026-11-27 (Opus):** looked at Forecast, Watch & Briefing and Evidence in screenshots. The Plotly band, the "NAAQS 24 h" line, the green best-value cells, the two-column briefing layout and the `[K101]` citations all render and read well.
- **Closed 2026-11-27 (Opus):** Evidence wording kept. It is accurate.
- **Closed 2026-11-27 (Opus):** already committed: `briefings/` (9), `sample_run/briefings/` (12), `watch/` (3).
- **Closed 2026-11-27 (Opus):** `.env` already has `LLM_MODEL=gemini-3.5-flash-lite` (checked without printing keys).
- **Closed 2026-11-27 (Opus):** flash-lite is the default (above). Ollama on the P6000 remains an untested option, not a task.
- **Closed 2026-11-27 (Opus):** superseded by the Phase 7 demo cache, which is committed.
- **Closed 2026-11-27 (Opus):** done with the Phase 7 reading above.
- **Closed 2026-11-27 (Opus):** explained. The FIRMS FAQ says new S-NPP production ceased on 2 November 2026, and Terra observations cease on 1 December 2026. No effect on 2022–2025. Any future season must use NOAA-20/21 (and Aqua while it lasts).
- **Closed 2026-11-27 (Opus):** curated notes checked against pages that were opened. **Confirmed:** VIIRS 375 m, low/nominal/high classes, S-NPP 13:30 / 01:30 mean local time (FIRMS FAQ; the note now says mid-latitudes get 3–4 looks a day near those times); CAMS global 0.4° (~45 km), from August 2022 (Open-Meteo docs); CAMS uses GFAS fire emissions (atmosphere.copernicus.eu); NAAQS PM2.5 60 (24 h) and 40 (annual), G.S.R. 826(E) of 16 Nov 2009 (scanned notification); GRAP Stages II–IV with AQI ranges and the CAQM name (newsonair.gov.in, Dec 2024); the 15–20 day sowing window (Punjab status report to the NGT, 2024). **Removed because no page could be read:** the WHO 15 and 5 µg/m³ figures (WHO's table is an image, other copies refused access; the note keeps only "much lower than India's standard" and the confirmed 3–4 exceedance days), the Stage I band name, and the harvest and sowing months. `grap.md`, `standards_pm25.md` and `crop_residue_context.md` now cite exact pages. KB rebuilt: chunk ids unchanged (saved briefings still resolve), retrieval eval unchanged (hybrid 17/17, 16 at rank 1).
- **Closed 2026-11-27 (Opus):** nothing to do.

## Review notes (Reviewer: GPT 6 Luna)
- Phase 2: `src/psw/app/pages/1_Replay_Map.py:54` builds the fire layer with a Python loop over every detection for the selected day. This conflicts with the Phase 2 review checklist requirement to avoid Python loops over fire rows; construct the layer data with vectorized columns instead.
  - **Resolved:** the fire layer is now built from vectorised pandas columns (`radius`, `tip`) and passed as a DataFrame. The only remaining loop is over the 22 district polygons. Both layers now use one top-level `tip` field for the tooltip (`{tip}`). Re-run the app and recheck the tooltips.
