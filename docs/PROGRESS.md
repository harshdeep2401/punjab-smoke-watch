# Progress

Update this file at the end of every phase. Newest notes go at the top of each section.

**Current phase:** Part 1 (0–7) done, plus Part 1 stretch 5 (hybrid retrieval) and 6 (Planner). **Part 2 (Audit + Response) planned on 2026-11-26. Phase 8 done on 2026-11-26. Waiting for the user's go-ahead (decision point, see Open questions) before Phase 9.** See AGENTS.md §1b and PLAN.md Part 2.

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
- [ ] Phase 9: Audit data (multi-sensor FIRMS, burned area, manual CSVs, district rain)
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
- Phase 8: **D17 does not work as written.** Planetary Computer is missing Sep 2023 and Sep–Oct 2025, so the proposed burned-area path is the monthly `burndate` GeoTIFFs from the University of Maryland SFTP server (window 18). Phase 9 would add `rasterio` and `paramiko` instead of `pystac-client` + `planetary-computer` + `rasterio`. Not yet approved, and nothing was added to `pyproject.toml`.
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
- **Decided 2026-11-26 (decision 1):** the user approved the University of Maryland server for MCD64A1, with new deps `rasterio` and `paramiko`. Add them and log the decision in AGENTS.md in Phase 9.
- **Decision 2 resolved (2026-11-27): district counts found for all four seasons.** 2022 and 2023: [PPCB action taken report, 19 Mar 2024](https://www.greentribunal.gov.in/sites/default/files/news_updates/Action%20Taken%20Report%20on%20behalf%20of%20Punjab%20Pollution%20Control%20Board%20in%20OA%20No.%20632%20of%202023%20%28In%20re%20%20News%20item%20appearing%20in%20Hindu%20dated%2006.10.2023%20....pdf), PDF page 57. 2024 and 2025: the PPCB reports in `docs/feasibility.md` §3. To be entered in `official_counts.csv` in Phase 9; nothing entered yet.
- Phase 9/12: district CRM machine counts exist (`docs/feasibility.md` §5). The first Phase 8 check found none. Decide whether `crm_machines.csv` holds them; fleet size stays a user input (D19).
- **Opus (before Phase 10, not blocking Phase 9): the signals are not independent, and the gap formula needs a review.** The three VIIRS satellites share the same pass time, MCD64A1 is guided by MODIS active fires, and official counts are satellite-based. A rough state-level run of the PLAN formula for 2025 against 2022 gives implied under-detection of about 0.13 (official), 0.84 (burned area) and 0.92 (night detections, which rest on 82 and 96 counts). So the range would be very wide, and "confidence = number of signals agreeing in sign" would overstate it. This arithmetic is a design check, not an audit result. I have not changed the interface.
- User (Phase 9, if approved): the PPCB reports state no licence. We would cite the address on each row of `official_counts.csv`. Malerkotla's row gets added to Sangrur (D10).
- User (Phase 8): optional, only if you want a real harvest signal: download historical paddy arrivals from the Agmarknet site in a browser. It refused my script (HTTP 403).
- Phase 13: Open-Meteo data is CC BY 4.0 and asks for a link where it is shown. The app names Open-Meteo but does not link to it, and neither README nor app states the licence.
- Not verified: the ISRO *Current Science* paper (only two news reports about it were opened). Open it before writing `evening_burning_reports.md` in Phase 10.
- User (Phase 3): decide whether the negative result is acceptable for the demo. Plan: the Evidence page shows it as is. Say if you want an extra experiment (not tuned on test seasons) before Phase 4.
- User (Phase 2): run `uv run streamlit run src/psw/app/main.py`, open Replay Map, and check that it looks right and Play works.
- User (Stretch 6): the committed demo briefings were written **without** the planner, so the Plan block does not show in the app yet. To show it, run `uv run python -m psw.pipeline brief --date 2025-10-31 --district Ludhiana --plan` (overwrites that one saved briefing; about 3 LLM calls), or `demo-cache --plan --force` for the whole set. Read the result first: I have no evidence yet that plans make briefings better.
- User (Phase 7): **take the screenshots** (home, Replay map, Forecast, Watch & Briefing on 2025-10-31 with the Ludhiana briefing, Evidence) into `docs/screenshots/` if you want them in the README or slides. I could not.
- User (Phase 7): **check a clean clone** (needs your last commit to include `data/processed/`): `git clone . /tmp/psw-check && cd /tmp/psw-check && uv sync && uv run streamlit run src/psw/app/main.py` with no `.env`. Click through all four pages and the Ludhiana briefing.
- User (Phase 7): read the Ludhiana and the two Tarn Taran demo briefings (`data/processed/briefings/2025-10-31_Ludhiana.json`, `2025-10-18_...`, `2025-10-26_...`) as a reader. They pass the Verifier; I read them and found no wrong figures, but the Verifier cannot judge the wording. One says "rank 1 in Punjab out of a total state count of 42", which is clumsy but correct.
- User (Phase 7): `briefings/2025-10-23_Sangrur.json` is an old test run (agent_revised, verified) that is not part of the demo set. Keep it or delete it. It counts in the reliability table.
- User (Phase 7): the demo briefings come from two models: the peak-day Tarn Taran, Bathinda and Moga from `gemini-3.5-flash`, the rest from `gemini-3.5-flash-lite`. Run `uv run python -m psw.pipeline demo-cache --force` if you want one model throughout (it spends free-tier quota).
- User (Phase 6): run `uv run streamlit run src/psw/app/main.py` and look at Forecast, Watch & Briefing and Evidence. Things I could not see: the Plotly band and the "NAAQS 24 h" label, the green best-value highlighting, the layout of the two columns, and whether the briefing text reads well with `[K103]` citations.
- User (Phase 6): the Evidence page tells the reader the Verifier does not check that a source supports a sentence, and that Sentinel flags can come from `smog_risk` alone. Say if you want that wording changed.
- User (Phase 6): decide what to commit under `data/processed/`. The page now reads both `briefings/` (4 files) and `sample_run/briefings/` (12 files). If you commit only `briefings/`, the reliability table shows one row. Phase 7 `demo-cache` should decide the final set.
- User (Phase 5): **set `LLM_MODEL=gemini-3.5-flash` in `.env`.** It still says `gemini-3.8-flash`, which returned 503 "high demand" on every request when I tested it. The key itself works. Tested live with `gemini-3.5-flash`: plain, JSON mode, tool calls, and a full agent run.
- User (Phase 5): `gemini-3.5-flash` hit its free-tier quota (429 after all retries) during the first sample try, so I used `gemini-3.5-flash-lite` for the rest. Consider setting `LLM_MODEL=gemini-3.5-flash-lite` in `.env`. The retry backoff only costs time. Ollama on the P6000 is untested (you said you would set it up).
- User (Phase 5): `data/processed/briefings/` and `watch/` hold the real 2025-10-31 watch (3 briefings written by the agent) and one template briefing for Sangrur from the `--no-llm` check (it is overwritten by the next run for the same district-day). `data/processed/sample_run/` and `brief_sample_report.json` come from the sample run. Decide what to commit. Phase 7 `demo-cache` rebuilds the demo set anyway.
- User (Phase 5): read 2 or 3 saved briefings (`data/processed/briefings/*.json`) as a reader. The Verifier proves the numbers exist in the tool output. It does not prove the sentences use them correctly.
- User: the 2026 FIRMS feed looks stale after about Nov 2. Decide if that matters (it only affects the stretch Live mode).
- User (Phase 4): **review `kb/curated/*.md` for factual accuracy.** I wrote them from memory and did not browse. Check especially: the S-NPP overpass times (about 01:30 / 13:30), the CAMS note (coarse grid, GFAS fire emissions), NAAQS 60 and WHO 15 µg/m³ (24 h) and the annual values, the GRAP stage names (Poor / Very Poor / Severe / Severe+) and that GRAP covers Delhi-NCR, and the paddy harvest / wheat sowing dates. The `source_url` for `grap.md` (caqm.nic.in) and `crop_residue_context.md` (agri.punjab.gov.in) are site roots, not exact pages. Replace them with the exact page.
- User (Phase 4): the first `pipeline kb` run downloaded the bge-small model (about 130 MB) to the local cache. Done on this machine already.

## Review notes (Reviewer: GPT 6 Luna)
- Phase 2: `src/psw/app/pages/1_Replay_Map.py:54` builds the fire layer with a Python loop over every detection for the selected day. This conflicts with the Phase 2 review checklist requirement to avoid Python loops over fire rows; construct the layer data with vectorized columns instead.
  - **Resolved:** the fire layer is now built from vectorised pandas columns (`radius`, `tip`) and passed as a DataFrame. The only remaining loop is over the 22 district polygons. Both layers now use one top-level `tip` field for the tooltip (`{tip}`). Re-run the app and recheck the tooltips.
