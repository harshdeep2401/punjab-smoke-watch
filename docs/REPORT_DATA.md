# Punjab Smoke Watch: consolidated project data for the report and slides

> **Instructions for the AI writing the report and the slides**
> - This document is the **only source of truth**. Use only the facts and numbers given here. Do not invent results, citations, user studies or accuracy figures. If something is missing, write `[TO FILL]`.
> - Report results **as they are**. The main forecasting result is negative (the fire features did not help, and persistence was best). Present this honestly as a finding, not a failure.
> - Section 13b describes the **Part 2 direction (Audit + Response)**, which is planned but not built. Present it as the project direction and ongoing work, and keep every `[TO FILL]` result as a placeholder unless the user supplies numbers.
> - Suggested report outline: §16. Suggested slide outline: §17.
> - The author name, institution, course and date are `[TO FILL]`.

---

## 1. One-line summary

**Punjab Smoke Watch** is an agentic, RAG-grounded pipeline. It connects NASA satellite fire detections in Punjab, India, to next-day PM2.5 forecasts (with uncertainty bands) for four cities, and it writes a cited, **number-verified** daily briefing for any district. The LLM never does the maths: every number comes from Python code, and a deterministic Verifier rejects any number the code didn't produce.

It is **not a chatbot**. It has no free-text Q&A box. The core is a data pipeline, a forecasting model with a backtest, and a verified report generator.

## 2. Problem statement

Every October and November, crop-residue (paddy stubble) burning in Punjab coincides with severe smog across north India. Three things make this hard to reason about:

1. **The chain is invisible.** People can't easily see *fires → wind → PM2.5*. Fire maps, weather and air-quality dashboards live in separate places.
2. **Detection is uncertain.** Satellites only see a fire if it is burning during an overpass and isn't under cloud, so a single confident fire count is misleading.
3. **Decision makers need a short read.** Officials and the public need a district-level "what is happening and what is likely tomorrow", not raw CSVs.

## 3. Objectives / research questions

1. Build a reproducible pipeline that joins fires, weather and PM2.5 at the district and city level.
2. **RQ1:** does an *upwind fire-pressure* feature improve next-day PM2.5 forecasts over persistence and over the same model without fire features? This is answered with a time-based backtest, not a claim.
3. **RQ2:** can an LLM agent write a daily district briefing that a reader can trust because every number and citation in it is checked automatically?
4. Ship a Streamlit app that runs offline from cached data, for demos.

## 4. Scope

- **In scope:** 22 Punjab districts; 4 cities (Ludhiana, Amritsar, Patiala, Bathinda); fire seasons 2022–2025 (Sep 15 to Nov 30); next-day PM2.5 forecasts; agent briefings.
- **Out of scope:** identifying individual farmers or fields, enforcement use, health advice, hardware, user accounts.
- **Designed as a one-day build,** with phased implementation (§13).

## 5. Data sources

| Need | Source | Access / notes |
|---|---|---|
| Active fires | **NASA FIRMS**, VIIRS S-NPP 375 m. Archive product `VIIRS_SNPP_SP` for 2022–2025. | Free MAP_KEY. Area API; **max 5 days per request** (measured). Low-confidence detections dropped. |
| Weather | **Open-Meteo Historical** (ERA5 reanalysis, Copernicus C3S) | No key. Hourly temperature, humidity, 10 m wind speed and direction, boundary layer height. Archive lag measured at 1 day. |
| PM2.5 (the forecast target) | **Copernicus Atmosphere Monitoring Service (CAMS)** global forecasts via the **Open-Meteo Air Quality API** | No key. **Modelled, not a ground monitor.** Earliest usable data ≈ 2022-08-02 (2020–2021 return nothing). |
| District boundaries | **geoBoundaries IND ADM2** (gbOpen), licence ODbL 1.0 | 22 Punjab districts. **Malerkotla** (created 2021) is not in any source we checked, so it is counted inside Sangrur. |
| Embeddings | `BAAI/bge-small-en-v1.5` via fastembed (runs locally on CPU) | ~130 MB download, once. |
| LLM | Gemini free tier via its OpenAI-compatible endpoint (`gemini-3.5-flash`, `gemini-3.5-flash-lite`) | Swappable for Ollama or any OpenAI-compatible server through `.env`. |

Study-area bounding box for fires: lon 72.5–78.0, lat 28.0–32.8 (Punjab, Haryana, Delhi and the border belt). Fires outside Indian Punjab are kept for the upwind feature, because smoke doesn't respect borders.

### Feasibility findings (measured on 2026-11-25)

- CAMS PM2.5 via Open-Meteo starts around Aug 2022, so the training window is at most **4 full seasons (2022–2025)**.
- FIRMS archive (SP) covers 2022–2025 fully. The 2026 near-real-time feed stopped at about Nov 2, 2026, so **2026 was excluded**.
- FIRMS allows a maximum of 5 days per request (not 10 as first assumed), so requests are chunked.
- All weather variables were fully available for 2020–2026.

## 6. Data at a glance (computed from the processed data)

- **Fire detections:** 139,419 in the bounding box after dropping 6,707 low-confidence rows. **88,487** of these fall inside the 22 Punjab districts.
- **Ingest scale:** 1,472 daily PM2.5 and weather rows each (4 cities × 92 days × 4 seasons), with 0 missing values. The first ingest took about 3 minutes; a cached rerun takes about 1 second.

**Fire detections in Punjab per season (Sep 15 to Nov 30)**

| Season | Detections | Peak day (detections) | Top 5 districts (detections) |
|---|---|---|---|
| 2022 | 45,387 | 2022-11-11 (3,086) | Sangrur 5,273; Bathinda 4,428; Fazilka 3,516; Sri Muktsar Sahib 3,493; Moga 3,153 |
| 2023 | 30,390 | 2023-11-05 (3,005) | Sangrur 5,002; Bathinda 2,615; Ferozepur 2,401; Moga 2,286; Fazilka 2,054 |
| 2024 | 8,672 | 2024-11-18 (963) | Sangrur 1,476; Ferozepur 800; Tarn Taran 676; Bathinda 658; Sri Muktsar Sahib 657 |
| 2025 | 4,038 | 2025-10-31 (273) | Tarn Taran 621; Sangrur 465; Amritsar 353; Fazilka 320; Ferozepur 302 |

Detections fell by about **91% from 2022 to 2025**. The hotspot districts (Sangrur, Bathinda, Ferozepur, Moga, Tarn Taran) match public reporting.

**City PM2.5 (CAMS, season Sep 15 to Nov 30): mean / max daily mean, µg/m³, and days above the 60 µg/m³ NAAQS 24 h standard**

| Season | Amritsar | Bathinda | Ludhiana | Patiala |
|---|---|---|---|---|
| 2022 | 78.7 / 187.6 / 60 d | 79.2 / 159.4 / 52 d | 66.6 / 120.3 / 49 d | 64.0 / 125.1 / 45 d |
| 2023 | 57.4 / 115.9 / 32 d | 61.3 / 150.5 / 31 d | 51.5 / 101.2 / 25 d | 49.1 / 91.3 / 25 d |
| 2024 | 80.3 / 260.8 / 45 d | 75.0 / 253.0 / 37 d | 75.8 / 202.3 / 45 d | 68.1 / 172.0 / 36 d |
| 2025 | 74.2 / 150.5 / 44 d | 72.0 / 168.2 / 41 d | 71.2 / 158.8 / 50 d | 71.4 / 140.1 / 52 d |

**Key observation (good for a slide):** fire detections fell about 91% from 2022 to 2025, but season-mean PM2.5 in the four cities did **not** fall in step (2024 and 2025 are similar to 2022). This is descriptive only. Other sources (weather, traffic, industry, regional transport, and fires missed by satellite) also drive PM2.5. It is consistent with the forecasting result in §10.

## 7. System architecture

```
 NASA FIRMS ─┐
 Open-Meteo ─┼─> ingest (cached) ─> districts + daily tables ─> features (upwind pressure) ─> LightGBM p10/p50/p90
 CAMS PM2.5 ─┘                                                                                    │
                                                                                                  v
 curated notes ─┐                                                              forecasts.parquet, metrics.json
 generated docs ┴─> chunks + embeddings (as_of dates) ──────────────┐                            │
                                                                     │                            │
 SENTINEL (rules) ──> ranked watchlist ──> ANALYST (LLM loop) <──── tools ────────────────────────┘
                                              │   tools: district snapshot, city forecast, wind & upwind,
                                              │          model reliability, search_knowledge (RAG, as_of filter)
                                              v
                                       draft {headline, briefing, citations}
                                              v
                                    VERIFIER (rules, no LLM)
                                     ok ──────────────> saved briefing
                                     fail ─> one revision with the error list ─> Verifier again
                                     fail again / LLM down ─> fixed TEMPLATE (also verified)
                                              v
                                    Streamlit app (reads data/processed/ only; works offline)
```

**Pipeline commands:** `ingest → process → train → kb → brief → demo-cache`, all run as `uv run python -m psw.pipeline <cmd>`.

## 8. Methods

### 8.1 Ingestion and processing
- Every API call is cached on disk, so reruns make no network calls. A shared retry/backoff helper scrubs the API key from error messages.
- Fires are assigned to districts by point-in-polygon spatial join (GeoPandas). The daily district table is a full district × day grid with zero-fill, holding the fire count, total FRP (MW) and a trailing 7-day mean.
- Daily wind direction is a **vector mean** (average of u and v, then atan2), not an arithmetic mean of degrees, which would be wrong across 0°/360°.

### 8.2 Upwind fire pressure (the main original feature)
For each city and day:

**upwind_frp = Σ FRP · max(0, cos θ) · exp(−d / 150 km)**, over fires within d ≤ 300 km

- θ is the angle between the direction the wind blows *to* and the bearing from the fire to the city. A fire directly upwind gets full weight; a fire downwind or crosswind gets zero.
- The exp term decays the weight with distance.
- Computed with lags of 0, 1 and 2 days, each using that day's wind.
- A separate helper reports the **top upwind contributing districts** (share %). These are used in briefings, not as a model feature.

### 8.3 Forecasting model
- **Target:** next-day (t+1) mean CAMS PM2.5 per city. All features use data up to day t only (no leakage, checked by a test).
- **Features:**
  - PM2.5 today, lag 1 and 3-day mean;
  - temperature, humidity, wind speed, u and v, boundary layer height;
  - **fire features:** upwind_frp lags 0–2, unweighted FRP within 300 km, and Punjab fire count today and over 3 days;
  - day of week, day of season, city (categorical).
  - Feature table: 1,232 rows.
- **Model:** LightGBM quantile regression at α = 0.1, 0.5 and 0.9 (p10/p50/p90). One model pooled across the 4 cities. Quantiles are sorted after prediction so they don't cross.
- **Calibration:** split-conformal (CQR) correction. The raw 80% band covered only 41–51% of the inner validation season, so it was widened by 8–10 µg/m³ in every fold.
- **Evaluation:** expanding-window backtest. Test 2024 (train 2022–2023) and test 2025 (train 2022–2024). No shuffling, and the test seasons are never used for tuning.
- **Variants compared:**
  - `full`: all features;
  - `no_fire`: the ablation, with the fire features dropped;
  - `persistence`: tomorrow = today, with an empirical band.
- **Metrics:** MAE of p50, mean pinball loss over the 3 quantiles, 80% interval coverage, mean band width.

### 8.4 RAG knowledge base
- **Corpus:** 113 documents in 137 chunks; the index is 0.24 MB.
  - **8 curated notes**, short and written for this project, each with a source URL:
    - how VIIRS/FIRMS detection works and what it misses;
    - what CAMS PM2.5 is;
    - the India NAAQS (60 µg/m³, 24 h) and the WHO 2021 guideline (15 µg/m³, 24 h);
    - a GRAP (Graded Response Action Plan) summary;
    - crop-residue context (the harvest-to-sowing window);
    - the upwind-pressure method;
    - limitations;
    - how to read p10/p50/p90.
  - **105 documents generated from the pipeline's own outputs:** 88 district-season fire summaries, 16 city-season PM2.5 summaries and 1 model card built from metrics.json. This is RAG over our own data.
- **Retrieval:** fastembed bge-small embeddings and numpy cosine top-k, with no vector database server. A small boost is given to the briefing's district.
- **The `as_of` rule (no leakage):** every chunk has an `as_of` date, and retrieval returns only chunks dated before the briefing date. The agent can never "read the future". This is covered by a unit test.

### 8.5 Agents (plain Python, no agent framework)
- **Runtime:** one reusable `run_agent(AgentSpec, …)` loop, about 100 lines. An agent is a system prompt plus a list of tools, the required tools and a turn limit. The loop handles:
  - tool dispatch and tool errors;
  - a "you haven't called X yet" nudge;
  - tolerant JSON parsing with one retry;
  - a full trace.
- **Sentinel** (deterministic, no LLM): scans all 22 districts for a date. It flags:
  - **spike:** fires ≥ max(15, 2 × the 7-day mean);
  - **high_count:** in the top 3 and ≥ 30 detections;
  - **smog_risk:** the nearest city's p90 forecast > 120 µg/m³.

  Districts are ranked by a priority score (rank of fire count + rank of ratio to the 7-day mean + rank of p90), with a readable reason for each flag. The Sentinel decides which districts get a briefing, so there is no user prompt.
- **Analyst** (the only LLM agent): calls 5 tools and writes `{headline, briefing, citations}` in 120–160 words. Four of the tools are required.

  | Tool | Returns |
  |---|---|
  | `get_district_snapshot` | fire count, FRP, 7-day mean, % change, rank in Punjab, Punjab total, last 7 days |
  | `get_city_forecast` | p10/p50/p90 for tomorrow, today's PM2.5, the persistence value, a hint relative to the 60 µg/m³ standard |
  | `get_wind_and_upwind` | wind direction and speed, the upwind pressure score, the top upwind districts with share % |
  | `get_model_reliability` | test-season MAE and coverage against the baselines |
  | `search_knowledge` | RAG chunks with ids such as `K101` |

  - **Anti-leakage injection:** the orchestrator, not the LLM, fixes the date, district, city and the retrieval `before` filter in every tool call. The model can't request tomorrow's data.
  - Key prompt rules: use only numbers that appear in tool results, exactly as returned; never calculate; cite chunks as [K###]; no health advice; never identify farmers or fields; present upwind fire as context, not proof of cause; mention the uncertainty band and one limitation.
- **Verifier** (deterministic, no LLM). It checks:
  - **Numbers:** every number in the text must match a number in a tool output, a retrieved chunk or a date involved, allowing for rounding. Tokens like PM2.5, p90 and K012 are not treated as numbers.
  - **Citations:** each must be a chunk that was actually retrieved, and there must be at least one.
  - **Length:** 100–180 words; the headline at most 12 words.
  - **Banned content:** health advice ("wear a mask", "N95"…) and personal names after "farmer"/"Mr"/"Smt"…
  - **Raw tool field names** such as `rank_in_punjab`.
  - It returns all errors at once.
- **Fallback chain:** on failure the Analyst gets one revision, in the same conversation, with the error list (one extra LLM call). If that still fails, or the LLM is down or rate-limited, a fixed **template** built from the same tool results is used. The template is also verified.
- **Saved record per briefing:** mode (agent / agent_revised / template), text, citations, sources, verification report, full tool trace with timings, facts, model name.
- **Gemini 3.x quirk:** tool calls return a `thought_signature` that must be echoed back, or the next request fails with HTTP 400. The client handles this.

### 8.6 Streamlit app (5 pages, runs with no API keys and no network)
1. **Home:** explanation, limitations and credits.
2. **Replay Map:** season and date slider with Play/Pause. Districts are shaded by fire count, and fire points are sized by √FRP (pydeck), with the top districts.
3. **Forecast:** per city and test season, the p10–p90 band, the p50 line, actual values, an optional persistence line and the NAAQS 60 µg/m³ line (Plotly). The selected-day card shows the upwind contributors. Labelled "out-of-sample backtest, CAMS modelled PM2.5".
4. **Daily Watch & Briefing:**
   - The live Sentinel watchlist, with reasons.
   - The selected briefing with a badge: ✅ Numbers verified / ⚠️ Template fallback / ❌ failed.
   - Tabs: **Agent trace** (each tool call, its arguments, result and time), **Sources**, **Facts JSON** and **Verifier report** (each number mapped to the tool value it matched).
   - A "How the agent works" expander, and a "Run agent" button when a key is present.
5. **Evidence:**
   - The backtest table and per-city table, the ablation chart and feature importance.
   - An interpretation generated from metrics.json, with no hard-coded numbers.
   - Agent reliability stats, data sources and credits.

## 9. Engineering facts

- **Stack:**

  | Area | Tools |
  |---|---|
  | Language and packaging | Python 3.12, uv |
  | Data | pandas, pyarrow, GeoPandas/shapely, requests |
  | Model and metrics | LightGBM, scikit-learn (metrics) |
  | RAG | fastembed, numpy |
  | LLM | openai SDK (generic OpenAI-compatible client) |
  | App | Streamlit, pydeck, Plotly |
  | Tests | pytest |

- **Code size:** about 3,300 lines in `src/` and about 1,500 lines of tests.
- **Tests:** **149 passing**, all offline. The tests use recorded API fixtures, a fake embedder and a scripted fake LLM. They cover: ingest and caching, wind maths, district joins, features and no-leakage, the model, RAG and `as_of`, the runtime loop, tools, the Verifier, the agent, LLM client handling, and the app via Streamlit AppTest.
- **Reproducibility:** processed data, the index and the demo briefings are committed, so a clean clone runs the app with no keys. Raw caches are gitignored. Secrets live only in `.env`, which is gitignored.
- **Development process (AI-assisted, phased):**
  - **Claude Opus:** architecture and the plan.
  - **Claude Sonnet:** implementation and tests, phase by phase (Phases 0–7).
  - **GPT 6 Luna:** code review against a per-phase checklist.
  - **The author:** ran all git commits, reviewed outputs and made the decisions.
  - A decisions log (D1–D12) and a progress log with every deviation were kept in the repo.
- **Phases:**
  - 0: setup and feasibility;
  - 1: ingestion and caching;
  - 2: districts and the replay map;
  - 3: features, model, backtest and ablation;
  - 4: RAG;
  - 5: agents;
  - 6: app pages;
  - 7: demo cache and README.

## 10. Results

### 10.1 Forecast backtest (RQ1)
Test seasons 2024 and 2025, out of sample. 616 next-day forecasts per variant (4 cities × 154 days).

| Variant | MAE (µg/m³) | Pinball loss | 80% band coverage | Band width (µg/m³) |
|---|---|---|---|---|
| full (with fire features) | 11.94 | 4.05 | 84.4% | 41.6 |
| no_fire (ablation) | 11.88 | 3.97 | 83.8% | 40.5 |
| **persistence** (tomorrow = today) | **11.20** | **3.83** | 76.9% | **33.9** |

**MAE by city (µg/m³)**

| City | full | no_fire | persistence |
|---|---|---|---|
| Amritsar | 13.21 | 13.14 | 12.36 |
| Bathinda | 13.10 | 13.54 | 12.02 |
| Ludhiana | 11.00 | 10.83 | 10.32 |
| Patiala | 10.46 | 10.00 | 10.09 |

**80% coverage by city:**
- full: Amritsar 83.1%, Bathinda 83.8%, Ludhiana 87.7%, Patiala 83.1%;
- persistence: 73.4%, 74.7%, 80.5%, 79.2% in the same order.

**Top features by gain** (final model): pm25_today (far ahead), then day_of_season, temp_mean, wind_speed_mean, pm25_3d_mean, punjab_fire_count_3d, rh_mean, u_mean, upwind_frp_l1, upwind_frp_l0.

**Interpretation (honest):**
- The fire features **did not improve** the forecast: full has MAE 11.94 against 11.88 for no_fire. Bathinda is the only city where full beat no_fire (13.10 vs 13.54).
- **Persistence was best** on MAE and pinball loss, but its band **under-covers** (76.9% against the nominal 80%). The model's conformal-calibrated band is **well calibrated** (84.4%), at the cost of a wider band.
- Plausible reasons (not tested):
  1. CAMS PM2.5 already assimilates satellite fire emissions (GFAS), so today's PM2.5 already carries the fire signal and persistence is a strong baseline.
  2. Only 2–3 training seasons (about 600–900 rows), and fire activity fell sharply from 2022 to 2025, so the test seasons look different from training.
  3. A daily city mean is a coarse target.
- The test seasons were never used for tuning.
- **Value delivered anyway:** calibrated uncertainty bands, upwind-contributor attribution for briefings, and a rigorous negative result.

### 10.2 Agent briefings (RQ2)
- **Sampled run:** 12 random flagged district-days from the 2024 and 2025 test seasons (seed 42, out of 779 flagged district-days). **12/12 passed the Verifier**, all written by the agent (0 needed a revision, 0 fell back to the template). The target was at least 10.
  - Models: 9 by `gemini-3.5-flash-lite` and 3 by `gemini-3.5-flash`.
  - Length: 108–147 words (mean 128).
  - Token use for 9 briefings: 18 calls, 35,862 prompt and 3,887 completion tokens. That is about 2 LLM calls and about 4,400 tokens per briefing.
- **Demo set:** 3 demo dates in 2025, chosen by rule:
  - peak 2025-10-31, with 273 detections;
  - moderate 2025-10-26;
  - early 2025-10-18, the first day with a fire-driven flag.

  On the peak day 15 of 22 districts were flagged. All 7 demo briefings (Tarn Taran, Bathinda, Moga, Ludhiana and Sangrur on the peak day, plus Tarn Taran on each of the other two days) were written by the agent and verified, using 5–6 tool calls and 16–24 numbers checked each.
- **Problems found by reading the briefings, and what was done:**
  1. A briefing wrote a raw tool field name ("ranked rank_in_punjab"), which passed because it has no number in it. **Fixed:** the Verifier now flags snake_case field names. Two offending briefings were regenerated, and the 12/12 result holds under the stronger Verifier.
  2. Wording that is correct but clumsy ("rank 1 in Punjab out of a total state count of 42"). Fixed in the prompt.
  3. **Citation support gap:** the Verifier proves that a cited chunk was retrieved, not that it supports the sentence. Example: the Ludhiana demo briefing cites a "Ludhiana fire season 2024" summary chunk after a sentence about satellite detection limits. This motivates the Critic agent (§13).
  4. Sign blindness: the Verifier matches magnitudes, so it can't tell "rose" from "fell".
  5. Some Sentinel flags are `smog_risk` only, on zero-fire days: every district near a smoggy city gets flagged.

**Example verified briefing** (Ludhiana, 2025-10-31, mode = agent, 16 numbers checked, 5 tool calls):
> *Headline:* "Ludhiana Fire Detections Rise as PM2.5 Forecast Nears 81.9 µg/m3"
> "On 2025-10-31, satellite monitoring recorded 11 fire detections in Ludhiana district, with a total fire radiative power of 41.2 MW. The district ranked 8 in Punjab, where the total reached 273 detections. Winds blew from the northwest at 6.1 km/h, with upwind fire pressure coming primarily from Outside Punjab at 61.0% and Tarn Taran at 17.5%. For the next day, the PM2.5 forecast for Ludhiana city centres around a p50 value of 81.9 µg/m3, with an 80% uncertainty band ranging from a p10 of 53.6 to a p90 of 102.5 µg/m3. Today's value stood at 81.5 µg/m3. Satellite fire detections represent a lower bound due to missed fires, cloud cover, and overpass timing limitations [K101]."

(Note for the report: [K101] here is the Ludhiana 2024 season summary, which is an instance of the citation-support gap above.)

## 11. Limitations

- Satellites detect **heat anomalies**, not "stubble burning" specifically. They miss fires under cloud or smoke and outside overpass times (S-NPP passes at about 01:30 and 13:30 local), so counts are a lower bound.
- FIRMS dates are UTC and are treated as the local day.
- PM2.5 is **modelled (CAMS)**, not measured at street level. Forecasts are for 4 cities, not every district.
- A short training window: 4 seasons (2022–2025), because the CAMS history via Open-Meteo starts in 2022. 2026 was excluded because its fire feed is incomplete.
- 22 districts: Malerkotla is merged into Sangrur.
- The fire features did not improve the forecast.
- Sentinel flags are screening rules, not alerts. The Verifier doesn't prove that a source supports a sentence, and it can't detect wrong direction words.
- The curated knowledge notes were written for the project and should be checked against their sources.
- The LLM free tier has rate limits. Demo briefings are cached so the demo doesn't depend on a live LLM.
- This is a student project, not a regulatory or health tool.

## 12. Ethics and responsible-AI design

- No individual farmers or fields are identified; data is aggregated to the district level. The banned-pattern check blocks personal names.
- No health advice; health-advice phrases are blocked.
- Upwind fire is presented as context, never as proof of cause.
- Uncertainty is shown everywhere: p10–p90 bands, coverage and detection caveats.
- Negative results are reported, not hidden.
- All data sources are credited (§15).

## 13. Deeper RAG and agents (Part 1 stretch)

| # | Feature | Status | Result |
|---|---|---|---|
| 5 | **Hybrid retrieval:** BM25 plus embeddings, merged by reciprocal rank fusion (k = 60), with the `as_of` filter applied before ranking | Done (default mode) | 17-query eval set, k = 3. hit@3: hybrid 17/17, embeddings 17/17, BM25 16/17. hit@1: 16 / 15 / 15. MRR: 0.971 / 0.931 / 0.912. Hybrid helped on exact terms ("Happy Seeder" moved from rank 3 to 1) and hurt on one query (BM25 has no stemming: "satellites" vs "satellite"). The 17 queries were written by the team after reading the corpus, so this is a sanity check, not a benchmark. |
| 6 | **Planner agent:** an optional AgentSpec with no tools. It sees only the *kinds* of flag (never numbers) and writes a plan that goes into the Analyst's first message and the trace. | Done (off by default) | 4 live briefings with the planner on were all verified (3 agent, 1 agent_revised). It adds about 1 LLM call (≈1.1k prompt + 0.3k completion tokens). **Not measured:** whether plans improve the briefings. |
| 7 | **Critic agent** (clarity, tone, unsupported citations; any rewrite must pass the Verifier again) | Planned in Phase 10 (on by default for audit briefs; may only delete, reorder or reword; every rewrite re-verified) | `[TO FILL]` |
| 8 | **Comparison agent** | Superseded by the Auditor agent's season comparisons (§13b) | n/a |

**Why no framework:** the system is small. Plain Python keeps every tool result and chunk visible to the Verifier, avoids framework API churn, and makes the agent design the author's own work. New agents cost only a prompt and a tool list.

## 13b. Part 2 direction: Audit + Response (planned 2026-11-26, not built yet)

**Status: planned. Present this as the project's direction and ongoing work. Every result below is `[TO FILL]` until the user supplies it.**

### The finding that motivates it (measured from our data)
S-NPP VIIRS detections in Indian Punjab, Sep 15 to Nov 30, by overpass:

| Season | Day detections (≈13:30 IST pass) | Night detections (≈01:30 IST pass) | Night share |
|---|---|---|---|
| 2022 | 45,305 | 82 | 0.2% |
| 2023 | 30,300 | 90 | 0.3% |
| 2024 | 8,530 | 142 | 1.6% |
| 2025 | 3,942 | 96 | 2.4% |

- Daytime detections fell about 91%, but night detections did not fall. City PM2.5 did not fall either (§6), and fire counts did not improve the PM2.5 forecast (§10).
- Public reporting since 2024 suggests burning may have shifted to after satellite overpass times. If so, official "fires down ~90%" figures may partly be a **detection artefact**, and response plans built on them would miss real burning.
- Night counts are small, so this is a lead to test, not proof.

### Reframed problem
1. **Monitoring:** nobody can say with confidence how much stubble burning has *really* declined, because the headline numbers come mainly from one satellite pass a day.
2. **Response:** authorities must place a limited number of residue-management machines (balers, Super Seeders) and awareness teams in a short harvest-to-sowing window. Decisions based on yesterday's detected fires come late and may be biased.

### Module A: Audit ("what is really happening?"), for monitoring and audits
- **Users:** CAQM/PPCB monitoring cells, researchers, journalists, citizens.
- **Signals, triangulated:**
  - several satellites at different overpass times (VIIRS S-NPP, NOAA-20, NOAA-21; MODIS Terra ≈10:30 and Aqua ≈13:30);
  - the time-of-day profile and the night share;
  - **burned area** (MODIS MCD64A1, 500 m; Sentinel-2 dNBR for a few districts if feasible);
  - official counts (hand-entered with sources; noting that they are largely satellite-based too).
- **Detection-gap range:** for each district-season, each signal implies an under-detection = 1 − (change in detections ÷ change in that signal), clipped to 0–95%.
  - The reported range is the min to max across signals, with a confidence level (low / medium / high) based on how many signals agree.
  - With fewer than 2 signals: "insufficient evidence".
  - "Ground truth" here means the best available triangulation, **never field verification**.
- **Auditor agent:** a verified, cited audit brief in 4 labelled parts:
  1. what detections say;
  2. what other signals say;
  3. the best range and how sure we are;
  4. what would settle it.

### Module B: Response ("where should help go in the next 3 days?"), decision support for authorities
- **Users:** the Punjab agriculture department, district officers, cooperative machinery banks / custom hiring centres.
- **3-day district fire-risk forecast:** LightGBM quantile, the same expanding-window backtest.
  - **Features:** recent detections, the harvest-progress signal (mandi paddy arrivals if historical data exists, otherwise proxies such as season progress and last season's curve), rain over the last and next 3 days, weather and day of season.
  - **Judged mainly on ranking:** the top-5 district hit rate and Spearman correlation.
  - **"Likely burning" view:** detections are adjusted by Module A's gap range from the previous season (no leakage), as a separate, labelled view.
- **Deterministic allocator:** distributes N units across districts by greedy marginal gain with diminishing returns, cover(u) = 1 − exp(−u/k), with a per-district cap. The fleet size and caps are user inputs.
  - Within a district, 5 km grid hotspot cells guide awareness teams.
- **Decision backtest:** for each test-season day, allocation from the forecast vs proportional-to-last-season vs an oracle that knows the actual burning. The result is the "% of oracle" covered. This is the key "does it help decisions?" number.
- **Dispatcher agent:** a verified 3-day action brief: where units go and why, the uncertainty, the detection-gap caveat, and rain to watch.
- **Support, not enforcement:** no farmer, field or village names; the finest unit is a 5 km cell; words like penalties, FIRs and enforcement are banned in dispatch briefs.

### How they connect
Audit → gap range → Response plans for *likely* rather than *detected* burning → (future) after the season, Audit checks whether burning fell where help went.

### Part 2 results (fill in when built)
- Data coverage per sensor and burned-area source: `[TO FILL]`
- Punjab detection-gap range per season, with methods and confidence: `[TO FILL]`
- Fire-risk backtest (MAE, top-5 hit rate, Spearman, coverage vs persistence and climatology): `[TO FILL]`
- Decision backtest (forecast vs proportional vs oracle, % of oracle): `[TO FILL]`
- Auditor and Dispatcher verifier pass rates: `[TO FILL]`

### Other future work
- A geostationary fire archive (GEO-KOMPSAT-2A or INSAT-3D), to see the whole day and settle the gap question.
- A season-end "did help work?" analysis (descriptive, not causal).
- The Critic agent.
- Live mode, Delhi, CPCB ground checks, Malerkotla boundaries, a local LLM (Ollama).

## 14. What is original in this project

- The **upwind fire-pressure feature** design.
- The **expanding-window backtest and ablation**, with conformal calibration.
- A **RAG corpus generated from the pipeline's own outputs**, with the leakage-proof `as_of` rule.
- The **Sentinel → Analyst → Verifier → template** agent design, with orchestrator-injected arguments.
- The **deterministic number- and citation-checking Verifier**.

Open data and libraries are credited in §15.

## 15. Credits and attribution

- NASA FIRMS (VIIRS S-NPP 375 m active fire product), https://firms.modaps.eosdis.nasa.gov
- Copernicus Atmosphere Monitoring Service (CAMS) global forecasts, via the Open-Meteo Air Quality API.
- Open-Meteo historical weather (ERA5 reanalysis, Copernicus Climate Change Service), https://open-meteo.com
- geoBoundaries IND ADM2 (gbOpen), ODbL 1.0, wmgeolab, https://www.geoboundaries.org
- BAAI bge-small-en-v1.5 embeddings via fastembed.
- India NAAQS (CPCB), and the WHO Global Air Quality Guidelines 2021.
- Libraries: pandas, GeoPandas, LightGBM, scikit-learn, Streamlit, pydeck, Plotly, the OpenAI Python SDK.

## 16. Suggested report outline

1. Title page `[TO FILL: author, institution, course, date]`
2. Abstract (150–200 words: the problem, the pipeline, RAG and agents, the honest negative forecast result, 12/12 verified briefings)
3. Introduction and problem statement (§2)
4. Objectives and research questions (§3)
5. Background: stubble burning; satellite fire detection; CAMS; PM2.5 standards; RAG; LLM agents; hallucination risk
6. Data sources and feasibility (§5, §6)
7. System architecture (§7, with a diagram)
8. Methodology: pipeline, upwind feature, model and calibration, RAG, agents and Verifier (§8)
9. Implementation and testing (§9)
10. Results: backtest, ablation, agent reliability, example briefing (§10)
11. Discussion: why the fire features didn't help; what verification does and does not guarantee
12. Limitations and ethics (§11, §12)
13. Ongoing and future work (§13)
14. Conclusion
15. References and credits (§15)
16. Appendix: the Analyst system prompt rules, the tool list, the Verifier rules, the decisions log

## 17. Suggested slide outline (about 12 slides, 5–7 minutes)

1. **Title:** "Punjab Smoke Watch: verified, uncertainty-aware stubble-fire and air-quality intelligence"
2. **Problem:** fires, wind and smog are seen separately, and single fire counts hide detection uncertainty.
3. **What we built:** a one-line pipeline graphic from data to forecast to agent briefing to app. "Not a chatbot."
4. **Data:** 4 sources, 139k fire detections, 4 seasons, 22 districts, 4 cities.
5. **Insight:** a chart of Punjab detections per season (45,387 → 4,038, −91%) against city PM2.5 season means (roughly flat).
6. **Upwind fire pressure:** the formula and a sketch of the wind-angle weighting.
7. **Forecast and uncertainty:** the p10–p90 band chart and conformal calibration (coverage 84.4%).
8. **Honest result:** the backtest table. Persistence wins and the fire features don't help, with the reasons.
9. **Agents and RAG:** the Sentinel → Analyst (5 tools, RAG with `as_of`) → Verifier → template diagram.
10. **Trust:** the example briefing with the ✅ badge, plus "12/12 verified" and what the Verifier does *not* prove.
11. **Next:** hybrid retrieval, the Planner, Critic and Comparison agents; the detection-gap panel.
12. **Limitations and credits.**

**Live demo order:** Replay map on 2025-10-31 → Forecast (Ludhiana 2025) → Evidence → Watch & Briefing (Ludhiana, the trace, sources, Verifier report).

## 18. Screenshots

There are no screenshots yet; the user has to take them. Put placeholders in the slides and report: `[SCREENSHOT: Replay map 2025-10-31]`, `[SCREENSHOT: Forecast Ludhiana 2025]`, `[SCREENSHOT: Watch & Briefing, Ludhiana, trace tab]`, `[SCREENSHOT: Evidence page]`.

## 19. Glossary

- **FRP:** fire radiative power (MW), roughly how intense a fire is.
- **VIIRS:** the satellite sensor (375 m) on S-NPP.
- **FIRMS:** NASA's fire data service.
- **CAMS:** the Copernicus atmospheric composition model.
- **p10/p50/p90:** forecast quantiles; p10–p90 is the 80% interval.
- **Pinball loss:** the error metric for quantile forecasts.
- **Coverage:** the share of actual values that fall inside the band.
- **Persistence:** the baseline forecast "tomorrow = today".
- **Ablation:** removing a feature group to measure its value.
- **Conformal (CQR):** a method to calibrate interval width from held-out data.
- **RAG:** retrieval-augmented generation.
- **BM25:** keyword ranking.
- **RRF:** reciprocal rank fusion.
- **NAAQS:** India's National Ambient Air Quality Standards.
- **GRAP:** the Graded Response Action Plan (Delhi-NCR).
