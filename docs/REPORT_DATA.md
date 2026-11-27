# Punjab Smoke Watch: full context for the report and the slides

> **Instructions for the AI writing the report and the slides**
> - This document is the **only source of truth**. Use only the facts and numbers given here. Do not invent results, citations, user studies, accuracy figures or quotes. If something is missing, write `[TO FILL]`.
> - **The headline of the project is Part 2** (§2, §7, §8): an *Audit* of how much burning really fell, and a *Response* planner for where help should go. The PM2.5 forecast and the briefing agents (§9–§11) are the platform it is built on.
> - Report results **as they are**, including the negative ones. Keep every caveat attached to its number. In particular:
>   - the detection gap is **not** the share of burning missed;
>   - nothing here is field-verified;
>   - the allocation score does **not** measure fires prevented.
> - Author, institution, course and date: `[TO FILL]`. Screenshots are listed in §21.

---

## 1. Summary

**Punjab Smoke Watch** is a data pipeline with AI agents. It works from open satellite and weather data and answers two practical questions about stubble burning in Punjab, India:

1. **Audit: "Has burning really fallen as much as the fire counts say?"** Official and satellite fire counts fell about 90% between 2022 and 2025. We triangulate those counts against burned area mapped from space and against the time-of-day pattern of detections. The result is a **detection-gap range** with a stated confidence.
2. **Response: "Where should limited help go in the next 3 days?"** A 3-day district fire-risk forecast feeds a fixed allocation rule. The rule spreads residue-management machines and awareness teams across districts and 5 km cells. An agent then writes a checked action brief.

**It is not a chatbot.** There is no question box. Rules decide what to analyse, Python computes every number, LLM agents only write the prose, and a deterministic **Verifier** rejects any number the code didn't produce.

**Key results**

| Result | Value | Caveat |
|---|---|---|
| Punjab S-NPP satellite detections, 2022→2025 | −91.1% | — |
| Official counts, 2022→2025 | −89.8% | — |
| MODIS burned area, 2022→2025 | **−44.9%** | Its own error in cropland is large and unmeasured here |
| Detection gap, 2025 vs 2022/2023 | **77% to 84%** fall in detections per burned km², medium confidence | Not the share of burning missed |
| Fire-risk allocation vs a perfect-knowledge allocation | **95.3%**, against 91.1% for splitting by last season and 80.5% for an equal split | Lead +4.2 points (95% interval 2.4 to 7.0). What one unit achieves is assumed. |
| Fire features in the next-day PM2.5 forecast | Did **not** help; persistence was best | An honest negative result |
| Saved agent briefs | All verified | Verification of numbers ≠ verification of meaning (§11.6) |

## 2. The problem

### 2.1 Background
Every October and November, paddy-stubble burning in Punjab coincides with severe smog across north India. Farmers burn because the window between the paddy harvest and wheat sowing is short. The state says demand for straw machines falls almost entirely in a 15–20 day sowing window. Fire counts from satellites drive policy, court reporting and public claims of success.

### 2.2 Why the counts may mislead
- Polar-orbiting satellites see a fire only if it burns during an overpass. The VIIRS satellites pass at about 13:30 and 01:30 IST; MODIS Terra at about 10:30, Aqua at about 13:30.
- **Our data:** S-NPP detections in Indian Punjab (Sep 15 to Nov 30):

  | Season | Day detections (≈13:30 IST) | Night detections (≈01:30 IST) | Night share |
  |---|---|---|---|
  | 2022 | 45,305 | 82 | 0.2% |
  | 2023 | 30,300 | 90 | 0.3% |
  | 2024 | 8,530 | 142 | 1.6% |
  | 2025 | 3,942 | 96 | 2.4% |

  Daytime detections fell about 91%, but night detections did not. Night counts are small, so this is **a lead, not proof**.
- **City PM2.5 did not fall in step.** 2024 and 2025 season means are similar to 2022 (§9.3). This is descriptive only, because many sources drive PM2.5.

### 2.3 The public debate (from sources the project read)
- **Nov 2024, The Tribune:** a CREAMS (IARI) scientist said farmers had learned the pass times, and that evening and night fires were rising. A former PPCB chairman said a real decline can't be ruled out.
- **Nov 2025, ISRO study** (Current Science 129(10)), using the geostationary SEVIRI sensor every 15 minutes, 2020–2024:
  - peak fire time moved from about 13:30 to about 17:00 IST;
  - S-NPP counts fell steeply while SEVIRI counts rose slightly.

  Its limits: a coarse 2–5 km sensor, Punjab and Haryana together, and avoidance as a motive is the authors' interpretation.
- **Dec 2025, NASA Earth Observatory:** using GEO-KOMPSAT-2A, most fires now occur at 4–6 pm, and MODIS/VIIRS-only monitoring misses many.
- **Dissent** (reported by ThePrint, Dec 2025): a PGIMER researcher called evasion claims unsubstantiated and credits enforcement. The government credits its action plans, machinery and enforcement for the roughly 90% fall.
- **Official counts are satellite-based too.** They come from the Punjab Remote Sensing Centre, so agreement with NASA counts is **not** independent confirmation. In 2024, field teams visited 10,791 of 10,909 reported sites and saw no crop burning at 5,090 (marked "under verification"). Field visits only check places where a fire was already detected.

**So the question is open, and it matters.** If counts fall partly because burning moved out of view, success claims are inflated and help gets sent to the wrong places.

## 3. Objectives and research questions

- **RQ1 (Audit):** how far do fire counts and burned area disagree, and how sure can we be?
- **RQ2 (Response):** does a 3-day fire forecast help decide where limited help should go, compared with simple rules?
- **RQ3 (Platform):** does an upwind fire-pressure feature improve next-day PM2.5 forecasts over persistence?
- **RQ4 (Trust):** can LLM agents write briefs a reader can trust, because every number and citation is checked automatically?
- **Engineering goal:** a reproducible pipeline and an app that runs offline from cached data.

## 4. Users and scope

| Module | Users |
|---|---|
| Audit | CAQM/PPCB monitoring cells, researchers, journalists, citizens checking official claims |
| Response | the Punjab agriculture department, district officers, cooperative machinery banks / custom hiring centres |
| Briefings | officials and the public (a daily district read) |

- **In scope:**
  - 22 districts;
  - fire seasons 2022–2025, Sep 15 to Nov 30;
  - 4 cities for PM2.5: Ludhiana, Amritsar, Patiala, Bathinda;
  - district and 5 km-cell planning.
- **Out of scope:** identifying farmers or fields, enforcement, health advice, hardware, user accounts.

## 5. Data sources

| Need | Source | Notes |
|---|---|---|
| Fire detections | NASA FIRMS: VIIRS S-NPP (main), NOAA-20, NOAA-21 (from 2023), MODIS Terra/Aqua | Free key; max 5 days per request; low-confidence detections dropped. 408,056 detections in the study box, 253,800 in Punjab. |
| Burned area | NASA MODIS **MCD64A1** C6.1, 500 m, monthly burn date (University of Maryland server) | All months 2022–25. Its user guide reports high omission error globally and says cropland burns are low confidence. |
| Official counts | Punjab reports to the National Green Tribunal (PRSC satellite counts), district by district, 2022–25 | Hand-checked; each season sums to the stated total. Some tables state no start date. |
| Residue machines | State status report (26 Nov 2024): 146,540 machines supplied on subsidy, 2018-19 to 2024-25 | Context only; not used in the allocation maths. |
| Weather | Open-Meteo historical (ERA5) at 4 cities and 22 district centroids | Rain, temperature, humidity, wind, boundary-layer height. |
| PM2.5 | Copernicus CAMS global forecasts via the Open-Meteo Air Quality API | **Modelled, not a ground monitor.** Available from about Aug 2022. |
| Boundaries | geoBoundaries IND ADM2 (ODbL 1.0), 22 districts | Malerkotla (created 2021) is inside Sangrur. |
| Embeddings | BAAI bge-small-en-v1.5 via fastembed (local CPU) | — |
| LLM | Gemini free tier (`gemini-3.5-flash`, `gemini-3.5-flash-lite`) through an OpenAI-compatible client | Swappable (e.g. Ollama) through `.env`. |

**Feasibility decisions (measured, not assumed):**
- 4 usable seasons. 2026 is excluded because its fire feed stops around Nov 2.
- Sentinel-2 burn index was rejected because it measured the harvest, not burning (86.6% of Sangrur exceeded the burn threshold).
- No historical mandi paddy-arrival data could be downloaded (HTTP 403), so harvest progress uses proxies.

**Detections inside Punjab by sensor**

| Sensor | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|
| S-NPP | 45,387 | 30,390 | 8,672 | 4,038 |
| NOAA-20 | 48,365 | 32,778 | 9,958 | 4,153 |
| NOAA-21 | — | 31,244 | 8,949 | 4,050 |
| MODIS | 10,565 | 8,347 | 3,853 | 3,051 |

**State totals side by side**

| Season | S-NPP detections | MODIS burned area (km²) | Official count |
|---|---|---|---|
| 2022 | 45,387 | 16,532 | 49,922 |
| 2023 | 30,390 | 14,980 | 36,663 |
| 2024 | 8,672 | 6,343 | 10,909 |
| 2025 | 4,038 | 9,102 | 5,114 |

Burned area fell much less than detections, and it **rose** from 2024 to 2025 while detections halved.

**Hotspot districts by S-NPP detections:**
- **2022:** Sangrur 5,273, Bathinda 4,428, Fazilka 3,516.
- **2025:** Tarn Taran 621, Sangrur 465, Amritsar 353.
- **Peak days:** 2022-11-11 (3,086) and 2025-10-31 (273).

## 6. System overview

```
DATA                      ANALYTICS (Python, deterministic)              AGENTS (LLM writes, code checks)        APP
FIRMS (4 sensors) ─┐
MCD64A1 burned ────┼──> Audit: trends, pass-time profile, ─────────────> Auditor ─> Critic ─┐
Official counts ───┘     gap ranges + confidence                                            │
                                  │ last season's gap                                         │
FIRMS + rain + weather ──> 3-day fire-risk forecast ──> "likely burning" ──> Allocator ───> Dispatcher ─┤──> VERIFIER ──> Streamlit
                                                                       (fixed rule) + 5 km cells       │    (numbers,     (7 pages,
FIRMS + wind + CAMS ─────> upwind pressure ──> PM2.5 p10/p50/p90 ──> Sentinel (rules) ──> Analyst ─────┘     citations,    offline)
                                                                                                             rules) ──> fail: one revision
Curated notes + pipeline-generated docs ──> hybrid RAG index (as_of dates) ──> search_knowledge (all agents)          ──> fail: template
```

**Principle:** the LLM never does the maths. Rules choose what to analyse, Python computes, agents explain, and the Verifier checks.

## 6A. How it works under the hood

### 6A.1 The six layers

| Layer | What happens | Main modules |
|---|---|---|
| 1. Ingestion | Cached API calls (see the list below). | `ingest/firms.py`, `weather.py`, `airquality.py`, `http.py`; `audit/sensors.py`, `burned_area.py`, `official.py` |
| 2. Processing | Fires are joined to districts by point-in-polygon (GeoPandas `sjoin`, EPSG:4326). A full district × day table is zero-filled. Each detection gets an IST hour and a pass label ("morning ≈10:30", "afternoon ≈13:30", "night ≈01:30"). Wind is vector-averaged (u, v then atan2). | `processing/districts.py`, `wind.py`, `features.py` |
| 3. Analytics and models | Upwind pressure and the PM2.5 quantile model; the audit gap; the 3-day fire-risk model; the allocator; the 5 km grid (UTM zone 43N, EPSG:32643). Every result is written to `data/processed/`. | `models/train.py`, `evaluate.py`; `audit/gap.py`; `response/risk.py`, `harvest.py`, `allocate.py`, `grid.py` |
| 4. Knowledge (RAG) | `build_kb` writes markdown documents with metadata (id, title, type, `as_of`, district, source URL) from the processed data, then chunks, embeds and indexes them (§6A.5). | `rag/build_kb.py`, `index.py`, `evaluate.py` |
| 5. Agents | Sentinel rules, then LLM agents in the `run_agent` loop with tools, then the Critic, the Verifier and the template fallback. Each result is saved as a JSON record. | `agents/*.py`, `llm.py` |
| 6. App | Streamlit pages read only `data/processed/` through cached loaders. They never call an API, and only call the LLM if the user presses "Run". | `app/main.py`, `common.py`, `helpers.py`, `pages/1–6` |

**Ingestion sources:**
- **FIRMS:** in 5-day chunks, using the archive (SP) product and falling back to near-real-time (NRT).
- **Open-Meteo:** hourly weather and PM2.5, turned into daily values.
- **MCD64A1:** monthly burned-area GeoTIFFs from the University of Maryland SFTP server, read with rasterio. Pixels with a burn date between Sep 15 and Nov 30 are summed per district into km².
- **Official counts and machine counts:** hand-checked CSVs with a `source_url` on every row, validated on load.
- **Retries and caching:** a shared retry/backoff helper scrubs the API key from errors. Empty results are cached too, so a rerun makes no network calls.

### 6A.2 Pipeline commands (in order), all run as `uv run python -m psw.pipeline <cmd>`

| Command | Produces |
|---|---|
| `ingest` | raw caches (gitignored) |
| `process` | `fires.parquet`, `district_daily.parquet`, `pm25_daily.parquet`, `weather_daily.parquet` |
| `train` | `features.parquet`, `forecasts.parquet`, `metrics.json`, `model_full_{0.1,0.5,0.9}.txt` |
| `audit-ingest` | `fires_multisensor.parquet`, `burned_area.parquet`, `district_weather_daily.parquet`, the manual CSVs checked |
| `audit` | `audit/*.parquet` (diurnal profile, sensor trend, night share, burned vs detected, official vs satellite, timing indicators, gap range), `audit_summary.json`, `audits/{season}_{district}.json` |
| `risk-train` | `risk_forecasts.parquet`, `risk_metrics.json`, `decision_backtest.json` |
| `kb` / `kb-eval` | `kb_vectors.npy` + `kb_chunks.parquet` (the index); `retrieval_eval.json` |
| `brief` | Sentinel watchlist and Analyst briefings: `watch/{date}.json`, `briefings/{date}_{district}.json` |
| `dispatch` | `dispatch/{date}.json` (with `--machines`, `--district`) |
| `demo-cache` | everything the demo shows; it reuses verified agent records |

### 6A.3 Life of one agent run, step by step
Example: a dispatch brief for 2025-10-28 with 500 units.

1. **Trigger.** The user (CLI or the "Run" button) or `demo-cache` asks for it. For daily briefings, the **Sentinel** picks the districts.
2. **Code computes first.** The orchestrator loads the data store and runs the **allocation rule** with the user's settings (units, cap, likely or detected). The `Settings` object is documented as "what the user chose, never the LLM".
3. **Prepare the agent.** It builds the `AgentSpec` (system prompt, tool schemas, required tools, turn limit, word range), the user message, and the **injected arguments**. The date, units, cap and likely flag here (date, district, city or season for the other agents, plus the retrieval cut-off) are merged *over* whatever the LLM sends, so the model can't change them.
4. **The `run_agent` loop** (at most 8 turns, 14 for the Dispatcher):
   - send the messages and tool schemas to the LLM;
   - if it asks for tools, run each tool in Python with the injected arguments. Tool errors are returned to the model as `{"error": …}`, so the loop never crashes. Every call (tool, arguments, result summary, milliseconds) goes into the **trace**, and every full result is kept as **evidence**;
   - if it answers without the required tools, it gets one nudge ("you have not called X yet");
   - the final answer must be JSON (`headline`, text, `citations`). On bad JSON it gets one retry in JSON mode.
5. **Verify.** The Verifier builds the evidence set from every tool result and every retrieved chunk, then checks the draft (§11.3). If it fails, the agent gets **one revision in the same conversation** with the list of errors (one extra LLM call), and the result is verified again.
6. **Critic** (on for audits and dispatches). It reads the verified draft and the cited chunk texts. A deterministic cited-figure check runs beside it. A rewrite is kept only if it adds no number and passes the Verifier again. If the Critic fails, the verified draft stands.
7. **Fallback.** If the LLM is down, rate-limited or still failing, a fixed **template** is filled from the same tool calls. It is also verified.
8. **Save.** A JSON record holds:
   - the mode (`agent`, `agent_revised`, `+critic` or `template`), headline and text, citations and sources;
   - the Verifier report (each number and what it matched), the full trace, the facts, the Critic's notes;
   - the model, the settings and the time.

   The app shows this record: the badge plus the Trace, Sources, Facts and Verifier tabs.

### 6A.4 Agent tools (pure Python functions; the LLM only chooses which to call)

| Agent | Tools (required in **bold**) | Injected by the orchestrator | Length |
|---|---|---|---|
| Analyst | **get_district_snapshot**, **get_city_forecast**, **get_wind_and_upwind**, get_model_reliability, **search_knowledge** | date, district, nearest city, retrieval cut-off | 120–160 words |
| Auditor | **get_detection_trend**, **get_diurnal_profile**, get_burned_area_comparison, get_official_vs_satellite, **get_gap_range**, **search_knowledge** | district, season, retrieval cut-off (Dec 1 of the season) | 150–200 words, 4 labelled parts |
| Dispatcher | **get_risk_forecast**, **get_allocation**, get_hotspot_cells, get_rain_outlook, get_gap_range (previous season), get_machine_stock (context only), **search_knowledge** | date, units, per-district cap, likely or detected | 150–200 words, labelled parts |
| Planner / Critic | none | — | — |

### 6A.5 Retrieval internals
- **Chunking:** about 900 characters, split on headings and paragraphs. Each chunk gets a stable id (`K###`).
- **Embeddings:** the title and the chunk are embedded together with bge-small (384 dimensions, normalised). Queries use the model's query prefix.
- **Index:** a numpy matrix (`kb_vectors.npy`) and a parquet table of the chunk texts and metadata. There is no database server.
- **Search** (`mode="hybrid"` by default):
  1. Filter out every chunk whose `as_of` date isn't before the brief's date.
  2. Rank by cosine similarity.
  3. Rank by BM25 keywords.
  4. Merge the two rankings by reciprocal rank fusion: score = Σ 1/(60 + rank).
  5. Give a small boost to chunks about the brief's district.
- The embedding model downloads once (about 130 MB) and runs locally on CPU.

### 6A.6 LLM client
- One wrapper around the `openai` SDK. Changing `LLM_BASE_URL`, `LLM_API_KEY` and `LLM_MODEL` in `.env` switches between Gemini, Ollama or any OpenAI-compatible server.
- **Free-tier friendly:**
  - at least 4 s between calls;
  - up to 5 retries with backoff on rate-limit, connection and server errors;
  - no retry on auth or bad-request errors.
- It supports tool calling and JSON mode. It echoes back Gemini 3's `thought_signature`, which the API requires for multi-turn tool use.

### 6A.7 Verifier internals
- **Number extraction:** a regex handles `1,234`, `12.5`, `40%` and signs. It skips tokens glued to letters (PM2.5, p90, K012, N95) and checks ISO dates part by part.
- **Evidence set:** every numeric value in the tool results and in the retrieved chunks, the date parts, and a few window constants (7-day, 80%, 24-hour).
- **Match rule:** a number matches if it equals an evidence value rounded to the same precision (half-up), is within 0.5 for integers ≥ 10, or equals a ratio × 100. Signs are ignored, which is a known weakness.
- **Checks shared by all briefs:** citations were retrieved; length; banned health phrases; personal names; raw snake_case field names.
- **Audit-only checks:**
  - the 4 labels are present;
  - a burned-area caveat is present;
  - gap figures and direction match `get_gap_range`;
  - no other season's figures appear as this season's.
- **Dispatch-only checks:**
  - unit figures only as "District (N units)" or "from N units to M units";
  - capitalised words must be a district, a project term or an ordinary word (it fails closed against village and person names);
  - no penalty or enforcement words;
  - next-3-day rain must be called archive data;
  - "likely burning" must not be written as detections;
  - cells must carry their detection counts;
  - a district's rain and cells must not appear under another district's name.
- It returns **all** errors at once, and they go back to the agent as the revision list.

### 6A.8 Core algorithms

**Gap:**
- gap = 1 − (count_t / count_base) ÷ (burned_t / burned_base), for {S-NPP, official} × {base 2022, 2023}, giving a minimum-to-maximum range.
- Small-number rule: ≥ 50 km² burned and ≥ 100 base detections.
- Timing indicators set the confidence:
  - the night share, with a bootstrap 95% interval;
  - MODIS change ÷ S-NPP change, which supports a timing shift if ≥ 1.5.

**Risk forecast:**
- Target log1p(detections over t+1..t+3), back-transformed to counts.
- 3 LightGBM quantile models with conformal widening from an inner validation season.
- Ranking is scored on days with ≥ 20 detections statewide.

**Likely burning:** forecast ÷ (1 − the previous season's district gap), with the gap clipped to 0–0.9.

**Allocation:**
- Hand out units one at a time. Each unit goes to the district with the largest marginal gain, risk × (e^(−u/20) − e^(−(u+1)/20)), without exceeding the cap (25% of units).
- It is deterministic and runs in milliseconds, so the planner page updates live.

**Decision backtest:**
- Score = Σ actual detections over the next 3 days × cover(units), as a share of the same score for an allocation that knew those detections.
- 95% intervals come from a 7-day block bootstrap.

**PM2.5 model:**
- 3 LightGBM quantile models (α = 0.1, 0.5, 0.9), pooled over cities, quantiles sorted so they don't cross, plus CQR widening.

### 6A.9 Repo layout
```
src/psw/
  config.py  pipeline.py (CLI)  llm.py
  ingest/      firms, weather, airquality, http (retry/backoff)
  processing/  districts, wind, features
  models/      train (quantile + conformal), evaluate (backtest, ablation)
  audit/       sensors, burned_area, official, gap
  response/    harvest, risk, allocate, grid, intervals, machines
  rag/         build_kb, index (hybrid search), evaluate
  agents/      runtime (run_agent), sentinel, analyst, auditor, dispatcher, planner, critic,
               tools / audit_tools / dispatch_tools, verifier, template / audit_template / dispatch_template,
               orchestrator, demo
  app/         main, common, helpers, pages/1_Replay_Map … 6_Response_Planner
data/  geo/ (boundaries)  manual/ (official counts, machines)  processed/ (committed)  raw/ (gitignored)
kb/    curated/ (16 notes)  generated/ (rebuilt)
tests/ (492 offline tests, fixtures, FakeLLM)   docs/ (plan, progress, feasibility, screenshots)
```

## 7. Module A: Audit ("what is really happening?")

### 7.1 Method
Each signal has a role, because the signals aren't independent:

| Role | Signals | Why |
|---|---|---|
| Counts being tested | S-NPP detections; official counts | Both are satellite counts at fixed pass times |
| The only reference | MODIS burned area | A scar stays visible after the fire, so it is less tied to the pass time (but it is partly guided by MODIS fire detections) |
| Timing indicators (Punjab only; they set confidence and never enter the range) | Night share of detections; MODIS change ÷ S-NPP change (supports a timing shift if ≥ 1.5) | MODIS passes at other times |
| Robustness check only | NOAA-20, NOAA-21 | Same pass time as S-NPP |

- **Gap** = the fall in detections per burned km² since a base season: gap = 1 − (count_t / count_base) ÷ (burned_t / burned_base).
- **Range:** the minimum to maximum over {S-NPP, official} × {base 2022, base 2023}. No clipping.
- **Small-number rule:** a pair needs ≥ 50 km² burned in both seasons and a base count ≥ 100, otherwise the result is "insufficient evidence".
- **Confidence is never "high".** Punjab is "medium" at best and districts are "low" at best. It is raised only when the timing indicators agree.

### 7.2 Results: Punjab, against 2022

| Season | S-NPP | Burned area | Official | Gap range | Confidence | Night share | MODIS ÷ S-NPP change |
|---|---|---|---|---|---|---|---|
| 2023 | −33.0% | −9.4% | −26.6% | 19% to 26% | low | 0.3% (supports) | 1.18 (does not support) |
| 2024 | −80.9% | −61.6% | −78.1% | 30% to 50% | medium | 1.6% (supports) | 1.91 (supports) |
| 2025 | −91.1% | −44.9% | −89.8% | **77% to 84%** | medium | 2.4% (supports) | 3.25 (supports) |

**How to read it:**
- Counts fell much faster than burned area. In 2025, detections per burned km² were 77–84% lower than in 2022/2023.
- This is **not** the share of burning missed, and it excludes the burned-area product's own error, which is the largest uncertainty.
- Three explanations fit: burning moved outside pass times (consistent with the timing indicators and the ISRO/NASA findings), fires became smaller or cooler, or the product's sensitivity changed.

**Districts are weaker** (all "low" at best):
- **2025:** 13 of 22 districts have a range. In 11 of them counts fell more than burned area; **Ludhiana shows no sign of a gap**; Tarn Taran is unclear. The other 9 are "insufficient evidence".
- **2024:** Amritsar and Patiala show no sign of a gap.

So the gap is not uniform, and the tool says so.

### 7.3 Output
- The Audit page: headline, season cards, pass-time charts, burned area against detections, official against satellite, and a district table.
- **Auditor agent briefs**, cited, in four labelled parts:
  1. what detections say;
  2. what other signals say;
  3. our best range and how sure we are;
  4. what would settle it (field surveys, all-day geostationary data).

## 8. Module B: Response ("where should help go in the next 3 days?")

### 8.1 Three-day district fire-risk forecast
- **Target:** S-NPP detections per district over days t+1 to t+3.
- **Model:** LightGBM quantile (p10/p50/p90) with conformal calibration. Trained on earlier seasons, tested on 2024 and 2025 (22 districts, 3,256 district-days).
- **Features:**
  - recent counts (today, 3-day, 7-day, trend);
  - harvest-progress proxies: season progress against last season, last season's curve, days since the first fires;
  - rain over the last and next 3 days;
  - weather and day of season.
- **Judged mainly on ranking**, because allocation needs the right order. Ranking is measured on the 128 days with ≥ 20 detections statewide.

| Variant | MAE (detections) | 80% coverage | Top-5 hit | Spearman | Top-5 2024 / 2025 |
|---|---|---|---|---|---|
| **LightGBM model** | 10.72 | 87.6% | **63.7%** | **0.73** | 69.4% / 57.6% |
| Persistence (last 3 days) | **9.61** | 84.4% | 59.7% | 0.66 | 61.5% / 57.8% |
| Climatology (last season, scaled) | 13.34 | 81.3% | 57.1% | 0.63 | 59.6% / 54.4% |

**Honest reading:**
- The model ranks best but isn't the most accurate. Random choice would give about 23% top-5 hits, so all three variants carry real signal.
- Its top-5 lead over persistence is **not consistent**: +3.9 points pooled (95% interval 1.1 to 7.8), but +7.9 in 2024 and −0.3 in 2025. The variant was also picked on the same test seasons. Its Spearman lead (+0.07, 95% interval 0.04 to 0.12) is clearer.
- Next-3-day rain comes from the archive, which is optimistic. Without it the ranking is unchanged.

### 8.2 "Likely burning"
- Likely burning = forecast ÷ (1 − last season's district gap), with the gap clipped to 0–0.9. It uses the **previous** season, so there is no look-ahead.
- It is the planner's default view. It is an estimate, **not a count of fires**, and it can't be checked against satellite data.
- **Example (2025-10-28):** it moves Fazilka from 34 to 57 units, on a **low**-confidence range. The 2025 multipliers reach 5.8 (Fazilka).

### 8.3 Allocation rule (deterministic; the LLM can't change it)
- Greedy marginal gain with diminishing returns: cover(u) = 1 − exp(−u/20), at most 25% of the units per district.
- The fleet size, the per-district limit and likely-vs-detected are **user inputs**.
- The share of burning one unit "covers" is an **assumption**.
- **5 km grid:** 2,200 cells. Only cells with ≥ 2 detections in the last 7 days are offered, so teams aren't sent to single detections. No place or person names appear.

### 8.4 Does it help decisions? The decision backtest
500 units, 148 issue days in 2024–25. Each allocation is scored by the detections seen in the next 3 days, weighted by cover(units), as a share of an allocation that knew them in advance:

| Allocation | All days | 2024 | 2025 |
|---|---|---|---|
| **Model forecast** | **95.3%** | 95.5% | 95.0% |
| Share of last season's fires (naive) | 91.1% | 90.9% | 91.5% |
| Climatology forecast | 90.1% | 92.8% | 84.3% |
| Persistence forecast | 87.6% | 85.1% | 93.0% |
| Equal split | 80.5% | 80.5% | 80.7% |

- The lead over the naive split is **+4.2 points** (95% interval 2.4 to 7.0), and it holds in both seasons. That is real but modest.
- The lead is largest when units are scarce: +16 points with 100 units, +4 with 500, +2 with 2,000.
- It compares allocations. It does **not** measure fires prevented.
- The "likely burning" allocation scored 94.4% against detections, which can't see what it plans for.

### 8.5 Dispatcher agent: an example committed brief
Dispatch brief, 2025-10-28, 500 units, verified:

> **Where the units go:** Of 500 units, the largest shares go to Fazilka (57 units), Sangrur (46 units), Tarn Taran (41 units), Ferozepur (40 units), and Bathinda (39 units); the other districts get smaller shares. The limit per district is 125 units.
> **Why:** Fazilka has 35 forecast detections, with an 80% band from 3 to 59. That is more than the 12 detections of the last 3 days. Its likely burning is estimated at 204. The adjustment moves Fazilka from 34 units to 57 units.
> **How sure we are:** The forecast is uncertain. Likely burning scales each forecast by the previous season's detection gap. The gap range behind Fazilka has low confidence. Likely burning is not a count of fires. What one unit achieves is assumed.
> **What to watch:** Fazilka had 0.0 mm of rain in the last 3 days. The weather archive, standing in for a forecast, shows 0.0 mm for the next 3. Teams might start at cell 1070 (north Fazilka), with 6 detections, and at cell 1022 (north-east Fazilka), with 5 detections. The state says demand for straw machines falls almost entirely in the wheat sowing window of 15 to 20 days [K434].

## 9. The platform: fires → wind → PM2.5 (Part 1)

### 9.1 Pipeline
- Cached ingestion: reruns make no API calls.
- Fires are assigned to districts by point-in-polygon join.
- Daily district × day tables with zero-fill.
- Wind direction is a vector mean.
- Every feature uses data up to day t only, and tests check this.

### 9.2 Upwind fire pressure (our feature)
**upwind_frp = Σ FRP · max(0, cos θ) · exp(−d / 150 km)** over fires within 300 km of a city, at lags of 0–2 days.
- θ is the angle between where the wind blows *to* and the bearing from the fire to the city. Upwind fires count fully; downwind and crosswind fires count zero.
- A helper also reports the top upwind districts (share %), which are used in briefings.

### 9.3 Next-day PM2.5 forecast
- **Model:** LightGBM quantile (p10/p50/p90) for 4 cities, with conformal calibration. The raw 80% band covered only 41–51% of validation data, so it was widened by 8–10 µg/m³.
- **Backtest:** expanding window, testing 2024 (trained on 2022–23) and 2025 (trained on 2022–24). 616 forecasts per variant.

| Variant | MAE (µg/m³) | Pinball | 80% coverage | Band width |
|---|---|---|---|---|
| Full (with fire features) | 11.94 | 4.05 | 84.4% | 41.6 |
| No fire features (ablation) | 11.88 | 3.97 | 83.8% | 40.5 |
| **Persistence** | **11.20** | **3.83** | 76.9% | 33.9 |

**Reading:**
- The fire features did not help, and persistence was best on error. The model's band is well calibrated, while persistence's under-covers.
- Plausible reasons (not tested):
  - CAMS already assimilates fire emissions, so today's PM2.5 carries the fire signal;
  - there are only 2–3 training seasons, during a sharp fall in fires;
  - a city daily mean is a coarse target.
- This result is part of the motivation for Part 2.

**City PM2.5 (CAMS), season mean, µg/m³** (days above the 60 µg/m³ NAAQS 24-hour standard in brackets):

| Season | Amritsar | Bathinda | Ludhiana | Patiala |
|---|---|---|---|---|
| 2022 | 78.7 (60 d) | 79.2 (52 d) | 66.6 (49 d) | 64.0 (45 d) |
| 2023 | 57.4 (32 d) | 61.3 (31 d) | 51.5 (25 d) | 49.1 (25 d) |
| 2024 | 80.3 (45 d) | 75.0 (37 d) | 75.8 (45 d) | 68.1 (36 d) |
| 2025 | 74.2 (44 d) | 72.0 (41 d) | 71.2 (50 d) | 71.4 (52 d) |

## 10. RAG knowledge base
- **Corpus:** 213 documents, 435 chunks, a 0.7 MB index, and no vector-database server.
  - **16 curated notes**, written for the project, each with a source URL. Topics:
    - how satellite fire detection works;
    - pass times;
    - the MCD64A1 product and its error;
    - how official counts are made;
    - the 2024 reporting, the ISRO study and the debate;
    - the gap method;
    - straw-management options;
    - CAMS;
    - PM2.5 standards;
    - GRAP;
    - crop-residue context;
    - the upwind method;
    - how to read forecasts;
    - limitations.
  - **197 documents generated from the pipeline's own outputs:**
    - 88 district-season fire summaries;
    - 16 city-season PM2.5 summaries;
    - 92 audit documents;
    - 1 model card.

    So the agents retrieve facts about our own data, not just background.
- **Hybrid retrieval:** BM25 keyword search plus embeddings, merged by reciprocal rank fusion. On a 23-query sanity set (written by us, so not a benchmark):

  | Mode | hit@3 | hit@1 | MRR |
  |---|---|---|---|
  | Hybrid | 23/23 | 20 | 0.935 |
  | Embeddings | 22/23 | 21 | 0.935 |
  | BM25 | 20/23 | 17 | 0.797 |

  Keyword search helps with exact terms such as "Happy Seeder" and "MCD64A1".
- **No-leakage rule:** every chunk has an `as_of` date, and retrieval returns only chunks dated before the brief's date. Tests check this.

## 11. Agents and verification (plain Python, no agent framework)

### 11.1 Runtime
One reusable `run_agent(AgentSpec, …)` loop. An agent is a prompt plus tools. The loop handles:
- tool dispatch;
- a "you haven't called X yet" nudge;
- JSON parsing with a retry;
- a turn limit;
- a full trace.

The **orchestrator injects** the date, district, units and retrieval filters into every tool call, so the LLM can't read the future or change the plan.

### 11.2 The agents

| Agent | Type | Job |
|---|---|---|
| Sentinel | Rules | Flags districts each day: a spike (≥ max(15, 2 × the 7-day mean)), a high count (top 3 and ≥ 30), or smog risk (nearest city's p90 > 120 µg/m³). It decides who gets a briefing. |
| Analyst | LLM + 5 tools | Daily district briefing (fires, forecast band, wind and upwind, model reliability, RAG) |
| Auditor | LLM + audit tools | The 4-part audit brief (§7.3) |
| Dispatcher | LLM + response tools | The 3-day action brief explaining an allocation it cannot change (§8.5) |
| Planner | LLM, no tools, optional | Plans which facts or signals to check first; sees only flag kinds or signal names, never numbers |
| Critic | LLM, no tools | Reads a verified draft together with the cited source texts, and flags unsupported citations and overclaims. It may only delete, reorder or reword; a rewrite that adds a number is rejected, and every rewrite is re-verified. |
| Verifier | Rules | See §11.3 |

### 11.3 The Verifier (deterministic)
- **Numbers:** every number must match a tool output, a retrieved chunk or a date, allowing for rounding.
- **Citations:** every citation must be a chunk that was actually retrieved.
- **Length and structure:** the length limits, and the required labelled parts.
- **Banned content:** health advice, personal and village names, and raw field names. For dispatch, also penalties, FIRs and enforcement.
- **Dispatch-specific checks:** unit figures must be written as "District (N units)"; a district's rain and cell figures can't appear under another district's name.
- **Cited-figure check:** a code check that a cited chunk actually contains the figures in its sentence.
- **Fallback:** on failure there is one revision with the error list. If that fails, a fixed **template** is used, which is also verified. The app works offline using the saved briefs.

### 11.4 Reliability (saved briefs, all verified)

| Set | Saved | First draft passed | After one revision | Template |
|---|---|---|---|---|
| District briefings, random sample (seed 42) | 12 | 12 | 0 | 0 |
| Audit briefs | 7 | 2 | 5 | 0 |
| Dispatch briefs | 3 | 2 | 1 | 0 |

- **These are not clean pass rates.** The audit and dispatch sets are what was kept after reruns and fixes.
- 3 audits carry a Critic warning, and 1 audit had a citation marker removed by hand (recorded in the file).
- A briefing costs about 2 LLM calls and about 4,400 tokens on the free tier.

### 11.5 The Critic, honestly
- On the 7 audits it raised 6 issues, all unsupported citations. 2 rewrites were accepted and 1 was rejected because it moved a citation.
- It is **weak on its own**: it missed plainly unsupported citations, once wrongly called a plain data sentence an overclaim, and never raised a missing caveat or tone issue.
- That's why the deterministic cited-figure check was added beside it.

### 11.6 Lesson: verifying numbers isn't verifying meaning
Reading live output found briefs that **passed** the Verifier but were wrong:
- one district's rain and cell figures written under another district's name;
- half a source sentence quoted, saying there were enough machines, the opposite of its point;
- an estimate ("likely burning") written as a number of detections;
- an earlier briefing citing a season summary for a ranking it didn't contain;
- a raw tool field name in the text.

Each led to a new deterministic check. The Verifier still can't catch:
- a claim with no number in it ("Sangrur gets the largest share" when it doesn't);
- "rose" vs "fell";
- every misplaced figure.

A strong discussion point for the report: **checked numbers and retrieved sources are necessary, not sufficient**, and a human still has to read the brief.

## 12. The app (Streamlit, 7 pages, runs with no API keys and no network)
1. **Home:** the two questions, limitations and credits.
2. **Replay Map:** play a season day by day, with districts shaded by detections.
3. **Forecast:** next-day PM2.5 per city with the 80% band, actuals, persistence and the 60 µg/m³ line.
4. **Daily Watch & Briefing:** the Sentinel watchlist, the briefing with a ✅ verified badge, and tabs for agent trace, sources, facts and the Verifier report.
5. **Evidence:** the PM2.5 backtest and ablation, the fire-risk backtest, the decision backtest, the audit method and limits, retrieval and agent reliability. All wording about results is computed from the result files.
6. **Audit:** "Are fires really down?"
7. **Response Planner:**
   - **live inputs:** date, units, per-district limit, likely vs detected;
   - the district map and table;
   - the allocation chart;
   - the 5 km cells;
   - the dispatch brief.

   The caption reads: "Decision support, not enforcement."

## 13. Engineering

### Tech stack, and why each piece

| Area | Tool | Used for | Why this choice |
|---|---|---|---|
| Language and environment | Python 3.12, **uv** | everything; dependency locking | One language for data, ML and app; uv is fast and reproducible |
| Data | **pandas**, **pyarrow** (parquet) | tables and committed outputs | Small, fast files that a clean clone can read |
| Geospatial | **GeoPandas**, **shapely** | district joins, 5 km grid | Standard and vectorised |
| Rasters | **rasterio**, **paramiko** (SFTP) | reading and downloading MCD64A1 GeoTIFFs | The only reliable source of all the months (Planetary Computer had gaps) |
| HTTP | **requests**, **python-dotenv** | APIs; keys in `.env` | Simple; secrets stay out of git |
| ML | **LightGBM** (quantile objective), **scikit-learn** (metrics) | PM2.5 and fire-risk forecasts | Fast on small tabular data, gives native quantiles, easy to explain |
| Embeddings | **fastembed** + BAAI **bge-small-en-v1.5** | RAG vectors | Runs locally on CPU, no API cost |
| Keyword search | **rank-bm25** | the BM25 half of hybrid search | Handles exact terms that embeddings miss |
| Vector store | **numpy** arrays + parquet | the index | Under 1 MB; no server to run |
| LLM | **Gemini free tier** through the **openai** SDK (OpenAI-compatible endpoint) | the agents | Free; swap to Ollama or another provider by changing `.env` |
| Agents | **plain Python** (`run_agent`, about 100 lines) | all LLM agents | No framework: every tool result stays visible to the Verifier, there is no API churn, and the design is ours |
| App | **Streamlit**, **pydeck** (maps), **Plotly** (charts) | the 7-page app | The quickest way to build interactive data apps in Python |
| Tests | **pytest**, a FakeLLM, a fake embedder, recorded fixtures | 492 offline tests | Deterministic, with no keys or network needed |

- **Code size:** about 9,300 lines of source and 4,600 lines of tests.
- **Tests:** **492 passing, all offline.** They use recorded API fixtures, a fake embedder and a scripted fake LLM. They cover:
  - leakage;
  - the gap method;
  - the allocator;
  - the Verifier rules;
  - every agent;
  - every app page;
  - that every committed brief still passes the current Verifier.
- **Reproducibility:** processed data, the index and the demo briefs are committed, so a clean clone runs the demo. Raw caches and `.env` (the keys) are gitignored.
- **Process:** AI-assisted and phased (Phases 0–13).
  - Claude Opus did the architecture and plan.
  - Claude Sonnet implemented and tested.
  - GPT 6 Luna reviewed each phase.
  - The author made the decisions, read the outputs and ran all git commits.
  - A decisions log and a deviations log were kept.

## 14. Limitations
- Satellites detect **heat**, not "stubble burning". They miss fires outside pass times or under cloud, so counts are a lower bound.
- **The audit is triangulation, not field verification.** MCD64A1's own error in Punjab cropland is unknown and is the biggest uncertainty. District results are low confidence, and some districts show no gap.
- **Only 4 seasons (2022–25).** CAMS history starts in 2022, and 2026 was excluded.
- **"Likely burning" is an estimate, not a count.** What one machine or team achieves is assumed, so the decision backtest compares allocations, not outcomes.
- **The fire-risk forecast:** its ranking lead over persistence is unproven, and the rain look-ahead uses the archive.
- **No harvest data:** no historical mandi arrivals, so harvest progress uses proxies.
- **PM2.5** is modelled (CAMS), only for 4 cities, and the fire features didn't help.
- **Boundaries:** 22 districts, with Malerkotla inside Sangrur. FIRMS dates are UTC.
- **The Verifier and the Critic** don't guarantee meaning (§11.6).
- **The Sentinel's flags** are screening rules; smog risk alone can flag zero-fire districts.
- **Free-tier LLM limits:** this is why demo briefs are cached.
- **The curated notes** were written for the project from the cited sources and should be checked against them.
- **It is a student project,** not a regulatory or health tool.

## 15. Ethics and responsible design
- **Support, not enforcement:**
  - nothing below a 5 km cell, and no farmer, field or village names (enforced by the Verifier);
  - the planner maps have no place names;
  - dispatch briefs may not mention penalties or enforcement.
- **No accusations:** the Auditor is told not to say farmers hide fires or that official figures are wrong, and the Critic is asked to flag it. (No code check enforces this.)
- **Uncertainty everywhere:** ranges, confidence levels, bands and coverage are shown, and negative results are published.
- **No health advice:** health phrases are blocked.
- **Context, not cause:** upwind fire is presented as context, not proof of cause.
- **All sources credited** (§18).

## 16. What is original
1. The **detection-gap method**: roles for dependent signals, a range instead of a single number, and capped confidence.
2. The **fire-risk forecast judged on ranking**, plus a **decision backtest** of the allocation rule.
3. **"Likely burning",** which links the Audit to the Response without look-ahead.
4. **Upwind fire pressure,** with an expanding-window backtest, ablation and conformal calibration.
5. **RAG over the pipeline's own outputs,** with the `as_of` no-leakage rule.
6. The **Sentinel → agent → Critic → Verifier → template** design, with orchestrator-injected arguments, and a deterministic Verifier with checks for each type of brief.

## 17. Future work
- **Settle the gap:** field surveys and an all-day geostationary fire archive (GEO-KOMPSAT-2A, INSAT, SEVIRI).
- **A season-end "did help work?" analysis** (descriptive, not causal).
- **Better inputs:** historical mandi arrivals as the harvest signal, and real weather forecasts.
- **Better checks:** a code check for claims that carry no number.
- **More coverage:** Live mode, Delhi, CPCB ground monitors, Malerkotla boundaries, and a local LLM (Ollama).

## 18. Credits and references
- NASA FIRMS (VIIRS S-NPP, NOAA-20, NOAA-21; MODIS Terra/Aqua), https://firms.modaps.eosdis.nasa.gov
- NASA MODIS MCD64A1 Collection 6.1 burned area (University of Maryland), user guide: https://lpdaac.usgs.gov/documents/1006/MCD64_User_Guide_V61.pdf
- Punjab reports to the National Green Tribunal; PPCB action taken report, 30 Nov 2024: https://ppcb.punjab.gov.in/media/documents/ATR_30.11.2024.pdf
- Singh, Pradhan, Shukla & Pandya (ISRO SAC), *Current Science* 129(10), 25 Nov 2025: https://www.currentscience.ac.in/Volumes/129/10/0921.pdf
- NASA Earth Observatory, "New timing for stubble burning in India", 15 Dec 2025: https://science.nasa.gov/earth/earth-observatory/new-timing-for-stubble-burning-in-india
- The Tribune, 7 Nov 2024: https://www.tribuneindia.com/news/punjab/to-dodge-satellite-farmers-in-punjab-resort-to-stubble-burning-during-night
- The Tribune, 6 Dec 2025: https://www.tribuneindia.com/news/india/reduced-farm-fire-claims-misleading-farmers-evading-satellites-isro-study/
- ThePrint, 3 Dec 2025: https://theprint.in/environment/are-farm-fires-going-uncounted-isro-study-suggests-farmers-have-figured-out-a-way-to-evade-satellites/2797583/
- Copernicus CAMS via the Open-Meteo Air Quality API; Open-Meteo historical weather (ERA5, Copernicus C3S), https://open-meteo.com
- geoBoundaries IND ADM2 (ODbL 1.0), https://www.geoboundaries.org
- BAAI bge-small-en-v1.5 (fastembed); India NAAQS (CPCB)
- Libraries: pandas, GeoPandas, rasterio, paramiko, LightGBM, scikit-learn, rank-bm25, Streamlit, pydeck, Plotly, the OpenAI Python SDK

## 19. Suggested report outline
1. Title page `[TO FILL]`
2. **Abstract** (150–200 words): the question; the gap range with its caveat; the decision backtest; checked agents; the negative PM2.5 result
3. Introduction: the problem and why counts may mislead (§2)
4. Objectives and research questions (§3); users and scope (§4)
5. Background: stubble burning; satellite fire detection and pass times; burned-area products; CAMS; RAG; LLM agents and hallucination
6. Data and feasibility (§5)
7. System design and architecture: the layers, the data flow, the life of an agent run, tools, retrieval and Verifier internals, the tech stack (§6, §6A, §13)
8. Method and results: Audit (§7)
9. Method and results: Response (§8)
10. The platform and the PM2.5 result (§9)
11. RAG and agents, verification and reliability (§10–§11)
12. Implementation and testing (§12–§13)
13. Discussion: what the gap does and doesn't show; what verification does and doesn't guarantee; modest but real decision value
14. Limitations and ethics (§14–§15)
15. Future work and conclusion (§17)
16. References (§18)
17. Appendix: the core algorithms (§6A.8), the Verifier rules (§6A.7), the agent tools (§6A.4), pipeline commands and outputs (§6A.2), the repo layout (§6A.9)

## 20. Suggested slide outline (14 slides, 6–8 minutes)
1. **Title:** "Punjab Smoke Watch: has stubble burning really fallen, and where should help go?"
2. **Hook:** day detections −91% while night detections stayed flat (the §2.2 table). "Has burning really fallen that much?"
3. **Why it matters:** policy and success claims rest on satellite pass times; the ISRO and NASA findings vs the dissent.
4. **What we built:** Audit + Response on a verified agent platform. "Not a chatbot; the LLM never does the maths."
4b. **Architecture:** the §6 diagram and the 6 layers (§6A.1); the tech stack (§13).
4c. **Under the hood:** the life of an agent run (§6A.3) as a flow: rules → code computes → agent calls tools → Verifier → Critic → template fallback.
5. **Data:** 4 satellites, burned area, official counts, weather; 4 seasons, 22 districts.
6. **Audit method:** the role of each signal (the §7.1 table).
7. **Audit result:** the Punjab table. 77–84% in 2025, medium confidence. "Not the share missed."
8. **Response:** forecast → likely burning → allocation rule → 5 km cells → brief.
9. **Does it help?** 95.3% vs 91.1% vs 80.5%, plus the honest ranking caveat.
10. **Trust:** the Verifier and Critic, and a brief that passed but was wrong (§11.6).
11. **Honest results and limits:** fire features didn't help PM2.5; triangulation, not ground truth; units' effect assumed.
12. **Ethics, future work, credits.**

**Live demo (5 minutes):**

| Step | What to show | Time |
|---|---|---|
| 1 | Home: the hook | 30 s |
| 2 | Audit: the 2025 card, burned area vs detections, the district table (Ludhiana shows no gap) | 60 s |
| 3 | Response Planner (opens on 2025-10-28, 500 units): change to 300 units, then switch "likely burning" off, and watch the allocation move | 90 s |
| 4 | Dispatch brief: the verified badge, then the trace, sources and Verifier report | 45 s |
| 5 | Evidence: the decision backtest | 45 s |
| 6 | Close: limitations | 30 s |

Don't press "Run" or "Regenerate" during the demo: everything shown is saved.

## 21. Screenshots (in `docs/screenshots/`)
`1_home`, `2_replay_map`, `3_forecast`, `4_watch_list`, `5_briefing_ludhiana`, `6_evidence`, `7_audit`, `8_response_planner`, `9_dispatch_brief` (all `.png`).

## 22. Glossary
- **VIIRS / MODIS:** satellite sensors that detect active fires (375 m / 1 km). S-NPP, NOAA-20 and NOAA-21 carry VIIRS; Terra and Aqua carry MODIS.
- **FIRMS:** NASA's fire data service.
- **MCD64A1:** the MODIS monthly burned-area map (500 m).
- **Geostationary satellite:** stays over one spot and sees the whole day (e.g. SEVIRI, GEO-KOMPSAT-2A).
- **FRP:** fire radiative power (MW), roughly a fire's intensity.
- **Detection gap:** the fall in detections per burned km² since a base season. Not the share of burning missed.
- **Likely burning:** a forecast scaled by last season's gap. An estimate, not a count.
- **p10 / p50 / p90:** forecast quantiles; p10–p90 is the 80% band. **Coverage** is the share of actual values inside the band.
- **Pinball loss:** the error metric for quantile forecasts.
- **Persistence:** the baseline "tomorrow = today".
- **Climatology:** the baseline "same days last season, scaled".
- **Ablation:** removing a feature group to measure its value.
- **Conformal calibration:** widening a band using held-out data so it covers what it claims.
- **Top-5 hit rate:** the share of the actual top-5 districts found in the predicted top 5.
- **Spearman:** rank correlation.
- **Oracle:** an allocation that knows the future, used as a ceiling.
- **RAG:** retrieval-augmented generation.
- **BM25:** keyword ranking.
- **RRF:** reciprocal rank fusion, a way to merge two rankings.
- **CAMS:** the Copernicus atmosphere model.
- **NAAQS:** India's air-quality standards (PM2.5, 24 h: 60 µg/m³).
- **CRM:** crop residue management (machines such as balers and Super Seeders).
- **PRSC / PPCB / CAQM:** the Punjab Remote Sensing Centre, the Punjab Pollution Control Board, and the Commission for Air Quality Management.
