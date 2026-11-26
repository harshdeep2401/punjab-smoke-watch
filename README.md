# Punjab Smoke Watch

An agentic, RAG-grounded pipeline that connects satellite fire detections in Punjab to next-day PM2.5 in four cities, and writes a cited, number-checked briefing for any district-day.

It is not a chatbot and has no free-text Q&A box. **The LLM never does the maths:** every number comes from a Python tool, and a deterministic Verifier checks that every number in a briefing appears in the tool outputs.

## The problem

Every winter the fires, the wind and the smog in north India are visible separately, but they are rarely connected. Single fire counts also hide how uncertain satellite detection is. This project asks two questions and answers them with a backtest rather than a claim:

1. Does upwind fire pressure help forecast next-day PM2.5 in Ludhiana, Amritsar, Patiala and Bathinda?
2. Can an LLM write a daily district briefing that a reader can trust, because every number and source in it is checked?

## What it does

1. **Pipeline.** NASA FIRMS fires (VIIRS S-NPP), Open-Meteo weather (ERA5) and CAMS PM2.5 are joined, assigned to 22 districts, and turned into daily tables and an **upwind fire-pressure** feature.
2. **Model.** LightGBM quantile regression (p10 / p50 / p90) for next-day PM2.5, with an expanding-window backtest on two out-of-sample seasons and an ablation against persistence and a no-fire-features model. The 80% band is calibrated with conformal prediction.
3. **RAG.** A local index (fastembed embeddings + numpy, plus BM25 keyword search merged by reciprocal rank fusion, no server) over curated method and policy notes and over documents the pipeline writes about itself (season summaries, the model card). Every chunk has an `as_of` date, and retrieval only returns chunks older than the briefing date, so it cannot see the future.
4. **Agents.** A Sentinel scans every district, an Analyst writes the briefing with tools and RAG, and a Verifier checks it (see below).
5. **App.** Streamlit with four pages: Replay map, Forecast, Daily Watch & Briefing, and Evidence.

## Architecture

```
 NASA FIRMS ─┐
 Open-Meteo ─┼─> ingest (cached) ─> districts + daily tables ─> features (upwind pressure) ─> LightGBM p10/p50/p90
 CAMS PM2.5 ─┘                                                                                    │
                                                                                                  v
 curated notes ─┐                                                              forecasts.parquet, metrics.json
 generated docs ┴─> chunks + embeddings (as_of dates) ──────────────┐                            │
                                                                     │                            │
 SENTINEL (rules) ──> ranked watchlist ──> ANALYST (LLM loop) <──── tools ────────────────────────┘
                                              │   calls: district snapshot, city forecast, wind & upwind,
                                              │          model reliability, search_knowledge (RAG, as_of filter)
                                              v
                                       draft {headline, briefing, citations}
                                              v
                                    VERIFIER (rules, no LLM)
                                     ok ──────────────> saved briefing
                                     fail ─> one revision with the error list ─> Verifier again
                                     fail again / LLM down ─> fixed TEMPLATE (also verified)
                                              v
                                    Streamlit app (reads data/processed/ only)
```

### Agent design

- **Sentinel** (deterministic): flags a district on a date if it has a fire spike, a high count (top 3 and at least 30 detections), or its nearest city's p90 PM2.5 forecast is above 120 µg/m³. It ranks districts by a priority score and gives a plain-text reason for each flag.
- **Planner** (optional, `--plan` or a checkbox in the app; LLM, no tools): sees only the district, the date and the kinds of Sentinel flag, never a number, and writes a short plan that is passed to the Analyst and shown at the top of the Agent trace tab. It costs one extra LLM call. The plan is guidance, not evidence, and a deterministic filter drops plan items that contain numbers or raw field names. If the Planner fails, the Analyst runs without a plan.
- **Analyst** (LLM tool-calling loop in plain Python, no agent framework): `agents/runtime.py` is a reusable `run_agent(AgentSpec, ...)` loop, and the Analyst is a prompt plus five tools run through it. The orchestrator fixes the date, district and city and the retrieval filters in every tool call, so the model cannot read other dates.
- **Verifier** (deterministic): every number in the text must match a number in a tool output or a retrieved chunk. Every citation must be a chunk that was actually retrieved. It also checks length, banned content (health advice, personal names) and raw field names. On failure the Analyst gets one revision, then a fixed template is used. The template is verified too.
- **What the Verifier does not prove:** it shows that a number exists in the evidence and that a cited chunk was retrieved. It cannot tell "rose" from "fell", and it does not prove that a cited source supports the sentence. Read the Sources tab.

## Quickstart (a clean clone, no keys, no network)

```bash
git clone <this repo> && cd punjab-smoke-watch
uv sync                                   # Python 3.12 via uv
uv run streamlit run src/psw/app/main.py
```

The app reads only the committed `data/processed/`. Replay, Forecast, Evidence and the saved briefings all work without a `.env`. The "Run agent" button needs an LLM key and shows a message without one.

Demo defaults: the Watch page opens on the peak day of the 2025 season with Ludhiana selected. `data/processed/demo.json` lists the three demo dates, which all have saved briefings.

## Regenerating the data

Copy `.env.example` to `.env` and fill in the keys (never commit `.env`):

| Key | Used for |
|---|---|
| `FIRMS_MAP_KEY` | NASA FIRMS (free) |
| `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL` | any OpenAI-compatible endpoint: the Gemini free tier by default, or Ollama |

```bash
uv run python -m psw.pipeline ingest       # fetch and cache raw data (data/raw/, gitignored); first run takes minutes
uv run python -m psw.pipeline process      # fires -> districts, daily tables
uv run python -m psw.pipeline train        # features, backtest, ablation, forecasts.parquet, metrics.json
uv run python -m psw.pipeline kb           # generate docs and build the vector index (downloads a ~130 MB embedding model once)
uv run python -m psw.pipeline kb-eval      # hit@3 / hit@1 / MRR for embeddings vs BM25 vs hybrid (tests/fixtures/retrieval_eval.json)
uv run python -m psw.pipeline brief --date 2025-10-31 [--district Sangrur]   # Sentinel + agent
uv run python -m psw.pipeline brief --date 2025-10-31 --district Ludhiana --plan   # with the optional Planner step
uv run python -m psw.pipeline brief --sample 12                              # verifier pass rate on sampled flagged district-days
uv run python -m psw.pipeline demo-cache   # briefings for the three demo dates (reuses saved agent briefings; --force redoes them)
uv run pytest -q                           # tests, all offline (a scripted fake LLM, no network)
```

On macOS, LightGBM needs `brew install libomp`.

## Results

Expanding-window backtest: for each test season (2024 and 2025) the model is trained only on earlier seasons. 616 next-day forecasts per variant (4 cities × 154 days). From `data/processed/metrics.json`:

| Variant | MAE (µg/m³) | Pinball loss | 80% band coverage | Band width (µg/m³) |
|---|---|---|---|---|
| full (with fire features) | 11.94 | 4.05 | 84.4% | 41.6 |
| no fire features | 11.88 | 3.97 | 83.8% | 40.5 |
| persistence (tomorrow = today) | **11.20** | **3.83** | 76.9% | **33.9** |

**The honest reading: the fire features did not help, and persistence is the best on MAE and pinball.** We did not tune on the test seasons. Plausible reasons (not tested here): CAMS PM2.5 already assimilates fire emissions, so today's PM2.5 carries the fire signal; only two to three training seasons are available and fires fall sharply from 2022 to 2025; and a daily city mean is a coarse target. The Evidence page shows this table, the by-city breakdown and the feature importance.

**Retrieval.** On a 17-query eval set, hybrid (embeddings + BM25) finds the right document in the top 3 for 17 of 17 queries, the same as embeddings alone, so the set is too easy to separate them at that level. Hybrid puts it first on 16 (embeddings 15, BM25 alone 15) and has the best MRR (0.97 vs 0.93 and 0.91). It fixed two queries that embeddings ranked lower (an exact term and a district-and-year query) and moved one down a place. The queries were written by us after reading the corpus, so this is a sanity check, not a benchmark. The Evidence page shows the table and the per-query ranks.

**Agent reliability.** On 12 sampled flagged district-days from the two test seasons (seed 42), all 12 briefings passed the Verifier, all written by the agent (none by the template, none needing a revision). That sample is small and it checks numbers and citations, not the quality of the writing. Two checks found real problems while building this: a briefing quoted a source that did not support its sentence, and two briefings printed a raw tool field name. The second is now caught by the Verifier.

## Data sources and credits

- **Fire detections:** NASA FIRMS, VIIRS S-NPP 375 m, archive product (low-confidence detections dropped). https://firms.modaps.eosdis.nasa.gov
- **PM2.5 (the forecast target):** Copernicus Atmosphere Monitoring Service (CAMS) global atmospheric composition forecasts, retrieved through the Open-Meteo air-quality API. This is modelled PM2.5, not a ground monitor.
- **Weather:** Open-Meteo historical weather (ERA5 reanalysis, Copernicus Climate Change Service). https://open-meteo.com
- **District boundaries:** geoBoundaries IND ADM2 (gbOpen), licence Open Data Commons Open Database License (ODbL) 1.0, attribution to geoBoundaries (wmgeolab). https://www.geoboundaries.org
- **Embeddings:** `BAAI/bge-small-en-v1.5`, run locally with fastembed.
- **PM2.5 reference line:** India's National Ambient Air Quality Standards, 24-hour value of 60 µg/m³.
- **Libraries:** pandas, GeoPandas, LightGBM, scikit-learn, Streamlit, pydeck, Plotly, the OpenAI Python SDK (used as a generic client).

## Limitations

- Satellites detect **heat anomalies**, not "stubble burning" specifically, and miss fires under cloud and outside overpass times. Counts are a lower bound.
- FIRMS dates are UTC; we treat `acq_date` as the local day.
- PM2.5 is modelled (CAMS), not measured at street level, and the forecast is for four cities, not districts.
- Four seasons (2022 to 2025) is a short training window, and the fire features did not improve the forecast.
- 22 districts. **Malerkotla** (created in 2021) is counted inside Sangrur because neither boundary source has it.
- Sentinel flags are screening rules. A district can be flagged on a smoggy day because of its nearest city's forecast, with no fires of its own.
- The Verifier does not prove that a source supports a sentence.
- This is a student project, not a regulatory or health tool. No individual farms or farmers are identified.

## What is ours

The upwind fire-pressure feature (`Σ FRP · max(0, cos θ) · exp(−d / 150 km)` over fires within 300 km, lags 0 to 2), the backtest and ablation, the RAG corpus built from the pipeline's own outputs with the `as_of` rule, the Sentinel, the Analyst loop, and the number- and citation-checking Verifier. Open data and libraries are credited above.

## More

`AGENTS.md` (context for AI sessions and the decisions log), `docs/PLAN.md` (the build plan), `docs/PROGRESS.md` (what was done, deviations, open items), `docs/feasibility.md` (data availability), `Punjab_Smoke_Watch.md` (the original brief).
