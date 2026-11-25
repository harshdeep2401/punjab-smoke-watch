# Implementation plan (for the Implementer: Claude Sonnet)

Read `AGENTS.md` first. Run the phases **in order**. Each phase has tasks, interfaces, acceptance criteria and a review checklist for the Reviewer. Time budget: about 10 hours in total.

**Kick-off prompt for each phase (paste into a new Sonnet chat):**
> Read AGENTS.md, docs/PROGRESS.md and the "Phase N" section of docs/PLAN.md. Implement Phase N exactly. Run the tests. Use read-only git commands only (status, diff, log): after each green step, give me a commit message and the files to stage, and wait for me to commit. At the end, update docs/PROGRESS.md (tick tasks, note deviations, and list anything I must do).

**Review prompt (GPT 6 Luna):**
> Read AGENTS.md and the "Phase N → Review checklist" in docs/PLAN.md. Review the diff for Phase N (`git log` / `git diff <phase-start>..HEAD`; read-only git only). Report only concrete problems, each with file:line. Don't rewrite the code.

| Phase | Goal | Est. |
|---|---|---|
| 0 | Setup and feasibility checks | 1.0 h |
| 1 | Ingestion with caching | 1.5 h |
| 2 | Districts, daily tables, Replay map | 1.5 h |
| 3 | Features, quantile model, backtest, ablation | 2.0 h |
| 4 | RAG knowledge base | 1.0 h |
| 5 | Agents: tools, Sentinel, Analyst, Verifier | 2.0 h |
| 6 | Forecast, Watch & Briefing, Evidence pages | 1.0 h |
| 7 | Demo cache, README, rehearsal | 0.5 h |
| S | Stretch: Live mode, Delhi, detection-gap panel | only if time remains |

---

## Phase 0: Setup and feasibility

**User must do (Sonnet: ask for these at the start of the phase):**
1. Get a FIRMS MAP_KEY at https://firms.modaps.eosdis.nasa.gov/api/map_key (it arrives by email).
2. Get a Gemini API key at https://aistudio.google.com/apikey. Alternatively, run Ollama on the P6000 machine.
3. Put both in `.env`.

**Tasks**
1. `uv init --package --name psw` (or write the pyproject by hand) with src layout, `uv python pin 3.12`, and the dependencies listed in AGENTS.md §3, plus `pytest` as a dev dependency.
2. `.gitignore`: `.env`, `data/raw/`, `kb/generated/`, `.venv/`, `__pycache__/`, `.pytest_cache/`, `*.egg-info`, `.DS_Store`.
3. `.env.example`:
   ```
   FIRMS_MAP_KEY=
   LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
   LLM_API_KEY=
   LLM_MODEL=gemini-2.5-flash   # use the current free-tier Flash model name
   ```
4. `src/psw/config.py`: paths (`ROOT`, `RAW`, `PROCESSED`, `GEO`, `KB`), `BBOX = (72.5, 28.0, 78.0, 32.8)`, `CITIES` (dict name → (lat, lon)), `SEASON_START = (9, 15)`, `SEASON_END = (11, 30)`, and `SEASONS` (filled in from the feasibility results). Load `.env` with python-dotenv.
5. A throwaway script `scripts/feasibility.py` (keep it, it documents the checks) that:
   - **FIRMS:** calls the area API once for `VIIRS_SNPP_NRT`, the bbox, `DAY_RANGE=1` and a recent date, and prints the row count and columns. Then checks the maximum `DAY_RANGE` (try 10; if it errors, try 5) and how far back `VIIRS_SNPP_SP` goes (try the 1 Nov of each year from 2022). Also checks whether `SP` covers the 2026 season or only `NRT` does.
   - **Open-Meteo AQ:** for Ludhiana, `hourly=pm2_5` with `start_date`/`end_date`, binary-searched for the earliest date that returns non-null values. (Expectation, unverified: CAMS global data from about Aug 2022.)
   - **Weather archive:** `https://archive-api.open-meteo.com/v1/archive` for the same dates, with `hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,boundary_layer_height`. Records which variables come back non-null and the archive lag (ERA5 runs about 5 days behind; recent days need the forecast API with `past_days`).
   - **District GeoJSON:** try candidate sources in order and pick the first that has Punjab with **23 districts including Malerkotla**:
     (a) the `india-maps-data` GitHub repo (udit-001), Punjab state file;
     (b) geoBoundaries IND ADM2, filtered to Punjab;
     (c) DataMeet `maps` repo, 2011 census districts (22 districts; Malerkotla is inside Sangrur).
     Save the result to `data/geo/punjab_districts.geojson` with one normalised property, `district`, in title case. Record the source URL and licence.
6. Write `docs/feasibility.md`: a table of source × usable date range × caveats, the chosen `SEASONS` list (e.g. `[2022, 2023, 2024, 2025, 2026]`, only seasons where both FIRMS and AQ cover Sep 15 to Nov 30, or most of it for 2026), and the district source and licence.
7. `docs/PROGRESS.md`: tick Phase 0.

**Training window rule:** with ≥4 seasons, test on the last 2 (expanding window, D5). With 3, test on the last 1. With <3, stop and tell the user.

**Acceptance:** `uv run pytest -q` runs (zero tests is fine), `uv run python -c "import psw.config, lightgbm, fastembed, geopandas"` works, feasibility.md exists, and the GeoJSON is committed.

**Review checklist:** `.env` is not tracked (`git ls-files | grep .env` should show only `.env.example`); no key appears in any committed file or printed log; feasibility.md states the real ranges, not assumptions.

---

## Phase 1: Ingestion with caching

All ingest functions follow one pattern: **build the cache path → return the cached file if it exists → otherwise fetch, validate, save and return it.** Use `requests` with a timeout of 60 s and 3 retries with backoff on 429/5xx. Add a 0.3 s sleep between FIRMS calls.

**`src/psw/ingest/firms.py`**
```python
def fetch_chunk(source: str, start: date, day_range: int, key: str) -> pd.DataFrame
    # cache: data/raw/firms/{source}/{start:%Y-%m-%d}_{day_range}.csv  (also cache empty results)
def fetch_season(year: int, key: str) -> pd.DataFrame
    # loop Sep 15..Nov 30 in DAY_RANGE chunks; try SP first, fall back to NRT per chunk;
    # add column `source`; dedupe on (latitude, longitude, acq_date, acq_time)
def load_all_fires(seasons: list[int]) -> pd.DataFrame
    # concat; parse acq_date → date; add `season`; keep columns:
    # latitude, longitude, acq_date, acq_time, frp, confidence, daynight, source, season
    # drop confidence == 'l' (low)  ← log how many rows were dropped
```
Note: the FIRMS date parameter is the **start** date and `DAY_RANGE` counts forward from it. Confirm this in Phase 0 and adjust if the API counts backward.

**`src/psw/ingest/airquality.py`**
```python
def fetch_city_pm25(city: str, lat: float, lon: float, start: date, end: date) -> pd.DataFrame
    # cache: data/raw/aq/{city}_{start}_{end}.json ; one request per city per season
    # returns hourly: time, pm2_5
def daily_pm25(seasons) -> pd.DataFrame   # columns: date, city, pm25_mean (needs ≥18 hourly values, else NaN)
```
Fetch from **Sep 1** (not Sep 15) so the lag features have history, through **Dec 1** so the target exists for Nov 30. Use `timezone=Asia/Kolkata`.

**`src/psw/ingest/weather.py`**
```python
def fetch_city_weather(city, lat, lon, start, end) -> pd.DataFrame   # archive API; forecast API past_days for the gap
def daily_weather(seasons) -> pd.DataFrame
    # columns: date, city, temp_mean, rh_mean, wind_speed_mean, wind_dir_mean, u_mean, v_mean, blh_mean
    # wind_dir_mean = vector mean (average u,v then atan2), NOT arithmetic mean of degrees
```
Put the wind maths in a small pure helper (`psw/processing/wind.py`): `uv_from_dir_speed(dir_deg, speed)` and `dir_from_uv(u, v)` in the meteorological "from" convention. **Test it**: a north wind (from 0°) has v < 0; the mean of 350° and 10° is 0°.

**Tests (`tests/test_ingest.py`, `tests/test_wind.py`)**
- Record small fixtures: one FIRMS CSV with about 20 rows, one AQ JSON for 3 days, one weather JSON for 3 days. Save them in `tests/fixtures/` (**strip the key from any URL**).
- Monkeypatch `requests.get` (or the fetch helper). Check that the parsers produce the right columns and dtypes, and that a second call reads from the cache (the mock is called once).
- Wind helper tests as above.

**Acceptance:** `uv run python -m psw.pipeline ingest` downloads every season. A second run finishes in seconds with no network calls. Tests pass.

**Review checklist:** cache hits skip the network; the vector-mean wind direction is right; the key is not written into the cache filenames or logs; empty FIRMS chunks are cached too, so they aren't re-fetched forever.

---

## Phase 2: Districts, daily tables, Replay map

**`src/psw/processing/districts.py`**
```python
def assign_districts(fires: pd.DataFrame, geo_path) -> pd.DataFrame
    # geopandas sjoin (predicate="within"); adds `district` (NaN outside Indian Punjab), `in_punjab` bool
def district_daily(fires) -> pd.DataFrame
    # full grid: every district × every season day (fill 0)
    # columns: date, season, district, fire_count, frp_sum, fire_count_7d_mean (trailing, incl. today)
```
Output: `data/processed/fires.parquet` (all fires in the bbox, with `district`) and `data/processed/district_daily.parquet`. If `fires.parquet` is over 25 MB, round lat/lon to 4 decimals and drop columns you don't need.

**`src/psw/pipeline.py`:** an argparse CLI with subcommands `ingest | process | train | kb | brief | demo-cache | all`. Each subcommand calls library functions and does the file I/O.

**App skeleton (`src/psw/app/main.py` + `pages/1_Replay_Map.py`)**
- `main.py`: title, a one-paragraph explanation, the limitations box (AGENTS/brief §11) and credits. Cache data loaders with `@st.cache_data`.
- Replay Map page:
  - A season selectbox and a date slider over Sep 15 to Nov 30.
  - A **Play/Pause** toggle: when playing, advance one day, `time.sleep(0.6)`, `st.rerun()`. Keep the state in `st.session_state`.
  - pydeck: a `GeoJsonLayer` of districts shaded by `fire_count` for that date (sequential colour scale) and a `ScatterplotLayer` of fires sized by `sqrt(frp)`, with tooltips (district, count; FRP, time).
  - Side metrics: fires in Punjab today, the 7-day mean and the top 5 districts.
  - A caption: "Satellite detections of heat anomalies (VIIRS 375 m). Fires outside overpass times or under cloud are missed."

**Tests (`tests/test_districts.py`):** two hand-made square polygons, with points inside, outside and on the edge. Check the assignment, the zero-filled grid and the 7-day mean.

**Acceptance:** the map plays through a season. Peak days (early-to-mid November) show Sangrur, Ferozepur, Bathinda, Moga and Tarn Taran style hotspots. Check that this matches public reporting, and note it in PROGRESS.md if it doesn't.

**Review checklist:** no Python loops over fire rows (vectorised or sjoin only); CRS is EPSG:4326 on both sides; the zero-fill covers every district-day; the app doesn't reload the parquet files on every rerun.

---

## Phase 3: Features, quantile model, backtest, ablation

**`src/psw/processing/features.py`**
```python
def upwind_pressure(fires_day: pd.DataFrame, city_latlon, wind_dir_from_deg: float,
                    max_km=300, decay_km=150) -> float
    # D4: w = max(0, cos(theta)) * exp(-d/decay_km), theta = angle between (wind_dir_from+180)
    # and bearing(fire → city); sum(frp * w) over fires with d <= max_km. Vectorised haversine/bearing.
def upwind_contributors(fires_day, city_latlon, wind_dir_from_deg, top=3) -> list[dict]
    # [{district, share_pct}] of the weighted FRP: used by the agent tool, NOT a model feature
def build_features(fires, pm25_daily, weather_daily) -> pd.DataFrame
```
Feature table `data/processed/features.parquet`, one row per (city, date in the season window):

| column | meaning |
|---|---|
| `date, city, season` | keys |
| `target_pm25_next` | pm25_mean at date+1 (the target) |
| `pm25_today`, `pm25_lag1`, `pm25_3d_mean` | persistence information |
| `temp_mean, rh_mean, wind_speed_mean, u_mean, v_mean, blh_mean` | weather (drop `blh_mean` if unavailable) |
| `upwind_frp_l0, upwind_frp_l1, upwind_frp_l2` | **fire feature**: upwind pressure for days t, t-1, t-2, using that day's wind |
| `frp_300km_l0` | **fire feature**: FRP within 300 km, unweighted |
| `punjab_fire_count_l0`, `punjab_fire_count_3d` | **fire feature** |
| `dow`, `day_of_season`, `city` (categorical) | calendar and city |

Keep `FIRE_FEATURES = [...]` as a constant in features.py. The ablation drops exactly these.

**`src/psw/models/train.py`**
- `fit_quantile_models(X, y, features) -> dict[alpha, LGBMRegressor]` for α ∈ {0.1, 0.5, 0.9}, with `objective="quantile"`, `n_estimators=300`, `learning_rate=0.05`, `num_leaves=15`, `min_child_samples=20`, `subsample=0.8`, `subsample_freq=1`, `colsample_bytree=0.8`. A pooled model across cities, with `city` as a categorical.
- Sort the quantiles after predicting (`p10 ≤ p50 ≤ p90`) to avoid crossing.
- **Conformal fix (small, optional):** if the 80% coverage on the *last training season* (an inner validation split) is outside 70–90%, widen or narrow the band by the empirical correction from that season (CQR). Log whether it was applied.

**`src/psw/models/evaluate.py`**
- An expanding-window backtest (D5): for each test season S, train on seasons < S and predict S.
- Variants: `full`, `no_fire` (drop FIRE_FEATURES), and `persistence` (p50 = pm25_today; the band is ± the empirical 10th/90th percentile of the day-to-day change in the training data).
- Metrics per variant, overall and per city: MAE(p50), mean pinball loss (averaged over the 3 quantiles, `sklearn.metrics.mean_pinball_loss`), 80% coverage and mean band width.
- Feature importance (gain) of the final `full` p50 model.
- Outputs:
  - `data/processed/forecasts.parquet`: `issue_date, target_date, city, season, variant, p10, p50, p90, actual`
  - `data/processed/metrics.json`: `{seasons_train, seasons_test, variants: {name: {overall: {...}, by_city: {...}}}, feature_importance: {...}, conformal_applied: bool, generated_at}`
- Also train a **final model on all seasons** and save it to `data/processed/model_full_{alpha}.txt` (`booster_.save_model`). It is used only for a "next day" forecast past the last date, i.e. Live/stretch.

**Tests (`tests/test_features.py`, `tests/test_model.py`)**
- upwind_pressure: a fire directly upwind gets weight ≈ exp(−d/150); directly downwind gets 0; crosswind gets ≈ 0; fires beyond 300 km are excluded.
- No leakage: on synthetic data, the features at date t don't change when the data after t is altered.
- Backtest on a tiny synthetic dataset runs end to end and produces all 3 variants and the metric keys.

**Acceptance:** `pipeline train` writes forecasts.parquet and metrics.json. Paste the summary table (variant × MAE × pinball × coverage) into PROGRESS.md. **Report the result honestly.** If `full` doesn't beat `no_fire`, say so. Don't tune on the test seasons to force a win. Possible honest explanation: CAMS already assimilates fire emissions (GFAS), so the target partly "knows" about the fires.

**Review checklist:** the target is t+1 and every feature is ≤ t; the test season is never in training; the persistence baseline uses the same rows as the model; coverage is computed on the test seasons only; the random seed is fixed.

---

## Phase 4: RAG knowledge base

**Corpus** (markdown with a YAML frontmatter: `id, title, doc_type, as_of, district?, city?, source_url?`)

*Curated* (`kb/curated/`, committed, about 8 short notes of 150–400 words each, written in our own words). **Only state facts you can back with the `source_url`. If unsure, leave it out.** The user will review these.
1. `firms_detection.md`: how VIIRS active fire detection works, the 375 m resolution, the S-NPP overpass times (around 01:30 and 13:30 local), and why fires are missed (overpass timing, cloud, small or short burns). Source: the FIRMS FAQ.
2. `cams_pm25.md`: what CAMS global forecasts are, their coarse resolution, the fact that they are modelled and not measured, and that they use GFAS fire emissions. Source: the CAMS/ECMWF documentation and Open-Meteo.
3. `standards_pm25.md`: India NAAQS PM2.5 (24 h: 60 µg/m³) and the WHO 2021 guideline (24 h: 15 µg/m³). Source: CPCB and WHO.
4. `grap.md`: CAQM's Graded Response Action Plan stages by AQI band (a summary only, not advice). Source: CAQM.
5. `crop_residue_context.md`: the paddy harvest and wheat sowing window, and in-situ/ex-situ management options. Qualitative. Source: PPCB or the Punjab agriculture department.
6. `method_upwind_pressure.md`: our D4 formula in plain words.
7. `limitations.md`: brief §11.
8. `how_to_read_forecast.md`: what p10/p50/p90 and an 80% interval mean.

*Generated* (`kb/generated/`, written by `rag/build_kb.py` from the processed data with deterministic templates):
- `season_{year}_{district}.md` for every district × season: total detections, total FRP, peak date and peak count, rank among districts, comparison with the previous season (%). `as_of` = Dec 1 of that year.
- `city_season_{year}_{city}.md`: season mean and max PM2.5, the number of days above 60 µg/m³, and the peak date. `as_of` = Dec 1 of that year.
- `model_card.md`: generated from metrics.json (variants, MAE, coverage, top features, training seasons, caveats). `as_of` = the date of the first test season's start, so the agent can cite it when writing about test seasons. (The model card describes the method, not future data, so this is acceptable. Note it in the doc.)

**`src/psw/rag/index.py`**
```python
def chunk(doc_text, max_chars=900, overlap=100) -> list[str]   # split on paragraphs/headings first
def build_index(docs: list[Doc]) -> None
    # embed with fastembed TextEmbedding("BAAI/bge-small-en-v1.5"); normalise
    # save data/processed/kb_vectors.npy + data/processed/kb_chunks.parquet
    #   (chunk_id like "K012", doc_id, title, doc_type, as_of, district, city, text)
class KnowledgeBase:
    def search(self, query: str, k=4, before: date | None = None,
               district: str | None = None) -> list[dict]
    # cosine top-k; filter as_of < before; if district is given, boost (+0.05) chunks from that district
    # (don't hard-filter, or curated docs would be excluded)
```
Load the embedding model lazily and once (module-level singleton). The first run downloads about 130 MB, so tell the user.

**Tests (`tests/test_rag.py`):** use a **fake embedder** (inject it, e.g. a bag-of-words hash vector) so the tests don't download anything. Check chunking limits, that the `as_of` filter excludes future docs, the district boost, and that chunk ids are stable across rebuilds.

**Acceptance:** `pipeline kb` builds the index. `KnowledgeBase().search("why do satellites miss fires", k=3)` returns firms_detection chunks first. `search("Sangrur last season", before=2025-11-05, district="Sangrur")` returns season_2024_Sangrur, never 2025.

**Review checklist:** no future docs leak past `before`; the generated docs contain only numbers computed from the data; curated docs have a source_url; the index files are small enough to commit (well under 10 MB).

---

## Phase 5: Agents

### 5.1 `src/psw/llm.py`
```python
class LLM:
    def __init__(self, base_url, api_key, model)             # from .env by default
    def chat(self, messages, tools=None, json_mode=False) -> ChatResult   # openai SDK
```
Retry on 429 with exponential backoff (free-tier limits are tight) and a minimum interval between calls (env `LLM_MIN_INTERVAL_S`, default 4). Log the token usage. Keep a `FakeLLM` in `tests/` that replays a scripted list of responses (tool calls, then a final message).

### 5.2 `src/psw/agents/tools.py`: data tools (pure Python, read the processed parquet)
Each tool returns a **small JSON-serialisable dict**. Round floats to 1 decimal (they must be quotable).
| tool | args | returns |
|---|---|---|
| `get_district_snapshot` | district, date | fire_count, frp_sum_mw, fire_count_7d_mean, change_vs_7d_pct, rank_in_punjab, punjab_total, last_7_days: [{date, count}], data_as_of |
| `get_city_forecast` | city, date | issue_date, target_date, p10, p50, p90, pm25_today, persistence_p50, band_width, exceeds_naaqs_60_prob_hint ("p50 above 60" / "p90 above 60" / "below") |
| `get_wind_and_upwind` | city, date | wind_dir_from_deg, wind_dir_compass (e.g. "NW"), wind_speed_kmh, upwind_frp_l0, top_upwind_districts: [{district, share_pct}] |
| `get_model_reliability` | city | test-season MAE, coverage_80_pct, and MAE vs persistence and no_fire (from metrics.json) |
| `search_knowledge` | query, k=4 | [{chunk_id, title, text}]. `before`=briefing date and district are injected by the orchestrator, not chosen by the LLM |

Plus `nearest_city(district) -> str` (a centroid distance helper, not an LLM tool). Expose OpenAI-format JSON schemas for the 5 tools in `TOOL_SCHEMAS`.

### 5.3 `src/psw/agents/sentinel.py` (deterministic)
```python
def scan(date) -> list[dict]   # every district: fire_count, 7d mean, ratio, nearest city p90
# flag reasons: "spike" if fire_count >= max(15, 2 * 7d_mean); "high_count" if top-3 in Punjab and >= 30;
#               "smog_risk" if nearest-city p90 > 120
# priority score = rank sum; return sorted, flagged first
```
Thresholds go in config.py, and the reasons are human-readable strings shown in the app.

### 5.4a `src/psw/agents/runtime.py`: a reusable agent loop (write this first)
Every LLM agent (the Analyst now, and the stretch Planner, Critic and Comparison agents later) is **a prompt plus a tool list run through this one function**. Don't write a second loop.
```python
@dataclass
class Tool:
    name: str
    description: str
    parameters: dict              # JSON schema (OpenAI function format)
    fn: Callable[..., dict]       # returns a small JSON-serialisable dict

@dataclass
class AgentSpec:
    name: str                     # "analyst", "critic", ...
    system_prompt: str
    tools: list[Tool]
    required_tools: list[str] = field(default_factory=list)   # nudged if missing
    max_turns: int = 8
    json_output: bool = True      # final answer must parse as JSON

@dataclass
class AgentResult:
    output: dict | str            # parsed final answer
    trace: list[dict]             # {step, agent, tool, args, result_summary, ms}
    tool_results: list[dict]      # full results → feeds Verifier evidence
    turns: int
    error: str | None

def run_agent(spec: AgentSpec, user_message: str, llm: LLM,
              injected_args: dict | None = None) -> AgentResult
```
- `injected_args` are merged into the tool call arguments and override any value the LLM sent (e.g. `{"search_knowledge": {"before": date, "district": d}}`). This way the LLM can never control the leakage filters.
- It handles: dispatching tool calls, catching tool exceptions (returned to the LLM as `{"error": ...}`), the required-tools nudge, tolerant JSON parsing with one "Return only the JSON" retry, the turn limit, and the trace.
- `run_agent` must not know about districts or PM2.5. Domain logic stays in the tools and prompts.
- Tests (`tests/test_runtime.py`, FakeLLM): tools are dispatched with the injected args overriding, the nudge fires for a missing required tool, the turn limit stops the loop, bad JSON gets one retry, and a tool exception doesn't crash the loop.

### 5.4b `src/psw/agents/analyst.py`: the Analyst, an `AgentSpec` run through `run_agent`
- `ANALYST = AgentSpec(name="analyst", system_prompt=..., tools=[the 5 tools], required_tools=[the 4 below])`, plus `def analyse(district, date, llm) -> AgentResult`, a thin wrapper that builds the user message and the injected args.
- The system prompt (store it as a constant and show it in the app's "How it works" expander) says:
  - "You are an environmental analyst writing a district briefing for officials and the public. Use the tools to gather facts. You MUST call get_district_snapshot, get_city_forecast, get_wind_and_upwind and search_knowledge at least once. Use ONLY numbers that appear in tool results. Never calculate new numbers. Cite knowledge chunks as [K###] after the sentence they support. No health or medical advice. Never name or identify individual farmers or fields. 120–160 words. Plain language. Mention the uncertainty band and one limitation."
  - Final output is JSON: `{"headline": str (≤12 words), "briefing": str, "citations": ["K012", ...]}`
- `run_agent` handles the loop, the nudge, JSON parsing and the trace.

### 5.5 `src/psw/agents/verifier.py` (deterministic)
```python
def verify(briefing: dict, evidence: Evidence) -> VerifyResult  # ok, errors: list[str], checked_numbers: list
```
- `evidence.allowed_numbers` = every numeric value in all tool results, plus numbers in retrieved chunk texts, plus date parts (day, year) of the dates involved, plus the window constants {7, 80, 10, 50, 90, 24} (from "7-day", "80% interval", p10/p50/p90, "24-hour").
- Extract numbers from the text with a regex that handles `1,234`, `12.5`, `40%` and `−3`, and ignores `PM2.5` (treat "PM2.5" as a token, not the number 2.5) and citation ids like `K012`.
- A number n matches if some allowed value a satisfies `round(a, decimals(n)) == n` or `|n − a| ≤ 0.5` for integers ≥ 10, or n == round(a*100) for ratios. Unmatched numbers are errors: `"number 87 not found in evidence"`.
- Citations: each must be in the retrieved chunk ids, and there must be at least 1 citation.
- Word count between 100 and 180. Banned patterns: health advice ("wear a mask", "should stay indoors", "N95"), and personal names after "farmer"/"Mr"/"Smt". Keep the list short and in config.
- Return all errors (don't stop at the first).

### 5.6 `src/psw/agents/template.py`: deterministic fallback
Fill a fixed paragraph from the same tool results, so it always passes the verifier. Label it "template (LLM unavailable or failed verification)".

### 5.7 `src/psw/agents/orchestrator.py`
```python
def run_briefing(district, date, llm=None) -> BriefingRecord
    # 1. evidence = run analyst (tool loop)        → draft
    # 2. verify(draft) → if fail: send errors back to the analyst once ("Fix these: ..."), re-verify
    # 3. still failing or LLM error → template.render(evidence), mode="template"
    # 4. save data/processed/briefings/{date}_{district}.json:
    #    {district, date, mode: "agent"|"agent_revised"|"template", headline, briefing, citations,
    #     sources: [{chunk_id, title, doc_type}], verification: {ok, errors, checked_numbers},
    #     trace, facts: {tool results}, model, generated_at}
def run_daily_watch(date, top_n=3, llm=None) -> dict   # sentinel.scan → run_briefing for top_n flagged
    # save data/processed/watch/{date}.json
```

**Tests (`tests/test_verifier.py`, `tests/test_agent.py`)**
- Verifier: passes clean text; catches an invented number; accepts a number that matches after rounding (87.4 → 87); doesn't flag "PM2.5" or "K012"; catches a citation that wasn't retrieved; catches banned phrases. **Write these first. The verifier is the heart of the "not a chatbot" claim.**
- Agent with FakeLLM: a scripted sequence of tool calls, then a final answer, produces a record with mode="agent" and the full trace. A script that skips a required tool gets the nudge. A final answer with a bad number goes through revision and then falls back to the template.
- Template output always passes `verify`.

**Acceptance:** `pipeline brief --date <a peak day in the latest test season>` produces a watch file plus 3 briefings using the real LLM. Run 12 sampled district-days (`pipeline brief --sample 12`) and report the pass rate and mode counts in PROGRESS.md. The definition of done needs ≥10 to pass, with agent or template mode counted honestly.

**Review checklist:** `run_agent` is the only LLM loop in the codebase and contains no domain logic; the LLM never receives the raw parquet, only tool outputs; `before` and `district` for RAG are injected and not LLM-controlled; the loop is bounded; API errors fall back cleanly; there are no `eval` or exec calls on model output; the verifier tolerance can't be gamed (e.g. it allows every integer below 100).

---

## Phase 6: Remaining app pages

**`pages/2_Forecast.py`**
- City and season selectors (test seasons only, with the label "out-of-sample backtest").
- A Plotly chart over the season: a shaded p10–p90 band, the p50 line, actual dots, a persistence dashed line (toggleable), and a horizontal line at 60 µg/m³ labelled "NAAQS 24 h".
- A selected-date card: "Forecast for {target_date}: {p50} µg/m³ (80% range {p10}–{p90})", plus the upwind contributors.
- The caption: "Target is CAMS modelled PM2.5 (via Open-Meteo), not a ground monitor."

**`pages/3_Watch_and_Briefing.py`**
- Pick a date (default: the demo peak day) → show the Sentinel watchlist table (district, count, 7-day mean, reasons) from `watch/{date}.json`, or compute it live (Sentinel is fast and offline).
- Pick a district → load the cached briefing if it exists, otherwise offer a **"Run agent"** button (needs the LLM key; show a friendly message if it's missing).
- Layout: on the left, the headline, the briefing text with [K###] citations, and a badge (**✅ Numbers verified** / **⚠️ Template fallback** / **❌ failed**, with the errors). On the right, tabs for **Agent trace** (each step: tool, args, result, ms), **Sources** (retrieved chunks with their titles), **Facts JSON** and **Verifier report** (each number → the evidence value it matched).
- An expander, "How the agent works": the system prompt, the tools list and a short diagram in text.

**`pages/4_Evidence.py`**
- A backtest table from metrics.json (variant × MAE, pinball, coverage, width), with the best value highlighted, plus a by-city breakdown.
- A bar chart of MAE for full vs no_fire vs persistence (the ablation), a feature importance bar, and a short honest interpretation written from the numbers.
- Agent reliability: the pass rate over the sampled briefings (from `briefings/`) and the mode counts.
- Data sources and credits: NASA FIRMS (VIIRS S-NPP), Copernicus Atmosphere Monitoring Service via Open-Meteo, Open-Meteo weather (ERA5), the district boundary source and licence, and the fastembed/BGE model. Also the limitations.

**Acceptance:** all 4 pages load from committed data with `.env` removed (test this: `mv .env .env.bak`).

**Review checklist:** no page crashes on a missing briefing or date; every page has its caveat caption; there are no hard-coded numbers in the narrative text, which must be computed from metrics.json.

---

## Phase 7: Demo cache, README, rehearsal

1. Pick **3 demo dates** in the latest test season (the peak fire day, a moderate day and an early-season day). Run `pipeline demo-cache`, which does `run_daily_watch` for each and also briefs Ludhiana's district and Sangrur on the peak day. Ask the user to commit `data/processed/briefings/` and `watch/`.
2. Set the app defaults to the peak demo date and Ludhiana.
3. README.md: the problem, the solution, an architecture diagram (ASCII is fine, include the agent loop), a quickstart (clean clone → `uv sync` → `streamlit run`), how to regenerate the data, the agent design (Sentinel → Analyst ↔ tools/RAG → Verifier → template fallback), the results table (copied from metrics.json by a script, or written by hand from it), data sources with attribution, limitations, and "what is ours".
4. Give the user these commands to check a clean clone (the user runs them): `git clone . /tmp/psw-check && cd /tmp/psw-check && uv sync && uv run streamlit run src/psw/app/main.py` with no `.env`.
5. Take screenshots of each page into `docs/screenshots/`.
6. Update the demo script in the brief (§12): step 5 now shows the Sentinel watchlist → open the briefing → walk through the agent trace → the sources → the verified badge.

**Definition of done (replaces brief §10)**
- [ ] The app runs from a clean clone with no keys and no network (Replay and cached briefings).
- [ ] The backtest table shows full vs persistence vs no_fire on ≥1 out-of-sample season.
- [ ] ≥10 sampled district-day briefings pass the verifier, with the agent/template split reported.
- [ ] RAG retrieval respects `as_of`, as shown by a test.
- [ ] The README credits every source and lists the limitations.
- [ ] No secrets in git (`git log -p | grep -i "key="` should print nothing).

---

## Stretch (in this order, only after Phase 7)
1. **Live mode:** FIRMS NRT for the last 7 days, the forecast API for weather and AQ, and the saved final model predicts tomorrow. A sidebar toggle. It only makes sense until Nov 30.
2. **Delhi** as a 5th city (it's a target only; keep the bbox as it is).
3. **Detection-gap panel:** hand-entered `data/official_counts.csv` (`date,district,count`) vs FIRMS counts, with an indicative ratio range and a strong caveat. Add `official_counts` as an extra fact for the agent when it's present.
4. CPCB ground sanity check via data.gov.in.

### Stretch: deeper RAG and agents (no framework, all built on `run_agent`)
Do these in order. Each must keep the demo working: cached briefings still render, and the Verifier still runs last on any text shown as verified.
5. **Hybrid retrieval:** add `rank_bm25` keyword search next to the embedding search and merge with reciprocal rank fusion (`score = Σ 1/(60 + rank)`). Add `tests/fixtures/retrieval_eval.json`, about 15 (query → expected doc_id) pairs. Report hit@3 for embeddings alone, BM25 alone and hybrid on the Evidence page.
6. **Planner step:** an `AgentSpec(name="planner", tools=[], json_output=True)` that returns `{"steps": [...], "questions_to_answer": [...]}` for the district and date. Its plan is passed into the Analyst's user message and shown at the top of the Agent trace tab.
7. **Critic agent:** an `AgentSpec(name="critic", tools=[])`. Its input is the verified briefing; its output is `{"ok": bool, "issues": [...], "suggested_briefing": str}`, covering clarity, tone and missing caveats. If it suggests a rewrite, **run the Verifier again on the rewrite**, and use it only if it passes. Otherwise keep the original. Show the critic's notes in the trace.
8. **Comparison agent:** `AgentSpec(name="comparer")` with tools `get_district_snapshot`, `search_knowledge` and a new `compare_districts(a, b, date)` / `same_week_last_season(district, date)` (pure Python, all the maths in the tool). It writes a short comparison card on the Briefing page. Same Verifier and same template fallback.

## Risks and fallbacks
| Risk | Fallback |
|---|---|
| Open-Meteo AQ history is shorter than 3 seasons | Use the training-window rule in Phase 0, and say so in the app. |
| Gemini free-tier rate limit or model name change | `LLM_MIN_INTERVAL_S`, backoff, the model name in .env, Ollama on the P6000, and the template fallback. |
| LLM tool calling is flaky | The required-tools nudge, one revision, then the template. Cached demo briefings mean the demo never depends on a live LLM. |
| Fire features don't beat no_fire | Report it honestly and explain (CAMS assimilates GFAS). The upwind contributors are still useful in briefings. |
| District GeoJSON has no Malerkotla | Use 22 districts and state the limitation in the app and README. |
| fastembed download blocked | The cached briefings still render. Rebuild the index on another network. |
