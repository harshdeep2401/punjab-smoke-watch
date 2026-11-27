# AGENTS.md: Punjab Smoke Watch

Shared context for every AI session working in this repo, whichever model or tool it is. **Read this file first, then `docs/PROGRESS.md` to see where work stopped.**

- Original brief: `Punjab_Smoke_Watch.md`. Where it conflicts with this file or `docs/PLAN.md`, **this file and PLAN.md win.** They add RAG and agents and cut the scope to one day. **Part 2 (§1b) redirects the project toward a real problem: audit plus response.**
- Build plan: `docs/PLAN.md`, with phases, interfaces and acceptance criteria.
- Progress log: `docs/PROGRESS.md`. Update it at the end of every phase.
- Data availability: `docs/feasibility.md`, written in Phase 0.

## 1. What we're building (30-second version)

An **agentic, RAG-grounded stubble-fire and air-quality intelligence pipeline** for Punjab. It is not a chatbot and has no free-text Q&A box.

1. **Pipeline:** NASA FIRMS fires, Open-Meteo weather and CAMS PM2.5 go into district aggregation and an upwind fire-pressure feature.
2. **Model:** LightGBM quantile regression (p10/p50/p90) for next-day PM2.5 in 4 cities, with an expanding-window backtest and an ablation against persistence and a no-fire-features model.
3. **RAG:** a local vector index over (a) curated method and policy notes and (b) **auto-generated documents from our own pipeline** (season summaries, the model card). Each chunk carries an `as_of` date, so retrieval never sees the future.
4. **Agents:**
   - **Sentinel:** deterministic and rule-based. Scans every district for a date and builds a ranked watchlist.
   - **Analyst agent:** an LLM tool-calling loop. It calls data tools and `search_knowledge` (RAG), then writes a cited district briefing as JSON.
   - **Verifier:** deterministic. Checks that every number in the text appears in the tool outputs or retrieved chunks, that every citation was actually retrieved, and the length and banned content. It sends feedback to the Analyst once, then falls back to a template.
5. **App:** Streamlit with 4 pages: Replay map, Forecast, Daily Watch & Briefing (shows the agent trace, sources and a "numbers verified" badge), and Evidence. Part 2 adds Audit and Response Planner pages.

**Core rule: the LLM never does the maths.** Every number comes from a Python tool, and the Verifier enforces this.

### 1b. Part 2 direction (from 2026-11-26): from a demo to a real problem

Phases 0–7 built the platform above. Part 2 (Phases 8–13 in PLAN.md) points it at two connected problems:

**Why:** our own data shows a gap. Punjab's **daytime** detections (the ≈13:30 S-NPP pass) fell 91% from 2022 to 2025 (45,305 → 3,942). **Night** detections didn't fall (82 → 96), so the night share rose from 0.2% to 2.4%. City PM2.5 didn't fall, and fire features didn't help the forecast. Public reporting since 2024 says burning may have moved outside satellite overpass times. So official "fires down 90%" claims may partly be a detection artefact, and response planning built on those counts may miss the real burning.

1. **Module A, Audit ("what is really happening?"): monitoring and audits.** We **triangulate** the true scale of burning from independent signals instead of trusting one satellite pass:
   - several sensors at different overpass times (VIIRS S-NPP, NOAA-20, NOAA-21; MODIS Terra ≈10:30 and Aqua ≈13:30);
   - the time-of-day profile of detections;
   - **burned area** (MODIS MCD64A1, plus Sentinel-2 dNBR if feasible);
   - official counts (hand-entered, with sources).

   The output is a **detection-gap range** per district and season, and a verified audit brief written by an **Auditor agent**.
   - **Users:** CAQM/PPCB monitoring cells, researchers, journalists and citizens checking official claims.
   - "Ground truth" here means the **best available triangulation, with uncertainty**. We have no field verification, and we say so.
2. **Module B, Response ("where should help go in the next 3 days?"): decision support for authorities.** It has three parts:
   - A **3-day district fire-risk forecast**, using recent detections, the harvest-progress signal (mandi paddy arrivals if available, otherwise proxies), rain and past seasons. It is adjusted by Module A's gap range, so it plans for *likely* burning, not just *detected* burning.
   - A **deterministic allocator** that distributes a limited fleet of residue-management machines and awareness teams across districts and 5 km grid cells.
   - A **Dispatcher agent** that writes a verified 3-day action brief for the district agriculture officer or state control room.
   - **Users:** the Punjab agriculture department, district officers and cooperative machinery banks.
   - This is **support, not enforcement:** it never identifies farmers or fields. The finest unit is a 5 km grid cell.

**How they connect:** Audit → gap range → Response plans for likely burning → after the season, Audit checks whether burning fell where help went (future work).

The Phase 0–7 platform stays as it is: pipeline, PM2.5 forecast, RAG, `run_agent`, Sentinel/Analyst/Verifier and the app. The new agents are new `AgentSpec`s run through `run_agent`, they use the same Verifier, and they have template fallbacks.

## 2. Model roles (who does what)

| Role | Model | Responsibilities |
|---|---|---|
| Architect | Claude Opus | Architecture, PLAN.md, hard technical decisions, debugging dead ends. Changes to interfaces go through Opus. |
| Implementer | Claude Sonnet | Executes PLAN.md one phase at a time: code, tests, suggests commit messages (the user commits), updates PROGRESS.md. |
| Reviewer | GPT 6 Luna | Reviews code at the end of each phase against the phase's "Review checklist" in PLAN.md, and handles trivial chores. |

Implementer rules:
- Work **one phase at a time**, in order. Don't start phase N+1 until phase N's acceptance criteria pass.
- If you have to deviate from an interface in PLAN.md, write the deviation and the reason in `docs/PROGRESS.md` under "Deviations". Don't silently redesign.
- If a decision is genuinely ambiguous and blocks progress, stop and ask the user. Otherwise pick the simplest option and log it.

## 3. Tech stack (fixed, don't add to it without a reason)

- Python **3.12** managed by **uv** (`uv sync`, `uv run ...`). Package `psw` lives in `src/psw/`.
- Data: `pandas`, `pyarrow`, `geopandas`/`shapely`, `requests`, `python-dotenv`.
- Model: `lightgbm`, `scikit-learn` (metrics only). On macOS run `brew install libomp` if LightGBM fails to import.
- RAG: `fastembed` (`BAAI/bge-small-en-v1.5`, runs locally on CPU) plus numpy cosine search, and `rank-bm25` keyword search merged by reciprocal rank fusion (D13). **No vector DB server.**
- LLM: the `openai` SDK pointed at any OpenAI-compatible endpoint, configured in `.env`:
  - Default: **Gemini free tier** via `https://generativelanguage.googleapis.com/v1beta/openai/`.
  - Alternative: **Ollama** at `http://<host>:11434/v1`. The user has a 22 GB P6000 and can serve e.g. a Qwen 14B-class model with tool support.
  - No agent framework (LangChain, CrewAI and the like). The agent loop is about 100 lines of plain Python.
- App: `streamlit`, `pydeck` (maps), `plotly` (charts).
- Part 2, **only if Phase 8 feasibility confirms they're needed:** `pystac-client` and `planetary-computer` (Microsoft Planetary Computer STAC, no key) and `rasterio` for MODIS MCD64A1 burned area, plus Sentinel-2 dNBR if it's feasible. Log the addition as a decision.
- Tests: `pytest`. **No network in tests.** Use recorded fixtures in `tests/fixtures/` and a scripted fake LLM.

## 4. Repo layout

```
AGENTS.md  CLAUDE.md  Punjab_Smoke_Watch.md  README.md  pyproject.toml  .env.example
data/
  raw/            # API caches, gitignored
  geo/            # punjab_districts.geojson, committed (record its licence)
  processed/      # parquet, kb index, metrics, demo briefings, committed so a clean clone runs
  manual/         # Part 2 hand-entered CSVs, each row with a source_url: official_counts.csv, crm_machines.csv, mandi_arrivals.csv (if no API)
kb/
  curated/        # hand-written method/policy notes (markdown + frontmatter)
  generated/      # written by the pipeline, gitignored and rebuilt by `psw.pipeline kb`
src/psw/
  config.py  pipeline.py  llm.py
  ingest/      firms.py  weather.py  airquality.py
  processing/  districts.py  features.py
  models/      train.py  evaluate.py
  rag/         build_kb.py  index.py
  agents/      runtime.py  tools.py  sentinel.py  analyst.py  planner.py  verifier.py  orchestrator.py  template.py
               auditor.py  critic.py  dispatcher.py    # Part 2 agents (AgentSpecs)
  audit/       sensors.py  burned_area.py  official.py  gap.py      # Part 2, Module A
  response/    harvest.py  risk.py  allocate.py  grid.py          # Part 2, Module B
  app/         main.py  pages/   # + 5_Audit.py, 6_Response_Planner.py
tests/  fixtures/
docs/   PLAN.md  PROGRESS.md  feasibility.md
```

## 5. Key commands

```bash
uv sync                                        # install
uv run pytest -q                               # tests (offline)
uv run python -m psw.pipeline ingest           # fetch + cache raw data
uv run python -m psw.pipeline process          # fires->districts, daily tables, features
uv run python -m psw.pipeline train            # backtest, ablation, forecasts.parquet, metrics.json
uv run python -m psw.pipeline kb               # generate docs + build vector index
uv run python -m psw.pipeline brief --date 2025-11-05 [--district Sangrur]   # sentinel + agent
uv run python -m psw.pipeline demo-cache       # precompute briefings for the demo
# Part 2
uv run python -m psw.pipeline audit-ingest     # extra FIRMS sensors, burned area, manual CSVs
uv run python -m psw.pipeline audit            # diurnal profiles, sensor/burned-area comparison, gap ranges, audit briefs
uv run python -m psw.pipeline risk-train       # 3-day district fire-risk backtest + forecasts
uv run python -m psw.pipeline dispatch --date 2025-10-28 --machines 500 [--district Sangrur]   # allocation + dispatch brief
uv run streamlit run src/psw/app/main.py       # app
```

## 6. Conventions

- **Secrets:** only in `.env` (gitignored). Keys: `FIRMS_MAP_KEY`, `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`. Never print keys or commit them.
- **Caching:** every ingest function checks for its cache file on disk first. Re-running the pipeline makes no repeat API calls.
- **Dates:** a `datetime.date` or an ISO string `YYYY-MM-DD`. FIRMS dates are UTC, so we treat `acq_date` as the local day (document this as a limitation). Season year = calendar year; season window = Sep 15 to Nov 30.
- **Units:** PM2.5 in µg/m³, FRP in MW, distances in km, wind direction in degrees, meteorological convention (the direction the wind blows *from*).
- **No leakage:** features for day *t* use data up to *t* only. The target is the mean PM2.5 on *t+1*. RAG retrieval filters `as_of < briefing_date`.
- **Git: AI tools may run read-only git commands only** (`git status`, `git diff`, `git log`, `git show`, `git ls-files`, `git blame`). **The user runs every command that changes anything**: add, commit, push, pull, checkout, branch, merge, rebase, reset, stash, tag, clone and the like. When a step is green, stop and give the user a suggested commit message in conventional style (`feat(ingest): ...`, `test(...)`, `docs: ...`) and the list of files to stage. Then wait for them to commit before starting the next step.
- Keep functions small and pure where you can, and keep I/O at the edges (the `pipeline.py` CLI).
- The app must run from committed `data/processed/` with **no API keys and no network** (Replay mode, cached briefings). LLM regeneration is an optional button.

## 7. Decisions log (append; don't rewrite)

| # | Decision | Why |
|---|---|---|
| D1 | One day of scope. Delhi, Live mode, the detection-gap panel and CPCB data are **stretch**. | Time |
| D2 | 4 cities: Ludhiana (30.90, 75.85), Amritsar (31.63, 74.87), Patiala (30.34, 76.39), Bathinda (30.21, 74.95). | Spread across Punjab |
| D3 | FIRMS: VIIRS S-NPP only. Use `VIIRS_SNPP_SP` (archive) when it returns data, otherwise `VIIRS_SNPP_NRT`. | One consistent sensor across years |
| D4 | Upwind pressure = Σ FRP · max(0, cos θ) · exp(−d/150 km) over fires within 300 km, where θ is the angle between the wind's *to* direction and the bearing from fire to city. Lags 0, 1, 2. | Explainable, a single formula |
| D5 | Expanding-window backtest: test on each of the last 2 seasons, training on all earlier seasons. | Time-based, no shuffle, gives 2 out-of-sample seasons for the app |
| D6 | LLM through an OpenAI-compatible client. Default Gemini free tier, Ollama as an alternative. | Free, swappable, no framework |
| D7 | RAG = fastembed + numpy and an `as_of` filter. Corpus = curated notes plus pipeline-generated docs. | No server; RAG grounded in our own data |
| D8 | Sentinel and Verifier are deterministic. Only the Analyst uses the LLM. | Reliability on the free tier, and verifiable |
| D9 | Commit `data/processed/` (including demo briefings) and gitignore `data/raw/`. | The demo runs from a clean clone, offline |
| D11 | One reusable `run_agent(AgentSpec, ...)` loop in `agents/runtime.py`. Every LLM agent is a prompt plus tools run through it. No agent framework. | New agents (Planner, Critic, Comparison) cost a prompt, not a new loop |
| D12 | AI tools run read-only git commands only. The user runs every git command that changes anything (add, commit, push, etc.). AI tools suggest commit messages and the files to stage. | User preference |
| D13 | Retrieval is hybrid by default: BM25 (`rank-bm25`) plus embeddings, merged by reciprocal rank fusion (k = 60). The `as_of` filter applies in every mode before ranking. `search(..., mode="embedding"\|"bm25"\|"hybrid")`. | Exact names, years and terms are weak spots for embeddings; measured on a 17-query eval set (`pipeline kb-eval`), hybrid was never worse than rank 3 and best on MRR. |
| D14 | The Planner (`agents/planner.py`) is an optional `AgentSpec` with no tools, off by default (`use_planner` / `--plan`). It sees flag kinds, never numbers; its plan goes into the Analyst's first message and is stored in the briefing record. A failed or unusable plan never blocks a briefing. The Verifier still runs last on the Analyst's text. | One more LLM call per briefing is costly on a free tier, and a plan can steer the Analyst but cannot supply evidence. |
| D21 | The Critic (`agents/critic.py`) is built in Phase 10 and is on by default for audit briefs. It gets the cited chunk texts, flags unsupported citations, overclaims and missing caveats, and may only delete, reorder or reword: a rewrite with any new number is rejected, and every rewrite is re-verified. If the Critic fails, the verified original stands. The Planner (D14) is also available to the Auditor; it sees only the names of the available signals. | Audit briefs make sensitive claims about official numbers, so citations that don't support their sentence (seen in Part 1) matter most there |
| D15 | Part 2 direction: two modules. **A: Audit** (triangulate the true scale of burning, with a detection-gap range) and **B: Response** (3-day fire-risk forecast, then allocation of machines and teams, then a dispatch brief). B uses A's gap range. | Turns the platform toward a real problem: whether counts are falling because burning fell or because detection fell, and where to send help |
| D16 | "Ground truth" = triangulation of independent signals (multi-sensor, time of day, burned area, official counts), always reported as a **range** with the method for each bound. Never a single "true" number. No field verification exists, and we say so. | Honesty; each signal has its own bias |
| D17 | Burned area: MODIS MCD64A1 (500 m, monthly) via Planetary Computer STAC is the default. Sentinel-2 dNBR only for a few hotspot districts, if Phase 8 shows it's practical. If neither works, run the audit on sensors, time of day and official counts only, and say so. | MCD64A1 is easy to get but under-detects small fields; S2 is sharper but heavy |
| D18 | Response is **support, not enforcement**. The smallest spatial unit is a 5 km grid cell. No field, farmer or village names. Briefs talk about machines, awareness teams and timing, never penalties. | Ethics and the brief's out-of-scope list |
| D19 | The allocator is deterministic: greedy marginal gain with diminishing returns, cover(u) = 1 − exp(−u / k). Fleet size and per-district limits are user inputs. The Dispatcher agent explains the allocation; it never sets it. | Same rule as before: the LLM never does the maths |
| D20 | Fire-risk target = detections in a district over the next 3 days (log1p), LightGBM quantile, the same expanding-window backtest. It is judged mainly on **ranking** (top-5 hit rate, Spearman across districts per day), because allocation needs the right order more than exact counts. | Matches the decision the user makes |
| D10 | District boundaries: chosen in Phase 0 (see feasibility.md). Prefer 23 districts including Malerkotla; the fallback is the 2011-era 22 districts with a stated limitation. | Must verify the data |
| D22 | Replaces D17 for Part 2. Burned area = MCD64A1 monthly `burndate` GeoTIFFs from the University of Maryland SFTP server (window 18), with the shared login published in the user guide. No Sentinel-2 dNBR. Adds `rasterio` and `paramiko`; `pystac-client` and `planetary-computer` are not added. Approved by the user on 2026-11-26. | Planetary Computer is missing months we need, and Sentinel-2 dNBR measured the harvest, not burning (Phase 8) |
| D23 | Gap method revised (Opus, 2026-11-27), D16's principle kept. Signals get roles: counts (S-NPP, official) are the numerator; MCD64A1 burned area is the only reference; night share and MODIS-vs-S-NPP are Punjab-level timing indicators that set confidence only; NOAA-20/21 are a robustness check. Gap = relative fall in detections per burned km² since a base season; range over {S-NPP, official} × {base 2022, 2023}; no clipping; small-number rule (≥ 50 km², ≥ 100 base detections); confidence never "high" (Punjab "medium" at best, districts "low" at best). Phase 11 clips the gap to [0, 0.9] for the likely-burning adjustment only. | The original formula treated dependent signals as independent: VIIRS satellites share a pass time, official counts are satellite-based, and night counts are tiny, so the min-max range ran 0.13 to 0.92 and "signals agreeing in sign" overstated confidence |
| D24 | District CRM machine counts go in `data/manual/crm_machines.csv` as context only (Dispatcher tool and page), never in the allocation maths. | They count machines supplied on subsidy since 2018-19, not machines that still work or can move |
