# AGENTS.md: Punjab Smoke Watch

Shared context for every AI session working in this repo, whichever model or tool it is. **Read this file first, then `docs/PROGRESS.md` to see where work stopped.**

- Original brief: `Punjab_Smoke_Watch.md`. Where it conflicts with this file or `docs/PLAN.md`, **this file and PLAN.md win.** They add RAG and agents and cut the scope to one day.
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
5. **App:** Streamlit with 4 pages: Replay map, Forecast, Daily Watch & Briefing (shows the agent trace, sources and a "numbers verified" badge), and Evidence.

**Core rule: the LLM never does the maths.** Every number comes from a Python tool, and the Verifier enforces this.

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
- RAG: `fastembed` (`BAAI/bge-small-en-v1.5`, runs locally on CPU) plus numpy cosine search. **No vector DB server.**
- LLM: the `openai` SDK pointed at any OpenAI-compatible endpoint, configured in `.env`:
  - Default: **Gemini free tier** via `https://generativelanguage.googleapis.com/v1beta/openai/`.
  - Alternative: **Ollama** at `http://<host>:11434/v1`. The user has a 22 GB P6000 and can serve e.g. a Qwen 14B-class model with tool support.
  - No agent framework (LangChain, CrewAI and the like). The agent loop is about 100 lines of plain Python.
- App: `streamlit`, `pydeck` (maps), `plotly` (charts).
- Tests: `pytest`. **No network in tests.** Use recorded fixtures in `tests/fixtures/` and a scripted fake LLM.

## 4. Repo layout

```
AGENTS.md  CLAUDE.md  Punjab_Smoke_Watch.md  README.md  pyproject.toml  .env.example
data/
  raw/            # API caches, gitignored
  geo/            # punjab_districts.geojson, committed (record its licence)
  processed/      # parquet, kb index, metrics, demo briefings, committed so a clean clone runs
  official_counts.csv   # stretch only
kb/
  curated/        # hand-written method/policy notes (markdown + frontmatter)
  generated/      # written by the pipeline, gitignored and rebuilt by `psw.pipeline kb`
src/psw/
  config.py  pipeline.py  llm.py
  ingest/      firms.py  weather.py  airquality.py
  processing/  districts.py  features.py
  models/      train.py  evaluate.py
  rag/         build_kb.py  index.py
  agents/      runtime.py  tools.py  sentinel.py  analyst.py  verifier.py  orchestrator.py  template.py
  app/         main.py  pages/
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
| D10 | District boundaries: chosen in Phase 0 (see feasibility.md). Prefer 23 districts including Malerkotla; the fallback is the 2011-era 22 districts with a stated limitation. | Must verify the data |
