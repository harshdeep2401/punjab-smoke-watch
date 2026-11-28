# Frontend polish plan (2026-11-28)

Goal: the app tells one story, problem → response → trust, and every element on a page makes a point. Nothing in the
pipeline, agents or saved data changes; only `src/psw/app/` (plus a theme file and tests).

## Rule for every element
Keep it on the page only if it answers that page's question. Figures that support a point but are not the point go in
an expander or tab. Method notes, tool lists and system prompts go to the Evidence page.

## Navigation (`st.navigation`, grouped)
- **The story**
  1. Overview: the problem in three numbers and one chart, then links to the next steps.
  2. Is burning really down? (Audit)
  3. Where should help go? (Response Planner + Dispatcher)
  4. Can you trust it? (Evidence)
- **Explore the data** (Part 1 tools, kept but secondary): Fire replay, PM2.5 forecast, Daily district briefing.

## Pages
- **Overview** (`home.py`, new): "Satellite counts fell 91%. Burned area fell 45%. City PM2.5 did not fall." Three
  metrics computed from the audit tables and `pm25_daily.parquet`, the detections-vs-burned-area chart, three step cards
  with links, a collapsed limitations expander. `main.py` becomes the router plus theme.
- **Audit**: lead sentence → key chart (detections vs burned area) beside one gap-range card per season (replaces the
  15-metric wall) → "Why we suspect detection fell, not just burning" as tabs (pass time, night share, official counts,
  city air) → districts in an expander → the audit brief.
- **Response**: plan in words + 3 metrics → map of expected burning beside the allocation chart → the dispatch brief
  (the deliverable) → district drill-down (cells, rain, machines). Big tables and method captions go into expanders.
- **Evidence**: tabs in story order: Audit method, Fire risk and allocation, Agents (reliability, Verifier limits,
  how each agent works, tools, prompt), PM2.5 forecast, Retrieval, Sources and limits.
- **Agent records** (shared by three pages): text full width in a bordered box with the badge; trace, sources, facts
  and Verifier report move into one collapsed "How this brief was checked" expander.
- **Part 1 pages**: drop per-page "how it works" blocks (moved to Evidence), trim duplicated captions.
- Theme: `.streamlit/config.toml`, one accent colour, minimal toolbar.

## Done when
- All pages load with no `.env`; `uv run pytest -q` passes (tests updated only where a label moved, never a check
  removed).
- Screenshots of each page reviewed in light theme.
- PROGRESS.md notes the change.
