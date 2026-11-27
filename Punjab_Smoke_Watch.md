# Punjab Smoke Watch

Uncertainty-aware stubble-fire and air-quality intelligence for Punjab. Software only, no hardware, not a chatbot.

> **For Claude Code:** read this whole file first. Start with "Step 0: feasibility checks", then follow the build plan in order. Keep each step small, test it, and commit. Ask me only when a choice below is marked **DECISION**. Never commit API keys.

---

## 1. Problem statement

Every October and November, crop-residue burning in Punjab coincides with smog across north India. Three things make this hard to reason about:

1. People cannot easily see the chain *fires -> wind -> PM2.5*. Fire maps, weather and air-quality dashboards live in separate places.
2. Satellites only see a fire if it is burning during an overpass, so satellite counts and official counts can disagree. A tool that reports one confident number is misleading.
3. Officials and the public need a short, district-level read of "what is happening and what is likely tomorrow", not raw CSVs.

## 2. Solution (one paragraph, for the teacher)

A pipeline that pulls open satellite fire detections (NASA FIRMS), weather and modelled PM2.5 (Open-Meteo), assigns fires to districts, and computes upwind fire-pressure features for a few Punjab cities. A LightGBM quantile model forecasts next-day PM2.5 **with a prediction interval**, and an ablation proves the fire features help. A Streamlit app shows a replayable hotspot map, the forecast bands, and an auto-written district briefing. The LLM only writes prose from a structured JSON of computed numbers, and a checker rejects any number in the text that is not in the JSON. The LLM never does the maths.

**Why it is not a chatbot:** the core is a data pipeline, a forecasting model with a backtest, and a verified report generator. There is no Q&A interface.

## 3. Scope

**Core (must ship):**
- Data ingestion with local caching (fires, weather, PM2.5).
- Fire-to-district aggregation and upwind fire-pressure features.
- Quantile forecast for 4-5 cities, with backtest and ablation.
- Streamlit app: replay map, forecast bands, district briefing.
- Number-checked LLM briefing.

**Stretch (only if core is done):**
- Detection-gap panel: compare FIRMS counts with hand-entered official daily counts and show an indicative under-detection range.
- Optional CPCB ground-truth check via data.gov.in.

**Out of scope:** identifying individual farmers or fields, enforcement use, health advice, hardware, user accounts.

## 4. Data sources

| Need | Source | Access |
|---|---|---|
| Active fires | NASA FIRMS area API (VIIRS preferred, 375 m) | Free `MAP_KEY` from firms.modaps.eosdis.nasa.gov/api/map_key. Endpoint: `https://firms.modaps.eosdis.nasa.gov/api/area/csv/{MAP_KEY}/{SOURCE}/{west,south,east,north}/{DAY_RANGE}/{YYYY-MM-DD}`. Sources include `VIIRS_SNPP_NRT`, `VIIRS_SNPP_SP` (archive), `VIIRS_NOAA20_NRT`. Rate limit about 5,000 requests per 10 min per key. |
| Weather | Open-Meteo forecast and archive APIs | No key, free for non-commercial use. Wind speed/direction, temperature, humidity, boundary layer height if offered. |
| PM2.5 (modelled) | Open-Meteo Air Quality API (CAMS) | No key. `https://air-quality-api.open-meteo.com/v1/air-quality`. Model data, not a ground monitor. Must credit CAMS and Open-Meteo. |
| District boundaries | A Punjab district GeoJSON (e.g. from the DataMeet maps repo) | **DECISION:** pick one and confirm it includes recent districts such as Malerkotla. |
| Official fire counts (stretch) | PPCB / CAQM daily reports | No API. Hand-copy a few days into `data/official_counts.csv` with columns `date,district,count`. |
| Ground PM2.5 (optional) | CPCB via data.gov.in "Real time Air Quality Index" | Free data.gov.in API key, real-time only. Use only as a sanity check. |

Study area bounding box (fires): roughly `72.5,28.0,78.0,32.8` (Punjab, Haryana, Delhi and the border belt). Then assign Indian-Punjab fires to districts by point-in-polygon. Keep the other fires for upwind features, since smoke does not respect borders.

Fire columns to expect: `latitude, longitude, bright_ti4, scan, track, acq_date, acq_time, satellite, instrument, confidence, version, bright_ti5, frp, daynight`.

## 5. Step 0: feasibility checks (do these before building)

1. Get a FIRMS `MAP_KEY`, put it in `.env` as `FIRMS_MAP_KEY`. Confirm one area call returns Punjab data. Check the max `DAY_RANGE` per call in the API docs and loop in chunks.
2. **Check how far back Open-Meteo's Air Quality API returns data for Ludhiana (30.9, 75.85).** My assumption was 2023-2025, but I have not verified it. The answer decides the training window. If history is short, train on the seasons available and say so in the app.
3. Confirm the weather archive API returns hourly wind direction and speed for the same period.
4. Report back the usable date range for each source in `docs/feasibility.md` before moving on.

## 6. Architecture

```
FIRMS API ----\
Open-Meteo wx --+--> ingest/ (cached to data/raw/) --> processing/ --> features.parquet
Open-Meteo AQ -/                                          |
District GeoJSON ----> fire -> district join --------------+
                                                          v
                                   models/ LightGBM quantile (p10, p50, p90)
                                                          |
                         district_daily.parquet + forecasts.parquet
                                                          v
                      briefing/ JSON facts -> LLM prose -> number checker
                                                          v
                                  app/ Streamlit (map, forecast, briefing, evidence)
```

### Repo layout

```
punjab-smoke-watch/
  PROJECT.md
  .env.example            # FIRMS_MAP_KEY, LLM keys (never commit .env)
  data/raw/  data/processed/  data/official_counts.csv
  src/
    ingest/firms.py  weather.py  airquality.py
    processing/districts.py  features.py
    models/train.py  evaluate.py
    briefing/facts.py  llm.py  check.py
    app/main.py
  tests/
  docs/feasibility.md
```

## 7. Key technical design

**Upwind fire pressure (the main feature).** For each city and day, sum fire radiative power (FRP) of detections within ~300 km, weighting each fire by how closely the wind direction points from the fire toward the city (e.g. `max(0, cos(angle))`), with 0, 1 and 2 day lags. Also include raw district counts, wind speed, temperature, humidity, boundary layer height (if available), day-of-week and day-of-season. Keep features explainable.

**Target and model.** Next-day mean PM2.5 per city (Ludhiana, Amritsar, Patiala, Bathinda, plus Delhi if it fits). LightGBM with quantile objective at 0.1, 0.5 and 0.9.

**Evaluation (this is what makes it credible).**
- Time-based split: train on earlier seasons, test on the latest. No random shuffling.
- Metrics: MAE on the median, pinball loss, and coverage of the 80% interval.
- Baselines: persistence (tomorrow = today) and the same model **without** fire features. Show the ablation in the app.
- Honest caveat to display: the target is CAMS modelled PM2.5, not a ground monitor.

**Briefing generator.**
1. `facts.py` builds a JSON for one district and date: fire counts, 7-day trend, top villages/areas by detection density if available, forecast p10/p50/p90, dominant wind direction, data-freshness date.
2. `llm.py` sends only that JSON with a strict prompt: write 120-150 words, plain language, use only the numbers given. Make the provider swappable (Gemini free tier or local Ollama).
3. `check.py` extracts every number from the text and asserts it exists in the JSON (allow rounding and unit changes). If it fails, retry once, then fall back to a plain template. Log failures.

**Detection-gap panel (stretch).** Where official counts exist, show FIRMS count vs official count per day and an indicative ratio range. State clearly that the two systems use different methods and detection windows, so the ratio is indicative only.

## 8. App pages (Streamlit)

1. **Replay map:** date slider across a past season, fire points sized by FRP, district shading by count, play button.
2. **Forecast:** per-city next-day PM2.5 with p10-p90 band over recent days.
3. **Briefing:** choose district and date, see the generated briefing next to the JSON facts it used, with a "numbers verified" badge.
4. **Evidence:** backtest metrics, ablation (with vs without fire features), data sources and credits.

Support two modes: **Replay** (a stored past season, works offline from cache) and **Live** (last 7 days from FIRMS). Monitoring season is Sep 15 to Nov 30, so Live will be thin after this week; Replay must be the default for demos.

## 9. Build plan

**Day 1**
1. Step 0 feasibility checks and `docs/feasibility.md`.
2. Ingestion modules with caching and tests (recorded fixtures, no live calls in tests).
3. District join and `district_daily.parquet`.
4. Replay map page working end to end.

**Day 2**
5. Features and quantile model, backtest, ablation.
6. Forecast page and Evidence page.
7. Facts JSON, LLM briefing, number checker, Briefing page.
8. README, screenshots, demo rehearsal. Stretch items only if time remains.

## 10. Definition of done

- [ ] `streamlit run src/app/main.py` works from a clean clone using cached data.
- [ ] Backtest table shows the model against persistence and the no-fire ablation.
- [ ] Briefing text passes the number checker on at least 10 sampled district-days.
- [ ] README lists every data source with attribution (NASA FIRMS, Copernicus CAMS via Open-Meteo, any boundary data) and the limitations below.
- [ ] No secrets in git.

## 11. Limitations to state openly

- Satellites detect heat anomalies, not "stubble burning" specifically, and miss fires outside overpass windows or under cloud.
- PM2.5 is modelled (CAMS), not measured at street level.
- This is a student project, not a regulatory or health tool. No individual farms or farmers are identified.
- Forecast accuracy is limited by the training window we can actually obtain.

## 12. Demo script (5 minutes)

Updated in Phase 13 for the Audit and Response modules (this replaces the earlier script).

**Before you start:** `uv run streamlit run src/psw/app/main.py --server.address localhost`. No key is needed: every brief shown is saved. **Do not press a "Run" or "Regenerate" button during the demo.** It takes about a minute or more, uses your LLM key and, on the Watch page, can replace a saved briefing with the template. The Replay map's background needs a network.

| Step | What to do | What to say | Time |
|---|---|---|---|
| 1. Hook | Home page | Daytime fire detections in Punjab fell about 91% from 2022 to 2025 (45,305 to 3,942). Night detections did not fall (82 to 96). "That is a lead, not proof. So how much has burning really fallen?" | 30 s |
| 2. Audit | Audit page: the headline sentence, the 2025 card, "Burned area against detections", then the district table | S-NPP detections down 91.1%, burned area down 44.9%: detections per burned km² fell 77% to 84%, medium confidence. It is **not** the share of burning missed, there is no field check, and the burned-area product's own error is unknown. Districts are weaker: Ludhiana shows no sign of a gap, Tarn Taran is unclear. | 60 s |
| 3. Response Planner | Opens on 2025-10-28 with 500 units. Change the units to 300, then switch "Plan on likely burning" off. Put both back. | With 500 units Fazilka gets 57. With 300 the limit per district drops to 75 and Fazilka gets 45. On detections only, Sangrur leads with 44. Likely burning moves Fazilka from 34 to 57 units, and the range behind that is **low** confidence. The rule places units; no language model does. | 90 s |
| 4. Dispatch brief | Same page, "Dispatch brief": the badge, then Agent trace, Sources, Verifier report | Every number was checked against a tool output. The badge does **not** prove a source supports its sentence: an earlier brief passed while saying there were enough machines, the opposite of its source, and that is now checked too. | 45 s |
| 5. Evidence | "Does the forecast help decide where units go?", then the fire-risk bullets | The forecast reaches 95.3% of a perfect-knowledge allocation, against 91.1% for splitting by last season and 80.5% for an equal split: real but modest, and it assumes what a unit achieves. Its ranking lead over "last 3 days" is unproven. Fire features did not help the PM2.5 forecast. | 45 s |
| 6. Close | Stay on Evidence, "Limitations" | Triangulation, not ground truth. Support, not enforcement: nothing below a 5 km cell. Next: field checks and all-day geostationary fire data to settle the gap. | 30 s |

If you change the planner inputs, the saved brief says it was written for other inputs. That is expected: say so, or put the inputs back first. If asked about the other pages, show the Watch page (it opens on the 2025 peak day with Ludhiana) and the Replay map.

## 13. Pitch lines for the teacher

- **Problem:** fire counts fell sharply, but a satellite only sees what burns while it passes, so "how much fell" and "where is help needed" are both open.
- **Solution:** a triangulated audit with a stated range, a 3-day allocation of help, and briefs whose numbers are checked by code.
- **What is mine:** the detection-gap method, the allocator and its decision backtest, the upwind-pressure feature, and the number-checking Verifier. Open data and libraries are credited.
