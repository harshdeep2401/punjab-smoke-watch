"""Audit: are fires really down? Triangulation of several signals, with a range and its confidence, never one number."""
import streamlit as st

from psw import config
from psw.agents import audit_tools, orchestrator
from psw.agents.audit_tools import AuditStore
from psw.app import audit_charts as charts
from psw.app import audit_view as view
from psw.app.common import get_kb
from psw.app.record_view import show_provenance, show_tabs, show_text
from psw.audit import gap
from psw.llm import LLM, LLMError

st.set_page_config(page_title="Audit", layout="wide")
st.title("Audit: are fires really down?")
st.caption(view.CAPTION)

tables = view.load_audit_tables(config.PROCESSED)
all_seasons = view.seasons(tables)
later = [s for s in all_seasons if s > view.BASE]
if not later:
    st.warning("No audit tables were found. Run `uv run python -m psw.pipeline audit --tables-only`.")
    st.stop()
theme = st.context.theme.type

# ---------- headline ----------
st.header(f"Punjab against {view.BASE}")
st.markdown(view.headline_text(view.season_headline(tables, later[-1])))
for column, season in zip(st.columns(len(later)), later):
    row = view.season_headline(tables, season)
    with column.container(border=True):
        st.subheader(str(season))
        st.metric("S-NPP detections", view.pct_text(row["snpp_change"]))
        st.metric("All sensors with both seasons", view.pct_text(row["all_change"]),
                  help="Summed detections of " + ", ".join(row["all_sensors"]) + ".")
        st.metric("MODIS burned area", view.pct_text(row["burned_change"]))
        night = None if row["night_share"] is None else row["night_share"] * 100
        base_night = None if row["base_night_share"] is None else row["base_night_share"] * 100
        st.metric("Night share of S-NPP", view.pct_text(night, signed=False),
                  help=f"{view.BASE}: {view.pct_text(base_night, signed=False)}.")
        st.metric("Gap range", view.gap_text(row), help=row["methods"] or row["reason"])
        st.caption(f"Confidence: {row['confidence'] or 'not estimated'}")
st.info("**What the gap range is:** the fall in fire detections per burned km² since the base season. It is not the "
        f"share of burning that is missed. {gap.GAP_CAVEAT}\n\n{gap.EXPLANATIONS}")

# ---------- charts ----------
st.header("The signals")

st.subheader("Detections by time of satellite pass (Punjab)")
passes = view.pass_time_frame(tables)
if passes.empty:
    st.info("No pass-time table is available.")
else:
    st.plotly_chart(charts.pass_time_figure(passes, theme), width="stretch")
    st.caption("A satellite only sees what is burning when it passes. The three VIIRS satellites pass at about the same "
               "time, so they are not independent looks. A missing bar means the sensor has no data for that season.")
    if note := view.night_visibility_note(passes):
        st.caption(note)
    with st.expander("Show the numbers (detections by pass time)"):
        st.dataframe(passes.pivot_table(index=["sensor", "pass_label"], columns="season", values="count"),
                     width="stretch")

left, right = st.columns(2)
with left:
    st.subheader("Night share of S-NPP detections")
    nights = view.night_share_frame(tables)
    if nights.empty:
        st.info("No timing table is available.")
    else:
        st.plotly_chart(charts.night_share_figure(nights, theme), width="stretch")
        st.caption("Bars show 95% intervals. Night detections are few, so a rise is a lead, not proof.")
        with st.expander("Show the numbers (night share)"):
            st.dataframe(nights.style.format({"share_pct": "{:.2f}", "low_pct": "{:.2f}", "high_pct": "{:.2f}"}),
                         hide_index=True, width="stretch")
with right:
    st.subheader("Burned area against detections")
    indexed = view.indexed_frame(tables)
    if view.BURNED_SERIES not in set(indexed["series"]):
        st.info("Burned area is not available, so detections cannot be compared with it.")
    else:
        st.plotly_chart(charts.indexed_figure(indexed, theme), width="stretch")
        st.caption("Each series is 100 in the base season. MCD64A1 is 500 m, misses small fields and is low confidence "
                   "in cropland, and its own error here is unknown.")
        with st.expander("Show the numbers (burned area and detections)"):
            st.dataframe(indexed.style.format({"value": "{:,.0f}", "index": "{:.1f}"}), hide_index=True,
                         width="stretch")

st.subheader("Official counts against satellite detections")
official = view.official_frame(tables)
if official.empty:
    st.info("No official counts are available.")
else:
    st.plotly_chart(charts.official_figure(official, theme), width="stretch")
    st.caption("The official counts are satellite-based too, so agreement with S-NPP is not independent confirmation. "
               "Some official tables state no start date.")
    with st.expander("Show the numbers (official counts) and their sources"):
        st.dataframe(official.style.format({"ratio": "{:.2f}"}), hide_index=True, width="stretch")
        for url in view.official_sources(tables):
            st.markdown(f"- <{url}>")

# ---------- districts ----------
st.header("Districts")
season = st.selectbox("Season", later, index=len(later) - 1, key="table_season")
table = view.district_table(tables, season)
st.markdown(view.district_summary(table, season))
st.dataframe(table.style.format({"S-NPP detections": "{:,.0f}", "detections change (%)": "{:+.1f}",
                                 "burned area (km²)": "{:,.0f}", "burned change (%)": "{:+.1f}",
                                 "gap low (%)": "{:.0f}", "gap high (%)": "{:.0f}"}, na_rep="-"),
             hide_index=True, width="stretch", placeholder="-",
             height=35 * (len(table) + 1) + 3)  # every district without scrolling
st.caption("The last column, `note`, says why a district has no range (scroll the table sideways). District figures are weaker than the state's: burned area collapses implausibly in places and night "
           "detections are too few to show a change in timing, so no district is above low confidence. The ranges do "
           "not include the error of the burned-area product.")

# ---------- audit brief ----------
st.header("Audit brief")
places = [view.STATE] + view.districts(tables)
c1, c2 = st.columns(2)
place = c1.selectbox("Place", places, key="brief_place")
brief_season = c2.selectbox("Season", later, index=len(later) - 1, key="brief_season")

saved = orchestrator.audit_path(brief_season, place)
try:
    rec = orchestrator.load_record(saved)
except (OSError, ValueError):
    rec = None


def run_button(label: str) -> None:
    use_planner = st.checkbox("Plan first (one extra LLM call)", key=f"plan_{label}")
    if not st.button(label, help="Calls the LLM configured in .env. Saves the result to data/processed/audits/."):
        return
    try:
        llm = LLM.from_env()
    except LLMError:
        st.warning("No LLM is configured, so the auditor cannot run here. Set `LLM_BASE_URL`, `LLM_API_KEY` and "
                   "`LLM_MODEL` in `.env` (see `.env.example`). Saved audits can still be read.")
        return
    with st.spinner(f"Auditing {place}, {brief_season} (about a minute on a free tier)…"):
        try:
            new = orchestrator.run_audit(place, brief_season, llm=llm, store=AuditStore(), kb=get_kb(), save=False,
                                         use_planner=use_planner)
        except Exception as e:  # noqa: BLE001 - the page must not crash on a provider or data problem
            st.error(f"The run failed: {type(e).__name__}: {e}")
            return
    if new.mode == "template" and rec is not None and rec["mode"] != "template":
        st.warning("The agent failed or its draft did not pass the Verifier, so the saved audit was kept."
                   + (f" Reason: {new.llm_error}" if new.llm_error else ""))
        return
    orchestrator.save_audit(new)
    st.rerun()


if rec is None:
    st.info(f"No saved audit for {place} in {brief_season}.")
    run_button("Run auditor")
else:
    left, right = st.columns([5, 4])
    with left:
        show_text(rec)
        st.caption(rec.get("caveat") or gap.GAP_CAVEAT)
        for note in rec.get("manual_edits") or []:
            st.caption(f"Edited by hand after the run: {note}")
        show_provenance(rec)
        run_button("Regenerate with the auditor")
    with right:
        show_tabs(rec, "Place, season and the retrieval filters are fixed by the orchestrator, not chosen by the model.")

with st.expander("How the auditor works"):
    st.markdown("An LLM agent calls the tools below, writes the brief in four labelled parts, and the same Verifier "
                "checks it, with the statements every audit must make. One revision, then a fixed template. A Critic "
                "agent then reads it and may only delete, reorder or reword.")
    for tool in audit_tools.TOOL_SCHEMAS:
        st.markdown(f"- `{tool['function']['name']}`: {tool['function']['description']}")

st.divider()
st.caption("Fire detections: NASA FIRMS (VIIRS S-NPP, NOAA-20, NOAA-21; MODIS Terra and Aqua). Burned area: MODIS "
           "MCD64A1 from the University of Maryland. Official counts: Punjab Pollution Control Board reports.")
st.caption(view.CAPTION)
