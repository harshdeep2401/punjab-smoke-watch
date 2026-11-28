"""Audit: are fires really down? Triangulation of several signals, with a range and its confidence, never one number."""
import streamlit as st

from psw import config
from psw.agents import orchestrator
from psw.agents.audit_tools import AuditStore
from psw.app import audit_charts as charts
from psw.app import audit_view as view
from psw.app.common import get_kb, load_pm25
from psw.app.record_view import show_record
from psw.audit import gap
from psw.llm import LLM, LLMError

st.title("Is burning really down?")
st.caption(view.CAPTION)

tables = view.load_audit_tables(config.PROCESSED)
all_seasons = view.seasons(tables)
later = [s for s in all_seasons if s > view.BASE]
if not later:
    st.warning("No audit tables were found. Run `uv run python -m psw.pipeline audit --tables-only`.")
    st.stop()
theme = st.context.theme.type

# ---------- the finding ----------
st.markdown(view.headline_text(view.season_headline(tables, later[-1])))

left, right = st.columns([3, 2], gap="large")
with left:
    st.markdown(f"**Detections against burned area, Punjab ({view.BASE} = 100)**")
    indexed = view.indexed_frame(tables)
    if view.BURNED_SERIES not in set(indexed["series"]):
        st.info("Burned area is not available, so detections cannot be compared with it.")
    else:
        st.plotly_chart(charts.indexed_figure(indexed, theme), width="stretch")
        st.caption("MCD64A1 burned area is 500 m, misses small fields and is low confidence in cropland; its own error "
                   "here is unknown.")
with right:
    st.markdown("**Gap range by season**")
    for season in later:
        row = view.season_headline(tables, season)
        with st.container(border=True):
            a, b = st.columns([3, 2], vertical_alignment="center")
            a.metric(f"{season}", view.gap_text(row), help=row["methods"] or row["reason"])
            b.caption(f"Confidence: **{row['confidence'] or 'not estimated'}**  \n"
                      f"S-NPP {view.pct_text(row['snpp_change'])}  \nburned area {view.pct_text(row['burned_change'])}")
    st.caption("**The gap range** is the fall in detections per burned km² since the base season. It is not the share "
               f"of burning that is missed. {gap.GAP_CAVEAT}")

# ---------- why ----------
st.subheader("Why we suspect detection fell, not only burning")
st.caption(gap.EXPLANATIONS + " These signals speak to the first: timing.")
t_night, t_pass, t_air, t_official = st.tabs(["Night detections", "Each satellite's pass", "Smoke in cities",
                                              "Official counts"])
with t_pass:
    passes = view.pass_time_frame(tables)
    if passes.empty:
        st.info("No pass-time table is available.")
    else:
        st.plotly_chart(charts.pass_time_figure(passes, theme), width="stretch")
        st.caption("A satellite sees only what burns while it passes. The three VIIRS satellites pass at about the "
                   "same time, so they are not independent looks; MODIS Terra passes in the morning."
                   + (f" {note}" if (note := view.night_visibility_note(passes)) else ""))
        with st.expander("Numbers"):
            st.dataframe(passes.pivot_table(index=["sensor", "pass_label"], columns="season", values="count"),
                         width="stretch")
with t_night:
    nights = view.night_share_frame(tables)
    if nights.empty:
        st.info("No timing table is available.")
    else:
        st.plotly_chart(charts.night_share_figure(nights, theme), width="stretch")
        st.caption("Share of S-NPP detections from the night pass, with 95% intervals. Night detections are few, so a "
                   "rise is a lead, not proof.")
with t_air:
    pm25 = view.city_pm25_frame(load_pm25(str(config.PROCESSED)))
    if pm25.empty:
        st.info("No PM2.5 data is available.")
    else:
        st.plotly_chart(charts.pm25_figure(pm25, theme), width="stretch")
        st.caption(view.PM25_NOTE + " Not part of the gap range.")
with t_official:
    official = view.official_frame(tables)
    if official.empty:
        st.info("No official counts are available.")
    else:
        st.plotly_chart(charts.official_figure(official, theme), width="stretch")
        st.caption("The official counts are satellite-based too, so agreement with S-NPP is not independent "
                   "confirmation. Some official tables state no start date.")
        with st.expander("Numbers and sources"):
            st.dataframe(official.style.format({"ratio": "{:.2f}"}), hide_index=True, width="stretch")
            for url in view.official_sources(tables):
                st.markdown(f"- <{url}>")

# ---------- districts ----------
with st.expander("District by district"):
    season = st.selectbox("Season", later, index=len(later) - 1, key="table_season")
    table = view.district_table(tables, season)
    st.markdown(view.district_summary(table, season))
    st.dataframe(table.style.format({"S-NPP detections": "{:,.0f}", "detections change (%)": "{:+.1f}",
                                     "burned area (km²)": "{:,.0f}", "burned change (%)": "{:+.1f}",
                                     "gap low (%)": "{:.0f}", "gap high (%)": "{:.0f}"}, na_rep="-"),
                 hide_index=True, width="stretch", placeholder="-", height=35 * (len(table) + 1) + 3)
    st.caption("District figures are weaker than the state's: burned area collapses implausibly in places and night "
               "detections are too few to show a change in timing, so no district is above low confidence. `note` says "
               "why a district has no range.")

# ---------- audit brief ----------
st.subheader("The audit brief")
st.caption("Written by the **Auditor agent** (an LLM that calls data tools and searches the knowledge base), "
           "optionally after a **Planner agent**. The rule-based **Verifier** checks every number and citation, then "
           "the **Critic agent** reads it against its sources.")
places = [view.STATE] + view.districts(tables)
c1, c2, _ = st.columns([2, 1, 3])
place = c1.selectbox("Place", places, key="brief_place")
brief_season = c2.selectbox("Season", later, index=len(later) - 1, key="brief_season")

saved = orchestrator.audit_path(brief_season, place)
try:
    rec = orchestrator.load_record(saved)
except (OSError, ValueError):
    rec = None


def run_button(label: str) -> None:
    a, b = st.columns([1, 3], vertical_alignment="center")
    use_planner = b.checkbox("Plan first (one extra LLM call)", key=f"plan_{label}")
    if not a.button(label, help="Calls the LLM configured in .env. Saves the result to data/processed/audits/."):
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
    notes = tuple(f"Edited by hand after the run: {n}" for n in rec.get("manual_edits") or [])
    show_record(rec, "Place, season and the retrieval filters are fixed by the orchestrator, not chosen by the model.",
                caveat=rec.get("caveat") or gap.GAP_CAVEAT, notes=notes, writer="Auditor")
    run_button("Regenerate with the auditor")
