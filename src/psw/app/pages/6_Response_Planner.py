"""Response Planner: where should residue-management help go in the next 3 days? Support, not enforcement."""
from datetime import date as Date

import pydeck as pdk
import streamlit as st

from psw import config
from psw.agents import dispatch_tools, orchestrator
from psw.agents.demo import load_demo
from psw.agents.dispatch_tools import ResponseStore, Settings, plan_allocation
from psw.app import planner_charts as charts
from psw.app import planner_view as view
from psw.app.common import get_kb, load_district_daily, load_geojson
from psw.app.helpers import load_risk_metrics
from psw.app.record_view import show_record
from psw.llm import LLM, LLMError
from psw.response import allocate, harvest

st.title("Where should help go in the next 3 days?")
st.caption(view.CAPTION)


@st.cache_resource(show_spinner=False)
def get_response_store(root: str) -> ResponseStore:
    """One store per data folder: it holds the forecasts and the 5 km grid."""
    return ResponseStore(root)


store = get_response_store(str(config.PROCESSED))
try:
    dates, variant = store.dates, store.variant
    metrics = load_risk_metrics(config.PROCESSED)
except (OSError, KeyError, ValueError):
    st.warning("No fire-risk forecasts were found. Run `uv run python -m psw.pipeline risk-train`.")
    st.stop()

# ---------- inputs ----------
st.sidebar.markdown("**Plan settings**")
day = st.sidebar.date_input("Issue date", value=view.default_date(load_district_daily(), dates), min_value=dates[0],
                            max_value=dates[-1], format="YYYY-MM-DD")
units = int(st.sidebar.number_input("Machines or teams available", min_value=1, max_value=5000,
                                    value=allocate.DEFAULT_UNITS, step=50))
max_share = st.sidebar.slider("Most one district may take (%)", 5, 100, int(allocate.DEFAULT_MAX_SHARE * 100), 5) / 100
use_likely = st.sidebar.toggle("Plan on likely burning", value=True,
                               help="On: the forecast scaled up by last season's detection gap (from the audit). "
                                    "Off: forecast detections only.")
saved_days = [d["date"] for d in (load_demo(config.PROCESSED) or {}).get("dispatches", [])]
st.sidebar.caption("Forecasts exist from 15 September to 27 November of each test season."
                   + (f" Saved briefs: {', '.join(saved_days)}." if saved_days else ""))

if not isinstance(day, Date) or day not in set(dates):
    st.warning(f"No fire-risk forecast was issued on {day}. Pick a date from 15 September to 27 November of "
               f"{' or '.join(str(s) for s in sorted({d.year for d in dates}))}.")
    st.stop()

allocation = plan_allocation(store, day, Settings(units, max_share, use_likely))
first, last = view.window(day)
basis = "likely burning" if use_likely else "forecast detections"

# ---------- the plan ----------
st.subheader(f"Plan for {first:%d %b} to {last:%d %b %Y}")
for line in view.plan_sentences(allocation):
    st.markdown(line)
figures = [("Units placed", allocation.placed), ("Districts with units", int((allocation.table["units"] > 0).sum())),
           ("Limit per district", allocation.cap)]
if allocation.placed < units:
    figures.append(("Not placed", units - allocation.placed))
for column, (label, value) in zip(st.columns(4), figures):
    column.metric(label, value)

left, right = st.columns([1, 1], gap="large")
with left:
    st.markdown(f"**Expected {basis}, next 3 days**")
    features, vmax = view.risk_features(load_geojson(), allocation)
    lat, lon, zoom = view.state_view(store, width=560, height=460)
    st.pydeck_chart(pdk.Deck(
        layers=[pdk.Layer("GeoJsonLayer", features, get_fill_color="properties.color",
                          get_line_color=[90, 90, 90, 220], line_width_min_pixels=1, pickable=True, stroked=True,
                          filled=True)],
        initial_view_state=pdk.ViewState(latitude=lat, longitude=lon, zoom=zoom), map_provider=None,
        tooltip={"text": "{tip}"}), height=460)
    st.caption(f"Darker means more; the darkest district is at {vmax:.0f}. Hover for figures.")
with right:
    st.markdown("**Units by district**")
    table = view.allocation_table(allocation)
    st.plotly_chart(charts.allocation_figure(table, use_likely, st.context.theme.type), width="stretch")

with st.expander("District figures and how the split is made"):
    st.dataframe(view.risk_table(store, allocation, day), hide_index=True, width="stretch", placeholder="-")
    st.dataframe(table, hide_index=True, width="stretch")
    variant_words = {"model": "the LightGBM model", "persistence": "persistence", "climatology": "climatology"}[variant]
    st.markdown(
        f"- **Forecast detections** are the median of {variant_words}, the variant that ranked districts best in the "
        "backtest. The 80% band is its own uncertainty.\n"
        f"- **The split:** a fixed rule places units one at a time where the next one adds most, assuming u units deal "
        f"with 1 − exp(−u / {allocate.DEFAULT_K:g}) of a district's burning. That share is an assumption, not a "
        "measurement. No language model sets any figure here.\n"
        "- **Gap confidence** belongs to the range named under 'scaled by'. A 'medium' beside 'the Punjab range' is "
        "about the state as a whole and says nothing about that district.")
    if metrics:
        st.caption(metrics["likely_note"])
        st.caption(metrics["rain_assumption"])

# ---------- dispatch brief ----------
st.subheader("The dispatch brief")
st.caption("What a district agriculture officer or the state control room would receive. The **Dispatcher agent** "
           "(an LLM with data tools) explains the plan above but cannot change it; the rule-based **Verifier** checks "
           "every figure against the right district, and the **Critic agent** reads it against its sources.")
path = orchestrator.dispatch_path(str(day))
try:
    saved = orchestrator.load_record(path)
except (OSError, ValueError):
    saved = None
session = st.session_state.setdefault("dispatch_runs", {})
session_key = (str(day), units, max_share, use_likely)
rec = session.get(session_key) or saved
names = list(allocation.table["district"])


def run_button(label: str) -> None:
    a, b = st.columns([1, 3], vertical_alignment="center")
    focus = b.selectbox("Add a paragraph about", ["no district"] + names, key=f"paragraph_{label}",
                        label_visibility="collapsed", format_func=lambda d: f"Add a paragraph about {d}"
                        if d != "no district" else "No district paragraph")
    if not a.button(label, help="Calls the LLM configured in .env, with the plan settings in the sidebar."):
        return
    try:
        llm = LLM.from_env()
    except LLMError:
        st.warning("No LLM is configured, so the dispatcher cannot run here. Set `LLM_BASE_URL`, `LLM_API_KEY` and "
                   "`LLM_MODEL` in `.env` (see `.env.example`). Saved briefs can still be read.")
        return
    with st.spinner(f"Writing the dispatch for {day} (a few minutes on a free tier)…"):
        try:
            new = orchestrator.run_dispatch(day, units=units, max_share=max_share, use_likely=use_likely,
                                            district=None if focus == "no district" else focus, llm=llm,
                                            store=store, kb=get_kb(), save=False)
        except Exception as e:  # noqa: BLE001 - the page must not crash on a provider or data problem
            st.error(f"The run failed: {type(e).__name__}: {e}")
            return
    outcome = view.keep_dispatch(new.to_dict(), saved)
    if outcome == "kept":
        st.warning("The agent failed or its draft did not pass the Verifier, so the saved brief was kept."
                   + (f" Reason: {new.llm_error}" if new.llm_error else ""))
        return
    if outcome == "save":
        orchestrator.save_dispatch(new)
        session.pop(session_key, None)
    else:
        session[session_key] = new.to_dict()
    st.rerun()


if rec is None:
    st.info(f"No saved dispatch brief for {day}.")
    run_button("Run dispatcher")
else:
    if not view.matches(rec, units, max_share, use_likely):
        old = "likely burning" if rec["use_likely"] else "forecast detections"
        st.warning(f"**This brief was written for other inputs:** {rec['units']} units, at most "
                   f"{rec['max_share']:.0%} per district, planned on {old}. Its unit figures do not match the plan "
                   "above.")
    if session_key in session:
        st.caption("Made in this session. The brief saved for this date, written for other inputs, is unchanged.")
    show_record(rec, "The date, units, limit, planning basis and retrieval filter are fixed by the orchestrator, not "
                     "chosen by the model.", caveat=rec.get("caveat") or view.CAPTION, writer="Dispatcher")
    run_button("Run dispatcher for these inputs")

# ---------- one district ----------
st.subheader("Inside one district")
district = st.selectbox("District", names, key=f"cells_{day}_{use_likely}")
cells, singles = view.cluster_cells(store, district, day)
left, right = st.columns([1, 1], gap="large")
with left:
    lat, lon, zoom = view.district_view(store, district, width=420, height=420)
    outlines = [{"type": "Feature", "geometry": f["geometry"], "tip": f["properties"]["district"],
                 "properties": {"fill": [150, 150, 150, 60 if f["properties"]["district"] == district else 0]}}
                for f in load_geojson()["features"]]
    st.pydeck_chart(pdk.Deck(layers=[
        pdk.Layer("GeoJsonLayer", {"type": "FeatureCollection", "features": outlines}, get_fill_color="properties.fill",
                  get_line_color=[110, 110, 110, 200], line_width_min_pixels=1, stroked=True, filled=True),
        pdk.Layer("GeoJsonLayer", view.cell_features(store, cells), get_fill_color=[214, 60, 40, 190],
                  get_line_color=[130, 20, 10, 255], line_width_min_pixels=1, pickable=True, stroked=True, filled=True),
    ], initial_view_state=pdk.ViewState(latitude=lat, longitude=lon, zoom=zoom), map_provider=None,
        tooltip={"text": "{tip}"}), height=420)
with right:
    units_here = int(allocation.table.set_index("district")["units"][district])
    rain = dispatch_tools.get_rain_outlook(store, district, str(day))
    a, b = st.columns(2)
    a.metric("Units in this plan", units_here)
    b.metric("Rain, next 3 days", f"{rain['rain_next_3_days_mm']} mm",
             help=f"{rain['rain_last_3_days_mm']} mm in the last 3 days. {harvest.RAIN_ASSUMPTION}")
    st.markdown("**Where awareness teams might start** (5 km cells, detections in the last 7 days)")
    if cells:
        st.dataframe([{"cell": c["label"], "detections, 7 days": c["detections_7d"],
                       "share of district (%)": c["share_pct"]} for c in cells], hide_index=True, width="stretch")
    else:
        st.info(f"No 5 km cell in {district} had more than a single detection in the last 7 days, so there is no "
                "cluster to point to.")
    st.caption((f"{singles} other cell(s) had a single detection and are not drawn: one detection is not a cluster. "
                if singles else "") + "Cells come from detections, so they cannot show burning that was not detected.")
    stock = dispatch_tools.get_machine_stock(store, district, str(day))
    if "error" not in stock:
        largest = ", ".join(f"{t['type']} {t['supplied']:,}" for t in stock["largest_types"][:3])
        st.caption(f"Machines supplied on subsidy in {district}, {stock['period']}: {stock['machines_supplied']:,} "
                   f"(largest: {largest}). Context only, not used in the split. [Source]({stock['source_url']})")
