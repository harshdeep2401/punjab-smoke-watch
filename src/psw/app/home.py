"""Overview: the problem in three numbers and one chart, then the way through the app."""
import streamlit as st

from psw import config
from psw.app import audit_charts as charts
from psw.app import audit_view as view
from psw.app.common import CREDITS, LIMITATIONS, LLM_RULE, load_pm25
from psw.app.helpers import AGENTS

tables = view.load_audit_tables(config.PROCESSED)
later = [s for s in view.seasons(tables) if s > view.BASE]
pm25 = view.city_pm25_frame(load_pm25(str(config.PROCESSED)))

st.title("Punjab Smoke Watch")
st.markdown("##### Satellite fire counts in Punjab have collapsed. Has the burning?")

if later:
    last = later[-1]
    row = view.season_headline(tables, last)
    cards = (
        ("Satellite fire detections", row["snpp_change"], "VIIRS S-NPP, the afternoon pass most counts rely on."),
        ("Burned area", row["burned_change"], "MODIS MCD64A1: the scars left behind, whenever the fire burned."),
        ("Smoke in four cities", view.pm25_change(pm25, last), "Season-mean PM2.5 (CAMS, modelled) in Ludhiana, "
                                                               "Amritsar, Patiala and Bathinda."),
    )
    for column, (label, value, note) in zip(st.columns(3), cards):
        with column.container(border=True):
            st.metric(f"{label}, {view.BASE} → {last}", view.pct_text(value))
            st.caption(note)

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.plotly_chart(charts.indexed_figure(view.with_pm25(view.indexed_frame(tables), pm25),
                                              st.context.theme.type, height=380), width="stretch")
    with right:
        st.markdown("#### The problem")
        fell_less = (row["burned_change"] is not None and row["snpp_change"] is not None
                     and row["burned_change"] - row["snpp_change"] >= 20)
        st.markdown(
            "If burning had fallen as steeply as the counts, the burned area and the smoke should have fallen with it. "
            + ("They fell far less. " if fell_less else "Compare the lines on the left. ")
            + "One afternoon satellite pass carries most of the counting, and public reports since 2024 say burning "
              "may have moved outside it. A plan built only on those counts could send help to the wrong places.")
        st.markdown("#### What this app does")
        st.markdown(
            "It **audits** the fall against independent signals and gives a range for the gap, never one number. It "
            "then **plans** where a limited fleet of residue machines and awareness teams should go over the next "
            "3 days, for the burning that is *likely*, not just detected.")
else:
    st.warning("No audit tables were found. Run `uv run python -m psw.pipeline audit --tables-only`.")

st.divider()
STEPS = (
    ("pages/5_Audit.py", "1. Is burning really down?",
     "Several satellites, pass times, burned area and official counts, triangulated into a gap range."),
    ("pages/6_Response_Planner.py", "2. Where should help go?",
     "A 3-day district forecast, a fixed rule that splits the fleet, and a checked brief for district officers."),
    ("pages/4_Evidence.py", "3. Can you trust it?",
     "Backtests, a forward test on the 2026 season, how often the agents pass their checks, and what is still "
     "unknown."),
)
for column, (page, title, text) in zip(st.columns(3), STEPS):
    with column.container(border=True, height="stretch"):
        st.page_link(page, label=f"**{title}**", icon=":material/arrow_forward:")
        st.caption(text)

st.markdown("#### The AI agents behind it")


def chip(name: str) -> str:
    llm = AGENTS[name][0].startswith("LLM")
    return f":{'violet' if llm else 'gray'}-badge[{':material/smart_toy:' if llm else ':material/rule:'} {name}]"


FLOW = [["Sentinel"], ["Planner"], ["Analyst", "Auditor", "Dispatcher"], ["Verifier"], ["Critic"]]
st.markdown(" &nbsp;→&nbsp; ".join(" / ".join(chip(n) for n in step) for step in FLOW))
st.caption("Violet: an LLM agent. Grey: rules, no LLM. The middle step is one of three writing agents, depending on the "
           "page.")
st.markdown("\n".join(f"- **{name}** ({'rule-based, no LLM' if kind == 'rules' else 'an LLM agent' + (' with data tools' if 'tools' in kind else '')}): {role}."
                      for name, (kind, role) in AGENTS.items()))
st.caption(LLM_RULE + " Each brief on the next pages shows which agents wrote and checked it. Decision support, not "
           "enforcement: nothing finer than a 5 km cell, no farms or farmers named.")
with st.expander("Limitations"):
    st.markdown(LIMITATIONS)
st.caption(CREDITS)
