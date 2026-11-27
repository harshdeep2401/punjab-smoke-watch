"""Punjab Smoke Watch: Streamlit entry point. Run: uv run streamlit run src/psw/app/main.py"""
import streamlit as st

from psw.app.common import CREDITS, LIMITATIONS

st.set_page_config(page_title="Punjab Smoke Watch", page_icon="🔥", layout="wide")

st.title("Punjab Smoke Watch")
st.write(
    "Satellite fire counts for Punjab have fallen steeply. This project asks two questions about that: **how much has "
    "burning really fallen**, and **where should help go in the next 3 days?** Everything runs offline from the "
    "committed `data/processed/` files."
)

left, right = st.columns(2)
with left.container(border=True):
    st.subheader("Audit: are fires really down?")
    st.write("Compares fire detections from several satellites and pass times with burned area and official counts, and "
             "reports a range with its confidence, never one number.")
    st.page_link("pages/5_Audit.py", label="Open the Audit page")
with right.container(border=True):
    st.subheader("Response: where should help go?")
    st.write("Forecasts the next 3 days of burning by district, splits a fleet of machines or teams with a fixed rule, "
             "and explains the split in a checked brief. Support, not enforcement.")
    st.page_link("pages/6_Response_Planner.py", label="Open the Response Planner")

st.markdown(
    "**Also here**\n"
    "- **Replay Map:** fire detections day by day.\n"
    "- **Forecast:** next-day PM2.5 for four cities with an 80% band.\n"
    "- **Watch and Briefing:** a rule-based district watchlist and cited, number-checked briefings.\n"
    "- **Evidence:** the backtests, whether the forecast helps decisions, and how reliable the agents are.\n\n"
    "**Rule: the language model never does the maths.** Every number comes from Python, and a Verifier that is not a "
    "language model checks it. It does not prove that a cited source supports a sentence."
)
with st.expander("Limitations (please read)", expanded=True):
    st.markdown(LIMITATIONS)
st.caption(CREDITS)
