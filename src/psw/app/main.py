"""Punjab Smoke Watch: Streamlit entry point. Run: uv run streamlit run src/psw/app/main.py

The pages are grouped so the app reads as one story: the problem, the response, and how far to trust both. The
Part 1 tools stay available under "Explore the data".
"""
import streamlit as st

st.set_page_config(page_title="Punjab Smoke Watch", page_icon="🔥", layout="wide")

STORY = [
    st.Page("home.py", title="Overview", icon=":material/home:", default=True),
    st.Page("pages/5_Audit.py", title="1. Is burning really down?", icon=":material/search:", url_path="audit"),
    st.Page("pages/6_Response_Planner.py", title="2. Where should help go?", icon=":material/agriculture:",
            url_path="response"),
    st.Page("pages/4_Evidence.py", title="3. Can you trust it?", icon=":material/fact_check:", url_path="evidence"),
]
EXPLORE = [
    st.Page("pages/1_Replay_Map.py", title="Fire replay", icon=":material/map:", url_path="replay"),
    st.Page("pages/2_Forecast.py", title="City PM2.5 forecast", icon=":material/air:", url_path="forecast"),
    st.Page("pages/3_Watch_and_Briefing.py", title="Daily district briefing", icon=":material/description:",
            url_path="briefing"),
]

st.navigation({"The story": STORY, "Explore the data": EXPLORE}).run()
