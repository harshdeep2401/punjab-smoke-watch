"""Punjab Smoke Watch: Streamlit entry point. Run: uv run streamlit run src/psw/app/main.py"""
import streamlit as st

from psw.app.common import CREDITS, LIMITATIONS

st.set_page_config(page_title="Punjab Smoke Watch", page_icon="🔥", layout="wide")

st.title("Punjab Smoke Watch")
st.write(
    "A stubble-fire and air-quality intelligence pipeline for Punjab. It joins satellite fire detections "
    "(NASA FIRMS) with weather and modelled PM2.5, aggregates them to districts, and measures how much "
    "upwind fire pressure helps forecast next-day PM2.5 in four cities. Use the pages in the sidebar. "
    "Everything runs offline from the committed `data/processed/` files."
)
with st.expander("Limitations (please read)", expanded=True):
    st.markdown(LIMITATIONS)
st.caption(CREDITS)
