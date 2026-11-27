"""Replay map: fires and district fire counts, day by day."""
import time
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

from psw import config
from psw.app.common import load_district_daily, load_fires, load_geojson

st.set_page_config(page_title="Replay map", layout="wide")
st.title("Replay map")

daily, fires, geo = load_district_daily(), load_fires(), load_geojson()
seasons = sorted(daily["season"].unique())

season = st.sidebar.selectbox("Season", seasons, index=len(seasons) - 1)
lo, hi = date(season, *config.SEASON_START), date(season, *config.SEASON_END)
key = f"day_{season}"
if key not in st.session_state:
    st.session_state[key] = date(season, 11, 5)
if st.session_state.get("pending_day") is not None:  # set by the play loop; applied before the slider exists
    st.session_state[key] = st.session_state.pop("pending_day")
st.session_state.setdefault("playing", False)

c1, c2 = st.columns([1, 6])
label = "⏸ Pause" if st.session_state["playing"] else "▶ Play"
if c1.button(label, key="play_toggle"):  # a stable key, so a label change does not leave a stale button
    st.session_state["playing"] = not st.session_state["playing"]
    st.rerun()
day = c2.slider("Date", min_value=lo, max_value=hi, key=key, format="YYYY-MM-DD")

today = daily[daily["date"] == day].set_index("district")
day_fires = fires[(fires["acq_date"] == day) & (fires["season"] == season)]


def colour(n: float, vmax: float) -> list[int]:
    t = float(np.sqrt(min(n / vmax, 1.0))) if vmax > 0 else 0.0  # white -> dark red
    return [int(255 - 115 * t), int(245 - 210 * t), int(235 - 215 * t), 170]


vmax = max(float(daily[daily["season"] == season]["fire_count"].quantile(0.99)), 1.0)
features = []
for f in geo["features"]:
    name = f["properties"]["district"]
    n = int(today["fire_count"].get(name, 0))
    features.append({**f, "tip": f"{name}: {n} fires",
                     "properties": {"district": name, "fire_count": n, "color": colour(n, vmax)}})
fc = {"type": "FeatureCollection", "features": features}

t = day_fires["acq_time"].astype(int).astype(str).str.zfill(4)
pts = day_fires[["longitude", "latitude"]].assign(
    radius=500 + 350 * np.sqrt(day_fires["frp"]),
    tip=(day_fires["district"].fillna("outside Punjab") + ": FRP " + day_fires["frp"].round(1).astype(str)
         + " MW at " + t.str[:2] + ":" + t.str[2:] + " UTC"),
)

layers = [
    pdk.Layer("GeoJsonLayer", fc, get_fill_color="properties.color", get_line_color=[90, 90, 90, 200],
              line_width_min_pixels=1, pickable=True, stroked=True, filled=True),
    pdk.Layer("ScatterplotLayer", pts, get_position=["longitude", "latitude"], get_radius="radius",
              get_fill_color=[220, 40, 20, 200], pickable=True, radius_min_pixels=2, radius_max_pixels=14),
]
view = pdk.ViewState(latitude=30.9, longitude=75.4, zoom=6.4)
left, right = st.columns([3, 1])
left.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=view, map_style=None,
                           tooltip={"text": "{tip}"}))
left.caption("Satellite detections of heat anomalies (VIIRS 375 m). Fires outside overpass times or under "
             "cloud are missed.")

in_punjab = int(today["fire_count"].sum())
mean7 = float(today["fire_count_7d_mean"].sum())
right.metric("Fires in Punjab today", in_punjab)
right.metric("7-day mean (all districts)", f"{mean7:.0f}")
right.markdown("**Top 5 districts**")
top = today["fire_count"].sort_values(ascending=False).head(5)
right.dataframe(pd.DataFrame({"district": top.index, "fires": top.values}), hide_index=True)

if st.session_state["playing"]:
    time.sleep(0.6)
    nxt = day + timedelta(days=1)
    if nxt > hi:
        st.session_state["playing"] = False
    else:
        st.session_state["pending_day"] = nxt
    st.rerun()
