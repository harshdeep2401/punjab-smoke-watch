"""Forecast: next-day PM2.5 with an 80% band, for the out-of-sample (test) seasons only."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from psw import config
from psw.agents import tools
from psw.app.common import get_store, load_forecasts, load_metrics

st.title("City PM2.5 forecast")

fc = load_forecasts()
metrics = load_metrics()
store = get_store()
test_seasons = sorted(int(s) for s in metrics["seasons_test"])

city = st.sidebar.selectbox("City", list(config.CITIES))
season = st.sidebar.selectbox("Season", test_seasons, index=len(test_seasons) - 1, help="Test seasons only")
show_persistence = st.sidebar.toggle("Show persistence baseline", value=True)
st.caption(f"Next-day PM2.5 with an 80% band. Season {season} is out of sample: its model was trained on earlier seasons only.")

sel = fc[(fc["city"] == city) & (fc["season"] == season)]
full = sel[sel["variant"] == "full"].sort_values("target_date").reset_index(drop=True)
pers = sel[sel["variant"] == "persistence"].sort_values("target_date").reset_index(drop=True)
if full.empty:
    st.warning(f"No forecasts for {city} in {season}.")
    st.stop()

x = pd.to_datetime(full["target_date"])
fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=full["p90"], mode="lines", line={"width": 0}, hoverinfo="skip", showlegend=False))
fig.add_trace(go.Scatter(x=x, y=full["p10"], mode="lines", line={"width": 0}, fill="tonexty",
                         fillcolor="rgba(214, 96, 77, 0.25)", name="80% range (p10 to p90)", hoverinfo="skip"))
fig.add_trace(go.Scatter(x=x, y=full["p50"], mode="lines", line={"color": "#b2182b", "width": 2}, name="Forecast (p50)"))
if show_persistence and not pers.empty:
    fig.add_trace(go.Scatter(x=pd.to_datetime(pers["target_date"]), y=pers["p50"], mode="lines",
                             line={"color": "#555", "width": 1.5, "dash": "dash"}, name="Persistence (today's value)"))
fig.add_trace(go.Scatter(x=x, y=full["actual"], mode="markers", marker={"color": "#1f1f1f", "size": 5},
                         name="Actual (CAMS)"))
fig.add_hline(y=config.NAAQS_PM25_24H, line={"color": "#2166ac", "dash": "dot"},
              annotation_text="NAAQS 24 h", annotation_position="top left")

dates = [pd.Timestamp(d).date() for d in full["target_date"]]
default = dates[int(full["p50"].idxmax())]
chosen = st.slider("Forecast for (target date)", min_value=dates[0], max_value=dates[-1], value=default,
                   key=f"fc_{city}_{season}", format="YYYY-MM-DD")
fig.add_shape(type="line", x0=pd.Timestamp(chosen), x1=pd.Timestamp(chosen), y0=0, y1=1, yref="paper",
              line={"color": "#888", "width": 1})
fig.update_layout(height=430, margin={"l": 10, "r": 10, "t": 10, "b": 10}, yaxis_title="PM2.5 (µg/m³)",
                  legend={"orientation": "h", "y": 1.08}, hovermode="x unified")
st.plotly_chart(fig, width="stretch")

row = full[full["target_date"].astype(str) == chosen.isoformat()]
if row.empty:
    st.info("No forecast for that date.")
else:
    r = row.iloc[0]
    issue = str(r["issue_date"])
    left, right = st.columns([3, 2])
    with left:
        st.subheader(f"Forecast for {chosen.isoformat()}: {r['p50']:.1f} µg/m³ (80% range {r['p10']:.1f}–{r['p90']:.1f})")
        st.write(f"{city}, issued on {issue} using data up to that day. "
                 f"CAMS value on the target day: **{r['actual']:.1f} µg/m³** (only known because this is a backtest).")
        p = pers[pers["target_date"].astype(str) == chosen.isoformat()]
        if not p.empty:
            st.write(f"Persistence baseline: {p.iloc[0]['p50']:.1f} µg/m³.")
    with right:
        w = tools.get_wind_and_upwind(store, city, issue)
        st.markdown(f"**Upwind fires on {issue}**")
        if "error" in w:
            st.write("No wind data for this date.")
        else:
            st.write(f"Wind at {city} blows from the {w['wind_dir_compass']} ({w['wind_dir_from_deg']:.0f}°) "
                     f"at {w['wind_speed_kmh']:.1f} km/h.")
            if w["top_upwind_districts"]:
                st.dataframe(pd.DataFrame(w["top_upwind_districts"]).rename(
                    columns={"district": "district", "share_pct": "share of upwind pressure (%)"}), hide_index=True)
            else:
                st.write("No detections upwind within 300 km.")

# season-level accuracy for this city, computed from the table above
err = (full["p50"] - full["actual"]).abs().mean()
cover = ((full["actual"] >= full["p10"]) & (full["actual"] <= full["p90"])).mean() * 100
pers_err = (pers["p50"] - pers["actual"]).abs().mean() if not pers.empty else float("nan")
st.caption(f"{city}, {season}: {len(full)} forecasts, MAE {err:.1f} µg/m³ (persistence {pers_err:.1f}), "
           f"80% band covered {cover:.0f}% of actual values.")
st.caption("Target is CAMS modelled PM2.5 (via Open-Meteo), not a ground monitor. "
           "Upwind pressure is context for the forecast, not proof of cause.")
