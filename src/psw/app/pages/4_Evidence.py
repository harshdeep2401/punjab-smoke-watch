"""Evidence: the backtest, the ablation, how reliable the agent is, data sources and limitations."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from psw import config
from psw.app.common import CREDITS, LIMITATIONS, load_metrics
from psw.app.helpers import (VARIANT_LABEL, VARIANTS, by_city_frame, importance_frame, interpret, metrics_frame,
                             reliability)

st.set_page_config(page_title="Evidence", layout="wide")
st.title("Evidence")
st.caption("Everything on this page is computed from `data/processed/metrics.json` and the saved briefings. "
           "The target is CAMS modelled PM2.5 (via Open-Meteo), not a ground monitor.")

m = load_metrics()
mf = metrics_frame(m)
BEST = "background-color: #cfe8cf; font-weight: bold"


def highlight_best(df: pd.DataFrame, lower: list[str], target: dict[str, float] | None = None) -> pd.DataFrame:
    """Style: the best value in each column (lowest, or closest to a target)."""
    css = pd.DataFrame("", index=df.index, columns=df.columns)
    for c in lower:
        css.loc[df[c].idxmin(), c] = BEST
    for c, t in (target or {}).items():
        css.loc[(df[c] - t).abs().idxmin(), c] = BEST
    return css


# ---------- backtest ----------
st.header("Backtest: does the model beat simple baselines?")
st.write(f"Expanding-window backtest: for each test season ({', '.join(str(s) for s in m['seasons_test'])}) the model "
         "was trained only on earlier seasons. Green marks the best value in each column.")
st.dataframe(
    mf.style.apply(highlight_best, lower=["MAE (µg/m³)", "pinball", "band width (µg/m³)"],
                   target={"80% coverage (%)": 80.0}, axis=None)
    .format({"MAE (µg/m³)": "{:.2f}", "pinball": "{:.2f}", "80% coverage (%)": "{:.1f}", "band width (µg/m³)": "{:.1f}"}),
    hide_index=True, width="stretch")
st.caption("MAE is the error of the median forecast. Pinball loss scores all three quantiles. Coverage should be close to 80%; "
           "a narrower band is only better if it still covers.")

st.subheader("By city")
city_score = st.radio("Score", ["mae", "pinball", "coverage80", "band_width"], horizontal=True,
                      format_func=lambda s: {"mae": "MAE", "pinball": "Pinball", "coverage80": "80% coverage", "band_width": "Band width"}[s])
bc = by_city_frame(m, city_score)
if city_score == "coverage80":
    bc = bc * 100
    styled = bc.style.apply(lambda r: [BEST if abs(v - 80) == (r - 80).abs().min() else "" for v in r], axis=1).format("{:.1f}")
else:
    styled = bc.style.highlight_min(axis=1, props=BEST).format("{:.2f}")
st.dataframe(styled, width="stretch")

# ---------- ablation + importance ----------
st.header("Ablation: what do the fire features add?")
c1, c2 = st.columns(2)
with c1:
    best = mf["MAE (µg/m³)"].idxmin()
    fig = go.Figure(go.Bar(x=[VARIANT_LABEL[v] for v in VARIANTS], y=mf["MAE (µg/m³)"],
                           marker_color=["#2166ac" if i == best else "#b2b2b2" for i in range(len(mf))],
                           text=[f"{v:.2f}" for v in mf["MAE (µg/m³)"]], textposition="outside"))
    fig.update_layout(title="MAE by variant (lower is better)", yaxis_title="MAE (µg/m³)", height=380,
                      margin={"l": 10, "r": 10, "t": 50, "b": 10}, yaxis_range=[0, mf["MAE (µg/m³)"].max() * 1.2])
    st.plotly_chart(fig, width="stretch")
with c2:
    imp = importance_frame(m).iloc[::-1]
    fig = go.Figure(go.Bar(x=imp["gain"], y=imp["feature"], orientation="h",
                           marker_color=["#d6604d" if f else "#b2b2b2" for f in imp["fire feature"]]))
    fig.update_layout(title="Feature importance (gain, full model; red = fire feature)", height=380,
                      margin={"l": 10, "r": 10, "t": 50, "b": 10})
    st.plotly_chart(fig, width="stretch")

st.subheader("Reading the numbers")
for line in interpret(m):
    st.markdown(f"- {line}")
st.caption("Feature importance describes the final model trained on all seasons. It shows what the model used, "
           "not what causes PM2.5. Settings were not tuned on the test seasons.")

# ---------- agent reliability ----------
st.header("Agent reliability")
rel = reliability()
if rel.empty:
    st.info("No saved briefings were found in `data/processed/briefings/`.")
else:
    st.dataframe(rel.style.format({"pass rate (%)": "{:.0f}", "mean words": "{:.0f}"}), hide_index=True, width="stretch")
    st.write("`agent` means the first draft passed. `agent_revised` means it passed after one revision with the error list. "
             "`template` means the agent failed or was unavailable and a fixed template was used.")
    if rel["verified"].sum() == rel["briefings"].sum():
        st.write("Every saved briefing passed the Verifier.")
    else:
        st.warning(f"{int(rel['briefings'].sum() - rel['verified'].sum())} saved briefing(s) did not pass the Verifier.")
st.markdown(
    "- The Verifier checks that each number appears in a tool output or retrieved chunk, that each citation was retrieved, "
    "and the length and banned content. **It does not check that a cited source supports the sentence**, and it cannot tell "
    "\"rose\" from \"fell\".\n"
    "- The Sentinel flags a district on a smoggy day because of the nearest city's forecast, even when that district has no "
    "fires, so a flag is not by itself a sign of burning. The Watch page lists the most fire-heavy districts first.")

# ---------- sources ----------
st.header("Data sources and credits")
st.markdown(
    "| Data | Source |\n|---|---|\n"
    "| Fire detections | NASA FIRMS, VIIRS S-NPP 375 m (archive product, low-confidence detections dropped) |\n"
    "| PM2.5 (the forecast target) | Copernicus Atmosphere Monitoring Service (CAMS) via the Open-Meteo air-quality API |\n"
    "| Weather | Open-Meteo historical weather (ERA5 reanalysis) |\n"
    "| District boundaries | geoBoundaries IND ADM2 (gbOpen), licence ODbL 1.0, attribution to geoBoundaries (wmgeolab) |\n"
    "| Embeddings for the knowledge base | `BAAI/bge-small-en-v1.5` run locally with fastembed |\n"
    f"| PM2.5 reference line | India NAAQS 24-hour standard, {config.NAAQS_PM25_24H} µg/m³ |")
st.caption(CREDITS)

st.header("Limitations")
st.markdown(LIMITATIONS)
