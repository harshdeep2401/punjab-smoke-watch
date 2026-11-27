"""Plotly figures for the Audit page. Pure: a frame from audit_view in, a figure out.

Colour follows the thing measured, the same on every chart: S-NPP, burned area and official counts keep one colour
each, and the pass times have their own set. Both sets, light and dark, passed the dataviz skill's validator.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from psw.app.audit_view import BASE, BURNED_SERIES, SNPP_SERIES
from psw.audit.sensors import AFTERNOON, MORNING, NIGHT, OTHER

COLOURS = {
    "light": {"snpp": "#257de0", "burned": "#f47230", "official": "#00b894", MORNING: "#6a4ce2",
              AFTERNOON: "#f2a600", NIGHT: "#ee7fac", OTHER: "#b2b2b2", "surface": "#ffffff"},
    "dark": {"snpp": "#338ced", "burned": "#e76624", "official": "#00a888", MORNING: "#9887ef",
             AFTERNOON: "#d28e05", NIGHT: "#e24d8b", OTHER: "#6b6b6b", "surface": "#0e1117"},
}
# the hours of audit.sensors.pass_label, in Indian time: Aqua has drifted well past 13:30
PASS_NAMES = {MORNING: "morning (10 to 12 h)", AFTERNOON: "afternoon (12 to 16 h)", NIGHT: "night (0 to 4 h)",
              OTHER: "other hours"}
MARGIN = {"l": 10, "r": 10, "t": 40, "b": 10}


def _colours(theme: str | None) -> dict:
    return COLOURS["dark" if theme == "dark" else "light"]


def _layout(fig: go.Figure, height: int, **kw) -> go.Figure:
    fig.update_layout(height=height, margin=MARGIN, barcornerradius=4,
                      legend={"orientation": "h", "y": -0.15, "x": 0}, **kw)
    return fig


def pass_time_figure(frame: pd.DataFrame, theme: str | None = None) -> go.Figure:
    """One panel per sensor, one bar per season, stacked by the time of the pass. One shared count axis."""
    c = _colours(theme)
    sensors = list(dict.fromkeys(frame["sensor"]))
    fig = make_subplots(rows=1, cols=len(sensors), shared_yaxes=True, subplot_titles=sensors, horizontal_spacing=0.03)
    seasons = sorted(frame["season"].unique())
    shown = set()
    for col, sensor in enumerate(sensors, 1):
        for label, name in PASS_NAMES.items():
            part = frame[(frame["sensor"] == sensor) & (frame["pass_label"] == label)]
            if part.empty or part["count"].sum() == 0:
                continue
            # every season on every panel, so a season without data shows as a gap and the panels line up
            counts = part.set_index("season")["count"].reindex(seasons)
            fig.add_trace(go.Bar(x=[str(s) for s in seasons], y=counts, name=name, legendgroup=label,
                                 showlegend=label not in shown, width=0.55,
                                 marker={"color": c[label], "line": {"color": c["surface"], "width": 1}},
                                 hovertemplate=f"{sensor}, %{{x}}<br>{name}: %{{y:,}}<extra></extra>"), row=1, col=col)
            shown.add(label)
        fig.update_xaxes(type="category", categoryorder="array", categoryarray=[str(s) for s in seasons],
                         range=[-0.5, len(seasons) - 0.5], row=1, col=col)
    fig.update_yaxes(title_text="detections", row=1, col=1)
    return _layout(fig, 380, barmode="stack")


def night_share_figure(frame: pd.DataFrame, theme: str | None = None) -> go.Figure:
    """The night share of S-NPP detections with its 95% interval. One series, so no legend."""
    c = _colours(theme)
    x = frame["season"].astype(str)
    last = len(frame) - 1
    labels = [f"{v:.1f}%" if i in (0, last) else "" for i, v in enumerate(frame["share_pct"])]
    places = ["top right" if i == 0 else "top left" for i in range(len(frame))]  # inside the plot at both ends
    fig = go.Figure(go.Scatter(
        x=x, y=frame["share_pct"], mode="lines+markers+text", text=labels, textposition=places,
        line={"color": c["snpp"], "width": 2}, marker={"size": 9, "line": {"color": c["surface"], "width": 2}},
        error_y={"type": "data", "symmetric": False, "array": frame["high_pct"] - frame["share_pct"],
                 "arrayminus": frame["share_pct"] - frame["low_pct"], "color": c["snpp"], "thickness": 1},
        customdata=frame[["night", "total", "low_pct", "high_pct"]],
        hovertemplate="%{x}: %{y:.2f}%<br>%{customdata[0]:,} of %{customdata[1]:,} detections<br>"
                      "95% interval %{customdata[2]:.2f}% to %{customdata[3]:.2f}%<extra></extra>"))
    fig.update_xaxes(type="category")
    fig.update_yaxes(title_text="night share (%)", rangemode="tozero")
    return _layout(fig, 340, showlegend=False)


def indexed_figure(frame: pd.DataFrame, theme: str | None = None) -> go.Figure:
    """Detections and burned area, each indexed to the base season = 100, on one axis."""
    c = _colours(theme)
    fig = go.Figure()
    for name, key in ((SNPP_SERIES, "snpp"), (BURNED_SERIES, "burned")):
        part = frame[frame["series"] == name].sort_values("season")
        if part.empty:
            continue
        labels = [""] * (len(part) - 1) + [f"{part['index'].iloc[-1]:.0f}"]
        fig.add_trace(go.Scatter(
            x=part["season"].astype(str), y=part["index"], name=name, mode="lines+markers+text", text=labels,
            textposition="top center", line={"color": c[key], "width": 2},
            marker={"size": 9, "line": {"color": c["surface"], "width": 2}}, customdata=part["value"],
            hovertemplate=f"{name}: %{{y:.0f}} (%{{customdata:,.0f}})<extra></extra>"))
    fig.update_xaxes(type="category")
    fig.update_yaxes(title_text=f"index ({BASE} = 100)", rangemode="tozero")
    return _layout(fig, 340, hovermode="x unified")


def official_figure(frame: pd.DataFrame, theme: str | None = None) -> go.Figure:
    """Official counts beside S-NPP detections: the same unit, so one axis."""
    c = _colours(theme)
    fig = go.Figure()
    for column, name, key in (("snpp", SNPP_SERIES, "snpp"), ("official", "official counts", "official")):
        fig.add_trace(go.Bar(x=frame["season"].astype(str), y=frame[column], name=name,
                             marker={"color": c[key], "line": {"color": c["surface"], "width": 1}},
                             hovertemplate=f"{name}, %{{x}}: %{{y:,}}<extra></extra>"))
    fig.update_xaxes(type="category")
    fig.update_yaxes(title_text="fires counted")
    return _layout(fig, 340, barmode="group", bargap=0.4)
