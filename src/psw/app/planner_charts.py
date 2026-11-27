"""The allocation chart of the Response Planner page. Pure: the allocation table in, a Plotly figure out.

The plan is the point and the split without the adjustment is context, so one is coloured and the other grey.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from psw.app.audit_charts import COLOURS

CONTEXT = {"light": "#b2b2b2", "dark": "#6b6b6b"}
LABELLED = 5  # direct labels on the largest shares only; the table has every figure


def allocation_figure(table: pd.DataFrame, use_likely: bool, theme: str | None = None) -> go.Figure:
    """Horizontal bars, most units at the top. `table` is planner_view.allocation_table."""
    mode = "dark" if theme == "dark" else "light"
    labels = [str(u) if i < LABELLED and u > 0 else "" for i, u in enumerate(table["units"])]
    name = "planned on likely burning" if use_likely else "planned on forecast detections"
    fig = go.Figure(go.Bar(y=table["district"], x=table["units"], orientation="h", name=name, text=labels,
                           textposition="outside", cliponaxis=False, marker={"color": COLOURS[mode]["snpp"]},
                           hovertemplate="%{y}: %{x} units<extra>" + name + "</extra>"))
    if "on detections only" in table:
        fig.add_trace(go.Bar(y=table["district"], x=table["on detections only"], orientation="h",
                             name="on forecast detections only", marker={"color": CONTEXT[mode]},
                             hovertemplate="%{y}: %{x} units<extra>on forecast detections only</extra>"))
    fig.update_yaxes(autorange="reversed", dtick=1)
    fig.update_xaxes(title_text="units")
    fig.update_layout(barmode="group", bargap=0.25, barcornerradius=3, height=26 * len(table) + 110,
                      margin={"l": 10, "r": 30, "t": 30, "b": 10},
                      legend={"orientation": "h", "y": 1.0, "yanchor": "bottom", "x": 0, "traceorder": "normal"},
                      showlegend=True)
    return fig
