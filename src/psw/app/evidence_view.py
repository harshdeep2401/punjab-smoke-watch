"""Pure helpers for the Part 2 sections of the Evidence page (no Streamlit import).

Every figure and every verdict is computed from risk_metrics.json, decision_backtest.json and the audit tables, so the
wording changes when the results do.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from psw import config
from psw.app import audit_view
from psw.app.audit_charts import COLOURS
from psw.app.helpers import RISK_LABEL
from psw.app.planner_charts import CONTEXT
from psw.audit import gap
from psw.response import risk

STRATEGY_LABEL = {"model": "model forecast", "persistence": "persistence forecast",
                  "climatology": "climatology forecast", "last_season_share": "share of last season's fires (naive)",
                  "equal_split": "equal split"}


def load_decision(root: Path | None = None) -> dict | None:
    try:
        return json.loads((Path(root or config.PROCESSED) / "decision_backtest.json").read_text())
    except (OSError, json.JSONDecodeError):
        return None


# ---------- fire-risk forecast ----------

def risk_frame(m: dict) -> pd.DataFrame:
    """One row per variant: counts error, band, and the two ranking scores."""
    rows = []
    for v, s in m["variants"].items():
        o = s["overall"]
        row = {"variant": RISK_LABEL.get(v, v), "MAE (detections)": o["mae"], "pinball": o["pinball"],
               "80% coverage (%)": o["coverage80"] * 100, "top-5 hit (%)": o["top5_hit"] * 100,
               "Spearman": o["spearman"]}
        for season, b in s.get("by_season", {}).items():
            row[f"top-5 hit {season} (%)"] = b["top5_hit"] * 100
        rows.append(row)
    return pd.DataFrame(rows)


def random_hit(districts: int) -> float:
    """What picking the top districts blindly would score."""
    return min(1.0, risk.TOP_K / districts)


def _verdict(lead: dict, by_season: dict) -> str:
    """How much to believe a lead, from its interval and the intervals of each season."""
    if lead["high"] < 0:
        return "it is behind, by more than the choice of days explains"
    if lead["low"] <= 0:
        return "this is not clearly more than chance"
    failing = [f"{s} ({v['points']:+.1f}, {v['low']:+.1f} to {v['high']:+.1f})" for s, v in by_season.items()
               if v["low"] <= 0]
    if failing:
        return f"**but it does not hold in {' or '.join(failing)}, so treat it as unproven**"
    return "and it holds in each test season (" + "; ".join(
        f"{s}: {v['points']:+.1f}, {v['low']:+.1f} to {v['high']:+.1f}" for s, v in by_season.items()) + ")"


def interpret_ranking(d: dict) -> str | None:
    """Whether the selected variant really ranks districts better than the next best."""
    r = d.get("ranking_lead")
    if not r:
        return None
    top, rho = r["top5"], r["spearman"]
    chosen = RISK_LABEL.get(r["of"], r["of"])
    if r["of"] == "model":
        chosen = f"the {chosen}"
    return (f"Against {RISK_LABEL.get(r['against'], r['against'])}, the next best, {chosen} "
            f"changes the top-5 hit rate by {top['points']:+.1f} points over {r['n_days']} days (95% interval "
            f"{top['low']:+.1f} to {top['high']:+.1f}); {_verdict(top, top['by_season'])}. Spearman differs by "
            f"{rho['difference']:+.2f} ({rho['low']:+.2f} to {rho['high']:+.2f}).")


# ---------- decision backtest ----------

def decision_frame(d: dict) -> pd.DataFrame:
    """Each strategy's score as % of an allocation that knew the next days, pooled and per season, best first."""
    rows = [{"allocation": STRATEGY_LABEL.get(name, name), "key": name, "all days (%)": value,
             **{f"{s} (%)": v[name] for s, v in d["by_season"].items()}}
            for name, value in d["pct_of_oracle"].items()]
    return pd.DataFrame(rows).sort_values("all days (%)", ascending=False).reset_index(drop=True)


def interpret_decision(d: dict) -> list[str]:
    """Does the forecast help decide where units go? In words, from decision_backtest.json."""
    pct = d["pct_of_oracle"]
    lead = d.get("lead_over_last_season_share")
    chosen = lead["of"] if lead else "model"
    out = [f"With {d['units']} units, allocating on the {STRATEGY_LABEL.get(chosen, chosen)} reached "
           f"**{pct[chosen]:.1f}%** of what an allocation that knew the next 3 days would have, over {d['n_days']} issue "
           f"days. Splitting by last season's fires reached {pct['last_season_share']:.1f}% and an equal split "
           f"{pct['equal_split']:.1f}%, so much of the score needs no forecast at all."]
    if lead:
        out.append(f"The lead over the naive split is {lead['points']:+.1f} points (95% interval {lead['low']:+.1f} to "
                   f"{lead['high']:+.1f}); {_verdict(lead, lead['by_season'])}.")
    best = max(pct, key=pct.get)
    if best != chosen:
        out.append(f"**{STRATEGY_LABEL.get(best, best).capitalize()} scored higher** ({pct[best]:.1f}%).")
    if "likely_check" in d:
        out.append(f"The planner's default, allocating on likely burning, scored {d['likely_check']['pct_of_oracle']:.1f}% "
                   "against detections. " + d["likely_check"]["note"].split(". ", 1)[1])
    return out


def decision_figure(d: dict, theme: str | None = None) -> go.Figure:
    """Bars from zero, the allocation in use coloured and the others grey."""
    mode = "dark" if theme == "dark" else "light"
    frame = decision_frame(d)
    chosen = d.get("lead_over_last_season_share", {}).get("of", "model")
    colours = [COLOURS[mode]["snpp"] if k == chosen else CONTEXT[mode] for k in frame["key"]]
    fig = go.Figure(go.Bar(y=frame["allocation"], x=frame["all days (%)"], orientation="h",
                           marker={"color": colours}, text=[f"{v:.1f}" for v in frame["all days (%)"]],
                           textposition="outside", cliponaxis=False,
                           hovertemplate="%{y}: %{x:.2f}%<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title_text="% of an allocation that knew the next 3 days", range=[0, 105])
    fig.update_layout(height=46 * len(frame) + 80, barcornerradius=4, bargap=0.35, showlegend=False,
                      margin={"l": 10, "r": 30, "t": 10, "b": 10})
    return fig


# ---------- audit method ----------

def audit_gap_frame(tables: dict) -> pd.DataFrame:
    """Punjab's gap range per season with what it rests on."""
    rows = []
    for season in (s for s in audit_view.seasons(tables) if s > audit_view.BASE):
        h = audit_view.season_headline(tables, season)
        rows.append({"season": str(season), "S-NPP change (%)": h["snpp_change"],
                     "burned area change (%)": h["burned_change"], "gap range": audit_view.gap_text(h),
                     "confidence": h["confidence"] or "not estimated",
                     "estimates used": len(h["methods"].split("; ")) if h["methods"] else 0})
    return pd.DataFrame(rows)


def audit_method_lines() -> list[str]:
    """The gap method, with the thresholds read from the code that applies them."""
    bases = " or ".join(str(b) for b in gap.BASES)
    return [
        "**What is measured:** the fall in fire counts per burned km² since a base season. It is not the share of "
        "burning that is missed, because the base season's own miss rate is unknown.",
        "**Roles:** S-NPP detections and official counts are the numerator; MODIS burned area is the only reference; "
        "night share and MODIS-against-S-NPP only set the confidence; NOAA-20 and NOAA-21 are a robustness check, "
        "because they pass at about the same time as S-NPP.",
        f"**Range:** one estimate per count series and base season ({bases}); a pair is used only with at least "
        f"{gap.MIN_BURNED_KM2} km² burned in both seasons and a base count of at least {gap.MIN_BASE_COUNT}. Nothing is "
        "clipped.",
        f"**Confidence:** never high. Punjab is medium only if the lowest estimate is at least "
        f"{gap.MEDIUM_MIN_GAP:.0%} and both timing indicators support a shift (the MODIS rule: at least "
        f"{gap.MODIS_SUPPORT_RATIO} times the S-NPP change). Districts are low at most.",
    ]


AUDIT_LIMITS = [
    "The range does not include the error of the burned-area product. That error is unknown and is the largest "
    "uncertainty; the product is 500 m, misses small fields and is low confidence in cropland.",
    gap.EXPLANATIONS + " The timing indicators speak only to the first.",
    "The official counts are satellite-based too, so their agreement with S-NPP is not independent confirmation.",
    "There is no field verification. This is triangulation, not ground truth.",
]
