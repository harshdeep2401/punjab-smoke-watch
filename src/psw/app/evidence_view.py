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


# ---------- forward test on 2026 (docs/FORWARD_TEST_PLAN.md) ----------

FORWARD_FILE = Path("forward_2026") / "forward_test.json"
ARM_LABEL = {"snpp": "S-NPP", "noaa20": "NOAA-20 stand-in"}


def load_forward(root: Path | None = None) -> dict | None:
    try:
        return json.loads((Path(root or config.PROCESSED) / FORWARD_FILE).read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _short(day: str) -> str:
    return pd.Timestamp(day).strftime("%d %b").lstrip("0")


def arm_column(report: dict, arm: str) -> str:
    e = report["arms"][arm]
    return f"2026, {ARM_LABEL[arm]} ({_short(e['first_issue'])} to {_short(e['last_issue'])})"


def forward_frame(report: dict, risk_metrics: dict | None, decision: dict | None) -> pd.DataFrame:
    """The 2026 arms beside the backtest seasons, one row per measure, values as text (mixed units)."""
    chosen = report["chosen_variant"]
    measures = [  # (label, how to read one column's numbers)
        ("Top-5 hit rate, model", lambda v, a: v[chosen]["top5_hit"] * 100, "{:.1f}%"),
        ("Top-5 hit rate, persistence", lambda v, a: v["persistence"]["top5_hit"] * 100, "{:.1f}%"),
        ("Spearman, model", lambda v, a: v[chosen]["spearman"], "{:.2f}"),
        ("80% band coverage, model", lambda v, a: v[chosen]["coverage80"] * 100, "{:.1f}%"),
        ("Allocation, model (% of oracle)", lambda v, a: a[chosen], "{:.1f}"),
        ("Allocation, last season's split (% of oracle)", lambda v, a: a["last_season_share"], "{:.1f}"),
    ]
    columns = {}
    if risk_metrics and decision:
        for season in sorted(risk_metrics["variants"][chosen]["by_season"]):
            v = {name: s["by_season"][season] for name, s in risk_metrics["variants"].items()}
            columns[f"backtest {season}"] = (v, decision["by_season"][season])
    for arm, e in report["arms"].items():
        columns[arm_column(report, arm)] = (e["variants"], e["allocation_pct_of_oracle"])
    rows = []
    for label, get, fmt in measures:
        row = {"measure": label}
        for name, (v, a) in columns.items():
            try:
                value = get(v, a)
                row[name] = "-" if value is None else fmt.format(value)
            except (KeyError, TypeError):
                row[name] = "-"
        rows.append(row)
    return pd.DataFrame(rows)


def criteria_lines(report: dict, arm: str) -> list[str]:
    out = []
    model = report["arms"][arm]["variants"][report["chosen_variant"]]
    for c in report["arms"][arm]["criteria"]:
        mark = "✅ met" if c["met"] else "❌ not met" if c["met"] is False else "– not scored"
        value = c["value"]
        shown = ("-" if value is None
                 else f"top-5 {model['top5_hit']:.1%}, Spearman {model['spearman']:.2f}" if c["name"] == "ranking holds"
                 else f"{value:+.1f} points" if "allocation" in c["name"]
                 else f"{value * 100:+.1f} points" if "persistence" in c["name"]
                 else f"{value:.1%}" if isinstance(value, float) else str(value))
        out.append(f"{mark}: **{c['name']}** ({shown}; rule: {c['rule']})")
    return out


def forward_summary(report: dict) -> list[str]:
    """The forward test in words, computed from the report."""
    a, b = report["arms"]["snpp"], report["arms"]["noaa20"]
    chosen = report["chosen_variant"]
    met = lambda e: sum(bool(c["met"]) for c in e["criteria"])  # noqa: E731
    lead = lambda e: e["allocation_lead"]  # noqa: E731
    out = [f"The method was frozen (code as of `{report['frozen_code_commit']}`), trained on "
           f"{report['trained_on'][0]} to {report['trained_on'][-1]} and run on 2026, a season never used for any "
           f"choice. The pass marks were written down before the run. The S-NPP arm met {met(a)} of "
           f"{len(a['criteria'])}, the NOAA-20 arm {met(b)} of {len(b['criteria'])}."]
    out.append(f"**What held: the allocation.** Placing units on the {chosen} forecast beat the split by last "
               f"season's fires in both arms: {lead(a)['points']:+.1f} points (95% interval {lead(a)['low']:+.1f} to "
               f"{lead(a)['high']:+.1f}) and {lead(b)['points']:+.1f} ({lead(b)['low']:+.1f} to {lead(b)['high']:+.1f}).")
    ra, rb = a.get("ranking_lead"), b.get("ranking_lead")
    if ra and rb:
        out.append(f"**What did not clearly hold: beating persistence at ranking.** The top-5 lead over "
                   f"{ra['against']} was {ra['top5']['points']:+.1f} points ({ra['top5']['low']:+.1f} to "
                   f"{ra['top5']['high']:+.1f}) on S-NPP and {rb['top5']['points']:+.1f} ({rb['top5']['low']:+.1f} to "
                   f"{rb['top5']['high']:+.1f}) on NOAA-20: within chance either way, as in the backtest.")
    same = a.get("post_hoc_backtest_same_window", {})
    if same:
        rho = a["variants"][chosen]["spearman"]
        ref = ", ".join(f"{s}: {v[chosen]['spearman']:.2f}" for s, v in sorted(same.items()) if chosen in v)
        weaker = all(rho < v[chosen]["spearman"] for v in same.values() if chosen in v)
        out.append(f"The S-NPP arm covers only the early season, when ranking is hardest. On the same dates the "
                   f"backtest's Spearman was {ref}; 2026's {rho:.2f} is "
                   + ("lower than every backtest season, so the ranking really was weaker in 2026. "
                      if weaker else "within that range. ")
                   + "(This comparison was added after seeing the result.)")
    return out


def detection_note(report: dict) -> str:
    """2026 detections against earlier seasons on the same dates. Context only."""
    parts = []
    for arm, e in report["arms"].items():
        by = e["detections_same_dates_by_season"]
        prev = by.get(str(report["season"] - 1))
        parts.append(f"{ARM_LABEL[arm].split()[0]} to {_short(e['data_end'])}: {by[str(report['season'])]:,} in "
                     f"{report['season']}" + (f" against {prev:,} in {report['season'] - 1}" if prev else ""))
    return "Detections in Punjab by the same date: " + "; ".join(parts) + "."
