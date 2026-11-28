"""Forward test of the 3-day fire-risk forecast and the allocation on the 2026 season (docs/FORWARD_TEST_PLAN.md).

Pure functions, no I/O. The method is frozen: the same `risk.build_table` and `risk.run_backtest` as the backtest,
trained on every season before 2026, and the same allocation scoring. Only the 2026 inputs are new.

Two arms, because S-NPP near-real-time data stops on 2026-11-02:
  snpp    the sensor the model was trained on, up to its last full day (primary; ends before the usual peak)
  noaa20  NOAA-20 counts times a fixed S-NPP/NOAA-20 ratio from 2022-2025, up to the data cutoff (secondary)
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from psw.models.evaluate import score
from psw.processing.districts import district_daily
from psw.response import allocate, intervals, risk

SEASON = 2026
ARMS = {
    "snpp": "S-NPP, the sensor the model was trained on (primary; ends before the usual peak)",
    "noaa20": "NOAA-20 scaled to S-NPP, the full season to date (secondary; the input sensor is swapped)",
}
# Fixed in docs/FORWARD_TEST_PLAN.md before any 2026 result was computed. Backtest reference for `model`:
# top-5 hit 69.4% (2024) and 57.6% (2025), Spearman 0.772 and 0.687.
MIN_TOP5 = 0.576 - 0.05
MIN_SPEARMAN = 0.687 - 0.05
COVERAGE_OK = (0.70, 0.90)
CHOSEN = "model"  # the planner's variant, picked on 2024 and 2025


# ---------- inputs ----------

def snpp_ratio(multisensor: pd.DataFrame, before: int = SEASON) -> float:
    """S-NPP / NOAA-20 detections inside Punjab, pooled over the seasons before `before` that have both."""
    d = multisensor[multisensor["district"].notna() & (multisensor["season"] < before)]
    counts = d.groupby(["season", "sensor"]).size().unstack()
    both = counts.dropna(subset=["snpp", "noaa20"])
    if both.empty:
        raise ValueError("no season with both S-NPP and NOAA-20 detections")
    return float(both["snpp"].sum() / both["noaa20"].sum())


def counts(detections: pd.DataFrame, sensor: str, end: date, names: list[str], scale: float = 1.0,
           season: int = SEASON) -> pd.DataFrame:
    """date, season, district, fire_count for one sensor, every district-day from the season start to `end`.

    Days after `end` are left out, never filled with zeros: a missing day is not a day without fires.
    """
    rows = detections[(detections["sensor"] == sensor) & (detections["season"] == season)]
    rows = rows[pd.to_datetime(rows["acq_date"]).dt.date <= end]
    daily = district_daily(rows, names, [season])
    daily = daily[pd.to_datetime(daily["date"]).dt.date <= end].copy()
    daily["fire_count"] = np.round(daily["fire_count"] * scale).astype(int)
    return daily[["date", "season", "district", "fire_count"]].reset_index(drop=True)


def pad_weather(weather: pd.DataFrame, days: int = risk.HORIZON) -> pd.DataFrame:
    """Repeat each district's last weather row for `days` more days.

    `risk.build_table` needs weather 3 days past every issue day, including the last 3 days of data, whose 3-day
    target is incomplete and which it then drops. The padded rows only ever reach those dropped rows
    (checked by `build`).
    """
    last = weather.sort_values("date").groupby("district").tail(1)
    pads = [last.assign(date=pd.to_datetime(last["date"]).dt.date + timedelta(days=i)) for i in range(1, days + 1)]
    return pd.concat([weather, *pads], ignore_index=True)


def build(history: pd.DataFrame, current: pd.DataFrame, weather_history: pd.DataFrame,
          weather_current: pd.DataFrame) -> pd.DataFrame:
    """The backtest's feature table over the history and 2026. Raises if a kept 2026 row used padded weather."""
    weather_end = pd.to_datetime(weather_current["date"]).dt.date.max()
    data_end = pd.to_datetime(current["date"]).dt.date.max()
    if data_end > weather_end:
        raise ValueError(f"fire data runs to {data_end} but weather only to {weather_end}")
    table = risk.build_table(pd.concat([history, current], ignore_index=True),
                             pd.concat([weather_history, pad_weather(weather_current)], ignore_index=True))
    kept = pd.to_datetime(table.loc[table["season"] == SEASON, "date"]).dt.date
    if kept.empty or kept.max() + timedelta(days=risk.HORIZON) > weather_end:
        raise ValueError("a 2026 issue day would rely on padded weather")
    return table


# ---------- the frozen recipe ----------

def run(table: pd.DataFrame, gap_range: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Train on seasons before 2026, forecast 2026 (all variants), add the likely-burning columns."""
    forecasts, conformal, _ = risk.run_backtest(table, [SEASON])
    return risk.add_likely(forecasts, gap_range), conformal[str(SEASON)]


def evaluate(forecasts: pd.DataFrame, history: pd.DataFrame, runner_up: str) -> dict:
    """The backtest's scores for each variant, the allocation score and the intervals on the leads."""
    variants = {v: score(g) | risk.ranking_metrics(g) for v, g in forecasts.groupby("variant")}
    decision = allocate.decision_backtest(forecasts, history)
    scores = allocate.daily_scores(forecasts, history)
    issue = pd.to_datetime(forecasts["issue_date"]).dt.date
    out = {"issue_days": int(forecasts["issue_date"].nunique()), "first_issue": str(issue.min()),
           "last_issue": str(issue.max()), "variants": variants,
           "allocation_pct_of_oracle": decision["pct_of_oracle"],
           "likely_allocation_pct_of_oracle": decision.get("likely_check", {}).get("pct_of_oracle"),
           "allocation_lead": intervals.allocation_lead(scores, CHOSEN, "last_season_share")}
    if variants[CHOSEN]["n_days"] >= 2:
        out["ranking_lead"] = intervals.ranking_lead(forecasts, CHOSEN, runner_up)
    return out


def criteria(result: dict) -> list[dict]:
    """The four pre-set checks: [{name, met, value, rule}]. `met` is None when there is nothing to score."""
    m, p = result["variants"][CHOSEN], result["variants"]["persistence"]
    alloc = result["allocation_pct_of_oracle"]

    def check(name, value, ok, rule):
        bad = value is None or (isinstance(value, float) and np.isnan(value))
        return {"name": name, "met": None if bad else bool(ok), "value": value, "rule": rule}

    lo, hi = COVERAGE_OK
    return [
        check("ranking holds", m["top5_hit"], m["top5_hit"] >= MIN_TOP5 and m["spearman"] >= MIN_SPEARMAN,
              f"top-5 hit at least {MIN_TOP5:.1%} and Spearman at least {MIN_SPEARMAN:.2f}"),
        check("not worse than persistence", m["top5_hit"] - p["top5_hit"], m["top5_hit"] >= p["top5_hit"],
              "top-5 hit rate at least persistence's"),
        check("allocation beats the naive split", alloc[CHOSEN] - alloc["last_season_share"],
              alloc[CHOSEN] > alloc["last_season_share"], "% of the oracle above the split by last season's fires"),
        check("band is honest", m["coverage80"], lo <= m["coverage80"] <= hi,
              f"80% band coverage between {lo:.0%} and {hi:.0%}"),
    ]


def backtest_window(backtest: pd.DataFrame, first_issue: str, last_issue: str) -> dict:
    """Ranking scores of the backtest seasons over the same calendar window as a 2026 arm. {season: {variant: ...}}.

    Post-hoc context, added after the 2026 result was seen: arm A covers only the early season, when counts are low
    and ranking is harder, so its scores are compared with the backtest on the same dates. Not one of the criteria.
    """
    md = lambda d: (d.month, d.day)  # noqa: E731
    lo, hi = md(date.fromisoformat(first_issue)), md(date.fromisoformat(last_issue))
    days = pd.to_datetime(backtest["issue_date"])
    inside = [lo <= (d.month, d.day) <= hi for d in days]
    out = {}
    for (season, variant), g in backtest[inside].groupby(["season", "variant"]):
        r = risk.ranking_metrics(g)
        out.setdefault(str(int(season)), {})[variant] = {"top5_hit": r["top5_hit"], "spearman": r["spearman"],
                                                         "n_days": r["n_days"]}
    return out


def detections_to_date(multisensor: pd.DataFrame, current: pd.DataFrame, sensor: str, end: date) -> dict:
    """Detections in Punjab from the season start to the same day and month as `end`, per season, 2026 included."""
    d = multisensor[(multisensor["sensor"] == sensor) & multisensor["district"].notna()]
    when = pd.to_datetime(d["acq_date"])
    upto = (when.dt.month < end.month) | ((when.dt.month == end.month) & (when.dt.day <= end.day))
    out = {str(int(s)): int(n) for s, n in d[upto].groupby("season").size().items()}
    return out | {str(SEASON): int(current["fire_count"].sum())}


def on_dates(forecasts: pd.DataFrame, last_issue: str) -> pd.DataFrame:
    """Forecasts issued up to `last_issue`: arm B on arm A's dates, to measure the cost of the sensor swap."""
    return forecasts[pd.to_datetime(forecasts["issue_date"]).dt.date <= date.fromisoformat(last_issue)]
