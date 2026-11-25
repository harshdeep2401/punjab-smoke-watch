"""Sentinel: a deterministic scan of every district for one date. No LLM.

Flags (thresholds in config.py), each with a human-readable reason shown in the app:
  spike       fire_count >= max(SPIKE_MIN, SPIKE_RATIO * 7-day mean)
  high_count  top-3 in Punjab that day and at least HIGH_COUNT_MIN detections
  smog_risk   the nearest city's p90 PM2.5 forecast is above SMOG_P90
Priority = sum of three ranks (fire count, ratio to the 7-day mean, nearest-city p90); lower is more urgent.
"""
from __future__ import annotations

import pandas as pd

from psw import config
from psw.agents.tools import DataStore, nearest_city


def _rank(s: pd.Series) -> pd.Series:
    """1 = highest; ties share the best rank; missing values rank last."""
    return s.rank(method="min", ascending=False, na_option="bottom").astype(int)


def scan(date, store: DataStore | None = None) -> list[dict]:
    """One dict per district, flagged ones first, then by priority score (lower first)."""
    store = store or DataStore()
    day = str(date)
    dd = store.district_daily
    dd = dd[dd["date"] == day].copy()
    if dd.empty:
        raise ValueError(f"no district data for {day}")

    f = store.forecasts
    f = f[(f["variant"] == "full") & (f["issue_date"] == day)].set_index("city")["p90"]
    dd["nearest_city"] = [nearest_city(d, store) for d in dd["district"]]
    dd["city_p90"] = dd["nearest_city"].map(f)  # NaN on days without a forecast (non-test seasons)
    dd["ratio"] = dd["fire_count"] / dd["fire_count_7d_mean"].where(dd["fire_count_7d_mean"] > 0)

    dd["rank_count"] = _rank(dd["fire_count"])
    dd["priority_score"] = dd["rank_count"] + _rank(dd["ratio"]) + _rank(dd["city_p90"])

    rows = []
    for r in dd.itertuples():
        mean7 = float(r.fire_count_7d_mean)
        flags = []
        spike_at = max(config.SENTINEL_SPIKE_MIN, config.SENTINEL_SPIKE_RATIO * mean7)
        if r.fire_count >= spike_at:
            flags.append(f"spike: {r.fire_count} detections vs a 7-day mean of {mean7:.1f} (threshold {spike_at:.1f})")
        if r.rank_count <= config.SENTINEL_HIGH_COUNT_TOP and r.fire_count >= config.SENTINEL_HIGH_COUNT_MIN:
            flags.append(f"high_count: rank {r.rank_count} in Punjab with {r.fire_count} detections "
                         f"(top {config.SENTINEL_HIGH_COUNT_TOP} and at least {config.SENTINEL_HIGH_COUNT_MIN})")
        if pd.notna(r.city_p90) and r.city_p90 > config.SENTINEL_SMOG_P90:
            flags.append(f"smog_risk: p90 PM2.5 forecast for {r.nearest_city} is {r.city_p90:.1f} ug/m3 "
                         f"(above {config.SENTINEL_SMOG_P90})")
        rows.append({
            "district": r.district, "date": day,
            "fire_count": int(r.fire_count), "fire_count_7d_mean": round(mean7, 1),
            "ratio": None if pd.isna(r.ratio) else round(float(r.ratio), 1),
            "nearest_city": r.nearest_city,
            "nearest_city_p90": None if pd.isna(r.city_p90) else round(float(r.city_p90), 1),
            "flags": flags, "priority_score": int(r.priority_score),
        })
    return sorted(rows, key=lambda x: (not x["flags"], x["priority_score"], -x["fire_count"], x["district"]))
