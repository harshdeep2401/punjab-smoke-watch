"""Harvest-progress signals for the fire-risk model (PLAN Phase 11). Pure functions, no I/O.

Mandi paddy arrivals were a no-go (docs/feasibility.md), so these are proxies built from detections and rain.
A value for day t uses this season's data up to t and last season's data, with one labelled exception:
`rain_next3_mm` (see RAIN_ASSUMPTION).
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from psw import config

FIRST_FIRE_MIN = 3  # detections in one day that count as the district's burning having started
NOT_STARTED = -1  # days_since_first_fire before that day
RAIN_DAYS = 3

PROXY_FEATURES = ["season_progress", "days_since_first_fire", "last_season_curve_t"]
RAIN_FEATURES = ["rain_last3_mm", "rain_next3_mm"]
FORECAST_STANDIN = ("rain_next3_mm",)  # the only features that look past day t
RAIN_ASSUMPTION = ("Rain over the next 3 days is taken from the weather archive, standing in for a forecast. A real "
                   "forecast is less accurate, so backtest results using it are optimistic.")


def season_matrix(daily: pd.DataFrame, season: int) -> pd.DataFrame:
    """One season's fire_count as day-of-season (Sep 15 = 0) x district. Raises if a district-day is missing."""
    d = daily[daily["season"] == season]
    start = pd.Timestamp(date(season, *config.SEASON_START))
    day = (pd.to_datetime(d["date"]) - start).dt.days
    m = d.assign(day=day.to_numpy()).pivot(index="day", columns="district", values="fire_count").sort_index()
    if m.empty or list(m.index) != list(range(len(m))) or m.isna().any().any():
        raise ValueError(f"daily counts for {season} are not one row per district per day from the season start")
    return m


def previous_matrix(daily: pd.DataFrame, season: int) -> pd.DataFrame | None:
    """Last calendar year's matrix, or None if the data has no such season."""
    return season_matrix(daily, season - 1) if (daily["season"] == season - 1).any() else None


def _long(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Day x district matrices (same shape) -> columns day, district and one per matrix."""
    return pd.concat({name: m.stack() for name, m in frames.items()}, axis=1).rename_axis(["day", "district"]).reset_index()


def proxy_features(daily: pd.DataFrame) -> pd.DataFrame:
    """date, season, district and PROXY_FEATURES, from `daily` (date, season, district, fire_count).

    season_progress: detections so far this season / the district's total last season (can exceed 1).
    days_since_first_fire: days since the first day with at least FIRST_FIRE_MIN detections, else NOT_STARTED.
    last_season_curve_t: the share of last season's total that had been detected by the same day of season.
    The two that need last season are NaN without one, or when the district had no detections in it.
    """
    out = []
    for season in sorted(int(s) for s in daily["season"].unique()):
        m = season_matrix(daily, season)
        prev = previous_matrix(daily, season)
        if prev is None:
            progress = curve = pd.DataFrame(np.nan, index=m.index, columns=m.columns)
        else:
            total = prev.sum().where(lambda t: t > 0)
            progress = m.cumsum() / total
            curve = (prev.cumsum() / total).reindex(index=m.index, columns=m.columns, method=None).ffill()
        started = (m >= FIRST_FIRE_MIN).cummax()
        frame = _long({"season_progress": progress, "days_since_first_fire": started.cumsum() + NOT_STARTED,
                       "last_season_curve_t": curve})
        start = pd.Timestamp(date(season, *config.SEASON_START))
        frame.insert(0, "season", season)
        frame.insert(0, "date", (start + pd.to_timedelta(frame.pop("day"), unit="D")).dt.date)
        out.append(frame)
    cols = ["date", "season", "district", *PROXY_FEATURES]
    return pd.concat(out, ignore_index=True)[cols].sort_values(["district", "date"]).reset_index(drop=True)


def rain_features(weather: pd.DataFrame) -> pd.DataFrame:
    """date, district, rain_last3_mm (t-2..t) and rain_next3_mm (t+1..t+3), from daily `precip_mm`.

    NaN where the window is not complete within the same year.
    """
    w = weather[["date", "district", "precip_mm"]].sort_values(["district", "date"]).reset_index(drop=True)
    keys = [w["district"], pd.to_datetime(w["date"]).dt.year]
    g = w.groupby(keys)["precip_mm"]
    w["rain_last3_mm"] = g.transform(lambda s: s.rolling(RAIN_DAYS, min_periods=RAIN_DAYS).sum())
    w["rain_next3_mm"] = g.transform(
        lambda s: s[::-1].rolling(RAIN_DAYS, min_periods=RAIN_DAYS).sum()[::-1].shift(-1))
    return w[["date", "district", *RAIN_FEATURES]]
