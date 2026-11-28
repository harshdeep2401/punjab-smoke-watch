"""2026 forward test inputs: near-real-time FIRMS detections and archive district weather up to an end date.

Standard (archive) FIRMS products stop at 2026-08-31, so 2026 comes from the NRT products. Every cache file name carries
the end date, so a partial season fetched now is never read by a later run that wants more days.
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from psw import config
from psw.audit import sensors
from psw.ingest import firms, weather

SEASON = 2026


def fetch_nrt(sensor: str, end: date, key: str | None = None, season: int = SEASON) -> pd.DataFrame:
    """Raw NRT rows for `sensor` (snpp, noaa20, ...) from the season start to `end`, in 5-day chunks.

    The last chunk is shortened to stop at `end`; its cache name (start_days) differs from a full chunk's.
    """
    key = config.FIRMS_MAP_KEY if key is None else key
    if not key:
        raise RuntimeError("FIRMS_MAP_KEY is not set in .env")
    source = f"{sensors.SENSORS[sensor]}_NRT"
    d, parts = date(season, *config.SEASON_START), []
    while d <= end:
        n = min(firms.DAY_RANGE, (end - d).days + 1)
        chunk = firms.fetch_chunk(source, d, n, key)
        if not chunk.empty:
            parts.append(chunk.assign(source=source))
        d += timedelta(days=n)
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True).drop_duplicates(subset=firms.DEDUPE)
    df = df[pd.to_datetime(df["acq_date"]).dt.date <= end]  # FIRMS counts the range forward; never past `end`
    return df.reset_index(drop=True)


def load_detections(ends: dict[str, date], key: str | None = None, geo_path=None) -> pd.DataFrame:
    """Normalised, district-assigned 2026 detections, each sensor up to its own end date ({sensor: end})."""
    frames = []
    for s, end in ends.items():
        raw = fetch_nrt(s, end, key)
        if raw.empty:
            raise RuntimeError(f"FIRMS returned no {s} NRT detections for {SEASON} up to {end}")
        frames.append(sensors.normalise(raw, s, SEASON))
    from psw.processing.districts import assign_districts
    return assign_districts(pd.concat(frames, ignore_index=True), geo_path or config.GEO / "punjab_districts.geojson")[
        sensors.COLUMNS]


def last_weather_day(today: date | None = None) -> date:
    """The latest day the weather archive is expected to cover."""
    return (today or date.today()) - timedelta(days=weather.ARCHIVE_LAG_DAYS)


def district_weather(end: date, geo_path=None, season: int = SEASON) -> pd.DataFrame:
    """2026 district weather from the archive, from two weeks before the season to `end` (cached per end date)."""
    start = date(season, *weather.DISTRICT_START)
    parts = [weather.fetch_district_weather(d, lat, lon, start, end)
             for d, (lat, lon) in weather.district_centroids(geo_path).items()]
    return pd.concat(parts, ignore_index=True).sort_values(["district", "date"]).reset_index(drop=True)
