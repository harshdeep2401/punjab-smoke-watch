"""Fire detections from several satellites, labelled by the time of day each one saw them.

`acq_date` stays the UTC date, as in fires.parquet. A night pass at 20:00 UTC is 01:30 IST on the *next* day,
so night detections carry the previous day's date.
"""
from __future__ import annotations

import logging

import pandas as pd

from psw import config
from psw.ingest import firms
from psw.processing.districts import assign_districts

log = logging.getLogger(__name__)

SENSORS = {"snpp": "VIIRS_SNPP", "noaa20": "VIIRS_NOAA20", "noaa21": "VIIRS_NOAA21", "modis": "MODIS"}
FIRST_SEASON = {"noaa21": 2023}  # NOAA-21 standard-processing data starts 2023-03-01 (docs/feasibility.md)
MODIS_MIN_CONFIDENCE = 30  # MODIS confidence is 0-100; VIIRS uses l/n/h
PLATFORMS = {"Terra": "T", "Aqua": "A"}
IST_OFFSET_MIN = 330  # UTC+5:30

MORNING, AFTERNOON, NIGHT, OTHER = "morning ~10:30", "afternoon ~13:30", "night ~01:30", "other"
PASS_HOURS = {MORNING: (10, 11), AFTERNOON: (12, 13, 14, 15), NIGHT: (0, 1, 2, 3)}

COLUMNS = ["latitude", "longitude", "acq_date", "acq_time", "ist_hour", "pass_label", "sensor", "platform",
           "frp", "confidence", "daynight", "season", "district"]


def ist_hour(acq_time: pd.Series) -> pd.Series:
    """Hour of day in IST (0-23) from FIRMS `acq_time`, which is HHMM in UTC."""
    t = acq_time.astype(int)
    return ((t // 100) * 60 + t % 100 + IST_OFFSET_MIN) % 1440 // 60


def pass_label(hours: pd.Series) -> pd.Series:
    labels = {h: name for name, hs in PASS_HOURS.items() for h in hs}
    return hours.map(labels).fillna(OTHER)


def filter_confidence(df: pd.DataFrame, sensor: str) -> pd.DataFrame:
    """MODIS: keep confidence >= 30 on its numeric scale. VIIRS: drop 'l', as for fires.parquet."""
    if sensor == "modis":
        keep = pd.to_numeric(df["confidence"], errors="coerce") >= MODIS_MIN_CONFIDENCE
    else:
        keep = df["confidence"].astype(str).str.strip().str.lower() != "l"
    log.info("%s: dropping %d of %d low-confidence detections", sensor, int((~keep).sum()), len(df))
    return df[keep]


def fetch_sensor_season(sensor: str, year: int, key: str) -> pd.DataFrame:
    """Raw FIRMS rows for one sensor and season (cached per chunk; SP, then NRT)."""
    return firms.fetch_season(year, key, product=SENSORS[sensor])


def normalise(raw: pd.DataFrame, sensor: str, year: int) -> pd.DataFrame:
    """Filtered rows with every column of the multi-sensor table except `district`."""
    df = filter_confidence(raw, sensor).copy()
    df["acq_date"] = pd.to_datetime(df["acq_date"]).dt.date
    df["acq_time"] = df["acq_time"].astype(int)
    df["ist_hour"] = ist_hour(df["acq_time"]).astype(int)
    df["pass_label"] = pass_label(df["ist_hour"])
    df["sensor"] = sensor
    df["platform"] = df["satellite"].map(PLATFORMS) if sensor == "modis" else None
    df["confidence"] = df["confidence"].astype(str).str.strip().str.lower()  # one type for both scales
    df["season"] = year
    return df[[c for c in COLUMNS if c != "district"]]


def load_multisensor(seasons: list[int], key: str | None = None, geo_path=None) -> pd.DataFrame:
    """All sensors for the seasons. Raises if a sensor that should have data for a season has none."""
    key = config.FIRMS_MAP_KEY if key is None else key
    geo_path = geo_path or config.GEO / "punjab_districts.geojson"
    frames = []
    for sensor in SENSORS:
        for year in seasons:
            raw = fetch_sensor_season(sensor, year, key)
            if raw.empty:
                if year >= FIRST_SEASON.get(sensor, 0):
                    raise RuntimeError(f"FIRMS returned no {sensor} detections for the {year} season")
                continue
            frames.append(normalise(raw, sensor, year))
    df = pd.concat(frames, ignore_index=True)
    return assign_districts(df, geo_path)[COLUMNS]
