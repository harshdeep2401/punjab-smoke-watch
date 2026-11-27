"""Open-Meteo weather (ERA5 archive; forecast API past_days for the most recent gap)."""
from __future__ import annotations

import json
import re
import time
from datetime import date, timedelta

import geopandas as gpd
import pandas as pd

from psw import config
from psw.ingest import http
from psw.processing.wind import dir_from_uv, uv_from_dir_speed

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
HOURLY = "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,boundary_layer_height"
ARCHIVE_LAG_DAYS = 2  # measured lag was 1 day in Phase 0; keep a margin
# Open-Meteo daily variable -> our column
DISTRICT_DAILY = {"precipitation_sum": "precip_mm", "temperature_2m_mean": "temp_mean",
                  "relative_humidity_2m_mean": "rh_mean", "wind_speed_10m_mean": "wind_speed_mean"}
DISTRICT_START = (9, 1)  # two weeks before the season, for trailing rain
DISTRICT_END = (12, 3)  # three days after it, for "rain over the next 3 days"
SLEEP = 0.2  # between district requests


def _request(url: str, params: dict) -> dict:
    r = http.get(url, params)
    if r.status_code != 200:
        raise RuntimeError(f"Open-Meteo weather: HTTP {r.status_code}: {r.text[:100]!r}")
    return r.json()


def _to_frame(payload: dict) -> pd.DataFrame:
    h = payload["hourly"]
    df = pd.DataFrame({k: h[k] for k in ["time", *HOURLY.split(",")]})
    df["time"] = pd.to_datetime(df["time"])
    return df


def fetch_city_weather(city: str, lat: float, lon: float, start: date, end: date) -> pd.DataFrame:
    """Hourly weather, cached per request. Days newer than the archive lag come from the forecast API."""
    path = config.RAW / "weather" / f"{city}_{start}_{end}.json"
    if path.exists():
        return _to_frame(json.loads(path.read_text()))
    base = {"latitude": lat, "longitude": lon, "hourly": HOURLY, "timezone": "Asia/Kolkata"}
    cutoff = date.today() - timedelta(days=ARCHIVE_LAG_DAYS)
    frames = []
    if start <= cutoff:
        frames.append(_to_frame(_request(ARCHIVE_URL, {**base, "start_date": str(start),
                                                        "end_date": str(min(end, cutoff))})))
    if end > cutoff:
        gap_start = max(start, cutoff + timedelta(days=1))
        past = (date.today() - gap_start).days + 1
        fc = _to_frame(_request(FORECAST_URL, {**base, "past_days": past, "forecast_days": 1}))
        frames.append(fc[(fc["time"].dt.date >= gap_start) & (fc["time"].dt.date <= end)])
    df = pd.concat(frames, ignore_index=True).drop_duplicates("time").sort_values("time")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"hourly": {**{"time": df["time"].dt.strftime("%Y-%m-%dT%H:%M").tolist()},
                                           **{k: df[k].tolist() for k in HOURLY.split(",")}}}))
    return df.reset_index(drop=True)


def daily_from_hourly(df: pd.DataFrame, city: str) -> pd.DataFrame:
    u, v = uv_from_dir_speed(df["wind_direction_10m"].to_numpy(float), df["wind_speed_10m"].to_numpy(float))
    df = df.assign(date=df["time"].dt.date, u=u, v=v)
    g = df.groupby("date")
    out = pd.DataFrame({
        "temp_mean": g["temperature_2m"].mean(),
        "rh_mean": g["relative_humidity_2m"].mean(),
        "wind_speed_mean": g["wind_speed_10m"].mean(),
        "u_mean": g["u"].mean(),
        "v_mean": g["v"].mean(),
        "blh_mean": g["boundary_layer_height"].mean(),
    })
    out["wind_dir_mean"] = dir_from_uv(out["u_mean"].to_numpy(), out["v_mean"].to_numpy())  # vector mean
    out = out.reset_index()
    out["city"] = city
    return out[["date", "city", "temp_mean", "rh_mean", "wind_speed_mean", "wind_dir_mean",
                "u_mean", "v_mean", "blh_mean"]]


def daily_weather(seasons: list[int]) -> pd.DataFrame:
    parts = []
    for year in seasons:
        start, end = date(year, 9, 1), date(year, 12, 1)
        for city, (lat, lon) in config.CITIES.items():
            parts.append(daily_from_hourly(fetch_city_weather(city, lat, lon, start, end), city))
    return pd.concat(parts, ignore_index=True).sort_values(["city", "date"]).reset_index(drop=True)


# ---------- district weather (ERA5 reanalysis at one point per district, not rain gauges) ----------


def district_centroids(geo_path=None) -> dict[str, tuple[float, float]]:
    """district -> (lat, lon) of its centroid, computed in a metric projection (UTM 43N)."""
    g = gpd.read_file(geo_path or config.GEO / "punjab_districts.geojson")
    points = g.to_crs("EPSG:32643").centroid.to_crs("EPSG:4326")
    return {d: (round(p.y, 3), round(p.x, 3)) for d, p in zip(g["district"], points)}


def _district_frame(payload: dict, district: str, start: date, end: date) -> pd.DataFrame:
    """One row per day from `start` to `end`. Raises if a day or a value is missing."""
    daily = payload["daily"]
    df = pd.DataFrame({new: pd.to_numeric(daily[old]) for old, new in DISTRICT_DAILY.items()})
    df.insert(0, "district", district)
    df.insert(0, "date", pd.to_datetime(daily["time"]).date)
    expected = (end - start).days + 1
    if len(df) != expected or df.isna().any().any():
        raise RuntimeError(f"Open-Meteo district weather: {district} {start} to {end} is incomplete "
                           f"({len(df)} of {expected} days, {int(df.isna().any(axis=1).sum())} with gaps)")
    return df


def fetch_district_weather(district: str, lat: float, lon: float, start: date, end: date) -> pd.DataFrame:
    """Daily weather from the archive, cached per request. An incomplete answer is not cached."""
    slug = re.sub(r"[^A-Za-z0-9]+", "-", district).strip("-")
    path = config.RAW / "weather" / f"district_{slug}_{start}_{end}.json"
    if path.exists():
        return _district_frame(json.loads(path.read_text()), district, start, end)
    payload = _request(ARCHIVE_URL, {"latitude": lat, "longitude": lon, "daily": ",".join(DISTRICT_DAILY),
                                     "start_date": str(start), "end_date": str(end), "timezone": "Asia/Kolkata"})
    df = _district_frame(payload, district, start, end)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"daily": payload["daily"]}))
    time.sleep(SLEEP)
    return df


def district_daily_weather(seasons: list[int], geo_path=None) -> pd.DataFrame:
    """date, district, precip_mm, temp_mean, rh_mean, wind_speed_mean (mm, deg C, %, km/h)."""
    centroids = district_centroids(geo_path)
    parts = []
    for year in seasons:
        start, end = date(year, *DISTRICT_START), date(year, *DISTRICT_END)
        for district, (lat, lon) in centroids.items():
            parts.append(fetch_district_weather(district, lat, lon, start, end))
    return pd.concat(parts, ignore_index=True).sort_values(["district", "date"]).reset_index(drop=True)
