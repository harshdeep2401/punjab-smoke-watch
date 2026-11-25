"""Open-Meteo air-quality (CAMS) hourly PM2.5, cached as JSON per city and season."""
from __future__ import annotations

import json
from datetime import date

import pandas as pd

from psw import config
from psw.ingest import http

URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
MIN_HOURS = 18


def season_fetch_window(year: int) -> tuple[date, date]:
    """Sep 1 (history for lags) to Dec 1 (so the target exists for Nov 30)."""
    return date(year, 9, 1), date(year, 12, 1)


def fetch_city_pm25(city: str, lat: float, lon: float, start: date, end: date) -> pd.DataFrame:
    """Hourly PM2.5 (ug/m3), columns: time, pm2_5. Cached per request."""
    path = config.RAW / "aq" / f"{city}_{start}_{end}.json"
    if path.exists():
        payload = json.loads(path.read_text())
    else:
        r = http.get(URL, {"latitude": lat, "longitude": lon, "hourly": "pm2_5", "start_date": str(start),
                           "end_date": str(end), "timezone": "Asia/Kolkata"})
        if r.status_code != 200:
            raise RuntimeError(f"Open-Meteo AQ {city}: HTTP {r.status_code}: {r.text[:100]!r}")
        payload = r.json()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))
    h = payload["hourly"]
    df = pd.DataFrame({"time": pd.to_datetime(h["time"]), "pm2_5": pd.to_numeric(h["pm2_5"])})
    return df


def daily_from_hourly(df: pd.DataFrame, city: str) -> pd.DataFrame:
    df = df.assign(date=df["time"].dt.date)
    g = df.groupby("date")["pm2_5"]
    out = pd.DataFrame({"pm25_mean": g.mean().where(g.count() >= MIN_HOURS)}).reset_index()
    out["city"] = city
    return out[["date", "city", "pm25_mean"]]


def daily_pm25(seasons: list[int]) -> pd.DataFrame:
    """Columns: date, city, pm25_mean (NaN when fewer than 18 hourly values)."""
    parts = []
    for year in seasons:
        start, end = season_fetch_window(year)
        for city, (lat, lon) in config.CITIES.items():
            parts.append(daily_from_hourly(fetch_city_pm25(city, lat, lon, start, end), city))
    return pd.concat(parts, ignore_index=True).sort_values(["city", "date"]).reset_index(drop=True)
