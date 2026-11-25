"""Data tools for the agents. Pure Python over the processed parquet; each returns a small JSON-serialisable dict.

Floats are rounded to 1 decimal so they can be quoted. The LLM never sees raw tables, only these outputs.
The forecast table's `actual` column is never exposed (it would leak the future).
"""
from __future__ import annotations

import json
from datetime import date as Date
from pathlib import Path

import numpy as np
import pandas as pd

from psw import config
from psw.agents.runtime import Tool
from psw.processing.features import _haversine_bearing, upwind_contributors
from psw.processing.wind import dir_from_uv

COMPASS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
LIMITATIONS_DOC = "limitations"


def compass(deg: float) -> str:
    return COMPASS[int((deg % 360) / 22.5 + 0.5) % 16]


def _iso(d) -> str:
    return d if isinstance(d, str) else Date.isoformat(d)


def _r1(x) -> float | None:
    return None if x is None or pd.isna(x) else round(float(x), 1)


class DataStore:
    """Lazy, cached access to the processed tables. Pass `processed_dir` to point it at test fixtures."""

    def __init__(self, processed_dir: Path | None = None, geo_path: Path | None = None):
        self.dir = Path(processed_dir or config.PROCESSED)
        self.geo_path = Path(geo_path or config.GEO / "punjab_districts.geojson")
        self._cache: dict = {}

    def _get(self, key, loader):
        if key not in self._cache:
            self._cache[key] = loader()
        return self._cache[key]

    def _table(self, name: str, date_cols: list[str]) -> pd.DataFrame:
        def load():
            df = pd.read_parquet(self.dir / f"{name}.parquet")
            for c in date_cols:
                df[c] = df[c].astype(str)
            return df
        return self._get(name, load)

    @property
    def district_daily(self) -> pd.DataFrame:
        return self._table("district_daily", ["date"])

    @property
    def features(self) -> pd.DataFrame:
        return self._table("features", ["date"])

    @property
    def forecasts(self) -> pd.DataFrame:
        return self._table("forecasts", ["issue_date", "target_date"])

    @property
    def metrics(self) -> dict:
        return self._get("metrics", lambda: json.loads((self.dir / "metrics.json").read_text()))

    @property
    def districts(self) -> list[str]:
        return sorted(self.district_daily["district"].unique())

    def fires_on(self, day: str) -> pd.DataFrame:
        def load():
            cols = ["latitude", "longitude", "acq_date", "frp", "district"]
            df = pd.read_parquet(self.dir / "fires.parquet", columns=cols)
            df["acq_date"] = df["acq_date"].astype(str)
            return df
        fires = self._get("fires", load)
        return fires[fires["acq_date"] == day]

    def centroids(self) -> dict[str, tuple[float, float]]:
        """District centroid (lat, lon). Computed in a projected CRS so the centroid is not skewed."""
        def load():
            import geopandas as gpd
            g = gpd.read_file(self.geo_path)
            c = g.to_crs(32643).geometry.centroid.to_crs(4326)
            return {n: (float(p.y), float(p.x)) for n, p in zip(g["district"], c)}
        return self._get("centroids", load)


def nearest_city(district: str, store: DataStore | None = None) -> str:
    """Closest of the 4 monitored cities to the district's centroid. Not an LLM tool."""
    store = store or DataStore()
    cents = store.centroids()
    if district not in cents:
        raise ValueError(f"unknown district '{district}'")
    lat, lon = cents[district]
    dists = {c: float(_haversine_bearing(lat, lon, clat, clon)[0]) for c, (clat, clon) in config.CITIES.items()}
    return min(dists, key=dists.get)


def _day_rank(df_day: pd.DataFrame, col: str = "fire_count") -> pd.Series:
    """1 = highest. Ties share the best rank."""
    return df_day[col].rank(method="min", ascending=False).astype(int)


# ---------- the five tools ----------

def get_district_snapshot(store: DataStore, district: str, date: str) -> dict:
    date = _iso(date)
    day = store.district_daily[store.district_daily["date"] == date]
    row = day[day["district"] == district]
    if row.empty:
        return {"error": f"no fire data for {district} on {date}"}
    row = row.iloc[0]
    count, mean7 = int(row["fire_count"]), float(row["fire_count_7d_mean"])
    ranks = _day_rank(day).set_axis(day["district"])
    hist = store.district_daily[(store.district_daily["district"] == district) & (store.district_daily["date"] <= date)].tail(7)
    return {
        "district": district,
        "date": date,
        "fire_count": count,
        "frp_sum_mw": _r1(row["frp_sum"]),
        "fire_count_7d_mean": _r1(mean7),
        "change_vs_7d_pct": _r1((count - mean7) / mean7 * 100) if mean7 > 0 else None,
        "rank_in_punjab": int(ranks[district]),
        "punjab_total": int(day["fire_count"].sum()),
        "last_7_days": [{"date": d, "count": int(c)} for d, c in zip(hist["date"], hist["fire_count"])],
        "nearest_city": nearest_city(district, store),
        "data_as_of": date,
    }


def get_city_forecast(store: DataStore, city: str, date: str) -> dict:
    date = _iso(date)
    f = store.forecasts
    f = f[(f["city"] == city) & (f["issue_date"] == date)]
    full = f[f["variant"] == "full"]
    if full.empty:
        return {"error": f"no forecast for {city} issued on {date} (forecasts exist only for the test seasons)"}
    r = full.iloc[0]
    pers = f[f["variant"] == "persistence"]
    feat = store.features
    today = feat[(feat["city"] == city) & (feat["date"] == date)]
    p10, p50, p90 = _r1(r["p10"]), _r1(r["p50"]), _r1(r["p90"])
    limit = config.NAAQS_PM25_24H
    hint = f"p50 above {limit}" if p50 > limit else (f"p90 above {limit}" if p90 > limit else f"p90 below {limit}")
    return {
        "city": city,
        "issue_date": date,
        "target_date": str(r["target_date"]),
        "p10": p10, "p50": p50, "p90": p90,
        "pm25_today": _r1(today.iloc[0]["pm25_today"]) if not today.empty else None,
        "persistence_p50": _r1(pers.iloc[0]["p50"]) if not pers.empty else None,
        "band_width": _r1(p90 - p10),
        "exceeds_naaqs_60_prob_hint": hint,
    }


def get_wind_and_upwind(store: DataStore, city: str, date: str) -> dict:
    date = _iso(date)
    feat = store.features
    row = feat[(feat["city"] == city) & (feat["date"] == date)]
    if row.empty:
        return {"error": f"no weather features for {city} on {date}"}
    row = row.iloc[0]
    deg = float(dir_from_uv(row["u_mean"], row["v_mean"]))
    contributors = upwind_contributors(store.fires_on(date), config.CITIES[city], deg)
    return {
        "city": city,
        "date": date,
        "wind_dir_from_deg": _r1(deg),
        "wind_dir_compass": compass(deg),
        "wind_speed_kmh": _r1(row["wind_speed_mean"]),
        "upwind_frp_l0": _r1(row["upwind_frp_l0"]),
        "top_upwind_districts": contributors,
    }


def get_model_reliability(store: DataStore, city: str) -> dict:
    m = store.metrics
    try:
        full = m["variants"]["full"]["by_city"][city]
        no_fire = m["variants"]["no_fire"]["by_city"][city]
        pers = m["variants"]["persistence"]["by_city"][city]
    except KeyError:
        return {"error": f"no reliability metrics for {city}"}
    return {
        "city": city,
        "test_seasons": m["seasons_test"],
        "mae": _r1(full["mae"]),
        "coverage_80_pct": _r1(full["coverage80"] * 100),
        "mae_persistence": _r1(pers["mae"]),
        "mae_no_fire": _r1(no_fire["mae"]),
        "n_forecasts": int(full["n"]),
    }


def search_knowledge(kb, query: str, k: int = 4, before=None, district: str | None = None) -> dict:
    """`before` and `district` are injected by the orchestrator. The model cannot choose them."""
    hits = kb.search(query, k=max(1, min(int(k), 6)), before=before, district=district)
    return {"results": [{"chunk_id": h["chunk_id"], "title": h["title"], "text": h["text"]} for h in hits]}


def limitations_chunk(kb) -> dict:
    """Deterministic (no embedding) retrieval of the limitations note, for the template fallback.

    Same output shape as `search_knowledge`, so it counts as retrieved evidence.
    """
    c = kb.chunks
    c = c[c["doc_id"] == LIMITATIONS_DOC].head(1)
    return {"results": [{"chunk_id": r["chunk_id"], "title": r["title"], "text": r["text"]} for _, r in c.iterrows()]}


# ---------- schemas shown to the LLM ----------

def _fn(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {"name": name, "description": description,
                                             "parameters": {"type": "object", "properties": properties, "required": required}}}


_DATE = {"type": "string", "description": "Briefing date, YYYY-MM-DD."}
TOOL_SCHEMAS: list[dict] = [
    _fn("get_district_snapshot",
        "Fire detections for one Punjab district on one day: count, FRP (MW), 7-day mean, change vs the mean, "
        "rank among districts, Punjab total, the last 7 days, and the nearest monitored city.",
        {"district": {"type": "string"}, "date": _DATE}, ["district", "date"]),
    _fn("get_city_forecast",
        "Next-day PM2.5 forecast (p10/p50/p90, ug/m3) for one monitored city, with today's value and the persistence baseline.",
        {"city": {"type": "string", "enum": list(config.CITIES)}, "date": _DATE}, ["city", "date"]),
    _fn("get_wind_and_upwind",
        "Wind direction (the direction it blows FROM) and speed at a city, the upwind fire pressure, and the districts "
        "that contribute most to it.",
        {"city": {"type": "string", "enum": list(config.CITIES)}, "date": _DATE}, ["city", "date"]),
    _fn("get_model_reliability",
        "How accurate the forecast model was for a city on the held-out test seasons: error, 80% interval coverage, and the baselines.",
        {"city": {"type": "string", "enum": list(config.CITIES)}}, ["city"]),
    _fn("search_knowledge",
        "Search the knowledge base (method notes, limitations, standards, past season summaries, the model card). "
        "Put names and years in the query, for example 'Sangrur 2024 season' rather than 'last season'. "
        "Returns chunks with ids like K012 that you can cite.",
        {"query": {"type": "string"}, "k": {"type": "integer", "description": "Number of chunks, 1 to 6 (default 4)."}}, ["query"]),
]
_SCHEMA = {s["function"]["name"]: s["function"] for s in TOOL_SCHEMAS}


def build_tools(store: DataStore, kb) -> list[Tool]:
    """Bind the five tools to a data store and a knowledge base."""
    fns = {
        "get_district_snapshot": lambda district, date: get_district_snapshot(store, district, date),
        "get_city_forecast": lambda city, date: get_city_forecast(store, city, date),
        "get_wind_and_upwind": lambda city, date: get_wind_and_upwind(store, city, date),
        "get_model_reliability": lambda city: get_model_reliability(store, city),
        "search_knowledge": lambda query, k=4, before=None, district=None: search_knowledge(kb, query, k, before, district),
    }
    return [Tool(n, _SCHEMA[n]["description"], _SCHEMA[n]["parameters"], fns[n]) for n in fns]
