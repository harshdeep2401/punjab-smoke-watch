"""Shared helpers for the Streamlit app: cached data loaders and the limitations text."""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from psw import config

LLM_RULE = ("**The language model never does the maths.** Every number comes from Python, and a Verifier that is not a "
            "language model checks it before a brief is shown.")

LIMITATIONS = """
- Satellites detect **heat anomalies**, not "stubble burning" specifically, and miss fires outside overpass
  times or under cloud. FIRMS dates are UTC; we treat `acq_date` as the local day.
- PM2.5 is **modelled** (CAMS), not measured at street level.
- This is a student project, not a regulatory or health tool. No individual farms or farmers are identified.
- Forecast accuracy is limited by the training window we can actually obtain (2022 to 2025).
- Districts: 22 districts. **Malerkotla** (created in 2021) is counted inside Sangrur.
- The audit **triangulates** satellite and official signals. There is no field verification, and the error of the
  burned-area product is unknown.
- The response planner is **support, not enforcement**. What a machine or team achieves is an assumption, "likely
  burning" is not a count of fires, and nothing finer than a 5 km cell is shown.
"""

CREDITS = (
    "Data: NASA FIRMS (VIIRS S-NPP, NOAA-20, NOAA-21; MODIS); MODIS MCD64A1 burned area (University of Maryland); "
    "official fire counts and machine figures from Punjab Pollution Control Board and state reports; weather and air-quality data by [Open-Meteo.com](https://open-meteo.com/) "
    "(ERA5 weather, CAMS air quality), licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); "
    "district boundaries from geoBoundaries (ODbL)."
)


@st.cache_data(show_spinner=False)
def load_district_daily() -> pd.DataFrame:
    return pd.read_parquet(config.PROCESSED / "district_daily.parquet")


@st.cache_data(show_spinner=False)
def load_fires() -> pd.DataFrame:
    cols = ["latitude", "longitude", "acq_date", "acq_time", "frp", "district", "season"]
    return pd.read_parquet(config.PROCESSED / "fires.parquet", columns=cols)


@st.cache_data(show_spinner=False)
def load_geojson() -> dict:
    return json.loads((config.GEO / "punjab_districts.geojson").read_text())


@st.cache_data(show_spinner=False)
def load_pm25(root: str) -> pd.DataFrame | None:
    """Daily CAMS PM2.5 per city, or None when the file is missing. `root` is the data folder (part of the cache key)."""
    try:
        return pd.read_parquet(f"{root}/pm25_daily.parquet")
    except (OSError, ValueError):
        return None


# ---------- Phase 6 loaders ----------

@st.cache_resource(show_spinner=False)
def get_store():
    """One DataStore per server process (it caches the parquet tables itself)."""
    from psw.agents.tools import DataStore
    return DataStore()


@st.cache_resource(show_spinner="Loading the knowledge base…")
def get_kb():
    from psw.rag.index import KnowledgeBase
    return KnowledgeBase()


@st.cache_data(show_spinner=False)
def load_forecasts() -> pd.DataFrame:
    return pd.read_parquet(config.PROCESSED / "forecasts.parquet")


@st.cache_data(show_spinner=False)
def load_metrics() -> dict:
    return json.loads((config.PROCESSED / "metrics.json").read_text())


@st.cache_data(show_spinner=False)
def scan_watchlist(date: str) -> list[dict]:
    """Sentinel for one date (deterministic, offline, fast). Raises ValueError when there is no data for the date."""
    from psw.agents import sentinel
    return sentinel.scan(date, get_store())
