"""Shared helpers for the Streamlit app: cached data loaders and the limitations text."""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from psw import config

LIMITATIONS = """
- Satellites detect **heat anomalies**, not "stubble burning" specifically, and miss fires outside overpass
  times or under cloud. FIRMS dates are UTC; we treat `acq_date` as the local day.
- PM2.5 is **modelled** (CAMS), not measured at street level.
- This is a student project, not a regulatory or health tool. No individual farms or farmers are identified.
- Forecast accuracy is limited by the training window we can actually obtain (2022 to 2025).
- Districts: 22 districts. **Malerkotla** (created in 2021) is counted inside Sangrur.
"""

CREDITS = (
    "Data: NASA FIRMS (VIIRS S-NPP); weather and air-quality data by [Open-Meteo.com](https://open-meteo.com/) "
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
