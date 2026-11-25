"""Central configuration: paths, bounding box, cities, season window, env vars."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

DATA = ROOT / "data"
RAW = DATA / "raw"
GEO = DATA / "geo"
PROCESSED = DATA / "processed"
KB = ROOT / "kb"

# (min_lon, min_lat, max_lon, max_lat) around Punjab with a margin for upwind fires
BBOX = (72.5, 28.0, 78.0, 32.8)

# Decision D2. (lat, lon)
CITIES: dict[str, tuple[float, float]] = {
    "Ludhiana": (30.90, 75.85),
    "Amritsar": (31.63, 74.87),
    "Patiala": (30.34, 76.39),
    "Bathinda": (30.21, 74.95),
}

SEASON_START = (9, 15)  # Sep 15
SEASON_END = (11, 30)  # Nov 30

SEASONS: list[int] = [2022, 2023, 2024, 2025]  # 2026 has FIRMS data only to ~Nov 2; see docs/feasibility.md

FIRMS_MAP_KEY = os.getenv("FIRMS_MAP_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "")
