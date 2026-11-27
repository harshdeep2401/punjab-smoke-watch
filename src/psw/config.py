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

# ---------- Phase 5: LLM, Sentinel, Verifier ----------
LLM_MIN_INTERVAL_S = float(os.getenv("LLM_MIN_INTERVAL_S", "4"))  # free tiers are tight

NAAQS_PM25_24H = 60  # India NAAQS, 24 h, ug/m3

# Sentinel thresholds (PLAN 5.3)
SENTINEL_SPIKE_MIN = 15  # "spike": fire_count >= max(SPIKE_MIN, SPIKE_RATIO * 7d mean)
SENTINEL_SPIKE_RATIO = 2.0
SENTINEL_HIGH_COUNT_TOP = 3  # "high_count": top-3 in Punjab and at least HIGH_COUNT_MIN detections
SENTINEL_HIGH_COUNT_MIN = 30
SENTINEL_SMOG_P90 = 120  # "smog_risk": nearest-city p90 forecast above this (ug/m3)

# Verifier (PLAN 5.5)
BRIEFING_WORDS = (100, 180)
HEADLINE_MAX_WORDS = 12
# Phase 10: audit briefs
AUDIT_WORDS = (150, 200)
AUDIT_LABELS = ("What detections say", "What other signals say", "Our best range and how sure we are", "What would settle it")
AUDIT_BANNED_PATTERNS = [
    (r"\bhigh confidence\b|\bconfidence\s*(is|of|:)\s*high\b", "an audit is never high confidence"),
    (r"\b(fall|fell|drop|dropped|declined?|decreased?|reduction|rise|rose|increased?)\s+(of\s+|by\s+)?[-−]\s?\d",
     "write a change without the minus sign"),
    (r"\bnever (call|say|claim|present|cite) (it|the|that|a)\b|\bdo not copy\b", "an instruction copied into the brief"),
    (r"\[(?!K\d{3}\])[^\]\n]{0,20}\]", "a bracketed marker that is not a citation"),
]
# numbers that may appear without being in a tool result: "7-day", "80% interval", p10/p50/p90, "24-hour"
VERIFY_CONSTANTS = (7, 80, 10, 50, 90, 24)
BANNED_PATTERNS = [  # (regex, why). Matched case-insensitively except the name rule, which needs a capital letter.
    (r"\bwear(ing)?\s+(a\s+|an\s+)?(face\s+)?mask", "health advice"),
    (r"\bstay(ing)?\s+(indoors|inside)\b", "health advice"),
    (r"\bshould\s+stay\s+(indoors|inside|home)\b", "health advice"),
    (r"\bN95\b", "health advice"),
    (r"\b(see|consult)\s+a\s+(doctor|physician)\b", "health advice"),
]
BANNED_NAME_PATTERN = r"\b(?:[Ff]armer|Mr|Mrs|Ms|Smt|Shri|Sardar)\.?\s+[A-Z][a-z]+"  # personal names (case-sensitive)
