"""Tools for the Auditor. Pure Python over data/processed/audit/*.parquet; each returns a small JSON-serialisable dict.

Percentages and ratios are rounded here so the agent can quote them as they are. No tool returns a raw table, and
none uses a field name as a value, so there is nothing machine-like to copy into a brief.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from psw import config
from psw.agents.runtime import Tool
from psw.audit import gap
from psw.audit.sensors import NIGHT

STATE = gap.STATE
SENSOR_NAMES = {"snpp": "VIIRS S-NPP", "noaa20": "VIIRS NOAA-20", "noaa21": "VIIRS NOAA-21", "modis": "MODIS (Terra and Aqua)"}
BURNED_NAME = "MODIS burned area (MCD64A1), low confidence in cropland"
SIGNAL_NAMES = {**SENSOR_NAMES, "burned_area": "MODIS burned area", "official": "official counts"}


def cutoff(season: int) -> str:
    """Retrieval and audit documents are cut off at Dec 1 of the season being audited."""
    return f"{int(season)}-12-01"


class AuditStore:
    """Lazy, cached access to the audit tables. Pass `processed_dir` to point it at test fixtures."""

    def __init__(self, processed_dir: Path | None = None):
        self.dir = Path(processed_dir or config.PROCESSED) / "audit"
        self._cache: dict[str, pd.DataFrame] = {}

    def table(self, name: str) -> pd.DataFrame:
        if name not in self._cache:
            self._cache[name] = pd.read_parquet(self.dir / f"{name}.parquet")
        return self._cache[name]

    def rows(self, name: str, district: str, season: int | None = None) -> pd.DataFrame:
        t = self.table(name)
        t = t[t["district"] == district]
        return t if season is None else t[t["season"] == int(season)]

    @property
    def districts(self) -> list[str]:
        return sorted(set(self.table("burned_vs_detected")["district"]) - {STATE})

    @property
    def seasons(self) -> list[int]:
        return sorted(int(s) for s in self.table("burned_vs_detected")["season"].unique())

    def check(self, district: str, season: int) -> None:
        """Raise ValueError for a district or season the audit tables do not cover."""
        if district != STATE and district not in self.districts:
            raise ValueError(f"unknown district '{district}'")
        if int(season) not in self.seasons:
            raise ValueError(f"no audit data for season {season}")


def _r(x, digits: int = 1):
    return None if x is None or pd.isna(x) else round(float(x), digits)


def change_words(pct, base: int) -> str | None:
    """-91.1 -> 'down 91.1% from 2022'. The agent quotes this instead of a signed number."""
    if pct is None or pd.isna(pct):
        return None
    pct = round(float(pct), 1)
    return f"unchanged from {base}" if pct == 0 else f"{'up' if pct > 0 else 'down'} {abs(pct)}% from {base}"


def _s(x) -> str | None:
    return x if isinstance(x, str) else None


def _none(district: str, season: int) -> dict:
    return {"error": f"not available for {district} in {season}"}


# ---------- the five data tools ----------

def get_detection_trend(store: AuditStore, district: str, season: int) -> dict:
    rows = {r["sensor"]: r for r in store.rows("sensor_trend", district, season).to_dict("records")}
    if not rows:
        return _none(district, season)
    base_counts = store.rows("sensor_trend", district).set_index(["sensor", "season"])["count"]
    sensors = []
    for key, name in SENSOR_NAMES.items():
        if key not in rows:
            continue
        r = rows[key]
        first = r["base_season"] == int(season)
        entry = {"satellite": name, "detections": int(r["count"]), "base_season": int(r["base_season"]),
                 "change_vs_base_pct": None if first else _r(r["change_pct"]),
                 "change_in_words": None if first else change_words(r["change_pct"], int(r["base_season"]))}
        warning = None if first else gap.small_base(base_counts[(key, r["base_season"])])
        if warning:
            entry["warning"] = warning
        sensors.append(entry)
    return {"district": district, "season": int(season), "sensors": sensors,
            "note": "NOAA-20 and NOAA-21 pass at about the same time as S-NPP, so they are a robustness check and "
                    "not an independent look. MODIS passes at other times."}


def get_diurnal_profile(store: AuditStore, district: str, season: int) -> dict:
    """Detections by time of pass, and the night share of S-NPP now and in the base season (one satellite, so the
    two seasons are comparable)."""
    season = int(season)
    prof = store.rows("diurnal_profile", district)
    now = prof[prof["season"] == season]
    if now.empty:
        return _none(district, season)

    def passes(sensor: str) -> list[dict]:
        p = now[now["sensor"] == sensor]
        return [{"pass": r["pass_label"], "detections": int(r["count"]), "share_pct": _r(r["share"] * 100)}
                for r in p.to_dict("records") if r["count"] > 0]

    def night_pct(y: int) -> float | None:
        p = prof[(prof["season"] == y) & (prof["sensor"] == "snpp") & (prof["pass_label"] == NIGHT)]
        return None if p.empty else _r(p["share"].iloc[0] * 100)

    base = gap.BASES[0]
    n, b = night_pct(season), night_pct(base)
    return {"district": district, "season": season, "base_season": base,
            "snpp_passes": passes("snpp"), "modis_passes": passes("modis"),
            "night_share_pct": n, "base_night_share_pct": b,
            "night_share_change_points": None if n is None or b is None else _r(n - b),
            "note": "Night share is of S-NPP detections. Satellites only see fires burning when they pass over."}


def get_burned_area_comparison(store: AuditStore, district: str, season: int) -> dict:
    rows = store.rows("burned_vs_detected", district, season)
    if rows.empty:
        return _none(district, season)
    r = rows.iloc[0]
    out = {"district": district, "season": int(season), "base_season": int(r["base_season"]), "product": BURNED_NAME,
           "burned_km2": _r(r["burned_km2"]), "snpp_detections": int(r["detections"]),
           "detections_per_burned_km2": _r(r["detections_per_km2"], 2)}
    if int(season) != int(r["base_season"]):
        out.update(burned_change_vs_base_pct=_r(r["burned_change_pct"]),
                   detections_change_vs_base_pct=_r(r["detections_change_pct"]),
                   per_km2_change_vs_base_pct=_r(r["per_km2_change_pct"]),
                   burned_change_in_words=change_words(r["burned_change_pct"], int(r["base_season"])),
                   detections_change_in_words=change_words(r["detections_change_pct"], int(r["base_season"])))
    if r["burned_km2"] < gap.MIN_BURNED_KM2:
        out["warning"] = f"burned area is under {gap.MIN_BURNED_KM2} km2, too small to rely on"
    return out


def get_official_vs_satellite(store: AuditStore, district: str, season: int) -> dict:
    t = store.rows("official_vs_satellite", district)
    now = t[t["season"] == int(season)]
    if now.empty:
        return _none(district, season)
    r = now.iloc[0]
    base = t[t["season"] == gap.BASES[0]]
    out = {"district": district, "season": int(season), "official_count": int(r["official"]), "snpp_detections": int(r["snpp"]),
           "official_to_snpp_ratio": _r(r["ratio"], 2), "source_url": r["source_url"], "how_counted": r["method_note"],
           "note": "The official count is satellite-based too, so it is not an independent check."}
    if int(season) != gap.BASES[0] and not base.empty:
        out.update(base_season=gap.BASES[0], base_official_count=int(base["official"].iloc[0]),
                   official_change_vs_base_pct=_r(gap.pct_change(r["official"], base["official"].iloc[0])),
                   official_change_in_words=change_words(gap.pct_change(r["official"], base["official"].iloc[0]), gap.BASES[0]))
        warning = gap.small_base(base["official"].iloc[0])
        if warning:
            out["warning"] = warning
    return out


def get_gap_range(store: AuditStore, district: str, season: int) -> dict:
    rows = store.rows("gap_range", district, season)
    if rows.empty:
        return {"error": f"no gap estimate for {district} in {season}: it needs an earlier base season"}
    g = rows.iloc[0]

    def pct(x):
        return None if pd.isna(x) else _r(x * 100)

    return {"district": district, "season": int(season),
            "what_the_gap_is": "the fall in fire counts per burned km2 since the base season; it is not the share of "
                               "burning that is missed",
            "detected_change_pct": _r(g["detected_change_pct"]), "reference_change_pct": _r(g["reference_change_pct"]),
            "changes_are_for": _s(g["detected_basis"]),
            "changes_in_words": None if pd.isna(g["detected_change_pct"]) else (
                f"counts changed {_r(g['detected_change_pct'])}% and burned area changed {_r(g['reference_change_pct'])}%"
                .replace("changed -", "fell ").replace("changed ", "rose ")),
            "gap_low_pct": pct(g["gap_low"]), "gap_high_pct": pct(g["gap_high"]), "methods": gap.method_words(g["methods"]),
            "confidence": g["confidence"], "direction": _s(g["direction"]), "reason": _s(g["reason"]),
            "timing_evidence": g["timing_evidence"], "other_explanations": gap.EXPLANATIONS, "caveat": g["caveat"]}


def available_signals(store: AuditStore, district: str, season: int) -> list[str]:
    """The names (never values) of the signals that exist for a district and season. This is all the Planner sees."""
    seen = set(store.rows("sensor_trend", district, season)["sensor"])
    names = [name for key, name in SIGNAL_NAMES.items() if key in seen]
    if not store.rows("burned_vs_detected", district, season).empty:
        names.append(SIGNAL_NAMES["burned_area"])
    if not store.rows("official_vs_satellite", district, season).empty:
        names.append(SIGNAL_NAMES["official"])
    return names


# ---------- schemas shown to the LLM ----------

_PLACE = {"district": {"type": "string", "description": "A district name, or 'Punjab' for the state total."},
          "season": {"type": "integer", "description": "Season year, for example 2025."}}


def _fn(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {"name": name, "description": description,
                                             "parameters": {"type": "object", "properties": properties, "required": required}}}


TOOL_SCHEMAS: list[dict] = [
    _fn("get_detection_trend", "Fire detections per satellite for a season, with the change against the base season.",
        _PLACE, ["district", "season"]),
    _fn("get_diurnal_profile", "Detections by time of satellite pass, and the night share now and in the base season.",
        _PLACE, ["district", "season"]),
    _fn("get_burned_area_comparison", "MODIS burned area, detections per burned km2, and changes against the base season.",
        _PLACE, ["district", "season"]),
    _fn("get_official_vs_satellite", "The official fire count, the S-NPP count, their ratio and the source address.",
        _PLACE, ["district", "season"]),
    _fn("get_gap_range", "The detection-gap range with its methods, confidence, timing evidence and caveat.",
        _PLACE, ["district", "season"]),
    _fn("search_knowledge",
        "Search the knowledge base (satellite pass times, official counting, the burned-area product, the gap method, "
        "published reporting, earlier season audits). Returns chunks with ids like K012 that you can cite.",
        {"query": {"type": "string"}, "k": {"type": "integer", "description": "Number of chunks, 1 to 6 (default 4)."}}, ["query"]),
]
_SCHEMA = {s["function"]["name"]: s["function"] for s in TOOL_SCHEMAS}
DATA_TOOLS = {"get_detection_trend": get_detection_trend, "get_diurnal_profile": get_diurnal_profile,
              "get_burned_area_comparison": get_burned_area_comparison,
              "get_official_vs_satellite": get_official_vs_satellite, "get_gap_range": get_gap_range}


def search_audit_knowledge(kb, query: str, k: int = 4, before=None, district: str | None = None,
                           place: str = STATE) -> dict:
    """`before`, `district` and `place` are injected by the orchestrator. Same result shape as `search_knowledge`."""
    hits = kb.search(query, k=max(1, min(int(k), 6)), before=before, district=district, audit_place=place)
    return {"results": [{"chunk_id": h["chunk_id"], "title": h["title"], "text": h["text"]} for h in hits]}


def build_tools(store: AuditStore, kb) -> list[Tool]:
    def bind(fn):
        return lambda district, season: fn(store, district, season)

    fns = {name: bind(fn) for name, fn in DATA_TOOLS.items()}
    fns["search_knowledge"] = lambda query, k=4, before=None, district=None, place=STATE: search_audit_knowledge(
        kb, query, k, before, district, place)
    return [Tool(n, _SCHEMA[n]["description"], _SCHEMA[n]["parameters"], fns[n]) for n in fns]
