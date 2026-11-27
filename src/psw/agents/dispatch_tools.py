"""Tools for the Dispatcher, and the plan they report. Pure Python over data/processed; no LLM maths.

The fleet size, the per-district limit and whether to plan on likely burning are the user's inputs. They are injected
into every call, so the model cannot choose them. Figures are rounded here so they can be quoted as they are, and no
tool uses a field name as a value.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, timedelta
from functools import cached_property
from pathlib import Path

import pandas as pd

from psw import config
from psw.agents import audit_tools
from psw.agents.runtime import Tool
from psw.agents.verifier import Plan
from psw.response import allocate, harvest, machines, risk
from psw.response import grid as grid_mod

TOP_DISTRICTS = 8
HOTSPOTS = 5
MIN_CLUSTER = 2  # detections in 7 days before a cell is offered to awareness teams
STATE = audit_tools.STATE
ASSUMPTION = "How much one unit achieves is an assumption, not a measurement."
LIKELY_WORDS = ("Likely burning scales the forecast by the previous season's detection gap. It estimates what would "
                "have been detected at the base season's detections per burned km2. It is not a count of fires.")


def as_date(day: date | str) -> date:
    return date.fromisoformat(day) if isinstance(day, str) else day


@dataclass(frozen=True)
class Settings:
    """What the user chose. Never the LLM."""
    units: int = allocate.DEFAULT_UNITS
    max_share: float = allocate.DEFAULT_MAX_SHARE
    use_likely: bool = True


class ResponseStore:
    """Lazy, cached access to the response data. Pass `processed_dir` to point it at test fixtures."""

    def __init__(self, processed_dir: Path | None = None, geo_path: Path | None = None):
        self.dir = Path(processed_dir or config.PROCESSED)
        self.geo_path = geo_path

    @cached_property
    def forecasts(self) -> pd.DataFrame:
        return pd.read_parquet(self.dir / risk.FORECAST_FILE)

    @cached_property
    def variant(self) -> str:
        return json.loads((self.dir / risk.METRICS_FILE).read_text())["best_ranking_variant"]

    @cached_property
    def daily(self) -> pd.DataFrame:
        return pd.read_parquet(self.dir / "district_daily.parquet")

    @cached_property
    def rain(self) -> pd.DataFrame:
        weather = pd.read_parquet(self.dir / "district_weather_daily.parquet")
        return harvest.rain_features(weather).set_index(["district", "date"])

    @cached_property
    def fires(self) -> pd.DataFrame:
        return pd.read_parquet(self.dir / "fires.parquet", columns=["latitude", "longitude", "acq_date", "district"])

    @cached_property
    def grid(self) -> grid_mod.Grid:
        return grid_mod.build_grid(self.geo_path)

    @cached_property
    def audit(self) -> audit_tools.AuditStore:
        return audit_tools.AuditStore(self.dir)

    @cached_property
    def stock(self) -> pd.DataFrame:
        return machines.load_machine_stock(geo_path=self.geo_path)

    @property
    def dates(self) -> list[date]:
        return sorted(self.forecasts["issue_date"].unique())

    def check(self, day: date | str, district: str | None = None) -> date:
        """The date, or a ValueError if there is no forecast for it (or no such district)."""
        day = as_date(day)
        if day not in set(self.dates):
            raise ValueError(f"no fire-risk forecast was issued on {day}: forecasts run {self.dates[0]} to "
                             f"{self.dates[-1]}, 15 Sep to 27 Nov of each season")
        if district is not None and district not in set(self.forecasts["district"]):
            raise ValueError(f"unknown district '{district}'")
        return day

    def risk_on(self, day: date | str) -> pd.DataFrame:
        """One row per district: the selected variant's forecast issued on `day`."""
        f = self.forecasts
        return f[(f["issue_date"] == self.check(day)) & (f["variant"] == self.variant)].reset_index(drop=True)

    def gap_confidence(self, district: str, season: int, source: str) -> str | None:
        """Confidence of the previous season's range that scales a district: its own, or Punjab's."""
        rows = self.audit.rows("gap_range", district if source == "district" else STATE, season - 1)
        return None if rows.empty else str(rows["confidence"].iloc[0])

    def last_days(self, district: str, day: date, days: int = 3) -> int:
        d = self.daily
        recent = d[(d["district"] == district) & (d["date"] > day - timedelta(days=days)) & (d["date"] <= day)]
        return int(recent["fire_count"].sum())


# ---------- the allocation, computed once and shared by the tool, the record, the template and the check ----------

@dataclass
class Allocation:
    table: pd.DataFrame  # allocate() output for the user's settings
    detected: pd.DataFrame  # the same fleet on forecast detections only
    moved: pd.DataFrame | None  # compare_to_detected, when planning on likely burning
    settings: Settings
    confidence: dict[str, str]  # district -> confidence of the gap range that scales it (likely planning only)

    @property
    def placed(self) -> int:
        return int(self.table["units"].sum())

    @property
    def cap(self) -> int:
        return allocate.unit_cap(self.settings.units, self.settings.max_share)

    def moves(self, top: int = 3) -> list[dict]:
        if self.moved is None:
            return []
        out = []
        for r in self.moved[self.moved["moved"] != 0].head(top).itertuples():
            out.append({"district": r.district, "units_with_adjustment": int(r.units_likely),
                        "units_without_adjustment": int(r.units_detected),
                        "change_in_words": f"{abs(int(r.moved))} {'more' if r.moved > 0 else 'fewer'} units with the adjustment",
                        "confidence_of_its_gap_range": self.confidence.get(r.district)})
        return out

    def plan(self, words: frozenset[str] = frozenset()) -> Plan:
        units = dict(zip(self.table["district"], self.table["units"].astype(int)))
        moves = self.moves(1)
        others = int(self.table["units"].iloc[TOP_DISTRICTS:].sum())
        # the range that matters most is the one behind the biggest move, else behind the leading district
        about = moves[0]["district"] if moves else str(self.table["district"].iloc[0])
        return Plan(units=units, detected_units=dict(zip(self.detected["district"], self.detected["units"].astype(int))),
                    totals={"fleet": self.settings.units, "placed": self.placed,
                            "unplaced": self.settings.units - self.placed, "cap": self.cap, "others": others},
                    use_likely=self.settings.use_likely, most_moved=moves[0]["district"] if moves else None,
                    confidence=self.confidence.get(about), words=words)


def plan_allocation(store: ResponseStore, day: date | str, settings: Settings) -> Allocation:
    table = store.risk_on(day)
    args = {"max_share": settings.max_share}
    chosen = allocate.allocate(table, settings.units, use_likely=settings.use_likely, **args)
    detected = allocate.allocate(table, settings.units, use_likely=False, **args)
    moved = allocate.compare_to_detected(table, settings.units, **args) if settings.use_likely else None
    confidence = {}
    if settings.use_likely:
        season = as_date(day).year
        for r in table.dropna(subset=["likely_p50"]).itertuples():
            confidence[r.district] = store.gap_confidence(r.district, season, r.gap_source)
    return Allocation(chosen, detected, moved, settings, confidence)


# ---------- tools ----------

def _district_forecast(store: ResponseStore, r, day: date, use_likely: bool) -> dict:
    last = store.last_days(r.district, day)
    forecast = int(round(r.p50))
    out = {"district": r.district, "forecast_detections": forecast, "band_80_low": int(round(r.p10)),
           "band_80_high": int(round(r.p90)), "detections_last_3_days": last,
           "compared_with_last_3_days": "more" if forecast > last else "fewer" if forecast < last else "about the same"}
    if use_likely and pd.notna(r.likely_p50):
        out.update(likely_burning=int(round(r.likely_p50)), likely_low=int(round(r.likely_low)),
                   likely_high=int(round(r.likely_high)),
                   detection_gap_from="its own district range" if r.gap_source == "district" else "the Punjab range",
                   confidence_of_its_gap_range=store.gap_confidence(r.district, day.year, r.gap_source))
    return out


def get_risk_forecast(store: ResponseStore, date: str, use_likely: bool = True, district: str | None = None) -> dict:
    day = store.check(date, district)
    table = store.risk_on(day)
    values, _ = allocate.risk_used(table, use_likely)
    table = table.assign(rank_by=values).sort_values(["rank_by", "district"], ascending=[False, True])
    out = {"date": str(day), "days_ahead": risk.HORIZON,
           "what_is_forecast": "satellite fire detections in each district over the next 3 days",
           "ranked_by": "likely burning" if use_likely else "forecast detections",
           "top_districts": [_district_forecast(store, r, day, use_likely)
                             for r in table.head(TOP_DISTRICTS).itertuples()],
           "number_of_districts": len(table), "uncertainty": "The 80% band is the forecast's own uncertainty.",
           "rain_note": harvest.RAIN_ASSUMPTION}
    if use_likely:
        out["likely_note"] = LIKELY_WORDS
    if district:
        row = next(table[table["district"] == district].itertuples())
        out["asked_district"] = _district_forecast(store, row, day, use_likely)
    return out


def get_allocation(store: ResponseStore, date: str, units: int = allocate.DEFAULT_UNITS,
                   max_share: float = allocate.DEFAULT_MAX_SHARE, use_likely: bool = True,
                   district: str | None = None) -> dict:
    day = store.check(date, district)
    a = plan_allocation(store, day, Settings(units, max_share, use_likely))

    def row(r) -> dict:
        name = "likely_burning" if r.basis == allocate.LIKELY else "forecast_detections"
        return {"district": r.district, "units": int(r.units), name: int(round(r.risk_used))}

    rest = a.table.iloc[TOP_DISTRICTS:]
    out = {"date": str(day), "days_ahead": risk.HORIZON, "units_available": units, "units_placed": a.placed,
           "limit_per_district": a.cap,
           "planned_on": "likely burning" if use_likely else "forecast detections",
           "districts": [row(r) for r in a.table.head(TOP_DISTRICTS).itertuples() if r.units > 0],
           "other_districts_with_units": {"number": int((rest["units"] > 0).sum()), "units_they_share": int(rest["units"].sum())},
           "note": "The split was computed by a fixed rule; you explain it and cannot change it. " + ASSUMPTION}
    if a.placed < units:
        out["units_not_placed"] = units - a.placed
    if use_likely:
        out["moved_by_the_likely_adjustment"] = a.moves()
    if district:
        found = a.table[a.table["district"] == district]
        without = a.detected.set_index("district")["units"]
        out["asked_district"] = {**row(next(found.itertuples())), "units_without_adjustment": int(without[district])}
    return out


def get_hotspot_cells(store: ResponseStore, district: str, date: str) -> dict:
    """Only cells with at least 2 detections are returned: a single detection is not a cluster, and a team should not
    be sent to one."""
    day = store.check(date, district)
    cells = grid_mod.hotspot_cells(district, day, store.fires, store.grid, top=HOTSPOTS)
    out = [{"cell": c["label"], "detections": c["detections_7d"], "share_of_district_pct": c["share_pct"]}
           for c in cells if c["detections_7d"] >= MIN_CLUSTER]
    note = ("Cells come from satellite detections in the last 7 days. They cannot show burning that was not detected.")
    if not out:
        note += (" No cell had more than a single detection, so there is no cluster to point to." if cells else
                 " This district had no detections in that time.")
    return {"district": district, "date": str(day), "window_days": grid_mod.DAYS,
            "district_detections": store.last_days(district, day, grid_mod.DAYS), "cells": out, "note": note}


def get_rain_outlook(store: ResponseStore, district: str, date: str) -> dict:
    day = store.check(date, district)
    r = store.rain.loc[(district, day)]
    return {"district": district, "date": str(day), "days": harvest.RAIN_DAYS,
            "rain_last_3_days_mm": round(float(r["rain_last3_mm"]), 1),
            "rain_next_3_days_mm": round(float(r["rain_next3_mm"]), 1),
            "note": "The next 3 days are taken from the weather archive, standing in for a forecast. Rain can delay "
                    "burning."}


def get_gap_range(store: ResponseStore, district: str, season: int) -> dict:
    """Module A's range for the PREVIOUS season (the orchestrator injects it)."""
    out = audit_tools.get_gap_range(store.audit, district, season)
    if "error" not in out:
        out["used_for"] = ("This is the previous season's range, which scales this season's forecast. A negative value "
                           "is treated as no adjustment.")
    return out


def get_machine_stock(store: ResponseStore, district: str, date: str) -> dict:
    day = as_date(date)
    if day < machines.PUBLISHED:
        return {"error": "the machine figures had not been published on this date"}
    rows = store.stock[store.stock["district"] == district]
    if rows.empty:
        return {"error": f"no machine figures for '{district}'"}
    largest = rows.sort_values(["count", "machine_type"], ascending=[False, True]).head(5)
    return {"district": district, "machines_supplied": int(rows["count"].sum()),
            "largest_types": [{"type": kind.replace("_", " "), "supplied": int(n)}
                              for kind, n in zip(largest["machine_type"], largest["count"])],
            "period": rows["period"].iloc[0], "source_url": rows["source_url"].iloc[0], "caveat": machines.CAVEAT}


# ---------- schemas shown to the LLM ----------

_fn = audit_tools._fn
_DATE = {"date": {"type": "string", "description": "The issue date, YYYY-MM-DD."}}
_DISTRICT = {"district": {"type": "string", "description": "A district name, as the tools write it."}}
_OPTIONAL = {"district": {"type": "string", "description": "Optional: also report this district."}}

TOOL_SCHEMAS: list[dict] = [
    _fn("get_risk_forecast", "The 3-day fire forecast for the districts that lead it, with its 80% band, likely burning "
        "and the detections of the last 3 days.", {**_DATE, **_OPTIONAL}, ["date"]),
    _fn("get_allocation", "How the units are split across districts, and what the likely-burning adjustment moves.",
        {**_DATE, **_OPTIONAL}, ["date"]),
    _fn("get_hotspot_cells", "The district's 5 km cells where detections clustered in the last 7 days, for awareness teams.",
        {**_DISTRICT, **_DATE}, ["district", "date"]),
    _fn("get_rain_outlook", "Rain in a district over the last and the next 3 days.", {**_DISTRICT, **_DATE},
        ["district", "date"]),
    _fn("get_gap_range", "The previous season's detection-gap range for a district or 'Punjab', with its confidence and "
        "caveat.", _DISTRICT, ["district"]),
    _fn("search_knowledge", "Search the knowledge base (ways to manage straw without burning, machines, the gap method, "
        "pass times, earlier audits). Returns chunks, each with a chunk_id you can cite.",
        {"query": {"type": "string"}, "k": {"type": "integer", "description": "Number of chunks, 1 to 6 (default 4)."}},
        ["query"]),
    _fn("get_machine_stock", "Machines supplied on subsidy in a district, by type. Context only: it does not change the "
        "split.", _DISTRICT, ["district"]),
]
_SCHEMA = {s["function"]["name"]: s["function"] for s in TOOL_SCHEMAS}


def build_tools(store: ResponseStore, kb) -> list[Tool]:
    fns = {
        "get_risk_forecast": lambda date, use_likely=True, district=None: get_risk_forecast(store, date, use_likely, district),
        "get_allocation": lambda date, units=allocate.DEFAULT_UNITS, max_share=allocate.DEFAULT_MAX_SHARE, use_likely=True,
        district=None: get_allocation(store, date, units, max_share, use_likely, district),
        "get_hotspot_cells": lambda district, date: get_hotspot_cells(store, district, date),
        "get_rain_outlook": lambda district, date: get_rain_outlook(store, district, date),
        "get_gap_range": lambda district, season: get_gap_range(store, district, season),
        "search_knowledge": lambda query, k=4, before=None: audit_tools.search_audit_knowledge(kb, query, k, before=before),
        "get_machine_stock": lambda district, date: get_machine_stock(store, district, date),
    }
    return [Tool(n, _SCHEMA[n]["description"], _SCHEMA[n]["parameters"], fns[n]) for n in fns]
