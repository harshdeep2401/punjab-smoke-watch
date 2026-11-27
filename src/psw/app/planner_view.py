"""Pure helpers for the Response Planner page (no Streamlit import).

The allocation is the same `plan_allocation` the Dispatcher's tools use, and figures are rounded the same way, so the
page and a brief written for the same inputs agree. Nothing here is finer than a 5 km cell (D18).
"""
from __future__ import annotations

from datetime import date, timedelta

import geopandas as gpd
import numpy as np
import pandas as pd

from psw.agents.critic import base_mode
from psw.agents.dispatch_tools import MIN_CLUSTER, ResponseStore, Allocation
from psw.app.helpers import peak_day
from psw.response import allocate, risk
from psw.response import grid as grid_mod

CAPTION = "Decision support for residue-management help. Not for enforcement. No farms or farmers identified."
SOURCE_WORDS = {"district": "its own district range", "Punjab": "the Punjab range"}
CELLS_SHOWN = 5


def default_date(daily: pd.DataFrame, available: list[date]) -> date:
    """Three days before the latest season's busiest day, or the last date with a forecast."""
    wanted = peak_day(daily) - timedelta(days=risk.HORIZON)
    return wanted if wanted in set(available) else max(available)


def window(day: date) -> tuple[date, date]:
    """The days a forecast issued on `day` covers."""
    return day + timedelta(days=1), day + timedelta(days=risk.HORIZON)


# ---------- district risk ----------

def risk_table(store: ResponseStore, allocation: Allocation, day: date) -> pd.DataFrame:
    """One row per district, in the allocation's order (most units first)."""
    rows = store.risk_on(day).set_index("district")
    out = []
    for district in allocation.table["district"]:
        r = rows.loc[district]
        likely = pd.notna(r["likely_p50"])
        out.append({
            "district": district,
            "forecast detections": int(round(r["p50"])),
            "80% band low": int(round(r["p10"])),
            "80% band high": int(round(r["p90"])),
            "likely burning": int(round(r["likely_p50"])) if likely else None,
            "likely low": int(round(r["likely_low"])) if likely else None,
            "likely high": int(round(r["likely_high"])) if likely else None,
            "gap confidence": store.gap_confidence(district, day.year, r["gap_source"]) if likely else None,
            "scaled by": SOURCE_WORDS.get(r["gap_source"]) if likely else None,
            "detections, last 3 days": store.last_days(district, day),
        })
    return pd.DataFrame(out)


def shade(value: float, vmax: float) -> list[int]:
    """White to dark red on a square-root scale, the same ramp as the replay map."""
    t = float(np.sqrt(min(value / vmax, 1.0))) if vmax > 0 else 0.0
    return [int(255 - 115 * t), int(245 - 210 * t), int(235 - 215 * t), 220]


def risk_features(geo: dict, allocation: Allocation) -> tuple[dict, float]:
    """(GeoJSON of the districts coloured by the expected burning the plan uses, the largest value)."""
    rows = allocation.table.set_index("district")
    vmax = float(rows["risk_used"].max()) if len(rows) else 0.0
    features = []
    for f in geo["features"]:
        name = f["properties"]["district"]
        r = rows.loc[name]
        kind = "likely burning" if r["basis"] == allocate.LIKELY else "forecast detections"
        features.append({"type": "Feature", "geometry": f["geometry"],
                         "properties": {"color": shade(float(r["risk_used"]), vmax)},
                         "tip": f"{name}: {kind} {r['risk_used']:.0f}, {int(r['units'])} units"})
    return {"type": "FeatureCollection", "features": features}, vmax


# ---------- allocation ----------

def allocation_table(allocation: Allocation) -> pd.DataFrame:
    """The plan, with the same fleet on forecast detections alone beside it when planning on likely burning."""
    out = allocation.table[["district", "units", "reason"]].copy()
    if allocation.settings.use_likely:
        without = allocation.detected.set_index("district")["units"]
        out.insert(2, "on detections only", [int(without[d]) for d in out["district"]])
        out.insert(3, "moved", out["units"] - out["on detections only"])
    return out.reset_index(drop=True)


def plan_sentences(allocation: Allocation, top: int = 3) -> list[str]:
    """The plan in words, computed from the allocation."""
    s = allocation.settings
    placed = allocation.table[allocation.table["units"] > 0]
    if placed.empty:
        return [f"No burning is forecast in any district, so none of the {s.units} units are placed."]
    lead = ", ".join(f"{d} ({int(u)})" for d, u in zip(placed["district"].head(top), placed["units"].head(top)))
    out = [f"Of {s.units} units, the largest shares go to {lead}. {len(placed)} districts get units; "
           f"no district may take more than {allocation.cap}."]
    if allocation.placed < s.units:
        out.append(f"**{s.units - allocation.placed} units are not placed:** every district with expected burning is "
                   "at the limit.")
    if not s.use_likely:
        out.append("This plan uses forecast detections only, so burning that satellites miss is not counted.")
        return out
    moves = allocation.moves(top)
    if moves:
        words = "; ".join(f"{m['district']} {m['units_without_adjustment']} to {m['units_with_adjustment']} "
                          f"(gap range confidence {m['confidence_of_its_gap_range']})" for m in moves)
        out.append(f"Planning on likely burning instead of forecast detections moves most: {words}.")
    else:
        out.append("Planning on likely burning gives the same split as forecast detections on this date.")
    return out


# ---------- hotspot cells ----------

def cluster_cells(store: ResponseStore, district: str, day: date) -> tuple[list[dict], int]:
    """(cells with at least 2 detections in the last 7 days, busiest first; how many cells had just one).

    A single detection is not a cluster, so those cells are counted but never offered or drawn as hotspots. The same
    rule and the same top 5 as the Dispatcher's tool.
    """
    found = grid_mod.hotspot_cells(district, day, store.fires, store.grid, top=CELLS_SHOWN)
    cells = [c for c in found if c["detections_7d"] >= MIN_CLUSTER]
    every = grid_mod.cell_detections(store.fires, store.grid, day)
    singles = int(((every["district"] == district) & (every["detections"] < MIN_CLUSTER)).sum())
    return cells, singles


def cell_features(store: ResponseStore, cells: list[dict]) -> dict:
    """The 5 km squares of `cells` (clipped to the state) as GeoJSON."""
    shapes = store.grid.cells.set_index("cell_id").geometry
    features = []
    if cells:
        squares = gpd.GeoSeries([shapes[c["cell_id"]] for c in cells], crs=grid_mod.METRIC_CRS).to_crs("EPSG:4326")
        features = [{"type": "Feature", "geometry": square.__geo_interface__, "properties": {},
                     "tip": f"{c['label']}: {c['detections_7d']} detections in 7 days"}
                    for c, square in zip(cells, squares)]
    return {"type": "FeatureCollection", "features": features}


def fit_view(bounds, width: int, height: int, margin: float = 0.15) -> tuple[float, float, float]:
    """(lat, lon, zoom) that shows all of `bounds` (min lon, min lat, max lon, max lat) in a map of this size.

    Web Mercator with 512 px tiles. `width` should be the narrowest the map is likely to be drawn.
    """
    west, south, east, north = (float(v) for v in bounds)
    y = lambda lat: np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))  # noqa: E731
    across = max(east - west, 1e-6) / 360
    up = max(y(north) - y(south), 1e-6) / (2 * np.pi)
    zoom = min(np.log2(width / (512 * across)), np.log2(height / (512 * up))) - margin
    return (south + north) / 2, (west + east) / 2, round(float(zoom), 2)


def district_view(store: ResponseStore, district: str, width: int, height: int) -> tuple[float, float, float]:
    """(lat, lon, zoom) that shows the whole district."""
    shape = gpd.GeoSeries([store.grid.districts.geometry[district]], crs=grid_mod.METRIC_CRS).to_crs("EPSG:4326")
    return fit_view(shape.total_bounds, width, height)


def state_view(store: ResponseStore, width: int, height: int) -> tuple[float, float, float]:
    return fit_view(store.grid.districts.to_crs("EPSG:4326").total_bounds, width, height)


# ---------- the dispatch brief ----------

def matches(record: dict, units: int, max_share: float, use_likely: bool) -> bool:
    """Whether a brief was written for these inputs, so its unit figures are the ones on the page."""
    return (record.get("units"), record.get("max_share"), record.get("use_likely")) == (units, max_share, use_likely)


def keep_dispatch(new: dict, saved: dict | None) -> str:
    """What to do with a brief the page just made. There is one saved brief per date, so:

    "save"     nothing is saved for the date, or the saved one has the same inputs and can be replaced
    "kept"     the run fell back to the template and an agent's brief for the same inputs is saved: keep that one
    "session"  the saved brief was written for other inputs: show the new one without overwriting it
    """
    if saved is None:
        return "save"
    inputs = ("units", "max_share", "use_likely", "district")
    if any(new.get(k) != saved.get(k) for k in inputs):
        return "session"
    if base_mode(new["mode"]) == "template" and base_mode(saved.get("mode")) != "template":
        return "kept"
    return "save"
