"""A 5 km grid over Punjab and the cells where detections have clustered lately (D18).

The grid cell is the smallest unit the project reports: no villages, fields or farmers, and cell centres are rounded
to 0.01 degrees. Cells are aligned to multiples of 5 km in UTM 43N and clipped to the state.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

from psw import config

METRIC_CRS = "EPSG:32643"  # UTM 43N
CELL_M = 5000
DAYS = 7  # the look-back window, ending on the day asked about
CENTRAL_RADIUS = 0.33  # share of the district's half-width within which a cell is "central"
COMPASS = ("east", "north-east", "north", "north-west", "west", "south-west", "south", "south-east")


@dataclass
class Grid:
    cells: gpd.GeoDataFrame  # cell_id, row, col, district, area_km2, lat, lon, geometry (clipped, METRIC_CRS)
    districts: gpd.GeoDataFrame  # district, geometry (METRIC_CRS), indexed by district


def build_grid(geo_path=None) -> Grid:
    """Every 5 km cell that overlaps Punjab. A cell belongs to the district that covers most of it."""
    districts = gpd.read_file(geo_path or config.GEO / "punjab_districts.geojson").to_crs(METRIC_CRS)
    minx, miny, maxx, maxy = districts.total_bounds
    index = [(r, c) for r in range(math.floor(miny / CELL_M), math.ceil(maxy / CELL_M))
             for c in range(math.floor(minx / CELL_M), math.ceil(maxx / CELL_M))]
    squares = gpd.GeoDataFrame(
        {"row": [r for r, _ in index], "col": [c for _, c in index]},
        geometry=[box(c * CELL_M, r * CELL_M, (c + 1) * CELL_M, (r + 1) * CELL_M) for r, c in index], crs=METRIC_CRS)
    pieces = gpd.overlay(squares, districts[["district", "geometry"]], how="intersection", keep_geom_type=True)
    pieces = pieces.assign(area=pieces.area)
    pieces = pieces[pieces["area"] > 1.0]  # square metres: drop edges that only touch
    main = pieces.sort_values(["area", "district"], ascending=[False, True]).drop_duplicates(["row", "col"])
    # merge the pieces, so the slivers where two district shapes overlap are not counted twice
    cells = pieces[["row", "col", "geometry"]].dissolve(by=["row", "col"]).reset_index()
    cells = cells.merge(main[["row", "col", "district"]], on=["row", "col"]).sort_values(["row", "col"]).reset_index(drop=True)
    centres = cells.geometry.centroid.to_crs("EPSG:4326")
    cells = cells.assign(cell_id=np.arange(1, len(cells) + 1), area_km2=cells.area / 1e6,
                         lat=centres.y.round(2), lon=centres.x.round(2))
    cells = cells[["cell_id", "row", "col", "district", "area_km2", "lat", "lon", "geometry"]]
    return Grid(cells=cells, districts=districts.set_index("district", drop=False)[["district", "geometry"]])


def cell_detections(fires: pd.DataFrame, grid: Grid, day: date | str, days: int = DAYS) -> pd.DataFrame:
    """district, cell_id, detections, share for the `days` days ending on `day` (nothing later).

    `fires` has latitude, longitude, acq_date and district. A detection counts for its own district, so a cell on a
    border can appear under two, and each district's shares add up to 1.
    """
    day = date.fromisoformat(day) if isinstance(day, str) else day
    when = pd.to_datetime(fires["acq_date"]).dt.date
    recent = fires[fires["district"].notna() & (when > day - timedelta(days=days)) & (when <= day)]
    empty = pd.DataFrame(columns=["district", "cell_id", "detections", "share"])
    if recent.empty:
        return empty
    points = gpd.GeoSeries(gpd.points_from_xy(recent["longitude"], recent["latitude"]), crs="EPSG:4326").to_crs(METRIC_CRS)
    at = pd.DataFrame({"district": recent["district"].to_numpy(),
                       "row": np.floor(points.y.to_numpy() / CELL_M).astype(int),
                       "col": np.floor(points.x.to_numpy() / CELL_M).astype(int)})
    at = at.merge(grid.cells[["row", "col", "cell_id"]], how="left", on=["row", "col"])
    if at["cell_id"].isna().any():
        raise ValueError("a detection inside a district falls in no grid cell")
    out = at.groupby(["district", "cell_id"]).size().rename("detections").reset_index()
    out["cell_id"] = out["cell_id"].astype(int)
    out["share"] = out["detections"] / out.groupby("district")["detections"].transform("sum")
    return out


def _part_of(point, shape) -> str:
    """Where a point lies within a district: "central" or a compass direction."""
    minx, miny, maxx, maxy = shape.bounds
    centre = shape.centroid
    dx = (point.x - centre.x) / ((maxx - minx) / 2)
    dy = (point.y - centre.y) / ((maxy - miny) / 2)
    if math.hypot(dx, dy) < CENTRAL_RADIUS:
        return "central"
    return COMPASS[round(math.degrees(math.atan2(dy, dx)) / 45) % 8]


def hotspot_cells(district: str, day: date | str, fires: pd.DataFrame, grid: Grid, top: int = 5) -> list[dict]:
    """The district's cells with the most detections in the last 7 days, busiest first. Empty if it had none."""
    if district not in grid.districts.index:
        raise ValueError(f"unknown district {district!r}")
    found = cell_detections(fires, grid, day)
    found = found[found["district"] == district].sort_values(["detections", "cell_id"], ascending=[False, True]).head(top)
    cells = grid.cells.set_index("cell_id")
    shape = grid.districts.geometry[district]
    return [{"cell_id": int(r.cell_id),
             "label": f"cell {r.cell_id} ({_part_of(cells.geometry[r.cell_id].centroid, shape)} {district})",
             "lat": float(cells["lat"][r.cell_id]), "lon": float(cells["lon"][r.cell_id]),
             "detections_7d": int(r.detections), "share_pct": round(float(r.share) * 100, 1)}
            for r in found.itertuples()]
