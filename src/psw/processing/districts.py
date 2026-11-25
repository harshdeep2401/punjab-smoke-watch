"""Assign fires to districts and build the district x day table."""
from __future__ import annotations

from datetime import date

import geopandas as gpd
import pandas as pd

from psw import config


def assign_districts(fires: pd.DataFrame, geo_path) -> pd.DataFrame:
    """Point-in-polygon join (EPSG:4326 both sides). Adds `district` (NaN outside Indian Punjab) and `in_punjab`.

    Points exactly on a boundary are not 'within' any polygon and get NaN.
    """
    districts = gpd.read_file(geo_path).to_crs("EPSG:4326")[["district", "geometry"]]
    pts = gpd.GeoDataFrame(
        fires[["latitude", "longitude"]].copy(),
        geometry=gpd.points_from_xy(fires["longitude"], fires["latitude"]),
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(pts, districts, how="left", predicate="within")
    # districts do not overlap, but guard against a point matching two polygons
    joined = joined[~joined.index.duplicated(keep="first")]
    out = fires.copy()
    out["district"] = joined["district"].reindex(out.index)
    out["in_punjab"] = out["district"].notna()
    return out


def season_days(seasons: list[int]) -> pd.DataFrame:
    rows = []
    for y in seasons:
        days = pd.date_range(date(y, *config.SEASON_START), date(y, *config.SEASON_END), freq="D").date
        rows.append(pd.DataFrame({"date": days, "season": y}))
    return pd.concat(rows, ignore_index=True)


def district_daily(fires: pd.DataFrame, district_names: list[str], seasons: list[int] | None = None) -> pd.DataFrame:
    """Full grid of every district x every season day (zeros filled).

    Columns: date, season, district, fire_count, frp_sum, fire_count_7d_mean (trailing 7 days incl. today,
    computed within each season/district; the first days of a season use the days available).
    """
    seasons = seasons if seasons is not None else sorted(fires["season"].unique())
    grid = season_days(seasons).merge(pd.DataFrame({"district": district_names}), how="cross")
    agg = (fires[fires["district"].notna()]
           .groupby(["acq_date", "district"])
           .agg(fire_count=("frp", "size"), frp_sum=("frp", "sum"))
           .reset_index()
           .rename(columns={"acq_date": "date"}))
    df = grid.merge(agg, on=["date", "district"], how="left")
    df[["fire_count", "frp_sum"]] = df[["fire_count", "frp_sum"]].fillna(0)
    df["fire_count"] = df["fire_count"].astype(int)
    df = df.sort_values(["district", "season", "date"]).reset_index(drop=True)
    df["fire_count_7d_mean"] = (df.groupby(["district", "season"])["fire_count"]
                                .transform(lambda s: s.rolling(7, min_periods=1).mean()))
    return df[["date", "season", "district", "fire_count", "frp_sum", "fire_count_7d_mean"]]
