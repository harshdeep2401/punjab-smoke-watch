"""Burned area per district and season from MODIS MCD64A1 (500 m, monthly burn-date GeoTIFFs).

Source (D22): the University of Maryland SFTP server, window 18 (South Asia). The product's user guide says
burned area in cropland is low confidence and misses many small fires, so read these figures as a floor-like
signal and not as a measurement of everything that burned.
"""
from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window, from_bounds

from psw import config

log = logging.getLogger(__name__)

# Server and shared login are published in the MCD64A1 Collection 6.1 User's Guide, section 4.1
# (https://lpdaac.usgs.gov/documents/1006/MCD64_User_Guide_V61.pdf): "we openly share the password".
UMD_HOST, UMD_USER, UMD_PASSWORD = "fuoco.geog.umd.edu", "fire", "burnt"
UMD_DIR = "data/MODIS/C61/MCD64A1/TIFF/Win18/{year}"

MONTHS = (9, 10, 11)  # the months the season window touches
UNMAPPED = -1  # too little data to decide; 0 is unburned, -2 water, 1..366 the burn day of year
KM_PER_DEG = 111.195
COLUMNS = ["district", "season", "burned_km2", "n_pixels"]


def file_name(year: int, month: int) -> str:
    """The server names each monthly file by the day of year of the month's first day."""
    return f"MCD64monthly.A{year}{date(year, month, 1).timetuple().tm_yday:03d}.Win18.061.burndate.tif"


def _download(year: int, paths: list[Path]) -> None:
    """Fetch the named files. The server's host key is not checked (public data, public password)."""
    import paramiko

    transport = paramiko.Transport((UMD_HOST, 22))
    try:
        transport.connect(username=UMD_USER, password=UMD_PASSWORD)
        sftp = paramiko.SFTPClient.from_transport(transport)
        for path in paths:
            part = path.with_name(path.name + ".part")
            try:
                sftp.get(f"{UMD_DIR.format(year=year)}/{path.name}", str(part))
            except OSError as exc:
                raise RuntimeError(f"could not download MCD64A1 file {path.name}: {exc}") from exc
            part.rename(path)
    finally:
        transport.close()


def fetch_rasters(year: int) -> list[Path]:
    """The Sep, Oct and Nov rasters for a season, downloading only those not cached. All three or an error."""
    cache = config.RAW / "burned" / "mcd64_win18"
    paths = [cache / file_name(year, m) for m in MONTHS]
    missing = [p for p in paths if not p.exists()]
    if missing:
        cache.mkdir(parents=True, exist_ok=True)
        log.info("downloading %d MCD64A1 files for %d", len(missing), year)
        _download(year, missing)
    return paths


def read_window(paths: list[Path], districts: gpd.GeoDataFrame) -> tuple[list[np.ndarray], rasterio.Affine]:
    """The part of each raster that covers the districts, plus that part's transform."""
    arrays, transforms = [], []
    for path in paths:
        with rasterio.open(path) as src:
            exact = from_bounds(*districts.total_bounds, transform=src.transform)
            # round outwards so edge pixels are not lost, then stay inside the raster
            col, row = int(np.floor(exact.col_off)), int(np.floor(exact.row_off))
            win = Window(col, row, int(np.ceil(exact.col_off + exact.width)) - col,
                         int(np.ceil(exact.row_off + exact.height)) - row)
            win = win.intersection(Window(0, 0, src.width, src.height))
            arrays.append(src.read(1, window=win))
            transforms.append(src.window_transform(win))
    if any(t != transforms[0] for t in transforms):
        raise RuntimeError("MCD64A1 monthly rasters are not on the same grid")
    return arrays, transforms[0]


def district_burned_area(monthly: list[np.ndarray], transform: rasterio.Affine, districts: gpd.GeoDataFrame,
                         season: int) -> pd.DataFrame:
    """Burned km2 per district for burn dates inside the season window.

    A pixel belongs to the district holding its centre, and counts once even if it burned in two months.
    The grid is lat/lon, so each row of pixels has its own area. `n_pixels` is the number of burned pixels.
    """
    lo = date(season, *config.SEASON_START).timetuple().tm_yday
    hi = date(season, *config.SEASON_END).timetuple().tm_yday
    stack = np.stack(monthly)
    burned = ((stack >= lo) & (stack <= hi)).any(axis=0)
    ids = rasterize([(g, i) for i, g in enumerate(districts.geometry, start=1)], out_shape=burned.shape,
                    transform=transform, fill=0, dtype="uint16")
    unmapped = int(((stack == UNMAPPED).any(axis=0) & (ids > 0)).sum())
    if unmapped:
        log.warning("%d: %d pixels inside districts are unmapped in at least one month", season, unmapped)

    lat = transform.f + (np.arange(burned.shape[0]) + 0.5) * transform.e
    row_km2 = abs(transform.e) * KM_PER_DEG * transform.a * KM_PER_DEG * np.cos(np.radians(lat))
    area = np.broadcast_to(row_km2[:, None], burned.shape)
    n = len(districts) + 1
    km2 = np.bincount(ids[burned], weights=area[burned], minlength=n)
    pixels = np.bincount(ids[burned], minlength=n)
    return pd.DataFrame({"district": list(districts["district"]), "season": season,
                         "burned_km2": km2[1:], "n_pixels": pixels[1:].astype(int)})[COLUMNS]


def mcd64_district_season(year: int, geo_path=None) -> pd.DataFrame:
    """district, season, burned_km2 (burn dates Sep 15 to Nov 30), n_pixels. One row per district."""
    districts = gpd.read_file(geo_path or config.GEO / "punjab_districts.geojson").to_crs("EPSG:4326")
    arrays, transform = read_window(fetch_rasters(year), districts)
    return district_burned_area(arrays, transform, districts, year)


def load_burned_area(seasons: list[int], geo_path=None) -> pd.DataFrame:
    out = pd.concat([mcd64_district_season(y, geo_path) for y in seasons], ignore_index=True)
    return out.sort_values(["season", "district"]).reset_index(drop=True)
