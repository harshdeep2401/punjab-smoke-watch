import math

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from psw import config
from psw.audit import burned_area as ba

RES = 0.01
TRANSFORM = from_origin(75.0, 30.03, RES, RES)  # 3 x 3 pixels: lon 75.00-75.03, lat 30.00-30.03
YEAR = 2023
LO, HI = 258, 334  # 15 Sep and 30 Nov 2023 as days of year


def pixel_km2(row: int) -> float:
    lat = 30.03 - (row + 0.5) * RES
    return (RES * 111.195) ** 2 * math.cos(math.radians(lat))


def two_districts() -> gpd.GeoDataFrame:
    """West takes columns 0 and 1, East takes column 2."""
    return gpd.GeoDataFrame({"district": ["West", "East"]},
                            geometry=[box(75.0, 30.0, 75.02, 30.03), box(75.02, 30.0, 75.03, 30.03)],
                            crs="EPSG:4326")


def month(cells: dict[tuple[int, int], int]) -> np.ndarray:
    arr = np.zeros((3, 3), dtype="int16")
    for rc, value in cells.items():
        arr[rc] = value
    return arr


def test_known_area_per_district():
    monthly = [
        month({(0, 0): LO, (1, 1): LO - 1, (2, 2): -2}),   # in window, day before it, water
        month({(1, 0): 290, (0, 2): 300}),
        month({(2, 1): HI, (2, 0): HI + 1, (1, 2): -1}),   # last day, day after it, unmapped
    ]
    out = ba.district_burned_area(monthly, TRANSFORM, two_districts(), YEAR).set_index("district")
    assert out.loc["West", "n_pixels"] == 3  # (0,0), (1,0), (2,1)
    assert out.loc["West", "burned_km2"] == pytest.approx(pixel_km2(0) + pixel_km2(1) + pixel_km2(2))
    assert out.loc["East", "n_pixels"] == 1
    assert out.loc["East", "burned_km2"] == pytest.approx(pixel_km2(0))
    assert list(out["season"]) == [YEAR, YEAR]


def test_pixel_burned_in_two_months_counts_once_and_empty_district_is_zero():
    monthly = [month({(0, 0): 270}), month({(0, 0): 300}), month({})]
    out = ba.district_burned_area(monthly, TRANSFORM, two_districts(), YEAR).set_index("district")
    assert out.loc["West", "n_pixels"] == 1
    assert out.loc["East", "n_pixels"] == 0
    assert out.loc["East", "burned_km2"] == 0


def test_file_names_follow_day_of_year_including_leap_years():
    assert ba.file_name(2023, 9) == "MCD64monthly.A2023244.Win18.061.burndate.tif"
    assert ba.file_name(2024, 9) == "MCD64monthly.A2024245.Win18.061.burndate.tif"


def write_raster(path, arr):
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", height=3, width=3, count=1, dtype="int16",
                       crs="EPSG:4326", transform=TRANSFORM) as dst:
        dst.write(arr, 1)


def test_cached_rasters_are_read_without_connecting(tmp_path, monkeypatch):
    def no_network(*_args):
        raise AssertionError("downloaded although cached")

    monkeypatch.setattr(ba, "_download", no_network)
    for m, arr in zip(ba.MONTHS, (month({(0, 0): 270}), month({(1, 2): 300}), month({}))):
        write_raster(config.RAW / "burned" / "mcd64_win18" / ba.file_name(YEAR, m), arr)
    geo = tmp_path / "geo.geojson"
    two_districts().to_file(geo, driver="GeoJSON")
    out = ba.mcd64_district_season(YEAR, geo).set_index("district")
    assert out["n_pixels"].to_dict() == {"West": 1, "East": 1}
    assert out.loc["East", "burned_km2"] == pytest.approx(pixel_km2(1))


def test_only_missing_months_are_downloaded_and_a_failure_is_loud(monkeypatch):
    asked = []

    def fake_download(year, paths):
        asked.extend(p.name for p in paths)
        raise RuntimeError(f"could not download MCD64A1 file {paths[0].name}")

    monkeypatch.setattr(ba, "_download", fake_download)
    write_raster(config.RAW / "burned" / "mcd64_win18" / ba.file_name(YEAR, 9), month({}))
    with pytest.raises(RuntimeError, match="could not download"):
        ba.fetch_rasters(YEAR)
    assert asked == [ba.file_name(YEAR, 10), ba.file_name(YEAR, 11)]
