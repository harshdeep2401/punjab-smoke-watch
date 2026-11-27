import re
from datetime import date

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from psw import config
from psw.response import grid as g


@pytest.fixture(scope="module")
def punjab():
    return g.build_grid()


@pytest.fixture
def two_districts(tmp_path):
    """Two 20 km x 20 km squares side by side, aligned to the 5 km lines, with the border inside a cell."""
    x, y = 500_000, 3_400_000
    shapes = [box(x, y, x + 22_000, y + 20_000), box(x + 22_000, y, x + 40_000, y + 20_000)]
    path = tmp_path / "two.geojson"
    gpd.GeoDataFrame({"district": ["West", "East"]}, geometry=shapes, crs=g.METRIC_CRS).to_crs("EPSG:4326") \
        .to_file(path, driver="GeoJSON")
    return g.build_grid(path)


def fire(x, y, district, day=date(2025, 10, 20)) -> dict:
    p = gpd.GeoSeries(gpd.points_from_xy([x], [y]), crs=g.METRIC_CRS).to_crs("EPSG:4326")
    return {"latitude": p.y.iloc[0], "longitude": p.x.iloc[0], "district": district, "acq_date": day}


def test_cells_cover_punjab(punjab):
    area = punjab.districts.union_all().area / 1e6
    cells = punjab.cells
    assert cells["area_km2"].sum() == pytest.approx(area, rel=0.002)
    assert area / 25 <= len(cells) <= area / 25 * 1.25  # whole cells plus the clipped ones along the border
    assert cells["area_km2"].max() <= 25.01
    assert cells["cell_id"].is_unique and not cells.duplicated(["row", "col"]).any()
    assert set(cells["district"]) == set(punjab.districts.index)  # every cell has one, and every district has cells


def test_a_border_cell_belongs_to_the_district_covering_most_of_it(two_districts):
    cells = two_districts.cells
    assert len(cells) == 8 * 4  # 40 km x 20 km
    border = cells[cells["col"] == (500_000 + 20_000) // g.CELL_M]  # 2 km West, 3 km East
    assert set(border["district"]) == {"East"}


def test_hotspots_count_the_last_seven_days_of_the_districts_own_detections(two_districts):
    x, y = 500_000, 3_400_000
    rows = ([fire(x + 1_000, y + 1_000, "West")] * 3 + [fire(x + 11_000, y + 11_000, "West")]
            + [fire(x + 1_000, y + 1_000, "West", date(2025, 10, 14))] * 2  # first day of the window
            + [fire(x + 1_000, y + 1_000, "West", date(2025, 10, 13))] * 9  # too old
            + [fire(x + 11_000, y + 11_000, "West", date(2025, 10, 21))] * 9  # after the date asked about
            + [fire(x + 35_000, y + 18_000, "East")])
    out = g.hotspot_cells("West", "2025-10-20", pd.DataFrame(rows), two_districts)
    assert [c["detections_7d"] for c in out] == [5, 1]
    assert [c["share_pct"] for c in out] == [83.3, 16.7]
    assert out[0]["label"] == f"cell {out[0]['cell_id']} (south-west West)"
    assert out[1]["label"].endswith("(central West)")
    assert len(g.hotspot_cells("West", date(2025, 10, 20), pd.DataFrame(rows), two_districts, top=1)) == 1


def test_no_detections_means_no_hotspots_and_unknown_districts_fail(two_districts):
    fires = pd.DataFrame([fire(501_000, 3_401_000, "West")])
    assert g.hotspot_cells("East", "2025-10-20", fires, two_districts) == []
    assert g.hotspot_cells("West", "2025-09-20", fires, two_districts) == []
    with pytest.raises(ValueError):
        g.hotspot_cells("Nowhere", "2025-10-20", fires, two_districts)


def test_committed_detections_all_land_in_a_cell_and_labels_name_nothing_below_a_district(punjab):
    fires = pd.read_parquet(config.PROCESSED / "fires.parquet")
    found = g.cell_detections(fires, punjab, "2025-11-30", days=2000)  # every season at once
    assert found["detections"].sum() == int(fires["district"].notna().sum())
    pattern = rf"cell \d+ \((central|{'|'.join(g.COMPASS)}) Sangrur\)"
    out = g.hotspot_cells("Sangrur", "2025-10-31", fires, punjab)
    assert out and all(re.fullmatch(pattern, c["label"]) for c in out)
    assert all(set(c) == {"cell_id", "label", "lat", "lon", "detections_7d", "share_pct"} for c in out)
    assert all(round(c["lat"], 2) == c["lat"] and round(c["lon"], 2) == c["lon"] for c in out)
