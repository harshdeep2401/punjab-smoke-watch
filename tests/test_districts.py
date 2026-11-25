from datetime import date

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from psw.processing import districts


def make_geo(tmp_path):
    g = gpd.GeoDataFrame({"district": ["A", "B"], "geometry": [box(0, 0, 1, 1), box(1, 0, 2, 1)]}, crs="EPSG:4326")
    p = tmp_path / "geo.geojson"
    g.to_file(p, driver="GeoJSON")
    return p


def make_fires(points, d=date(2024, 10, 1)):
    return pd.DataFrame([{"latitude": lat, "longitude": lon, "acq_date": d, "frp": frp, "season": 2024}
                         for lat, lon, frp in points])


def test_assignment_inside_outside_edge(tmp_path):
    fires = make_fires([(0.5, 0.5, 1.0), (0.5, 1.5, 1.0), (0.5, 5.0, 1.0), (0.5, 1.0, 1.0)])
    out = districts.assign_districts(fires, make_geo(tmp_path))
    assert out.loc[0, "district"] == "A" and out.loc[1, "district"] == "B"
    assert pd.isna(out.loc[2, "district"]) and not out.loc[2, "in_punjab"]
    assert pd.isna(out.loc[3, "district"])  # on the shared edge: not 'within' either polygon
    assert out["in_punjab"].tolist() == [True, True, False, False]
    assert len(out) == len(fires)


def test_zero_filled_grid_and_7day_mean(monkeypatch):
    from psw import config
    monkeypatch.setattr(config, "SEASON_START", (10, 1))
    monkeypatch.setattr(config, "SEASON_END", (10, 10))
    fires = pd.concat([make_fires([(0, 0, 2.0)] * 3, date(2024, 10, 2)),
                       make_fires([(0, 0, 1.0)], date(2024, 10, 5))], ignore_index=True)
    fires["district"] = "A"
    out = districts.district_daily(fires, ["A", "B"], [2024])
    assert len(out) == 2 * 10  # every district x every day
    a = out[out.district == "A"].set_index("date")
    assert a.loc[date(2024, 10, 1), "fire_count"] == 0
    assert a.loc[date(2024, 10, 2), "fire_count"] == 3 and a.loc[date(2024, 10, 2), "frp_sum"] == 6.0
    assert (out[out.district == "B"]["fire_count"] == 0).all()
    # trailing 7 days incl. today, first days use what is available
    assert a.loc[date(2024, 10, 1), "fire_count_7d_mean"] == 0
    assert a.loc[date(2024, 10, 5), "fire_count_7d_mean"] == (3 + 1) / 5
    assert a.loc[date(2024, 10, 8), "fire_count_7d_mean"] == 4 / 7
    assert a.loc[date(2024, 10, 9), "fire_count_7d_mean"] == 1 / 7  # Oct 2 dropped out of the window
