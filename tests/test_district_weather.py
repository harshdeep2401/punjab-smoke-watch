import json
from datetime import date
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import box

from psw import config
from psw.ingest import weather

PAYLOAD = json.loads((Path(__file__).parent / "fixtures" / "openmeteo_district_daily.json").read_text())
START, END = date(2024, 9, 25), date(2024, 9, 29)


@pytest.fixture
def requests_made(monkeypatch):
    """Answers every request with the recorded payload (or whatever the test puts in `.payload`)."""
    class Log(list):
        payload = PAYLOAD

    log = Log()

    def fake(url, params):
        log.append((url, params))
        return log.payload

    monkeypatch.setattr(weather, "_request", fake)
    return log


def fetch(district="SAS Nagar (Mohali)"):
    return weather.fetch_district_weather(district, 30.2, 75.9, START, END)


def test_columns_values_and_cache(requests_made):
    df = fetch()
    again = fetch()
    assert len(requests_made) == 1
    assert requests_made[0][1]["daily"].split(",")[0] == "precipitation_sum"
    assert list(df.columns) == ["date", "district", "precip_mm", "temp_mean", "rh_mean", "wind_speed_mean"]
    assert df["date"].tolist() == [date(2024, 9, d) for d in range(25, 30)]
    assert df["precip_mm"].tolist() == [0.2, 6.2, 0.1, 0.0, 0.1]
    assert df.equals(again)


@pytest.mark.parametrize("break_it", [
    lambda daily: daily["precipitation_sum"].__setitem__(2, None),  # a gap in one value
    lambda daily: [values.pop() for values in daily.values()],  # a day short
])
def test_incomplete_answer_fails_and_is_not_cached(requests_made, break_it):
    payload = json.loads(json.dumps(PAYLOAD))
    break_it(payload["daily"])
    requests_made.payload = payload
    with pytest.raises(RuntimeError, match="incomplete"):
        fetch()
    assert not list((config.RAW / "weather").glob("district_*"))


def test_one_request_per_district_and_season(tmp_path, monkeypatch, requests_made):
    geo = tmp_path / "geo.geojson"
    gpd.GeoDataFrame({"district": ["B", "A"]}, geometry=[box(75.0, 30.0, 75.2, 30.2), box(76.0, 31.0, 76.4, 31.4)],
                     crs="EPSG:4326").to_file(geo, driver="GeoJSON")
    monkeypatch.setattr(weather, "DISTRICT_START", (9, 25))
    monkeypatch.setattr(weather, "DISTRICT_END", (9, 29))
    centroids = weather.district_centroids(geo)
    assert centroids["B"] == pytest.approx((30.1, 75.1), abs=0.01)
    df = weather.district_daily_weather([2024], geo)
    assert len(requests_made) == 2 and len(df) == 10
    assert df["district"].tolist() == ["A"] * 5 + ["B"] * 5
