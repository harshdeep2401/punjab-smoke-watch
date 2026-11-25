import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from psw.ingest import airquality, firms, http, weather

FX = Path(__file__).parent / "fixtures"
KEY = "SECRETKEY123"


class FakeResp:
    def __init__(self, text="", status=200, payload=None):
        self.text, self.status_code, self._payload = text, status, payload

    def json(self):
        return self._payload


class CallLog(list):
    def install(self, responder):
        def fake_get(url, params=None, timeout=None):
            self.append((url, params))
            return responder(url, params)
        http.requests.get = fake_get


@pytest.fixture
def calls(monkeypatch):
    log = CallLog()
    monkeypatch.setattr(http, "BACKOFF", 0)
    monkeypatch.setattr(http.requests, "get", http.requests.get)  # restores the real get after the test

    return log


def test_firms_chunk_parses_and_caches(calls):
    text = (FX / "firms_chunk.csv").read_text()
    calls.install(lambda u, p: FakeResp(text))
    a = firms.fetch_chunk("VIIRS_SNPP_SP", date(2024, 10, 25), 5, KEY)
    b = firms.fetch_chunk("VIIRS_SNPP_SP", date(2024, 10, 25), 5, KEY)
    assert len(calls) == 1 and len(a) == len(b) == 18
    assert {"latitude", "longitude", "acq_date", "acq_time", "frp", "confidence", "daynight"} <= set(a.columns)
    assert KEY not in str(firms._cache_path("VIIRS_SNPP_SP", date(2024, 10, 25), 5))


def test_firms_empty_chunk_is_cached(calls):
    header = (FX / "firms_chunk.csv").read_text().splitlines()[0] + "\n"
    calls.install(lambda u, p: FakeResp(header))
    assert firms.fetch_chunk("VIIRS_SNPP_NRT", date(2022, 9, 15), 5, KEY).empty
    assert firms.fetch_chunk("VIIRS_SNPP_NRT", date(2022, 9, 15), 5, KEY).empty
    assert len(calls) == 1


def test_firms_error_does_not_leak_key(calls):
    calls.install(lambda u, p: FakeResp("Invalid MAP_KEY", 400))
    with pytest.raises(RuntimeError) as e:
        firms.fetch_chunk("VIIRS_SNPP_SP", date(2022, 9, 15), 5, KEY)
    assert KEY not in str(e.value)


def test_firms_season_falls_back_to_nrt_and_filters_low_confidence(calls):
    text = (FX / "firms_chunk.csv").read_text()
    header = text.splitlines()[0] + "\n"
    calls.install(lambda u, p: FakeResp(header if "VIIRS_SNPP_SP" in u else text))
    df = firms.load_all_fires([2026], key=KEY)
    assert set(df["source"]) == {"VIIRS_SNPP_NRT"}
    assert (df["confidence"] != "l").all() and list(df.columns) == firms.KEEP
    assert df["season"].eq(2026).all() and df["acq_date"].map(lambda d: isinstance(d, date)).all()
    assert df.duplicated(firms.DEDUPE).sum() == 0  # same chunk returned for every window is deduped


def test_http_retries_then_raises_without_leaking(calls):
    calls.install(lambda u, p: FakeResp("x", 503))
    with pytest.raises(RuntimeError) as e:
        http.get("https://x/" + KEY, secret=KEY)
    assert len(calls) == http.RETRIES + 1 and KEY not in str(e.value)


def test_aq_daily_requires_18_hours(calls):
    payload = json.loads((FX / "aq_sample.json").read_text())
    calls.install(lambda u, p: FakeResp(payload=payload))
    start, end = date(2024, 10, 10), date(2024, 10, 12)
    hourly = airquality.fetch_city_pm25("Ludhiana", 30.9, 75.85, start, end)
    assert list(hourly.columns) == ["time", "pm2_5"] and len(hourly) == 72
    airquality.fetch_city_pm25("Ludhiana", 30.9, 75.85, start, end)
    assert len(calls) == 1
    daily = airquality.daily_from_hourly(hourly, "Ludhiana")
    assert list(daily.columns) == ["date", "city", "pm25_mean"] and len(daily) == 3
    assert daily["pm25_mean"].notna().all()
    short = hourly.iloc[:30].copy()  # day 1 full (24 h), day 2 only 6 h
    d2 = airquality.daily_from_hourly(short, "Ludhiana")
    assert d2["pm25_mean"].notna().tolist() == [True, False]


def test_weather_daily_columns_and_vector_mean(calls):
    payload = json.loads((FX / "weather_sample.json").read_text())
    calls.install(lambda u, p: FakeResp(payload=payload))
    hourly = weather.fetch_city_weather("Ludhiana", 30.9, 75.85, date(2024, 10, 10), date(2024, 10, 12))
    weather.fetch_city_weather("Ludhiana", 30.9, 75.85, date(2024, 10, 10), date(2024, 10, 12))
    assert len(calls) == 1
    daily = weather.daily_from_hourly(hourly, "Ludhiana")
    assert list(daily.columns) == ["date", "city", "temp_mean", "rh_mean", "wind_speed_mean",
                                   "wind_dir_mean", "u_mean", "v_mean", "blh_mean"]
    assert len(daily) == 3 and daily["wind_dir_mean"].between(0, 360).all() and daily.notna().all().all()


def test_weather_direction_wraparound():
    t = pd.date_range("2024-10-10", periods=2, freq="h")
    h = pd.DataFrame({"time": t, "temperature_2m": 20.0, "relative_humidity_2m": 50.0,
                      "wind_speed_10m": 4.0, "wind_direction_10m": [350.0, 10.0], "boundary_layer_height": 500.0})
    d = float(weather.daily_from_hourly(h, "X")["wind_dir_mean"].iloc[0])
    assert min(d, 360 - d) < 1e-6
