from pathlib import Path

import pandas as pd
import pytest

from psw import config
from psw.audit import sensors
from psw.ingest import firms

FX = Path(__file__).parent / "fixtures"
FIXTURE = {"VIIRS_SNPP": "firms_chunk.csv", "VIIRS_NOAA20": "firms_noaa20.csv",
           "VIIRS_NOAA21": "firms_noaa20.csv", "MODIS": "firms_modis.csv"}


def raw(name):
    return pd.read_csv(FX / name)


@pytest.fixture
def fake_firms(monkeypatch):
    """Each sensor returns its recorded rows on the first chunk of a season and nothing after."""
    asked = []

    def fetch_chunk(source, start, day_range, key):
        asked.append((source, start))
        df = raw(FIXTURE[source.rsplit("_", 1)[0]])
        first = (start.month, start.day) == config.SEASON_START
        exists = not (source.startswith("VIIRS_NOAA21") and start.year < 2023)  # as in the real archive
        return df if exists and first and source.endswith("_SP") else df.iloc[0:0]

    monkeypatch.setattr(firms, "fetch_chunk", fetch_chunk)
    return asked


def test_ist_hour_adds_five_thirty_and_rolls_over():
    utc = pd.Series([0, 429, 430, 717, 1829, 1830, 2055, 2359])  # HHMM
    assert sensors.ist_hour(utc).tolist() == [5, 9, 10, 12, 23, 0, 2, 5]


def test_pass_label_boundaries():
    m, a, n, o = sensors.MORNING, sensors.AFTERNOON, sensors.NIGHT, sensors.OTHER
    hours = pd.Series([23, 0, 3, 4, 9, 10, 11, 12, 15, 16, 21])
    assert sensors.pass_label(hours).tolist() == [o, n, n, o, o, m, m, a, a, o, o]


def test_modis_confidence_uses_the_numeric_scale():
    df = raw("firms_modis.csv")
    kept = sensors.filter_confidence(df, "modis")
    assert kept["confidence"].min() == 30  # 30 stays, 29 and below go
    assert len(df) - len(kept) == int((df["confidence"] < 30).sum()) == 5


def test_viirs_confidence_drops_low_only():
    df = raw("firms_noaa20.csv")
    kept = sensors.filter_confidence(df, "noaa20")
    assert set(kept["confidence"]) == {"n", "h"} and len(df) - len(kept) == 4


def test_normalise_marks_platform_and_night_pass():
    modis = sensors.normalise(raw("firms_modis.csv"), "modis", 2024)
    assert set(modis["platform"]) == {"T", "A"}
    noaa = sensors.normalise(raw("firms_noaa20.csv"), "noaa20", 2024)
    assert noaa["platform"].isna().all()
    night = noaa[noaa["acq_time"] == 2055]  # 20:55 UTC is 02:25 IST the next day
    assert len(night) and (night["ist_hour"] == 2).all() and (night["pass_label"] == sensors.NIGHT).all()


def test_load_multisensor_columns_and_missing_noaa21_2022(fake_firms):
    ms = sensors.load_multisensor([2022, 2023], key="k")
    assert list(ms.columns) == sensors.COLUMNS
    assert set(ms[ms["season"] == 2022]["sensor"]) == {"snpp", "noaa20", "modis"}
    assert set(ms[ms["season"] == 2023]["sensor"]) == set(sensors.SENSORS)
    assert ms["confidence"].map(type).eq(str).all()
    assert ("VIIRS_NOAA21_NRT", firms.season_window(2022)[0]) in fake_firms  # SP empty, so NRT was tried


def test_load_multisensor_fails_loudly_when_a_sensor_is_missing(monkeypatch):
    empty = raw("firms_chunk.csv").iloc[0:0]
    monkeypatch.setattr(firms, "fetch_chunk", lambda *a: empty)
    with pytest.raises(RuntimeError, match="no snpp detections for the 2022"):
        sensors.load_multisensor([2022], key="k")


def test_snpp_rows_match_the_single_sensor_loader(fake_firms):
    ms = sensors.load_multisensor([2024], key="k")
    old = firms.load_all_fires([2024], key="k")
    keys = ["latitude", "longitude", "acq_date", "acq_time", "frp"]
    got = ms[ms["sensor"] == "snpp"][keys].sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(got, old[keys].sort_values(keys).reset_index(drop=True))
