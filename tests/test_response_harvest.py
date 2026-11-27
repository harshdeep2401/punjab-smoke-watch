from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from psw.response import harvest

DISTRICTS = ("A", "B")


def make_daily(counts: dict[int, dict[str, list[int]]]) -> pd.DataFrame:
    """{season: {district: [count on Sep 15, Sep 16, ...]}}"""
    return pd.DataFrame([{"date": date(y, 9, 15) + timedelta(days=i), "season": y, "district": d, "fire_count": n}
                         for y, by in counts.items() for d, series in by.items() for i, n in enumerate(series)])


def random_daily(seed=0, days=20, seasons=(2022, 2023)) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return make_daily({y: {d: [int(x) for x in rng.integers(0, 8, days)] for d in DISTRICTS} for y in seasons})


def row(f, district, day, season=2023):
    d = date(season, 9, 15) + timedelta(days=day)
    return f[(f["district"] == district) & (f["date"] == d)].iloc[0]


def test_proxy_values_on_a_hand_made_season():
    daily = make_daily({2022: {"A": [2, 3, 5], "B": [0, 0, 0]}, 2023: {"A": [1, 3, 0], "B": [4, 0, 9]}})
    f = harvest.proxy_features(daily)
    assert [row(f, "A", i)["season_progress"] for i in range(3)] == pytest.approx([0.1, 0.4, 0.4])
    assert [row(f, "A", i)["last_season_curve_t"] for i in range(3)] == pytest.approx([0.2, 0.5, 1.0])
    assert [row(f, "A", i)["days_since_first_fire"] for i in range(3)] == [-1, 0, 1]
    assert [row(f, "B", i)["days_since_first_fire"] for i in range(3)] == [0, 1, 2]
    assert np.isnan(row(f, "B", 2)["season_progress"])  # no detections last season
    assert np.isnan(row(f, "A", 2, season=2022)["season_progress"])  # no last season at all


def test_proxy_features_use_nothing_after_day_t():
    daily = random_daily()
    cut = date(2023, 9, 22)
    changed = daily.copy()
    changed.loc[changed["date"] > cut, "fire_count"] += 50
    a, b = harvest.proxy_features(daily), harvest.proxy_features(changed)
    pd.testing.assert_frame_equal(a[a["date"] <= cut], b[b["date"] <= cut])


def test_current_season_totals_never_normalise():
    daily = random_daily()
    changed = daily.copy()
    changed.loc[changed["season"] == 2023, "fire_count"] *= 10
    a, b = harvest.proxy_features(daily), harvest.proxy_features(changed)
    pd.testing.assert_series_equal(a["last_season_curve_t"], b["last_season_curve_t"])
    now = a["season"] == 2023
    np.testing.assert_allclose(b.loc[now, "season_progress"], a.loc[now, "season_progress"] * 10)


def test_a_missing_district_day_is_an_error():
    daily = random_daily()
    with pytest.raises(ValueError):
        harvest.proxy_features(daily.drop(daily.index[3]))


def weather(values: list[float], year=2023) -> pd.DataFrame:
    return pd.DataFrame({"date": [date(year, 9, 1) + timedelta(days=i) for i in range(len(values))],
                         "district": "A", "precip_mm": values})


def test_rain_windows():
    f = harvest.rain_features(weather([1, 2, 4, 8, 16, 32, 64]))
    assert f["rain_last3_mm"].tolist()[2:] == [7, 14, 28, 56, 112]
    assert f["rain_next3_mm"].tolist()[:4] == [14, 28, 56, 112]
    assert f["rain_last3_mm"][:2].isna().all() and f["rain_next3_mm"][4:].isna().all()


def test_rain_windows_do_not_cross_years():
    f = harvest.rain_features(pd.concat([weather([1, 1, 1], 2022), weather([5, 5, 5], 2023)]))
    assert f["rain_next3_mm"].isna().all()
    assert f["rain_last3_mm"].dropna().tolist() == [3, 15]


def test_only_the_labelled_rain_feature_looks_ahead():
    w = weather([float(i) for i in range(12)])
    changed = w.copy()
    cut = date(2023, 9, 6)
    changed.loc[changed["date"] > cut, "precip_mm"] += 100
    a, b = harvest.rain_features(w), harvest.rain_features(changed)
    past = a["date"] <= cut
    differs = [c for c in harvest.RAIN_FEATURES if not a.loc[past, c].equals(b.loc[past, c])]
    assert differs == list(harvest.FORECAST_STANDIN)
