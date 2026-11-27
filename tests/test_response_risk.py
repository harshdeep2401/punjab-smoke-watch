import json
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from psw.response import harvest, risk

DISTRICTS = ("A", "B", "C", "D", "E", "F", "G", "H")
DAYS = 30


def make_daily(seed=0, seasons=(2022, 2023, 2024), days=DAYS) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame([{"date": date(y, 9, 15) + timedelta(days=i), "season": y, "district": d,
                          "fire_count": int(rng.poisson(1 + 2 * k))}
                         for y in seasons for k, d in enumerate(DISTRICTS) for i in range(days)])


def make_weather(seed=1, seasons=(2022, 2023, 2024), days=DAYS) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame([{"date": date(y, 9, 12) + timedelta(days=i), "district": d, "precip_mm": float(rng.random()),
                          "temp_mean": float(25 + rng.random()), "rh_mean": float(60 + rng.random()),
                          "wind_speed_mean": float(5 + rng.random())}
                         for y in seasons for d in DISTRICTS for i in range(days + 6)])


def gaps(*rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["district", "season", "gap_low", "gap_high"])


def at(table, district, day, season=2023):
    d = date(season, 9, 15) + timedelta(days=day)
    return table[(table["district"] == district) & (table["date"] == d)].iloc[0]


def test_target_is_exactly_the_next_three_days():
    daily = make_daily()
    daily.loc[daily["district"] == "A", "fire_count"] = [i for _ in range(3) for i in range(DAYS)]
    t = risk.build_table(daily, make_weather())
    row = at(t, "A", 10)
    assert row["actual"] == 11 + 12 + 13
    assert row[risk.TARGET] == pytest.approx(np.log1p(36))
    assert row["count_today"] == 10 and row["count_3d"] == 8 + 9 + 10 and row["count_7d"] == sum(range(4, 11))
    assert row["count_slope_7d"] == pytest.approx(1.0)
    # the last issue day is the one whose 3-day window still fits the season
    assert t["date"].max() == date(2024, 9, 15) + timedelta(days=DAYS - 1 - risk.HORIZON)


def test_climatology_is_last_seasons_window_scaled_by_the_to_date_ratio():
    daily = make_daily()
    a = daily["district"] == "A"
    daily.loc[a & (daily["season"] == 2022), "fire_count"] = 2
    daily.loc[a & (daily["season"] == 2023), "fire_count"] = 3
    t = risk.build_table(daily, make_weather())
    assert at(t, "A", 5)["climatology"] == pytest.approx(6 * 1.5)
    assert np.isnan(at(t, "A", 5, season=2022)["climatology"])
    share = daily[daily["season"] == 2022].groupby("district")["fire_count"].sum()
    assert at(t, "A", 5)["district_prev_share"] == pytest.approx(share["A"] / share.sum())


def test_no_feature_or_baseline_uses_anything_after_day_t():
    daily, weather = make_daily(), make_weather()
    cut = date(2024, 9, 25)
    daily2, weather2 = daily.copy(), weather.copy()
    daily2.loc[daily2["date"] > cut, "fire_count"] += 40
    for c in ("precip_mm", *risk.WEATHER_FEATURES):
        weather2.loc[weather2["date"] > cut, c] += 100
    a, b = risk.build_table(daily, weather), risk.build_table(daily2, weather2)
    past = a["date"] <= cut
    columns = [c for c in risk.FEATURES if c not in harvest.FORECAST_STANDIN] + ["climatology"]
    pd.testing.assert_frame_equal(a.loc[past, columns], b.loc[past, columns])
    assert not a.loc[past, "rain_next3_mm"].equals(b.loc[past, "rain_next3_mm"])  # the labelled exception


def test_missing_weather_is_an_error():
    weather = make_weather()
    with pytest.raises(ValueError):
        risk.build_table(make_daily(), weather[weather["district"] != "B"])


def test_top5_hit_rate():
    actual = [50, 40, 30, 20, 10, 1, 0]
    assert risk.top_k_hit(actual, [7, 6, 5, 4, 3, 2, 1]) == 1.0
    assert risk.top_k_hit(actual, [7, 6, 5, 4, 1, 3, 2]) == pytest.approx(0.8)  # E (10) is out, F is in
    # a forecast that ties everything gets the expected value of a random pick: 5 of 7
    assert risk.top_k_hit(actual, [0] * 7) == pytest.approx(5 / 7)
    # only two districts burned: they are the whole actual top
    assert risk.top_k_hit([9, 3, 0, 0, 0, 0, 0], [9, 0, 8, 7, 6, 5, 4]) == pytest.approx(0.5)
    assert np.isnan(risk.top_k_hit([0] * 7, actual))


def test_spearman_edge_cases():
    assert risk.spearman([1, 2, 3], [10, 20, 30]) == pytest.approx(1.0)
    assert risk.spearman([1, 2, 3], [5, 5, 5]) == 0.0
    assert np.isnan(risk.spearman([4, 4, 4], [1, 2, 3]))


def test_quiet_days_are_not_ranked():
    day = lambda d, actual, p50: pd.DataFrame({"season": 2023, "issue_date": d, "actual": actual, "p50": p50})  # noqa: E731
    f = pd.concat([day(date(2023, 10, 1), [10, 9, 8, 7, 6, 0], [6, 5, 4, 3, 2, 1]),
                   day(date(2023, 10, 2), [3, 0, 0, 0, 0, 0], [0, 1, 2, 3, 4, 5])])  # 3 detections: quiet
    m = risk.ranking_metrics(f)
    assert m["n_days"] == 1 and m["top5_hit"] == 1.0 and m["spearman"] == pytest.approx(1.0)


def forecast_rows(districts, season=2025, p50=10.0) -> pd.DataFrame:
    return pd.DataFrame({"issue_date": date(season, 10, 1), "district": list(districts), "season": season,
                         "variant": "model", "p10": 0.0, "p50": p50, "p90": 20.0, "actual": 5})


def test_likely_burning_uses_the_previous_seasons_gap():
    g = gaps(("A", 2024, 0.4, 0.6), ("A", 2025, 0.8, 0.9), ("Punjab", 2024, 0.2, 0.2), ("Punjab", 2025, 0.7, 0.7),
             ("B", 2024, np.nan, np.nan), ("C", 2024, -2.0, -0.9), ("D", 2024, 0.95, 0.99))
    out = risk.add_likely(forecast_rows("ABCD"), g).set_index("district")
    assert out.loc["A", "likely_p50"] == pytest.approx(20.0)
    assert out.loc["A", "likely_low"] == pytest.approx(10 / 0.6, abs=0.01)
    assert out.loc["A", "likely_high"] == pytest.approx(25.0)
    assert out.loc["A", "gap_source"] == "district"
    assert out.loc["B", "gap_source"] == "Punjab" and out.loc["B", "likely_p50"] == pytest.approx(12.5)
    assert out.loc["C", "likely_p50"] == 10.0  # negative gap: never plan for less than detected
    assert out.loc["D", "likely_high"] == pytest.approx(100.0)  # capped at 10 times
    assert list(out["p50"]) == [10.0] * 4  # the forecast itself is untouched


def test_likely_burning_without_a_previous_gap_is_empty():
    out = risk.add_likely(forecast_rows("A", season=2023), gaps(("A", 2023, 0.4, 0.6)))
    assert out["likely_p50"].isna().all() and out["gap_source"].isna().all()


def test_run_all_writes_outputs_and_the_gap_never_touches_the_scores(tmp_path):
    four = (2022, 2023, 2024, 2025)  # climatology needs a previous season inside the training years
    daily, weather = make_daily(seasons=four), make_weather(seasons=four)
    m = risk.run_all(daily, weather, gaps(("Punjab", 2023, 0.5, 0.5), ("Punjab", 2024, 0.5, 0.5)), out_dir=tmp_path)
    m2 = risk.run_all(daily, weather, gaps(("Punjab", 2023, 0.1, 0.1), ("Punjab", 2024, 0.1, 0.1)),
                      out_dir=tmp_path / "other")
    assert m["variants"] == m2["variants"]
    f = pd.read_parquet(tmp_path / risk.FORECAST_FILE)
    assert list(f.columns) == risk.FORECAST_COLUMNS
    assert sorted(f["season"].unique()) == [2024, 2025] and set(f["variant"]) == set(risk.VARIANTS)
    assert len(f) == 2 * len(risk.VARIANTS) * len(DISTRICTS) * (DAYS - risk.HORIZON)
    assert (f["p10"] <= f["p50"]).all() and (f["p50"] <= f["p90"]).all() and (f["p10"] >= 0).all()
    assert m["best_ranking_variant"] in risk.VARIANTS
    assert json.loads((tmp_path / risk.METRICS_FILE).read_text())["folds"] == {"2024": [2022, 2023], "2025": [2022, 2023, 2024]}


def test_forecasts_up_to_day_t_ignore_later_data():
    daily, weather = make_daily(), make_weather()
    cut = date(2024, 9, 25)
    changed = daily.copy()
    changed.loc[changed["date"] > cut, "fire_count"] += 40
    a = risk.run_backtest(risk.build_table(daily, weather), [2024])[0]
    b = risk.run_backtest(risk.build_table(changed, weather), [2024])[0]
    past = a["issue_date"] <= cut
    pd.testing.assert_frame_equal(a.loc[past, ["p10", "p50", "p90"]], b.loc[past, ["p10", "p50", "p90"]])
