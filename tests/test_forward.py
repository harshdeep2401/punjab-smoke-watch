"""2026 forward test (docs/FORWARD_TEST_PLAN.md): synthetic data, offline."""
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from psw.response import forward, risk
from tests.test_response_risk import DISTRICTS, make_daily, make_weather

HISTORY = (2023, 2024, 2025)


def detections(sensor="snpp", days=20, per_day=2, season=2026, district="A"):
    return pd.DataFrame([{"latitude": 30.0, "longitude": 75.0, "acq_date": date(season, 9, 15) + timedelta(days=i),
                          "frp": 5.0, "sensor": sensor, "season": season, "district": district}
                         for i in range(days) for _ in range(per_day)])


def test_counts_stop_at_the_end_date_and_never_invent_zeros_after_it():
    end = date(2026, 9, 24)
    c = forward.counts(detections(days=20), "snpp", end, list(DISTRICTS))
    assert pd.to_datetime(c["date"]).dt.date.max() == end  # nothing after the end, not even zero rows
    assert len(c) == 10 * len(DISTRICTS)
    a = c[c["district"] == "A"]["fire_count"]
    assert (a == 2).all() and (c[c["district"] == "B"]["fire_count"] == 0).all()
    scaled = forward.counts(detections(per_day=3), "snpp", end, list(DISTRICTS), scale=0.5)
    assert set(scaled[scaled["district"] == "A"]["fire_count"]) == {2}  # 1.5 rounds to 2
    assert forward.counts(detections(sensor="noaa20"), "snpp", end, list(DISTRICTS))["fire_count"].sum() == 0


def test_the_scaling_ratio_uses_no_2026_data_and_only_punjab():
    rows = ([{"sensor": "snpp", "season": 2024, "district": "A"}] * 9 + [{"sensor": "noaa20", "season": 2024, "district": "A"}] * 10
            + [{"sensor": "snpp", "season": 2026, "district": "A"}] * 100 + [{"sensor": "noaa20", "season": 2026, "district": "A"}]
            + [{"sensor": "snpp", "season": 2024, "district": None}] * 50
            + [{"sensor": "noaa21", "season": 2022, "district": "A"}] * 7)
    assert forward.snpp_ratio(pd.DataFrame(rows)) == pytest.approx(0.9)


def _inputs(days_2026=25, weather_days=25):
    history = make_daily(seasons=HISTORY)
    wx_hist = make_weather(seasons=HISTORY)
    current = make_daily(seed=5, seasons=(2026,), days=days_2026)
    wx_now = make_weather(seed=6, seasons=(2026,), days=weather_days - 3)  # make_weather adds 3 days either side
    wx_now = wx_now[pd.to_datetime(wx_now["date"]).dt.date <= date(2026, 9, 15) + timedelta(days=weather_days - 1)]
    return history, current, wx_hist, wx_now


def test_build_uses_real_weather_for_every_kept_2026_row_and_refuses_data_past_the_weather():
    history, current, wx_hist, wx_now = _inputs()
    table = forward.build(history, current, wx_hist, wx_now)
    kept = table[table["season"] == 2026]
    assert kept["date"].max() == date(2026, 9, 15) + timedelta(days=25 - 1 - risk.HORIZON)
    assert kept["actual"].notna().all()
    short = wx_now[pd.to_datetime(wx_now["date"]).dt.date <= date(2026, 9, 30)]
    with pytest.raises(ValueError, match="weather only to"):
        forward.build(history, current, wx_hist, short)


def test_the_frozen_recipe_trains_on_earlier_seasons_only_and_scores_2026():
    history, current, wx_hist, wx_now = _inputs()
    table = forward.build(history, current, wx_hist, wx_now)
    gaps = pd.DataFrame([{"district": "Punjab", "season": 2025, "gap_low": 0.5, "gap_high": 0.7}])
    fc, conformal = forward.run(table, gaps)
    assert set(fc["season"]) == {2026} and set(fc["variant"]) == set(risk.VARIANTS)
    assert fc["likely_p50"].notna().all()  # the 2025 range, as the planner would have used
    # changing 2026's actuals cannot change the model's 2026 forecasts: it never saw them
    later = current.copy()
    later.loc[pd.to_datetime(later["date"]).dt.date > date(2026, 9, 30), "fire_count"] = 99
    fc2, _ = forward.run(forward.build(history, later, wx_hist, wx_now), gaps)
    early = lambda f: f[(f["variant"] == "model") & (pd.to_datetime(f["issue_date"]).dt.date <= date(2026, 9, 26))]  # noqa: E731
    assert np.allclose(early(fc)["p50"].to_numpy(), early(fc2)["p50"].to_numpy())
    result = forward.evaluate(fc, history, "persistence")
    assert result["issue_days"] == 22 and set(result["allocation_pct_of_oracle"]) >= {"model", "last_season_share"}


def _result(top5=0.6, rho=0.7, p_top5=0.55, alloc=95.0, naive=90.0, cov=0.8):
    return {"variants": {"model": {"top5_hit": top5, "spearman": rho, "coverage80": cov},
                         "persistence": {"top5_hit": p_top5}},
            "allocation_pct_of_oracle": {"model": alloc, "last_season_share": naive}}


def test_criteria_follow_the_numbers_and_the_thresholds_set_in_the_plan():
    assert forward.MIN_TOP5 == pytest.approx(0.526) and forward.MIN_SPEARMAN == pytest.approx(0.637)
    assert [c["met"] for c in forward.criteria(_result())] == [True, True, True, True]
    missed = forward.criteria(_result(top5=0.61, rho=0.55, p_top5=0.7, alloc=88.0, cov=0.95))
    assert [c["met"] for c in missed] == [False, False, False, False]
    assert forward.criteria(_result(top5=float("nan")))[0]["met"] is None


def test_the_same_window_diagnostic_ignores_the_year():
    rows = [{"issue_date": date(s, 9, 15) + timedelta(days=i), "season": s, "variant": "model", "district": d,
             "p50": float(k), "actual": k * 10} for s in (2024, 2025) for i in range(60) for k, d in enumerate(DISTRICTS)]
    out = forward.backtest_window(pd.DataFrame(rows), "2026-09-15", "2026-09-24")
    assert out["2024"]["model"]["n_days"] == 10 and out["2025"]["model"]["spearman"] == pytest.approx(1.0)
