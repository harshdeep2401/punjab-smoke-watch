from datetime import date, timedelta
from itertools import product

import numpy as np
import pandas as pd
import pytest

from psw.response import allocate as al


def risk(p50: dict, likely: dict | None = None) -> pd.DataFrame:
    out = pd.DataFrame({"district": list(p50), "p50": list(p50.values())})
    if likely is not None:
        out["likely_p50"] = [likely.get(d, np.nan) for d in p50]
    return out


def units_of(table: pd.DataFrame) -> dict:
    return dict(zip(table["district"], table["units"]))


SPREAD = {"A": 100.0, "B": 60.0, "C": 30.0, "D": 10.0, "E": 2.0, "F": 0.0}


def test_totals_limits_and_zero_risk():
    out = al.allocate(risk(SPREAD), 200, max_share=0.25)
    u = units_of(out)
    assert sum(u.values()) == 200 and max(u.values()) <= 50 and u["F"] == 0
    assert u["A"] >= u["B"] >= u["C"] >= u["D"] >= u["E"]
    assert list(out.columns) == al.COLUMNS and list(out["units"]) == sorted(out["units"], reverse=True)
    assert "limit of 50 units" in out.set_index("district").loc["A", "reason"]
    assert out.set_index("district").loc["F", "reason"] == "no burning expected in the next 3 days"


def test_diminishing_returns_spread_the_units():
    u = units_of(al.allocate(risk({"A": 10.0, "B": 9.0, "C": 8.0}), 60, max_share=1.0))
    assert min(u.values()) > 0 and u["A"] > u["B"] > u["C"]  # not all to the top district
    assert units_of(al.allocate(risk({"B": 5.0, "A": 5.0}), 7, max_share=1.0)) == {"A": 4, "B": 3}  # tie: by name


def test_greedy_is_the_best_split():
    values = {"A": 9.0, "B": 4.0, "C": 1.0}
    got = units_of(al.allocate(risk(values), 12, k=5.0, max_share=1.0))
    total = lambda u: sum(values[d] * al.cover(n, 5.0) for d, n in u.items())  # noqa: E731
    best = max(total({"A": a, "B": b, "C": 12 - a - b}) for a, b in product(range(13), range(13)) if a + b <= 12)
    assert total(got) == pytest.approx(best)


def test_minimum_per_district():
    u = units_of(al.allocate(risk(SPREAD), 100, min_per_district=3))
    assert sum(u.values()) == 100 and u["F"] == 3 and min(u.values()) >= 3
    with pytest.raises(ValueError):
        al.allocate(risk(SPREAD), 10, min_per_district=3)


def test_units_stay_unplaced_when_every_burning_district_is_full():
    u = units_of(al.allocate(risk({"A": 5.0, "B": 0.0, "C": 0.0, "D": 0.0}), 100, max_share=0.25))
    assert u == {"A": 25, "B": 0, "C": 0, "D": 0}


def test_same_answer_whatever_the_row_order():
    table = risk(SPREAD)
    a, b = al.allocate(table, 137), al.allocate(table.iloc[::-1], 137)
    pd.testing.assert_frame_equal(a, b)


def test_likely_burning_is_used_where_it_exists():
    table = risk({"A": 10.0, "B": 10.0}, likely={"A": 40.0})
    out = al.allocate(table, 20, max_share=1.0).set_index("district")
    assert list(out.loc[["A", "B"], "basis"]) == [al.LIKELY, al.DETECTED]
    assert out.loc["A", "units"] > out.loc["B", "units"]
    assert "likely burning 40.0" in out.loc["A", "reason"] and "forecast detections 10.0" in out.loc["B", "reason"]
    assert units_of(al.allocate(table, 20, max_share=1.0, use_likely=False)) == {"A": 10, "B": 10}
    moved = al.compare_to_detected(table, 20, max_share=1.0).set_index("district")
    assert moved.loc["A", "moved"] == moved.loc["A", "units_likely"] - 10 > 0
    assert moved["moved"].sum() == 0


def test_bad_inputs():
    with pytest.raises(ValueError):
        al.allocate(risk({"A": np.nan}), 5)
    with pytest.raises(ValueError):
        al.allocate(pd.concat([risk({"A": 1.0}), risk({"A": 2.0})]), 5)
    with pytest.raises(ValueError):
        al.allocate(risk(SPREAD), 5, max_share=0)


def test_proportional_split():
    w = pd.Series({"A": 3.0, "B": 1.0, "C": 1.0, "D": 1.0})
    got = al.proportional(w, 10)
    assert got.sum() == 10 and got["A"] == 5
    assert list(al.proportional(pd.Series({"A": 0.0, "B": 0.0}), 5)) == [3, 2]
    table = al.compare_to_proportional(al.allocate(risk(SPREAD), 100), pd.Series({"A": 1.0, "B": 3.0}))
    assert table["proportional_units"].sum() == 100
    assert table.set_index("district").loc["B", "proportional_units"] == 75
    assert (table["difference"] == table["units"] - table["proportional_units"]).all()


def synthetic_backtest(seed=0, days=15):
    """Forecasts of three qualities: exact, noisy and random. Last season is unrelated to this one."""
    rng = np.random.default_rng(seed)
    districts = list("ABCDEFGH")
    daily = pd.DataFrame([{"date": date(2023, 9, 15), "season": 2023, "district": d, "fire_count": int(rng.integers(1, 50))}
                          for d in districts])
    rows = []
    for i in range(days):
        actual = rng.poisson([1, 2, 4, 8, 16, 32, 64, 128])
        rng.shuffle(actual)
        guesses = {"model": actual, "persistence": actual * rng.lognormal(0, 0.7, 8), "climatology": rng.random(8) * 128}
        for variant, p50 in guesses.items():
            rows.append(pd.DataFrame({"issue_date": date(2024, 10, 1) + timedelta(days=i), "district": districts,
                                      "season": 2024, "variant": variant, "p50": p50.astype(float),
                                      "likely_p50": p50 * 2.0, "actual": actual}))
    return pd.concat(rows, ignore_index=True), daily


def test_decision_backtest_orders_the_strategies():
    forecasts, daily = synthetic_backtest()
    scores = al.daily_scores(forecasts, daily, units=100)
    for name in ("model", "persistence", "climatology", "last_season_share", "equal_split"):
        assert (scores["oracle"] >= scores[name] - 1e-9).all()
    d = al.decision_backtest(forecasts, daily, units=100)
    pct = d["pct_of_oracle"]
    assert pct["model"] == 100.0  # an exact forecast places the units exactly as the oracle does
    assert pct["model"] > pct["persistence"] > pct["climatology"]
    assert d["n_days"] == 15 and d["by_season"]["2024"] == pct
    assert "model_likely" not in pct and d["likely_check"]["pct_of_oracle"] == 100.0  # a uniform factor changes nothing


def test_decision_backtest_needs_the_previous_season():
    forecasts, daily = synthetic_backtest()
    with pytest.raises(ValueError):
        al.decision_backtest(forecasts, daily.assign(season=2020))
