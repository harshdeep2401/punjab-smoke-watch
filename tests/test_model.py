import json

import numpy as np
import pandas as pd

from psw.models import evaluate, train
from psw.processing import features as F


def _table(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for season in (2021, 2022, 2023):
        for city in ["Ludhiana", "Amritsar", "Patiala", "Bathinda"]:
            for i in range(60):
                today = 80 + 30 * rng.random()
                r = {"date": (pd.Timestamp(f"{season}-09-15") + pd.Timedelta(days=i)).date(), "city": city,
                     "season": season, F.TARGET: today + rng.normal(0, 10), "pm25_today": today,
                     "pm25_lag1": today + rng.normal(0, 5), "pm25_3d_mean": today, "dow": i % 7, "day_of_season": i}
                for c in F.WEATHER_FEATURES + F.FIRE_FEATURES:
                    r[c] = rng.random()
                rows.append(r)
    t = pd.DataFrame(rows)
    t["city"] = pd.Categorical(t["city"], categories=["Ludhiana", "Amritsar", "Patiala", "Bathinda"])
    return t


def test_quantiles_never_cross_and_seeded():
    t = _table()
    m = train.fit_quantile_models(t, t[F.TARGET], F.MODEL_FEATURES)
    q = train.predict_quantiles(m, t, F.MODEL_FEATURES)
    assert (q["p10"] <= q["p50"]).all() and (q["p50"] <= q["p90"]).all()
    q2 = train.predict_quantiles(train.fit_quantile_models(t, t[F.TARGET], F.MODEL_FEATURES), t, F.MODEL_FEATURES)
    pd.testing.assert_frame_equal(q, q2)


def test_cqr_correction_hits_target_coverage():
    rng = np.random.default_rng(1)
    y = rng.normal(0, 1, 500)
    p10, p90 = np.full(500, -0.2), np.full(500, 0.2)  # far too narrow
    q = train.cqr_correction(p10, p90, y)
    assert q > 0
    assert 0.77 <= train.coverage(p10 - q, p90 + q, y) <= 0.85


def test_backtest_end_to_end(tmp_path):
    t = _table()
    m = evaluate.run_all(t, n_test_seasons=2, out_dir=tmp_path)
    assert set(m["variants"]) == {"full", "no_fire", "persistence"}
    for v in m["variants"].values():
        assert {"mae", "pinball", "coverage80", "band_width"} <= set(v["overall"])
        assert set(v["by_city"]) == {"Ludhiana", "Amritsar", "Patiala", "Bathinda"}
    assert m["seasons_train"] == [2021] and not set(m["seasons_train"]) & set(m["seasons_test"])
    assert m["folds"] == {"2022": [2021], "2023": [2021, 2022]}
    assert m["seasons_test"] == [2022, 2023] and "conformal_applied" in m
    fc = pd.read_parquet(tmp_path / "forecasts.parquet")
    assert list(fc.columns) == ["issue_date", "target_date", "city", "season", "variant", "p10", "p50", "p90", "actual"]
    # test season never in training; every variant uses the same rows
    assert fc.groupby("variant").size().nunique() == 1
    assert set(fc["season"]) == {2022, 2023}
    assert (tmp_path / "model_full_0.5.txt").exists()
    assert json.loads((tmp_path / "metrics.json").read_text())["feature_importance"]
