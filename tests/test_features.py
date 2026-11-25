import numpy as np
import pandas as pd
import pytest

from psw import config
from psw.processing import features as F

CITY = (30.90, 75.85)


def _fire(lat, lon, frp=10.0, district="X"):
    return pd.DataFrame({"latitude": [lat], "longitude": [lon], "frp": [frp], "district": [district]})


def _offset_north(km):  # a point `km` north of the city
    return CITY[0] + km / 111.195, CITY[1]


def test_directly_upwind_weight():
    lat, lon = _offset_north(100)  # fire north of city; wind FROM north (0 deg) blows toward the city
    v = F.upwind_pressure(_fire(lat, lon), CITY, 0.0)
    assert v == pytest.approx(10 * np.exp(-100 / 150), rel=0.01)


def test_directly_downwind_is_zero():
    lat, lon = _offset_north(100)
    assert F.upwind_pressure(_fire(lat, lon), CITY, 180.0) == 0.0


def test_crosswind_is_about_zero():
    lat, lon = _offset_north(100)
    assert F.upwind_pressure(_fire(lat, lon), CITY, 90.0) == pytest.approx(0.0, abs=0.05)


def test_beyond_300km_excluded():
    lat, lon = _offset_north(310)
    assert F.upwind_pressure(_fire(lat, lon), CITY, 0.0) == 0.0
    assert F.upwind_pressure(_fire(*_offset_north(290)), CITY, 0.0) > 0


def test_empty_and_nan_wind():
    assert F.upwind_pressure(_fire(31, 76).iloc[:0], CITY, 0.0) == 0.0
    assert F.upwind_pressure(_fire(31, 76), CITY, float("nan")) == 0.0


def test_contributors_shares():
    a = _fire(*_offset_north(50), frp=30, district="A")
    b = _fire(*_offset_north(50), frp=10, district="B")
    out = F.upwind_contributors(pd.concat([a, b]), CITY, 0.0, top=2)
    assert out[0] == {"district": "A", "share_pct": 75.0}
    assert out[1]["district"] == "B"


def _synthetic(n_days_after=0, seed=0):
    rng = np.random.default_rng(seed)
    fires, pm, wx = [], [], []
    for season in (2022, 2023):
        days = pd.date_range(f"{season}-09-01", f"{season}-12-01")
        for d in days:
            if pd.Timestamp(f"{season}-09-15") <= d <= pd.Timestamp(f"{season}-11-30"):
                for _ in range(3):
                    lat, lon = _offset_north(rng.uniform(10, 200))
                    fires.append({"latitude": lat, "longitude": lon + rng.normal(0, .1), "frp": rng.uniform(1, 50),
                                  "acq_date": d.date(), "in_punjab": True, "district": "X", "season": season})
            for city in config.CITIES:
                pm.append({"date": d.date(), "city": city, "pm25_mean": 50 + 40 * rng.random()})
                wx.append({"date": d.date(), "city": city, "temp_mean": 20., "rh_mean": 60., "wind_speed_mean": 3.,
                           "wind_dir_mean": 0., "u_mean": rng.normal(), "v_mean": rng.normal() - 1, "blh_mean": 300.})
    return pd.DataFrame(fires), pd.DataFrame(pm), pd.DataFrame(wx)


def test_feature_table_shape_and_target():
    fires, pm, wx = _synthetic()
    t = F.build_features(fires, pm, wx)
    assert set(F.FIRE_FEATURES) <= set(t.columns) and F.TARGET in t.columns
    assert t["date"].min().month == 9 and t["date"].min().day == 15
    assert t[F.MODEL_FEATURES + [F.TARGET]].isna().sum().sum() == 0
    # target is t+1 PM2.5
    row = t[(t["city"] == "Ludhiana")].iloc[5]
    nxt = pm[(pm["city"] == "Ludhiana") & (pm["date"] == row["date"] + pd.Timedelta(days=1))]["pm25_mean"].iloc[0]
    assert row[F.TARGET] == pytest.approx(nxt)


def test_no_leakage_features_at_t_ignore_later_data():
    fires, pm, wx = _synthetic()
    base = F.build_features(fires, pm, wx)
    cutoff = pd.Timestamp("2023-10-20").date()
    fires2, pm2, wx2 = fires.copy(), pm.copy(), wx.copy()
    # corrupt everything strictly after the cutoff (fires, weather) and PM2.5 after cutoff + 1 (the t+1 target of cutoff)
    fires2.loc[fires2["acq_date"] > cutoff, "frp"] *= 100
    wx2.loc[wx2["date"] > cutoff, ["u_mean", "v_mean", "temp_mean"]] += 50
    pm2.loc[pm2["date"] > cutoff, "pm25_mean"] += 500
    alt = F.build_features(fires2, pm2, wx2)
    feats = [c for c in F.MODEL_FEATURES if c != "city"]
    a = base[base["date"] <= cutoff].set_index(["city", "date"])[feats]
    b = alt[alt["date"] <= cutoff].set_index(["city", "date"])[feats]
    pd.testing.assert_frame_equal(a, b)
