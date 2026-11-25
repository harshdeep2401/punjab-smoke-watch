"""Feature engineering: upwind fire pressure (D4) and the model feature table.

Every feature for day t uses data up to and including t only. The target is the mean PM2.5 on t+1.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from psw import config

EARTH_KM = 6371.0088

FIRE_FEATURES = [
    "upwind_frp_l0", "upwind_frp_l1", "upwind_frp_l2",
    "frp_300km_l0", "punjab_fire_count_l0", "punjab_fire_count_3d",
]
WEATHER_FEATURES = ["temp_mean", "rh_mean", "wind_speed_mean", "u_mean", "v_mean", "blh_mean"]
PERSISTENCE_FEATURES = ["pm25_today", "pm25_lag1", "pm25_3d_mean"]
CALENDAR_FEATURES = ["dow", "day_of_season", "city"]
MODEL_FEATURES = PERSISTENCE_FEATURES + WEATHER_FEATURES + FIRE_FEATURES + CALENDAR_FEATURES
TARGET = "target_pm25_next"


def _haversine_bearing(lat1, lon1, lat2, lon2):
    """Distance (km) and initial bearing (deg, clockwise from north) from point 1 to point 2. Vectorised."""
    p1, p2 = np.deg2rad(lat1), np.deg2rad(lat2)
    dlon = np.deg2rad(np.asarray(lon2) - np.asarray(lon1))
    dlat = p2 - p1
    a = np.sin(dlat / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlon / 2) ** 2
    dist = 2 * EARTH_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    y = np.sin(dlon) * np.cos(p2)
    x = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dlon)
    return dist, np.mod(np.rad2deg(np.arctan2(y, x)), 360.0)


def _weights(lat, lon, wind_from_deg, city_latlon, max_km: float, decay_km: float):
    """D4 weight per fire: max(0, cos(theta)) * exp(-d/decay), zero beyond max_km.

    theta = angle between the wind's *to* direction (from + 180) and the bearing fire -> city.
    `wind_from_deg` may be a scalar or an array aligned with the fires. Returns (weights, distance_km).
    """
    clat, clon = city_latlon
    dist, bearing = _haversine_bearing(np.asarray(lat), np.asarray(lon), clat, clon)
    theta = np.deg2rad(np.mod(np.asarray(wind_from_deg) + 180.0, 360.0) - bearing)
    w = np.maximum(0.0, np.cos(theta)) * np.exp(-dist / decay_km)
    return np.where(dist <= max_km, w, 0.0), dist


def upwind_pressure(fires_day: pd.DataFrame, city_latlon, wind_dir_from_deg: float,
                    max_km: float = 300, decay_km: float = 150) -> float:
    """Sum of FRP * weight over one day's fires within `max_km` of the city (D4)."""
    if fires_day is None or len(fires_day) == 0 or not np.isfinite(wind_dir_from_deg):
        return 0.0
    w, _ = _weights(fires_day["latitude"].to_numpy(), fires_day["longitude"].to_numpy(),
                    wind_dir_from_deg, city_latlon, max_km, decay_km)
    return float((fires_day["frp"].to_numpy() * w).sum())


def upwind_contributors(fires_day: pd.DataFrame, city_latlon, wind_dir_from_deg: float,
                        top: int = 3, max_km: float = 300, decay_km: float = 150) -> list[dict]:
    """Top districts by share of the weighted FRP: [{district, share_pct}]. For the agent tool, not a model feature.

    Fires outside the 22 districts are grouped as 'Outside Punjab'.
    """
    if fires_day is None or len(fires_day) == 0 or not np.isfinite(wind_dir_from_deg):
        return []
    w, _ = _weights(fires_day["latitude"].to_numpy(), fires_day["longitude"].to_numpy(),
                    wind_dir_from_deg, city_latlon, max_km, decay_km)
    contrib = pd.DataFrame({
        "district": fires_day["district"].fillna("Outside Punjab").to_numpy(),
        "wfrp": fires_day["frp"].to_numpy() * w,
    })
    total = contrib["wfrp"].sum()
    if total <= 0:
        return []
    by = contrib.groupby("district")["wfrp"].sum().sort_values(ascending=False).head(top)
    return [{"district": d, "share_pct": round(float(v / total * 100), 1)} for d, v in by.items()]


def _as_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s).dt.date


def _daily_fire_features(fires: pd.DataFrame, wind: pd.DataFrame, city: str, days: pd.DatetimeIndex) -> pd.DataFrame:
    """Per-day (index = days) same-day fire aggregates for one city, using that day's wind for the upwind sum."""
    f = fires[["latitude", "longitude", "frp", "acq_date", "in_punjab"]].copy()
    f["date"] = pd.to_datetime(f["acq_date"])
    wdir = wind.set_index("date")["wind_dir"]
    f["wind_dir"] = f["date"].map(wdir)
    f = f[f["wind_dir"].notna()]
    w, dist = _weights(f["latitude"].to_numpy(), f["longitude"].to_numpy(), f["wind_dir"].to_numpy(),
                       config.CITIES[city], 300, 150)
    f["wfrp"] = f["frp"].to_numpy() * w
    f["frp300"] = np.where(dist <= 300, f["frp"], 0.0)
    g = f.groupby("date").agg(upwind_frp=("wfrp", "sum"), frp_300km=("frp300", "sum"))
    out = g.reindex(days, fill_value=0.0)
    return out


def build_features(fires: pd.DataFrame, pm25_daily: pd.DataFrame, weather_daily: pd.DataFrame) -> pd.DataFrame:
    """One row per (city, date in the season window) with the target (next-day PM2.5).

    Fire data exist only inside the season window, so lagged fire features on the first days of a season
    treat the missing days before Sep 15 as zero fires.
    """
    fires = fires.copy()
    pm = pm25_daily.copy()
    wx = weather_daily.copy()
    pm["date"] = pd.to_datetime(pm["date"])
    wx["date"] = pd.to_datetime(wx["date"])
    wx["wind_dir"] = _wind_dir(wx)

    all_days = pd.date_range(min(pm["date"].min(), wx["date"].min()), max(pm["date"].max(), wx["date"].max()), freq="D")
    punjab = fires[fires["in_punjab"]].copy()
    punjab["date"] = pd.to_datetime(punjab["acq_date"])
    pj_count = punjab.groupby("date").size().reindex(all_days, fill_value=0)

    rows = []
    for city in config.CITIES:
        pmc = pm[pm["city"] == city].set_index("date")["pm25_mean"].reindex(all_days)
        wxc = wx[wx["city"] == city].sort_values("date")
        wxi = wxc.set_index("date").reindex(all_days)
        fd = _daily_fire_features(fires, wxc[["date", "wind_dir"]], city, all_days)

        df = pd.DataFrame(index=all_days)
        df["city"] = city
        df["pm25_today"] = pmc
        df["pm25_lag1"] = pmc.shift(1)
        df["pm25_3d_mean"] = pmc.rolling(3, min_periods=3).mean()
        df[TARGET] = pmc.shift(-1)
        for c in WEATHER_FEATURES:
            df[c] = wxi[c]
        df["upwind_frp_l0"] = fd["upwind_frp"]
        df["upwind_frp_l1"] = fd["upwind_frp"].shift(1)
        df["upwind_frp_l2"] = fd["upwind_frp"].shift(2)
        df["frp_300km_l0"] = fd["frp_300km"]
        df["punjab_fire_count_l0"] = pj_count
        df["punjab_fire_count_3d"] = pj_count.rolling(3, min_periods=1).sum()
        rows.append(df.rename_axis("date").reset_index())

    out = pd.concat(rows, ignore_index=True)
    out["season"] = out["date"].dt.year
    start = out["season"].map(lambda y: pd.Timestamp(date(y, *config.SEASON_START)))
    end = out["season"].map(lambda y: pd.Timestamp(date(y, *config.SEASON_END)))
    out = out[(out["date"] >= start) & (out["date"] <= end)]
    out["dow"] = out["date"].dt.day_of_week
    out["day_of_season"] = (out["date"] - out["season"].map(lambda y: pd.Timestamp(date(y, *config.SEASON_START)))).dt.days
    out["date"] = out["date"].dt.date
    # fires/lags before Sep 15 are zero by construction; lag-1/2 of fire features at Sep 15 are therefore 0, not NaN
    for c in ("upwind_frp_l1", "upwind_frp_l2"):
        out[c] = out[c].fillna(0.0)
    out = out.dropna(subset=[TARGET, "pm25_today", "pm25_lag1", "pm25_3d_mean"] + WEATHER_FEATURES)
    out["city"] = pd.Categorical(out["city"], categories=list(config.CITIES))
    cols = ["date", "city", "season", TARGET] + PERSISTENCE_FEATURES + WEATHER_FEATURES + FIRE_FEATURES + ["dow", "day_of_season"]
    return out[cols].sort_values(["city", "date"]).reset_index(drop=True)


def _wind_dir(wx: pd.DataFrame) -> pd.Series:
    """Daily wind-from direction from the vector-mean (u, v), which is correct across the 0/360 wrap."""
    from psw.processing.wind import dir_from_uv
    return pd.Series(dir_from_uv(wx["u_mean"].to_numpy(), wx["v_mean"].to_numpy()), index=wx.index)
