"""3-day district fire-risk forecast (D20): features, expanding-window backtest, baselines, ranking metrics.

One row per district and issue day t. The target is log1p of the S-NPP detections in t+1..t+3. Only S-NPP is used
because NOAA-21 has no 2022 data, so an all-VIIRS sum would not compare across seasons. Quantiles are fitted on the
log scale and reported as counts.

The "likely burning" columns scale a forecast by last season's detection gap (Module A). They are added after the
scores are computed and never enter them.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd

from psw import config
from psw.models.evaluate import score
from psw.models.train import fit_with_conformal, predict_with_correction
from psw.response import harvest

HORIZON = 3
TARGET = "target_log_next3"
COUNT_FEATURES = ["count_today", "count_3d", "count_7d", "count_slope_7d"]
WEATHER_FEATURES = ["temp_mean", "rh_mean", "wind_speed_mean"]
CONTEXT_FEATURES = ["day_of_season", "district_prev_share", "district"]
FEATURES = COUNT_FEATURES + harvest.PROXY_FEATURES + harvest.RAIN_FEATURES + WEATHER_FEATURES + CONTEXT_FEATURES

VARIANTS = ("model", "persistence", "climatology")
BASELINE_COLUMN = {"persistence": "count_3d", "climatology": "climatology"}

TOP_K = 5
MIN_STATE_COUNT = 20  # ranking is scored only on days with at least this many detections statewide in t+1..t+3
GAP_CLIP = (0.0, 0.9)  # for planning only (D23): never below detected, at most 10 times it
STATE = "Punjab"

FORECAST_FILE = "risk_forecasts.parquet"
METRICS_FILE = "risk_metrics.json"
FORECAST_COLUMNS = ["issue_date", "district", "season", "variant", "p10", "p50", "p90", "likely_p50", "likely_low",
                    "likely_high", "gap_low", "gap_high", "gap_source", "actual"]
LIKELY_NOTE = ("'Likely' columns divide the forecast by (1 - gap), using the previous season's detection gap clipped to "
               "0 to 0.9. They estimate what would have been detected at the base season's detections per burned km2. "
               "They are not a count of fires, cannot be checked against satellite data and are not in any score.")


def _slope_7d(m: pd.DataFrame) -> pd.DataFrame:
    """Least-squares slope (detections per day) over the last 7 days; days before the season count as zero."""
    weights = np.arange(-3, 4)
    return sum(w * m.shift(6 - i, fill_value=0) for i, w in enumerate(weights)) / float((weights ** 2).sum())


def _next_sum(m: pd.DataFrame) -> pd.DataFrame:
    """Sum over t+1..t+HORIZON; NaN where the season ends first."""
    return sum(m.shift(-k) for k in range(1, HORIZON + 1))


def _climatology(m: pd.DataFrame, prev: pd.DataFrame | None) -> pd.DataFrame:
    """Last season's detections over the same 3 days, scaled by this season's to-date total / last season's.

    A district with nothing to date last season uses the statewide ratio (1 if the state had nothing either).
    """
    if prev is None:
        return pd.DataFrame(np.nan, index=m.index, columns=m.columns)
    prev = prev.reindex(index=m.index, columns=m.columns)
    now, before = m.cumsum(), prev.cumsum()
    state = (now.sum(axis=1) / before.sum(axis=1).where(lambda s: s > 0)).fillna(1.0)
    ratio = now / before.where(before > 0)
    ratio = ratio.apply(lambda col: col.fillna(state))
    return _next_sum(prev) * ratio


def _season_table(daily: pd.DataFrame, season: int) -> pd.DataFrame:
    """date, season, district, count features, previous-season share, climatology and the 3-day actual."""
    m = harvest.season_matrix(daily, season)
    prev = harvest.previous_matrix(daily, season)
    share = pd.Series(np.nan, index=m.columns)
    if prev is not None and prev.to_numpy().sum() > 0:
        share = (prev.sum() / prev.to_numpy().sum()).reindex(m.columns)
    frames = {"count_today": m, "count_3d": m.rolling(3, min_periods=1).sum(),
              "count_7d": m.rolling(7, min_periods=1).sum(), "count_slope_7d": _slope_7d(m),
              "climatology": _climatology(m, prev), "actual": _next_sum(m)}
    out = pd.concat({k: v.stack(future_stack=True) for k, v in frames.items()}, axis=1)
    out = out.rename_axis(["day_of_season", "district"]).reset_index()
    start = date(season, *config.SEASON_START)
    out["date"] = [start + timedelta(days=int(d)) for d in out["day_of_season"]]
    out["season"] = season
    out["district_prev_share"] = out["district"].map(share)
    return out


def build_table(daily: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """One row per district and issue day whose 3-day window lies inside the season, with FEATURES and TARGET.

    `daily`: date, season, district, fire_count (S-NPP). `weather`: date, district, precip_mm and WEATHER_FEATURES.
    Every feature for day t uses data up to t, except harvest.FORECAST_STANDIN.
    """
    seasons = sorted(int(s) for s in daily["season"].unique())
    out = pd.concat([_season_table(daily, s) for s in seasons], ignore_index=True)
    n = len(out)
    out = out.merge(harvest.proxy_features(daily), how="left", on=["date", "season", "district"])
    out = out.merge(harvest.rain_features(weather), how="left", on=["date", "district"])
    out = out.merge(weather[["date", "district", *WEATHER_FEATURES]], how="left", on=["date", "district"])
    needed = harvest.RAIN_FEATURES + WEATHER_FEATURES
    if len(out) != n or out[needed].isna().any().any():
        raise ValueError("district weather does not cover every district and day (including 3 days either side)")
    out = out.dropna(subset=["actual"]).copy()
    out[TARGET] = np.log1p(out["actual"])
    out["district"] = pd.Categorical(out["district"], categories=sorted(daily["district"].unique()))
    return out.sort_values(["season", "date", "district"]).reset_index(drop=True)


# ---------- forecasts ----------


def _to_counts(log_q: pd.DataFrame) -> pd.DataFrame:
    """Back-transform p10/p50/p90 from log1p; no negative counts, and 2 decimals so a tiny p10 does not miss a 0."""
    return np.expm1(log_q.clip(lower=0)).round(2)


def baseline_forecast(train: pd.DataFrame, test: pd.DataFrame, column: str) -> pd.DataFrame:
    """p50 = the baseline count; band = it plus the 10th/90th percentile of its training error on the log scale."""
    error = (train[TARGET] - np.log1p(train[column])).dropna()
    if error.empty:
        raise ValueError(f"no training rows with a {column} baseline")
    lo, hi = np.percentile(error, [10, 90])
    p50 = np.log1p(test[column])
    return _to_counts(pd.DataFrame({"p10": p50 + min(lo, 0), "p50": p50, "p90": p50 + max(hi, 0)}, index=test.index))


def run_backtest(table: pd.DataFrame, test_seasons: list[int]) -> tuple[pd.DataFrame, dict, dict]:
    """For each test season S, train on seasons < S and predict S. Returns (forecasts, conformal log, gain)."""
    out, conformal, gain = [], {}, {}
    for s in test_seasons:
        train, test = table[table["season"] < s], table[table["season"] == s]
        if train.empty or test.empty:
            raise ValueError(f"no training or test rows for season {s}")
        fit = fit_with_conformal(train, FEATURES, target=TARGET)
        preds = {"model": _to_counts(predict_with_correction(fit, test, FEATURES))}
        preds |= {name: baseline_forecast(train, test, col) for name, col in BASELINE_COLUMN.items()}
        conformal[str(s)] = {k: fit[k] for k in ("applied", "inner_coverage", "correction")}
        gain[str(s)] = dict(sorted(zip(FEATURES, map(float, fit["models"][0.5].booster_.feature_importance("gain"))),
                                   key=lambda kv: -kv[1]))
        for name in VARIANTS:
            p = preds[name]
            out.append(pd.DataFrame({
                "issue_date": test["date"].to_numpy(), "district": test["district"].astype(str).to_numpy(),
                "season": s, "variant": name, "p10": p["p10"].to_numpy(), "p50": p["p50"].to_numpy(),
                "p90": p["p90"].to_numpy(), "actual": test["actual"].to_numpy().astype(int),
            }))
    return pd.concat(out, ignore_index=True), conformal, gain


# ---------- ranking metrics (D20) ----------


def _top_weights(values: np.ndarray, k: int) -> np.ndarray:
    """Each item's chance of being in the top k when ties are broken at random."""
    if len(values) <= k:
        return np.ones(len(values))
    cut = np.sort(values)[::-1][k - 1]
    above, tied = values > cut, values == cut
    weights = above.astype(float)
    weights[tied] = (k - above.sum()) / tied.sum()
    return weights


def top_k_hit(actual, predicted, k: int = TOP_K) -> float:
    """Share of the actual top-k districts that are in the predicted top k.

    A district with no detections is never counted as an actual top district. Ties are scored by their expected
    value, so a forecast cannot win or lose on row order.
    """
    actual, predicted = np.asarray(actual, float), np.asarray(predicted, float)
    truth = _top_weights(actual, k) * (actual > 0)
    if truth.sum() == 0:
        return float("nan")
    return float((_top_weights(predicted, k) * truth).sum() / truth.sum())


def spearman(actual, predicted) -> float:
    """Rank correlation. NaN if the actuals are all equal; 0 if only the forecast is (it ranks nothing)."""
    a, p = pd.Series(np.asarray(actual, float)), pd.Series(np.asarray(predicted, float))
    if a.nunique() < 2:
        return float("nan")
    if p.nunique() < 2:
        return 0.0
    return float(a.rank().corr(p.rank()))


def ranking_metrics(forecasts: pd.DataFrame, min_count: int = MIN_STATE_COUNT) -> dict:
    """Mean top-5 hit rate and Spearman over the days with at least `min_count` actual detections statewide."""
    hits, rhos = [], []
    for _, day in forecasts.groupby(["season", "issue_date"]):
        if day["actual"].sum() < min_count:
            continue
        hits.append(top_k_hit(day["actual"], day["p50"]))
        rhos.append(spearman(day["actual"], day["p50"]))
    mean = lambda xs: float(np.nanmean(xs)) if xs else float("nan")  # noqa: E731
    return {"top5_hit": mean(hits), "spearman": mean(rhos), "n_days": len(hits)}


def build_metrics(forecasts: pd.DataFrame, seasons: list[int], test_seasons: list[int], conformal: dict,
                  gain: dict) -> dict:
    variants = {}
    for name in VARIANTS:
        d = forecasts[forecasts["variant"] == name]
        variants[name] = {"overall": score(d) | ranking_metrics(d),
                          "by_season": {str(s): score(g) | ranking_metrics(g) for s, g in d.groupby("season")}}
    best = max(VARIANTS, key=lambda v: (variants[v]["overall"]["top5_hit"], variants[v]["overall"]["spearman"]))
    return {
        "target": f"S-NPP VIIRS detections per district over the next {HORIZON} days",
        "seasons_test": test_seasons,
        "folds": {str(t): [s for s in seasons if s < t] for t in test_seasons},
        "variants": variants,
        "best_ranking_variant": best,
        "best_ranking_rule": "highest top-5 hit rate on the test seasons, then Spearman",
        "ranking_days_rule": f"days with at least {MIN_STATE_COUNT} detections statewide in the 3 days forecast",
        "rain_assumption": harvest.RAIN_ASSUMPTION,
        "likely_note": LIKELY_NOTE,
        "feature_importance": gain,
        "conformal": conformal,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


# ---------- likely burning ----------


def add_likely(forecasts: pd.DataFrame, gap_range: pd.DataFrame) -> pd.DataFrame:
    """Add the 'likely burning' columns from the PREVIOUS season's gap (D23), so nothing from this season is used.

    A district without its own range that season uses the Punjab range. With neither, the columns are NaN.
    """
    gaps = gap_range.dropna(subset=["gap_low", "gap_high"])
    gaps = gaps.assign(season=gaps["season"] + 1)[["district", "season", "gap_low", "gap_high"]]
    state = gaps[gaps["district"] == STATE].drop(columns="district")
    out = forecasts.merge(gaps, how="left", on=["district", "season"])
    out = out.merge(state, how="left", on="season", suffixes=("", "_state"))
    own = out["gap_low"].notna()
    out["gap_source"] = np.where(own, "district", np.where(out["gap_low_state"].notna(), STATE, None))
    for col in ("gap_low", "gap_high"):
        out[col] = out[col].where(own, out[f"{col}_state"]).clip(*GAP_CLIP)
    mid = (out["gap_low"] + out["gap_high"]) / 2
    for name, gap in (("likely_p50", mid), ("likely_low", out["gap_low"]), ("likely_high", out["gap_high"])):
        out[name] = (out["p50"] / (1 - gap)).round(2)
    return out.drop(columns=["gap_low_state", "gap_high_state"])


def run_all(daily: pd.DataFrame, weather: pd.DataFrame, gap_range: pd.DataFrame, n_test_seasons: int = 2,
            out_dir=None) -> dict:
    """Backtest, score, add the likely columns and write both output files. Returns the metrics."""
    out_dir = out_dir or config.PROCESSED
    out_dir.mkdir(parents=True, exist_ok=True)
    table = build_table(daily, weather)
    seasons = [int(s) for s in sorted(table["season"].unique())]
    test_seasons = seasons[-n_test_seasons:]
    forecasts, conformal, gain = run_backtest(table, test_seasons)
    metrics = build_metrics(forecasts, seasons, test_seasons, conformal, gain)
    add_likely(forecasts, gap_range)[FORECAST_COLUMNS].to_parquet(out_dir / FORECAST_FILE, index=False)
    (out_dir / METRICS_FILE).write_text(json.dumps(metrics, indent=1, allow_nan=False))
    return metrics
