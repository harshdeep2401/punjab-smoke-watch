"""Expanding-window backtest (D5), ablation and artefacts (forecasts.parquet, metrics.json, final models)."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import mean_pinball_loss

from psw import config
from psw.models.train import (ALPHAS, coverage, fit_quantile_models, fit_with_conformal,
                              predict_with_correction)
from psw.processing.features import FIRE_FEATURES, MODEL_FEATURES, TARGET

VARIANTS = ("full", "no_fire", "persistence")
NO_FIRE = [f for f in MODEL_FEATURES if f not in FIRE_FEATURES]


def persistence_forecast(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """p50 = today's PM2.5; band = today + empirical 10th/90th percentile of the training day-to-day change."""
    delta = train[TARGET] - train["pm25_today"]
    lo, hi = np.percentile(delta, [10, 90])
    p50 = test["pm25_today"]
    return pd.DataFrame({"p10": p50 + min(lo, 0), "p50": p50, "p90": p50 + max(hi, 0)}, index=test.index)


def score(df: pd.DataFrame) -> dict:
    """df has p10, p50, p90, actual."""
    y = df["actual"].to_numpy()
    pin = np.mean([mean_pinball_loss(y, df[c], alpha=a) for c, a in zip(("p10", "p50", "p90"), ALPHAS)])
    return {
        "n": int(len(df)),
        "mae": float(np.mean(np.abs(y - df["p50"]))),
        "pinball": float(pin),
        "coverage80": coverage(df["p10"], df["p90"], y),
        "band_width": float(np.mean(df["p90"] - df["p10"])),
    }


def run_backtest(features: pd.DataFrame, test_seasons: list[int]) -> tuple[pd.DataFrame, dict]:
    """For each test season S, train on seasons < S and predict S. Returns (forecasts, conformal log)."""
    out, conformal = [], {}
    for s in test_seasons:
        train = features[features["season"] < s]
        test = features[features["season"] == s]
        if train.empty or test.empty:
            raise ValueError(f"no training or test rows for season {s}")
        fits = {"full": fit_with_conformal(train, MODEL_FEATURES), "no_fire": fit_with_conformal(train, NO_FIRE)}
        preds = {
            "full": predict_with_correction(fits["full"], test, MODEL_FEATURES),
            "no_fire": predict_with_correction(fits["no_fire"], test, NO_FIRE),
            "persistence": persistence_forecast(train, test),
        }
        conformal[str(s)] = {k: {"applied": v["applied"], "inner_coverage": v["inner_coverage"],
                                 "correction": v["correction"]} for k, v in fits.items()}
        for name in VARIANTS:
            p = preds[name]
            out.append(pd.DataFrame({
                "issue_date": test["date"].to_numpy(),
                "target_date": (pd.to_datetime(test["date"]) + pd.Timedelta(days=1)).dt.date.to_numpy(),
                "city": test["city"].astype(str).to_numpy(), "season": s, "variant": name,
                "p10": p["p10"].to_numpy(), "p50": p["p50"].to_numpy(), "p90": p["p90"].to_numpy(),
                "actual": test[TARGET].to_numpy(),
            }))
    return pd.concat(out, ignore_index=True), conformal


def build_metrics(forecasts: pd.DataFrame, all_seasons: list[int], test_seasons: list[int],
                  importance: dict, conformal: dict) -> dict:
    """`seasons_train` = seasons before the first test season (the smallest training set); `folds` gives the
    exact training seasons for each test season (expanding window)."""
    train_seasons = [x for x in all_seasons if x < min(test_seasons)]
    variants = {}
    for name, d in forecasts.groupby("variant"):
        variants[name] = {"overall": score(d), "by_city": {c: score(g) for c, g in d.groupby("city")}}
    return {
        "seasons_train": train_seasons, "seasons_test": test_seasons,
        "folds": {str(t): [x for x in all_seasons if x < t] for t in test_seasons},
        "variants": variants,
        "feature_importance": importance,
        "conformal_applied": any(v["applied"] for s in conformal.values() for v in s.values()),
        "conformal": conformal,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def train_final(features: pd.DataFrame, out_dir=None) -> dict:
    """Final full-variant model on all seasons: saves boosters (no conformal correction), returns gain importance."""
    out_dir = out_dir or config.PROCESSED
    out_dir.mkdir(parents=True, exist_ok=True)
    models = fit_quantile_models(features, features[TARGET], MODEL_FEATURES)
    for a, m in models.items():
        m.booster_.save_model(str(out_dir / f"model_full_{a}.txt"))
    gain = models[0.5].booster_.feature_importance(importance_type="gain")
    return dict(sorted(zip(MODEL_FEATURES, map(float, gain)), key=lambda kv: -kv[1]))


def run_all(features: pd.DataFrame, n_test_seasons: int = 2, out_dir=None) -> dict:
    out_dir = out_dir or config.PROCESSED
    seasons = [int(s) for s in sorted(features["season"].unique())]
    test_seasons = seasons[-n_test_seasons:]
    forecasts, conformal = run_backtest(features, test_seasons)
    imp = train_final(features, out_dir)
    metrics = build_metrics(forecasts, seasons, test_seasons, imp, conformal)
    forecasts.to_parquet(out_dir / "forecasts.parquet", index=False)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics
