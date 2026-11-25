"""Pooled LightGBM quantile models (p10/p50/p90) with an optional conformal band correction (CQR)."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from psw.processing.features import TARGET

ALPHAS = (0.1, 0.5, 0.9)
SEED = 42
COVERAGE_OK = (0.70, 0.90)  # acceptable 80% band coverage on the inner validation season
TARGET_COVERAGE = 0.80


def fit_quantile_models(X: pd.DataFrame, y, features: list[str]) -> dict[float, LGBMRegressor]:
    models = {}
    for a in ALPHAS:
        m = LGBMRegressor(
            objective="quantile", alpha=a, n_estimators=300, learning_rate=0.05, num_leaves=15,
            min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            random_state=SEED, deterministic=True, force_row_wise=True, n_jobs=1, verbose=-1,
        )
        m.fit(X[features], y)
        models[a] = m
    return models


def predict_quantiles(models: dict[float, LGBMRegressor], X: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Returns p10, p50, p90 sorted per row so the quantiles never cross."""
    raw = np.column_stack([models[a].predict(X[features]) for a in ALPHAS])
    raw.sort(axis=1)
    return pd.DataFrame(raw, columns=["p10", "p50", "p90"], index=X.index)


def cqr_correction(p10, p90, y, level: float = TARGET_COVERAGE) -> float:
    """Conformalised-quantile-regression offset q: the band becomes [p10 - q, p90 + q]. q < 0 narrows it."""
    p10, p90, y = map(np.asarray, (p10, p90, y))
    scores = np.maximum(p10 - y, y - p90)
    n = len(scores)
    k = min(n, math.ceil((n + 1) * level))
    return float(np.sort(scores)[k - 1])


def coverage(p10, p90, y) -> float:
    p10, p90, y = map(np.asarray, (p10, p90, y))
    return float(np.mean((y >= p10) & (y <= p90)))


def fit_with_conformal(train: pd.DataFrame, features: list[str]) -> dict:
    """Fit on all of `train`; calibrate the band on an inner split (train on seasons < last, validate on last).

    Returns {models, correction, applied, inner_coverage}. The correction is applied only if the inner
    80% coverage falls outside COVERAGE_OK. With a single training season there is no inner split, so no correction.
    """
    seasons = sorted(train["season"].unique())
    correction, applied, inner_cov = 0.0, False, None
    if len(seasons) >= 2:
        inner_tr = train[train["season"] < seasons[-1]]
        inner_va = train[train["season"] == seasons[-1]]
        inner_models = fit_quantile_models(inner_tr, inner_tr[TARGET], features)
        q = predict_quantiles(inner_models, inner_va, features)
        inner_cov = coverage(q["p10"], q["p90"], inner_va[TARGET])
        if not (COVERAGE_OK[0] <= inner_cov <= COVERAGE_OK[1]):
            correction = cqr_correction(q["p10"], q["p90"], inner_va[TARGET])
            applied = True
    models = fit_quantile_models(train, train[TARGET], features)
    return {"models": models, "correction": correction, "applied": applied, "inner_coverage": inner_cov}


def predict_with_correction(fit: dict, X: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    q = predict_quantiles(fit["models"], X, features)
    q["p10"] = np.minimum(q["p10"] - fit["correction"], q["p50"])  # keep p10 <= p50 <= p90 after narrowing
    q["p90"] = np.maximum(q["p90"] + fit["correction"], q["p50"])
    return q
