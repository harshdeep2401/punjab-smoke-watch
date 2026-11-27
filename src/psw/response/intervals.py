"""How sure can we be that one variant's lead over another is more than chance? Block bootstrap over issue days.

Neighbouring days are alike (their 3-day windows overlap), so days are resampled in runs of 7 within a season, which
keeps that dependence. The seed is fixed, so a rerun gives the same intervals.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from psw.response import risk

BLOCK_DAYS = 7
REPS = 2000
SEED = 42
METHOD = (f"95% interval from a block bootstrap: issue days are resampled in runs of {BLOCK_DAYS} within each season, "
          f"{REPS} times with a fixed seed. It covers the choice of days only, not the choice of variant, which was made "
          "on the same test seasons.")


def block_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """n row numbers made of random runs of `block` consecutive rows."""
    block = min(block, n)
    starts = rng.integers(0, n - block + 1, size=int(np.ceil(n / block)))
    return (starts[:, None] + np.arange(block)).ravel()[:n]


def interval(seasons: list[np.ndarray], statistic: Callable[[np.ndarray], float], reps: int = REPS,
             block: int = BLOCK_DAYS, seed: int = SEED) -> tuple[float, float]:
    """(2.5%, 97.5%) of `statistic` over resampled data. `seasons`: one array per season, a row per day, in order."""
    rng = np.random.default_rng(seed)
    draws = [statistic(np.concatenate([s[block_indices(len(s), block, rng)] for s in seasons])) for _ in range(reps)]
    low, high = np.percentile(draws, [2.5, 97.5])
    return float(low), float(high)


def _lead(seasons: list[np.ndarray], statistic) -> dict:
    low, high = interval(seasons, statistic)
    return {"points": round(float(statistic(np.concatenate(seasons))), 2), "low": round(low, 2), "high": round(high, 2)}


def allocation_lead(scores: pd.DataFrame, of: str, against: str) -> dict:
    """Lead of one strategy over another in % of the oracle's score, pooled and per season.

    `scores` is allocate.daily_scores.
    """
    ordered = scores.sort_values(["season", "issue_date"])
    seasons = {str(s): g[[of, against, "oracle"]].to_numpy(float) for s, g in ordered.groupby("season")}

    def lead(x):
        return (x[:, 0].sum() - x[:, 1].sum()) / x[:, 2].sum() * 100

    return {"of": of, "against": against, **_lead(list(seasons.values()), lead),
            "by_season": {s: _lead([x], lead) for s, x in seasons.items()}}


def ranking_lead(forecasts: pd.DataFrame, of: str, against: str, min_count: int = risk.MIN_STATE_COUNT) -> dict:
    """Lead of one variant over another in top-5 hit rate (points) and Spearman, over the days that are scored.

    Per season as well as pooled: a pooled lead can come from a single season.
    """
    rows = []
    for (season, day), group in forecasts.groupby(["season", "issue_date"]):
        by = {v: g.sort_values("district") for v, g in group.groupby("variant")}
        a, b = by[of], by[against]
        if a["actual"].sum() < min_count:
            continue
        rows.append({"season": season, "day": day,
                     "hit": (risk.top_k_hit(a["actual"], a["p50"]) - risk.top_k_hit(b["actual"], b["p50"])) * 100,
                     "rho": risk.spearman(a["actual"], a["p50"]) - risk.spearman(b["actual"], b["p50"])})
    table = pd.DataFrame(rows).sort_values(["season", "day"])

    def mean(x):
        return np.nanmean(x[:, 0])

    hits = {str(s): g[["hit"]].to_numpy(float) for s, g in table.groupby("season")}
    rhos = [g[["rho"]].to_numpy(float) for _, g in table.groupby("season")]
    low, high = interval(rhos, mean)
    return {"of": of, "against": against, "n_days": int(len(table)),
            "top5": {**_lead(list(hits.values()), mean), "by_season": {s: _lead([x], mean) for s, x in hits.items()}},
            "spearman": {"difference": round(float(np.nanmean(table["rho"])), 3), "low": round(low, 3),
                         "high": round(high, 3)}}


def runner_up(metrics: dict) -> str:
    """The second-best variant by the same rule that picked the best."""
    others = [v for v in metrics["variants"] if v != metrics["best_ranking_variant"]]
    return max(others, key=lambda v: (metrics["variants"][v]["overall"]["top5_hit"],
                                      metrics["variants"][v]["overall"]["spearman"]))
