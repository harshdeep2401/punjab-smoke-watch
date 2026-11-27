"""Deterministic allocation of machines and teams across districts (D19), and the backtest of that decision.

cover(u) = 1 - exp(-u / k): the share of a district's expected burning that u units are assumed to deal with. Units are
placed one at a time where the next one adds the most. k is an assumption, not something measured: no data here says
what a machine achieves. The LLM never sets any of this.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_UNITS = 500
DEFAULT_K = 20.0
DEFAULT_MAX_SHARE = 0.25
LIKELY, DETECTED = "likely", "detected"
COLUMNS = ["district", "units", "risk_used", "basis", "covered_share", "marginal_gain_last", "reason"]


def cover(units, k: float = DEFAULT_K):
    return 1.0 - np.exp(-np.asarray(units, float) / k)


def unit_cap(units: int, max_share: float) -> int:
    """The most one district can get. At least 1, so a very small fleet can still be placed."""
    return max(1, int(max_share * units + 1e-9))


def _greedy(risk: np.ndarray, units: int, k: float, low: int, cap: int) -> np.ndarray:
    """Units per row. Ties go to the earlier row, so the caller fixes the order. Stops when no unit adds anything."""
    got = np.full(len(risk), low, dtype=int)
    step = 1.0 - np.exp(-1.0 / k)
    gain = np.where(got < cap, risk * np.exp(-got / k) * step, 0.0)
    for _ in range(units - int(got.sum())):
        i = int(np.argmax(gain))
        if gain[i] <= 0:
            break
        got[i] += 1
        gain[i] = risk[i] * np.exp(-got[i] / k) * step if got[i] < cap else 0.0
    return got


def risk_used(risk: pd.DataFrame, use_likely: bool) -> tuple[pd.Series, pd.Series]:
    """(expected 3-day burning, "likely" or "detected") per row: likely_p50 where asked for and present, else p50."""
    values = risk["p50"].astype(float)
    basis = pd.Series(DETECTED, index=risk.index)
    if use_likely and "likely_p50" in risk.columns:
        has = risk["likely_p50"].notna()
        values = values.where(~has, risk["likely_p50"].astype(float))
        basis = basis.where(~has, LIKELY)
    return values, basis


def _reason(units: int, risk: float, basis: str, rank: int, n: int, cap: int) -> str:
    if risk <= 0:
        return "no burning expected in the next 3 days"
    kind = "likely burning" if basis == LIKELY else "forecast detections"
    text = f"{kind} {risk:.1f} over the next 3 days, rank {rank} of {n}"
    if units >= cap:
        return f"{text}; held at the limit of {cap} units per district"
    if units == 0:
        return f"{text}; every unit adds more elsewhere"
    return text


def allocate(risk: pd.DataFrame, units: int, k: float = DEFAULT_K, min_per_district: int = 0,
             max_share: float = DEFAULT_MAX_SHARE, use_likely: bool = True) -> pd.DataFrame:
    """Split `units` over the districts of `risk` (district, p50 and optionally likely_p50), one row per district.

    No district gets more than max_share of the fleet or fewer than min_per_district. A district with no expected
    burning gets no more than the minimum. If every district with burning is at its limit, the rest stays unplaced.
    Sorted by units, then expected burning.
    """
    if units < 0 or min_per_district < 0 or k <= 0 or not 0 < max_share <= 1:
        raise ValueError("units and min_per_district must be 0 or more, k above 0 and max_share in (0, 1]")
    if risk["district"].duplicated().any():
        raise ValueError("one row per district is needed")
    table = risk.sort_values("district").reset_index(drop=True)
    values, basis = risk_used(table, use_likely)
    if values.isna().any() or (values < 0).any():
        raise ValueError("expected burning must be a number of 0 or more for every district")
    cap = unit_cap(units, max_share)
    if min_per_district > cap or min_per_district * len(table) > units:
        raise ValueError(f"cannot give every district {min_per_district} of {units} units (limit {cap} per district)")
    got = _greedy(values.to_numpy(), units, k, min_per_district, cap)
    ranks = values.rank(ascending=False, method="min").astype(int)
    covered = cover(got, k)
    last = np.where(got > 0, values.to_numpy() * (covered - cover(np.maximum(got - 1, 0), k)), 0.0)
    out = pd.DataFrame({
        "district": table["district"], "units": got, "risk_used": values.round(2), "basis": basis,
        "covered_share": covered.round(4), "marginal_gain_last": last.round(4),
        "reason": [_reason(int(u), float(v), b, int(r), len(table), cap)
                   for u, v, b, r in zip(got, values, basis, ranks)],
    })
    return out.sort_values(["units", "risk_used", "district"], ascending=[False, False, True])[COLUMNS] \
        .reset_index(drop=True)


def compare_to_detected(risk: pd.DataFrame, units: int, k: float = DEFAULT_K, min_per_district: int = 0,
                        max_share: float = DEFAULT_MAX_SHARE) -> pd.DataFrame:
    """What the likely-burning adjustment moves: district, units_likely, units_detected, moved (likely - detected).

    Sorted by the size of the move.
    """
    args = {"k": k, "min_per_district": min_per_district, "max_share": max_share}
    likely = allocate(risk, units, use_likely=True, **args)[["district", "units"]]
    detected = allocate(risk, units, use_likely=False, **args)[["district", "units"]]
    out = likely.merge(detected, on="district", suffixes=("_likely", "_detected"))
    out["moved"] = out["units_likely"] - out["units_detected"]
    return out.assign(size=out["moved"].abs()).sort_values(["size", "district"], ascending=[False, True]) \
        .drop(columns="size").reset_index(drop=True)


def proportional(weights: pd.Series, units: int) -> pd.Series:
    """Whole units in proportion to `weights` (largest remainder). All zero weights give an equal split."""
    w = weights.sort_index().astype(float)
    if (w < 0).any():
        raise ValueError("weights must be 0 or more")
    share = w / w.sum() if w.sum() > 0 else pd.Series(1.0 / len(w), index=w.index)
    exact = share * units
    got = np.floor(exact + 1e-9).astype(int)
    spare = units - int(got.sum())
    got[(exact - got).sort_values(ascending=False, kind="stable").index[:spare]] += 1
    return got.reindex(weights.index)


def compare_to_proportional(allocation: pd.DataFrame, last_season: pd.Series) -> pd.DataFrame:
    """The allocation beside the naive one, "share of last season's detections".

    `last_season` is district -> detections over the whole of the previous season. Returns district, units,
    proportional_units and difference.
    """
    districts = pd.Index(allocation["district"])
    naive = proportional(last_season.reindex(districts, fill_value=0), int(allocation["units"].sum()))
    out = allocation[["district", "units"]].copy()
    out["proportional_units"] = naive.to_numpy()
    out["difference"] = out["units"] - out["proportional_units"]
    return out.reset_index(drop=True)


# ---------- decision backtest ----------

LIKELY_CHECK_NOTE = ("Allocation on likely burning (the planner's default), scored the same way against detections. "
                     "Detections cannot show the undetected burning it plans for, so this does not say whether it helps "
                     "more or less; it shows how far it departs from what satellites later saw.")
SCORE_NOTE = ("Score: for each issue day, the sum over districts of the detections actually seen in the next 3 days "
              "times cover(units). Shown as a percentage of an allocation that knew those detections in advance, summed "
              "over all days. cover() and k are assumptions, so this compares allocations; it does not measure fires "
              "prevented.")


def _score(actual: pd.Series, units: pd.Series, k: float) -> float:
    return float((actual * cover(units.reindex(actual.index), k)).sum())


def _units(risk: pd.DataFrame, units: int, k: float, max_share: float, use_likely: bool = False) -> pd.Series:
    return allocate(risk, units, k=k, max_share=max_share, use_likely=use_likely).set_index("district")["units"]


def daily_scores(forecasts: pd.DataFrame, daily: pd.DataFrame, units: int = DEFAULT_UNITS, k: float = DEFAULT_K,
                 max_share: float = DEFAULT_MAX_SHARE) -> pd.DataFrame:
    """One row per (season, issue_date): the score of every strategy, of the oracle and the actual detections.

    `forecasts` is risk_forecasts (all variants). `daily` (date, season, district, fire_count) gives the previous
    season's totals for the proportional allocation. Every forecast was issued on day t for t+1..t+3 by a model
    trained on earlier seasons.
    """
    totals = daily.groupby(["season", "district"])["fire_count"].sum()
    rows = []
    for (season, day), group in forecasts.groupby(["season", "issue_date"]):
        if season - 1 not in totals.index.get_level_values("season"):
            raise ValueError(f"no season {season - 1} for the proportional allocation of {season}")
        by_variant = {v: g.sort_values("district").reset_index(drop=True) for v, g in group.groupby("variant")}
        first = next(iter(by_variant.values()))
        actual = first.set_index("district")["actual"].astype(float)
        placed = {v: _units(g, units, k, max_share) for v, g in by_variant.items()}
        placed["last_season_share"] = proportional(totals.loc[season - 1].reindex(actual.index, fill_value=0), units)
        placed["equal_split"] = proportional(pd.Series(1.0, index=actual.index), units)
        placed["oracle"] = _units(pd.DataFrame({"district": actual.index, "p50": actual.to_numpy()}), units, k, max_share)
        if "model" in by_variant and "likely_p50" in by_variant["model"].columns:
            placed["model_likely"] = _units(by_variant["model"], units, k, max_share, use_likely=True)
        rows.append({"season": season, "issue_date": day, "actual_total": float(actual.sum()),
                     **{name: _score(actual, u, k) for name, u in placed.items()}})
    return pd.DataFrame(rows)


def decision_backtest(forecasts: pd.DataFrame, daily: pd.DataFrame, units: int = DEFAULT_UNITS, k: float = DEFAULT_K,
                      max_share: float = DEFAULT_MAX_SHARE) -> dict:
    """Each strategy's score as a percentage of the oracle's, pooled and per season (see SCORE_NOTE).

    The likely-burning allocation is reported apart from the comparison, under `likely_check`.
    """
    scores = daily_scores(forecasts, daily, units, k, max_share)
    strategies = [c for c in scores.columns if c not in ("season", "issue_date", "actual_total", "oracle", "model_likely")]

    def pct(frame: pd.DataFrame, name: str) -> float:
        best = frame["oracle"].sum()
        return round(float(frame[name].sum() / best * 100), 2) if best > 0 else None

    seasons = {str(s): g for s, g in scores.groupby("season")}
    out = {
        "units": units, "k": k, "max_share": max_share, "n_days": int(len(scores)),
        "seasons": sorted(int(s) for s in seasons),
        "score_note": SCORE_NOTE,
        "pct_of_oracle": {name: pct(scores, name) for name in strategies},
        "by_season": {s: {name: pct(g, name) for name in strategies} for s, g in seasons.items()},
        "oracle_covered_pct_of_detections": round(float(scores["oracle"].sum() / scores["actual_total"].sum() * 100), 2)
        if scores["actual_total"].sum() > 0 else None,
    }
    if "model_likely" in scores.columns:
        out["likely_check"] = {"pct_of_oracle": pct(scores, "model_likely"),
                               "by_season": {s: pct(g, "model_likely") for s, g in seasons.items()},
                               "note": LIKELY_CHECK_NOTE}
    return out
