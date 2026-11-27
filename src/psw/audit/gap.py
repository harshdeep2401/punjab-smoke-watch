"""Module A analytics: how far do counts and burned area disagree? Pure functions, no I/O.

The method is D23 (docs/PLAN.md, Phase 10). The signals are not independent, so each has one role:

  count series       S-NPP detections, official counts        what claims rest on (the numerator)
  reference          MCD64A1 burned area                      the only one we have (the denominator)
  timing indicators  night share, MODIS vs S-NPP (Punjab)     set the confidence, never a number in the range
  robustness check   NOAA-20, NOAA-21                         in `sensor_trend` only

gap = 1 - (C_t / B_t) / (C_base / B_base): the fall in detections per burned km2 since a base season. It is a relative
loss of detection efficiency, not the share of burning that is missed.

Every table has a `district` column; the value "Punjab" is the state total (rows inside the 22 districts only).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from psw.audit.sensors import AFTERNOON, MORNING, NIGHT, OTHER

STATE = "Punjab"
PASSES = (MORNING, AFTERNOON, NIGHT, OTHER)
VIIRS = ("snpp", "noaa20", "noaa21")
COUNT_SERIES = ("snpp", "official")  # the only series that can be a numerator; NOAA-20/21 never enter the range
SERIES_NAME = {"snpp": "S-NPP detections", "official": "official counts"}
BASES = (2022, 2023)

MIN_BURNED_KM2 = 50  # both seasons
MIN_BASE_COUNT = 100  # the base season
MEDIUM_MIN_GAP = 0.25  # Punjab "medium" needs gap_low at least this, and both timing indicators
MODIS_SUPPORT_RATIO = 1.5
Z95 = 1.959963984540054

INSUFFICIENT = "insufficient evidence"
GAP_CAVEAT = ("The range shows only how the result changes with the count series and the base season. It does not include "
              "the error of the MODIS burned-area product, which is unknown and is the largest uncertainty. This is "
              "triangulation of satellite and official signals, not field verification.")
EXPLANATIONS = ("A fall in detections per burned area fits three explanations: burning moved outside satellite pass times, "
                "fires became smaller or cooler, or the sensitivity of the burned-area product changed.")
MODIS_CONFOUND = "MODIS passes at other times (Terra in the morning, Aqua drifting later in the afternoon), which is a confound"


def _inside(ms: pd.DataFrame) -> pd.DataFrame:
    """Punjab rows only, each once under its district and once under the state total."""
    inside = ms[ms["district"].notna()]
    return pd.concat([inside, inside.assign(district=STATE)], ignore_index=True)


def _districts(ms: pd.DataFrame) -> list[str]:
    return sorted(ms["district"].dropna().unique()) + [STATE]


def _available(ms: pd.DataFrame) -> list[tuple[str, int]]:
    """(sensor, season) pairs that have data anywhere, so a district with none gets a real 0 and not a gap."""
    return sorted({(s, int(y)) for s, y in ms[["sensor", "season"]].drop_duplicates().itertuples(index=False)})


def _grid(ms: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([(d, s, y) for d in _districts(ms) for s, y in _available(ms)], columns=["district", "sensor", "season"])


def pct_change(new, old) -> float:
    """Percent change, NaN when there is nothing to compare with."""
    if pd.isna(new) or pd.isna(old) or old == 0:
        return float("nan")
    return (new - old) / old * 100


def _counts(ms: pd.DataFrame) -> pd.DataFrame:
    """district, sensor, season, count (zeros filled)."""
    n = _inside(ms).groupby(["district", "sensor", "season"]).size().rename("count").reset_index()
    out = _grid(ms).merge(n, how="left", on=["district", "sensor", "season"])
    return out.assign(count=out["count"].fillna(0).astype(int))


def diurnal_profile(ms: pd.DataFrame) -> pd.DataFrame:
    """district x season x sensor x pass_label: count and its share of that sensor's season in that district."""
    n = _inside(ms).groupby(["district", "sensor", "season", "pass_label"]).size().rename("count").reset_index()
    grid = _grid(ms).merge(pd.DataFrame({"pass_label": list(PASSES)}), how="cross")
    out = grid.merge(n, how="left", on=["district", "sensor", "season", "pass_label"])
    out["count"] = out["count"].fillna(0).astype(int)
    total = out.groupby(["district", "sensor", "season"])["count"].transform("sum")
    out["share"] = out["count"] / total.where(total > 0)
    return out.sort_values(["district", "season", "sensor"], kind="stable").reset_index(drop=True)


def sensor_trend(ms: pd.DataFrame) -> pd.DataFrame:
    """district x season x sensor: count and % change vs the sensor's first season (2022, except NOAA-21: 2023)."""
    out = _counts(ms).sort_values(["district", "sensor", "season"]).reset_index(drop=True)
    first = out.groupby(["district", "sensor"])[["season", "count"]].transform("first")
    out["base_season"] = first["season"]
    out["change_pct"] = [pct_change(n, b) for n, b in zip(out["count"], first["count"])]
    return out[["district", "season", "sensor", "count", "base_season", "change_pct"]]


def night_share(ms: pd.DataFrame) -> pd.DataFrame:
    """season x district: night passes over all passes, all VIIRS satellites together."""
    v = _inside(ms)
    v = v[v["sensor"].isin(VIIRS)]
    g = v.assign(night=v["pass_label"] == NIGHT).groupby(["district", "season"])["night"].agg(["sum", "size"])
    full = pd.MultiIndex.from_product([_districts(ms), sorted(ms["season"].unique())], names=["district", "season"])
    out = g.reindex(full, fill_value=0).rename(columns={"sum": "night", "size": "total"}).reset_index()
    out[["night", "total"]] = out[["night", "total"]].astype(int)
    out["night_share"] = out["night"] / out["total"].where(out["total"] > 0)
    return out


def _burned(ba: pd.DataFrame) -> pd.DataFrame:
    """district, season, burned_km2, with the state total."""
    state = ba.groupby("season", as_index=False)["burned_km2"].sum().assign(district=STATE)
    return pd.concat([ba[["district", "season", "burned_km2"]], state], ignore_index=True)


def burned_vs_detected(ms: pd.DataFrame, ba: pd.DataFrame, base: int = BASES[0]) -> pd.DataFrame:
    """district x season: S-NPP detections, burned km2, detections per km2, and % changes vs `base`."""
    snpp = _counts(ms)
    snpp = snpp[snpp["sensor"] == "snpp"][["district", "season", "count"]].rename(columns={"count": "detections"})
    out = _burned(ba).merge(snpp, how="left", on=["district", "season"])
    out["detections"] = out["detections"].fillna(0).astype(int)
    out["detections_per_km2"] = out["detections"] / out["burned_km2"].where(out["burned_km2"] > 0)
    at_base = out[out["season"] == base].set_index("district")
    for col, name in (("detections", "detections"), ("burned_km2", "burned"), ("detections_per_km2", "per_km2")):
        out[f"{name}_change_pct"] = [pct_change(v, at_base[col].get(d, np.nan)) for d, v in zip(out["district"], out[col])]
    out["base_season"] = base
    return out.sort_values(["district", "season"]).reset_index(drop=True)


def official_vs_satellite(ms: pd.DataFrame, off: pd.DataFrame) -> pd.DataFrame:
    """district x season: the official count, S-NPP detections and their ratio (official / S-NPP).

    The state row exists only when every district has an official count for the season, so it is never a partial sum.
    """
    cols = ["district", "season", "count", "source_url", "method_note"]
    n_districts = len(_districts(ms)) - 1
    states = []
    for season, g in off.groupby("season"):
        if g["district"].nunique() == n_districts:
            states.append({"district": STATE, "season": season, "count": g["count"].sum(),
                           "source_url": " ".join(sorted(g["source_url"].unique())),
                           "method_note": "; ".join(sorted({n.split("; includes")[0] for n in g["method_note"]}))})
    both = pd.concat([off[cols], pd.DataFrame(states, columns=cols)], ignore_index=True).rename(columns={"count": "official"})
    snpp = _counts(ms)
    snpp = snpp[snpp["sensor"] == "snpp"][["district", "season", "count"]].rename(columns={"count": "snpp"})
    out = both.merge(snpp, how="inner", on=["district", "season"])
    out["official"] = out["official"].astype(int)
    out["ratio"] = out["official"] / out["snpp"].where(out["snpp"] > 0)
    return out[["district", "season", "official", "snpp", "ratio", "source_url", "method_note"]] \
        .sort_values(["district", "season"]).reset_index(drop=True)


def wilson_interval(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """95% Wilson score interval for the share k / n. (nan, nan) when n is 0."""
    if n <= 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def timing_indicators(ms: pd.DataFrame, base: int = BASES[0]) -> pd.DataFrame:
    """Punjab x season: does the timing of detections suggest burning moved outside the afternoon pass?

    night_share is of S-NPP detections only, so the satellite mix is the same every season.
    supports_night: the season's 95% Wilson interval lies wholly above the base season's.
    supports_modis: (MODIS season/base) / (S-NPP season/base) is at least 1.5.
    """
    inside = ms[ms["district"].notna()]
    snpp = inside[inside["sensor"] == "snpp"]
    n_snpp = snpp.groupby("season").size()
    n_night = snpp[snpp["pass_label"] == NIGHT].groupby("season").size()
    n_modis = inside[inside["sensor"] == "modis"].groupby("season").size()
    if base not in n_snpp.index:
        raise ValueError(f"timing indicators: no S-NPP detections in the base season {base}")
    base_hi = wilson_interval(int(n_night.get(base, 0)), int(n_snpp[base]))[1]

    rows = []
    for season in sorted(inside["season"].unique()):
        total, night, modis = int(n_snpp.get(season, 0)), int(n_night.get(season, 0)), int(n_modis.get(season, 0))
        lo, hi = wilson_interval(night, total)
        snpp_ratio = total / n_snpp[base]
        modis_ratio = modis / n_modis[base] if n_modis.get(base, 0) else float("nan")
        both = modis_ratio / snpp_ratio if snpp_ratio > 0 else float("nan")
        rows.append({"district": STATE, "season": int(season), "base_season": base, "night": night, "total": total,
                     "night_share": night / total if total else float("nan"), "ci_low": lo, "ci_high": hi,
                     "modis": modis, "modis_ratio": modis_ratio, "snpp_ratio": snpp_ratio, "modis_vs_viirs": both,
                     "supports_night": bool(season > base and lo > base_hi),
                     "supports_modis": bool(season > base and both >= MODIS_SUPPORT_RATIO)})
    return pd.DataFrame(rows)


def timing_text(timing: pd.DataFrame, season: int) -> str:
    """The two timing indicators of one season, in words with their values."""
    t = timing.set_index("season")
    if season not in t.index:
        return "no timing indicators for this season"
    r = t.loc[season]
    base = int(r["base_season"])
    b = t.loc[base]
    yes = {True: "supports a shift in timing", False: "does not support a shift in timing"}
    modis = "not available" if pd.isna(r["modis_vs_viirs"]) else f"{r['modis_vs_viirs']:.2f} times"
    return (f"Punjab-wide only. Night share of S-NPP detections: {r['night_share']:.1%} "
            f"(95% interval {r['ci_low']:.1%} to {r['ci_high']:.1%}) in {season} against {b['night_share']:.1%} "
            f"({b['ci_low']:.1%} to {b['ci_high']:.1%}) in {base}: {yes[bool(r['supports_night'])]}. "
            f"MODIS change relative to the S-NPP change since {base}: {modis} (rule: at least {MODIS_SUPPORT_RATIO}): "
            f"{yes[bool(r['supports_modis'])]}. {MODIS_CONFOUND}.")


def method_words(methods: str | None) -> str | None:
    """'snpp / burned_area, base 2022; ...' in plain words, for text a person (or an agent that quotes it) reads."""
    if not isinstance(methods, str):
        return None
    return methods.replace("snpp /", f"{SERIES_NAME['snpp']} /").replace("official /", f"{SERIES_NAME['official']} /") \
        .replace("/ burned_area, base", "against burned area, base season")


def small_base(count) -> str | None:
    """Percent changes from a handful of detections mislead (1 to 3 is "up 200%")."""
    if count is None or pd.isna(count) or count >= MIN_BASE_COUNT:
        return None
    return f"the base count is under {MIN_BASE_COUNT}, so percentage changes are unreliable"


def _why_not(count_base, burned_t, burned_base, count_t, season: int, base: int) -> str | None:
    """Why a (count series, base) pair cannot be used, or None if it can."""
    if pd.isna(count_t) or pd.isna(count_base):
        return f"no count for {season} or {base}"
    low = [f"{y} ({v:,.0f} km²)" for y, v in ((base, burned_base), (season, burned_t)) if pd.isna(v) or v < MIN_BURNED_KM2]
    if low:
        return f"burned area under {MIN_BURNED_KM2} km² in " + " and ".join(low)
    if count_base < MIN_BASE_COUNT:
        return f"base count under {MIN_BASE_COUNT} in {base} ({int(count_base)})"
    return None


def gap_estimates(ms: pd.DataFrame, ba: pd.DataFrame, off: pd.DataFrame, bases=BASES) -> pd.DataFrame:
    """One row per district x season x count series x base season (base < season), used or not."""
    snpp = _counts(ms)
    snpp = snpp[snpp["sensor"] == "snpp"]
    official = official_vs_satellite(ms, off)
    counts = {"snpp": {(d, int(y)): v for d, y, v in zip(snpp["district"], snpp["season"], snpp["count"])},
              "official": {(d, int(y)): v for d, y, v in zip(official["district"], official["season"], official["official"])}}
    burned = _burned(ba)
    area = {(d, int(y)): v for d, y, v in zip(burned["district"], burned["season"], burned["burned_km2"])}
    seasons = sorted({y for _, y in area})

    rows = []
    for district in _districts(ms):
        for season in seasons:
            for base in (b for b in bases if b < season and b in seasons):
                for series in COUNT_SERIES:
                    c_t, c_b = counts[series].get((district, season), np.nan), counts[series].get((district, base), np.nan)
                    b_t, b_b = area.get((district, season), np.nan), area.get((district, base), np.nan)
                    why = _why_not(c_b, b_t, b_b, c_t, season, base)
                    gap = np.nan if why else 1 - (c_t / b_t) / (c_b / b_b)
                    rows.append({"district": district, "season": season, "count_series": series, "base_season": base,
                                 "method": f"{series} / burned_area, base {base}",
                                 "count_change_pct": pct_change(c_t, c_b), "burned_change_pct": pct_change(b_t, b_b),
                                 "gap": gap, "usable": why is None, "why_not": why})
    return pd.DataFrame(rows)


def _confidence(district: str, low: float, timing: pd.DataFrame, season: int) -> str:
    """Never "high": one reference signal and no field data. Districts are "low" at most."""
    if district != STATE or low < MEDIUM_MIN_GAP:
        return "low"
    t = timing[timing["season"] == season]
    supported = not t.empty and bool(t["supports_night"].iloc[0]) and bool(t["supports_modis"].iloc[0])
    return "medium" if supported else "low"


def direction(low: float, high: float) -> str:
    if low > 0:
        return "counts fell more than burned area"
    if high <= 0:
        return "no sign of a detection gap"
    return "direction unclear"


def gap_range(ms: pd.DataFrame, ba: pd.DataFrame, off: pd.DataFrame, timing: pd.DataFrame, bases=BASES) -> pd.DataFrame:
    """district|Punjab x season: the range of the gap over the usable (count series, base) pairs. Nothing is clipped.

    detected_change_pct and reference_change_pct describe one pair, named in detected_basis: S-NPP against the earliest
    base if usable, otherwise the first usable pair. With no usable pair there is no range, and `reason` says why.
    """
    rows = []
    for (district, season), g in gap_estimates(ms, ba, off, bases).groupby(["district", "season"], sort=True):
        used = g[g["usable"]].sort_values(["base_season", "count_series"], ascending=[True, False])  # snpp first
        row = {"district": district, "season": int(season), "detected_change_pct": np.nan, "reference_change_pct": np.nan,
               "detected_basis": None, "gap_low": np.nan, "gap_high": np.nan, "methods": None,
               "confidence": INSUFFICIENT, "direction": None, "reason": None,
               "timing_evidence": timing_text(timing, int(season)), "caveat": GAP_CAVEAT}
        if used.empty:
            row["reason"] = "; ".join(dict.fromkeys(g["why_not"]))
        else:
            first = used.iloc[0]
            low, high = float(used["gap"].min()), float(used["gap"].max())
            row.update(detected_change_pct=first["count_change_pct"], reference_change_pct=first["burned_change_pct"],
                       detected_basis=f"{SERIES_NAME[first['count_series']]} against {first['base_season']}",
                       gap_low=low, gap_high=high, methods="; ".join(used["method"]),
                       confidence=_confidence(district, low, timing, int(season)), direction=direction(low, high))
        rows.append(row)
    return pd.DataFrame(rows)


def _num(x, digits: int = 1):
    return None if x is None or pd.isna(x) else round(float(x), digits)


def _text(x) -> str | None:
    """pandas turns None in a text column into NaN, which is not valid JSON."""
    return x if isinstance(x, str) else None


def audit_tables(ms: pd.DataFrame, ba: pd.DataFrame, off: pd.DataFrame, bases=BASES) -> dict[str, pd.DataFrame]:
    """Every audit table by name (the file names under data/processed/audit/)."""
    timing = timing_indicators(ms, base=bases[0])
    return {
        "diurnal_profile": diurnal_profile(ms),
        "sensor_trend": sensor_trend(ms),
        "night_share": night_share(ms),
        "burned_vs_detected": burned_vs_detected(ms, ba, base=bases[0]),
        "official_vs_satellite": official_vs_satellite(ms, off),
        "timing_indicators": timing,
        "gap_estimates": gap_estimates(ms, ba, off, bases),
        "gap_range": gap_range(ms, ba, off, timing, bases),
    }


def audit_summary(tables: dict[str, pd.DataFrame]) -> dict:
    """Punjab headline numbers per season, as percentages rounded for reading."""
    def state(name: str) -> pd.DataFrame:
        t = tables[name]
        return t[t["district"] == STATE]

    trend, night, bvd = state("sensor_trend"), state("night_share").set_index("season"), state("burned_vs_detected").set_index("season")
    official, gaps = state("official_vs_satellite").set_index("season"), state("gap_range").set_index("season")
    timing = state("timing_indicators").set_index("season")
    base = int(bvd["base_season"].iloc[0])
    seasons = {}
    for season in sorted(bvd.index):
        entry = {
            "sensors": {r["sensor"]: {"detections": int(r["count"]), "base_season": int(r["base_season"]),
                                      "change_pct": _num(r["change_pct"])}
                        for r in trend[trend["season"] == season].to_dict("records")},
            "viirs_night_share_pct": _num(night.loc[season, "night_share"] * 100),  # all VIIRS satellites
            "snpp_night_share_pct": _num(timing.loc[season, "night_share"] * 100),  # what the timing indicator uses
            "burned_km2": _num(bvd.loc[season, "burned_km2"], 0),
            "burned_change_pct": _num(bvd.loc[season, "burned_change_pct"]),
            "official_count": None, "official_change_pct": None, "gap": None,
        }
        if season in official.index:
            entry["official_count"] = int(official.loc[season, "official"])
            if base in official.index:
                entry["official_change_pct"] = _num(pct_change(official.loc[season, "official"], official.loc[base, "official"]))
        if season in gaps.index:
            g = gaps.loc[season]
            entry["gap"] = {"low_pct": _num(g["gap_low"] * 100), "high_pct": _num(g["gap_high"] * 100),
                            "methods": _text(g["methods"]), "confidence": g["confidence"], "direction": _text(g["direction"]),
                            "reason": _text(g["reason"]), "timing_evidence": g["timing_evidence"]}
        seasons[str(int(season))] = entry
    return {"district": STATE, "base_season": base, "seasons": seasons, "caveat": GAP_CAVEAT, "explanations": EXPLANATIONS}
