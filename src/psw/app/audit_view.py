"""Pure helpers for the Audit page (no Streamlit import). Every number and every sentence the page shows about the
audit is computed here from the committed tables in data/processed/audit/.

A table that is missing or empty (for example when burned area or official counts could not be obtained) gives empty
frames or None values, never an error.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from psw import config
from psw.agents.audit_tools import SENSOR_NAMES, change_words
from psw.audit import gap
from psw.audit.sensors import NIGHT

STATE = gap.STATE
BASE = gap.BASES[0]
TABLES = ("sensor_trend", "diurnal_profile", "burned_vs_detected", "official_vs_satellite", "timing_indicators",
          "gap_range")
CAPTION = "Triangulation of satellite and official signals, not field verification."


def load_audit_tables(root: Path | str | None = None) -> dict[str, pd.DataFrame | None]:
    """name -> table, or None when the file is not there or cannot be read."""
    out: dict[str, pd.DataFrame | None] = {}
    for name in TABLES:
        path = Path(root or config.PROCESSED) / "audit" / f"{name}.parquet"
        try:
            out[name] = pd.read_parquet(path)
        except (OSError, ValueError):
            out[name] = None
    return out


def _rows(tables: dict, name: str, district: str | None = None, season: int | None = None) -> pd.DataFrame:
    t = tables.get(name)
    if t is None or t.empty:
        return pd.DataFrame()
    if district is not None:
        t = t[t["district"] == district]
    if season is not None:
        t = t[t["season"] == season]
    return t


def _value(rows: pd.DataFrame, column: str):
    if rows.empty or pd.isna(rows[column].iloc[0]):
        return None
    return rows[column].iloc[0]


def seasons(tables: dict) -> list[int]:
    t = _rows(tables, "sensor_trend")
    return [] if t.empty else sorted(int(s) for s in t["season"].unique())


def districts(tables: dict) -> list[str]:
    t = _rows(tables, "sensor_trend")
    return [] if t.empty else sorted(set(t["district"]) - {STATE})


# ---------- the Punjab headline, season by season ----------

def all_sensor_change(tables: dict, season: int, district: str = STATE) -> tuple[float | None, list[str]]:
    """Change since the base season in the detections of every sensor that has data for both seasons, summed.

    A sensor that starts later (NOAA-21) is left out, so the two seasons are compared on the same satellites.
    """
    t = _rows(tables, "sensor_trend", district)
    if t.empty:
        return None, []
    count = t.set_index(["sensor", "season"])["count"]
    both = [s for s in SENSOR_NAMES if (s, BASE) in count.index and (s, season) in count.index]
    before = sum(int(count[(s, BASE)]) for s in both)
    after = sum(int(count[(s, season)]) for s in both)
    change = gap.pct_change(after, before)
    return (None if pd.isna(change) else float(change)), [SENSOR_NAMES[s] for s in both]


def season_headline(tables: dict, season: int, district: str = STATE) -> dict:
    """The figures of one season against the base season. Missing signals are None."""
    snpp = _rows(tables, "sensor_trend", district, season)
    snpp = snpp[snpp["sensor"] == "snpp"] if not snpp.empty else snpp
    timing = _rows(tables, "timing_indicators", STATE)
    now = timing[timing["season"] == season] if not timing.empty else timing
    then = timing[timing["season"] == BASE] if not timing.empty else timing
    burned = _rows(tables, "burned_vs_detected", district, season)
    ranges = _rows(tables, "gap_range", district, season)
    combined, sensors = all_sensor_change(tables, season, district)
    return {
        "season": season,
        "snpp_change": _value(snpp, "change_pct"),
        "all_change": combined,
        "all_sensors": sensors,
        "night_share": _value(now, "night_share"),
        "base_night_share": _value(then, "night_share"),
        "burned_change": _value(burned, "burned_change_pct"),
        "gap_low": _value(ranges, "gap_low"),
        "gap_high": _value(ranges, "gap_high"),
        "confidence": _value(ranges, "confidence"),
        "direction": _value(ranges, "direction"),
        "reason": _value(ranges, "reason"),
        "methods": gap.method_words(_value(ranges, "methods")),
    }


def pct_text(value, signed: bool = True) -> str:
    if value is None or pd.isna(value):
        return "not available"
    return f"{value:+.1f}%" if signed else f"{value:.1f}%"


def gap_text(row: dict) -> str:
    """'77% to 84%', or why there is no range."""
    if row["gap_low"] is None:
        return "no range"
    return f"{row['gap_low'] * 100:.0f}% to {row['gap_high'] * 100:.0f}%"


def headline_text(row: dict) -> str:
    """One season in plain words. The wording follows the result."""
    season = row["season"]
    if row["snpp_change"] is None:
        return f"There is no S-NPP comparison for {season}."
    text = f"In {season}, Punjab's S-NPP detections were {change_words(row['snpp_change'], BASE)}"
    if row["burned_change"] is None:
        text += ". Burned area is not available, so there is nothing to compare them with."
    else:
        text += f", while MODIS burned area was {change_words(row['burned_change'], BASE)}."
    if row["gap_low"] is None:
        why = row["reason"] or "no reference signal"
        return text + f" There is insufficient evidence for a gap range ({why})."
    low, high = row["gap_low"] * 100, row["gap_high"] * 100
    if row["direction"] == "counts fell more than burned area":
        text += (f" Detections per burned km² fell by {low:.0f}% to {high:.0f}% (confidence: {row['confidence']}). "
                 "This is not the share of burning that is missed, and it does not prove hidden burning.")
    else:
        text += (f" The gap range is {low:.0f}% to {high:.0f}% (confidence: {row['confidence']}): "
                 f"{row['direction']}.")
    return text


# ---------- chart data ----------

def pass_time_frame(tables: dict, district: str = STATE) -> pd.DataFrame:
    """season, sensor (readable name), pass_label, count. Sensors in the usual order, S-NPP first."""
    t = _rows(tables, "diurnal_profile", district)
    if t.empty:
        return pd.DataFrame(columns=["season", "sensor", "pass_label", "count"])
    order = {s: i for i, s in enumerate(SENSOR_NAMES)}
    out = t.assign(at=t["sensor"].map(order), sensor=t["sensor"].map(SENSOR_NAMES)).sort_values(["at", "season"])
    return out[["season", "sensor", "pass_label", "count"]].reset_index(drop=True)


def night_visibility_note(frame: pd.DataFrame) -> str | None:
    """Said under the pass-time chart when night detections are too thin a slice to see there."""
    if frame.empty:
        return None
    night = frame[frame["pass_label"] == NIGHT]["count"].sum()
    share = night / frame["count"].sum() * 100 if frame["count"].sum() else 0
    if share >= 5:
        return None
    return (f"Night detections are {share:.1f}% of those drawn here, too thin to see at this scale; the night share "
            "chart below shows them for S-NPP.")


def night_share_frame(tables: dict) -> pd.DataFrame:
    """Punjab only (D23): season, night, total, share_pct, low_pct, high_pct for S-NPP."""
    t = _rows(tables, "timing_indicators", STATE)
    cols = ["season", "night", "total", "share_pct", "low_pct", "high_pct"]
    if t.empty:
        return pd.DataFrame(columns=cols)
    t = t.assign(share_pct=t["night_share"] * 100, low_pct=t["ci_low"] * 100, high_pct=t["ci_high"] * 100)
    return t.sort_values("season")[cols].reset_index(drop=True)


SNPP_SERIES, BURNED_SERIES = "S-NPP detections", "MODIS burned area"


def indexed_frame(tables: dict, district: str = STATE) -> pd.DataFrame:
    """season, series, value, index (base season = 100) for detections and burned area."""
    t = _rows(tables, "burned_vs_detected", district)
    rows = []
    if not t.empty:
        t = t.sort_values("season").set_index("season")
        for column, name in (("detections", SNPP_SERIES), ("burned_km2", BURNED_SERIES)):
            base = t[column].get(BASE)
            if base is None or pd.isna(base) or base == 0:
                continue
            rows += [{"season": int(s), "series": name, "value": float(v), "index": float(v) / float(base) * 100}
                     for s, v in t[column].items() if not pd.isna(v)]
    return pd.DataFrame(rows, columns=["season", "series", "value", "index"])


def official_frame(tables: dict, district: str = STATE) -> pd.DataFrame:
    """season, official, snpp, ratio (official / S-NPP). Empty when there are no official counts."""
    t = _rows(tables, "official_vs_satellite", district)
    cols = ["season", "official", "snpp", "ratio"]
    return pd.DataFrame(columns=cols) if t.empty else t.sort_values("season")[cols].reset_index(drop=True)


def official_sources(tables: dict) -> list[str]:
    t = _rows(tables, "official_vs_satellite")
    if t.empty:
        return []
    one = t[t["district"] != STATE]
    return sorted({url for urls in one["source_url"].dropna() for url in str(urls).split()})


# ---------- districts ----------

TABLE_COLUMNS = ["district", "reading", "confidence", "gap low (%)", "gap high (%)", "S-NPP detections",
                 "detections change (%)", "burned area (km²)", "burned change (%)", "note"]
NUMERIC_COLUMNS = TABLE_COLUMNS[3:-1]


def district_table(tables: dict, season: int) -> pd.DataFrame:
    """One row per district for a season, most S-NPP detections first. `note` says why there is no range."""
    rows = []
    for d in districts(tables):
        trend = _rows(tables, "sensor_trend", d)
        count = trend[trend["sensor"] == "snpp"].set_index("season")["count"] if not trend.empty else pd.Series(dtype=int)
        burned = _rows(tables, "burned_vs_detected", d, season)
        head = season_headline(tables, season, d)
        ranged = head["gap_low"] is not None
        notes = [] if ranged else [head["reason"] or "no gap estimate"]
        if warning := gap.small_base(count.get(BASE)):
            notes.append(warning)
        rows.append({
            "district": d,
            "S-NPP detections": count.get(season),
            "detections change (%)": head["snpp_change"],
            "burned area (km²)": _value(burned, "burned_km2"),
            "burned change (%)": head["burned_change"],
            "gap low (%)": None if head["gap_low"] is None else head["gap_low"] * 100,
            "gap high (%)": None if head["gap_high"] is None else head["gap_high"] * 100,
            "reading": head["direction"] if ranged else gap.INSUFFICIENT,
            "confidence": head["confidence"] if ranged else None,
            "note": "; ".join(notes),
        })
    out = pd.DataFrame(rows, columns=TABLE_COLUMNS)
    out[NUMERIC_COLUMNS] = out[NUMERIC_COLUMNS].apply(pd.to_numeric)  # a missing value is NaN, shown as "-"
    out["confidence"] = out["confidence"].fillna("-")
    return out.sort_values(["S-NPP detections", "district"], ascending=[False, True]).reset_index(drop=True)


def district_summary(table: pd.DataFrame, season: int) -> str:
    """How many districts have a range, and which way those ranges point."""
    if table.empty:
        return "No district figures are available."
    ranged = table[table["gap low (%)"].notna()]
    text = f"Of {len(table)} districts in {season}, {len(ranged)} have a gap range"
    if ranged.empty:
        return text + "."
    fell = int(((ranged["gap low (%)"] > 0)).sum())
    none = int(((ranged["gap high (%)"] <= 0)).sum())
    unclear = len(ranged) - fell - none
    text += (f": in {fell} counts fell more than burned area, in {none} there is no sign of a gap and in {unclear} the "
             "direction is unclear.")
    levels = sorted(ranged["confidence"].unique())
    return text + f" Confidence for these ranges is {' or '.join(levels)}."


def night_counts(tables: dict, district: str) -> dict[int, int]:
    """S-NPP night detections per season. For a district only counts are shown, never a share (D23)."""
    t = _rows(tables, "diurnal_profile", district)
    if t.empty:
        return {}
    t = t[(t["sensor"] == "snpp") & (t["pass_label"] == NIGHT)]
    return {int(s): int(n) for s, n in zip(t["season"], t["count"])}
