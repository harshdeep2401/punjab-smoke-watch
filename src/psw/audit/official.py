"""Official district fire counts, entered by hand in data/manual/official_counts.csv with a source on every row.

These counts are also satellite-based (PRSC / ISRO protocol), so they are not independent field truth.
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd

from psw import config

COLUMNS = ["season", "district", "count", "source_url", "method_note"]
# Created from Sangrur after our boundary file's date: counted separately in the sources, folded back here (D10).
MERGE_INTO = {"Malerkotla": "Sangrur"}


def known_districts(geo_path=None) -> set[str]:
    names = gpd.read_file(geo_path or config.GEO / "punjab_districts.geojson")["district"]
    return set(names) | set(MERGE_INTO)


def validate(df: pd.DataFrame, districts: set[str]) -> pd.DataFrame:
    """Check the raw CSV (all values as strings) and return it with integer season and count.

    Raises ValueError listing every problem, with the line numbers of the file.
    """
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"official counts: missing columns {missing}")
    if df.empty:
        raise ValueError("official counts: the file has no rows")
    df = df[COLUMNS].fillna("").apply(lambda s: s.str.strip())

    problems: list[str] = []

    def flag(bad: pd.Series, what: str) -> None:
        if bad.any():
            problems.append(f"{what} (line {', '.join(str(i + 2) for i in df.index[bad])})")

    for c in COLUMNS:
        flag(df[c] == "", f"blank {c}")
    flag((df["season"] != "") & ~df["season"].str.fullmatch(r"20\d\d"), "season is not a year")
    flag((df["count"] != "") & ~df["count"].str.fullmatch(r"\d+"), "count is not a whole number")
    unknown = (df["district"] != "") & ~df["district"].isin(districts)
    flag(unknown, f"unknown district {sorted(set(df.loc[unknown, 'district']))}")
    flag((df["source_url"] != "") & ~df["source_url"].str.fullmatch(r"https?://[^\s;,|]+"),
         "source_url must be exactly one web address")
    flag(df.duplicated(["season", "district"], keep=False), "duplicate season and district")
    for part, whole in MERGE_INTO.items():
        has_whole = set(df.loc[df["district"] == whole, "season"])
        flag((df["district"] == part) & ~df["season"].isin(has_whole), f"{part} has no {whole} row for its season")

    if problems:
        raise ValueError("official counts: " + "; ".join(problems))
    return df.astype({"season": int, "count": int})


def merge_new_districts(df: pd.DataFrame) -> pd.DataFrame:
    """Add each newer district's count to its parent's row, and say so in the method note."""
    out = df.copy()
    for part, whole in MERGE_INTO.items():
        for _, row in df[df["district"] == part].iterrows():
            at = (out["season"] == row["season"]) & (out["district"] == whole)
            if (out.loc[at, "source_url"] != row["source_url"]).any():
                raise ValueError(f"official counts: {part} and {whole} {row['season']} cite different sources")
            out.loc[at, "count"] += row["count"]
            out.loc[at, "method_note"] += f"; includes {part} (D10)"
        out = out[out["district"] != part]
    return out.sort_values(["season", "district"]).reset_index(drop=True)


def load_official(path: Path | None = None, geo_path=None) -> pd.DataFrame:
    """season, district, count, source_url, method_note. One row per district of our boundary file."""
    path = path or config.DATA / "manual" / "official_counts.csv"
    raw = pd.read_csv(path, dtype=str, keep_default_na=False, skip_blank_lines=False)
    return merge_new_districts(validate(raw, known_districts(geo_path)))
