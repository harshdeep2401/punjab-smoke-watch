"""District stock of crop-residue machines, from data/manual/crm_machines.csv (D24).

Context only: it is shown to people and to the Dispatcher and is never an input to the allocation. The counts are
machines supplied on subsidy since 2018-19, which says nothing about how many still work or can be moved.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from psw import config
from psw.audit.official import MERGE_INTO, known_districts

COLUMNS = ["district", "machine_type", "count", "period", "source_url", "method_note"]
PUBLISHED = date(2024, 11, 26)  # the status report's date: nothing dated earlier may show these figures
CAVEAT = ("These are machines supplied on subsidy from 2018-19 to 2024-25, as reported on 26 Nov 2024. They do not show "
          "how many still work or can be moved, and they are not used in the allocation.")


def validate(df: pd.DataFrame, districts: set[str]) -> pd.DataFrame:
    """Check the raw CSV (all values as strings) and return it with an integer count. Raises ValueError."""
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"machine stock: missing columns {missing}")
    if df.empty:
        raise ValueError("machine stock: the file has no rows")
    df = df[COLUMNS].fillna("").apply(lambda s: s.str.strip())
    problems: list[str] = []

    def flag(bad: pd.Series, what: str) -> None:
        if bad.any():
            problems.append(f"{what} (line {', '.join(str(i + 2) for i in df.index[bad][:10])})")

    for c in COLUMNS:
        flag(df[c] == "", f"blank {c}")
    flag((df["count"] != "") & ~df["count"].str.fullmatch(r"\d+"), "count is not a whole number")
    unknown = (df["district"] != "") & ~df["district"].isin(districts)
    flag(unknown, f"unknown district {sorted(set(df.loc[unknown, 'district']))}")
    flag((df["source_url"] != "") & ~df["source_url"].str.fullmatch(r"https?://[^\s;,|]+"),
         "source_url must be exactly one web address")
    flag(df.duplicated(["district", "machine_type"], keep=False), "duplicate district and machine type")
    types = df.groupby("district")["machine_type"].apply(frozenset)
    if types.nunique() > 1:
        problems.append("districts do not all have the same machine types")
    if problems:
        raise ValueError("machine stock: " + "; ".join(problems))
    return df.astype({"count": int})


def load_machine_stock(path: Path | None = None, geo_path=None) -> pd.DataFrame:
    """One row per district of our boundary file and machine type; newer districts are added to their parent (D10)."""
    path = path or config.DATA / "manual" / "crm_machines.csv"
    df = validate(pd.read_csv(path, dtype=str, keep_default_na=False, skip_blank_lines=False), known_districts(geo_path))
    for part, whole in MERGE_INTO.items():
        merged = df["district"] == part
        if not merged.any():
            continue
        extra = df[merged].set_index("machine_type")["count"]
        at = df["district"] == whole
        df.loc[at, "count"] += df.loc[at, "machine_type"].map(extra).fillna(0).astype(int)
        df.loc[at, "method_note"] += f"; includes {part} (D10)"
        df = df[~merged]
    return df.sort_values(["district", "machine_type"]).reset_index(drop=True)
