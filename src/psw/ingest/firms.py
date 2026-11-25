"""NASA FIRMS VIIRS S-NPP fire detections, cached per chunk (including empty chunks)."""
from __future__ import annotations

import io
import logging
import time
from datetime import date, timedelta

import pandas as pd

from psw import config
from psw.ingest import http

log = logging.getLogger(__name__)

URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv/{key}/{source}/{bbox}/{day_range}/{start}"
DAY_RANGE = 5  # API maximum (verified in Phase 0)
SLEEP = 0.3
KEEP = ["latitude", "longitude", "acq_date", "acq_time", "frp", "confidence", "daynight", "source", "season"]
DEDUPE = ["latitude", "longitude", "acq_date", "acq_time"]


def _cache_path(source: str, start: date, day_range: int):
    return config.RAW / "firms" / source / f"{start:%Y-%m-%d}_{day_range}.csv"


def fetch_chunk(source: str, start: date, day_range: int, key: str) -> pd.DataFrame:
    """One FIRMS call. `start` is the first day; the range counts forward. Cached on disk, empties too."""
    path = _cache_path(source, start, day_range)
    if path.exists():
        return pd.read_csv(path)
    bbox = ",".join(str(x) for x in config.BBOX)
    r = http.get(URL.format(key=key, source=source, bbox=bbox, day_range=day_range, start=f"{start:%Y-%m-%d}"),
                 secret=key)
    text = r.text
    header = text.split("\n", 1)[0]
    if r.status_code != 200 or "latitude" not in header:
        msg = f"FIRMS {source} {start}: HTTP {r.status_code}: {text[:100]!r}"
        raise RuntimeError(msg.replace(key, "***") if key else msg)
    df = pd.read_csv(io.StringIO(text))
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)  # a header-only file records an empty chunk
    time.sleep(SLEEP)
    return df


def season_window(year: int) -> tuple[date, date]:
    return date(year, *config.SEASON_START), date(year, *config.SEASON_END)


def fetch_season(year: int, key: str) -> pd.DataFrame:
    """All chunks Sep 15..Nov 30. SP first, NRT as fallback when SP returns nothing for a chunk."""
    start, end = season_window(year)
    parts = []
    d = start
    while d <= end:
        n = min(DAY_RANGE, (end - d).days + 1)
        chunk = fetch_chunk("VIIRS_SNPP_SP", d, n, key)
        source = "VIIRS_SNPP_SP"
        if chunk.empty:
            chunk = fetch_chunk("VIIRS_SNPP_NRT", d, n, key)
            source = "VIIRS_SNPP_NRT"
        if not chunk.empty:
            parts.append(chunk.assign(source=source))
        d += timedelta(days=n)
    if not parts:
        return pd.DataFrame(columns=[c for c in KEEP if c != "season"])
    df = pd.concat(parts, ignore_index=True)
    return df.drop_duplicates(subset=DEDUPE).reset_index(drop=True)


def load_all_fires(seasons: list[int], key: str | None = None) -> pd.DataFrame:
    key = config.FIRMS_MAP_KEY if key is None else key
    frames = [fetch_season(y, key).assign(season=y) for y in seasons]
    df = pd.concat(frames, ignore_index=True)
    df["acq_date"] = pd.to_datetime(df["acq_date"]).dt.date
    df["confidence"] = df["confidence"].astype(str).str.strip().str.lower()
    low = df["confidence"] == "l"
    log.info("dropping %d of %d low-confidence detections", int(low.sum()), len(df))
    df = df[~low]
    return df[KEEP].reset_index(drop=True)
