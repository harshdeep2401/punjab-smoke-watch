"""Phase 0 feasibility checks. Run: uv run python scripts/feasibility.py [firms|aq|weather|geo]

Prints what each data source really returns. Never prints API keys.
"""
from __future__ import annotations

import datetime as dt
import io
import json
import sys

import pandas as pd
import requests

from psw import config

AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WX_URL = "https://archive-api.open-meteo.com/v1/archive"
FIRMS_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv/{key}/{src}/{bbox}/{days}/{date}"
LDH = config.CITIES["Ludhiana"]


def firms_fetch(src: str, days: int, date: str) -> pd.DataFrame | str:
    bbox = ",".join(str(x) for x in config.BBOX)
    url = FIRMS_URL.format(key=config.FIRMS_MAP_KEY, src=src, bbox=bbox, days=days, date=date)
    r = requests.get(url, timeout=120)
    text = r.text
    if r.status_code != 200 or text.lstrip().startswith(("Invalid", "Error")) or "," not in text.split("\n")[0]:
        return f"HTTP {r.status_code}: {text[:120]!r}"
    return pd.read_csv(io.StringIO(text))


def check_firms() -> None:
    print("== FIRMS ==")
    if not config.FIRMS_MAP_KEY:
        print("FIRMS_MAP_KEY not set in .env: skipped")
        return
    recent = (dt.date.today() - dt.timedelta(days=3)).isoformat()
    r = firms_fetch("VIIRS_SNPP_NRT", 1, recent)
    print("NRT 1 day", recent, "->", r if isinstance(r, str) else f"{len(r)} rows; cols={list(r.columns)}")
    for n in (10, 5):
        r = firms_fetch("VIIRS_SNPP_NRT", n, recent)
        print(f"NRT DAY_RANGE={n} ->", r if isinstance(r, str) else f"OK {len(r)} rows")
        if not isinstance(r, str):
            break
    for y in range(2021, dt.date.today().year + 1):
        d = f"{y}-11-01"
        r = firms_fetch("VIIRS_SNPP_SP", 5, d)
        print(f"SP {d} (5 days) ->", r if isinstance(r, str) else f"{len(r)} rows")
    for y in range(2021, dt.date.today().year + 1):
        d = f"{y}-11-01"
        r = firms_fetch("VIIRS_SNPP_NRT", 5, d)
        print(f"NRT {d} (5 days) ->", r if isinstance(r, str) else f"{len(r)} rows")


def aq_nonnull(start: str, end: str) -> int:
    r = requests.get(AQ_URL, params={"latitude": LDH[0], "longitude": LDH[1], "hourly": "pm2_5",
                                     "start_date": start, "end_date": end, "timezone": "Asia/Kolkata"}, timeout=60)
    if r.status_code != 200:
        return 0
    return sum(v is not None for v in r.json()["hourly"]["pm2_5"])


def check_aq() -> None:
    print("== Open-Meteo air quality (Ludhiana, pm2_5) ==")
    for y in range(2020, dt.date.today().year + 1):
        print(f"{y}-11-01..07 non-null hours:", aq_nonnull(f"{y}-11-01", f"{y}-11-07"))
    lo, hi = dt.date(2020, 1, 1), dt.date(2023, 1, 1)  # earliest date with data: binary search by month start
    while (hi - lo).days > 1:
        mid = lo + (hi - lo) / 2
        if aq_nonnull(mid.isoformat(), (mid + dt.timedelta(days=2)).isoformat()) > 0:
            hi = mid
        else:
            lo = mid
    print("earliest date with non-null pm2_5 (approx):", hi)
    for y in range(2022, dt.date.today().year + 1):
        s, e = f"{y}-09-15", f"{y}-11-30"
        e = min(dt.date.fromisoformat(e), dt.date.today() - dt.timedelta(days=1)).isoformat()
        if e < s:
            continue
        r = requests.get(AQ_URL, params={"latitude": LDH[0], "longitude": LDH[1], "hourly": "pm2_5",
                                         "start_date": s, "end_date": e, "timezone": "Asia/Kolkata"}, timeout=60)
        v = r.json()["hourly"]["pm2_5"] if r.status_code == 200 else []
        nn = sum(x is not None for x in v)
        print(f"season {y} {s}..{e}: {nn}/{len(v)} non-null hours")


def check_weather() -> None:
    print("== Open-Meteo archive weather (Ludhiana) ==")
    vars_ = "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,boundary_layer_height"
    for y in range(2020, dt.date.today().year + 1):
        s, e = f"{y}-10-01", f"{y}-10-07"
        r = requests.get(WX_URL, params={"latitude": LDH[0], "longitude": LDH[1], "hourly": vars_,
                                         "start_date": s, "end_date": e, "timezone": "Asia/Kolkata"}, timeout=60)
        if r.status_code != 200:
            print(y, "HTTP", r.status_code, r.text[:100])
            continue
        h = r.json()["hourly"]
        print(y, {k: sum(x is not None for x in h[k]) for k in vars_.split(",")}, "of", len(h["time"]))
    # archive lag: latest date with non-null temperature
    today = dt.date.today()
    for lag in range(1, 12):
        d = (today - dt.timedelta(days=lag)).isoformat()
        r = requests.get(WX_URL, params={"latitude": LDH[0], "longitude": LDH[1], "hourly": "temperature_2m",
                                         "start_date": d, "end_date": d, "timezone": "Asia/Kolkata"}, timeout=60)
        if r.status_code == 200 and any(x is not None for x in r.json()["hourly"]["temperature_2m"]):
            print("archive latest non-null day:", d, f"(lag {lag} days)")
            break


def check_geo() -> None:
    import geopandas as gpd

    print("== District GeoJSON candidates ==")
    cands = {
        "a_udit-001": "https://raw.githubusercontent.com/udit-001/india-maps-data/main/geojson/states/punjab.geojson",
        "c_datameet": "https://raw.githubusercontent.com/datameet/maps/master/Districts/Census_2011/2011_Dist.json",
    }
    for name, url in cands.items():
        r = requests.get(url, timeout=120)
        print(name, url, "->", r.status_code, len(r.content), "bytes")
        if r.status_code == 200:
            g = gpd.GeoDataFrame.from_features(json.loads(r.text)["features"])
            print("  columns:", list(g.columns), "rows:", len(g))
            print("  sample:", g.drop(columns="geometry").head(30).to_dict("records")[:3])


if __name__ == "__main__":
    which = sys.argv[1:] or ["firms", "aq", "weather", "geo"]
    for w in which:
        {"firms": check_firms, "aq": check_aq, "weather": check_weather, "geo": check_geo}[w]()
