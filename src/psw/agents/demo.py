"""Demo cache: pick three demo dates in the latest test season and precompute their briefings, so the app can be
shown offline. Dates are chosen from the data by fixed rules, not typed in.

    peak       the day with the most detections in Punjab
    early      the earliest day with a fire-driven flag (spike or high_count)
    moderate   among the other days with a fire-driven flag, the one whose Punjab total is closest to half the peak's
Early-season days before any flag would give an empty watchlist, so "early" is the first day that has something to brief.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from psw import config
from psw.agents import orchestrator, sentinel
from psw.agents.tools import DataStore

FIRE_FLAGS = ("spike", "high_count")
EXTRA_DISTRICTS = ["Ludhiana", "Sangrur"]  # briefed on the peak day in addition to the top flagged ones


def demo_season(store: DataStore) -> int:
    return int(store.forecasts["season"].max())


def pick_demo_dates(store: DataStore, season: int | None = None) -> dict[str, str]:
    season = demo_season(store) if season is None else season
    dd = store.district_daily
    dd = dd[dd["season"] == season]
    totals = dd.groupby("date")["fire_count"].sum()
    peak = str(totals.idxmax())
    fire_days = [d for d in sorted(totals.index) if any(f.split(":")[0] in FIRE_FLAGS for r in sentinel.scan(d, store) for f in r["flags"])]
    if not fire_days:
        raise ValueError(f"no day in {season} has a fire-driven flag")
    early = str(fire_days[0])
    others = [d for d in fire_days if d not in (peak, early)]
    moderate = str(min(others, key=lambda d: abs(totals[d] - totals[peak] / 2))) if others else early
    return {"peak": peak, "moderate": moderate, "early": early}


def demo_path(root: Path | None = None) -> Path:
    return Path(root or config.PROCESSED) / "demo.json"


def load_demo(root: Path | None = None) -> dict | None:
    try:
        return json.loads(demo_path(root).read_text())
    except (OSError, json.JSONDecodeError):
        return None


def build_demo_cache(llm=None, store: DataStore | None = None, kb=None, root: Path | None = None,
                     top_n: int = 3, force: bool = False, use_planner: bool = False, log=print) -> dict:
    """Daily watch for each demo date, plus Ludhiana and Sangrur on the peak day. Saved agent briefings are reused
    unless `force`. Writes demo.json and returns it."""
    store = store or DataStore()
    dates = pick_demo_dates(store)
    log(f"demo dates: {dates}")
    modes: dict[str, list[str]] = {}
    for label, day in dates.items():
        watch = orchestrator.run_daily_watch(day, top_n=top_n, llm=llm, store=store, kb=kb, root=root, reuse=not force, use_planner=use_planner)
        modes[day] = [f"{b['district']}:{b['mode']}" for b in watch["briefings"]]
        log(f"{label} {day}: {watch['n_flagged']} flagged, briefed {modes[day]}")
    for d in EXTRA_DISTRICTS:
        rec = orchestrator.run_briefing(d, dates["peak"], llm=llm, store=store, kb=kb, root=root, reuse=not force,
                                        use_planner=use_planner)
        modes[dates["peak"]].append(f"{d}:{rec.mode}")
        log(f"peak {dates['peak']} {d}: {rec.mode}, verified={rec.verification['ok']}")
    out = {**dates, "season": demo_season(store), "default_district": EXTRA_DISTRICTS[0], "briefed": modes,
           "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    p = demo_path(root)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1))
    return out
