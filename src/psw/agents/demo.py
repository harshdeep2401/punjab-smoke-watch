"""Demo cache: pick the demo dates in the latest test season and precompute their briefings, audits and dispatches, so
the app can be shown offline. Dates and districts are chosen from the data by fixed rules, not typed in.

    peak       the day with the most detections in Punjab
    early      the earliest day with a fire-driven flag (spike or high_count)
    moderate   among the other days with a fire-driven flag, the one whose Punjab total is closest to half the peak's
Early-season days before any flag would give an empty watchlist, so "early" is the first day that has something to brief.

Part 2: audits of Punjab for each test season and of the latest season's 3 busiest districts, and dispatches issued
3 days before the peak, on the peak and on a late-season day (the first day after the peak on which the detections of the
last 3 days are under a quarter of the season's highest 3-day total).

A saved agent brief is reused unless asked to redo it, and a run that falls back to the template never replaces one.
A saved dispatch counts only while it still passes today's Verifier.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from psw import config
from psw.agents import orchestrator, sentinel
from psw.agents.audit_tools import AuditStore
from psw.agents.dispatch_tools import ResponseStore, Settings
from psw.agents.tools import DataStore
from psw.response import allocate, risk

FIRE_FLAGS = ("spike", "high_count")
EXTRA_DISTRICTS = ["Ludhiana", "Sangrur"]  # briefed on the peak day in addition to the top flagged ones
AUDIT_DISTRICTS = 3
LATE_SHARE = 0.25


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


# ---------- Part 2: audits and dispatches ----------

def audit_targets(store: AuditStore, seasons: list[int], top: int = AUDIT_DISTRICTS) -> list[tuple[str, int]]:
    """Punjab for each of `seasons`, then the districts with the most S-NPP detections in the latest of them."""
    latest = max(seasons)
    t = store.table("sensor_trend")
    busiest = t[(t["sensor"] == "snpp") & (t["season"] == latest) & (t["district"] != "Punjab")]
    busiest = busiest.sort_values(["count", "district"], ascending=[False, True]).head(top)
    return [("Punjab", s) for s in sorted(seasons)] + [(d, latest) for d in busiest["district"]]


def late_date(daily, peak: date) -> date:
    """The first day after `peak` whose last 3 days saw under a quarter of the season's highest 3-day total."""
    season = daily[daily["season"] == peak.year]
    totals = season.groupby("date")["fire_count"].sum().sort_index()
    recent = totals.rolling(risk.HORIZON, min_periods=1).sum()
    quiet = [d for d, n in recent.items() if d > peak and n < LATE_SHARE * recent.max()]
    return quiet[0] if quiet else recent.index[-1]


def dispatch_dates(daily, peak: date, issued: set[date]) -> list[date]:
    """3 days before the peak, the peak and a late-season day, keeping those a forecast was issued for."""
    wanted = [peak - timedelta(days=risk.HORIZON), peak, late_date(daily, peak)]
    return [d for d in dict.fromkeys(wanted) if d in issued]


def _cached(saved, redo: bool, make, save):
    """(record, what happened). `saved` is the reusable agent record or None; `make` runs the agent without saving."""
    if saved is not None and not redo:
        return saved, "reused"
    new = make()
    if new.mode == "template" and saved is not None:
        return saved, "kept the saved one: the new run fell back to the template"
    save(new)
    return new, "written"


def build_part2_cache(llm=None, audit_store: AuditStore | None = None, response_store: ResponseStore | None = None,
                      kb=None, root: Path | None = None, force: bool = False, redo_dispatch: bool = False,
                      log=print) -> dict:
    """Audits and dispatches for the demo. Adds `audits` and `dispatches` to demo.json and returns it."""
    audit_store, response_store = audit_store or AuditStore(), response_store or ResponseStore()
    peak = date.fromisoformat(load_demo(root)["peak"])
    seasons = sorted(int(s) for s in response_store.forecasts["season"].unique())
    out = load_demo(root)

    out["audits"] = []
    for district, season in audit_targets(audit_store, seasons):
        rec, what = _cached(
            orchestrator.load_reusable_audit(district, season, root), force,
            lambda: orchestrator.run_audit(district, season, llm=llm, store=audit_store, kb=kb, save=False),
            lambda r: orchestrator.save_audit(r, root))
        out["audits"].append({"district": district, "season": season, "mode": rec.mode,
                              "verified": rec.verification["ok"]})
        log(f"audit {district} {season}: {rec.mode}, verified={rec.verification['ok']} ({what})")

    settings = Settings(allocate.DEFAULT_UNITS, allocate.DEFAULT_MAX_SHARE, True)
    out["dispatches"] = []
    if kb is None:
        from psw.rag.index import KnowledgeBase
        kb = KnowledgeBase()
    for day in dispatch_dates(response_store.daily, peak, set(response_store.dates)):
        saved = orchestrator.load_reusable_dispatch(str(day), settings, None, root)
        if saved is not None and not orchestrator.recheck_dispatch(saved.to_dict(), response_store, kb).ok:
            log(f"dispatch {day}: the saved brief no longer passes the Verifier")
            saved = None
        rec, what = _cached(
            saved, force or redo_dispatch,
            lambda: orchestrator.run_dispatch(day, units=settings.units, max_share=settings.max_share,
                                              use_likely=settings.use_likely, llm=llm, store=response_store, kb=kb,
                                              save=False),
            lambda r: orchestrator.save_dispatch(r, root))
        out["dispatches"].append({"date": str(day), "units": rec.units, "mode": rec.mode,
                                  "verified": rec.verification["ok"]})
        log(f"dispatch {day}: {rec.mode}, verified={rec.verification['ok']} ({what})")

    out["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    demo_path(root).write_text(json.dumps(out, indent=1))
    return out
