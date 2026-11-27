"""CLI entry point: uv run python -m psw.pipeline <command>. I/O lives here."""
from __future__ import annotations

import argparse
import logging

from psw import config


def cmd_ingest(_args) -> None:
    from psw.ingest import airquality, firms, weather

    if not config.FIRMS_MAP_KEY:
        raise SystemExit("FIRMS_MAP_KEY is not set in .env")
    fires = firms.load_all_fires(config.SEASONS)
    print(f"fires: {len(fires)} rows; per season: {fires.groupby('season').size().to_dict()}")
    aq = airquality.daily_pm25(config.SEASONS)
    print(f"pm25 daily: {len(aq)} rows, {int(aq['pm25_mean'].isna().sum())} NaN")
    wx = weather.daily_weather(config.SEASONS)
    print(f"weather daily: {len(wx)} rows, {int(wx.isna().any(axis=1).sum())} rows with NaN")


def cmd_process(_args) -> None:
    import geopandas as gpd

    from psw.ingest import airquality, firms, weather
    from psw.processing import districts

    geo = config.GEO / "punjab_districts.geojson"
    names = sorted(gpd.read_file(geo)["district"])
    fires = districts.assign_districts(firms.load_all_fires(config.SEASONS), geo)
    for c in ("latitude", "longitude"):
        fires[c] = fires[c].round(4)
    config.PROCESSED.mkdir(parents=True, exist_ok=True)
    fires.to_parquet(config.PROCESSED / "fires.parquet", index=False)
    daily = districts.district_daily(fires, names, config.SEASONS)
    daily.to_parquet(config.PROCESSED / "district_daily.parquet", index=False)
    airquality.daily_pm25(config.SEASONS).to_parquet(config.PROCESSED / "pm25_daily.parquet", index=False)
    weather.daily_weather(config.SEASONS).to_parquet(config.PROCESSED / "weather_daily.parquet", index=False)
    print(f"fires: {len(fires)} rows, {int(fires['in_punjab'].sum())} inside Punjab; "
          f"district_daily: {len(daily)} rows for {len(names)} districts")


def cmd_train(_args) -> None:
    import pandas as pd

    from psw.models import evaluate
    from psw.processing import features as feat

    P = config.PROCESSED
    fires = pd.read_parquet(P / "fires.parquet")
    table = feat.build_features(fires, pd.read_parquet(P / "pm25_daily.parquet"), pd.read_parquet(P / "weather_daily.parquet"))
    table.to_parquet(P / "features.parquet", index=False)
    print(f"features: {len(table)} rows, {table.shape[1]} columns")
    m = evaluate.run_all(table)
    print(f"test seasons {m['seasons_test']}, conformal applied: {m['conformal_applied']}")
    print(f"{'variant':12s} {'MAE':>7s} {'pinball':>8s} {'cov80':>7s} {'width':>7s}")
    for name in evaluate.VARIANTS:
        o = m["variants"][name]["overall"]
        print(f"{name:12s} {o['mae']:7.2f} {o['pinball']:8.2f} {o['coverage80']:7.1%} {o['band_width']:7.1f}")
    print("top features (gain):", {k: round(v) for k, v in list(m["feature_importance"].items())[:6]})


def cmd_kb(_args) -> None:
    from psw.rag import build_kb, index

    print("First run downloads the embedding model (~130 MB) and caches it.")
    paths = build_kb.generate_docs()
    docs = index.load_docs(config.KB / "curated", config.KB / "generated")
    table = index.build_index(docs)
    size = sum((config.PROCESSED / f).stat().st_size for f in (index.VECTORS_FILE, index.CHUNKS_FILE))
    print(f"generated {len(paths)} docs; indexed {len(docs)} docs into {len(table)} chunks ({size / 1e6:.2f} MB)")


def cmd_kb_eval(_args) -> None:
    import json

    from psw.rag import evaluate
    from psw.rag.index import KnowledgeBase

    items = evaluate.load_eval()
    report = evaluate.evaluate(KnowledgeBase(), items, k=3)
    (config.PROCESSED / evaluate.REPORT_FILE).write_text(json.dumps(report, indent=1))
    print(f"retrieval eval: {report['n']} queries, hit@{report['k']} (hit@1)")
    for mode, m in report["modes"].items():
        print(f"  {mode:10s} {m['hit_at_k']:2d}/{report['n']}  ({m['hit_at_1']:2d})")
        for r in m["items"]:
            if r["rank"] is None:
                print(f"      miss: {r['query']!r} wanted {r['expected']} got {r['got']}")
    print(f"Saved data/processed/{evaluate.REPORT_FILE}")


MAX_PARQUET_MB = 40


def cmd_audit_ingest(_args) -> None:
    from psw.audit import sensors

    if not config.FIRMS_MAP_KEY:
        raise SystemExit("FIRMS_MAP_KEY is not set in .env")
    ms = sensors.load_multisensor(config.SEASONS)
    for c in ("latitude", "longitude"):
        ms[c] = ms[c].round(4)
    config.PROCESSED.mkdir(parents=True, exist_ok=True)
    out = config.PROCESSED / "fires_multisensor.parquet"
    ms.to_parquet(out, index=False)
    if out.stat().st_size > MAX_PARQUET_MB * 1e6:
        ms = ms[ms["district"].notna()]
        ms.to_parquet(out, index=False)
    inside = ms[ms["district"].notna()]
    print(f"fires_multisensor: {len(ms)} rows ({out.stat().st_size / 1e6:.1f} MB), {len(inside)} inside Punjab")
    for title, table in (("all of the bounding box", ms), ("inside Punjab", inside)):
        print(f"rows per sensor and season, {title}:")
        print(table.groupby(["sensor", "season"]).size().unstack(fill_value=0).to_string())
    _ingest_burned_area()
    _check_official_counts()
    _ingest_district_weather()


def _ingest_burned_area() -> None:
    from psw.audit import burned_area

    area = burned_area.load_burned_area(config.SEASONS)
    area.to_parquet(config.PROCESSED / "burned_area.parquet", index=False)
    totals = area.groupby("season")["burned_km2"].sum()
    print(f"burned_area: {len(area)} district-seasons; Punjab km2 per season: "
          f"{ {int(k): int(round(v)) for k, v in totals.items()} }")


def _check_official_counts() -> None:
    from psw.audit import official

    counts = official.load_official()  # raises on a bad row, so a hand edit cannot slip through
    totals = counts.groupby("season")["count"].sum()
    print(f"official_counts: {len(counts)} district-seasons; Punjab totals per season: "
          f"{ {int(k): int(v) for k, v in totals.items()} }")


def _ingest_district_weather() -> None:
    from psw.ingest import weather

    wx = weather.district_daily_weather(config.SEASONS)
    wx.to_parquet(config.PROCESSED / "district_weather_daily.parquet", index=False)
    print(f"district_weather_daily: {len(wx)} rows, {wx['district'].nunique()} districts, "
          f"{wx['date'].min()} to {wx['date'].max()}")


def _load_audit_inputs():
    """(multi-sensor fires, burned area, official counts) from the committed files."""
    import pandas as pd

    from psw.audit import official

    ms = pd.read_parquet(config.PROCESSED / "fires_multisensor.parquet")
    return ms, pd.read_parquet(config.PROCESSED / "burned_area.parquet"), official.load_official()


def _write_audit_tables() -> dict:
    """Compute the audit tables and the Punjab summary from committed data. No network, no LLM."""
    import json

    from psw.audit import gap

    tables = gap.audit_tables(*_load_audit_inputs())
    out = config.PROCESSED / "audit"
    out.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_parquet(out / f"{name}.parquet", index=False)
    summary = gap.audit_summary(tables)
    (config.PROCESSED / "audit_summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False, allow_nan=False))
    print(f"audit: {len(tables)} tables in data/processed/audit/ and data/processed/audit_summary.json")
    state = tables["gap_range"]
    for r in state[state["district"] == gap.STATE].itertuples():
        span = r.reason if r.confidence == gap.INSUFFICIENT else f"{r.gap_low:.2f} to {r.gap_high:.2f}"
        print(f"  Punjab {r.season}: gap {span} | confidence {r.confidence} | {r.methods}")
    print(f"  {gap.GAP_CAVEAT}")
    return tables


AUDIT_SEASONS = (2024, 2025)  # the Punjab audits
AUDIT_TOP_DISTRICTS = 5  # plus the busiest districts of the latest season


def _no_llm(reason: str):
    """A stand-in that forces the template path without trying .env."""
    from psw.llm import LLMError

    class NoLLM:
        model = "none"

        def chat(self, *a, **k):
            raise LLMError(reason)
    return NoLLM()


def _llm_or_template(no_llm: bool):
    from psw.llm import LLM, LLMError

    if no_llm:
        return _no_llm("--no-llm")
    try:
        llm = LLM.from_env()
    except LLMError as e:
        print(f"No LLM ({e}); using the template fallback.")
        return _no_llm(str(e))
    print(f"LLM: {llm.model} (min {llm.min_interval_s:.0f}s between calls)")
    return llm


def _default_audits(tables: dict) -> list[tuple[str, int]]:
    """Punjab for the audit seasons, then the districts with the most S-NPP detections in the latest season."""
    trend = tables["sensor_trend"]
    latest = max(AUDIT_SEASONS)
    top = trend[(trend["sensor"] == "snpp") & (trend["season"] == latest) & (trend["district"] != "Punjab")]
    top = top.sort_values(["count", "district"], ascending=[False, True]).head(AUDIT_TOP_DISTRICTS)
    return [("Punjab", s) for s in AUDIT_SEASONS] + [(d, latest) for d in top["district"]]


def cmd_audit(args) -> None:
    tables = _write_audit_tables()
    if args.tables_only:
        return

    from psw.agents import orchestrator
    from psw.agents.audit_tools import AuditStore
    from psw.rag.index import KnowledgeBase

    if args.district:
        wanted = [(args.district, args.season or max(AUDIT_SEASONS))]
    elif args.season:
        wanted = [("Punjab", args.season)]
    else:
        wanted = _default_audits(tables)
    store, kb, llm = AuditStore(), KnowledgeBase(), _llm_or_template(args.no_llm)
    records = []
    for district, season in wanted:
        rec = orchestrator.run_audit(district, season, llm=llm, store=store, kb=kb, reuse=not args.force,
                                     use_planner=args.plan)
        _print_audit(rec)
        records.append(rec)
    modes = [r.mode for r in records]
    print(f"\n{len(records)} audits, {sum(r.verification['ok'] for r in records)} verified; "
          f"modes: { {m: modes.count(m) for m in sorted(set(modes))} }. Saved in data/processed/audits/")
    if hasattr(llm, "usage"):
        print("token usage:", llm.usage)


def _print_audit(rec) -> None:
    v = rec.verification
    print(f"\n[{rec.mode}] {rec.district} {rec.season} | model {rec.model} | verified {v['ok']} "
          f"({len(v['checked_numbers'])} numbers checked)")
    print(f"  {rec.headline}\n{rec.briefing}")
    if rec.plan:
        print("  plan:", " | ".join(rec.plan["steps"]))
    elif rec.planner_error:
        print("  planner: no plan,", rec.planner_error)
    if rec.first_attempt_errors:
        print("  first draft errors:", rec.first_attempt_errors)
    if rec.revision_errors:
        print("  revision errors:", rec.revision_errors)
    if rec.llm_error:
        print("  llm:", rec.llm_error)
    if v["errors"]:
        print("  VERIFY ERRORS:", v["errors"])


def _print_briefing(rec) -> None:
    v = rec.verification
    print(f"\n[{rec.mode}] {rec.district} {rec.date} | model {rec.model} | verified {v['ok']} "
          f"({len(v['checked_numbers'])} numbers checked) | tools: {[t['tool'] for t in rec.trace]}")
    print(f"  {rec.headline}\n  {rec.briefing}")
    plan = getattr(rec, "plan", None)
    if plan:
        print("  plan:", " | ".join(plan["steps"]))
        print("  questions:", " | ".join(plan.get("questions_to_answer", [])))
    elif getattr(rec, "planner_error", None):
        print("  planner: no plan,", rec.planner_error)
    if rec.first_attempt_errors:
        print("  first draft errors:", rec.first_attempt_errors)
    if rec.llm_error:
        print("  llm:", rec.llm_error)
    if v["errors"]:
        print("  VERIFY ERRORS:", v["errors"])


def _sample_district_days(n: int, seed: int = 42) -> list[tuple[str, str]]:
    """n random (date, district) pairs the Sentinel flags in the test seasons: the days the product would brief."""
    import random

    from psw.agents import sentinel
    from psw.agents.tools import DataStore

    store = DataStore()
    days = sorted(store.forecasts["issue_date"].unique())
    flagged = [(d, r["district"]) for d in days for r in sentinel.scan(d, store) if r["flags"]]
    print(f"{len(flagged)} flagged district-days in the test seasons; sampling {n} (seed {seed})")
    return sorted(random.Random(seed).sample(flagged, min(n, len(flagged))))


def cmd_brief(args) -> None:
    import json
    from collections import Counter

    from psw.agents import orchestrator
    from psw.agents.tools import DataStore
    from psw.llm import LLM, LLMError
    from psw.rag.index import KnowledgeBase

    store, kb = DataStore(), KnowledgeBase()
    llm = None
    if not args.no_llm:
        try:
            llm = LLM.from_env()
            print(f"LLM: {llm.model} at {llm.base_url} (min {llm.min_interval_s:.0f}s between calls)")
        except LLMError as e:
            print(f"No LLM ({e}); using the template fallback.")
    if args.no_llm:
        llm = _no_llm("--no-llm")

    if args.sample:
        pairs = _sample_district_days(args.sample)
        root = config.PROCESSED / "sample_run"
        recs = []
        for i, (d, dist) in enumerate(pairs, 1):
            print(f"--- {i}/{len(pairs)}: {dist} {d}", flush=True)
            saved = orchestrator.briefing_path(d, dist, root)
            rec = _load(saved) if saved.exists() and json.loads(saved.read_text())["mode"] != "template" else None
            if rec is not None:
                print("    (reusing the saved agent briefing)", flush=True)
            else:  # new, or the earlier try fell back to the template (for example a rate limit): try again
                rec = orchestrator.run_briefing(dist, d, llm=llm, store=store, kb=kb, root=root, use_planner=args.plan)
            print(f"    -> {rec.mode}, verified={rec.verification['ok']}" + (f", llm: {rec.llm_error}" if rec.llm_error else ""))
            recs.append(rec)
        modes = Counter(r.mode for r in recs)
        passed = sum(r.verification["ok"] for r in recs)
        report = {"n": len(recs), "verified_ok": passed, "modes": dict(modes),
                  "agent_or_revised": modes["agent"] + modes["agent_revised"], "model": getattr(llm, "model", None),
                  "items": [{"date": r.date, "district": r.district, "mode": r.mode, "ok": r.verification["ok"],
                             "first_attempt_errors": r.first_attempt_errors, "llm_error": r.llm_error} for r in recs]}
        (config.PROCESSED / "brief_sample_report.json").write_text(json.dumps(report, indent=1))
        print(f"\nSAMPLE: {passed}/{len(recs)} verified; modes {dict(modes)}; "
              f"{report['agent_or_revised']} written by the agent (not the template)")
        if llm is not None and hasattr(llm, "usage"):
            print("token usage:", llm.usage)
        return

    if not args.date:
        raise SystemExit("--date YYYY-MM-DD is required (or use --sample N)")
    if args.district:
        _print_briefing(orchestrator.run_briefing(args.district, args.date, llm=llm, store=store, kb=kb, use_planner=args.plan))
        return
    watch = orchestrator.run_daily_watch(args.date, top_n=args.top_n, llm=llm, store=store, kb=kb, use_planner=args.plan)
    print(f"Daily watch {args.date}: {watch['n_flagged']} districts flagged, briefing the top {len(watch['briefings'])}")
    for r in watch["watchlist"][:max(args.top_n, 6)]:
        print(f"  {r['district']:26s} {r['fire_count']:4d} fires (7d mean {r['fire_count_7d_mean']:6.1f})  "
              f"{'; '.join(f.split(':')[0] for f in r['flags']) or '-'}")
    for b in watch["briefings"]:
        _print_briefing(_load(b["file"]))
    print(f"\nSaved data/processed/watch/{args.date}.json and {len(watch['briefings'])} briefings.")


def _load(path):
    """Reload a saved briefing record (a Path, or a file name inside data/processed/briefings)."""
    import json
    from pathlib import Path
    from types import SimpleNamespace as NS
    path = Path(path) if Path(path).is_absolute() else config.PROCESSED / "briefings" / path
    return NS(**json.loads(path.read_text()))


def cmd_demo_cache(args) -> None:
    from psw.agents import demo
    from psw.agents.tools import DataStore
    from psw.llm import LLM, LLMError
    from psw.rag.index import KnowledgeBase

    try:
        llm = LLM.from_env()
        print(f"LLM: {llm.model}")
    except LLMError as e:
        llm = None
        print(f"No LLM ({e}); briefings that are not already saved will use the template.")
    out = demo.build_demo_cache(llm=llm, store=DataStore(), kb=KnowledgeBase(), top_n=args.top_n, force=args.force, use_planner=args.plan)
    bad = [x for v in out["briefed"].values() for x in v if not x.endswith(("agent", "agent_revised"))]
    print(f"Saved data/processed/demo.json. Not written by the agent: {bad or 'none'}")


COMMANDS = {
    "ingest": cmd_ingest,
    "process": cmd_process,
    "train": cmd_train,
    "kb": cmd_kb,
    "kb-eval": cmd_kb_eval,
    "brief": cmd_brief,
    "demo-cache": cmd_demo_cache,
    "audit-ingest": cmd_audit_ingest,
    "audit": cmd_audit,
}


def cmd_all(args) -> None:
    for name in ("ingest", "process", "train", "kb"):  # extended as later phases land
        COMMANDS[name](args)


COMMANDS["all"] = cmd_all


def main() -> None:
    p = argparse.ArgumentParser(prog="psw.pipeline")
    p.add_argument("command", choices=sorted(COMMANDS))
    p.add_argument("--date")
    p.add_argument("--district")
    p.add_argument("--sample", type=int, help="brief N sampled flagged district-days in the test seasons and report the pass rate")
    p.add_argument("--top-n", type=int, default=3, help="daily watch: how many flagged districts to brief")
    p.add_argument("--season", type=int, help="audit: the season to audit (with --district, or Punjab alone)")
    p.add_argument("--tables-only", action="store_true", help="audit: write the tables and summary, no briefs")
    p.add_argument("--plan", action="store_true", help="brief / demo-cache / audit: run the Planner first (one extra LLM call each)")
    p.add_argument("--force", action="store_true", help="demo-cache / audit: regenerate what is already saved")
    p.add_argument("--no-llm", action="store_true", help="brief / audit: skip the LLM and use the template fallback")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    COMMANDS[args.command](args)


if __name__ == "__main__":
    main()
