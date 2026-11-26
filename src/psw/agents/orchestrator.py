"""Orchestrator: Sentinel -> Analyst -> Verifier -> (one revision) -> template fallback. Saves JSON for the app.

    run_briefing   one district-day. mode is "agent", "agent_revised" or "template".
    run_daily_watch   Sentinel scan, then a briefing for the top flagged districts.

Whatever the LLM does (errors, bad JSON, invented numbers), the result is a briefing that passed the Verifier.
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from psw import config
from psw.agents import analyst, planner, sentinel, template, tools
from psw.agents.runtime import AgentResult, run_agent
from psw.agents.tools import DataStore, nearest_city
from psw.agents.verifier import Evidence, verify
from psw.llm import LLM, LLMError

log = logging.getLogger(__name__)


@dataclass
class BriefingRecord:
    district: str
    date: str
    mode: str  # agent | agent_revised | template
    headline: str
    briefing: str
    citations: list[str]
    sources: list[dict]  # [{chunk_id, title, doc_type}]
    verification: dict  # {ok, errors, checked_numbers}
    trace: list[dict]
    facts: dict  # tool name -> result
    model: str
    generated_at: str
    nearest_city: str = ""
    first_attempt_errors: list[str] = field(default_factory=list)  # why the first draft failed, if it did
    llm_error: str | None = None
    label: str | None = None  # set for template mode
    plan: dict | None = None  # the Planner's {steps, questions_to_answer}, when the planner ran and produced a usable plan
    planner_error: str | None = None  # why there is no plan, when the planner was asked to run

    def to_dict(self) -> dict:
        return asdict(self)


def slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")


def briefing_path(date: str, district: str, root: Path | None = None) -> Path:
    return Path(root or config.PROCESSED) / "briefings" / f"{date}_{slug(district)}.json"


def watch_path(date: str, root: Path | None = None) -> Path:
    return Path(root or config.PROCESSED) / "watch" / f"{date}.json"


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=str))


def _evidence(results: list[dict], date: str) -> Evidence:
    return Evidence.from_tool_results(results, dates=[date])


def _facts(results: list[dict]) -> dict:
    """tool name -> result. The last successful call wins. Retrieved chunks from every search are merged."""
    facts: dict = {}
    chunks: dict[str, dict] = {}
    for r in results:
        name, res = r["tool"], r["result"]
        if name == "search_knowledge" and isinstance(res, dict):
            for c in res.get("results", []):
                chunks.setdefault(c["chunk_id"], c)
        if name not in facts or not (isinstance(res, dict) and "error" in res):
            facts[name] = res
    if chunks:
        facts["search_knowledge"] = {"results": list(chunks.values())}
    return facts


def _sources(citations: list[str], kb) -> list[dict]:
    c = kb.chunks.set_index("chunk_id")
    return [{"chunk_id": cid, "title": str(c.loc[cid, "title"]), "doc_type": str(c.loc[cid, "doc_type"])}
            for cid in citations if cid in c.index]


def _renumber(trace: list[dict]) -> list[dict]:
    return [{**t, "step": i} for i, t in enumerate(trace, 1)]


def _template_facts(store: DataStore, kb, district: str, date: str, city: str) -> tuple[dict, list[dict], list[dict]]:
    """Call the tools directly (no LLM). Returns (template facts, tool_results, trace)."""
    calls = [
        ("snapshot", "get_district_snapshot", {"district": district, "date": date}, lambda: tools.get_district_snapshot(store, district, date)),
        ("forecast", "get_city_forecast", {"city": city, "date": date}, lambda: tools.get_city_forecast(store, city, date)),
        ("wind", "get_wind_and_upwind", {"city": city, "date": date}, lambda: tools.get_wind_and_upwind(store, city, date)),
        ("reliability", "get_model_reliability", {"city": city}, lambda: tools.get_model_reliability(store, city)),
        ("limitations", "search_knowledge", {"doc": tools.LIMITATIONS_DOC}, lambda: tools.limitations_chunk(kb)),
    ]
    facts, results, trace = {}, [], []
    for key, name, args, fn in calls:
        t0 = time.perf_counter()
        res = fn()
        facts[key] = res
        results.append({"tool": name, "args": args, "result": res})
        s = json.dumps(res, ensure_ascii=False, default=str)
        trace.append({"agent": "template", "tool": name, "args": args, "result_summary": s if len(s) <= 180 else s[:179] + "…",
                      "ms": int((time.perf_counter() - t0) * 1000)})
    return facts, results, trace


def load_reusable(district: str, date: str, root: Path | None = None) -> BriefingRecord | None:
    """A saved briefing written by the agent that passed the Verifier. Template or failed ones are not reused."""
    path = briefing_path(date, district, root)
    try:
        d = json.loads(path.read_text())
        ok = d["mode"] in ("agent", "agent_revised") and d["verification"]["ok"]
        names = BriefingRecord.__dataclass_fields__
        return BriefingRecord(**{k: v for k, v in d.items() if k in names}) if ok else None
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return None


def run_briefing(district: str, date, llm=None, store: DataStore | None = None, kb=None,
                 root: Path | None = None, save: bool = True, reuse: bool = False, use_planner: bool = False) -> BriefingRecord:
    """One briefing. Never raises because of the LLM; raises ValueError only for an unknown district or date.

    reuse=True returns the saved briefing when it was written by the agent and verified (saves LLM quota on reruns).
    use_planner=True first asks the Planner for a plan (one extra LLM call) and passes it to the Analyst. A failed plan
    is recorded in `planner_error` and the Analyst runs without it.
    """
    date = str(date)
    if reuse and (saved := load_reusable(district, date, root)) is not None:
        return saved
    store = store or DataStore()
    if kb is None:
        from psw.rag.index import KnowledgeBase
        kb = KnowledgeBase()
    snap = tools.get_district_snapshot(store, district, date)
    if "error" in snap:
        raise ValueError(snap["error"])
    city = nearest_city(district, store)

    llm_error: str | None = None
    if llm is None:
        try:
            llm = LLM.from_env()
        except LLMError as e:
            llm_error = str(e)

    trace: list[dict] = []
    results: list[dict] = []
    first_errors: list[str] = []
    mode, draft, verdict = None, None, None

    plan = planner_error = None
    if llm is not None and use_planner:
        plan, planner_error, entry = planner.make_plan(district, date, city, planner.flag_kinds(store, district, date), llm)
        trace.append(entry)
        if planner_error:
            log.info("%s %s: no plan (%s), the Analyst runs without one", district, date, planner_error)

    if llm is not None:
        res: AgentResult = analyst.analyse(district, date, llm, store, kb, plan_text=planner.plan_text(plan) if plan else None)
        trace += res.trace
        results += res.tool_results
        if res.error:
            llm_error = res.error
        else:
            verdict = verify(res.output, _evidence(results, date))
            if verdict.ok:
                mode, draft = "agent", res.output
            else:
                first_errors = verdict.errors
                log.info("%s %s: draft failed verification (%d errors), asking for one revision", district, date, len(first_errors))
                spec = analyst.build_analyst(store, kb)
                rev = run_agent(spec, analyst.REVISE_PROMPT.format(errors="\n".join(f"- {e}" for e in first_errors)), llm,
                                injected_args=analyst.injected_args(district, date, city), history=res.messages)
                trace += rev.trace
                results += rev.tool_results
                if rev.error:
                    llm_error = f"revision failed: {rev.error}"
                else:
                    verdict = verify(rev.output, _evidence(results, date))
                    if verdict.ok:
                        mode, draft = "agent_revised", rev.output

    if mode is None:
        tfacts, tresults, ttrace = _template_facts(store, kb, district, date, city)
        draft = template.render(tfacts)
        verdict = verify(draft, _evidence(tresults, date))
        if not verdict.ok:  # a bug in the template or missing data: keep the record honest rather than hide it
            log.error("template briefing for %s %s failed verification: %s", district, date, verdict.errors)
        mode, results, trace = "template", tresults, trace + ttrace
        facts = _facts(tresults)
    else:
        facts = _facts(results)

    citations = [c for c in draft.get("citations", []) if isinstance(c, str)]
    record = BriefingRecord(
        district=district, date=date, mode=mode, headline=draft["headline"], briefing=draft["briefing"],
        citations=citations, sources=_sources(citations, kb),
        verification={"ok": verdict.ok, "errors": verdict.errors, "checked_numbers": verdict.checked_numbers},
        trace=_renumber(trace), facts=facts,
        model=getattr(llm, "model", "none") if mode != "template" else "template",
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        nearest_city=city, first_attempt_errors=first_errors, llm_error=llm_error,
        label=draft.get("label") if mode == "template" else None,
        plan=plan, planner_error=planner_error,
    )
    if save:
        _write_json(briefing_path(date, district, root), record.to_dict())
    return record


def run_daily_watch(date, top_n: int = 3, llm=None, store: DataStore | None = None, kb=None,
                    root: Path | None = None, reuse: bool = False, use_planner: bool = False) -> dict:
    """Sentinel scan, then a briefing for the top `top_n` flagged districts. Saves the watch file and returns it."""
    date = str(date)
    store = store or DataStore()
    rows = sentinel.scan(date, store)
    flagged = [r for r in rows if r["flags"]]
    if llm is None:
        try:
            llm = LLM.from_env()
        except LLMError as e:
            log.warning("no LLM (%s): briefings will use the template", e)
    briefings = []
    for r in flagged[:top_n]:
        rec = run_briefing(r["district"], date, llm=llm, store=store, kb=kb, root=root, reuse=reuse, use_planner=use_planner)
        briefings.append({"district": rec.district, "mode": rec.mode, "headline": rec.headline,
                          "verified": rec.verification["ok"], "file": briefing_path(date, rec.district, root).name})
    watch = {"date": date, "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "n_flagged": len(flagged), "top_n": top_n, "watchlist": rows, "briefings": briefings}
    _write_json(watch_path(date, root), watch)
    return watch
