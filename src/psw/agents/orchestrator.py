"""Orchestrator: Sentinel -> Analyst -> Verifier -> (one revision) -> template fallback. Saves JSON for the app.

    run_briefing   one district-day. mode is "agent", "agent_revised" or "template".
    run_daily_watch   Sentinel scan, then a briefing for the top flagged districts.
    run_audit   one district (or Punjab) and season: Auditor -> Verifier -> (one revision) -> template fallback.
    run_dispatch   one issue date: allocation (code) -> Dispatcher -> Verifier -> (one revision) -> template fallback.

With use_critic, the Critic reads an agent-written draft after the Verifier (on by default for audits and dispatches, off
for briefings).
A rewrite is used only if it passes the Verifier again; the mode then ends in "+critic".

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
from psw.agents import (analyst, audit_template, audit_tools, auditor, critic, dispatch_template, dispatch_tools,
                        dispatcher, planner, sentinel, template, tools)
from psw.agents.audit_tools import AuditStore
from psw.agents.dispatch_tools import ResponseStore, Settings
from psw.agents.runtime import AgentResult, run_agent
from psw.agents.tools import DataStore, nearest_city
from psw.agents.verifier import Evidence, lowercase_words, verify, verify_audit, verify_dispatch
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
    critic: dict | None = None  # {ok, issues, rewrite_used, error, ...} when the Critic ran

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


def _template_calls(calls: list[tuple]) -> tuple[dict, list[dict], list[dict]]:
    """Run (key, tool name, args, fn) calls directly (no LLM). Returns (template facts, tool_results, trace)."""
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


def _template_facts(store: DataStore, kb, district: str, date: str, city: str) -> tuple[dict, list[dict], list[dict]]:
    """Call the tools directly (no LLM). Returns (template facts, tool_results, trace)."""
    calls = [
        ("snapshot", "get_district_snapshot", {"district": district, "date": date}, lambda: tools.get_district_snapshot(store, district, date)),
        ("forecast", "get_city_forecast", {"city": city, "date": date}, lambda: tools.get_city_forecast(store, city, date)),
        ("wind", "get_wind_and_upwind", {"city": city, "date": date}, lambda: tools.get_wind_and_upwind(store, city, date)),
        ("reliability", "get_model_reliability", {"city": city}, lambda: tools.get_model_reliability(store, city)),
        ("limitations", "search_knowledge", {"doc": tools.LIMITATIONS_DOC}, lambda: tools.limitations_chunk(kb)),
    ]
    return _template_calls(calls)


@dataclass
class Attempt:
    """What came out of the LLM: the draft that passed the Verifier (mode is None if none did) and how it got there."""
    mode: str | None = None  # agent | agent_revised
    draft: dict | None = None
    verdict: object = None
    trace: list[dict] = field(default_factory=list)
    results: list[dict] = field(default_factory=list)
    first_errors: list[str] = field(default_factory=list)
    revision_errors: list[str] = field(default_factory=list)  # why the revision failed too, if it did
    llm_error: str | None = None


GROUPED_CITATION = re.compile(r"\[(K\d{3}(?:\s*[,;]\s*K\d{3})+)\]")


def _split_citations(draft):
    """'[K423, K425]' -> '[K423] [K425]'. Only the format changes; both ids are still checked."""
    if not isinstance(draft, dict) or not isinstance(draft.get("briefing"), str):
        return draft
    text = GROUPED_CITATION.sub(lambda m: " ".join(f"[{c}]" for c in re.findall(r"K\d{3}", m.group(1))), draft["briefing"])
    return {**draft, "briefing": text}


def _draft_and_verify(llm, first, revise, check, revise_prompt: str, what: str) -> Attempt:
    """Agent, Verifier, and at most one revision in the same conversation. Shared by briefings and audits.

    first() and revise(prompt, history) return an AgentResult; check(draft, tool_results) returns a VerifyResult.
    """
    out = Attempt()
    res: AgentResult = first()
    out.trace += res.trace
    out.results += res.tool_results
    if res.error:
        out.llm_error = res.error
        return out
    output = _split_citations(res.output)
    out.verdict = check(output, out.results)
    if out.verdict.ok:
        out.mode, out.draft = "agent", output
        return out
    out.first_errors = out.verdict.errors
    log.info("%s: draft failed verification (%d errors), asking for one revision", what, len(out.first_errors))
    rev = revise(revise_prompt.format(errors="\n".join(f"- {e}" for e in out.first_errors)), res.messages)
    out.trace += rev.trace
    out.results += rev.tool_results
    if rev.error:
        out.llm_error = f"revision failed: {rev.error}"
        return out
    output = _split_citations(rev.output)
    out.verdict = check(output, out.results)
    if out.verdict.ok:
        out.mode, out.draft = "agent_revised", output
    else:
        out.revision_errors = out.verdict.errors
    return out


def _criticise(llm, brief_type: str, mode: str, draft: dict, verdict, kb, trace: list[dict], check, word_range):
    """Run the Critic over a verified draft. Returns (mode, draft, verdict, critic info) and extends `trace`."""
    done = critic.review(draft, brief_type, critic.cited_chunks(draft, kb), llm, check, word_range)
    trace.append(done.entry)
    if done.info["error"]:
        log.info("%s: the Critic failed (%s), the verified draft stands", brief_type, done.info["error"])
    if done.info["rewrite_used"]:
        return f"{mode}+critic", done.draft, done.verdict, done.info
    return mode, draft, verdict, done.info


def load_reusable(district: str, date: str, root: Path | None = None) -> BriefingRecord | None:
    """A saved briefing written by the agent that passed the Verifier. Template or failed ones are not reused."""
    path = briefing_path(date, district, root)
    try:
        d = json.loads(path.read_text())
        ok = critic.base_mode(d["mode"]) in ("agent", "agent_revised") and d["verification"]["ok"]
        names = BriefingRecord.__dataclass_fields__
        return BriefingRecord(**{k: v for k, v in d.items() if k in names}) if ok else None
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return None


def run_briefing(district: str, date, llm=None, store: DataStore | None = None, kb=None,
                 root: Path | None = None, save: bool = True, reuse: bool = False, use_planner: bool = False,
                 use_critic: bool = False) -> BriefingRecord:
    """One briefing. Never raises because of the LLM; raises ValueError only for an unknown district or date.

    reuse=True returns the saved briefing when it was written by the agent and verified (saves LLM quota on reruns).
    use_planner=True first asks the Planner for a plan (one extra LLM call) and passes it to the Analyst. A failed plan
    is recorded in `planner_error` and the Analyst runs without it.
    use_critic=True has the Critic read an agent-written draft (one extra LLM call).
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
        att = _draft_and_verify(
            llm,
            first=lambda: analyst.analyse(district, date, llm, store, kb, plan_text=planner.plan_text(plan) if plan else None),
            revise=lambda prompt, history: run_agent(analyst.build_analyst(store, kb), prompt, llm,
                                                     injected_args=analyst.injected_args(district, date, city), history=history),
            check=lambda draft, found: verify(draft, _evidence(found, date)),
            revise_prompt=analyst.REVISE_PROMPT, what=f"{district} {date}")
        mode, draft, verdict, first_errors = att.mode, att.draft, att.verdict, att.first_errors
        trace += att.trace
        results += att.results
        llm_error = att.llm_error

    review_info = None
    if mode is not None and use_critic:
        mode, draft, verdict, review_info = _criticise(
            llm, "briefing", mode, draft, verdict, kb, trace,
            check=lambda d: verify(d, _evidence(results, date)), word_range=config.BRIEFING_WORDS)

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
        plan=plan, planner_error=planner_error, critic=review_info,
    )
    if save:
        _write_json(briefing_path(date, district, root), record.to_dict())
    return record


def run_daily_watch(date, top_n: int = 3, llm=None, store: DataStore | None = None, kb=None,
                    root: Path | None = None, reuse: bool = False, use_planner: bool = False, use_critic: bool = False) -> dict:
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
        rec = run_briefing(r["district"], date, llm=llm, store=store, kb=kb, root=root, reuse=reuse, use_planner=use_planner,
                           use_critic=use_critic)
        briefings.append({"district": rec.district, "mode": rec.mode, "headline": rec.headline,
                          "verified": rec.verification["ok"], "file": briefing_path(date, rec.district, root).name})
    watch = {"date": date, "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "n_flagged": len(flagged), "top_n": top_n, "watchlist": rows, "briefings": briefings}
    _write_json(watch_path(date, root), watch)
    return watch


# ---------- Part 2: audits ----------

@dataclass
class AuditRecord:
    district: str  # a district, or "Punjab"
    season: int
    mode: str  # agent | agent_revised | template
    headline: str
    briefing: str
    citations: list[str]
    sources: list[dict]
    verification: dict
    trace: list[dict]
    facts: dict
    model: str
    generated_at: str
    caveat: str = ""  # set by code, so every audit output carries it whatever the text says
    first_attempt_errors: list[str] = field(default_factory=list)
    revision_errors: list[str] = field(default_factory=list)
    llm_error: str | None = None
    label: str | None = None
    plan: dict | None = None
    planner_error: str | None = None
    critic: dict | None = None
    manual_edits: list[str] = field(default_factory=list)  # changes made by hand after the run, if any

    def to_dict(self) -> dict:
        return asdict(self)


def audit_path(season: int, district: str, root: Path | None = None) -> Path:
    return Path(root or config.PROCESSED) / "audits" / f"{int(season)}_{slug(district)}.json"


def load_reusable_audit(district: str, season: int, root: Path | None = None) -> AuditRecord | None:
    """A saved audit written by the agent that passed the Verifier. Template or failed ones are not reused."""
    try:
        d = json.loads(audit_path(season, district, root).read_text())
        ok = critic.base_mode(d["mode"]) in ("agent", "agent_revised") and d["verification"]["ok"]
        names = AuditRecord.__dataclass_fields__
        return AuditRecord(**{k: v for k, v in d.items() if k in names}) if ok else None
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return None


def _gap_result(results: list[dict]) -> dict | None:
    found = [r["result"] for r in results if r["tool"] == "get_gap_range" and "error" not in r["result"]]
    return found[-1] if found else None


AUDIT_TITLE = re.compile(r"burning audit, season (\d{4})")


def _other_audits(results: list[dict], season: int) -> dict[str, int]:
    """chunk id -> season, for retrieved audit documents about other seasons."""
    found = {}
    for r in results:
        if r["tool"] != "search_knowledge":
            continue
        for c in r["result"].get("results", []):
            m = AUDIT_TITLE.search(c.get("title", ""))
            if m and int(m.group(1)) != season:
                found[c["chunk_id"]] = int(m.group(1))
    return found


def _audit_check(season: int):
    def check(draft, found: list[dict]):
        evidence = Evidence.from_tool_results(found, dates=[audit_tools.cutoff(season)])
        return verify_audit(draft, evidence, gap=_gap_result(found), audit_seasons=_other_audits(found, season),
                            night_too_few=any(r["tool"] == "get_diurnal_profile" and "warning" in r["result"] for r in found))
    return check


def _audit_template_facts(store: AuditStore, kb, district: str, season: int) -> tuple[dict, list[dict], list[dict]]:
    place = {"district": district, "season": season}
    calls = [(key, name, place, lambda fn=audit_tools.DATA_TOOLS[name]: fn(store, district, season))
             for key, name in (("trend", "get_detection_trend"), ("diurnal", "get_diurnal_profile"),
                               ("burned", "get_burned_area_comparison"), ("official", "get_official_vs_satellite"),
                               ("gap", "get_gap_range"))]
    calls.append(("method", "search_knowledge", {"doc": audit_template.METHOD_DOC}, lambda: audit_template.method_chunks(kb)))
    return _template_calls(calls)


def run_audit(district: str, season: int, llm=None, store: AuditStore | None = None, kb=None, root: Path | None = None,
              save: bool = True, reuse: bool = False, use_planner: bool = False, use_critic: bool = True) -> AuditRecord:
    """One audit brief for a district or "Punjab". Never raises because of the LLM; raises ValueError for an unknown
    district or season.

    use_planner=True first asks the Planner for a plan from the names of the available signals (one extra LLM call).
    use_critic (on by default) has the Critic read an agent-written draft (one extra LLM call).
    """
    season = int(season)
    if reuse and (saved := load_reusable_audit(district, season, root)) is not None:
        return saved
    store = store or AuditStore()
    store.check(district, season)
    if kb is None:
        from psw.rag.index import KnowledgeBase
        kb = KnowledgeBase()

    llm_error: str | None = None
    if llm is None:
        try:
            llm = LLM.from_env()
        except LLMError as e:
            llm_error = str(e)

    trace: list[dict] = []
    att = Attempt(llm_error=llm_error)
    plan = planner_error = None
    if llm is not None:
        if use_planner:
            signals = audit_tools.available_signals(store, district, season)
            plan, planner_error, entry = planner.make_audit_plan(district, season, signals, llm, audit_tools.TOOL_SCHEMAS)
            trace.append(entry)
            if planner_error:
                log.info("%s %s: no plan (%s), the Auditor runs without one", district, season, planner_error)
        spec = auditor.build_auditor(store, kb)
        injected = auditor.injected_args(district, season)
        message = auditor.user_message(district, season, planner.plan_text(plan) if plan else None)
        att = _draft_and_verify(
            llm,
            first=lambda: run_agent(spec, message, llm, injected_args=injected),
            revise=lambda prompt, history: run_agent(spec, prompt, llm, injected_args=injected, history=history),
            check=_audit_check(season), revise_prompt=auditor.REVISE_PROMPT, what=f"audit {district} {season}")
        trace += att.trace

    mode, draft, verdict, results = att.mode, att.draft, att.verdict, att.results
    review_info = None
    if mode is not None and use_critic:
        mode, draft, verdict, review_info = _criticise(
            llm, "audit", mode, draft, verdict, kb, trace,
            check=lambda d: _audit_check(season)(d, att.results), word_range=config.AUDIT_WORDS)

    if mode is None:
        tfacts, results, ttrace = _audit_template_facts(store, kb, district, season)
        draft = audit_template.render(tfacts, district, season)
        verdict = _audit_check(season)(draft, results)
        if not verdict.ok:
            log.error("template audit for %s %s failed verification: %s", district, season, verdict.errors)
        mode, trace = "template", trace + ttrace

    citations = [c for c in draft.get("citations", []) if isinstance(c, str)]
    record = AuditRecord(
        district=district, season=season, mode=mode, headline=draft["headline"], briefing=draft["briefing"],
        citations=citations, sources=_sources(citations, kb),
        verification={"ok": verdict.ok, "errors": verdict.errors, "checked_numbers": verdict.checked_numbers},
        trace=_renumber(trace), facts=_facts(results),
        model=getattr(llm, "model", "none") if mode != "template" else "template",
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        caveat=audit_tools.gap.GAP_CAVEAT, first_attempt_errors=att.first_errors, revision_errors=att.revision_errors,
        llm_error=att.llm_error,
        label=draft.get("label") if mode == "template" else None, plan=plan, planner_error=planner_error,
        critic=review_info,
    )
    if save:
        _write_json(audit_path(season, district, root), record.to_dict())
    return record


# ---------- Part 2: dispatch ----------

DISPATCH_CAVEAT = ("Decision support for residue-management help. Not for enforcement. No farms or farmers identified. "
                   "The split was computed by a fixed rule, and what one unit achieves is an assumption.")


@dataclass
class DispatchRecord:
    date: str
    units: int
    max_share: float
    use_likely: bool
    district: str | None  # the district that has its own paragraph, if one was asked for
    variant: str  # which forecast variant was allocated on
    mode: str  # agent | agent_revised | template
    headline: str
    briefing: str
    district_paragraph: str | None
    citations: list[str]
    sources: list[dict]
    verification: dict
    trace: list[dict]
    facts: dict
    allocation: list[dict]  # computed by code, whatever the text says
    detected_only: list[dict]  # the same fleet without the likely-burning adjustment
    moved: list[dict]  # districts the adjustment moves, largest first
    units_not_placed: int
    model: str
    generated_at: str
    caveat: str = DISPATCH_CAVEAT
    first_attempt_errors: list[str] = field(default_factory=list)
    revision_errors: list[str] = field(default_factory=list)
    llm_error: str | None = None
    label: str | None = None
    critic: dict | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def dispatch_path(date: str, root: Path | None = None) -> Path:
    return Path(root or config.PROCESSED) / "dispatch" / f"{date}.json"


def load_reusable_dispatch(date: str, settings: Settings, district: str | None, root: Path | None = None
                           ) -> DispatchRecord | None:
    """A saved agent-written, verified dispatch for exactly these inputs."""
    try:
        d = json.loads(dispatch_path(date, root).read_text())
        same = (d["units"], d["max_share"], d["use_likely"], d["district"]) == (
            settings.units, settings.max_share, settings.use_likely, district)
        ok = critic.base_mode(d["mode"]) in ("agent", "agent_revised") and d["verification"]["ok"]
        names = DispatchRecord.__dataclass_fields__
        return DispatchRecord(**{k: v for k, v in d.items() if k in names}) if same and ok else None
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return None


def _dispatch_check(allocation: dispatch_tools.Allocation, day: str, district: str | None, kb):
    # words our own texts write in lower case; capitalised in a brief, they are ordinary words and not names
    own = lowercase_words([dispatcher.SYSTEM_PROMPT, *dispatch_template.PADDING, *kb.chunks["text"]])

    def check(draft, found: list[dict]):
        evidence = Evidence.from_tool_results(found, dates=[day])
        words = own | lowercase_words(json.dumps(r["result"], ensure_ascii=False, default=str) for r in found)
        return verify_dispatch(draft, evidence, allocation.plan(words), district=district)
    return check


def _dispatch_template_facts(store: ResponseStore, kb, day: str, settings: Settings,
                             district: str | None) -> tuple[dict, list[dict], list[dict]]:
    t = dispatch_tools
    s = (settings.units, settings.max_share, settings.use_likely)
    allocation = t.get_allocation(store, day, *s, district=district)
    leader = allocation["districts"][0]["district"] if allocation["districts"] else \
        t.get_risk_forecast(store, day, settings.use_likely)["top_districts"][0]["district"]
    season = t.as_date(day).year - 1
    moves = allocation.get("moved_by_the_likely_adjustment") or []
    about = moves[0]["district"] if moves else t.STATE
    calls = [
        ("risk", "get_risk_forecast", {"date": day, "district": district},
         lambda: t.get_risk_forecast(store, day, settings.use_likely, district)),
        ("allocation", "get_allocation", {"date": day, "units": settings.units}, lambda: allocation),
        ("gap", "get_gap_range", {"district": about, "season": season}, lambda: t.get_gap_range(store, about, season)),
        ("rain", "get_rain_outlook", {"district": leader, "date": day}, lambda: t.get_rain_outlook(store, leader, day)),
        ("hotspots", "get_hotspot_cells", {"district": leader, "date": day}, lambda: t.get_hotspot_cells(store, leader, day)),
        ("method", "search_knowledge", {"doc": audit_template.METHOD_DOC}, lambda: audit_template.method_chunks(kb)),
    ]
    if district:
        calls += [
            ("district_rain", "get_rain_outlook", {"district": district, "date": day},
             lambda: t.get_rain_outlook(store, district, day)),
            ("district_hotspots", "get_hotspot_cells", {"district": district, "date": day},
             lambda: t.get_hotspot_cells(store, district, day)),
        ]
    return _template_calls(calls)


def run_dispatch(date, units: int = 500, max_share: float = 0.25, use_likely: bool = True, district: str | None = None,
                 llm=None, store: ResponseStore | None = None, kb=None, root: Path | None = None, save: bool = True,
                 reuse: bool = False, use_critic: bool = True) -> DispatchRecord:
    """The 3-day dispatch issued on `date`. The allocation is computed by code; the Dispatcher only explains it.

    Never raises because of the LLM; raises ValueError for a date without a forecast, an unknown district or bad
    settings. `district` asks for an extra paragraph about it.
    """
    date, settings = str(date), Settings(int(units), float(max_share), bool(use_likely))
    if reuse and (saved := load_reusable_dispatch(date, settings, district, root)) is not None:
        return saved
    store = store or ResponseStore()
    store.check(date, district)
    allocation = dispatch_tools.plan_allocation(store, date, settings)
    if kb is None:
        from psw.rag.index import KnowledgeBase
        kb = KnowledgeBase()

    llm_error: str | None = None
    if llm is None:
        try:
            llm = LLM.from_env()
        except LLMError as e:
            llm_error = str(e)

    check = _dispatch_check(allocation, date, district, kb)
    att = Attempt(llm_error=llm_error)
    if llm is not None:
        spec = dispatcher.build_dispatcher(store, kb)
        injected = dispatcher.injected_args(date, settings)
        message = dispatcher.user_message(date, settings, district)
        att = _draft_and_verify(
            llm,
            first=lambda: run_agent(spec, message, llm, injected_args=injected),
            revise=lambda prompt, history: run_agent(spec, prompt, llm, injected_args=injected, history=history),
            check=check, revise_prompt=dispatcher.REVISE_PROMPT, what=f"dispatch {date}")

    mode, draft, verdict, results, trace = att.mode, att.draft, att.verdict, att.results, list(att.trace)
    review_info = None
    if mode is not None and use_critic:
        mode, draft, verdict, review_info = _criticise(
            llm, "dispatch", mode, draft, verdict, kb, trace, check=lambda d: check(d, att.results),
            word_range=config.DISPATCH_WORDS)

    if mode is None:
        tfacts, results, ttrace = _dispatch_template_facts(store, kb, date, settings, district)
        draft = dispatch_template.render(tfacts)
        verdict = check(draft, results)
        if not verdict.ok:
            log.error("template dispatch for %s failed verification: %s", date, verdict.errors)
        mode, trace = "template", trace + ttrace

    citations = [c for c in draft.get("citations", []) if isinstance(c, str)]
    record = DispatchRecord(
        date=date, units=settings.units, max_share=settings.max_share, use_likely=settings.use_likely, district=district,
        variant=store.variant, mode=mode, headline=draft["headline"], briefing=draft["briefing"],
        district_paragraph=draft.get("district_paragraph"), citations=citations, sources=_sources(citations, kb),
        verification={"ok": verdict.ok, "errors": verdict.errors, "checked_numbers": verdict.checked_numbers},
        trace=_renumber(trace), facts=_facts(results), allocation=allocation.table.to_dict("records"),
        detected_only=allocation.detected.to_dict("records"), moved=allocation.moves(top=len(allocation.table)),
        units_not_placed=settings.units - allocation.placed,
        model=getattr(llm, "model", "none") if mode != "template" else "template",
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        first_attempt_errors=att.first_errors, revision_errors=att.revision_errors, llm_error=att.llm_error,
        label=draft.get("label") if mode == "template" else None, critic=review_info,
    )
    if save:
        _write_json(dispatch_path(date, root), record.to_dict())
    return record
