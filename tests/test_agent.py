"""Analyst + Verifier + orchestrator with a scripted FakeLLM and the committed processed data. No network."""
import json

import pytest

from psw import config
from psw.agents import analyst, orchestrator, tools
from psw.agents.tools import DataStore
from psw.llm import LLMError
from tests.fakes import FakeKB, FakeLLM, calls, final

DAY = "2025-10-31"
DISTRICT = "Sangrur"
FILLER = ("This sentence is plain context with no figures and it helps the briefing reach a sensible length for readers. " * 5).strip()


@pytest.fixture(scope="module")
def store():
    return DataStore()


@pytest.fixture
def kb():
    return FakeKB()


def four_calls():
    """A model that asks for all four required tools in one turn, with arguments the orchestrator will override."""
    return calls(("get_district_snapshot", {"district": "Wrong", "date": "2030-01-01"}),
                 ("get_city_forecast", {"city": "Ludhiana", "date": "2030-01-01"}),
                 ("get_wind_and_upwind", {"city": "Ludhiana", "date": "2030-01-01"}),
                 ("search_knowledge", {"query": "Sangrur 2024 season", "before": "2099-01-01", "district": "Nowhere"}))


def good_draft(store, extra=""):
    snap = tools.get_district_snapshot(store, DISTRICT, DAY)
    fc = tools.get_city_forecast(store, snap["nearest_city"], DAY)
    text = (f"{DISTRICT} recorded {snap['fire_count']} fire detections on {DAY}, with {snap['frp_sum_mw']} MW of fire power. "
            f"The median PM2.5 forecast is {fc['p50']} with an 80% interval of {fc['p10']} to {fc['p90']}. {FILLER} [K001]{extra}")
    return {"headline": f"{DISTRICT} fire activity and PM2.5 outlook", "briefing": text, "citations": ["K001"]}


def bad_draft(store):
    d = good_draft(store)
    d["briefing"] += " Detections rose by 4321 percent."
    return d


def run(store, kb, script, tmp_path, **kw):
    llm = FakeLLM(script)
    return orchestrator.run_briefing(DISTRICT, DAY, llm=llm, store=store, kb=kb, root=tmp_path, **kw), llm


def test_clean_run_is_mode_agent_with_trace_sources_and_saved_file(store, kb, tmp_path):
    rec, llm = run(store, kb, [four_calls(), final(good_draft(store))], tmp_path)
    assert rec.mode == "agent" and rec.verification["ok"] and rec.first_attempt_errors == [] and rec.llm_error is None
    assert [t["tool"] for t in rec.trace] == ["get_district_snapshot", "get_city_forecast", "get_wind_and_upwind", "search_knowledge"]
    assert [t["step"] for t in rec.trace] == [1, 2, 3, 4]
    assert rec.sources == [{"chunk_id": "K001", "title": "How FIRMS detects fires", "doc_type": "curated"}]
    assert rec.model == "fake-llm" and rec.nearest_city in config.CITIES
    assert set(rec.facts) == {"get_district_snapshot", "get_city_forecast", "get_wind_and_upwind", "search_knowledge"}
    saved = json.loads((tmp_path / "briefings" / f"{DAY}_{DISTRICT}.json").read_text())
    assert saved["headline"] == rec.headline and saved["verification"]["ok"] and len(saved["trace"]) == 4


def test_the_model_cannot_change_the_date_district_or_leakage_filters(store, kb, tmp_path):
    rec, _ = run(store, kb, [four_calls(), final(good_draft(store))], tmp_path)
    assert kb.searches == [{"query": "Sangrur 2024 season", "k": 4, "before": DAY, "district": DISTRICT}]
    snap_args = rec.trace[0]["args"]
    assert snap_args == {"district": DISTRICT, "date": DAY}
    assert rec.trace[1]["args"]["date"] == DAY and rec.trace[1]["args"]["city"] == rec.nearest_city


def test_skipped_required_tools_get_the_nudge(store, kb, tmp_path):
    script = [final(good_draft(store)), four_calls(), final(good_draft(store))]
    rec, llm = run(store, kb, script, tmp_path)
    assert rec.mode == "agent"
    nudge = [m for m in llm.calls[1]["messages"] if m["role"] == "user"][-1]["content"]
    assert "get_district_snapshot" in nudge and "search_knowledge" in nudge


def test_invented_number_goes_to_revision_and_can_be_fixed(store, kb, tmp_path):
    rec, llm = run(store, kb, [four_calls(), final(bad_draft(store)), final(good_draft(store))], tmp_path)
    assert rec.mode == "agent_revised" and rec.verification["ok"]
    assert any("4321" in e for e in rec.first_attempt_errors)
    fix = llm.calls[2]["messages"][-1]["content"]
    assert fix.startswith("Your draft failed verification") and "4321" in fix
    assert len(llm.calls) == 3  # one revision, no more


def test_still_failing_after_the_revision_falls_back_to_the_template(store, kb, tmp_path):
    rec, llm = run(store, kb, [four_calls(), final(bad_draft(store)), final(bad_draft(store))], tmp_path)
    assert rec.mode == "template" and rec.verification["ok"] and rec.label == "template (LLM unavailable or failed verification)"
    assert rec.first_attempt_errors and rec.model == "template"
    assert rec.citations == ["K000"] and rec.sources[0]["title"] == "Limitations of this project"
    assert "4321" not in rec.briefing
    assert any(t["agent"] == "template" for t in rec.trace) and any(t["agent"] == "analyst" for t in rec.trace)


def test_llm_failure_falls_back_to_the_template(store, kb, tmp_path):
    rec, _ = run(store, kb, [LLMError("503 high demand")], tmp_path)
    assert rec.mode == "template" and rec.verification["ok"] and "503" in rec.llm_error


def test_invalid_json_twice_falls_back_to_the_template(store, kb, tmp_path):
    rec, _ = run(store, kb, [four_calls(), final("not json"), final("still not json")], tmp_path)
    assert rec.mode == "template" and rec.verification["ok"] and "JSON" in rec.llm_error


def test_no_llm_configured_uses_the_template(store, kb, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "LLM_API_KEY", "")
    rec = orchestrator.run_briefing(DISTRICT, DAY, llm=None, store=store, kb=kb, root=tmp_path)
    assert rec.mode == "template" and rec.verification["ok"] and "must be set" in rec.llm_error


def test_unknown_district_or_date_raises(store, kb, tmp_path):
    with pytest.raises(ValueError):
        orchestrator.run_briefing("Atlantis", DAY, llm=FakeLLM([]), store=store, kb=kb, root=tmp_path)
    with pytest.raises(ValueError):
        orchestrator.run_briefing(DISTRICT, "2030-01-01", llm=FakeLLM([]), store=store, kb=kb, root=tmp_path)


def test_analyst_prompt_and_wiring(store):
    spec = analyst.build_analyst(store, FakeKB())
    assert spec.name == "analyst" and len(spec.tools) == 5 and set(spec.required_tools) <= {t.name for t in spec.tools}
    assert "ONLY numbers that appear in tool results" in analyst.SYSTEM_PROMPT and "120 to 160 words" in analyst.SYSTEM_PROMPT
    inj = analyst.injected_args("Sangrur", DAY, "Bathinda")
    assert inj["search_knowledge"] == {"before": DAY, "district": "Sangrur"}


def test_slug_handles_brackets_and_spaces():
    assert orchestrator.slug("SAS Nagar (Mohali)") == "SAS_Nagar_Mohali"
    assert orchestrator.briefing_path("2025-10-31", "SBS Nagar (Nawanshahr)").name == "2025-10-31_SBS_Nagar_Nawanshahr.json"


def test_daily_watch_briefs_the_top_flagged_districts_and_saves_both_files(store, kb, tmp_path):
    n = 2
    watch = orchestrator.run_daily_watch(DAY, top_n=n, llm=FakeLLM([LLMError("down")] * n), store=store, kb=kb, root=tmp_path)
    assert len(watch["watchlist"]) == 22 and watch["n_flagged"] >= n and len(watch["briefings"]) == n
    assert all(b["mode"] == "template" and b["verified"] for b in watch["briefings"])
    flagged = [r["district"] for r in watch["watchlist"] if r["flags"]]
    assert [b["district"] for b in watch["briefings"]] == flagged[:n]
    saved = json.loads((tmp_path / "watch" / f"{DAY}.json").read_text())
    assert saved["briefings"] == watch["briefings"]
    for b in watch["briefings"]:
        assert (tmp_path / "briefings" / b["file"]).exists()


def test_watch_with_no_flagged_districts_briefs_nobody(store, kb, tmp_path, monkeypatch):
    monkeypatch.setattr(orchestrator.sentinel, "scan", lambda d, s: [{"district": "Sangrur", "flags": [], "priority_score": 3}])
    watch = orchestrator.run_daily_watch(DAY, llm=FakeLLM([]), store=store, kb=kb, root=tmp_path)
    assert watch["briefings"] == [] and watch["n_flagged"] == 0


# ---------- Phase 7: demo cache ----------

def test_reuse_keeps_a_verified_agent_briefing_and_redoes_a_template(store, kb, tmp_path):
    run(store, kb, [four_calls(), final(good_draft(store))], tmp_path)  # saves an agent briefing
    again = orchestrator.run_briefing(DISTRICT, DAY, llm=FakeLLM([]), store=store, kb=kb, root=tmp_path, reuse=True)
    assert again.mode == "agent" and again.model == "fake-llm"  # the empty script proves no LLM call was made
    # a template briefing is not reusable: it is regenerated
    path = orchestrator.briefing_path(DAY, DISTRICT, tmp_path)
    saved = json.loads(path.read_text())
    path.write_text(json.dumps({**saved, "mode": "template"}))
    redone = orchestrator.run_briefing(DISTRICT, DAY, llm=FakeLLM([four_calls(), final(good_draft(store))]), store=store, kb=kb,
                                       root=tmp_path, reuse=True)
    assert redone.mode == "agent"
    path.write_text("{broken")  # a corrupt file is ignored, not fatal
    assert orchestrator.load_reusable(DISTRICT, DAY, tmp_path) is None


def test_demo_dates_follow_the_rules(store):
    from psw.agents import demo
    d = demo.pick_demo_dates(store)
    assert d["peak"] == "2025-10-31"
    dd = store.district_daily
    totals = dd[dd["season"] == 2025].groupby("date")["fire_count"].sum()
    assert totals[d["peak"]] == totals.max()
    assert totals[d["moderate"]] < totals[d["peak"]]
    for day in d.values():  # each demo date has something to brief
        assert any(r["flags"] for r in orchestrator.sentinel.scan(day, store))


def test_build_demo_cache_writes_everything_with_the_template(store, kb, tmp_path):
    from psw.agents import demo
    out = demo.build_demo_cache(llm=FakeLLM([LLMError("down")] * 20), store=store, kb=kb, root=tmp_path, top_n=1, log=lambda *_: None)
    assert out["default_district"] == "Ludhiana" and out["season"] == 2025
    assert demo.load_demo(tmp_path)["peak"] == out["peak"]
    peak = out["peak"]
    assert (tmp_path / "briefings" / f"{peak}_Ludhiana.json").exists() and (tmp_path / "briefings" / f"{peak}_Sangrur.json").exists()
    for day in {out["peak"], out["moderate"], out["early"]}:
        assert (tmp_path / "watch" / f"{day}.json").exists()
    assert demo.load_demo(tmp_path / "nowhere") is None


# ---------- stretch: Planner ----------

from psw.agents import planner  # noqa: E402

PLAN = {"steps": ["Get the district snapshot for the date.", "Read the forecast and the wind for the nearest city.",
                  "Search the 2024 season summary for this district.", "Mention one limitation of satellite detection."],
        "questions_to_answer": ["How active were fires compared with recent days?", "What does the forecast band say?"]}


def run_with_planner(store, kb, script, tmp_path, **kw):
    llm = FakeLLM(script)
    return orchestrator.run_briefing(DISTRICT, DAY, llm=llm, store=store, kb=kb, root=tmp_path, use_planner=True, **kw), llm


def test_clean_plan_keeps_years_and_dates_and_drops_numbers_and_field_names():
    raw = {"steps": ["Search the 2024 season summary.", "Check fires on 2025-10-31.", "Quote 49 detections.",
                     "Use rank_in_punjab here.", "", 7, "Mention p90 and the 80 percent band"],
           "questions_to_answer": ["Is the forecast above 60?", "What is the wind direction?"]}
    plan = planner.clean_plan(raw)
    assert plan["steps"] == ["Search the 2024 season summary.", "Check fires on 2025-10-31.", "Mention p90 and the 80 percent band"]
    assert plan["questions_to_answer"] == ["What is the wind direction?"]  # "above 60" is dropped, constants like 80 and 7 are allowed
    assert planner.clean_plan({"steps": ["Look at the 7-day trend."]})["steps"] == ["Look at the 7-day trend."]


def test_clean_plan_limits_and_rejects_unusable_plans():
    many = {"steps": [f"Step {chr(97 + i)}" for i in range(9)], "questions_to_answer": [f"Why {chr(97 + i)}?" for i in range(9)]}
    plan = planner.clean_plan(many)
    assert len(plan["steps"]) == planner.MAX_STEPS and len(plan["questions_to_answer"]) == planner.MAX_QUESTIONS
    assert planner.clean_plan({"steps": "do things"}) is None and planner.clean_plan({"steps": []}) is None
    assert planner.clean_plan(["x"]) is None and planner.clean_plan({"steps": ["Quote 12 fires"]}) is None
    assert planner.clean_plan({"steps": ["ok" + "x" * 500]})["steps"][0].__len__() == planner.MAX_CHARS


def test_planner_spec_has_no_tools_and_its_prompt_lists_the_analyst_tools():
    spec = planner.build_planner()
    assert spec.name == "planner" and spec.tools == [] and spec.json_output
    for name in analyst.REQUIRED_TOOLS:
        assert name in spec.system_prompt
    assert not any(ch.isdigit() for ch in planner.user_message("Sangrur", "x", "Ludhiana", ["spike"]).replace("x", ""))


def test_flag_kinds_come_from_the_sentinel(store):
    kinds = planner.flag_kinds(store, "Tarn Taran", DAY)
    assert kinds and set(kinds) <= {"spike", "high_count", "smog_risk"}
    assert planner.flag_kinds(store, "Nowhere", DAY) == []


def test_the_plan_reaches_the_analyst_and_the_record(store, kb, tmp_path):
    rec, llm = run_with_planner(store, kb, [final(PLAN), four_calls(), final(good_draft(store))], tmp_path)
    assert rec.mode == "agent" and rec.verification["ok"] and rec.plan == PLAN and rec.planner_error is None
    assert not llm.calls[0]["tools"] and "plan a short district briefing" in llm.calls[0]["messages"][0]["content"]
    assert llm.calls[0]["messages"][-1]["content"].count("Sentinel flags") == 1
    analyst_msg = llm.calls[1]["messages"][-1]["content"]
    assert "Write the briefing for Sangrur" in analyst_msg and "1. Get the district snapshot" in analyst_msg
    assert "How active were fires" in analyst_msg and "not evidence" in analyst_msg
    assert [t["agent"] for t in rec.trace][:2] == ["planner", "analyst"] and rec.trace[0]["tool"] == "plan"
    assert [t["step"] for t in rec.trace] == list(range(1, len(rec.trace) + 1))
    saved = json.loads((tmp_path / "briefings" / f"{DAY}_{DISTRICT}.json").read_text())
    assert saved["plan"]["steps"][0] == PLAN["steps"][0]


def test_a_planner_failure_never_blocks_the_briefing(store, kb, tmp_path):
    rec, _ = run_with_planner(store, kb, [LLMError("rate limited"), four_calls(), final(good_draft(store))], tmp_path)
    assert rec.mode == "agent" and rec.plan is None and "rate limited" in rec.planner_error
    rec, llm = run_with_planner(store, kb, [final({"steps": ["Quote 12 fires"]}), four_calls(), final(good_draft(store))], tmp_path)
    assert rec.mode == "agent" and rec.plan is None and "all 1 steps were dropped" in rec.planner_error
    assert "planner wrote this plan" not in llm.calls[1]["messages"][-1]["content"]


def test_reject_reasons_are_specific():
    assert "JSON object" in planner.reject_reason(["x"]) and "no list of steps" in planner.reject_reason({"steps": "go"})
    assert "all 2 steps were dropped" in planner.reject_reason({"steps": ["Quote 12 fires", "Use rank_in_punjab"]})


def test_the_planner_is_off_by_default_and_the_verifier_still_runs_last(store, kb, tmp_path):
    rec, llm = run(store, kb, [four_calls(), final(good_draft(store))], tmp_path)
    assert rec.plan is None and rec.planner_error is None and len(llm.calls) == 2
    # a plan does not excuse an invented number: the draft fails, is revised once, and the revision is checked again
    rec, _ = run_with_planner(store, kb, [final(PLAN), four_calls(), final(bad_draft(store)), final(good_draft(store))], tmp_path)
    assert rec.mode == "agent_revised" and rec.first_attempt_errors and rec.plan == PLAN


def test_a_plan_survives_a_template_fallback(store, kb, tmp_path):
    rec, _ = run_with_planner(store, kb, [final(PLAN), LLMError("down")], tmp_path)
    assert rec.mode == "template" and rec.verification["ok"] and rec.plan == PLAN
    assert rec.trace[0]["agent"] == "planner" and rec.trace[-1]["agent"] == "template"


def test_daily_watch_passes_use_planner_through(store, kb, tmp_path):
    watch = orchestrator.run_daily_watch(DAY, top_n=1, llm=FakeLLM([final(PLAN), LLMError("down")]), store=store, kb=kb,
                                         root=tmp_path, use_planner=True)
    rec = json.loads((tmp_path / "briefings" / watch["briefings"][0]["file"]).read_text())
    assert rec["plan"] == PLAN and rec["mode"] == "template"
