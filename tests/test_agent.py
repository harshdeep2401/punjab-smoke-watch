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
