"""The Critic with a scripted FakeLLM, on the synthetic audit tables and the committed briefing data. No network."""
import json

import pytest

from psw.agents import critic, orchestrator
from psw.agents.verifier import VerifyResult
from psw.llm import LLMError
from tests.fakes import FakeLLM, final
from tests.test_auditor import DISTRICT, SEASON, KB, good_draft, required_calls, store  # noqa: F401 (store is a fixture)

REMOVABLE = "The signals are read together and each one has its own weakness that a reader should keep in mind. "


@pytest.fixture
def kb():
    return KB()


def issue(kind="overclaim", sentence="s", note="n"):
    return {"type": kind, "sentence": sentence, "note": note}


def review(issues=(), text=None, ok=None):
    return final({"ok": not issues if ok is None else ok, "issues": list(issues), "suggested_text": text})


def audit(store, kb, tmp_path, critic_turn, draft=None):
    draft = draft or good_draft(store)
    llm = FakeLLM([required_calls(), final(draft), critic_turn])
    rec = orchestrator.run_audit(DISTRICT, SEASON, llm=llm, store=store, kb=kb, root=tmp_path)  # the Critic is on by default
    return rec, llm, draft


def test_no_issues_leaves_the_brief_and_is_recorded(store, kb, tmp_path):
    rec, llm, draft = audit(store, kb, tmp_path, review())
    assert rec.mode == "agent" and rec.briefing == draft["briefing"]
    assert rec.critic == {"ok": True, "issues": [], "rewrite_used": False, "error": None, "rewrite_rejected": [],
                          "original_briefing": None, "warning": None}
    assert rec.trace[-1]["agent"] == "critic" and len(llm.calls) == 3


def test_the_critic_gets_the_brief_type_and_the_text_of_the_cited_chunks(store, kb, tmp_path):
    _, llm, draft = audit(store, kb, tmp_path, review())
    message = llm.calls[-1]["messages"][-1]["content"]
    assert "Brief type: audit" in message and draft["briefing"] in message and "150 to 200" in message
    assert "[K001] Burned-area product\nCropland burned area is low confidence." in message
    assert "Gap method" not in message  # K000 is not cited, so its text is not sent
    assert llm.calls[-1]["tools"] in (None, [])


def test_a_rewrite_that_passes_is_used(store, kb, tmp_path):
    new = good_draft(store)["briefing"].replace(REMOVABLE, "", 1)
    rec, _, draft = audit(store, kb, tmp_path, review([issue("tone")], new))
    assert rec.mode == "agent+critic" and rec.briefing == new.strip() and rec.verification["ok"]
    assert rec.critic["rewrite_used"] and rec.critic["original_briefing"] == draft["briefing"]
    saved = json.loads((tmp_path / "audits" / "2024_Sangrur.json").read_text())
    assert saved["mode"] == "agent+critic" and saved["critic"]["issues"][0]["type"] == "tone"
    assert orchestrator.load_reusable_audit(DISTRICT, SEASON, tmp_path).mode == "agent+critic"


@pytest.mark.parametrize("old, new, why", [
    # 2024 is in the evidence (the season), but it is not in the brief: still rejected
    ("field surveys", "field surveys in 2024", "adds or changes a number: 2024"),
    ("field surveys", "field surveys of 50 fields", "adds or changes a number: 50"),
    ("[K001]", "[K001] [K000]", "adds a citation: K000"),
    ("This is triangulation, not field verification.", "This is certain.", "fails the Verifier"),
    ("What would settle it:", "Next:", "fails the Verifier"),
])
def test_an_unacceptable_rewrite_is_rejected_and_the_original_kept(store, kb, tmp_path, old, new, why):
    text = good_draft(store)["briefing"].replace(old, new)
    rec, _, draft = audit(store, kb, tmp_path, review([issue()], text))
    assert rec.mode == "agent" and rec.briefing == draft["briefing"] and rec.verification["ok"]
    assert not rec.critic["rewrite_used"] and any(why in r for r in rec.critic["rewrite_rejected"]), rec.critic
    assert rec.critic["issues"]  # the issues are still stored


def test_a_rewrite_cannot_drop_a_minus_sign():
    draft = {"headline": "h", "briefing": "The estimate runs from -62.2% to 25.7%.", "citations": []}
    ok = lambda d: VerifyResult(True, [], [])
    kept = critic.apply_rewrite(draft, "From -62.2% to 25.7%, the estimate runs.", ok)
    assert kept[0] and kept[2] == []
    flipped = critic.apply_rewrite(draft, "The estimate runs from 62.2% to 25.7%.", ok)
    assert flipped[0] is None and "number: 62.2" in flipped[2][0]


def test_hyphens_in_ranges_and_names_are_not_minus_signs():
    assert critic._signed_numbers("30-50% from NOAA-20, up -3") == {(30.0, False), (50.0, False), (3.0, True)}


def test_an_unsupported_citation_with_no_rewrite_adds_a_warning(store, kb, tmp_path):
    rec, _, draft = audit(store, kb, tmp_path, review([issue("unsupported_citation")]))
    assert rec.mode == "agent" and rec.verification["ok"] and rec.critic["warning"] == critic.CITATION_WARNING
    rec, _, _ = audit(store, kb, tmp_path, review([issue("overclaim")]))
    assert rec.critic["warning"] is None


def test_a_rewrite_that_fixes_the_citation_has_no_warning(store, kb, tmp_path):
    draft = good_draft(store)
    draft["briefing"] += " Field notes. [K001]"
    new = draft["briefing"].replace(" Field notes. [K001]", " Field notes.")
    rec, _, _ = audit(store, kb, tmp_path, review([issue("unsupported_citation")], new), draft=draft)
    assert rec.critic["rewrite_used"] and rec.critic["warning"] is None and rec.citations == ["K001"]


@pytest.mark.parametrize("turn", [
    final("not json at all"),
    final({"verdict": "fine"}),
    final({"ok": False, "issues": [{"type": "nonsense"}]}),
    LLMError("quota"),
])
def test_a_critic_failure_keeps_the_verified_brief(store, kb, tmp_path, turn):
    llm = FakeLLM([required_calls(), final(good_draft(store)), turn, turn])  # bad JSON gets one retry
    rec = orchestrator.run_audit(DISTRICT, SEASON, llm=llm, store=store, kb=kb, root=tmp_path)
    assert rec.mode == "agent" and rec.verification["ok"] and rec.briefing == good_draft(store)["briefing"]
    assert rec.critic["error"] and not rec.critic["ok"] and rec.critic["warning"] is None


def test_the_critic_does_not_run_on_a_template_or_when_switched_off(store, kb, tmp_path):
    llm = FakeLLM([LLMError("down")])
    rec = orchestrator.run_audit(DISTRICT, SEASON, llm=llm, store=store, kb=kb, root=tmp_path)
    assert rec.mode == "template" and rec.critic is None and len(llm.calls) == 1
    llm = FakeLLM([required_calls(), final(good_draft(store))])
    rec = orchestrator.run_audit(DISTRICT, SEASON, llm=llm, store=store, kb=kb, root=tmp_path, use_critic=False)
    assert rec.mode == "agent" and rec.critic is None


def test_briefings_get_the_critic_only_when_asked(tmp_path):
    from psw.agents.tools import DataStore
    from tests.fakes import FakeKB
    from tests.test_agent import DAY, DISTRICT as PLACE, four_calls, good_draft as briefing_draft

    data = DataStore()
    llm = FakeLLM([four_calls(), final(briefing_draft(data))])
    rec = orchestrator.run_briefing(PLACE, DAY, llm=llm, store=data, kb=FakeKB(), root=tmp_path)
    assert rec.critic is None
    llm = FakeLLM([four_calls(), final(briefing_draft(data)), review([issue("tone")])])
    rec = orchestrator.run_briefing(PLACE, DAY, llm=llm, store=data, kb=FakeKB(), root=tmp_path, use_critic=True)
    assert rec.mode == "agent" and rec.critic["issues"] and "Brief type: briefing" in llm.calls[-1]["messages"][-1]["content"]


def test_the_app_treats_a_critic_mode_like_its_base_mode():
    from psw.app.helpers import badge

    assert critic.base_mode("agent_revised+critic") == "agent_revised"
    kind, text = badge({"mode": "agent_revised+critic", "verification": {"ok": True}})
    assert kind == "success" and "after one revision" in text


def test_a_rewrite_may_remove_a_citation_but_not_move_it():
    old = "Counts are satellite based. [K001] MODIS burned area is down from the base season."
    removed = "Counts are satellite based. MODIS burned area is down from the base season."
    moved = "Counts are satellite based. MODIS burned area is down from the base season. [K001]"
    reworded = "The counts are satellite based [K001]. Burned area from MODIS is down from the base season."
    reordered = "MODIS burned area is down from the base season. Counts are satellite based. [K001]"
    assert critic.moved_citations(old, removed) == [] and critic.moved_citations(old, reworded) == []
    assert critic.moved_citations(old, reordered) == [] and critic.moved_citations(old, moved) == ["K001"]
    draft = {"headline": "h", "briefing": old, "citations": ["K001"]}
    candidate, _, reasons = critic.apply_rewrite(draft, moved, lambda d: VerifyResult(True, [], []))
    assert candidate is None and "moves a citation" in reasons[0]


CHUNKS = [{"chunk_id": "K001", "title": "t", "text": "The product is mapped at 500 m. Omission error 72.6%. See 2022."}]


def test_a_figure_that_is_not_in_the_cited_chunk_is_found_by_code():
    text = ("Burned area fell 61.6% from 2022 [K001]. The omission error is 72.6% [K001]. "
            "Cropland is hard to map [K001]. Counts fell 80% from 2023 with no citation.")
    found = critic.figure_citations(text, CHUNKS)
    assert [f["note"] for f in found] == ["K001 does not contain the figure(s) 61.6"] and found[0]["by"] == "check"
    assert "61.6" in critic.user_message({"briefing": text, "headline": "h"}, "audit", CHUNKS, (150, 200)).split("automatic check")[1]


def test_the_code_finding_is_recorded_even_when_the_critic_says_ok_or_fails():
    draft = {"headline": "h", "briefing": "Burned area fell 61.6% [K001].", "citations": ["K001"]}
    ok = lambda d: VerifyResult(True, [], [])
    for turn in (review(), LLMError("quota")):
        done = critic.review(draft, "audit", CHUNKS, FakeLLM([turn]), ok, (1, 50))
        assert done.info["warning"] == critic.CITATION_WARNING and not done.info["ok"]
        assert done.info["issues"][0]["by"] == "check" and done.draft == draft


def test_a_rewrite_that_removes_the_marker_clears_the_code_finding():
    draft = {"headline": "h", "briefing": "Burned area fell 61.6% [K001]. Cropland is hard to map [K001].",
             "citations": ["K001"]}
    new = "Burned area fell 61.6%. Cropland is hard to map [K001]."
    done = critic.review(draft, "audit", CHUNKS, FakeLLM([review(text=new, ok=True)]),
                         lambda d: VerifyResult(True, [], []), (1, 50))
    assert done.info["rewrite_used"] and done.info["warning"] is None and done.draft["briefing"] == new
    assert done.info["issues"][0]["by"] == "check"  # recorded as found and fixed


def test_a_headline_repeated_on_top_of_a_rewrite_is_dropped():
    body = "What detections say: counts fell.\n\nWhat would settle it: field checks."
    assert critic.drop_headline("Audit of 2025\n\n" + body, "Audit of 2025") == body
    assert critic.drop_headline("Some other title\n" + body, "Audit of 2025") == body
    assert critic.drop_headline(body, "Audit of 2025") == body
    assert critic.drop_headline("Counts fell.\nMore text.", "Audit") == "Counts fell.\nMore text."


def test_one_problem_is_not_listed_twice():
    draft = {"headline": "h", "briefing": "What other signals say: burned area fell 61.6% [K001].", "citations": ["K001"]}
    said = issue("unsupported_citation", "burned area fell 61.6% [K001].")
    done = critic.review(draft, "audit", CHUNKS, FakeLLM([review([said])]), lambda d: VerifyResult(True, [], []), (1, 50))
    assert len(done.info["issues"]) == 1 and done.info["warning"] == critic.CITATION_WARNING


@pytest.mark.parametrize("text, ok", [
    ("The range is 10% to 20%, confidence low.", True),
    ("Our confidence in this range is low.", True),
    ("The confidence is low.", True),
    ("The MCD64A1 product is of low confidence.", False),
    ("Burned area in cropland is low confidence.", False),
])
def test_the_confidence_must_be_stated_for_the_range(text, ok):
    from psw.agents.verifier import _gap_errors

    gap_result = {"gap_low_pct": 10.0, "gap_high_pct": 20.0, "confidence": "low", "direction": "counts fell more than burned area"}
    errors = _gap_errors(text + " Counts per burned km². The range omits burned-area error.", gap_result)
    assert (not any("confidence of the range" in e for e in errors)) is ok


def test_a_method_note_cannot_be_cited_for_a_places_result():
    chunks = [{"chunk_id": "K001", "title": "method", "text": "A negative value means burned area fell more than counts."},
              {"chunk_id": "K002", "title": "audit", "text": "The gap is 30% (low; counts fell more than burned area)."}]
    assert critic.result_citations("Counts fell more than burned area [K001].", chunks)[0]["by"] == "check"
    assert critic.result_citations("The direction is unclear. [K001]", chunks)
    assert not critic.result_citations("Counts fell more than burned area [K002].", chunks)
    assert not critic.result_citations("Counts fell more than burned area. The method is explained here [K001].", chunks)
    draft = {"headline": "h", "briefing": "Counts fell more than burned area [K001].", "citations": ["K001"]}
    done = critic.review(draft, "audit", chunks, FakeLLM([review()]), lambda d: VerifyResult(True, [], []), (1, 50))
    assert done.info["warning"] == critic.CITATION_WARNING  # even when the model says all is well
