"""Dispatcher tools, template, Verifier checks and orchestrator, on the committed data with a scripted LLM. No network."""
import json
import re
from dataclasses import replace
from datetime import date

import pandas as pd
import pytest

from psw import config
from psw.agents import dispatch_template, dispatch_tools, dispatcher, orchestrator, verifier
from psw.agents.dispatch_tools import ResponseStore, Settings
from psw.agents.verifier import (FIELD_NAME, Evidence, Plan, lowercase_words, name_errors, unit_errors,
                                  verify_dispatch)
from psw.llm import LLMError
from tests.fakes import FakeLLM, calls, final

DAY = "2025-10-28"


@pytest.fixture(scope="module")
def store():
    return ResponseStore()


class KB:
    """The gap-method note (K000) and one ordinary chunk (K001). Records every search."""

    def __init__(self):
        self.chunks = pd.DataFrame([
            {"chunk_id": "K000", "doc_id": "method_detection_gap", "title": "Gap method", "doc_type": "curated",
             "text": "The range does not include the error of the burned-area product."},
            {"chunk_id": "K001", "doc_id": "crm_options", "title": "Straw options", "doc_type": "curated",
             "text": "Custom Hiring Centres let machines be shared."}])
        self.searches = []

    def search(self, query, k=4, before=None, district=None, audit_place=None):
        self.searches.append({"query": query, "before": before, "audit_place": audit_place})
        return [{**self.chunks.iloc[1].to_dict(), "score": 1.0}]


@pytest.fixture
def kb():
    return KB()


class RealChunks:
    """The committed knowledge base, without the embedding model: a re-check only reads the chunk texts."""

    def __init__(self):
        self.chunks = pd.read_parquet(config.PROCESSED / "kb_chunks.parquet")


def good_draft(store, day=DAY, district=None, settings=Settings()) -> tuple[dict, str]:
    """A draft that passes (the template's text, citing the chunk the fake search returns) and the leading district."""
    facts, _, _ = orchestrator._dispatch_template_facts(store, KB(), day, settings, district)
    draft = dispatch_template.render(facts)
    draft["briefing"] = draft["briefing"].replace("[K000]", "[K001]")
    draft["citations"] = ["K001"]
    del draft["label"]
    return draft, facts["allocation"]["districts"][0]["district"]


def tool_turn(leader, extra=(), allocation_args=None):
    return calls(("get_risk_forecast", {}), ("get_allocation", allocation_args or {}),
                 ("get_rain_outlook", {"district": leader}), ("get_hotspot_cells", {"district": leader}),
                 ("get_gap_range", {"district": "Punjab", "season": 2025}), ("search_knowledge", {"query": "straw"}), *extra)


CRITIC_OK = {"ok": True, "issues": [], "suggested_text": None}


# ---------- orchestrator ----------

def test_the_users_units_and_limits_override_whatever_the_model_asks_for(store, kb, tmp_path):
    draft, leader = good_draft(store)
    llm = FakeLLM([tool_turn(leader, allocation_args={"date": "2020-01-01", "units": 10_000, "max_share": 1.0,
                                                      "use_likely": False}), final(draft), final(CRITIC_OK)])
    rec = orchestrator.run_dispatch(DAY, llm=llm, store=store, kb=kb, root=tmp_path)
    assert rec.mode == "agent" and rec.verification["ok"], rec.first_attempt_errors
    used = next(t["args"] for t in rec.trace if t["tool"] == "get_allocation")
    assert used == {"date": DAY, "units": 500, "max_share": 0.25, "use_likely": True}
    result = rec.facts["get_allocation"]
    assert (result["units_available"], result["limit_per_district"], result["planned_on"]) == (500, 125, "likely burning")
    assert sum(r["units"] for r in rec.allocation) == 500 and max(r["units"] for r in rec.allocation) <= 125
    assert next(t["args"] for t in rec.trace if t["tool"] == "get_gap_range")["season"] == 2024  # the previous season
    assert kb.searches[0]["before"] == DAY
    assert rec.moved[0]["district"] == "Fazilka" and sum(r["units"] for r in rec.detected_only) == 500
    assert "Not for enforcement" in rec.caveat and rec.critic["ok"]
    assert [t["agent"] for t in rec.trace][-1] == "critic"
    saved = json.loads((tmp_path / "dispatch" / f"{DAY}.json").read_text())
    assert saved["allocation"] == rec.allocation


def test_an_enforcement_word_forces_a_revision_and_then_the_template(store, kb, tmp_path):
    draft, leader = good_draft(store)
    bad = {**draft, "briefing": draft["briefing"] + " Penalties will follow."}
    llm = FakeLLM([tool_turn(leader), final(bad), final(draft)])
    rec = orchestrator.run_dispatch(DAY, llm=llm, store=store, kb=kb, root=tmp_path, use_critic=False)
    assert rec.mode == "agent_revised" and any("not enforcement" in e for e in rec.first_attempt_errors)
    llm = FakeLLM([tool_turn(leader), final(bad), final(bad)])
    rec = orchestrator.run_dispatch(DAY, llm=llm, store=store, kb=kb, root=tmp_path)
    assert rec.mode == "template" and rec.verification["ok"] and "Penalt" not in rec.briefing


def test_a_dead_llm_gives_the_template_with_a_district_paragraph(store, kb, tmp_path):
    rec = orchestrator.run_dispatch(DAY, district="Sangrur", llm=FakeLLM([LLMError("down")]), store=store, kb=kb,
                                    root=tmp_path)
    assert rec.mode == "template" and rec.verification["ok"], rec.verification["errors"]
    assert rec.district_paragraph.startswith("This is Sangrur (") and rec.label


def test_a_saved_dispatch_is_reused_only_for_the_same_inputs(store, kb, tmp_path):
    draft, leader = good_draft(store)
    orchestrator.run_dispatch(DAY, llm=FakeLLM([tool_turn(leader), final(draft)]), store=store, kb=kb, root=tmp_path,
                              use_critic=False)
    again = orchestrator.run_dispatch(DAY, llm=FakeLLM([]), store=store, kb=kb, root=tmp_path, reuse=True)
    assert again.mode == "agent"
    other = orchestrator.run_dispatch(DAY, units=300, llm=FakeLLM([LLMError("x")]), store=store, kb=kb, root=tmp_path,
                                      reuse=True)
    assert other.mode == "template" and other.units == 300


def test_a_critic_rewrite_that_drops_a_required_part_is_rejected(store, kb, tmp_path):
    draft, leader = good_draft(store)
    shorter = re.sub(r"What to watch:.*", "", draft["briefing"], flags=re.S).strip()
    llm = FakeLLM([tool_turn(leader), final(draft),
                   final({"ok": False, "issues": [{"type": "tone", "sentence": "x", "note": "y"}],
                          "suggested_text": shorter})])
    rec = orchestrator.run_dispatch(DAY, llm=llm, store=store, kb=kb, root=tmp_path)
    assert rec.mode == "agent" and rec.briefing == draft["briefing"] and rec.critic["rewrite_rejected"]


def test_dates_without_a_forecast_are_refused(store, kb, tmp_path):
    for bad in ("2025-11-28", "2023-10-28"):
        with pytest.raises(ValueError):
            orchestrator.run_dispatch(bad, llm=FakeLLM([]), store=store, kb=kb, root=tmp_path)
    with pytest.raises(ValueError):
        orchestrator.run_dispatch(DAY, district="Nowhere", llm=FakeLLM([]), store=store, kb=kb, root=tmp_path)


def test_the_template_passes_on_sampled_dates_for_both_planning_bases(store, kb, tmp_path):
    for day in store.dates[::13]:
        for use_likely in (True, False):
            rec = orchestrator.run_dispatch(day, use_likely=use_likely, llm=FakeLLM([LLMError("off")]), store=store, kb=kb,
                                            save=False)
            assert rec.verification["ok"], (day, use_likely, rec.verification["errors"])
            assert ("not a count" in rec.briefing) == use_likely


# ---------- the checks ----------

PLAN = Plan(units={"Fazilka": 57, "Sangrur": 46, "SAS Nagar (Mohali)": 0},
            totals={"fleet": 500, "placed": 500, "unplaced": 0, "cap": 125, "others": 397},
            words=lowercase_words(["a sound brief uses ordinary words such as firm check sky estimates follow penalties apply "
                                   "file fines issued enforcement police ask fir tree fine particles data because without"]),
            detected_units={"Fazilka": 34, "Sangrur": 46, "SAS Nagar (Mohali)": 0}, use_likely=True,
            most_moved="Fazilka", confidence="low")
TEXT = """Where the units go: Of 500 units, Fazilka (57 units) and Sangrur (46 units) lead. The limit per district is 125 units.

Why: the adjustment moves Fazilka from 34 units to 57 units.

How sure we are: the forecast is uncertain and the detection gap figures are not a count of fires. [K001] The range behind Fazilka has low confidence.

What to watch: rain in Fazilka could delay burning."""


def check(text, plan=PLAN, **kw):
    evidence = Evidence(allowed_numbers=[0, 1, 3, 9, 34, 46, 57, 125, 500], retrieved_ids={"K001"})
    return verify_dispatch({"headline": "Plan", "briefing": text, "citations": ["K001"], **kw.pop("extra", {})},
                           evidence, plan, word_range=(1, 300), **kw)


def test_a_sound_brief_passes():
    assert check(TEXT).errors == []


@pytest.mark.parametrize("words", ["Penalties apply.", "File an FIR.", "Fines follow.", "A fine was issued.",
                                   "This is not for enforcement.", "Ask the police."])
def test_enforcement_words_are_banned(words):
    assert any("not enforcement" in e for e in check(f"{TEXT} {words}").errors)


def test_harmless_look_alikes_are_allowed():
    assert check(TEXT + " A fir tree, fine particles and FIRMS data.").errors == []


@pytest.mark.parametrize("old, new, message", [
    ("Fazilka (57 units) and Sangrur (46 units)", "Fazilka (46 units) and Sangrur (57 units)", "Fazilka gets 57 units"),
    ("from 34 units to 57 units", "from 57 units to 34 units", "from 34 units to 57 units, not"),
    ("could delay burning.", "could delay burning. Another 57 units wait.", "read as an allocation"),
    ("detection gap", "adjustment", "detection gap"),
    ("not a count of fires", "estimates", "not a count"),
    ("rain in Fazilka could delay burning", "check the sky over Fazilka", "rain"),
    ("rain in Fazilka", "rain in Sangrur", "must be about Fazilka"),
    ("uncertain", "firm", "uncertain"),
    ("Why:", "Because:", "labelled 'Why:'"),
    ("has low confidence", "has medium confidence", "low confidence"),
    ("uncertain and", "uncertain with high confidence and", "low confidence"),
    ("Why: the", "Why: expected burning is high; the", "likely burning"),
    ("rain in Fazilka could delay burning", "rain in Fazilka over the next 3 days is 0 mm", "weather archive"),
])
def test_what_a_brief_must_get_right(old, new, message):
    assert any(message in e for e in check(TEXT.replace(old, new)).errors)


def test_the_most_moved_and_the_leading_district_must_be_named():
    errors = check(TEXT.replace("Fazilka", "That district")).errors
    assert any("most units: Fazilka" in e for e in errors) and any("moves most: Fazilka" in e for e in errors)


@pytest.mark.parametrize("sentence", [
    "Fazilka gets 46 units and Sangrur gets 57 units.",  # swapped, both valid figures
    "Fazilka gets 57 units and Sangrur gets 46 units.",  # right, but not a form that can be checked
    "Fazilka (46 units) and Sangrur (57 units) lead.",
    "Fazilka gets 46 and Sangrur 57.",
    "Fazilka: 46, Sangrur: 57 units.",
    "Send 57 machines to Sangrur.",
    "Sangrur takes 125 units.",  # the limit is not Sangrur's figure
    "Sangrur takes all 500 units.",
    "Sangrur and Fazilka share 397 units.",
    "Sangrur has a limit of 57 units per district.",
    "Sangrur receives fifty-seven units.",
    "Sangrur should get 57 of 500 units.",
    "SAS Nagar gets 57 units.",
    "Next is Sangrur, 57.",
    "Then Sangrur at 57.",
])
def test_no_unit_figure_passes_unless_it_is_bound_to_the_right_district(sentence):
    assert unit_errors(sentence, PLAN), sentence
    assert check(f"{TEXT} {sentence}").errors


@pytest.mark.parametrize("sentence", [
    "Fazilka (57 units) leads, and SAS Nagar (0 units) waits.",
    "The adjustment moves Fazilka from 34 units to 57 units.",
    "The other districts share 397 units.",
    "The other 19 districts share 397 units.",
    "No district may take more than 125 units, and 0 units are not placed.",
    "Awareness teams might start at cell 1070 with 6 detections and expect 0.0 mm of rain over 3 days.",
    "What one unit achieves is assumed.",
    "Demand for each machine falls in a window of 15 to 20 days.",
])
def test_the_approved_forms_and_labelled_numbers_pass(sentence):
    assert unit_errors(sentence, PLAN) == [], sentence


def test_a_headline_may_carry_no_bare_figure():
    evidence = Evidence(allowed_numbers=[0, 1, 3, 34, 46, 57, 125, 500], retrieved_ids={"K001"})
    for headline, ok in (("Sangrur first with 57", False), ("3-day plan from 2025-10-28: Fazilka first", True),
                         ("Of 500 units, Fazilka first", True)):
        draft = {"headline": headline, "briefing": TEXT, "citations": ["K001"]}
        errors = verify_dispatch(draft, Evidence([*evidence.allowed_numbers, 10, 28, 2025], {"K001"}), PLAN,
                                 word_range=(1, 300)).errors
        assert (errors == []) == ok, (headline, errors)


def test_a_bare_number_in_the_where_part_is_rejected():
    text = TEXT.replace("The limit per district", "Sangrur is next at 46. The limit per district")
    assert any("read as an allocation" in e for e in check(text).errors)


@pytest.mark.parametrize("sentence", [
    "Start at Bhawanigarh village.", "Teams should visit Bhawanigarh.", "Bhawanigarh needs help.",
    "Ask Gurpreet Singh.", "Teams should go to the village of bhawanigarh.", "Longowal and Dirba tehsil come first.",
    "Visit BHAWANIGARH.", "Go to ਭਵਾਨੀਗੜ੍ਹ today.",
])
def test_village_and_personal_names_are_rejected(sentence):
    assert name_errors(sentence, PLAN), sentence
    assert check(f"{TEXT} {sentence}").errors


def test_district_names_project_terms_and_ordinary_capitalised_words_pass():
    text = ("Dispatch Brief for SAS Nagar (Mohali), Sangrur and Fazilka in Punjab. Custom Hiring Centres and VIIRS S-NPP. "
            "Overall, the Forecast uses October data.")
    assert name_errors(text, Plan(units=PLAN.units, detected_units=PLAN.detected_units, totals=PLAN.totals,
                                  use_likely=True, words=lowercase_words(["uses data forecast"]))) == []


def test_lowercase_words_ignore_names_and_web_addresses():
    assert lowercase_words(["Bhawanigarh has rain, see https://example.org/gurpreet"]) == {"has", "rain", "see"}


def test_a_village_in_the_headline_or_paragraph_is_rejected(store, kb, tmp_path):
    draft, leader = good_draft(store)
    bad = {**draft, "headline": "Units for Bhawanigarh"}
    llm = FakeLLM([tool_turn(leader), final(bad), final(draft)])
    rec = orchestrator.run_dispatch(DAY, llm=llm, store=store, kb=kb, root=tmp_path, use_critic=False)
    assert rec.mode == "agent_revised" and any("looks like a name" in e for e in rec.first_attempt_errors)
    paragraph = "Sangrur (46 units) should hear local requests. Start at Bhawanigarh. " + "Teams check the ground. " * 8
    assert any("district paragraph" in e and "Bhawanigarh" in e for e in
               check(TEXT, extra={"district_paragraph": paragraph}, district="Sangrur").errors)


def test_next_days_rain_must_be_called_archive_data():
    text = TEXT + " The archive, standing in for a forecast, shows 1 mm of rain over the next 3 days."
    assert not any("archive" in e for e in check(text).errors)


def test_the_confidence_is_that_of_the_district_moved_most(store):
    a = dispatch_tools.plan_allocation(store, DAY, Settings())
    assert a.moves(1)[0]["district"] == "Fazilka"
    assert a.moves(1)[0]["confidence_of_its_gap_range"] == "low" == a.plan().confidence  # Punjab's is medium
    assert store.audit.rows("gap_range", "Punjab", 2024)["confidence"].iloc[0] == "medium"
    assert dispatch_tools.plan_allocation(store, DAY, Settings(use_likely=False)).plan().confidence is None


def test_the_district_paragraph_is_checked_too():
    paragraph = "Sangrur (46 units) is in the plan. " + "Teams should check the ground and listen to local requests. " * 4
    assert check(TEXT, extra={"district_paragraph": paragraph}, district="Sangrur").errors == []
    assert any("missing" in e for e in check(TEXT, district="Sangrur").errors)
    wrong = check(TEXT, extra={"district_paragraph": paragraph.replace("46 units", "57 units")}, district="Sangrur")
    assert any("district paragraph" in e for e in wrong.errors)
    assert any("not enforcement" in e for e in
               check(TEXT, extra={"district_paragraph": paragraph + " Fines follow."}, district="Sangrur").errors)
    assert any("asked for" in e for e in check(TEXT, extra={"district_paragraph": paragraph}).errors)


# ---------- tools ----------

def test_tools_report_plain_values_and_no_field_names(store):
    outputs = [dispatch_tools.get_risk_forecast(store, DAY, True, "Moga"), dispatch_tools.get_allocation(store, DAY),
               dispatch_tools.get_hotspot_cells(store, "Fazilka", DAY), dispatch_tools.get_rain_outlook(store, "Moga", DAY),
               dispatch_tools.get_gap_range(store, "Punjab", 2024), dispatch_tools.get_machine_stock(store, "Sangrur", DAY)]

    def strings(x):
        if isinstance(x, dict):
            return [s for k, v in x.items() if k != "source_url" for s in strings(v)]
        if isinstance(x, list):
            return [s for v in x for s in strings(v)]
        return [x] if isinstance(x, str) else []

    assert not any("error" in o for o in outputs)
    assert [s for o in outputs for s in strings(o) if FIELD_NAME.search(s)] == []
    assert outputs[0]["asked_district"]["district"] == "Moga" and len(outputs[0]["top_districts"]) == 8
    hot = outputs[2]
    assert hot["cells"] and all(c["detections"] >= 2 for c in hot["cells"])  # a single detection is never offered
    assert outputs[1]["moved_by_the_likely_adjustment"][0]["change_in_words"] == "23 more units with the adjustment"


def test_detected_only_planning_never_mentions_likely_burning(store):
    r = dispatch_tools.get_risk_forecast(store, DAY, use_likely=False)
    a = dispatch_tools.get_allocation(store, DAY, use_likely=False)
    assert "likely" not in json.dumps([r["top_districts"], a["districts"]]) and "moved_by_the_likely_adjustment" not in a
    assert all("forecast_detections" in d for d in a["districts"]) and "units_not_placed" not in a
    assert a["planned_on"] == "forecast detections"


def test_machine_figures_are_hidden_before_they_were_published(store):
    assert "not been published" in dispatch_tools.get_machine_stock(store, "Sangrur", "2024-11-25")["error"]
    out = dispatch_tools.get_machine_stock(store, "Sangrur", date(2024, 11, 26))
    assert out["machines_supplied"] == 12900 and out["largest_types"][0] == {"type": "super seeder", "supplied": 6604}
    assert "not used in the allocation" in out["caveat"]


def test_the_dispatcher_spec_requires_the_planned_tools(store, kb):
    spec = dispatcher.build_dispatcher(store, kb)
    assert spec.required_tools == ["get_risk_forecast", "get_allocation", "search_knowledge"]
    assert {t.name for t in spec.tools} == {s["function"]["name"] for s in dispatch_tools.TOOL_SCHEMAS}
    shown = next(t for t in spec.tools if t.name == "get_allocation").parameters["properties"]
    assert set(shown) == {"date", "district"}  # no units, share or basis for the model to set


# ---------- the saved record keeps every figure, so it can be checked again ----------

def _recheck(rec, store, kb, facts):
    return orchestrator.recheck_dispatch({**rec.to_dict(), "facts": facts}, store, kb)


def test_a_record_with_two_districts_can_be_checked_again_from_its_facts(store, kb, tmp_path):
    rec = orchestrator.run_dispatch(DAY, district="Sangrur", llm=FakeLLM([LLMError("down")]), store=store, kb=kb,
                                    root=tmp_path)
    leader = rec.allocation[0]["district"]
    assert leader != "Sangrur"
    assert rec.facts["get_rain_outlook"]["district"] == "Sangrur"  # the plain name holds the last call
    assert rec.facts[f"get_rain_outlook ({leader})"]["district"] == leader
    assert rec.facts[f"get_hotspot_cells ({leader})"]["district"] == leader
    assert _recheck(rec, store, kb, rec.facts).ok
    last_only = {name: result for name, result in rec.facts.items() if " (" not in name}
    assert not _recheck(rec, store, kb, last_only).ok  # the leading district's cell and rain figures would be missing


def test_facts_keep_distinct_results_and_drop_repeats_and_errors():
    def call(district, result):
        return {"tool": "get_rain_outlook", "args": {"district": district}, "result": result}

    a, b = {"district": "Moga", "mm": 1}, {"district": "Barnala", "mm": 2}
    facts = orchestrator._facts([call("Moga", a), call("Moga", a), call("Nowhere", {"error": "x"}), call("Barnala", b)])
    assert facts == {"get_rain_outlook": b, "get_rain_outlook (Moga)": a}
    assert orchestrator._facts([call("Nowhere", {"error": "x"})]) == {"get_rain_outlook": {"error": "x"}}


def test_the_writing_instructions_name_no_tool_or_field():
    instructions = dispatcher.SYSTEM_PROMPT.split("The briefing text has exactly these four parts")[1]
    assert FIELD_NAME.findall(instructions) == []


# ---------- rain and cells must be the described district's own ----------

def test_a_brief_cannot_describe_a_district_whose_rain_and_cells_were_never_fetched(store, kb, tmp_path):
    """Seen live: only the asked district's tools were called, and its figures were written under the leader's name."""
    draft, leader = good_draft(store, district="Sangrur")
    only_sangrur = calls(("get_risk_forecast", {}), ("get_allocation", {"district": "Sangrur"}),
                         ("get_rain_outlook", {"district": "Sangrur"}), ("get_hotspot_cells", {"district": "Sangrur"}),
                         ("get_gap_range", {"district": "Punjab"}), ("search_knowledge", {"query": "straw"}))
    llm = FakeLLM([only_sangrur, final(draft), final(draft)])
    rec = orchestrator.run_dispatch(DAY, district="Sangrur", llm=llm, store=store, kb=kb, root=tmp_path,
                                    use_critic=False)
    assert rec.mode == "template"
    assert any(f"call the rain outlook and hotspot cell tools for {leader}" in e for e in rec.first_attempt_errors)


SITE = Plan(units={"Fazilka": 60, "Barnala": 20}, detected_units={"Fazilka": 40, "Barnala": 5}, totals={},
            use_likely=False, rain={"Fazilka": (1.5, 0.0), "Barnala": (4.0, 2.0)},
            cells={"Fazilka": [1070, 1022], "Barnala": []})


@pytest.mark.parametrize("text, message", [
    ("Fazilka had 4.0 mm of rain. Start at cell 1070.", "not a rain figure for Fazilka"),
    ("Start at cell 495.", "cell 495 is not one of the cells"),
    ("Awareness teams might start where there is no cluster, not at cell 1070.", "is wrong"),
    ("Awareness teams should wait.", "naming a cell"),
])
def test_site_claims_are_checked_against_what_was_fetched(text, message):
    assert any(message in e for e in verifier._site_errors(text, ["Fazilka"], SITE, need_cell=True))


def test_a_named_cell_must_come_with_its_detections():
    """Seen in the demo run: every brief named cells and none said how many detections they had."""
    plan = replace(SITE, cell_detections={1070: 6, 1022: 5})
    check = lambda text: verifier._site_errors(text, ["Fazilka"], plan, need_cell=True)  # noqa: E731
    bare = check("Start at cell 1070 (north Fazilka) and cell 1022 (north-east Fazilka).")
    assert len(bare) == 2 and "give the 6 detections of cell 1070" in bare[0]
    assert check("Start at cell 1070, with 6 detections, and cell 1022, with 5 detections.") == []
    assert check("Cell 1070 and cell 1022 had 6 and 5 detections.") == []
    assert check("Start at cell 1070. It had 6 detections.")  # not in the sentence that names it


@pytest.mark.parametrize("sentence", [
    "The state says the number of machines is enough for the paddy area by their field capacity.",
    "Machinery is sufficient in this district.",
    "There are adequate balers.",
])
def test_a_brief_cannot_say_there_are_enough_machines(sentence):
    """Seen in the demo run: half a source sentence was quoted, saying the opposite of what it says."""
    assert verifier._sufficiency_errors(sentence)


def test_the_timing_constraint_itself_is_allowed():
    assert verifier._sufficiency_errors("A machine may not be available when a farmer wants it.") == []
    assert verifier._sufficiency_errors("There is not enough evidence for a range.") == []


def test_site_claims_that_match_pass():
    good = "Fazilka had 1.5 mm and then 0.0 mm. Start at cell 1022 (north Fazilka)."
    assert verifier._site_errors(good, ["Fazilka"], SITE, need_cell=True) == []
    assert verifier._site_errors("No cell in Barnala had a cluster.", ["Barnala"], SITE, need_cell=False) == []


def test_an_instruction_copied_into_the_brief_is_rejected(store, kb, tmp_path):
    draft, leader = good_draft(store)
    draft["briefing"] += " Say that what one unit achieves is assumed."
    llm = FakeLLM([tool_turn(leader), final(draft), final(draft)])
    rec = orchestrator.run_dispatch(DAY, llm=llm, store=store, kb=kb, root=tmp_path, use_critic=False)
    assert any("instruction was copied" in e for e in rec.first_attempt_errors)


def test_why_must_explain_the_district_with_the_most_units():
    """Seen live: "Why" gave the second district's forecast, under a split that Fazilka led."""
    wrong = "Why: Barnala has 7 forecast detections. The adjustment moves Fazilka up. How sure we are: not very."
    assert any("start 'Why' with Fazilka" in e for e in verifier._why_errors(wrong, SITE))
    right = "Why: Fazilka has 35 forecast detections. Barnala moves too. How sure we are: not very."
    assert verifier._why_errors(right, SITE) == []


# ---------- demo cache, Part 2 ----------

def test_demo_dispatch_dates_and_audit_targets_follow_fixed_rules(store):
    from psw.agents import demo
    days = demo.dispatch_dates(store.daily, date(2025, 10, 31), set(store.dates))
    assert [str(d) for d in days] == ["2025-10-28", "2025-10-31", "2025-11-14"]
    totals = store.daily[store.daily["season"] == 2025].groupby("date")["fire_count"].sum().rolling(3, min_periods=1).sum()
    late = days[-1]
    assert totals[late] < totals.max() / 4 and all(totals[d] >= totals.max() / 4 for d in totals.index
                                                   if days[1] < d < late)
    assert demo.audit_targets(store.audit, [2024, 2025]) == [
        ("Punjab", 2024), ("Punjab", 2025), ("Tarn Taran", 2025), ("Sangrur", 2025), ("Amritsar", 2025)]


def test_demo_cache_never_replaces_an_agent_brief_with_the_template(store, kb, tmp_path):
    from psw.agents import demo
    (tmp_path / "demo.json").write_text(json.dumps({"peak": "2025-10-31"}))
    draft, leader = good_draft(store)
    orchestrator.run_dispatch(DAY, llm=FakeLLM([tool_turn(leader), final(draft)]), store=store, kb=kb, root=tmp_path,
                              use_critic=False)
    saved = orchestrator.dispatch_path(DAY, tmp_path).read_text()
    log = []
    out = demo.build_part2_cache(llm=FakeLLM([LLMError("down")] * 20), audit_store=store.audit, response_store=store,
                                 kb=kb, root=tmp_path, redo_dispatch=True, log=log.append)
    assert orchestrator.dispatch_path(DAY, tmp_path).read_text() == saved
    assert any("kept the saved one" in line for line in log)
    modes = {d["date"]: d["mode"] for d in out["dispatches"]}
    assert modes[DAY] == "agent" and modes["2025-10-31"] == "template"  # nothing was saved for it before
    assert len(out["audits"]) == 5 and all(a["verified"] for a in out["audits"])
    assert json.loads((tmp_path / "demo.json").read_text())["dispatches"] == out["dispatches"]


def test_demo_cache_reuses_a_saved_agent_brief_without_calling_the_llm(store, kb, tmp_path):
    from psw.agents import demo
    rec, what = demo._cached("SAVED", False, lambda: pytest.fail("ran the agent"), lambda r: pytest.fail("saved"))
    assert (rec, what) == ("SAVED", "reused")


# ---------- what is committed ----------

@pytest.mark.parametrize("path", sorted((config.PROCESSED / "dispatch").glob("*.json")), ids=lambda p: p.stem)
def test_every_committed_dispatch_passes_the_current_verifier(path, store):
    """If a check is tightened, the saved demo briefs must be regenerated, not left behind."""
    result = orchestrator.recheck_dispatch(json.loads(path.read_text()), store, RealChunks())
    assert result.ok, result.errors


@pytest.mark.parametrize("sentence", [
    "Its likely burning is 437 detections.",
    "Likely burning is estimated at 237 forecast detections.",
    "Sangrur has 437 detections of likely burning.",
    "Its likely burning is about 1,204 fires.",
])
def test_likely_burning_is_never_given_as_a_count(sentence):
    """Seen in the committed 2025-10-31 brief, against its own caveat that likely burning is not a count of fires."""
    assert verifier._likely_as_count_errors(sentence)


@pytest.mark.parametrize("sentence", [
    "Its likely burning is estimated at 437.",
    "Likely burning is not a count of fires.",
    "Sangrur leads with likely burning of 437; it has 215 detections forecast.",
    "Likely burning scales each forecast by the previous season's detection gap.",
])
def test_likely_burning_wording_that_is_allowed(sentence):
    assert verifier._likely_as_count_errors(sentence) == []


def test_a_sentence_about_demand_must_say_for_what_and_when():
    assert verifier._demand_errors("Demand falls within 15 to 20 days.")
    assert verifier._demand_errors("Almost all the demand falls in the wheat sowing window.")
    assert verifier._demand_errors("The state says demand for straw machines falls in the wheat sowing window.") == []


def test_the_demo_cache_drops_a_saved_brief_that_no_longer_passes(store, kb, tmp_path):
    from psw.agents import demo
    (tmp_path / "demo.json").write_text(json.dumps({"peak": "2025-10-31"}))
    draft, leader = good_draft(store)
    orchestrator.run_dispatch(DAY, llm=FakeLLM([tool_turn(leader), final(draft)]), store=store, kb=kb, root=tmp_path,
                              use_critic=False)
    path = orchestrator.dispatch_path(DAY, tmp_path)
    record = json.loads(path.read_text())
    record["briefing"] += " The number of machines is enough."  # as if written before the check existed
    path.write_text(json.dumps(record))
    log = []
    out = demo.build_part2_cache(llm=FakeLLM([LLMError("down")] * 20), audit_store=store.audit, response_store=store,
                                 kb=kb, root=tmp_path, log=log.append)
    assert any("no longer passes" in line for line in log)
    assert {d["date"]: d["mode"] for d in out["dispatches"]}[DAY] == "template"
