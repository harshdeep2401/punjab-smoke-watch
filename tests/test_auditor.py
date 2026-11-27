"""Auditor tools, template, Verifier checks, planner variant and orchestrator, on synthetic audit tables. No network."""
import json

import pytest

from psw.agents import audit_template, audit_tools, auditor, orchestrator, planner
from psw.agents.audit_tools import AuditStore
from psw.agents.verifier import Evidence, extract_numbers, verify_audit
from psw.audit import gap
from psw.llm import LLMError
from tests.fakes import FakeLLM, calls, final
from tests.test_audit_gap import area, fires, official

SEASON = 2024
DISTRICT = "Sangrur"  # Moga is too small for a range


class KB:
    """The gap-method note (K000) and one ordinary chunk (K001). Records every search."""

    def __init__(self):
        import pandas as pd
        self.chunks = pd.DataFrame([
            {"chunk_id": "K000", "doc_id": "method_detection_gap", "title": "Gap method", "doc_type": "curated",
             "text": "The range does not include the error of the burned-area product."},
            {"chunk_id": "K001", "doc_id": "mcd64a1", "title": "Burned-area product", "doc_type": "curated",
             "text": "Cropland burned area is low confidence. Source https://example.org/report_30.11.pdf"}])
        self.searches = []

    def search(self, query, k=4, before=None, district=None, audit_place=None):
        self.searches.append({"query": query, "before": before, "district": district, "audit_place": audit_place})
        return [{**self.chunks.iloc[1].to_dict(), "score": 1.0}]


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    ms, ba, off = {}, {}, {}
    for season, n, night in ((2022, 1000, 2), (2023, 1000, 2), (2024, 100, 40)):
        for district, scale, km2 in (("Sangrur", 1, 1000 if season < 2024 else 700), ("Moga", 0, 20)):
            count = n if scale else 3
            ms[(district, "snpp", season)] = (count - (night if scale else 0), night if scale else 0)
            ms[(district, "modis", season)] = (100 if season < 2024 else 40, 0)
            ms[(district, "noaa20", season)] = (count, 0)
            ba[(district, season)] = km2
            off[(district, season)] = count
    root = tmp_path_factory.mktemp("processed")
    (root / "audit").mkdir()
    for name, table in gap.audit_tables(fires(ms), area(ba), official(off)).items():
        table.to_parquet(root / "audit" / f"{name}.parquet", index=False)
    return AuditStore(root)


@pytest.fixture
def kb():
    return KB()


# ---------- tools ----------

def test_gap_tool_returns_percentages_in_plain_words_with_the_caveat(store):
    r = audit_tools.get_gap_range(store, DISTRICT, SEASON)
    assert r["gap_low_pct"] == r["gap_high_pct"] == 85.7 and r["confidence"] == "low"
    assert r["methods"].startswith("S-NPP detections against burned area, base season 2022")
    assert "_" not in r["methods"] and r["caveat"] == gap.GAP_CAVEAT and "not the share" in r["what_the_gap_is"]
    assert r["detected_change_pct"] == -90.0 and r["reference_change_pct"] == -30.0
    assert r["changes_in_words"] == "counts fell 90.0% and burned area fell 30.0%"
    trend = audit_tools.get_detection_trend(store, DISTRICT, SEASON)
    assert trend["sensors"][0]["change_in_words"] == "down 90.0% from 2022"
    assert audit_tools.change_words(50.0, 2022) == "up 50.0% from 2022" and audit_tools.change_words(float("nan"), 2022) is None


def test_gap_tool_gives_no_range_for_small_numbers_or_the_base_season(store):
    small = audit_tools.get_gap_range(store, "Moga", SEASON)
    assert small["gap_low_pct"] is None and small["gap_high_pct"] is None
    assert small["confidence"] == "insufficient evidence" and "under 50 km²" in small["reason"]
    assert "base season" in audit_tools.get_gap_range(store, DISTRICT, 2022)["error"]


def test_trend_diurnal_burned_and_official_tools(store):
    trend = audit_tools.get_detection_trend(store, DISTRICT, SEASON)
    assert [s["satellite"] for s in trend["sensors"]] == ["VIIRS S-NPP", "VIIRS NOAA-20", "MODIS (Terra and Aqua)"]
    assert trend["sensors"][0]["change_vs_base_pct"] == -90.0 and "warning" not in trend["sensors"][0]
    assert "unreliable" in audit_tools.get_detection_trend(store, "Moga", SEASON)["sensors"][0]["warning"]
    d = audit_tools.get_diurnal_profile(store, DISTRICT, SEASON)
    assert d["night_share_pct"] == 40.0 and d["base_night_share_pct"] == 0.2 and d["night_share_change_points"] == 39.8
    b = audit_tools.get_burned_area_comparison(store, DISTRICT, SEASON)
    assert b["burned_km2"] == 700.0 and b["burned_change_vs_base_pct"] == -30.0 and "warning" not in b
    assert "too small" in audit_tools.get_burned_area_comparison(store, "Moga", SEASON)["warning"]
    o = audit_tools.get_official_vs_satellite(store, DISTRICT, SEASON)
    assert o["official_to_snpp_ratio"] == 1.0 and o["source_url"].startswith("https://")
    assert "error" in audit_tools.get_burned_area_comparison(store, "Atlantis", SEASON)
    for result in (trend, d, b, o):
        json.dumps(result, allow_nan=False)


def test_the_planner_sees_signal_names_only(store):
    names = audit_tools.available_signals(store, DISTRICT, SEASON)
    assert names == ["VIIRS S-NPP", "VIIRS NOAA-20", "MODIS (Terra and Aqua)", "MODIS burned area", "official counts"]
    message = planner.audit_user_message(DISTRICT, SEASON, names)
    assert not [n for n in extract_numbers(message.replace(str(SEASON), "")) if n.value > 0]


def test_unknown_district_or_season(store, kb):
    for district, season in (("Atlantis", SEASON), (DISTRICT, 2019)):
        with pytest.raises(ValueError):
            orchestrator.run_audit(district, season, llm=FakeLLM([]), store=store, kb=kb, save=False)


# ---------- verifier ----------

def test_satellite_names_and_addresses_are_not_numbers():
    assert [n.value for n in extract_numbers("NOAA-20 and NOAA-21, NOAA-20/21, INSAT-3DS and MCD64A1")] == []
    ev = Evidence.from_tool_results([{"tool": "x", "result": {"source_url": "https://example.org/ATR_632.pdf"}}])
    assert 632.0 not in ev.allowed_numbers


def good_draft(store, extra="", citations=("K001",)):
    g = audit_tools.get_gap_range(store, DISTRICT, SEASON)
    filler = "The signals are read together and each one has its own weakness that a reader should keep in mind. "
    text = (f"What detections say: S-NPP detections fell {abs(g['detected_change_pct'])}% from 2022. {filler * 3}"
            f"What other signals say: burned area fell {abs(g['reference_change_pct'])}%. [K001] {filler * 2}"
            f"Our best range and how sure we are: counts per burned km² fell {g['gap_low_pct']}% to "
            f"{g['gap_high_pct']}%, confidence low. "
            "The range does not include the error of the burned-area product. "
            f"What would settle it: field surveys. This is triangulation, not field verification. {filler}{extra}")
    return {"headline": f"{DISTRICT} audit for {SEASON}", "briefing": text, "citations": list(citations)}


def check(store, draft):
    results = [{"tool": "get_gap_range", "result": audit_tools.get_gap_range(store, DISTRICT, SEASON)},
               {"tool": "search_knowledge", "result": {"results": [{"chunk_id": "K001", "text": "Cropland."}]}}]
    return verify_audit(draft, Evidence.from_tool_results(results), gap=results[0]["result"])


def test_a_good_audit_draft_verifies(store):
    r = check(store, good_draft(store))
    assert r.ok, r.errors


@pytest.mark.parametrize("label", ["What detections say", "What other signals say",
                                   "Our best range and how sure we are", "What would settle it"])
def test_each_label_is_required(store, label):
    d = good_draft(store)
    d["briefing"] = d["briefing"].replace(f"{label}:", "Also:")
    assert any(label in e for e in check(store, d).errors)


@pytest.mark.parametrize("old, new, expected", [
    ("This is triangulation, not field verification.", "This is our estimate.", "triangulation"),
    ("The range does not include the error of the burned-area product.", "The range has limits.", "burned-area product"),
    ("confidence low", "high confidence", "never high"),
    ("confidence low", "confidence is high", "never high"),
    ("confidence low", "confidence unknown", "confidence of the range is low"),
    ("counts per burned km² fell", "the gap range is", "per burned km²"),
    ("field surveys", "field surveys of 4321 fields", "4321"),
    ("field surveys", "field surveys after a fall of -90.0 percent", "minus sign"),
    ("field surveys", "field surveys; NEVER call it the share missed", "instruction copied"),
])
def test_audit_only_rules(store, old, new, expected):
    d = good_draft(store)
    d["briefing"] = d["briefing"].replace(old, new)
    errors = check(store, d).errors
    assert any(expected in e for e in errors), errors


def test_the_audit_word_range_applies(store):
    short = {"headline": "x", "briefing": "What detections say: little.", "citations": ["K001"]}
    assert any("outside 150-200" in e for e in check(store, short).errors)
    assert auditor.build_auditor(store, KB()).word_range == (150, 200)


# ---------- template ----------

def test_template_passes_for_every_district_and_season(store, kb):
    for season in store.seasons:
        for district in store.districts + ["Punjab"]:
            rec = orchestrator.run_audit(district, season, llm=FakeLLM([LLMError("down")]), store=store, kb=kb, save=False)
            assert rec.mode == "template" and rec.verification["ok"], (district, season, rec.verification["errors"])
            assert rec.citations == ["K000"] and rec.caveat == gap.GAP_CAVEAT and rec.label == audit_template.LABEL


def test_template_gives_no_range_when_there_is_insufficient_evidence(store, kb):
    rec = orchestrator.run_audit("Moga", SEASON, llm=FakeLLM([LLMError("down")]), store=store, kb=kb, save=False)
    assert "insufficient evidence" in rec.briefing and "is estimated at" not in rec.briefing
    assert "200.0%" not in rec.briefing and "from 2022, " not in rec.briefing.split("What other")[0]


# ---------- orchestrator with a scripted LLM ----------

def required_calls():
    wrong = {"district": "Moga", "season": 1999}
    return calls(("get_detection_trend", wrong), ("get_diurnal_profile", wrong), ("get_gap_range", wrong),
                 ("search_knowledge", {"query": "gap method", "before": "2099-01-01", "district": "Moga"}),
                 ("get_burned_area_comparison", wrong))


def run(store, kb, script, tmp_path, **kw):
    llm = FakeLLM(script)
    return orchestrator.run_audit(DISTRICT, SEASON, llm=llm, store=store, kb=kb, root=tmp_path, **kw), llm


def test_auditor_run_is_verified_and_saved_and_the_model_cannot_change_the_filters(store, kb, tmp_path):
    rec, llm = run(store, kb, [required_calls(), final(good_draft(store))], tmp_path)
    assert rec.mode == "agent" and rec.verification["ok"], rec.verification
    assert all(t["args"] == {"district": DISTRICT, "season": SEASON} for t in rec.trace if t["tool"].startswith("get_"))
    assert kb.searches == [{"query": "gap method", "before": "2024-12-01", "district": DISTRICT, "audit_place": DISTRICT}]
    saved = json.loads((tmp_path / "audits" / "2024_Sangrur.json").read_text())
    assert saved["mode"] == "agent" and saved["caveat"] == gap.GAP_CAVEAT and saved["plan"] is None
    again = orchestrator.run_audit(DISTRICT, SEASON, llm=FakeLLM([]), store=store, kb=kb, root=tmp_path, reuse=True)
    assert again.briefing == rec.briefing  # reused without an LLM call


def test_punjab_retrieval_has_no_district_filter():
    assert auditor.injected_args("Punjab", 2025)["search_knowledge"] == {"before": "2025-12-01", "district": None,
                                                                        "place": "Punjab"}


def test_a_missing_label_gets_one_revision(store, kb, tmp_path):
    bad = good_draft(store)
    bad["briefing"] = bad["briefing"].replace("What would settle it:", "Next:")
    rec, llm = run(store, kb, [required_calls(), final(bad), final(good_draft(store))], tmp_path)
    assert rec.mode == "agent_revised" and any("What would settle it" in e for e in rec.first_attempt_errors)
    assert "170 and 190 words" in llm.calls[-1]["messages"][-1]["content"]


def test_two_bad_drafts_fall_back_to_the_template(store, kb, tmp_path):
    bad = {"headline": "x", "briefing": "99999 fires were hidden.", "citations": []}
    rec, _ = run(store, kb, [required_calls(), final(bad), final(bad)], tmp_path)
    assert rec.mode == "template" and rec.verification["ok"] and "99999" not in rec.briefing
    assert rec.first_attempt_errors and rec.revision_errors


PLAN = {"steps": ["Check the burned area first", "Compare the satellites", "Search the gap method"],
        "questions_to_answer": ["Did counts fall more than burned area?"]}


def test_the_plan_reaches_the_first_message_and_the_trace(store, kb, tmp_path):
    rec, llm = run(store, kb, [final(PLAN), required_calls(), final(good_draft(store))], tmp_path, use_planner=True)
    assert rec.mode == "agent" and rec.plan["steps"] == PLAN["steps"] and rec.planner_error is None
    first = llm.calls[1]["messages"][1]["content"]
    assert "Check the burned area first" in first and "audit brief for Sangrur" in first
    planner_message = llm.calls[0]["messages"][1]["content"]
    assert "MODIS burned area" in planner_message and "700" not in planner_message  # names, never values
    assert rec.trace[0]["agent"] == "planner" and rec.trace[0]["args"]["signals"][-1] == "official counts"


def test_a_failing_planner_does_not_block_the_audit(store, kb, tmp_path):
    rec, _ = run(store, kb, [LLMError("quota"), required_calls(), final(good_draft(store))], tmp_path, use_planner=True)
    assert rec.mode == "agent" and rec.plan is None and "quota" in rec.planner_error


@pytest.mark.parametrize("place, expected", [("Sangrur", {"note", "audit_sangrur"}), ("Punjab", {"note", "audit_punjab"}),
                                             (None, {"note", "audit_sangrur", "audit_punjab", "audit_moga"})])
def test_audit_documents_about_other_places_are_not_retrieved(tmp_path, place, expected):
    from psw.rag import index
    from tests.test_rag import FakeEmbedder

    docs = [index.Doc(id=i, title="t", doc_type=t, as_of="2022-01-01", text="burned area gap", district=d)
            for i, t, d in (("note", "curated", None), ("audit_sangrur", "audit", "Sangrur"),
                            ("audit_moga", "audit", "Moga"), ("audit_punjab", "audit", None))]
    # unrelated documents, so BM25 does not score the shared words at zero
    filler = [index.Doc(id=f"x{i}", title="z", doc_type="curated", as_of="2022-01-01", text=f"unrelated words {i}")
              for i in range(8)]
    index.build_index(docs + filler, embedder=FakeEmbedder(), out_dir=tmp_path)
    kb = index.KnowledgeBase(tmp_path, embedder=FakeEmbedder())
    for mode in index.MODES:
        found = {h["doc_id"] for h in kb.search("burned area gap", k=20, mode=mode, audit_place=place)}
        assert {d for d in found if not d.startswith("x")} == expected, mode


@pytest.mark.parametrize("result, said, ok", [
    ({"gap_low_pct": -62.2, "gap_high_pct": 25.7, "confidence": "low", "direction": "direction unclear"}, "", False),
    ({"gap_low_pct": -62.2, "gap_high_pct": 25.7, "confidence": "low", "direction": "direction unclear"},
     "The direction is unclear.", True),
    ({"gap_low_pct": -60.1, "gap_high_pct": -24.4, "confidence": "low", "direction": "no sign of a detection gap"}, "", False),
    ({"gap_low_pct": -60.1, "gap_high_pct": -24.4, "confidence": "low", "direction": "no sign of a detection gap"},
     "There is no sign of a detection gap.", True),
    ({"gap_low_pct": None, "confidence": "insufficient evidence"}, "", False),
    ({"gap_low_pct": None, "confidence": "insufficient evidence"}, "There is insufficient evidence.", True),
])
def test_direction_and_insufficient_evidence_must_be_said(result, said, ok):
    from psw.agents.verifier import _gap_errors

    text = ("Counts per burned km² changed, confidence low. The range does not include the error of the burned-area "
            "product. " + said)
    assert (_gap_errors(text, result) == []) is ok
