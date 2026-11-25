"""Phase 6: pure helpers on synthetic data, plus an AppTest smoke run of every page on the committed data (offline)."""
import json
from datetime import date

import pandas as pd
import pytest

from psw import config
from psw.app import helpers

M = {
    "seasons_test": [2024, 2025],
    "variants": {
        v: {"overall": {"n": 100, "mae": mae, "pinball": pin, "coverage80": cov, "band_width": w},
            "by_city": {"A": {"n": 50, "mae": mae, "pinball": pin, "coverage80": cov, "band_width": w},
                        "B": {"n": 50, "mae": mae + 1, "pinball": pin, "coverage80": cov, "band_width": w}}}
        for v, (mae, pin, cov, w) in {"full": (12.0, 4.0, 0.84, 41.0), "no_fire": (11.5, 3.9, 0.83, 40.0),
                                      "persistence": (11.0, 3.8, 0.77, 34.0)}.items()
    },
    "feature_importance": {"pm25_today": 100.0, "upwind_frp_l0": 50.0, "temp_mean": 20.0},
    "conformal": {"2024": {"full": {"applied": True, "inner_coverage": 0.41, "correction": 10.0}},
                  "2025": {"full": {"applied": True, "inner_coverage": 0.49, "correction": 9.0}}},
}


def test_interpret_negative_result_is_stated():
    text = " ".join(helpers.interpret(M))
    assert "did not improve" in text and "Persistence (tomorrow = today) beat the model" in text
    assert "2024 and 2025" in text and "12.00" in text
    assert "2 cities" in text  # derived from by_city, not typed in


def test_interpret_positive_result_changes_wording():
    m = json.loads(json.dumps(M))
    m["variants"]["full"]["overall"].update(mae=9.0, pinball=3.0)
    text = " ".join(helpers.interpret(m))
    assert "lowered MAE from 11.50 to 9.00" in text and "did not improve" not in text
    assert "The model beat persistence" in text


def test_metrics_and_importance_frames():
    assert list(helpers.metrics_frame(M)["MAE (µg/m³)"]) == [12.0, 11.5, 11.0]
    assert helpers.by_city_frame(M).loc["B", helpers.VARIANT_LABEL["full"]] == 13.0
    imp = helpers.importance_frame(M)
    assert imp.loc[imp["feature"] == "upwind_frp_l0", "fire feature"].item() and not imp.loc[0, "fire feature"]


def test_peak_day_uses_latest_season():
    daily = pd.DataFrame({"date": [date(2024, 11, 1), date(2025, 11, 2), date(2025, 11, 2), date(2025, 11, 3)],
                          "season": [2024, 2025, 2025, 2025], "fire_count": [99, 5, 6, 8]})
    assert helpers.peak_day(daily) == date(2025, 11, 2)
    assert helpers.peak_day(daily, 2024) == date(2024, 11, 1)


def _briefing(**kw):
    rec = {"district": "Sangrur", "date": "2025-11-05", "mode": "agent", "headline": "Sangrur has 12 fires",
           "briefing": "Sangrur logged 12 detections, 3.5 above the 7-day mean [K001]. Forecast p50 is 80.2.",
           "citations": ["K001"], "verification": {"ok": True, "errors": [], "checked_numbers": []},
           "facts": {"get_district_snapshot": {"fire_count": 12, "x": 3.46}, "get_city_forecast": {"p50": 80.2},
                     "search_knowledge": {"results": [{"chunk_id": "K001", "title": "t", "text": "x"},
                                                      {"chunk_id": "K002", "title": "u", "text": "y"}]}}}
    return {**rec, **kw}


def test_explain_numbers_maps_each_number_to_its_tool():
    ex = helpers.explain_numbers(_briefing())
    got = {(r["in"], r["number"]): r["source"] for _, r in ex.iterrows()}
    assert got[("headline", "12")] == "get_district_snapshot"
    assert got[("briefing", "3.5")] == "get_district_snapshot"
    assert got[("briefing", "80.2")] == "get_city_forecast"
    assert got[("briefing", "7")].startswith("constant")  # "7-day"
    assert "NOT FOUND" not in set(ex["source"])


def test_explain_numbers_flags_an_unsupported_number():
    ex = helpers.explain_numbers(_briefing(briefing="Sangrur logged 13 detections [K001]."))
    assert ex.loc[ex["number"] == "13", "source"].item() == "NOT FOUND"


def test_retrieved_chunks_marks_cited():
    ch = helpers.retrieved_chunks(_briefing())
    assert {c["chunk_id"]: c["cited"] for c in ch} == {"K001": True, "K002": False}
    assert helpers.retrieved_chunks({"citations": [], "facts": {}}) == []


def test_badge():
    assert helpers.badge(_briefing())[0] == "success"
    assert helpers.badge(_briefing(mode="template"))[0] == "warning"
    assert helpers.badge(_briefing(verification={"ok": False, "errors": ["x"]}))[0] == "error"
    assert "revision" in helpers.badge(_briefing(mode="agent_revised"))[1]


def test_find_briefing_and_reliability(tmp_path):
    (tmp_path / "briefings").mkdir()
    (tmp_path / "sample_run" / "briefings").mkdir(parents=True)
    from psw.agents.orchestrator import briefing_path
    briefing_path("2025-11-05", "Sangrur", tmp_path).write_text(json.dumps(_briefing()))
    briefing_path("2025-11-06", "Moga", tmp_path / "sample_run").write_text(json.dumps(_briefing(
        district="Moga", date="2025-11-06", mode="template", verification={"ok": True, "errors": []})))
    (tmp_path / "briefings" / "2025-11-07_Bad.json").write_text("{not json")  # must not crash anything

    assert helpers.find_briefing("2025-11-05", "Sangrur", tmp_path)["district"] == "Sangrur"
    assert helpers.find_briefing("2025-11-06", "Moga", tmp_path)["mode"] == "template"  # falls back to sample_run
    assert helpers.find_briefing("2025-11-09", "Moga", tmp_path) is None
    assert helpers.cached_districts("2025-11-06", tmp_path) == {"Moga"}
    rel = helpers.reliability(tmp_path).set_index("set")
    assert rel.loc["demo briefings", "briefings"] == 1 and rel.loc["sampled run", "template"] == 1
    assert helpers.reliability(tmp_path / "nowhere").empty


# ---------- page smoke tests on the committed data ----------

PAGES = ["main.py", "pages/1_Replay_Map.py", "pages/2_Forecast.py", "pages/3_Watch_and_Briefing.py", "pages/4_Evidence.py"]
needs_data = pytest.mark.skipif(not (config.PROCESSED / "metrics.json").exists(), reason="processed data not present")


@needs_data
@pytest.mark.parametrize("page", PAGES)
def test_page_loads_without_error(page, monkeypatch):
    from streamlit.testing.v1 import AppTest
    for k in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL"):  # as if .env were removed
        monkeypatch.setattr(config, k, "")
    at = AppTest.from_file(str(config.ROOT / "src" / "psw" / "app" / page), default_timeout=60).run()
    assert not at.exception, [e.value for e in at.exception]


@needs_data
def test_watch_page_without_llm_and_with_unknown_date(monkeypatch):
    from streamlit.testing.v1 import AppTest
    for k in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL"):
        monkeypatch.setattr(config, k, "")
    at = AppTest.from_file(str(config.ROOT / "src/psw/app/pages/3_Watch_and_Briefing.py"), default_timeout=60).run()
    # a date outside any season: friendly warning, no crash
    at.sidebar.date_input[0].set_value(date(2025, 1, 15)).run()
    assert not at.exception and any("No fire data" in w.value for w in at.warning)
    # a flagged district with no saved briefing: the run button warns instead of crashing
    at.sidebar.date_input[0].set_value(date(2025, 11, 5)).run()
    assert not at.exception
    btn = [b for b in at.button if b.label == "Run agent"]
    if btn:
        btn[0].click().run()
        assert not at.exception and any("No LLM is configured" in w.value for w in at.warning)
