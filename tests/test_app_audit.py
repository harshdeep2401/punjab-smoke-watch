"""The Audit page: its pure helpers on synthetic tables, and the page itself on committed and on incomplete data."""
import json
import shutil

import pandas as pd
import pytest

from psw import config
from psw.app import audit_charts, audit_view, helpers


def tables(burned=True, official=True):
    trend = pd.DataFrame([
        {"district": d, "season": y, "sensor": s, "count": n, "base_season": 2023 if s == "noaa21" else 2022,
         "change_pct": ch}
        for d in ("Punjab", "Moga") for y, s, n, ch in [
            (2022, "snpp", 1000, 0.0), (2025, "snpp", 100, -90.0), (2022, "modis", 200, 0.0), (2025, "modis", 100, -50.0),
            (2023, "noaa21", 900, 0.0), (2025, "noaa21", 90, -90.0)]])
    diurnal = pd.DataFrame([{"district": "Punjab", "sensor": "snpp", "season": y, "pass_label": p, "count": n, "share": 0.5}
                            for y, p, n in [(2022, "afternoon ~13:30", 995), (2022, "night ~01:30", 5),
                                            (2025, "afternoon ~13:30", 96), (2025, "night ~01:30", 4)]])
    timing = pd.DataFrame([{"district": "Punjab", "season": y, "night": n, "total": t, "night_share": n / t,
                            "ci_low": n / t / 2, "ci_high": n / t * 2} for y, n, t in [(2022, 5, 1000), (2025, 4, 100)]])
    out = {"sensor_trend": trend, "diurnal_profile": diurnal, "timing_indicators": timing,
           "burned_vs_detected": None, "official_vs_satellite": None, "gap_range": None}
    if burned:
        out["burned_vs_detected"] = pd.DataFrame([
            {"district": d, "season": y, "burned_km2": km, "detections": n, "burned_change_pct": ch}
            for d in ("Punjab", "Moga") for y, km, n, ch in [(2022, 800.0, 1000, 0.0), (2025, 400.0, 100, -50.0)]])
        out["gap_range"] = pd.DataFrame([
            {"district": d, "season": 2025, "gap_low": lo, "gap_high": hi, "confidence": conf, "direction": way,
             "reason": why, "methods": methods}
            for d, lo, hi, conf, way, why, methods in [
                ("Punjab", 0.77, 0.84, "medium", "counts fell more than burned area", None,
                 "snpp / burned_area, base 2022"),
                ("Moga", None, None, "insufficient evidence", None, "burned area under 50 km²", None)]])
    if official:
        out["official_vs_satellite"] = pd.DataFrame([
            {"district": "Punjab", "season": y, "official": o, "snpp": s, "ratio": o / s, "source_url": "http://x"}
            for y, o, s in [(2022, 1100, 1000), (2025, 120, 100)]])
    return out


def test_all_sensor_change_uses_only_sensors_present_in_both_seasons():
    change, sensors = audit_view.all_sensor_change(tables(), 2025)
    assert change == pytest.approx((200 - 1200) / 1200 * 100)  # NOAA-21 starts in 2023 and is left out
    assert sensors == ["VIIRS S-NPP", "MODIS (Terra and Aqua)"]


def test_headline_follows_the_result():
    row = audit_view.season_headline(tables(), 2025)
    text = audit_view.headline_text(row)
    assert "down 90.0% from 2022" in text and "down 50.0% from 2022" in text
    assert "77% to 84%" in text and "confidence: medium" in text and "does not prove hidden burning" in text
    assert audit_view.gap_text(row) == "77% to 84%"
    flipped = audit_view.headline_text({**row, "gap_low": -0.6, "gap_high": -0.2, "direction": "no sign of a detection gap",
                                        "confidence": "low"})
    assert "no sign of a detection gap" in flipped and "hidden burning" not in flipped


def test_missing_burned_area_and_official_counts_do_not_break_anything():
    t = tables(burned=False, official=False)
    row = audit_view.season_headline(t, 2025)
    assert row["burned_change"] is None and row["gap_low"] is None and audit_view.gap_text(row) == "no range"
    text = audit_view.headline_text(row)
    assert "Burned area is not available" in text and "insufficient evidence" in text
    assert audit_view.official_frame(t).empty and audit_view.indexed_frame(t).empty
    assert audit_view.official_sources(t) == []
    table = audit_view.district_table(t, 2025)
    assert list(table["district"]) == ["Moga"] and table["reading"].iloc[0] == "insufficient evidence"
    assert table["confidence"].iloc[0] == "-" and "no gap estimate" in table["note"].iloc[0]
    assert "0 have a gap range" in audit_view.district_summary(table, 2025)
    empty = {name: None for name in audit_view.TABLES}
    assert audit_view.seasons(empty) == [] and audit_view.pass_time_frame(empty).empty
    assert audit_view.night_share_frame(empty).empty and audit_view.district_table(empty, 2025).empty


def test_indexed_frame_and_district_table():
    ix = audit_view.indexed_frame(tables()).set_index(["series", "season"])["index"]
    assert ix[(audit_view.SNPP_SERIES, 2025)] == 10 and ix[(audit_view.BURNED_SERIES, 2025)] == 50
    table = audit_view.district_table(tables(), 2025)
    assert table["reading"].iloc[0] == "insufficient evidence" and pd.isna(table["gap low (%)"].iloc[0])
    assert table["note"].iloc[0] == "burned area under 50 km²"


def test_figures_have_one_axis_and_fixed_colours_per_signal():
    t = tables()
    light = audit_charts.COLOURS["light"]
    indexed = audit_charts.indexed_figure(audit_view.indexed_frame(t))
    assert [tr.line.color for tr in indexed.data] == [light["snpp"], light["burned"]]
    official = audit_charts.official_figure(audit_view.official_frame(t))
    assert [tr.marker.color for tr in official.data] == [light["snpp"], light["official"]]
    assert "yaxis2" not in official.to_dict()["layout"] and "yaxis2" not in indexed.to_dict()["layout"]
    dark = audit_charts.night_share_figure(audit_view.night_share_frame(t), "dark")
    assert dark.data[0].line.color == audit_charts.COLOURS["dark"]["snpp"]
    assert len(audit_charts.pass_time_figure(audit_view.pass_time_frame(t)).data) == 2  # afternoon and night


def test_explain_numbers_works_for_an_audit_record():
    rec = {"season": 2025, "headline": "Punjab 2025", "briefing": "S-NPP recorded 4038 detections.",
           "facts": {"get_detection_trend": {"detections": 4038}}}
    ex = helpers.explain_numbers(rec)
    assert dict(zip(ex["number"], ex["source"])) == {"2025": "briefing date", "4038": "get_detection_trend"}


# ---------- the page ----------

PAGE = str(config.ROOT / "src/psw/app/pages/5_Audit.py")
needs_data = pytest.mark.skipif(not (config.PROCESSED / "audit" / "gap_range.parquet").exists(),
                                reason="processed data not present")


def _run(monkeypatch):
    from streamlit.testing.v1 import AppTest
    for k in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL"):  # as if .env were removed
        monkeypatch.setattr(config, k, "")
    return AppTest.from_file(PAGE, default_timeout=60).run()


def _text(at):
    return " ".join(x.value for kind in (at.markdown, at.caption, at.info, at.warning, at.subheader) for x in kind)


@needs_data
def test_audit_page_shows_the_saved_brief_and_the_caveats(monkeypatch):
    at = _run(monkeypatch)
    assert not at.exception, [e.value for e in at.exception]
    text = _text(at)
    assert audit_view.CAPTION in [c.value for c in at.caption]
    assert "not the share of burning that is missed" in text and "does not include the error" in text.replace("\n", " ")
    saved = json.loads((config.PROCESSED / "audits" / "2025_Punjab.json").read_text())
    assert saved["headline"] in [s.value for s in at.subheader]
    row = audit_view.season_headline(audit_view.load_audit_tables(), 2025)
    assert audit_view.gap_text(row) in [m.value for m in at.metric]  # the card shows the computed range


@needs_data
def test_audit_page_without_a_saved_brief_or_an_llm(monkeypatch):
    at = _run(monkeypatch)
    at.selectbox(key="brief_place").set_value("Moga").run()
    assert not at.exception and any("No saved audit for Moga" in i.value for i in at.info)
    next(b for b in at.button if b.label == "Run auditor").click().run()
    assert not at.exception and any("No LLM is configured" in w.value for w in at.warning)


@needs_data
def test_audit_page_survives_missing_burned_area_and_official_counts(monkeypatch, tmp_path):
    (tmp_path / "audit").mkdir()
    for name in ("sensor_trend", "diurnal_profile", "timing_indicators"):
        shutil.copy(config.PROCESSED / "audit" / f"{name}.parquet", tmp_path / "audit")
    monkeypatch.setattr(config, "PROCESSED", tmp_path)
    at = _run(monkeypatch)
    assert not at.exception, [e.value for e in at.exception]
    text = _text(at)
    assert "No official counts are available" in text and "Burned area is not available" in text
    assert audit_view.CAPTION in [c.value for c in at.caption]


def test_audit_page_without_any_tables_says_what_to_run(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "PROCESSED", tmp_path)
    at = _run(monkeypatch)
    assert not at.exception and any("No audit tables" in w.value for w in at.warning)


def test_pass_time_panels_share_the_seasons_and_put_snpp_first():
    t = tables()
    t["diurnal_profile"] = pd.concat([t["diurnal_profile"], pd.DataFrame([
        {"district": "Punjab", "sensor": "noaa21", "season": 2025, "pass_label": "afternoon ~13:30", "count": 90, "share": 1.0},
        {"district": "Punjab", "sensor": "modis", "season": 2022, "pass_label": "morning ~10:30", "count": 9, "share": 1.0}])])
    frame = audit_view.pass_time_frame(t)
    assert list(dict.fromkeys(frame["sensor"])) == ["VIIRS S-NPP", "VIIRS NOAA-21", "MODIS (Terra and Aqua)"]
    fig = audit_charts.pass_time_figure(frame)
    assert all(list(trace.x) == ["2022", "2025"] for trace in fig.data)  # NOAA-21 keeps an empty 2022 slot
    assert "too thin to see" in audit_view.night_visibility_note(frame)
    assert audit_view.night_visibility_note(frame[frame["pass_label"] == "night ~01:30"]) is None


@needs_data
def test_a_failed_regeneration_keeps_the_saved_audit(monkeypatch, tmp_path):
    """On a copy of the data: the button must not replace an agent's audit with the template."""
    from psw.llm import LLM, LLMError
    from tests.fakes import FakeLLM

    for folder in ("audit", "audits"):
        shutil.copytree(config.PROCESSED / folder, tmp_path / folder)
    for name in ("kb_chunks.parquet", "kb_vectors.npy"):
        shutil.copy(config.PROCESSED / name, tmp_path)
    monkeypatch.setattr(config, "PROCESSED", tmp_path)
    monkeypatch.setattr(LLM, "from_env", classmethod(lambda cls: FakeLLM([LLMError("rate limited")])))
    saved = tmp_path / "audits" / "2025_Punjab.json"
    before = saved.read_text()

    at = _run(monkeypatch)
    next(b for b in at.button if b.label == "Regenerate with the auditor").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("the saved audit was kept" in w.value and "rate limited" in w.value for w in at.warning)
    assert saved.read_text() == before
