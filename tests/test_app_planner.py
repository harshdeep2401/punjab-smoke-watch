"""The Response Planner page: its helpers and the page itself, on the committed data. No network, no LLM."""
import json
import shutil
from datetime import date

import pytest

from psw import config
from psw.agents.dispatch_tools import ResponseStore, Settings, get_allocation, plan_allocation
from psw.app import helpers, planner_charts, planner_view
from psw.app.common import load_geojson

needs_data = pytest.mark.skipif(not (config.PROCESSED / "risk_forecasts.parquet").exists(),
                                reason="processed data not present")
DAY = date(2025, 10, 28)


@pytest.fixture(scope="module")
def store():
    return ResponseStore()


def plan(store, units=500, share=0.25, likely=True, day=DAY):
    return plan_allocation(store, day, Settings(units, share, likely))


@needs_data
def test_default_date_is_three_days_before_the_latest_peak(store):
    import pandas as pd
    daily = pd.read_parquet(config.PROCESSED / "district_daily.parquet")
    assert planner_view.default_date(daily, store.dates) == DAY
    assert planner_view.default_date(daily, [date(2025, 9, 15), date(2025, 9, 16)]) == date(2025, 9, 16)
    assert planner_view.window(DAY) == (date(2025, 10, 29), date(2025, 10, 31))


@needs_data
def test_the_page_and_the_dispatcher_report_the_same_plan(store):
    allocation = plan(store)
    table = planner_view.allocation_table(allocation)
    tool = get_allocation(store, str(DAY))
    assert {r["district"]: r["units"] for r in tool["districts"]} == dict(zip(table["district"].head(8),
                                                                                 table["units"].head(8)))
    risk = planner_view.risk_table(store, allocation, DAY).set_index("district")
    assert list(risk.index) == list(table["district"])
    for r in tool["districts"]:
        assert risk.loc[r["district"], "likely burning"] == r["likely_burning"]
    assert table["units"].sum() == 500 and "on detections only" in table
    assert (table["moved"] == table["units"] - table["on detections only"]).all()


@needs_data
def test_the_plan_follows_the_inputs(store):
    text = " ".join(planner_view.plan_sentences(plan(store)))
    assert "Of 500 units" in text and "Fazilka (57)" in text and "Fazilka 34 to 57" in text
    only = plan(store, likely=False)
    assert "on detections only" not in planner_view.allocation_table(only)
    assert "forecast detections only" in " ".join(planner_view.plan_sentences(only))
    capped = plan(store, units=1000, share=0.05)
    assert capped.table["units"].max() == 50
    more = plan(store, units=2000, share=0.05)
    if more.placed < 2000:
        assert "not placed" in " ".join(planner_view.plan_sentences(more))


@needs_data
def test_a_single_detection_cell_is_never_offered(store):
    seen_singles = False
    for district in plan(store).table["district"]:
        cells, singles = planner_view.cluster_cells(store, district, DAY)
        seen_singles |= singles > 0
        assert all(c["detections_7d"] >= 2 for c in cells) and len(cells) <= 5
        assert len(planner_view.cell_features(store, cells)["features"]) == len(cells)
    assert seen_singles
    lat, lon, zoom = planner_view.district_view(store, "Fazilka", 420, 480)
    assert 29 < lat < 31.5 and 73.5 < lon < 75 and 6.5 < zoom < 9


def test_fit_view_shows_all_of_the_bounds():
    lat, lon, zoom = planner_view.fit_view((74.0, 30.0, 75.0, 31.0), 512, 512, margin=0)
    assert (lat, lon) == (30.5, 74.5)
    assert 512 * 2 ** zoom * (1 / 360) <= 513  # one degree of longitude fits the width
    assert planner_view.fit_view((74.0, 30.0, 76.0, 31.0), 512, 512)[2] < zoom


@needs_data
def test_map_features_and_chart(store):
    allocation = plan(store)
    features, vmax = planner_view.risk_features(load_geojson(), allocation)
    assert len(features["features"]) == 22 and vmax == allocation.table["risk_used"].max()
    assert planner_view.shade(vmax, vmax)[:3] == [140, 35, 20] and planner_view.shade(0, vmax)[:3] == [255, 245, 235]
    table = planner_view.allocation_table(allocation)
    fig = planner_charts.allocation_figure(table, True)
    assert [t.name for t in fig.data] == ["planned on likely burning", "on forecast detections only"]
    assert list(fig.data[0].y)[0] == table["district"].iloc[0] and fig.layout.yaxis.autorange == "reversed"
    assert len(planner_charts.allocation_figure(planner_view.allocation_table(plan(store, likely=False)), False).data) == 1


def _rec(**kw):
    return {"units": 500, "max_share": 0.25, "use_likely": True, "district": None, "mode": "agent", **kw}


def test_a_page_run_never_replaces_a_better_or_different_saved_brief():
    keep = planner_view.keep_dispatch
    assert keep(_rec(mode="template"), None) == "save"
    assert keep(_rec(), _rec()) == "save"
    assert keep(_rec(mode="template"), _rec(mode="agent_revised+critic")) == "kept"
    assert keep(_rec(mode="template"), _rec(mode="template")) == "save"
    assert keep(_rec(units=300), _rec()) == "session"
    assert keep(_rec(district="Moga"), _rec()) == "session"
    assert planner_view.matches(_rec(), 500, 0.25, True) and not planner_view.matches(_rec(), 500, 0.3, True)


def test_explain_numbers_covers_the_district_paragraph():
    rec = {"date": "2025-10-28", "headline": "Plan", "briefing": "57 units.", "district_paragraph": "Rain 99 mm.",
           "facts": {"get_allocation": {"units": 57}}}
    ex = helpers.explain_numbers(rec)
    assert ex.loc[ex["in"] == "district paragraph", "source"].item() == "NOT FOUND"


# ---------- the page ----------

PAGE = str(config.ROOT / "src/psw/app/pages/6_Response_Planner.py")


def _run(monkeypatch):
    from streamlit.testing.v1 import AppTest
    for k in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL"):  # as if .env were removed
        monkeypatch.setattr(config, k, "")
    return AppTest.from_file(PAGE, default_timeout=90).run()


def _text(at):
    return " ".join(x.value for kind in (at.markdown, at.caption, at.info, at.warning, at.subheader) for x in kind)


def _metric(at, label):
    return next(m.value for m in at.metric if m.label == label)


@needs_data
def test_planner_page_shows_the_plan_the_saved_brief_and_the_caption(monkeypatch):
    at = _run(monkeypatch)
    assert not at.exception, [e.value for e in at.exception]
    assert at.caption[0].value == planner_view.CAPTION  # the support-not-enforcement line opens the page
    saved = json.loads((config.PROCESSED / "dispatch" / "2025-10-28.json").read_text())
    assert saved["headline"] in [s.value for s in at.subheader]
    assert _metric(at, "Units placed") == "500" and "written for other inputs" not in _text(at)


@needs_data
def test_planner_page_updates_with_the_inputs_and_flags_a_brief_that_no_longer_matches(monkeypatch):
    at = _run(monkeypatch)
    at.sidebar.number_input[0].set_value(300).run()
    assert not at.exception and _metric(at, "Units placed") == "300" and _metric(at, "Limit per district") == "75"
    assert "written for other inputs" in _text(at)
    at.sidebar.toggle[0].set_value(False).run()
    assert not at.exception and "forecast detections only" in _text(at)


@needs_data
def test_planner_page_on_a_date_without_a_forecast_or_a_brief(monkeypatch):
    at = _run(monkeypatch)
    at.sidebar.date_input[0].set_value(date(2025, 1, 15)).run()
    assert not at.exception and any("No fire-risk forecast was issued" in w.value for w in at.warning)
    at.sidebar.date_input[0].set_value(date(2025, 11, 27)).run()
    assert not at.exception and any("No saved dispatch brief" in i.value for i in at.info)
    next(b for b in at.button if b.label == "Run dispatcher").click().run()
    assert not at.exception and any("No LLM is configured" in w.value for w in at.warning)


@needs_data
def test_a_run_with_other_inputs_leaves_the_saved_brief_alone(monkeypatch, tmp_path):
    """On a copy of the data, with a simulated LLM failure."""
    from psw.llm import LLM, LLMError
    from tests.fakes import FakeLLM

    for name in ("risk_forecasts.parquet", "risk_metrics.json", "district_daily.parquet", "fires.parquet",
                 "district_weather_daily.parquet", "kb_chunks.parquet", "kb_vectors.npy"):
        shutil.copy(config.PROCESSED / name, tmp_path)
    for folder in ("audit", "dispatch"):
        shutil.copytree(config.PROCESSED / folder, tmp_path / folder)
    monkeypatch.setattr(config, "PROCESSED", tmp_path)
    monkeypatch.setattr(LLM, "from_env", classmethod(lambda cls: FakeLLM([LLMError("rate limited")])))
    saved = tmp_path / "dispatch" / "2025-10-28.json"
    before = saved.read_text()
    button = "Run dispatcher for these inputs"

    at = _run(monkeypatch)
    next(b for b in at.button if b.label == button).click().run()  # same inputs: the template does not replace it
    assert not at.exception and any("the saved brief was kept" in w.value for w in at.warning)

    at.sidebar.number_input[0].set_value(300).run()
    next(b for b in at.button if b.label == button).click().run()  # other inputs: shown for this session only
    assert not at.exception, [e.value for e in at.exception]
    assert "Made in this session" in _text(at) and "Of 300 units" in _text(at).replace("of 300 units", "Of 300 units")
    assert saved.read_text() == before
