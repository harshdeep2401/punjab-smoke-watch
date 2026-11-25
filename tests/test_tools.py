"""Tools, Sentinel and the template fallback, run against the committed processed data (offline, no LLM)."""
import json

import pytest

from psw import config
from psw.agents import sentinel, template, tools
from psw.agents.tools import DataStore
from psw.agents.verifier import Evidence, verify
from psw.rag.index import KnowledgeBase

DAY = "2025-10-31"  # the busiest day of 2025 across Punjab


@pytest.fixture(scope="module")
def store():
    return DataStore()


@pytest.fixture(scope="module")
def kb():
    return KnowledgeBase()  # only .chunks is used by the template path, so no embedder is loaded


class FakeKB:
    """Records search arguments and returns one fixed chunk."""
    def __init__(self):
        self.calls = []

    def search(self, query, k=4, before=None, district=None):
        self.calls.append({"query": query, "k": k, "before": before, "district": district})
        return [{"chunk_id": "K001", "title": "T", "text": "x", "score": 0.9}] * 2


def test_snapshot_matches_the_table_and_is_json_serialisable(store):
    r = tools.get_district_snapshot(store, "Sangrur", DAY)
    row = store.district_daily.query("district == 'Sangrur' and date == @DAY").iloc[0]
    assert r["fire_count"] == row["fire_count"] and r["district"] == "Sangrur" and r["data_as_of"] == DAY
    assert r["rank_in_punjab"] >= 1 and r["punjab_total"] == store.district_daily.query("date == @DAY")["fire_count"].sum()
    assert [d["date"] for d in r["last_7_days"]][-1] == DAY and len(r["last_7_days"]) == 7
    assert r["nearest_city"] in config.CITIES
    json.dumps(r)


def test_snapshot_floats_are_rounded_to_one_decimal(store):
    r = tools.get_district_snapshot(store, "Sangrur", DAY)
    for k in ("frp_sum_mw", "fire_count_7d_mean", "change_vs_7d_pct"):
        assert r[k] is None or round(r[k], 1) == r[k]


def test_snapshot_unknown_district_returns_error_dict(store):
    assert "error" in tools.get_district_snapshot(store, "Atlantis", DAY)


def test_forecast_never_leaks_the_actual(store):
    r = tools.get_city_forecast(store, "Bathinda", DAY)
    assert r["issue_date"] == DAY and r["p10"] <= r["p50"] <= r["p90"]
    assert r["target_date"] == "2025-11-01"
    assert "actual" not in r and "actual" not in json.dumps(r)
    assert r["exceeds_naaqs_60_prob_hint"].split()[0] in {"p50", "p90"}


def test_forecast_outside_the_test_seasons_is_an_error_dict(store):
    assert "error" in tools.get_city_forecast(store, "Bathinda", "2023-10-31")


def test_wind_tool(store):
    r = tools.get_wind_and_upwind(store, "Ludhiana", DAY)
    assert 0 <= r["wind_dir_from_deg"] < 360 and r["wind_dir_compass"] in tools.COMPASS and r["wind_speed_kmh"] >= 0
    assert all(set(d) == {"district", "share_pct"} for d in r["top_upwind_districts"])
    assert sum(d["share_pct"] for d in r["top_upwind_districts"]) <= 100.3


def test_compass():
    assert [tools.compass(x) for x in (0, 90, 180, 270, 315, 359)] == ["N", "E", "S", "W", "NW", "N"]


def test_model_reliability(store):
    r = tools.get_model_reliability(store, "Ludhiana")
    assert r["test_seasons"] == [2024, 2025] and 0 < r["coverage_80_pct"] <= 100 and r["mae"] > 0
    assert "error" in tools.get_model_reliability(store, "Nowhere")


def test_nearest_city(store):
    assert tools.nearest_city("Sangrur", store) in config.CITIES
    assert tools.nearest_city("Pathankot", store) == "Amritsar"
    with pytest.raises(ValueError):
        tools.nearest_city("Atlantis", store)


def test_search_knowledge_passes_injected_filters_and_clamps_k():
    kb = FakeKB()
    out = tools.search_knowledge(kb, "q", k=99, before="2025-11-05", district="Sangrur")
    assert kb.calls == [{"query": "q", "k": 6, "before": "2025-11-05", "district": "Sangrur"}]
    assert out["results"][0] == {"chunk_id": "K001", "title": "T", "text": "x"}


def test_schemas_match_the_tools_and_hide_the_injected_arguments(store):
    names = [s["function"]["name"] for s in tools.TOOL_SCHEMAS]
    assert names == ["get_district_snapshot", "get_city_forecast", "get_wind_and_upwind", "get_model_reliability", "search_knowledge"]
    search = tools.TOOL_SCHEMAS[-1]["function"]["parameters"]["properties"]
    assert set(search) == {"query", "k"}  # `before` and `district` are not the model's to choose
    built = tools.build_tools(store, FakeKB())
    assert [t.name for t in built] == names
    out = built[-1].fn(query="q", before="2025-11-05", district="Sangrur")
    assert out["results"]


def test_limitations_chunk_needs_no_embedder(kb):
    r = tools.limitations_chunk(kb)
    assert len(r["results"]) == 1 and r["results"][0]["chunk_id"].startswith("K")


# ---------- Sentinel ----------

def test_sentinel_scans_every_district_and_flags_first(store):
    rows = sentinel.scan(DAY, store)
    assert len(rows) == 22 and {r["district"] for r in rows} == set(store.districts)
    flagged = [bool(r["flags"]) for r in rows]
    assert flagged == sorted(flagged, reverse=True)  # flagged rows come first
    top = rows[0]
    assert top["flags"] and all(isinstance(f, str) for f in top["flags"])
    scores = [r["priority_score"] for r in rows if r["flags"]]
    assert scores == sorted(scores)


def test_sentinel_flag_rules(store):
    rows = {r["district"]: r for r in sentinel.scan(DAY, store)}
    for r in rows.values():
        spike = r["fire_count"] >= max(config.SENTINEL_SPIKE_MIN, config.SENTINEL_SPIKE_RATIO * r["fire_count_7d_mean"])
        assert spike == any(f.startswith("spike") for f in r["flags"]), r
        smog = r["nearest_city_p90"] is not None and r["nearest_city_p90"] > config.SENTINEL_SMOG_P90
        assert smog == any(f.startswith("smog_risk") for f in r["flags"]), r
        if r["fire_count"] < config.SENTINEL_HIGH_COUNT_MIN:
            assert not any(f.startswith("high_count") for f in r["flags"])


def test_sentinel_works_on_days_without_a_forecast(store):
    rows = sentinel.scan("2023-10-31", store)
    assert len(rows) == 22 and all(r["nearest_city_p90"] is None for r in rows)
    assert not any(f.startswith("smog_risk") for r in rows for f in r["flags"])


def test_sentinel_unknown_date(store):
    with pytest.raises(ValueError):
        sentinel.scan("2030-01-01", store)


# ---------- Template: must always pass the verifier ----------

def facts_for(store, kb, district, day):
    snap = tools.get_district_snapshot(store, district, day)
    city = snap["nearest_city"]
    return {"snapshot": snap, "forecast": tools.get_city_forecast(store, city, day),
            "wind": tools.get_wind_and_upwind(store, city, day), "reliability": tools.get_model_reliability(store, city),
            "limitations": tools.limitations_chunk(kb)}


def check(store, kb, district, day):
    f = facts_for(store, kb, district, day)
    out = template.render(f)
    ev = Evidence.from_tool_results([{"tool": k, "args": {}, "result": v} for k, v in f.items()], dates=[day])
    r = verify(out, ev)
    assert r.ok, (district, day, r.errors, out["briefing"])
    return out


def test_template_passes_verify_on_every_day_of_the_latest_season(store, kb):
    days = sorted(store.district_daily.query("season == 2025")["date"].unique())
    for district in ("Sangrur", "Pathankot", "Amritsar", "Fazilka"):
        for day in days:
            check(store, kb, district, day)


def test_template_passes_verify_when_forecast_and_wind_are_missing(store, kb):
    for day in ("2023-09-15", "2023-10-31", "2022-11-30"):
        for district in ("Sangrur", "Pathankot"):
            out = check(store, kb, district, day)
            assert 100 <= len(out["briefing"].split()) <= 180


def test_template_without_any_optional_data_still_passes(store, kb):
    f = facts_for(store, kb, "Sangrur", DAY)
    f.update(forecast={"error": "x"}, wind={"error": "x"}, reliability={"error": "x"})
    out = template.render(f)
    ev = Evidence.from_tool_results([{"tool": k, "args": {}, "result": v} for k, v in f.items()], dates=[DAY])
    assert verify(out, ev).ok and out["label"] == template.LABEL


def test_template_states_the_model_vs_persistence_result_as_it_is(store, kb):
    out = check(store, kb, "Sangrur", DAY)
    rel = tools.get_model_reliability(store, tools.nearest_city("Sangrur", store))
    expected = "so it beat that simple baseline" if rel["mae"] < rel["mae_persistence"] else "so it did not beat that simple baseline"
    assert expected in out["briefing"]
