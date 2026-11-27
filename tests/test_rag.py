import hashlib
import json
import re
from datetime import date

import numpy as np
import pandas as pd
import pytest

from psw.rag import build_kb, index
from psw.rag.index import Doc, KnowledgeBase


class FakeEmbedder:
    """Bag-of-words hash vectors: no model download, texts that share words are similar."""
    DIM = 256

    def _vec(self, text):
        v = np.zeros(self.DIM, dtype=np.float32)
        for w in re.findall(r"[a-z0-9]+", text.lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.DIM] += 1
        return v

    def embed_passages(self, texts):
        return np.array([self._vec(t) for t in texts])

    def embed_query(self, text):
        return self._vec(text)


def doc(id_, text, as_of="2022-01-01", district=None, title=None):
    return Doc(id=id_, title=title or id_, doc_type="curated", as_of=as_of, text=text, district=district)


# ---------- chunking ----------

def test_chunk_respects_max_chars_and_keeps_all_text():
    paras = [f"Paragraph {i} " + "word " * 40 for i in range(12)]
    chunks = index.chunk("\n\n".join(paras), max_chars=500)
    assert len(chunks) > 1
    assert all(len(c) <= 500 for c in chunks)
    assert all(f"Paragraph {i} " in "\n\n".join(chunks) for i in range(12))


def test_chunk_splits_long_paragraph_with_overlap():
    text = "x" * 2000
    chunks = index.chunk(text, max_chars=900, overlap=100)
    assert all(len(c) <= 900 for c in chunks)
    assert len(chunks) == 3  # windows start at 0, 800, 1600
    assert sum(len(c) for c in chunks) == 2000 + 2 * 100  # two overlaps


def test_chunk_starts_new_chunk_at_heading():
    chunks = index.chunk("# A\n\nshort one\n\n# B\n\nshort two", max_chars=900)
    assert len(chunks) == 2 and chunks[1].startswith("# B")


def test_chunk_empty():
    assert index.chunk("  \n\n ") == []


# ---------- frontmatter ----------

def test_parse_markdown_and_load_docs(tmp_path):
    (tmp_path / "a.md").write_text("---\nid: a\ntitle: Title A\ndoc_type: curated\nas_of: 2022-01-01\n"
                                   "district: Sangrur\n---\n\nBody text.\n")
    (d,) = index.load_docs(tmp_path)
    assert (d.id, d.title, d.district, d.text) == ("a", "Title A", "Sangrur", "Body text.")


def test_load_docs_rejects_missing_field_and_duplicates(tmp_path):
    (tmp_path / "a.md").write_text("---\nid: a\ntitle: T\ndoc_type: curated\n---\nbody")
    with pytest.raises(ValueError, match="as_of"):
        index.load_docs(tmp_path)
    (tmp_path / "a.md").write_text("---\nid: a\ntitle: T\ndoc_type: curated\nas_of: 2022-01-01\n---\nbody")
    (tmp_path / "b.md").write_text("---\nid: a\ntitle: T\ndoc_type: curated\nas_of: 2022-01-01\n---\nbody")
    with pytest.raises(ValueError, match="duplicate"):
        index.load_docs(tmp_path)


# ---------- search ----------

@pytest.fixture
def kb(tmp_path):
    docs = [
        doc("firms", "Satellites miss fires because of cloud and overpass timing."),
        doc("grap", "Graded response action plan stages by air quality index band."),
        doc("season_2024_Sangrur", "Sangrur fire detections 2024 season.", as_of="2024-12-01", district="Sangrur"),
        doc("season_2025_Sangrur", "Sangrur fire detections 2025 season.", as_of="2025-12-01", district="Sangrur"),
        doc("season_2024_Moga", "Moga fire detections 2024 season.", as_of="2024-12-01", district="Moga"),
    ]
    index.build_index(docs, embedder=FakeEmbedder(), out_dir=tmp_path)
    return KnowledgeBase(tmp_path, embedder=FakeEmbedder())


def test_search_ranks_relevant_doc_first(kb):
    assert kb.search("why do satellites miss fires", k=2)[0]["doc_id"] == "firms"


def test_as_of_filter_excludes_future_docs(kb):
    res = kb.search("Sangrur fire detections season", k=10, before=date(2025, 11, 5))
    ids = {r["doc_id"] for r in res}
    assert "season_2025_Sangrur" not in ids and "season_2024_Sangrur" in ids
    assert all(r["as_of"] < "2025-11-05" for r in res)


def test_as_of_is_strict_and_accepts_iso_string(kb):
    ids = {r["doc_id"] for r in kb.search("Sangrur 2024", k=10, before="2024-12-01")}
    assert "season_2024_Sangrur" not in ids  # as_of == before is excluded


def test_no_before_returns_everything(kb):
    assert len(kb.search("fire", k=10)) == 5


def test_district_boost_reorders_but_does_not_filter(kb):
    plain = kb.search("fire detections 2024 season", k=5, mode="embedding")
    boosted = kb.search("fire detections 2024 season", k=5, district="Moga", mode="embedding")
    assert len(boosted) == len(plain) == 5  # nothing excluded
    score = {r["doc_id"]: r["score"] for r in plain}
    bscore = {r["doc_id"]: r["score"] for r in boosted}
    assert bscore["season_2024_Moga"] == pytest.approx(score["season_2024_Moga"] + index.DISTRICT_BOOST)
    assert bscore["firms"] == pytest.approx(score["firms"])  # curated docs are not penalised


# ---------- stable ids ----------

def test_chunk_ids_stable_across_rebuilds_and_input_order(tmp_path):
    docs = [doc("b", "Beta text. " * 150), doc("a", "Alpha text."), doc("c", "Gamma text.")]
    t1 = index.build_index(docs, embedder=FakeEmbedder(), out_dir=tmp_path / "1")
    t2 = index.build_index(list(reversed(docs)), embedder=FakeEmbedder(), out_dir=tmp_path / "2")
    assert t1["chunk_id"].tolist() == t2["chunk_id"].tolist()
    assert t1["doc_id"].tolist() == t2["doc_id"].tolist()
    assert t1["doc_id"].tolist()[0] == "a" and t1["doc_id"].tolist()[-1] == "c"
    assert t1["chunk_id"].tolist() == [f"K{i:03d}" for i in range(len(t1))]
    np.testing.assert_array_equal(np.load(tmp_path / "1" / index.VECTORS_FILE), np.load(tmp_path / "2" / index.VECTORS_FILE))


# ---------- generated docs ----------

@pytest.fixture
def processed(tmp_path):
    rows = []
    for season, counts in {2023: {"A": [5, 0], "B": [1, 2]}, 2024: {"A": [1, 1], "B": [4, 0]}}.items():
        for dist, c in counts.items():
            for i, n in enumerate(c):
                rows.append({"date": date(season, 10, 1 + i), "season": season, "district": dist,
                             "fire_count": n, "frp_sum": n * 10.0, "fire_count_7d_mean": 0.0})
    pd.DataFrame(rows).to_parquet(tmp_path / "district_daily.parquet")
    pm = [{"date": date(2024, 10, 1 + i), "city": "Ludhiana", "pm25_mean": v} for i, v in enumerate([50.0, 70.0, 90.0])]
    pm += [{"date": date(2023, 10, 1), "city": "Ludhiana", "pm25_mean": 40.0}]
    pm += [{"date": date(2024, 8, 1), "city": "Ludhiana", "pm25_mean": 500.0}]  # outside the season window
    pd.DataFrame(pm).to_parquet(tmp_path / "pm25_daily.parquet")
    v = lambda mae: {"overall": {"n": 10, "mae": mae, "pinball": mae / 3, "coverage80": 0.8, "band_width": 30.0}}
    metrics = {"seasons_test": [2024], "folds": {"2024": [2023]}, "conformal_applied": True,
               "variants": {"full": v(12.0), "no_fire": v(11.0), "persistence": v(10.0)},
               "feature_importance": {"pm25_today": 5.0, "temp_mean": 2.0}}
    (tmp_path / "metrics.json").write_text(json.dumps(metrics))
    return tmp_path


def test_season_summary_numbers_come_from_data(processed):
    daily = pd.read_parquet(processed / "district_daily.parquet").assign(date=lambda d: d["date"].astype(str))
    doc_id, md = build_kb.season_summary_doc("A", 2024, daily)
    assert doc_id == "season_2024_A"
    assert "as_of: 2024-12-01" in md and "district: A" in md
    assert "2 fire detections" in md and "20 MW" in md  # 1 + 1 detections, FRP 10 + 10
    assert "ranked number 2" in md  # B had 4
    assert "-60%" in md and "5 detections" in md  # 2 vs 5 in 2023
    _, first = build_kb.season_summary_doc("A", 2023, daily)
    assert "no earlier season" in first


def test_city_season_ignores_days_outside_window(processed):
    pm = pd.read_parquet(processed / "pm25_daily.parquet").assign(date=lambda d: d["date"].astype(str))
    _, md = build_kb.city_season_doc("Ludhiana", 2024, pm)
    assert "averaged 70.0" in md and "maximum daily mean of 90.0 µg/m³ on 2024-10-03" in md
    assert "2 of 3 days were above 60" in md


def test_model_card_states_negative_result_and_start_date(processed):
    doc_id, md = build_kb.model_card_doc(json.loads((processed / "metrics.json").read_text()))
    assert "as_of: 2024-09-15" in md
    assert "did not improve" in md and "did not beat" in md
    assert "12.00 vs 11.00" in md


def test_generate_docs_roundtrip_through_load_docs(processed, tmp_path):
    out = tmp_path / "gen"
    paths = build_kb.generate_docs(processed, out)
    assert len(paths) == 4 + 2 + 1  # 2 districts x 2 seasons, 1 city x 2 seasons, model card
    docs = index.load_docs(out)
    assert {d.doc_type for d in docs} == {"season_summary", "city_season", "model_card"}
    assert next(d for d in docs if d.id == "season_2024_A").district == "A"
    # regenerating replaces old files instead of piling up
    (out / "stale.md").write_text("old")
    assert len(build_kb.generate_docs(processed, out)) == 7 and not (out / "stale.md").exists()


# ---------- hybrid retrieval: BM25 + embeddings + reciprocal rank fusion ----------

MODES = index.MODES


def test_tokenize_keeps_years_and_numbers_whole():
    assert index.tokenize("Moga 2024: p90 > 60 µg/m3, FRP.") == ["moga", "2024", "p90", "60", "g", "m3", "frp"]


def test_rrf_scores_and_order():
    fused = index.rrf([[1, 2], [2, 3]], k=60)
    assert fused[1] == pytest.approx(1 / 61)
    assert fused[2] == pytest.approx(1 / 62 + 1 / 61)  # ranked 2nd in one list and 1st in the other
    assert fused[3] == pytest.approx(1 / 62)
    assert max(fused, key=fused.get) == 2
    assert index.rrf([]) == {} and index.rrf([[]]) == {}


class FlatEmbedder:
    """Every text gets the same vector, so embedding search cannot tell chunks apart. Only keywords can."""
    def embed_passages(self, texts):
        return np.ones((len(texts), 8), dtype=np.float32)

    def embed_query(self, text):
        return np.ones(8, dtype=np.float32)


@pytest.fixture
def flat_kb(tmp_path):
    docs = [doc(f"d{i}", f"Ordinary text number {i} about weather and wind.") for i in range(5)]
    docs.append(doc("rare", "A note about the zebrafish experiment."))
    docs.append(doc("rare_future", "The zebrafish experiment results.", as_of="2026-01-01"))
    index.build_index(docs, embedder=FlatEmbedder(), out_dir=tmp_path)
    return KnowledgeBase(tmp_path, embedder=FlatEmbedder())


def test_bm25_finds_an_exact_keyword_that_embeddings_cannot(flat_kb):
    assert flat_kb.search("zebrafish", k=1, mode="bm25")[0]["doc_id"] in ("rare", "rare_future")
    emb = [r["doc_id"] for r in flat_kb.search("zebrafish", k=3, mode="embedding")]
    assert "rare" not in emb  # all vectors tie, so the embedding ranking is just the index order
    assert flat_kb.search("zebrafish", k=1, before="2025-01-01", mode="hybrid")[0]["doc_id"] == "rare"


def test_bm25_returns_nothing_for_words_not_in_the_corpus(flat_kb):
    assert flat_kb.search("xylophone", k=5, mode="bm25") == []
    assert len(flat_kb.search("xylophone", k=5, mode="hybrid")) == 5  # hybrid falls back to the embedding ranking


@pytest.mark.parametrize("mode", MODES)
def test_as_of_filter_holds_in_every_mode(flat_kb, mode):
    res = flat_kb.search("zebrafish experiment", k=10, before="2025-12-31", mode=mode)
    assert res and "rare_future" not in {r["doc_id"] for r in res}
    assert all(r["as_of"] < "2025-12-31" for r in res)


@pytest.mark.parametrize("mode", MODES)
def test_every_mode_returns_k_results_with_scores_and_columns(kb, mode):
    res = kb.search("Sangrur fire detections 2024 season", k=3, mode=mode)
    assert 1 <= len(res) <= 3
    assert all("score" in r and "chunk_id" in r and "doc_id" in r for r in res)
    assert [r["score"] for r in res] == sorted((r["score"] for r in res), reverse=True)


def test_hybrid_is_the_default_and_unknown_modes_fail(kb):
    assert index.DEFAULT_MODE == "hybrid"
    assert kb.search("satellites miss fires", k=3) == kb.search("satellites miss fires", k=3, mode="hybrid")
    with pytest.raises(ValueError, match="mode must be one of"):
        kb.search("x", mode="dense")


def test_hybrid_score_is_a_fused_rank_score(kb):
    top = kb.search("satellites miss fires cloud", k=1, mode="hybrid")[0]
    assert top["score"] <= 2 / (index.RRF_K + 1) + 1e-9  # at most rank 1 in both lists


# ---------- retrieval evaluation ----------

def test_evaluate_counts_hits_and_ranks(kb):
    from psw.rag import evaluate
    items = [{"query": "why do satellites miss fires", "expected": ["firms"]},
             {"query": "Moga fire detections 2024 season", "expected": ["season_2024_Moga"], "before": "2025-01-01"},
             {"query": "graded response action plan", "expected": ["season_2024_Moga"]}]  # deliberately wrong
    rep = evaluate.evaluate(kb, items, k=3)
    assert rep["n"] == 3 and rep["k"] == 3 and set(rep["modes"]) == set(MODES)
    for mode in MODES:
        m = rep["modes"][mode]
        assert m["items"][0]["rank"] == 1 and m["items"][2]["rank"] != 1
        assert m["hit_at_1"] >= 1 and m["hit_at_k"] >= m["hit_at_1"]
        assert 0 < m["mrr"] <= 1 and m["mrr"] == pytest.approx(sum(1 / r["rank"] for r in m["items"] if r["rank"]) / 3)


def test_load_eval_rejects_bad_items(tmp_path):
    from psw.rag import evaluate
    (tmp_path / "e.json").write_text('[{"query": "x", "expected": []}]')
    with pytest.raises(ValueError, match="bad eval item"):
        evaluate.load_eval(tmp_path / "e.json")


def test_committed_eval_set_is_valid_against_the_committed_index():
    from psw import config
    from psw.rag import evaluate
    chunks_path = config.PROCESSED / index.CHUNKS_FILE
    if not chunks_path.exists():
        pytest.skip("index not built")
    docs = set(pd.read_parquet(chunks_path)["doc_id"])
    items = evaluate.load_eval()
    assert len(items) >= 15
    for it in items:
        assert set(it["expected"]) <= docs, it
        if it.get("before"):
            date.fromisoformat(it["before"])
    assert {it.get("kind") for it in items} >= {"meaning", "keyword"}


# ---------- Part 2: audit documents ----------

def audit_inputs():
    """Two districts and three seasons: enough for a range, an insufficient district and a base season."""
    import pandas as pd

    from tests.test_audit_gap import area, fires, official

    ms, ba, off = {}, {}, {}
    for season, n, night in ((2022, 1000, 2), (2023, 800, 2), (2024, 100, 40)):
        for district, km2 in (("Sangrur", 1000 if season < 2024 else 700), ("Moga", 20)):
            ms[(district, "snpp", season)] = (n - night, night)
            ms[(district, "modis", season)] = (100 if season < 2024 else 40, 0)
            ba[(district, season)] = km2
            off[(district, season)] = n
    return fires(ms), area(ba), official(off)


@pytest.fixture
def audit_processed(tmp_path):
    from psw.audit import gap

    folder = tmp_path / "audit"
    folder.mkdir()
    for name, table in gap.audit_tables(*audit_inputs()).items():
        table.to_parquet(folder / f"{name}.parquet", index=False)
    return tmp_path


def audit_text(processed, doc_id):
    return dict(build_kb.audit_docs(processed))[doc_id]


def test_audit_docs_cover_each_district_season_and_punjab(audit_processed):
    docs = dict(build_kb.audit_docs(audit_processed))
    assert len(docs) == 9 and {"audit_2024_punjab", "audit_2024_Sangrur", "audit_2022_Moga"} <= set(docs)
    meta, body = index.parse_markdown(docs["audit_2024_Sangrur"])
    assert meta["as_of"] == "2024-12-01" and meta["part"] == "2" and meta["district"] == "Sangrur"
    assert meta["doc_type"] == "audit" and "100 detections (down 90.0% from 2022)" in body
    assert "district" not in index.parse_markdown(docs["audit_2024_punjab"])[0]
    assert "Timing indicators" in docs["audit_2024_punjab"] and "Timing indicators" not in docs["audit_2024_Sangrur"]


def test_audit_docs_state_the_base_season_and_insufficient_evidence_without_a_range(audit_processed):
    assert "base season and has no gap" in audit_text(audit_processed, "audit_2022_Sangrur")
    moga = audit_text(audit_processed, "audit_2024_Moga")
    assert "Insufficient evidence" in moga and "under 50 km²" in moga and "The gap (the fall" not in moga


def test_every_chunk_with_a_gap_range_carries_the_caveat(audit_processed):
    from psw.audit import gap

    ranges = 0
    for _, text in build_kb.audit_docs(audit_processed):
        for piece in index.chunk(index.parse_markdown(text)[1]):
            assert len(piece) > 60  # no title-only chunks
            assert "burned_area" not in piece
            if "The gap (the fall" in piece:
                ranges += 1
                assert gap.GAP_CAVEAT in piece and "confidence" in piece
    assert ranges > 0


def test_no_audit_tables_means_no_audit_docs(tmp_path):
    assert build_kb.audit_docs(tmp_path) == []


def test_later_parts_sort_after_earlier_ones_so_chunk_ids_do_not_move(tmp_path):
    old = [index.Doc(id=i, title=i, doc_type="curated", as_of="2022-01-01", text=f"text of {i}") for i in ("method", "season")]
    new = index.Doc(id="audit_x", title="a", doc_type="audit", as_of="2022-12-01", text="audit text", part=2)
    before = index.build_index(old, embedder=FakeEmbedder(), out_dir=tmp_path / "a")
    after = index.build_index([new, *old], embedder=FakeEmbedder(), out_dir=tmp_path / "b")
    assert list(after["doc_id"]) == ["method", "season", "audit_x"]
    assert after.iloc[:2][["chunk_id", "doc_id"]].equals(before[["chunk_id", "doc_id"]])

