"""Retrieval evaluation: does the right document come back in the top k, for embeddings, BM25 and hybrid?

The eval set is `tests/fixtures/retrieval_eval.json`: a list of {query, expected: [doc_id, ...]} with optional
`before` and `district` (the filters the Analyst's tool passes). A query is a hit at k if any of the top-k chunks
comes from an expected document.
"""
from __future__ import annotations

import json
from pathlib import Path

from psw import config
from psw.rag.index import MODES, KnowledgeBase

EVAL_FILE = config.ROOT / "tests" / "fixtures" / "retrieval_eval.json"
REPORT_FILE = "retrieval_eval.json"


def load_eval(path: Path | None = None) -> list[dict]:
    items = json.loads(Path(path or EVAL_FILE).read_text())
    for it in items:
        if not it.get("query") or not it.get("expected"):
            raise ValueError(f"bad eval item: {it}")
    return items


def evaluate(kb: KnowledgeBase, items: list[dict], k: int = 3, modes: tuple[str, ...] = MODES) -> dict:
    """Per mode: hit@1, hit@k, MRR and the rank of the first expected chunk for each query (None = not in the top k)."""
    report: dict = {"k": k, "n": len(items), "modes": {}}
    for mode in modes:
        rows = []
        for it in items:
            hits = kb.search(it["query"], k=k, before=it.get("before"), district=it.get("district"), mode=mode)
            expected = set(it["expected"])
            rank = next((i for i, h in enumerate(hits, 1) if h["doc_id"] in expected), None)
            rows.append({"query": it["query"], "kind": it.get("kind", ""), "expected": it["expected"], "rank": rank,
                         "got": [h["doc_id"] for h in hits]})
        report["modes"][mode] = {
            "hit_at_k": sum(r["rank"] is not None for r in rows),
            "hit_at_1": sum(r["rank"] == 1 for r in rows),
            "mrr": sum(1 / r["rank"] for r in rows if r["rank"]) / len(rows),  # reciprocal rank, 0 when not in the top k
            "items": rows,
        }
    return report
