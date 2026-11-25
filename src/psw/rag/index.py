"""Local vector index: chunking, embedding (fastembed) and cosine search with an `as_of` filter.

No vector DB. Vectors live in a .npy file and chunk metadata in a parquet file.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from psw import config

MODEL_NAME = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
DISTRICT_BOOST = 0.05
VECTORS_FILE = "kb_vectors.npy"
CHUNKS_FILE = "kb_chunks.parquet"


@dataclass(frozen=True)
class Doc:
    id: str
    title: str
    doc_type: str  # curated | season_summary | city_season | model_card
    as_of: str  # ISO date; retrieval only sees docs with as_of < briefing date
    text: str
    district: str | None = None
    city: str | None = None
    source_url: str | None = None


# ---------- embedders: any object with embed_passages(list[str]) and embed_query(str) ----------

class FastEmbedder:
    """bge-small via fastembed (local CPU). The model loads lazily, once. First use downloads ~130 MB."""

    _model = None

    @classmethod
    def _get(cls):
        if cls._model is None:
            from fastembed import TextEmbedding
            cls._model = TextEmbedding(MODEL_NAME)
        return cls._model

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return np.array(list(self._get().embed(texts)), dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return np.array(list(self._get().embed([QUERY_PREFIX + text]))[0], dtype=np.float32)


def _normalise(m: np.ndarray) -> np.ndarray:
    m = np.atleast_2d(m).astype(np.float32)
    n = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.where(n == 0, 1, n)


# ---------- chunking ----------

def _hard_split(text: str, max_chars: int, overlap: int) -> list[str]:
    """Split one over-long paragraph into windows of at most max_chars, overlapping by `overlap`."""
    step = max(1, max_chars - overlap)
    return [text[i:i + max_chars].strip() for i in range(0, len(text), step) if text[i:i + max_chars].strip()]


def chunk(doc_text: str, max_chars: int = 900, overlap: int = 100) -> list[str]:
    """Pack paragraphs (split on blank lines) into chunks of at most max_chars.

    A heading starts a new chunk, so a chunk does not mix two sections. A paragraph longer than
    max_chars is cut into overlapping windows.
    """
    paras = [p.strip() for p in re.split(r"\n\s*\n", doc_text) if p.strip()]
    chunks: list[str] = []
    cur = ""
    for p in paras:
        if len(p) > max_chars:
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.extend(_hard_split(p, max_chars, overlap))
            continue
        starts_section = p.startswith("#")
        if cur and (starts_section or len(cur) + 2 + len(p) > max_chars):
            chunks.append(cur)
            cur = p
        else:
            cur = f"{cur}\n\n{p}" if cur else p
    if cur:
        chunks.append(cur)
    return chunks


# ---------- loading docs from markdown with frontmatter ----------

def parse_markdown(text: str) -> tuple[dict, str]:
    """Split `---` frontmatter (simple `key: value` lines) from the body."""
    m = re.match(r"\A---\n(.*?)\n---\n?(.*)\Z", text, re.S)
    if not m:
        raise ValueError("missing frontmatter")
    meta = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    return meta, m.group(2).strip()


def load_docs(*dirs: Path) -> list[Doc]:
    docs = []
    for d in dirs:
        for f in sorted(Path(d).glob("*.md")):
            meta, body = parse_markdown(f.read_text())
            for req in ("id", "title", "doc_type", "as_of"):
                if req not in meta:
                    raise ValueError(f"{f.name}: frontmatter is missing '{req}'")
            date.fromisoformat(meta["as_of"])  # fail early on a bad date
            docs.append(Doc(id=meta["id"], title=meta["title"], doc_type=meta["doc_type"], as_of=meta["as_of"],
                            text=body, district=meta.get("district"), city=meta.get("city"),
                            source_url=meta.get("source_url")))
    ids = [d.id for d in docs]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate doc ids")
    return sorted(docs, key=lambda d: d.id)  # sorted, so chunk ids are stable across rebuilds


# ---------- build and search ----------

def build_index(docs: list[Doc], embedder=None, out_dir: Path | None = None) -> pd.DataFrame:
    """Chunk and embed every doc; write kb_vectors.npy and kb_chunks.parquet. Returns the chunk table."""
    embedder = embedder or FastEmbedder()
    out_dir = Path(out_dir or config.PROCESSED)
    rows = []
    for d in sorted(docs, key=lambda d: d.id):
        for piece in chunk(d.text):
            rows.append({"chunk_id": f"K{len(rows):03d}", "doc_id": d.id, "title": d.title, "doc_type": d.doc_type,
                         "as_of": d.as_of, "district": d.district, "city": d.city,
                         "source_url": d.source_url, "text": piece})
    table = pd.DataFrame(rows)
    # the title goes into the embedded text so a chunk keeps its context
    vecs = _normalise(embedder.embed_passages([f"{r['title']}. {r['text']}" for r in rows]))
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / VECTORS_FILE, vecs.astype(np.float32))
    table.to_parquet(out_dir / CHUNKS_FILE, index=False)
    return table


class KnowledgeBase:
    def __init__(self, index_dir: Path | None = None, embedder=None):
        d = Path(index_dir or config.PROCESSED)
        self.vectors = np.load(d / VECTORS_FILE)
        self.chunks = pd.read_parquet(d / CHUNKS_FILE)
        self._embedder = embedder
        self._as_of = pd.to_datetime(self.chunks["as_of"]).dt.date.to_numpy()

    @property
    def embedder(self):
        if self._embedder is None:
            self._embedder = FastEmbedder()
        return self._embedder

    def search(self, query: str, k: int = 4, before: date | str | None = None,
               district: str | None = None) -> list[dict]:
        """Cosine top-k. Chunks with as_of >= `before` are excluded. A matching `district` adds +0.05 (a boost, not a filter)."""
        scores = self.vectors @ _normalise(self.embedder.embed_query(query))[0]
        if district:
            scores = scores + DISTRICT_BOOST * (self.chunks["district"] == district).to_numpy()
        mask = np.ones(len(scores), dtype=bool)
        if before is not None:
            cutoff = date.fromisoformat(before) if isinstance(before, str) else before
            mask = np.array([a < cutoff for a in self._as_of])
        idx = [i for i in np.argsort(-scores) if mask[i]][:k]
        out = []
        for i in idx:
            r = self.chunks.iloc[i]
            out.append({**{c: (None if pd.isna(r[c]) else r[c]) for c in self.chunks.columns}, "score": float(scores[i])})
        return out
