"""Scripted stand-ins for the LLM. Tests never touch the network."""
from __future__ import annotations

import json

from psw.llm import ChatResult, ToolCall


def calls(*specs: tuple[str, dict]) -> ChatResult:
    """An assistant turn that calls one or more tools: calls(("get_x", {"a": 1}), ...)."""
    return ChatResult(tool_calls=[ToolCall(id=f"c{i}", name=n, arguments=json.dumps(a)) for i, (n, a) in enumerate(specs)])


def final(content) -> ChatResult:
    """A final assistant message. A dict is sent as JSON."""
    return ChatResult(content=content if isinstance(content, str) else json.dumps(content))


class FakeLLM:
    """Replays a scripted list of ChatResult objects (or Exceptions to raise). Records every call."""

    model = "fake-llm"

    def __init__(self, script: list):
        self.script = list(script)
        self.calls: list[dict] = []

    def chat(self, messages, tools=None, json_mode=False):
        self.calls.append({"messages": [dict(m) for m in messages], "tools": tools, "json_mode": json_mode})
        if not self.script:
            raise AssertionError("FakeLLM script exhausted: the agent made more LLM calls than the test expected")
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeKB:
    """A two-chunk knowledge base: K000 is the limitations note, K001 an ordinary chunk. No embedder, no model download."""

    def __init__(self):
        import pandas as pd
        self.chunks = pd.DataFrame([
            {"chunk_id": "K000", "doc_id": "limitations", "title": "Limitations of this project", "doc_type": "curated",
             "as_of": "2022-01-01", "district": None, "city": None, "source_url": None,
             "text": "Satellites detect heat anomalies and miss fires under cloud."},
            {"chunk_id": "K001", "doc_id": "firms_detection", "title": "How FIRMS detects fires", "doc_type": "curated",
             "as_of": "2022-01-01", "district": None, "city": None, "source_url": None,
             "text": "VIIRS pixels are 375 m wide."},
        ])
        self.searches: list[dict] = []

    def search(self, query, k=4, before=None, district=None):
        self.searches.append({"query": query, "k": k, "before": before, "district": district})
        return [{**self.chunks.iloc[1].to_dict(), "score": 0.9}]
