"""The one reusable agent loop. Every LLM agent is a prompt plus a tool list run through `run_agent`.

This module knows nothing about districts, PM2.5 or fires. Domain logic lives in the tools and the prompts.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Callable

from psw.llm import ChatResult, dumps


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict  # JSON schema (OpenAI function format)
    fn: Callable[..., dict]  # returns a small JSON-serialisable dict

    def schema(self) -> dict:
        return {"type": "function", "function": {"name": self.name, "description": self.description,
                                                 "parameters": self.parameters}}


@dataclass
class AgentSpec:
    name: str
    system_prompt: str
    tools: list[Tool]
    required_tools: list[str] = field(default_factory=list)  # nudged once if missing from the final answer
    max_turns: int = 8
    json_output: bool = True  # the final answer must parse as a JSON object
    word_range: tuple[int, int] | None = None  # length of the verified text; None = the Verifier's default


@dataclass
class AgentResult:
    output: dict | str  # parsed final answer ({} when the run failed)
    trace: list[dict]  # {step, agent, tool, args, result_summary, ms}
    tool_results: list[dict]  # {tool, args, result}: full results, they feed the Verifier's evidence
    turns: int  # number of LLM calls
    error: str | None = None
    messages: list[dict] = field(default_factory=list)  # whole conversation, so a caller can continue it


NUDGE = "You have not called these required tools yet: {names}. Call them now, then give your final answer."
JSON_RETRY = "Return only the JSON object, nothing else."


def parse_json_object(text: str | None) -> dict | None:
    """Tolerant parse: plain JSON, ```json fences, or a JSON object embedded in prose. None if there is none.

    strict=False accepts real line breaks inside strings, which models write when asked for paragraphs.
    """
    if not text:
        return None
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    try:
        v = json.loads(s, strict=False)
        return v if isinstance(v, dict) else None
    except json.JSONDecodeError:
        pass
    dec = json.JSONDecoder(strict=False)
    for m in re.finditer(r"\{", s):
        try:
            v, _ = dec.raw_decode(s[m.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(v, dict):
            return v
    return None


def _jsonable(obj):
    return json.loads(dumps(obj))


def _summary(result, limit: int = 180) -> str:
    s = dumps(result)
    return s if len(s) <= limit else s[: limit - 1] + "…"


def _call_names(messages: list[dict]) -> set[str]:
    return {tc["function"]["name"] for m in messages for tc in m.get("tool_calls", [])}


def run_agent(spec: AgentSpec, user_message: str, llm, injected_args: dict | None = None,
              history: list[dict] | None = None) -> AgentResult:
    """Run the tool-calling loop.

    injected_args: {tool_name: {arg: value}}, merged over the model's arguments so the model can never control
    them (for example the `before` date and `district` used for retrieval).
    history: the `messages` of an earlier AgentResult, to continue that conversation (used for the one revision).
    """
    injected_args = injected_args or {}
    tools = {t.name: t for t in spec.tools}
    schemas = [t.schema() for t in spec.tools]
    messages = list(history) if history else [{"role": "system", "content": spec.system_prompt}]
    messages.append({"role": "user", "content": user_message})
    called = _call_names(messages)
    trace: list[dict] = []
    results: list[dict] = []
    nudged = json_retried = force_json = False
    turns = 0

    def finish(output, error=None) -> AgentResult:
        return AgentResult(output=output, trace=trace, tool_results=results, turns=turns, error=error, messages=messages)

    for _ in range(spec.max_turns):
        turns += 1
        try:
            res: ChatResult = llm.chat(messages, tools=None if force_json else schemas, json_mode=force_json)
        except Exception as e:  # any API failure: the caller falls back, the loop never crashes
            return finish({}, f"LLM error: {e}")
        messages.append(res.message)

        if res.tool_calls:
            for tc in res.tool_calls:
                result, args, ms = _dispatch(tools, tc, injected_args)
                called.add(tc.name)
                trace.append({"step": len(trace) + 1, "agent": spec.name, "tool": tc.name, "args": args,
                              "result_summary": _summary(result), "ms": ms})
                results.append({"tool": tc.name, "args": args, "result": result})
                messages.append({"role": "tool", "tool_call_id": tc.id, "name": tc.name, "content": dumps(result)})
            continue

        missing = [t for t in spec.required_tools if t not in called]
        if missing:
            if nudged:
                return finish({}, f"required tools not called: {', '.join(missing)}")
            nudged = True
            messages.append({"role": "user", "content": NUDGE.format(names=", ".join(missing))})
            continue

        text = (res.content or "").strip()
        if not spec.json_output:
            return finish(text)
        parsed = parse_json_object(text)
        if parsed is not None:
            return finish(parsed)
        if json_retried:
            return finish({}, "final answer is not valid JSON")
        json_retried = force_json = True
        messages.append({"role": "user", "content": JSON_RETRY})

    return finish({}, f"turn limit ({spec.max_turns}) reached")


def _dispatch(tools: dict[str, Tool], tc, injected_args: dict) -> tuple[dict, dict, int]:
    """Run one tool call. Returns (result, effective_args, ms). Never raises."""
    t0 = time.perf_counter()
    args: dict = {}
    try:
        tool = tools.get(tc.name)
        if tool is None:
            raise ValueError(f"unknown tool '{tc.name}'")
        raw = json.loads(tc.arguments) if tc.arguments and tc.arguments.strip() else {}
        if not isinstance(raw, dict):
            raise ValueError("tool arguments must be a JSON object")
        args = {**raw, **injected_args.get(tc.name, {})}  # injected values win
        result = tool.fn(**args)
    except Exception as e:  # returned to the model so it can correct itself
        result = {"error": f"{type(e).__name__}: {e}"}
    return _jsonable(result), _jsonable(args), int((time.perf_counter() - t0) * 1000)
