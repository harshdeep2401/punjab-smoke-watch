"""Thin OpenAI-compatible chat client (Gemini free tier by default, Ollama as an alternative).

Settings come from .env: LLM_BASE_URL, LLM_API_KEY, LLM_MODEL, LLM_MIN_INTERVAL_S.
It enforces a minimum gap between calls, retries 429 / 5xx / connection errors with exponential backoff,
and logs token usage. The agent loop that uses it lives in `agents/runtime.py`.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field

from psw import config

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """The LLM could not be reached or refused after all retries. Callers fall back to the template."""


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # raw JSON string, exactly as the model sent it
    extra: dict = field(default_factory=dict)  # provider fields to echo back, e.g. Gemini 3's thought_signature


@dataclass
class ChatResult:
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict = field(default_factory=dict)

    @property
    def message(self) -> dict:
        """The assistant message to append to the conversation (OpenAI format)."""
        msg: dict = {"role": "assistant", "content": self.content or ""}
        if self.tool_calls:
            msg["tool_calls"] = [{"id": t.id, "type": "function", "function": {"name": t.name, "arguments": t.arguments}, **t.extra}
                                 for t in self.tool_calls]
        return msg


class LLM:
    def __init__(self, base_url: str | None = None, api_key: str | None = None, model: str | None = None,
                 min_interval_s: float | None = None, max_retries: int = 5, backoff_s: float = 5.0,
                 client=None, sleep=time.sleep, clock=time.monotonic):
        self.base_url = base_url or config.LLM_BASE_URL
        self.api_key = api_key or config.LLM_API_KEY
        self.model = model or config.LLM_MODEL
        self.min_interval_s = config.LLM_MIN_INTERVAL_S if min_interval_s is None else min_interval_s
        self.max_retries = max_retries
        self.backoff_s = backoff_s
        self._client = client
        self._sleep = sleep
        self._clock = clock
        self._last_call: float | None = None
        self.usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}

    @classmethod
    def from_env(cls) -> "LLM":
        """Build from .env. Raises LLMError if the key or model is missing (the caller then uses the template)."""
        if not (config.LLM_API_KEY and config.LLM_MODEL and config.LLM_BASE_URL):
            raise LLMError("LLM_BASE_URL, LLM_API_KEY and LLM_MODEL must be set in .env")
        return cls()

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=90, max_retries=0)  # we retry ourselves
        return self._client

    def _wait_for_slot(self) -> None:
        if self._last_call is not None:
            gap = self.min_interval_s - (self._clock() - self._last_call)
            if gap > 0:
                self._sleep(gap)
        self._last_call = self._clock()

    def chat(self, messages: list[dict], tools: list[dict] | None = None, json_mode: bool = False) -> ChatResult:
        """One chat completion. `tools` is a list of OpenAI function schemas. Raises LLMError after the retries."""
        import openai

        kwargs: dict = {"model": self.model, "messages": messages, "temperature": 0.2}
        if tools:
            kwargs["tools"] = tools
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        retryable = (openai.RateLimitError, openai.APIConnectionError, openai.InternalServerError)
        for attempt in range(self.max_retries + 1):
            self._wait_for_slot()
            try:
                resp = self.client.chat.completions.create(**kwargs)
                break
            except retryable as e:
                if attempt == self.max_retries:
                    raise LLMError(f"{type(e).__name__} after {attempt + 1} attempts") from e
                wait = self.backoff_s * 2 ** attempt
                log.warning("LLM %s (attempt %d); waiting %.0fs", type(e).__name__, attempt + 1, wait)
                self._sleep(wait)
            except openai.OpenAIError as e:  # auth, bad request, unknown model: retrying will not help
                raise LLMError(f"{type(e).__name__}: {str(e)[:200]}") from e
        return self._parse(resp)

    def _parse(self, resp) -> ChatResult:
        if not getattr(resp, "choices", None):
            raise LLMError("empty response (no choices)")
        msg = resp.choices[0].message
        calls = []
        for i, tc in enumerate(msg.tool_calls or []):
            # some OpenAI-compatible servers return an empty tool-call id; the reply needs a non-empty one
            # Gemini 3 models return `extra_content` (a thought signature) that must be sent back with the call
            extra = {k: v for k, v in (getattr(tc, "model_extra", None) or {}).items() if k == "extra_content"}
            calls.append(ToolCall(id=tc.id or f"call_{i}", name=tc.function.name, arguments=tc.function.arguments or "{}", extra=extra))
        u = getattr(resp, "usage", None)
        usage = {"prompt_tokens": getattr(u, "prompt_tokens", 0) or 0, "completion_tokens": getattr(u, "completion_tokens", 0) or 0}
        self.usage["calls"] += 1
        self.usage["prompt_tokens"] += usage["prompt_tokens"]
        self.usage["completion_tokens"] += usage["completion_tokens"]
        log.info("LLM call %d: %d prompt + %d completion tokens", self.usage["calls"], usage["prompt_tokens"], usage["completion_tokens"])
        return ChatResult(content=msg.content, tool_calls=calls, usage=usage)


def dumps(obj) -> str:
    """Compact JSON for tool results sent back to the model."""
    return json.dumps(obj, ensure_ascii=False, default=str)
