from types import SimpleNamespace as NS

import openai
import pytest

from psw.llm import LLM, LLMError


def _resp(content="hi", tool_calls=None, usage=(10, 5)):
    msg = NS(content=content, tool_calls=tool_calls)
    return NS(choices=[NS(message=msg)], usage=NS(prompt_tokens=usage[0], completion_tokens=usage[1]))


class FakeClient:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.kwargs = []
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **kwargs):
        self.kwargs.append(kwargs)
        o = self.outcomes.pop(0)
        if isinstance(o, Exception):
            raise o
        return o


def _rate_limit():
    req = __import__("httpx").Request("POST", "http://x")
    return openai.RateLimitError("slow down", response=__import__("httpx").Response(429, request=req), body=None)


def _llm(outcomes, **kw):
    sleeps = []
    clock = iter(range(0, 10_000, 1))  # advances 1 s per read
    llm = LLM("http://x", "k", "m", client=FakeClient(outcomes), sleep=sleeps.append, clock=lambda: next(clock), **kw)
    return llm, sleeps


def test_chat_returns_text_and_counts_usage():
    llm, _ = _llm([_resp("hello")], min_interval_s=0)
    r = llm.chat([{"role": "user", "content": "x"}])
    assert r.content == "hello" and llm.usage == {"calls": 1, "prompt_tokens": 10, "completion_tokens": 5}


def test_tool_calls_parsed_and_empty_id_replaced():
    tc = NS(id="", function=NS(name="get_x", arguments='{"a": 1}'))
    llm, _ = _llm([_resp(None, [tc])], min_interval_s=0)
    r = llm.chat([{"role": "user", "content": "x"}])
    assert r.tool_calls[0].id == "call_0" and r.tool_calls[0].name == "get_x"
    assert r.message["tool_calls"][0]["function"]["arguments"] == '{"a": 1}'


def test_gemini_thought_signature_is_kept_and_echoed_back():
    sig = {"extra_content": {"google": {"thought_signature": "abc"}}}
    tc = NS(id="c1", function=NS(name="get_x", arguments="{}"), model_extra={**sig, "other": 1})
    llm, _ = _llm([_resp(None, [tc])], min_interval_s=0)
    r = llm.chat([{"role": "user", "content": "x"}])
    assert r.message["tool_calls"][0]["extra_content"] == sig["extra_content"]
    assert "other" not in r.message["tool_calls"][0]


def test_json_mode_and_tools_are_passed_through():
    llm, _ = _llm([_resp("{}")], min_interval_s=0)
    llm.chat([], tools=[{"type": "function"}], json_mode=True)
    kw = llm._client.kwargs[0]
    assert kw["response_format"] == {"type": "json_object"} and kw["tools"] == [{"type": "function"}]


def test_429_is_retried_with_exponential_backoff():
    llm, sleeps = _llm([_rate_limit(), _rate_limit(), _resp("ok")], min_interval_s=0, backoff_s=5)
    assert llm.chat([]).content == "ok"
    assert [s for s in sleeps if s >= 5] == [5, 10]


def test_gives_up_after_max_retries():
    llm, _ = _llm([_rate_limit()] * 3, min_interval_s=0, max_retries=2)
    with pytest.raises(LLMError):
        llm.chat([])


def test_min_interval_between_calls_is_enforced():
    llm, sleeps = _llm([_resp(), _resp()], min_interval_s=4)
    llm.chat([])
    llm.chat([])  # the fake clock only moved 1 s since the first call, so it must wait about 3 s
    assert sleeps and 2 <= sleeps[-1] <= 4


def test_non_retryable_error_becomes_llm_error_without_retry():
    req = __import__("httpx").Request("POST", "http://x")
    bad = openai.AuthenticationError("bad key", response=__import__("httpx").Response(401, request=req), body=None)
    llm, sleeps = _llm([bad, _resp()], min_interval_s=0)
    with pytest.raises(LLMError):
        llm.chat([])
    assert len(llm._client.kwargs) == 1
