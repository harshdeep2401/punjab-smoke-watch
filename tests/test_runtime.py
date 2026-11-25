"""run_agent with a scripted FakeLLM. The loop is generic: these tools know nothing about Punjab."""
from psw.agents.runtime import AgentSpec, Tool, parse_json_object, run_agent
from psw.llm import LLMError
from tests.fakes import FakeLLM, calls, final


def make_tools(log):
    def add(a, b=0):
        log.append(("add", a, b))
        return {"sum": a + b}

    def boom():
        raise RuntimeError("tool exploded")

    def search(query, before=None, district=None):
        log.append(("search", query, before, district))
        return {"results": [query]}

    return [
        Tool("add", "add numbers", {"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}}, add),
        Tool("boom", "always fails", {"type": "object", "properties": {}}, boom),
        Tool("search", "search", {"type": "object", "properties": {"query": {"type": "string"}}}, search),
    ]


def spec(log, **kw):
    return AgentSpec(name="t", system_prompt="sys", tools=make_tools(log), **kw)


def test_dispatches_tools_and_returns_parsed_json_with_trace():
    log = []
    llm = FakeLLM([calls(("add", {"a": 1, "b": 2})), final({"answer": 3})])
    r = run_agent(spec(log), "go", llm)
    assert r.output == {"answer": 3} and r.error is None and r.turns == 2
    assert log == [("add", 1, 2)]
    assert r.trace[0]["tool"] == "add" and r.trace[0]["agent"] == "t" and r.trace[0]["step"] == 1
    assert r.tool_results == [{"tool": "add", "args": {"a": 1, "b": 2}, "result": {"sum": 3}}]
    # the tool result went back to the model as a tool message
    last_call_messages = llm.calls[1]["messages"]
    assert last_call_messages[-1]["role"] == "tool" and '"sum": 3' in last_call_messages[-1]["content"]


def test_injected_args_override_what_the_model_sent():
    log = []
    llm = FakeLLM([calls(("search", {"query": "q", "before": "2099-01-01", "district": "Elsewhere"})), final({})])
    run_agent(spec(log), "go", llm, injected_args={"search": {"before": "2025-11-05", "district": "Sangrur"}})
    assert log == [("search", "q", "2025-11-05", "Sangrur")]


def test_missing_required_tool_is_nudged_once_then_loop_continues():
    log = []
    llm = FakeLLM([final({"early": True}), calls(("add", {"a": 1})), final({"ok": True})])
    r = run_agent(spec(log, required_tools=["add"]), "go", llm)
    assert r.output == {"ok": True} and r.error is None
    nudge = [m for m in llm.calls[1]["messages"] if m["role"] == "user"][-1]["content"]
    assert "add" in nudge


def test_required_tool_still_missing_after_nudge_is_an_error():
    llm = FakeLLM([final({"a": 1}), final({"a": 1})])
    r = run_agent(spec([], required_tools=["add"]), "go", llm)
    assert r.error and "add" in r.error and r.output == {}


def test_turn_limit_stops_the_loop():
    llm = FakeLLM([calls(("add", {"a": 1}))] * 10)
    r = run_agent(spec([], max_turns=3), "go", llm)
    assert r.turns == 3 and r.error and "turn limit" in r.error
    assert len(llm.calls) == 3


def test_bad_json_gets_exactly_one_retry_without_tools():
    llm = FakeLLM([final("Sure! here you go"), final({"fixed": 1})])
    r = run_agent(spec([]), "go", llm)
    assert r.output == {"fixed": 1}
    assert llm.calls[1]["tools"] is None and llm.calls[1]["json_mode"] is True
    assert "only the JSON" in llm.calls[1]["messages"][-1]["content"]


def test_bad_json_twice_is_an_error():
    llm = FakeLLM([final("nope"), final("still nope")])
    r = run_agent(spec([]), "go", llm)
    assert r.error == "final answer is not valid JSON" and r.output == {}


def test_tool_exception_does_not_crash_the_loop():
    llm = FakeLLM([calls(("boom", {})), final({"done": True})])
    r = run_agent(spec([]), "go", llm)
    assert r.output == {"done": True} and r.error is None
    assert "tool exploded" in r.tool_results[0]["result"]["error"]


def test_unknown_tool_and_bad_arguments_are_reported_to_the_model():
    llm = FakeLLM([calls(("nope", {}), ("add", {"zzz": 1})), final({})])
    r = run_agent(spec([]), "go", llm)
    assert "unknown tool" in r.tool_results[0]["result"]["error"]
    assert "error" in r.tool_results[1]["result"]  # unexpected keyword argument


def test_llm_error_is_returned_not_raised():
    r = run_agent(spec([]), "go", FakeLLM([LLMError("429 forever")]))
    assert r.error and "429 forever" in r.error and r.output == {}


def test_text_mode_returns_the_raw_string():
    r = run_agent(spec([], json_output=False), "go", FakeLLM([final("plain words")]))
    assert r.output == "plain words"


def test_history_continues_a_conversation_and_skips_the_nudge_for_tools_already_called():
    log = []
    first = run_agent(spec(log, required_tools=["add"]), "go", FakeLLM([calls(("add", {"a": 1})), final({"v": 1})]))
    llm2 = FakeLLM([final({"v": 2})])
    second = run_agent(spec(log, required_tools=["add"]), "fix it", llm2, history=first.messages)
    assert second.output == {"v": 2} and second.error is None
    msgs = llm2.calls[0]["messages"]
    assert msgs[0]["role"] == "system" and msgs[-1] == {"role": "user", "content": "fix it"} and len(msgs) > 4


def test_parse_json_object_is_tolerant():
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_object('Here: {"a": {"b": 2}} thanks') == {"a": {"b": 2}}
    assert parse_json_object("[1, 2]") is None and parse_json_object("") is None and parse_json_object(None) is None
