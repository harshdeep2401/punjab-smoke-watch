"""The Planner: an `AgentSpec` with no tools, run through `run_agent`. It writes a short plan before the Analyst starts.

It sees only the district, the date, the nearest city and the *kinds* of Sentinel flag (spike, high_count, smog_risk), never
a number. Its plan goes into the Analyst's first message and is shown at the top of the Agent trace tab.

The plan is guidance, not evidence: the Analyst's numbers still come only from tools and the Verifier still checks the final
text. A plan that is missing, invalid or empty never blocks a briefing; the Analyst then runs as it would without a planner.
"""
from __future__ import annotations

import re
import time

from psw import config
from psw.agents import sentinel
from psw.agents.runtime import AgentSpec, run_agent
from psw.agents.tools import TOOL_SCHEMAS, DataStore
from psw.agents.verifier import FIELD_NAME, extract_numbers

MAX_STEPS, MAX_QUESTIONS, MAX_CHARS = 5, 4, 200
YEAR_OR_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b|\b(?:19|20)\d{2}\b")

SYSTEM_PROMPT = """You plan a short district briefing about stubble-fire activity and air quality in Punjab, India. You do not write the briefing and you cannot call tools. A separate analyst will follow your plan using these tools:
{tools}

You are told the district, the date, the nearest monitored city, and which Sentinel rules flagged the district (spike = fire count far above its recent mean; high_count = one of the highest counts in Punjab that day; smog_risk = the nearest city's PM2.5 forecast is high; none = the user asked for it directly).

Rules:
- You have no data. Never write, guess or imply any number. The only digits allowed are years and dates, for example "2024 season".
- Never write tool field names with underscores in the text. Describe things in plain words.
- Adapt to the flags. If the district is flagged only for smog_risk, fire counts may be low, so put the forecast and the wind first and say fire activity is context. If it is flagged for fires, start with the fire activity.
- Always include a step to search the knowledge base for the district's earlier season (put the district name and the year in the query) and a step to cover one limitation.
- Fire detections are a lower bound and upwind fires are context, not proof of cause. No health advice.

Return only this JSON object, with no other text:
{{"steps": ["3 to 5 short imperative steps for the analyst"], "questions_to_answer": ["2 to 4 questions the briefing must answer"]}}"""


def build_planner(tool_schemas: list[dict] | None = None) -> AgentSpec:
    lines = "\n".join(f"- {s['function']['name']}: {s['function']['description']}" for s in (tool_schemas or TOOL_SCHEMAS))
    return AgentSpec(name="planner", system_prompt=SYSTEM_PROMPT.format(tools=lines), tools=[], max_turns=2, json_output=True)


def flag_kinds(store: DataStore, district: str, date: str) -> list[str]:
    """The kinds of Sentinel flag the district has on `date` ([] if none or the district is unknown)."""
    for row in sentinel.scan(date, store):
        if row["district"] == district:
            return [f.split(":")[0] for f in row["flags"]]
    return []


def user_message(district: str, date: str, city: str, kinds: list[str]) -> str:
    return (f"District: {district}\nDate: {date}\nNearest monitored city: {city}\n"
            f"Sentinel flags: {', '.join(kinds) if kinds else 'none'}\nWrite the plan.")


def _item_ok(text: str) -> bool:
    """No raw field names, and no numbers other than years, dates and the Verifier's own constants (7-day, 80% interval, ...)."""
    if FIELD_NAME.search(text):
        return False
    allowed = {float(c) for c in config.VERIFY_CONSTANTS}
    return all(n.value in allowed for n in extract_numbers(YEAR_OR_DATE.sub(" ", text)))


def _clean_list(raw, limit: int) -> list[str]:
    if not isinstance(raw, list):
        return []
    out = []
    for x in raw:
        if isinstance(x, str) and x.strip():
            t = " ".join(x.split())[:MAX_CHARS]
            if _item_ok(t):
                out.append(t)
    return out[:limit]


def clean_plan(raw) -> dict | None:
    """Keep only well-formed, number-free items. None when no usable step is left."""
    if not isinstance(raw, dict):
        return None
    steps = _clean_list(raw.get("steps"), MAX_STEPS)
    if not steps:
        return None
    return {"steps": steps, "questions_to_answer": _clean_list(raw.get("questions_to_answer"), MAX_QUESTIONS)}


def reject_reason(raw) -> str:
    """Why `clean_plan(raw)` returned None, in words for the trace and the app."""
    if not isinstance(raw, dict):
        return "the planner did not return a JSON object"
    steps = raw.get("steps")
    if not isinstance(steps, list) or not steps:
        return "the plan has no list of steps"
    return f"all {len(steps)} steps were dropped (they contained numbers, raw field names or no text)"


def plan_text(plan: dict) -> str:
    """The plan as it is appended to the Analyst's first message."""
    steps = "\n".join(f"{i}. {s}" for i, s in enumerate(plan["steps"], 1))
    qs = "\n".join(f"- {q}" for q in plan.get("questions_to_answer", []))
    return ("A planner wrote this plan. Follow it where it helps. It is guidance, not evidence: every rule above still applies, "
            "and every number must come from a tool result.\n" + steps + ("\nQuestions the briefing should answer:\n" + qs if qs else ""))


def make_plan(district: str, date: str, city: str, kinds: list[str], llm) -> tuple[dict | None, str | None, dict]:
    """Run the planner. Returns (plan or None, error or None, a trace entry). Never raises because of the LLM."""
    t0 = time.perf_counter()
    res = run_agent(build_planner(), user_message(district, date, city, kinds), llm)
    plan = None if res.error else clean_plan(res.output)
    error = res.error or (None if plan else reject_reason(res.output))
    summary = (str(plan) if plan else f"no plan: {error}")
    entry = {"step": 1, "agent": "planner", "tool": "plan", "args": {"sentinel_flags": kinds},
             "result_summary": summary if len(summary) <= 180 else summary[:179] + "…", "ms": int((time.perf_counter() - t0) * 1000)}
    return plan, error, entry
