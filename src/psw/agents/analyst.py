"""The Analyst: an `AgentSpec` (prompt + tools) run through `run_agent`. It is the only agent that uses the LLM.

The orchestrator injects the briefing date, district and nearest city into every tool call, so the model cannot
look at other dates (no leakage) or other places.
"""
from __future__ import annotations

from psw.agents.runtime import AgentResult, AgentSpec, run_agent
from psw.agents.tools import DataStore, build_tools, nearest_city

REQUIRED_TOOLS = ["get_district_snapshot", "get_city_forecast", "get_wind_and_upwind", "search_knowledge"]

SYSTEM_PROMPT = """You are an environmental analyst writing a district briefing about stubble-fire activity and air quality in Punjab, India, for officials and the public.

Use the tools to gather facts. You MUST call get_district_snapshot, get_city_forecast, get_wind_and_upwind and search_knowledge at least once. get_model_reliability tells you how accurate the forecast has been.

Rules:
- Use ONLY numbers that appear in tool results, written exactly as the tool returned them (same rounding). Never calculate new numbers: no sums, differences, averages or percentages of your own.
- Cite knowledge chunks as [K012] right after the sentence they support, and list every cited id in "citations". Only cite chunks that search_knowledge returned.
- When you search, put names and years in the query (for example "Sangrur 2024 season" or "limitations of satellite fire detection"), not words like "last season".
- Wind direction means the direction the wind blows FROM.
- rank_in_punjab is the district's rank among Punjab's districts that day (1 = most detections). punjab_total is the number of detections in all of Punjab. Never write "rank X out of <punjab_total>".
- upwind_frp_l0 is a pressure score without a unit, not megawatts. share_pct is a share of that score.
- Describe upwind fire activity as context, never as proof that it caused the PM2.5 level. Fire detections are a lower bound.
- If a tool returns an error, say that the information is unavailable. Do not guess.
- No health or medical advice. Never name or identify individual farmers or fields.
- Plain language, 120 to 160 words. Mention the forecast uncertainty band (the 80% interval from p10 to p90) and at least one limitation.

Your final answer must be only this JSON object, with no other text:
{"headline": "at most 12 words", "briefing": "the 120 to 160 word briefing with [K012]-style citations", "citations": ["K012"]}"""

REVISE_PROMPT = """Your draft failed verification. Fix these problems and return the corrected JSON object only:
{errors}

Quote only numbers that appear in the tool results, exactly as returned. Cite only chunks you retrieved, in the text as [K012] and in "citations". Keep the briefing between 120 and 160 words."""


def build_analyst(store: DataStore, kb) -> AgentSpec:
    return AgentSpec(name="analyst", system_prompt=SYSTEM_PROMPT, tools=build_tools(store, kb),
                     required_tools=REQUIRED_TOOLS, max_turns=8, json_output=True)


def injected_args(district: str, date: str, city: str) -> dict:
    """Arguments the model cannot override. `before` and `district` are the retrieval leakage filters."""
    return {
        "get_district_snapshot": {"district": district, "date": date},
        "get_city_forecast": {"city": city, "date": date},
        "get_wind_and_upwind": {"city": city, "date": date},
        "get_model_reliability": {"city": city},
        "search_knowledge": {"before": date, "district": district},
    }


def user_message(district: str, date: str, city: str) -> str:
    year = int(date[:4])
    return (f"Write the briefing for {district} district for {date}. The nearest monitored city is {city}. "
            f"Season summaries in the knowledge base exist for earlier seasons only (for example {year - 1}).")


def analyse(district: str, date: str, llm, store: DataStore | None = None, kb=None) -> AgentResult:
    """Run the Analyst for one district-day. `store` and `kb` default to the committed data and index."""
    store = store or DataStore()
    if kb is None:
        from psw.rag.index import KnowledgeBase
        kb = KnowledgeBase()
    city = nearest_city(district, store)
    return run_agent(build_analyst(store, kb), user_message(district, date, city), llm,
                     injected_args=injected_args(district, date, city))
