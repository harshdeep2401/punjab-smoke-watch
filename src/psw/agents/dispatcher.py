"""The Dispatcher: an `AgentSpec` (prompt + tools) run through `run_agent`. It explains a 3-day allocation that a
fixed rule has already made, for the state control room. It never sets or changes the allocation (D19).
"""
from __future__ import annotations

from psw import config
from psw.agents import dispatch_tools
from psw.agents.dispatch_tools import ResponseStore, Settings, as_date
from psw.agents.runtime import AgentSpec

REQUIRED_TOOLS = ["get_risk_forecast", "get_allocation", "search_knowledge"]

SYSTEM_PROMPT = """You write a short 3-day action brief for Punjab's state agriculture control room: where residue-management machines and awareness teams should go, and why. The split of units has already been computed by a fixed rule. You explain it. You cannot change it.

This is support for farmers, not enforcement. Never mention penalties, fines, FIRs, police, prosecution or enforcement, even to say the brief is not about them. Never name a person, farm, field or village.

Use the tools to gather facts. You MUST call get_risk_forecast, get_allocation and search_knowledge. For the district that gets the most units, also call get_rain_outlook and get_hotspot_cells. If planning on likely burning, call get_gap_range for the district the adjustment moves most. Search for "managing paddy straw without burning, machines and timing".

The briefing text has exactly these four parts, separated by blank lines, each starting with its label and a colon:
Where the units go:
Why:
How sure we are:
What to watch:

Length: 160 to 185 words in total. A shorter or longer text is rejected. Write short full sentences, one idea in each; do not chain clauses with commas.
- Where the units go: begin "Of N units, the largest shares go to", naming the five districts with the most units, each written as "Name (N units)"; then say that the other districts get smaller shares, and give the limit per district. Mention units not placed only if get_allocation reports units_not_placed.
- Why: for the leading district, its forecast detections with the 80% band and how that compares with the last 3 days. If planning on likely burning, also give its likely burning, always under that name, and in its own sentence name the district the adjustment moves most, as "The adjustment moves Name from N units to M units."
- How sure we are: begin "The forecast is uncertain." If planning on likely burning, write that likely burning scales each forecast by the previous season's detection gap; then "The gap range behind NAME has low confidence." (NAME is the district the adjustment moves most; use its confidence_of_its_gap_range, not Punjab's); then that likely burning is "not a count of fires". If planning on detections only, write that the figures are not adjusted for the detection gap. Say that what one unit achieves is assumed.
- What to watch: rain in the leading district over the last and next 3 days, saying in the same sentence that the next 3 days come from the weather archive standing in for a forecast; where awareness teams might start, giving each cell's detections (if get_hotspot_cells returns no cells, say there is no cluster); and, as a separate last sentence, one fact from the knowledge base about machines or timing, with its citation.

Rules:
- Use only numbers that appear in tool results, written as returned. Do no sums, differences, averages or percentages of your own.
- Write unit figures only in these forms, because any other is rejected: "Name (N units)" for a district; "The adjustment moves Name from N units to M units."; "Of N units, ..." for the whole fleet; "The limit per district is N units." Put a unit figure in a sentence with no other figures. Never write a figure for machines or teams, never write one in words, and never add up units.
- Name nothing smaller than a district except a cell: no village, town, block, person or farm.
- Name only cells that get_hotspot_cells returns, copied exactly, for example "cell 55 (south Sangrur)".
- Likely burning is not observed burning. Do not say machines will prevent fires.
- Put the citation marker (the chunk id in square brackets, copied exactly from search_knowledge) only after a sentence that says what that chunk's text says, and list it in "citations". A chunk about machines or timing does not support a sentence about the forecast, confidence or the detection gap. Figures from the data tools need no citation.
- No raw field names. Plain language.

Your final answer must be only this JSON object, with no other text:
{"headline": "at most 12 words", "briefing": "the four labelled parts with citation markers", "citations": ["chunk ids you cited"]}"""

DISTRICT_REQUEST = """Also add a key "district_paragraph" to the JSON: 45 to 70 words about {district} only, whose first sentence gives "{district} (N units)" and no other figure; then its forecast or likely burning, its rain and its cells. Call get_risk_forecast and get_allocation with district "{district}", and get_rain_outlook and get_hotspot_cells for it."""

REVISE_PROMPT = """Your draft failed verification. Fix these problems and return the corrected JSON object only:
{errors}

Quote only numbers that appear in the tool results, exactly as returned. Write unit figures only as "Name (N units)", "moves Name from N units to M units", "Of N units, ..." or "The limit per district is N units". Cite at least one chunk you retrieved, in square brackets, and list it in "citations". Keep the four labelled parts. Write between 160 and 185 words in the briefing: if the text was too short, add a full sentence to each part; if it was too long, delete whole sentences."""


def build_dispatcher(store: ResponseStore, kb) -> AgentSpec:
    return AgentSpec(name="dispatcher", system_prompt=SYSTEM_PROMPT, tools=dispatch_tools.build_tools(store, kb),
                     required_tools=REQUIRED_TOOLS, max_turns=10, json_output=True, word_range=config.DISPATCH_WORDS)


def injected_args(day: str, settings: Settings) -> dict:
    """Arguments the model cannot override: the date, the user's settings, the previous season for the gap range and
    the retrieval cut-off."""
    return {
        "get_risk_forecast": {"date": day, "use_likely": settings.use_likely},
        "get_allocation": {"date": day, "units": settings.units, "max_share": settings.max_share,
                           "use_likely": settings.use_likely},
        "get_hotspot_cells": {"date": day},
        "get_rain_outlook": {"date": day},
        "get_gap_range": {"season": as_date(day).year - 1},
        "search_knowledge": {"before": day},
        "get_machine_stock": {"date": day},
    }


def user_message(day: str, settings: Settings, district: str | None = None) -> str:
    basis = "likely burning (the forecast adjusted for the detection gap)" if settings.use_likely else \
        "forecast detections only"
    msg = (f"Write the dispatch brief issued on {day}, for the 3 days that follow. {settings.units} units are available "
           f"and the split is planned on {basis}.")
    return msg + ("\n\n" + DISTRICT_REQUEST.format(district=district) if district else "")
