"""The Auditor: an `AgentSpec` (prompt + tools) run through `run_agent`. It writes the audit brief for one district
(or Punjab) and one season.

The orchestrator injects the district, the season and the retrieval cut-off into every tool call.
"""
from __future__ import annotations

from psw import config
from psw.agents import audit_tools
from psw.agents.audit_tools import STATE, AuditStore
from psw.agents.runtime import AgentSpec

REQUIRED_TOOLS = ["get_detection_trend", "get_diurnal_profile", "get_gap_range", "search_knowledge"]

SYSTEM_PROMPT = """You are an auditor of satellite fire monitoring in Punjab, India. You write a short audit brief for monitoring cells, researchers, journalists and citizens who want to check claims that stubble fires have fallen.

Use the tools to gather facts. You MUST call get_detection_trend, get_diurnal_profile, get_gap_range and search_knowledge at least once. Also call get_burned_area_comparison and get_official_vs_satellite. Search at least once for "how the detection gap is calculated and its weaknesses".

The briefing text has exactly these four parts, separated by blank lines, each starting with its label and a colon:
What detections say:
What other signals say:
Our best range and how sure we are:
What would settle it:

Length: 170 to 190 words in total, about 45 words in each part. A shorter text is rejected.
Citations: a brief with no citation is rejected. Put at least one [K012]-style citation after a sentence about the method, the burned-area product, official counting, pass times or published reporting, and list it in "citations". Citations are chunk ids only, never web addresses. Cite a chunk only if its text supports that sentence. Figures from the data tools need no citation. Never cite a chunk about another season as support for this season's figures.

How to write it (these are instructions for you; do not copy them into the brief):
- Use only numbers that appear in tool results, written as returned. Do no sums, differences, averages or percentages of your own.
- For changes, use the "in words" fields ("down 91.1% from 2022"). Do not write "a fall of -91.1".
- Give the range in the words "counts per burned km² fell" or "the fall in counts per burned km²". It is not a measure of how many fires or how much burning went unseen, and you must not present it as one.
- Report only the range that get_gap_range returns for this district and season. If it returns no range, write that there is insufficient evidence and give its reason.
- Next to the range, write "confidence" followed by the level get_gap_range returns. If its direction is "direction unclear", use the word "unclear"; if it is "no sign of a detection gap", use the words "no sign".
- Include the word "triangulation" and state that this is not field verification.
- When you give a range, include one sentence saying the range does not include the error of the burned-area product, which is unknown and is the largest uncertainty.
- Mention that a fall in detections per burned area can also come from smaller or cooler fires or a change in the burned-area product.
- Mention that the official count is satellite-based, so agreeing with it is not independent confirmation.
- Do not assert that farmers hide fires or that official numbers are wrong. Others' claims may be reported, with a citation.
- No raw field names (write "lowest estimate", not a name with underscores). No individual farmers or fields. Plain language.

Your final answer must be only this JSON object, with no other text:
{"headline": "at most 12 words", "briefing": "the four labelled parts with [K012]-style citations", "citations": ["K012"]}"""

REVISE_PROMPT = """Your draft failed verification. Fix these problems and return the corrected JSON object only:
{errors}

Quote only numbers that appear in the tool results, exactly as returned. Cite at least one chunk you retrieved, in the text as [K012] and in "citations". Keep the four labelled parts. Write between 170 and 190 words: if the text was too short, add a full sentence to each part."""


def build_auditor(store: AuditStore, kb) -> AgentSpec:
    return AgentSpec(name="auditor", system_prompt=SYSTEM_PROMPT, tools=audit_tools.build_tools(store, kb),
                     required_tools=REQUIRED_TOOLS, max_turns=8, json_output=True, word_range=config.AUDIT_WORDS)


def injected_args(district: str, season: int) -> dict:
    """Arguments the model cannot override. `before` is the retrieval leakage filter."""
    place = {"district": district, "season": int(season)}
    args = {name: dict(place) for name in audit_tools.DATA_TOOLS}
    args["search_knowledge"] = {"before": audit_tools.cutoff(season), "district": None if district == STATE else district,
                                "place": district}
    return args


def user_message(district: str, season: int, plan_text: str | None = None) -> str:
    place = "the whole of Punjab" if district == STATE else f"{district} district"
    msg = (f"Write the audit brief for {place}, {season} season (Sep 15 to Nov 30). "
           f"Compare with the base season that the tools report.")
    return msg + ("\n\n" + plan_text if plan_text else "")
