"""How a saved agent record is shown: the text with its badge on the left, its evidence in tabs on the right.

Shared by the Watch, Audit and Response Planner pages.
"""
from __future__ import annotations

import json
import re

import pandas as pd
import streamlit as st

from psw.app.helpers import agent_steps, badge, explain_numbers, retrieved_chunks

VERIFIER_NOTE = ("The Verifier proves that every number appears in the tool outputs and that every cited chunk was "
                 "retrieved. It does not prove that a cited source supports the sentence, so read the sources.")


LABEL = re.compile(r"^([A-Z][A-Za-z ,']{2,40}):(?=\s|$)", re.MULTILINE)


def bold_labels(text: str) -> str:
    """Bold the section labels the agents write ("Why:", "What other signals say:") at the start of a line."""
    return LABEL.sub(r"**\1:**", text)


def agent_line(rec: dict, writer: str | None = None) -> str:
    """The agents that made this brief as one line of badges: LLM agents violet, rule-based agents grey, a failed
    check red and a template fallback orange."""
    kind, _ = badge(rec)
    parts = []
    for s in agent_steps(rec, writer):
        colour = "violet" if s["kind"].startswith("LLM") else "gray"
        if s["agent"] == "Verifier":
            colour = {"success": "green", "warning": "green", "error": "red"}[kind]
        if "template" in s["did"]:
            colour = "orange"
        icon = ":material/smart_toy:" if s["kind"].startswith("LLM") else ":material/rule:"
        suffix = " agent" if s["kind"].startswith("LLM") else ""
        parts.append(f":{colour}-badge[{icon} {s['agent']}{suffix}] {s['did']}")
    return " &nbsp;→&nbsp; ".join(parts)


def show_text(rec: dict, writer: str | None = None) -> None:
    """Headline, the agents that made it, errors, the text itself and the Critic's warning."""
    st.subheader(rec["headline"])
    st.markdown(agent_line(rec, writer))
    if rec.get("model"):
        st.caption(f"Language model: {rec['model']}. The LLM agents write words; every number comes from Python tools.")
    for e in rec["verification"].get("errors", []):
        st.error(e)
    if rec["mode"] == "template":
        st.caption(rec.get("label") or "Written from a fixed template because the agent was unavailable or failed "
                                       "verification.")
        if rec.get("llm_error"):
            st.caption(f"Reason: {rec['llm_error']}")
    st.markdown(re.sub(r"\n+", "\n\n", bold_labels(rec["briefing"])))
    if rec.get("district_paragraph"):
        st.markdown(f"**{rec['district']}:** {rec['district_paragraph']}")
    if (rec.get("critic") or {}).get("warning"):
        st.warning(rec["critic"]["warning"])


def show_provenance(rec: dict, extra: str = "") -> None:
    st.caption(f"Mode: {rec['mode']} · model: {rec.get('model')} · generated {rec.get('generated_at', '')[:19]} UTC"
               + (f" · {extra}" if extra else ""))
    st.caption(VERIFIER_NOTE)


def show_record(rec: dict, fixed: str, caveat: str | None = None, notes: tuple[str, ...] = (), extra: str = "",
                writer: str | None = None) -> None:
    """A saved brief as a reader sees it: the text in a box with the agents that made it, then how it was checked."""
    with st.container(border=True):
        show_text(rec, writer)
        if caveat:
            st.caption(caveat)
        for note in notes:
            st.caption(note)
    with st.expander("How the agents made and checked this brief: trace, sources, Verifier report"):
        show_provenance(rec, extra)
        show_tabs(rec, fixed)


def _plan(rec: dict) -> None:
    if rec.get("plan"):
        st.markdown("**Plan** (written by the Planner agent before the writing agent ran)")
        for i, step in enumerate(rec["plan"]["steps"], 1):
            st.markdown(f"{i}. {step}")
        if rec["plan"].get("questions_to_answer"):
            st.markdown("**Questions the text should answer**")
            for q in rec["plan"]["questions_to_answer"]:
                st.markdown(f"- {q}")
        st.caption("The plan is guidance, not evidence. Numbers still come only from tools, and the Verifier checks "
                   "the text.")
    elif rec.get("planner_error"):
        st.caption(f"The Planner was asked to run but produced no usable plan ({rec['planner_error']}), so the agent "
                   "ran without one.")


def show_tabs(rec: dict, fixed: str) -> None:
    """Trace, sources, facts and the Verifier report. `fixed` says which arguments the orchestrator injects."""
    t_trace, t_src, t_facts, t_ver = st.tabs(["Agent trace", "Sources", "Facts JSON", "Verifier report"])
    with t_trace:
        _plan(rec)
        if rec["trace"]:
            tr = pd.DataFrame(rec["trace"])
            tr["args"] = tr["args"].map(lambda a: json.dumps(a, ensure_ascii=False))
            cols = [c for c in ("step", "agent", "tool", "args", "result_summary", "ms") if c in tr]
            st.dataframe(tr[cols], hide_index=True, width="stretch")
            st.caption(f"Result summaries are cut at 180 characters; the full results are in Facts JSON. {fixed}")
        else:
            st.write("No trace was saved.")
    with t_src:
        chunks = retrieved_chunks(rec)
        if not chunks:
            st.write("No chunks were retrieved.")
        for c in chunks:
            with st.expander(f"[{c['chunk_id']}] {c['title']}" + ("  ✔ cited" if c["cited"] else ""),
                             expanded=c["cited"]):
                st.markdown(c["text"])
    with t_facts:
        st.json(rec["facts"], expanded=False)
    with t_ver:
        v = rec["verification"]
        st.write(f"**{'Passed' if v['ok'] else 'Failed'}**: {len(v.get('checked_numbers', []))} numbers checked, "
                 f"{len(v.get('errors', []))} errors.")
        if rec.get("first_attempt_errors"):
            st.write("The first draft failed on:")
            for e in rec["first_attempt_errors"]:
                st.write(f"- {e}")
        ex = explain_numbers(rec)
        if not ex.empty:
            st.dataframe(ex, hide_index=True, width="stretch")
        st.caption("Each number in the text, the evidence value it matched, and the tool that returned it. "
                   "Rounding and the sign are ignored when matching.")
