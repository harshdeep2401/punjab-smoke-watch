"""Daily Watch & Briefing: the Sentinel watchlist, then one cited, verified district briefing with its agent trace."""
import json
from datetime import date as Date

import pandas as pd
import streamlit as st

from psw.agents import analyst, tools
from psw.app.common import get_kb, get_store, load_district_daily, scan_watchlist
from psw.app.helpers import badge, cached_districts, explain_numbers, find_briefing, peak_day, retrieved_chunks
from psw.llm import LLM, LLMError

st.set_page_config(page_title="Watch & Briefing", layout="wide")
st.title("Daily Watch & Briefing")

daily = load_district_daily()
all_days = pd.to_datetime(daily["date"])
lo, hi = all_days.min().date(), all_days.max().date()
default_day = peak_day(daily)

day = st.sidebar.date_input("Date", value=default_day, min_value=lo, max_value=hi, format="YYYY-MM-DD")
st.sidebar.caption(f"Default is the busiest day of the latest season ({default_day.isoformat()}).")
date = day.isoformat() if isinstance(day, Date) else str(day)

# ---------- Sentinel ----------
try:
    rows = scan_watchlist(date)
except ValueError:
    st.warning(f"No fire data for {date}. Pick a date inside a season (15 September to 30 November).")
    st.stop()

flagged = [r for r in rows if r["flags"]]
have = cached_districts(date)
st.subheader(f"Sentinel watchlist for {date}")
st.write(f"**{len(flagged)}** of {len(rows)} districts are flagged by the rules (spike, high count, smog risk). "
         f"The Sentinel is deterministic: no LLM is involved in this table.")
if all(r["nearest_city_p90"] is None for r in rows):
    st.info("There is no PM2.5 forecast for this date (forecasts exist only for the test seasons), "
            "so the smog-risk rule cannot fire and a briefing will have no forecast section.")

show_all = st.checkbox("Show districts that are not flagged", value=False)
table = pd.DataFrame([{
    "district": r["district"], "fire count": r["fire_count"], "7-day mean": r["fire_count_7d_mean"],
    "× mean": r["ratio"], "nearest city": r["nearest_city"], "city p90 (µg/m³)": r["nearest_city_p90"],
    "briefing saved": r["district"] in have, "why flagged": "; ".join(f.split(":")[0] for f in r["flags"]) or "-",
    "reasons": " | ".join(r["flags"]),
} for r in (rows if show_all else flagged)])
if table.empty:
    st.write("No district is flagged on this date.")
else:
    st.dataframe(table.drop(columns=["reasons"]), hide_index=True, width="stretch")
    with st.expander("Full reasons for each flag"):
        for _, t in table.iterrows():
            st.markdown(f"**{t['district']}**: {t['reasons'] or 'not flagged'}")
st.caption("Flags are rule-based screening, not a diagnosis. Detections are a lower bound: satellites miss fires "
           "under cloud and outside overpass times.")

# ---------- district + briefing ----------
st.divider()
order = [r["district"] for r in flagged] + [r["district"] for r in rows if not r["flags"]]
flag_set = {r["district"] for r in flagged}


def _label(d: str) -> str:
    return f"{d}  ({'flagged' if d in flag_set else 'not flagged'}{', briefing saved' if d in have else ''})"


district = st.selectbox("District", order, format_func=_label, key=f"district_{date}")
rec = find_briefing(date, district)


def run_agent_button(label: str) -> None:
    if not st.button(label, help="Calls the LLM configured in .env. Saves the result to data/processed/briefings/."):
        return
    try:
        llm = LLM.from_env()
    except LLMError:
        st.warning("No LLM is configured, so the agent cannot run here. Set `LLM_BASE_URL`, `LLM_API_KEY` and "
                   "`LLM_MODEL` in `.env` (see `.env.example`). Saved briefings can still be read.")
        return
    from psw.agents.orchestrator import run_briefing
    with st.spinner(f"Running the agent for {district} on {date} (about a minute on a free tier)…"):
        try:
            run_briefing(district, date, llm=llm, store=get_store(), kb=get_kb())
        except Exception as e:  # noqa: BLE001 - the page must not crash on a provider or data problem
            st.error(f"The run failed: {type(e).__name__}: {e}")
            return
    st.rerun()


if rec is None:
    st.info(f"No saved briefing for {district} on {date}.")
    run_agent_button("Run agent")
else:
    left, right = st.columns([5, 4])
    kind, text = badge(rec)
    with left:
        st.subheader(rec["headline"])
        getattr(st, kind)(text)
        for e in rec["verification"].get("errors", []):
            st.error(e)
        if rec["mode"] == "template":
            st.caption(rec.get("label") or "Written from a fixed template because the agent was unavailable or failed verification.")
            if rec.get("llm_error"):
                st.caption(f"Reason: {rec['llm_error']}")
        st.markdown(rec["briefing"])
        st.caption(f"Mode: {rec['mode']} · model: {rec.get('model')} · generated {rec.get('generated_at', '')[:19]} UTC · "
                   f"nearest city: {rec.get('nearest_city')}")
        st.caption("The Verifier proves that every number appears in the tool outputs and that every cited chunk was "
                   "retrieved. It does not prove that a cited source supports the sentence, so read the sources.")
        run_agent_button("Regenerate with the agent")
    with right:
        t_trace, t_src, t_facts, t_ver = st.tabs(["Agent trace", "Sources", "Facts JSON", "Verifier report"])
        with t_trace:
            if rec["trace"]:
                tr = pd.DataFrame(rec["trace"])
                tr["args"] = tr["args"].map(lambda a: json.dumps(a, ensure_ascii=False))
                cols = [c for c in ("step", "agent", "tool", "args", "result_summary", "ms") if c in tr]
                st.dataframe(tr[cols], hide_index=True, width="stretch")
                st.caption("Result summaries are cut at 180 characters; the full results are in Facts JSON. "
                           "Date, city and the retrieval filters are fixed by the orchestrator, not chosen by the model.")
            else:
                st.write("No trace was saved.")
        with t_src:
            chunks = retrieved_chunks(rec)
            if not chunks:
                st.write("No chunks were retrieved.")
            for c in chunks:
                with st.expander(f"[{c['chunk_id']}] {c['title']}" + ("  ✔ cited" if c["cited"] else ""), expanded=c["cited"]):
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

# ---------- how it works ----------
with st.expander("How the agent works"):
    st.code(
        "Sentinel (rules, no LLM)  ->  ranked watchlist\n"
        "        |\n"
        "        v   top flagged district\n"
        "Analyst (LLM + tools)  ->  calls data tools and search_knowledge, writes JSON {headline, briefing, citations}\n"
        "        |\n"
        "        v\n"
        "Verifier (rules, no LLM)  ->  every number in the tool outputs? every citation retrieved? length, banned content\n"
        "        |  fail: one revision with the error list\n"
        "        v  fail again: fixed template\n"
        "Saved briefing (this page)",
        language="text")
    st.markdown("**Rule: the LLM never does the maths.** Every number comes from a Python tool, and the Verifier enforces it.")
    st.markdown("**Tools the Analyst can call**")
    for s in tools.TOOL_SCHEMAS:
        f = s["function"]
        st.markdown(f"- `{f['name']}`: {f['description']}")
    st.markdown("**System prompt**")
    st.code(analyst.SYSTEM_PROMPT, language="text")
