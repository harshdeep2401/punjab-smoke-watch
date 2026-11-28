"""Daily district briefing: the Sentinel watchlist, then one cited, verified district briefing with its agent trace."""
from datetime import date as Date

import pandas as pd
import streamlit as st

from psw.agents.demo import load_demo
from psw.app.common import get_kb, get_store, load_district_daily, scan_watchlist
from psw.app.helpers import cached_districts, find_briefing, peak_day
from psw.app.record_view import show_record
from psw.llm import LLM, LLMError

st.title("Daily district briefing")
st.caption("The rule-based **Sentinel** screens every district for a date; the **Analyst agent** (an LLM with data "
           "tools) writes a cited briefing for one, and the **Verifier** checks its numbers.")

daily = load_district_daily()
all_days = pd.to_datetime(daily["date"])
lo, hi = all_days.min().date(), all_days.max().date()
demo = load_demo() or {}
default_day = Date.fromisoformat(demo["peak"]) if demo.get("peak") else peak_day(daily)

day = st.sidebar.date_input("Date", value=default_day, min_value=lo, max_value=hi, format="YYYY-MM-DD")
if demo.get("peak"):
    st.sidebar.caption(f"Demo dates with saved briefings: peak {demo['peak']}, moderate {demo['moderate']}, early {demo['early']}.")
else:
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
st.write(f"**{len(flagged)}** of {len(rows)} districts are flagged by fixed rules (spike, high count, smog risk). "
         "No language model is involved in this table.")
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
st.subheader("Analyst briefing")
order = [r["district"] for r in flagged] + [r["district"] for r in rows if not r["flags"]]
flag_set = {r["district"] for r in flagged}


def _label(d: str) -> str:
    return f"{d}  ({'flagged' if d in flag_set else 'not flagged'}{', briefing saved' if d in have else ''})"


want = demo.get("default_district")
district = st.selectbox("District", order, index=order.index(want) if want in order else 0, format_func=_label,
                        key=f"district_{date}")
rec = find_briefing(date, district)


def run_agent_button(label: str) -> None:
    use_planner = st.checkbox("Plan first (one extra LLM call)", key=f"plan_{label}",
                              help="A Planner agent writes a short plan that the Analyst follows. The Verifier still checks the text.")
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
            run_briefing(district, date, llm=llm, store=get_store(), kb=get_kb(), use_planner=use_planner)
        except Exception as e:  # noqa: BLE001 - the page must not crash on a provider or data problem
            st.error(f"The run failed: {type(e).__name__}: {e}")
            return
    st.rerun()


if rec is None:
    st.info(f"No saved briefing for {district} on {date}.")
    run_agent_button("Run agent")
else:
    show_record(rec, "Date, city and the retrieval filters are fixed by the orchestrator, not chosen by the model.",
                extra=f"nearest city: {rec.get('nearest_city')}", writer="Analyst")
    run_agent_button("Regenerate with the agent")
