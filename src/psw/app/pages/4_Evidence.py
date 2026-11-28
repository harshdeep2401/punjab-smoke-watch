"""Evidence: can you trust it? The audit method, the fire-risk and allocation backtests, the agents, the PM2.5 forecast,
retrieval, sources and limitations, as tabs in the order of the story."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from psw import config
from psw.agents import analyst, audit_tools, dispatch_tools, tools
from psw.app import audit_view
from psw.app import evidence_view as ev
from psw.app.common import CREDITS, LIMITATIONS, LLM_RULE, load_geojson, load_metrics
from psw.app.helpers import (AGENTS, VARIANT_LABEL, VARIANTS, by_city_frame, importance_frame, interpret, interpret_retrieval,
                             interpret_risk, load_retrieval_report, load_risk_metrics, metrics_frame, reliability,
                             retrieval_frame, retrieval_queries)

st.title("Can you trust it?")
st.caption("Every result here is computed from the metrics files, audit tables and saved briefs in `data/processed/`.")

m = load_metrics()
districts_ = [f["properties"]["district"] for f in load_geojson()["features"]]
BEST = "background-color: #cfe8cf; color: #1a1a1a; font-weight: bold"


def highlight_best(df: pd.DataFrame, lower: list[str], target: dict[str, float] | None = None) -> pd.DataFrame:
    """Style: the best value in each column (lowest, or closest to a target)."""
    css = pd.DataFrame("", index=df.index, columns=df.columns)
    for c in lower:
        css.loc[df[c].idxmin(), c] = BEST
    for c, t in (target or {}).items():
        css.loc[(df[c] - t).abs().idxmin(), c] = BEST
    return css


def tool_list(schemas: list[dict]) -> None:
    for s in schemas:
        st.markdown(f"- `{s['function']['name']}`: {s['function']['description']}")


t_audit, t_resp, t_agents, t_pm, t_rag, t_src = st.tabs(
    ["Audit method", "Forecast and allocation", "Agents", "PM2.5 forecast", "Retrieval", "Sources and limits"])

# ---------- audit method ----------
with t_audit:
    st.subheader("How the gap range is built")
    for line in ev.audit_method_lines():
        st.markdown(f"- {line}")
    gaps_ = ev.audit_gap_frame(audit_view.load_audit_tables(config.PROCESSED))
    if gaps_.empty:
        st.info("No audit tables found. Run `uv run python -m psw.pipeline audit --tables-only`.")
    else:
        st.dataframe(gaps_.style.format({"S-NPP change (%)": "{:+.1f}", "burned area change (%)": "{:+.1f}"},
                                        na_rep="-"), hide_index=True, width="stretch")
    st.subheader("Limits")
    for line in ev.AUDIT_LIMITS:
        st.markdown(f"- {line}")
    st.caption(audit_view.CAPTION)

# ---------- fire risk and decision backtest ----------
with t_resp:
    decision_ = ev.load_decision()
    st.subheader("Forward test: did it hold on 2026?")
    forward_ = ev.load_forward()
    if forward_ is None:
        st.info("No forward test found. Run `uv run python -m psw.pipeline forward-test` (needs the network once).")
    else:
        for line in ev.forward_summary(forward_):
            st.markdown(line)
        st.dataframe(ev.forward_frame(forward_, load_risk_metrics(), decision_), hide_index=True, width="stretch")
        st.caption("S-NPP's near-real-time feed stops on 2 November 2026, so the clean arm ends before the usual peak. "
                   f"The second arm feeds the same model NOAA-20 counts times {forward_['noaa20_scale']:.3f} (the "
                   "S-NPP to NOAA-20 ratio in 2022 to 2025) to cover the peak; swapping the input sensor is itself a "
                   "change, so read it as secondary. " + ev.detection_note(forward_))
        with st.expander("The pass marks, set before the run"):
            for arm in forward_["arms"]:
                st.markdown(f"**{ev.arm_column(forward_, arm)}**")
                for line in ev.criteria_lines(forward_, arm):
                    st.markdown(f"- {line}")
            st.caption(" ".join(forward_["notes"]) + " Plan: `docs/FORWARD_TEST_PLAN.md`.")

    st.subheader("Does the forecast help decide where units go? (backtest)")
    if decision_ is None:
        st.info("No decision backtest found. Run `uv run python -m psw.pipeline risk-train`.")
    else:
        left, right = st.columns([3, 2], gap="large")
        with left:
            st.plotly_chart(ev.decision_figure(decision_, st.context.theme.type), width="stretch")
        with right:
            for line in ev.interpret_decision(decision_):
                st.markdown(f"- {line}")
        with st.expander("Per season, and the settings"):
            st.dataframe(ev.decision_frame(decision_).drop(columns="key").style.format(precision=1), hide_index=True,
                         width="stretch")
            st.caption(f"Settings: {decision_['units']} units, k = {decision_['k']:g}, at most "
                       f"{decision_['max_share']:.0%} per district. {decision_.get('interval_method', '')}")
        st.caption(decision_["score_note"])

    st.subheader("Which forecast ranks districts best?")
    risk_ = load_risk_metrics()
    if risk_ is None:
        st.info("No fire-risk backtest found. Run `uv run python -m psw.pipeline risk-train`.")
    else:
        for line in interpret_risk(risk_):
            st.markdown(f"- {line}")
        if decision_ and (line := ev.interpret_ranking(decision_)):
            st.markdown(f"- {line}")
        with st.expander("The backtest table"):
            st.write(f"Target: {risk_['target']}. For where to send help, the order of districts matters more than "
                     "the exact counts.")
            rf = ev.risk_frame(risk_)
            higher = [c for c in rf.columns if c.startswith("top-5") or c == "Spearman"]
            st.dataframe(
                rf.style.apply(highlight_best, lower=["MAE (detections)", "pinball"], target={"80% coverage (%)": 80.0},
                               axis=None)
                .highlight_max(subset=higher, props=BEST)
                .format({c: "{:.3f}" if c == "pinball" else "{:.2f}" if c in ("MAE (detections)", "Spearman")
                         else "{:.1f}" for c in rf.columns[1:]}),
                hide_index=True, width="stretch")
            st.caption(f"MAE and pinball are over every district and day. Top-5 hit rate and Spearman are over days "
                       f"with enough burning to rank ({risk_['ranking_days_rule']}). Picking 5 districts blindly would "
                       f"score about {ev.random_hit(len(districts_)):.0%}. Green marks the best value per column.")

# ---------- agents ----------
with t_agents:
    st.markdown(LLM_RULE)
    st.subheader("The agents")
    st.dataframe(pd.DataFrame([{"agent": name, "type": "rules, no LLM" if kind == "rules" else kind, "job": role}
                               for name, (kind, role) in AGENTS.items()]), hide_index=True, width="stretch")
    st.subheader("How often the saved briefs pass")
    rel = reliability()
    if rel.empty:
        st.info("No saved briefings were found in `data/processed/`.")
    else:
        st.dataframe(rel.style.format({"pass rate (%)": "{:.0f}", "mean words": "{:.0f}"}), hide_index=True,
                     width="stretch")
        if rel["verified"].sum() != rel["saved"].sum():
            st.warning(f"{int(rel['saved'].sum() - rel['verified'].sum())} saved brief(s) did not pass the Verifier.")
        st.caption("`agent`: the first draft passed. `agent_revised`: it passed after one revision with the error list. "
                   "`template`: the agent failed or was unavailable and a fixed template was used. **These are the "
                   "saved outputs, after reruns and after fixes to the checks. They are not the chance that a new run "
                   "passes.**")
    st.subheader("What the checks cannot catch")
    st.markdown(
        "- The Verifier checks that each number appears in a tool output or retrieved chunk, that each citation was "
        "retrieved, and the length and banned content. **It does not check that a cited source supports the "
        "sentence**, and it cannot tell \"rose\" from \"fell\".\n"
        "- For dispatch briefs it also checks units, rain and cells against the right district and rejects names below "
        "a district and enforcement talk. It still cannot catch a claim made without a number, or a name that is also "
        "an ordinary word.\n"
        "- The Critic is weak on its own: it has flagged citations that do support their sentence and missed ones that "
        "do not. Read the sources.\n"
        "- The Sentinel flags a district on a smoggy day because of the nearest city's forecast, even when that "
        "district has no fires, so a flag is not by itself a sign of burning.")
    st.subheader("How each agent works")
    st.code(
        "Code computes the facts (rules, models, allocator)\n"
        "        |\n"
        "        v\n"
        "(optional) Planner (LLM, no tools)  ->  a short plan from labels only, never numbers\n"
        "        |\n"
        "        v\n"
        "Writer agent (LLM + tools)  ->  Analyst, Auditor or Dispatcher; calls data tools and search_knowledge\n"
        "        |\n"
        "        v\n"
        "Verifier (rules, no LLM)  ->  every number in the tool outputs? every citation retrieved? banned content?\n"
        "        |  fail: one revision with the error list;  fail again: fixed template\n"
        "        v\n"
        "Critic (LLM, audit and dispatch)  ->  may only delete, reorder or reword; re-verified\n",
        language="text")
    with st.expander("Analyst (daily district briefing): tools and system prompt"):
        tool_list(tools.TOOL_SCHEMAS)
        st.code(analyst.SYSTEM_PROMPT, language="text")
    with st.expander("Auditor: tools"):
        tool_list(audit_tools.TOOL_SCHEMAS)
    with st.expander("Dispatcher: tools"):
        st.caption("The allocation is computed by code; the Dispatcher explains it and cannot change it.")
        tool_list(dispatch_tools.TOOL_SCHEMAS)

# ---------- PM2.5 forecast ----------
with t_pm:
    mf = metrics_frame(m)
    st.subheader("Do fire features help the next-day PM2.5 forecast?")
    for line in interpret(m):
        st.markdown(f"- {line}")
    c1, c2 = st.columns(2)
    with c1:
        best = mf["MAE (µg/m³)"].idxmin()
        fig = go.Figure(go.Bar(x=[VARIANT_LABEL[v] for v in VARIANTS], y=mf["MAE (µg/m³)"],
                               marker_color=["#2166ac" if i == best else "#b2b2b2" for i in range(len(mf))],
                               text=[f"{v:.2f}" for v in mf["MAE (µg/m³)"]], textposition="outside"))
        fig.update_layout(title="MAE by variant (lower is better)", yaxis_title="MAE (µg/m³)", height=360,
                          margin={"l": 10, "r": 10, "t": 50, "b": 10}, yaxis_range=[0, mf["MAE (µg/m³)"].max() * 1.2])
        st.plotly_chart(fig, width="stretch")
    with c2:
        imp = importance_frame(m).iloc[::-1]
        fig = go.Figure(go.Bar(x=imp["gain"], y=imp["feature"], orientation="h",
                               marker_color=["#d6604d" if f else "#b2b2b2" for f in imp["fire feature"]]))
        fig.update_layout(title="Feature importance (gain; red = fire feature)", height=360,
                          margin={"l": 10, "r": 10, "t": 50, "b": 10})
        st.plotly_chart(fig, width="stretch")
    with st.expander("Backtest tables"):
        st.write(f"Expanding window: each test season ({', '.join(str(s) for s in m['seasons_test'])}) uses a model "
                 "trained only on earlier seasons. Green marks the best value in each column.")
        st.dataframe(
            mf.style.apply(highlight_best, lower=["MAE (µg/m³)", "pinball", "band width (µg/m³)"],
                           target={"80% coverage (%)": 80.0}, axis=None)
            .format({"MAE (µg/m³)": "{:.2f}", "pinball": "{:.2f}", "80% coverage (%)": "{:.1f}",
                     "band width (µg/m³)": "{:.1f}"}),
            hide_index=True, width="stretch")
        city_score = st.radio("By city", ["mae", "pinball", "coverage80", "band_width"], horizontal=True,
                              format_func=lambda s: {"mae": "MAE", "pinball": "Pinball", "coverage80": "80% coverage",
                                                     "band_width": "Band width"}[s])
        bc = by_city_frame(m, city_score)
        if city_score == "coverage80":
            bc = bc * 100
            styled = bc.style.apply(lambda r: [BEST if abs(v - 80) == (r - 80).abs().min() else "" for v in r],
                                    axis=1).format("{:.1f}")
        else:
            styled = bc.style.highlight_min(axis=1, props=BEST).format("{:.2f}")
        st.dataframe(styled, width="stretch")
    st.caption("The target is CAMS modelled PM2.5 (via Open-Meteo), not a ground monitor. Feature importance shows what "
               "the final model used, not what causes PM2.5. Settings were not tuned on the test seasons.")

# ---------- retrieval ----------
with t_rag:
    st.subheader("Embeddings, keywords and hybrid")
    rep_ = load_retrieval_report()
    if rep_ is None:
        st.info("No retrieval evaluation found. Run `uv run python -m psw.pipeline kb-eval`.")
    else:
        st.write("The knowledge base is searched with embeddings (meaning), BM25 (exact words) and a hybrid of both "
                 "merged by reciprocal rank fusion. Every mode applies the same `as_of` filter, so none of them can "
                 "see the future.")
        st.dataframe(retrieval_frame(rep_).style.format({"MRR": "{:.2f}"}), hide_index=True, width="stretch")
        for line in interpret_retrieval(rep_):
            st.markdown(f"- {line}")
        with st.expander("Rank of the right document, per query (1 = best, - = not in the top 3)"):
            st.dataframe(retrieval_queries(rep_).astype(str), hide_index=True, width="stretch")

# ---------- sources ----------
with t_src:
    st.subheader("Limitations")
    st.markdown(LIMITATIONS)
    st.subheader("Data sources")
    st.markdown(
        "| Data | Source |\n|---|---|\n"
        "| Fire detections | NASA FIRMS, VIIRS S-NPP 375 m (archive product, low-confidence detections dropped) |\n"
        "| Other sensors (audit) | NASA FIRMS, VIIRS NOAA-20 and NOAA-21, MODIS Terra and Aqua |\n"
        "| Burned area (audit) | MODIS MCD64A1 monthly burn dates, University of Maryland |\n"
        "| Official fire counts | Punjab Pollution Control Board reports (satellite-based), address per row in "
        "`data/manual/official_counts.csv` |\n"
        "| Machine figures (context only) | State status report of 26 Nov 2024, address in "
        "`data/manual/crm_machines.csv` |\n"
        "| PM2.5 | Copernicus Atmosphere Monitoring Service (CAMS) via the [Open-Meteo](https://open-meteo.com/) "
        "air-quality API, CC BY 4.0 |\n"
        "| Weather | [Open-Meteo](https://open-meteo.com/) historical weather (ERA5 reanalysis), CC BY 4.0 |\n"
        "| District boundaries | geoBoundaries IND ADM2 (gbOpen), licence ODbL 1.0, attribution to geoBoundaries "
        "(wmgeolab) |\n"
        "| Embeddings for the knowledge base | `BAAI/bge-small-en-v1.5` run locally with fastembed |\n"
        f"| PM2.5 reference line | India NAAQS 24-hour standard, {config.NAAQS_PM25_24H} µg/m³ |")
    st.caption(CREDITS)
