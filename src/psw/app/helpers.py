"""Pure helpers for the Phase 6 pages (no Streamlit import, so they are unit-testable).

Everything the pages say about numbers is computed here from `metrics.json` or from saved briefings.
"""
from __future__ import annotations

import json
from collections import Counter
from datetime import date as Date
from pathlib import Path

import pandas as pd

from psw import config
from psw.agents.critic import base_mode
from psw.agents.orchestrator import briefing_path
from psw.agents.verifier import Evidence, _walk, extract_numbers

VARIANTS = ["full", "no_fire", "persistence"]
VARIANT_LABEL = {"full": "full (with fire features)", "no_fire": "no fire features", "persistence": "persistence (today = tomorrow)"}
SAMPLE_DIR = "sample_run"
FIRE_FEATURE_HINTS = ("fire", "frp", "upwind")


# ---------- dates and cached briefings ----------

def peak_day(daily: pd.DataFrame, season: int | None = None) -> Date:
    """The day with the most detections in Punjab in `season` (default: the latest season). The demo default date."""
    season = int(daily["season"].max()) if season is None else season
    totals = daily[daily["season"] == season].groupby("date")["fire_count"].sum()
    day = totals.idxmax()
    return day if isinstance(day, Date) else Date.fromisoformat(str(day)[:10])


def find_briefing(date: str, district: str, root: Path | None = None) -> dict | None:
    """The saved briefing for a district-day: `briefings/` first, then `sample_run/briefings/`. None if there is none."""
    root = Path(root or config.PROCESSED)
    for base in (root, root / SAMPLE_DIR):
        p = briefing_path(str(date), district, base)
        if p.exists():
            try:
                return json.loads(p.read_text())
            except (OSError, json.JSONDecodeError):
                return None  # a half-written file must not crash the page
    return None


def cached_districts(date: str, root: Path | None = None) -> set[str]:
    """Districts that have a saved briefing for `date`."""
    root = Path(root or config.PROCESSED)
    out: set[str] = set()
    for base in (root, root / SAMPLE_DIR):
        for p in (base / "briefings").glob(f"{date}_*.json"):
            try:
                out.add(json.loads(p.read_text())["district"])
            except (OSError, json.JSONDecodeError, KeyError):
                continue
    return out


def badge(rec: dict) -> tuple[str, str]:
    """(kind, text) for the verification badge. kind: success | warning | error."""
    ok = bool(rec.get("verification", {}).get("ok"))
    mode = base_mode(rec.get("mode"))
    if not ok:
        return "error", "❌ Verification failed"
    if mode == "template":
        return "warning", "⚠️ Template fallback (numbers verified)"
    if mode == "agent_revised":
        return "success", "✅ Numbers verified (after one revision)"
    return "success", "✅ Numbers verified"


# ---------- the Verifier report: number -> evidence value ----------

def _tool_numbers(result) -> list[float]:
    """Numbers one tool result could vouch for (same walk as Evidence.from_tool_results)."""
    if isinstance(result, dict) and "error" in result:
        return []
    out: list[float] = []
    for leaf in _walk(result):
        if isinstance(leaf, bool) or leaf is None:
            continue
        if isinstance(leaf, (int, float)):
            out.append(abs(float(leaf)))
        elif isinstance(leaf, str):
            out += [n.value for n in extract_numbers(leaf)]
    return out


def explain_numbers(rec: dict) -> pd.DataFrame:
    """One row per number written in the headline and briefing: the number, the evidence value it matched, and where.

    Re-runs the Verifier's matching rule (`Evidence.matches`) against each tool's output on its own.
    """
    date = Date.fromisoformat(str(rec["date"]))
    facts = rec.get("facts", {})
    pools = {name: sorted(set(_tool_numbers(res))) for name, res in facts.items()}
    constants = {float(c): "constant (7-day, 80% interval, p10/p50/p90, 24-hour)" for c in config.VERIFY_CONSTANTS}
    constants |= {float(date.year): "briefing date", float(date.month): "briefing date", float(date.day): "briefing date"}

    rows = []
    for label, text in (("headline", rec.get("headline", "")), ("briefing", rec.get("briefing", ""))):
        for n in extract_numbers(text):
            best: tuple[float, str] | None = None
            for tool, nums in pools.items():
                for a in nums:
                    if Evidence(allowed_numbers=[a]).matches(n.value, n.decimals) and (best is None or abs(a - n.value) < abs(best[0] - n.value)):
                        best = (a, tool)
            if best is None:
                for a, why in constants.items():
                    if Evidence(allowed_numbers=[a]).matches(n.value, n.decimals):
                        best = (a, why)
                        break
            shown = str(int(n.raw)) if n.raw.isdigit() else n.raw
            rows.append({"in": label, "number": shown,
                         "matched value": best[0] if best else None,
                         "source": best[1] if best else "NOT FOUND"})
    return pd.DataFrame(rows, columns=["in", "number", "matched value", "source"])


def retrieved_chunks(rec: dict) -> list[dict]:
    """All chunks the run retrieved (not only the cited ones), each marked as cited or not."""
    cited = set(rec.get("citations", []))
    res = rec.get("facts", {}).get("search_knowledge", {})
    chunks = res.get("results", []) if isinstance(res, dict) else []
    return [{**c, "cited": c.get("chunk_id") in cited} for c in chunks]


# ---------- Evidence page ----------

def metrics_frame(m: dict) -> pd.DataFrame:
    """Backtest table: one row per variant."""
    rows = []
    for v in VARIANTS:
        o = m["variants"][v]["overall"]
        rows.append({"variant": VARIANT_LABEL[v], "n": int(o["n"]), "MAE (µg/m³)": o["mae"], "pinball": o["pinball"],
                     "80% coverage (%)": o["coverage80"] * 100, "band width (µg/m³)": o["band_width"]})
    return pd.DataFrame(rows)


def by_city_frame(m: dict, score: str = "mae") -> pd.DataFrame:
    """City × variant for one score (mae, pinball, coverage80, band_width)."""
    cities = sorted(m["variants"]["full"]["by_city"])
    return pd.DataFrame({VARIANT_LABEL[v]: [m["variants"][v]["by_city"][c][score] for c in cities] for v in VARIANTS},
                        index=pd.Index(cities, name="city"))


def importance_frame(m: dict, top: int = 12) -> pd.DataFrame:
    s = pd.Series(m["feature_importance"]).sort_values(ascending=False).head(top)
    return pd.DataFrame({"feature": s.index, "gain": s.values,
                         "fire feature": [any(h in f for h in FIRE_FEATURE_HINTS) for f in s.index]})


def _seasons_text(seasons: list[int]) -> str:
    return " and ".join(str(s) for s in seasons) if len(seasons) <= 2 else f"{seasons[0]} to {seasons[-1]}"


def interpret(m: dict) -> list[str]:
    """An honest reading of the backtest, written only from numbers in metrics.json. Wording follows the result."""
    ov = {v: m["variants"][v]["overall"] for v in VARIANTS}
    full, nof, pers = ov["full"], ov["no_fire"], ov["persistence"]
    n_cities = len(m["variants"]["full"]["by_city"])
    out = [f"Each variant was scored on {full['n']} next-day forecasts ({n_cities} cities, test seasons {_seasons_text(m['seasons_test'])}), "
           f"always trained on earlier seasons only."]

    best = min(VARIANTS, key=lambda v: ov[v]["mae"])
    out.append(f"The lowest MAE is {VARIANT_LABEL[best]}: {ov[best]['mae']:.2f} µg/m³ "
               f"(full {full['mae']:.2f}, no fire features {nof['mae']:.2f}, persistence {pers['mae']:.2f}).")

    d = nof["mae"] - full["mae"]
    if d > 0:
        out.append(f"Fire features lowered MAE from {nof['mae']:.2f} to {full['mae']:.2f} µg/m³ (by {d:.2f}) "
                   f"and pinball loss from {nof['pinball']:.2f} to {full['pinball']:.2f}.")
    else:
        out.append(f"**Fire features did not improve the forecast:** MAE is {full['mae']:.2f} with them and {nof['mae']:.2f} "
                   f"without, and pinball loss is {full['pinball']:.2f} against {nof['pinball']:.2f}.")

    if pers["mae"] < full["mae"]:
        out.append(f"**Persistence (tomorrow = today) beat the model on point error** ({pers['mae']:.2f} vs {full['mae']:.2f} µg/m³ MAE).")
    else:
        out.append(f"The model beat persistence on MAE ({full['mae']:.2f} vs {pers['mae']:.2f} µg/m³).")

    out.append(f"The model's 80% band covered {full['coverage80'] * 100:.1f}% of actual values (band width {full['band_width']:.1f} µg/m³); "
               f"persistence's band covered {pers['coverage80'] * 100:.1f}% (width {pers['band_width']:.1f}). The target is 80%.")

    conf = m.get("conformal", {})
    inner = [f["full"]["inner_coverage"] * 100 for f in conf.values() if f.get("full", {}).get("applied")]
    corr = [f["full"]["correction"] for f in conf.values() if f.get("full", {}).get("applied")]
    if inner:
        out.append(f"Before the conformal correction, the raw 80% band covered only {min(inner):.0f} to {max(inner):.0f}% of the last "
                   f"training season, so it was widened by {min(corr):.1f} to {max(corr):.1f} µg/m³ in each fold.")

    top = list(pd.Series(m["feature_importance"]).sort_values(ascending=False).head(3).index)
    out.append("The three features with the most gain are " + ", ".join(f"`{t}`" for t in top) + ".")
    return out


def reliability(root: Path | None = None) -> pd.DataFrame:
    """Verifier pass rate and mode counts over saved briefings, split by folder (demo set vs sampled run)."""
    root = Path(root or config.PROCESSED)
    rows = []
    for name, base in (("demo briefings", root / "briefings"), ("sampled run", root / SAMPLE_DIR / "briefings")):
        recs = []
        for p in sorted(base.glob("*.json")):
            try:
                recs.append(json.loads(p.read_text()))
            except (OSError, json.JSONDecodeError):
                continue
        if not recs:
            continue
        modes = Counter(base_mode(r.get("mode")) for r in recs)
        verified = sum(bool(r.get("verification", {}).get("ok")) for r in recs)
        words = [len(r.get("briefing", "").split()) for r in recs]
        rows.append({"set": name, "briefings": len(recs), "verified": verified,
                     "pass rate (%)": 100 * verified / len(recs),
                     "agent": modes.get("agent", 0), "agent_revised": modes.get("agent_revised", 0),
                     "template": modes.get("template", 0),
                     "first draft failed": sum(bool(r.get("first_attempt_errors")) for r in recs),
                     "mean words": sum(words) / len(words),
                     "models": ", ".join(sorted({str(r.get("model")) for r in recs if r.get("mode") != "template"})) or "none"})
    return pd.DataFrame(rows)


# ---------- retrieval eval (hybrid search) ----------

RETRIEVAL_LABEL = {"hybrid": "hybrid (RRF)", "embedding": "embeddings only", "bm25": "BM25 only"}


def load_retrieval_report(root: Path | None = None) -> dict | None:
    try:
        return json.loads((Path(root or config.PROCESSED) / "retrieval_eval.json").read_text())
    except (OSError, json.JSONDecodeError):
        return None


def retrieval_frame(rep: dict) -> pd.DataFrame:
    k, n = rep["k"], rep["n"]
    return pd.DataFrame([{"retrieval": RETRIEVAL_LABEL.get(m, m), f"hit@{k}": f"{v['hit_at_k']} / {n}", "hit@1": f"{v['hit_at_1']} / {n}",
                          "MRR": v["mrr"]} for m, v in rep["modes"].items()])


def retrieval_queries(rep: dict) -> pd.DataFrame:
    """One row per query with the rank of the first right chunk in each mode ('-' = not in the top k)."""
    modes = list(rep["modes"])
    first = rep["modes"][modes[0]]["items"]
    rows = []
    for i, it in enumerate(first):
        row = {"query": it["query"], "kind": it["kind"], "expected": ", ".join(it["expected"])}
        for m in modes:
            r = rep["modes"][m]["items"][i]["rank"]
            row[RETRIEVAL_LABEL.get(m, m)] = "-" if r is None else r
        rows.append(row)
    return pd.DataFrame(rows)


def interpret_retrieval(rep: dict) -> list[str]:
    """What the retrieval eval does and does not show, from the report's own numbers."""
    k, n, ms = rep["k"], rep["n"], rep["modes"]
    h, e, b = ms["hybrid"], ms["embedding"], ms["bm25"]
    out = [f"{n} queries, each with the document that should come back. A query is a hit at {k} if any of the top {k} chunks is from that document."]
    if h["hit_at_k"] == e["hit_at_k"] == n:
        out.append(f"**This set is too easy to separate the methods at hit@{k}:** embeddings alone already find the right document for all {n} queries. "
                   f"The differences show only at hit@1 (hybrid {h['hit_at_1']}, embeddings {e['hit_at_1']}, BM25 {b['hit_at_1']}) and in MRR "
                   f"({h['mrr']:.2f}, {e['mrr']:.2f}, {b['mrr']:.2f}).")
    else:
        out.append(f"hit@{k}: hybrid {h['hit_at_k']}, embeddings {e['hit_at_k']}, BM25 {b['hit_at_k']} out of {n}.")
    better = [i for i, (x, y) in enumerate(zip(h["items"], e["items"]))
              if (x["rank"] or 99) < (y["rank"] or 99)]
    worse = [i for i, (x, y) in enumerate(zip(h["items"], e["items"]))
             if (x["rank"] or 99) > (y["rank"] or 99)]
    out.append(f"Against embeddings alone, hybrid ranked the right document higher on {len(better)} of {n} queries and lower on {len(worse)}.")
    missed = [x["query"] for x in b["items"] if x["rank"] is None]
    if missed:
        out.append("BM25 alone missed " + "; ".join(f"“{q}”" for q in missed) + ". It matches whole words only, with no stemming, "
                   "so a paraphrase that shares no exact word gets nothing from it.")
    out.append(f"The queries were written by us after reading the corpus and there are only {n}, so treat this as a sanity check, not a benchmark. "
               "Keyword search is expected to matter most for exact names, years and terms, which is why the Analyst is told to put them in its queries.")
    return out


# ---------- Part 2: fire-risk forecast ----------

RISK_LABEL = {"model": "LightGBM model", "persistence": "persistence (the last 3 days)",
              "climatology": "climatology (last season, scaled)"}


def load_risk_metrics(root: Path | None = None) -> dict | None:
    path = Path(root or config.PROCESSED) / "risk_metrics.json"
    return json.loads(path.read_text()) if path.exists() else None


def interpret_risk(m: dict) -> list[str]:
    """Which fire-risk variant was selected and what that does not show, from risk_metrics.json."""
    ov = {v: s["overall"] for v, s in m["variants"].items()}
    best = m["best_ranking_variant"]
    scores = "; ".join(f"{RISK_LABEL[v]} {o['top5_hit']:.1%} and {o['spearman']:.2f}" for v, o in ov.items())
    out = [f"**Best-ranking variant, the one to allocate with: {RISK_LABEL[best]}.** Rule: {m['best_ranking_rule']} "
           f"({_seasons_text(m['seasons_test'])}).",
           f"Top-5 hit rate and Spearman rank correlation: {scores}. Scored on {ov[best]['n_days']} days "
           f"({m['ranking_days_rule']})."]
    lowest = min(ov, key=lambda v: ov[v]["mae"])
    if lowest == best:
        out.append(f"It also has the lowest MAE ({ov[best]['mae']:.2f} detections per district over 3 days).")
    else:
        out.append(f"**It is not the most accurate:** {RISK_LABEL[lowest]} has the lower MAE "
                   f"({ov[lowest]['mae']:.2f} against {ov[best]['mae']:.2f} detections per district over 3 days).")
    out.append("The variant was picked on the same test seasons it is scored on, so its lead may be flattering.")
    if best == "model":
        out.append(m["rain_assumption"])
    return out
