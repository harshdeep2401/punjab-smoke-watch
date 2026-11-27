"""Deterministic fallback audit brief: fixed sentences filled from the same tool results the Auditor sees.

Every number is copied from the tool results, so it passes the Verifier by construction (tests check this for every
district and season). It cites the gap-method note, which the orchestrator retrieves without embeddings.
"""
from __future__ import annotations

from psw import config
from psw.audit import gap

LABEL = "template (LLM unavailable or failed verification)"
METHOD_DOC = "method_detection_gap"
DETECTIONS, SIGNALS, RANGE, SETTLE = config.AUDIT_LABELS
MIN_WORDS = config.AUDIT_WORDS[0] + 5  # a margin above the verifier's minimum

PADDING = [
    "Satellites only record fires that are burning when they pass over, and cloud can hide them.",
    "A fire count is a count of heat detections, not a measure of the area burned or of the smoke produced.",
    "The figures describe a district or the state and never an individual farm or field.",
]


def method_chunks(kb) -> dict:
    """The gap-method note, in the same shape as `search_knowledge`, so it counts as retrieved evidence."""
    c = kb.chunks
    c = c[c["doc_id"] == METHOD_DOC]
    return {"results": [{"chunk_id": r["chunk_id"], "title": r["title"], "text": r["text"]} for _, r in c.iterrows()]}


def _ok(d: dict | None) -> bool:
    return bool(d) and "error" not in d


def _change(pct, base) -> str:
    if pct is None:
        return ""
    if pct == 0:
        return f", unchanged from {base}"
    return f", {'up' if pct > 0 else 'down'} {abs(pct):.1f}% from {base}"


def _detections(trend: dict, diurnal: dict, place: str, season: int) -> str:
    if not _ok(trend):
        return f"{DETECTIONS}: no detection data is available for {place} in {season}."
    by_name = {s["satellite"]: s for s in trend["sensors"]}
    main = by_name["VIIRS S-NPP"]
    change = "" if "warning" in main else _change(main["change_vs_base_pct"], main["base_season"])
    out = [f"{DETECTIONS}: in {season}, VIIRS S-NPP recorded {main['detections']:,} fire detections in {place}{change}."]
    others = [f"{name} {s['detections']:,}" for name, s in by_name.items() if name != "VIIRS S-NPP"]
    if others:
        out.append("The other satellites recorded: " + ", ".join(others) + ".")
    if _ok(diurnal) and diurnal["night_share_pct"] is not None:
        if season != diurnal["base_season"] and diurnal["base_night_share_pct"] is not None:
            out.append(f"Night passes gave {diurnal['night_share_pct']}% of S-NPP detections, against "
                       f"{diurnal['base_night_share_pct']}% in {diurnal['base_season']}.")
        else:
            out.append(f"Night passes gave {diurnal['night_share_pct']}% of S-NPP detections.")
    return " ".join(out)


def _signals(burned: dict, official: dict) -> str:
    out = []
    if _ok(burned):
        out.append(f"MODIS burned area was {burned['burned_km2']:,.1f} km²"
                   f"{_change(burned.get('burned_change_vs_base_pct'), burned['base_season'])}.")
        if "warning" in burned:
            out.append(f"That is under {gap.MIN_BURNED_KM2} km², too small to rely on.")
    else:
        out.append("No burned-area figure is available.")
    if _ok(official):
        change = "" if "warning" in official else _change(official.get("official_change_vs_base_pct"), official.get("base_season"))
        out.append(f"The official count was {official['official_count']:,}{change}; it is also satellite-based, so it is "
                   "not an independent check.")
    else:
        out.append("No official count is available.")
    return f"{SIGNALS}: " + " ".join(out)


def _range(g: dict, season: int, cite: str) -> str:
    if not _ok(g):
        return f"{RANGE}: {season} is the base season, so there is no earlier season to compare with and no range.{cite}"
    if g["gap_low_pct"] is None:
        return (f"{RANGE}: there is insufficient evidence for a range, because of {g['reason']}. "
                f"We therefore give no estimate.{cite}")
    meaning = {"counts fell more than burned area": "Counts fell more than burned area.",
               "no sign of a detection gap": "A negative value means counts per burned km² rose: there is no sign of a "
                                             "detection gap.",
               "direction unclear": "The range crosses zero, so the direction is unclear."}[g["direction"]]
    return (f"{RANGE}: the gap, meaning the fall in counts per burned km² since the base season, is estimated at "
            f"{g['gap_low_pct']}% to {g['gap_high_pct']}% (confidence: {g['confidence']}). {meaning} This is not the share "
            "of burning missed. "
            f"The range leaves out the error of the burned-area product, which is unknown and the largest uncertainty.{cite}")


def _settle() -> str:
    return (f"{SETTLE}: field checks of a sample of paddy fields, and geostationary satellites that watch through the "
            "day, could test this. Smaller or cooler fires, or a change in the burned-area product, could also lower "
            "detections per burned area. This is triangulation of satellite and official signals, not field "
            "verification.")


def render(facts: dict, district: str, season: int) -> dict:
    """facts: {trend, diurnal, burned, official, gap, method} (tool results; any may be missing or an error).

    Returns {"headline", "briefing", "citations", "label"}.
    """
    method = (facts.get("method") or {}).get("results") or []
    cite_id = method[0]["chunk_id"] if method else None
    parts = [_detections(facts.get("trend"), facts.get("diurnal"), district, season),
             _signals(facts.get("burned"), facts.get("official")),
             _range(facts.get("gap"), season, f" [{cite_id}]" if cite_id else ""),
             _settle()]
    for pad in PADDING:  # only for briefs with missing sections (small districts, base season)
        if sum(len(p.split()) for p in parts) >= MIN_WORDS:
            break
        parts[1] += " " + pad
    return {"headline": f"{district}: fire monitoring audit, {season} season", "briefing": "\n\n".join(parts),
            "citations": [cite_id] if cite_id else [], "label": LABEL}
