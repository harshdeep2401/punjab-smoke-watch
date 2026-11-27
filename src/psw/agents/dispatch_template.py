"""Deterministic fallback dispatch brief: fixed sentences filled from the same tool results the Dispatcher sees.

Every number is copied from the tool results and every unit figure sits beside its district, so it passes the Verifier
by construction (tests check this for every forecast date). It cites the gap-method note, which the orchestrator reads
without embeddings.
"""
from __future__ import annotations

from psw import config

LABEL = "template (LLM unavailable or failed verification)"
WHERE, WHY, SURE, WATCH = config.DISPATCH_LABELS
MIN_WORDS = config.DISPATCH_WORDS[0] + 5

PADDING = [
    "Check the plan against what field staff report.",
    "Recompute the plan each day as new detections and weather arrive.",
    "The forecast covers satellite detections, which miss fires burning between passes or under cloud.",
    "The plan stops at districts and 5 km cells and names no person or farm.",
    "Custom hiring centres and balers are ways to deal with straw without burning it.",
    "Machine numbers available locally should be confirmed before units are moved.",
    "A district with few units may still need help, so local requests should be heard.",
]


def _units(n: int) -> str:
    return f"{n} unit" if n == 1 else f"{n} units"


def _list(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _where(a: dict) -> str:
    if not a["districts"]:
        return (f"{WHERE}: no burning is forecast in any district over the next 3 days, so none of the "
                f"{_units(a['units_available'])} are placed. Teams can use the time to prepare.")
    lead = [f"{d['district']} ({_units(d['units'])})" for d in a["districts"][:3]]
    text = (f"{WHERE}: of {_units(a['units_available'])}, the largest shares go to {_list(lead)}. No district may take "
            f"more than {_units(a['limit_per_district'])}.")
    if a.get("units_not_placed"):
        text += f" {_units(a['units_not_placed'])} are not placed."
    return text


def _forecast_words(f: dict) -> str:
    text = (f"{f['forecast_detections']} detections forecast (80% band {f['band_80_low']} to {f['band_80_high']}), "
            f"against {f['detections_last_3_days']} in the last 3 days")
    return text


def _why(a: dict, r: dict) -> str:
    lead = r["top_districts"][0]
    basis = "likely burning" if a["planned_on"] == "likely burning" else "forecast detections"
    out = [f"{WHY}: the split follows {basis} over the next 3 days."]
    if "likely_burning" in lead:
        out.append(f"{lead['district']} leads with likely burning of {lead['likely_burning']}; it has "
                   f"{_forecast_words(lead)}.")
    else:
        out.append(f"{lead['district']} leads, with {_forecast_words(lead)}.")
    moves = a.get("moved_by_the_likely_adjustment") or []
    if moves:
        m = moves[0]
        out.append(f"The adjustment moves {m['district']} from {_units(m['units_without_adjustment'])} to "
                   f"{_units(m['units_with_adjustment'])}, the biggest change of any district.")
    return " ".join(out)


def _sure(a: dict, gap: dict | None, cite: str) -> str:
    out = [f"{SURE}: the forecast is uncertain and its band is wide."]
    if a["planned_on"] == "likely burning":
        conf = gap.get("confidence") if gap and "error" not in gap else None
        moves = a.get("moved_by_the_likely_adjustment") or []
        conf = moves[0]["confidence_of_its_gap_range"] if moves else conf
        about = moves[0]["district"] if moves else a["districts"][0]["district"] if a["districts"] else None
        out.append(f"Likely burning scales each forecast by the previous season's detection gap; it is not a count of "
                   f"fires and cannot be checked against satellite data.{cite}")
        if conf and about:
            out.append(f"The range behind {about} has {conf} confidence.")
    else:
        out.append(f"This plan uses forecast detections only and is not adjusted for the detection gap, so burning that "
                   f"satellites miss is not counted.{cite}")
    out.append("What one unit achieves is assumed, not measured.")
    return " ".join(out)


def _cell_words(h: dict) -> str:
    if not h["cells"]:
        return f"No cell in {h['district']} had a cluster of detections in the last 7 days."
    c = h["cells"][0]
    return f"Awareness teams could start at {c['cell']}, with {c['detections']} detections in the last 7 days."


def _watch(rain: dict, hotspots: dict) -> str:
    return (f"{WATCH}: {rain['district']} had {rain['rain_last_3_days_mm']} mm of rain in the last 3 days, and for "
            f"the next 3 the weather archive, standing in for a forecast, shows {rain['rain_next_3_days_mm']} mm; rain "
            f"can delay burning. {_cell_words(hotspots)}")


def _paragraph(a: dict, r: dict, rain: dict, hotspots: dict) -> str:
    f, mine = r["asked_district"], a["asked_district"]
    likely = f" Likely burning is {f['likely_burning']}." if "likely_burning" in f else ""
    return (f"This is {f['district']} ({_units(mine['units'])}). It has {_forecast_words(f)}.{likely} Rain was "
            f"{rain['rain_last_3_days_mm']} mm in the last 3 days; for the next 3 the archive, standing in for a forecast, "
            f"shows {rain['rain_next_3_days_mm']} mm. {_cell_words(hotspots)} These figures describe the district only, never a farm.")


def render(facts: dict) -> dict:
    """facts: {risk, allocation, gap, rain, hotspots, method, district_rain?, district_hotspots?} (tool results).

    Returns {"headline", "briefing", "citations", "label"} and "district_paragraph" when a district was asked for.
    """
    method = (facts.get("method") or {}).get("results") or []
    cite_id = method[0]["chunk_id"] if method else None
    a, r = facts["allocation"], facts["risk"]
    parts = [_where(a), _why(a, r), _sure(a, facts.get("gap"), f" [{cite_id}]" if cite_id else ""),
             _watch(facts["rain"], facts["hotspots"])]
    for pad in PADDING:
        if sum(len(p.split()) for p in parts) >= MIN_WORDS:
            break
        parts[3] += " " + pad
    leader = a["districts"][0]["district"] if a["districts"] else "no district"
    headline = f"3-day plan from {a['date']}: {leader} first" if a["districts"] else f"3-day plan from {a['date']}: quiet"
    out = {"headline": headline, "briefing": "\n\n".join(parts), "citations": [cite_id] if cite_id else [], "label": LABEL}
    if "asked_district" in a:
        out["district_paragraph"] = _paragraph(a, r, facts["district_rain"], facts["district_hotspots"])
    return out
