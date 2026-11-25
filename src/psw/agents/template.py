"""Deterministic fallback briefing. Fixed sentences filled from the same tool results the Analyst would see.

Every number it writes is copied from `facts`, so it passes the Verifier by construction (tests check this on every
district-day of the test seasons). It cites the limitations chunk, which the orchestrator retrieves deterministically.
"""
from __future__ import annotations

from datetime import date as Date

from psw import config

LABEL = "template (LLM unavailable or failed verification)"
MIN_WORDS = config.BRIEFING_WORDS[0] + 8  # a margin above the verifier's minimum

PADDING = [
    "Fire counts are satellite detections from the VIIRS S-NPP instrument, and fire radiative power measures how much heat the detected fires release.",
    "Treat these figures as a daily indicator of burning activity and not as a measure of the smoke that reaches any one place.",
    "A larger fire radiative power value usually means more or hotter burning.",
]


def _d(iso: str) -> str:
    d = Date.fromisoformat(iso)
    return f"{d.day} {d.strftime('%B')} {d.year}"


def _f(x: float) -> str:
    return f"{x:,.1f}"


def _ok(d: dict | None) -> bool:
    return bool(d) and "error" not in d


def render(facts: dict) -> dict:
    """facts: {snapshot, forecast, wind, reliability, limitations} (tool results; any may be missing or an error).

    Returns {"headline", "briefing", "citations", "label"}.
    """
    snap, fc, wind, rel = (facts.get(k) for k in ("snapshot", "forecast", "wind", "reliability"))
    lim = (facts.get("limitations") or {}).get("results") or []
    s = []

    s.append(f"On {_d(snap['date'])}, satellites detected {snap['fire_count']} fires in {snap['district']} district, "
             f"releasing {_f(snap['frp_sum_mw'])} MW of fire radiative power.")
    ch = snap.get("change_vs_7d_pct")
    if ch is not None:
        rel_word = "above" if ch >= 0 else "below"
        s.append(f"That is {_f(abs(ch))}% {rel_word} the 7-day average of {_f(snap['fire_count_7d_mean'])}.")
    s.append(f"{snap['district']} ranks number {snap['rank_in_punjab']} in Punjab, which recorded {snap['punjab_total']} detections in total.")

    if _ok(fc):
        s.append(f"The nearest city, {fc['city']}, has a forecast median PM2.5 of {_f(fc['p50'])} µg/m³ for {_d(fc['target_date'])}, "
                 f"with an 80% interval of {_f(fc['p10'])} to {_f(fc['p90'])}.")
        limit = config.NAAQS_PM25_24H
        hint = fc["exceeds_naaqs_60_prob_hint"]
        if hint.startswith("p50 above"):
            s.append(f"The median is above the national 24-hour standard of {limit} µg/m³.")
        elif hint.startswith("p90 above"):
            s.append(f"The median is below the national 24-hour standard of {limit} µg/m³, but the top of the interval is above it.")
        else:
            s.append(f"The whole interval is below the national 24-hour standard of {limit} µg/m³.")
    else:
        s.append("No forecast is available for this date.")

    if _ok(wind):
        top = wind.get("top_upwind_districts") or []
        shares = " and ".join(f"{t['district']} ({_f(t['share_pct'])}%)" for t in top[:2])
        tail = f", with the largest upwind fire shares from {shares}" if shares else ""
        s.append(f"Wind at {wind['city']} blows from the {wind['wind_dir_compass']} ({_f(wind['wind_dir_from_deg'])}°) "
                 f"at {_f(wind['wind_speed_kmh'])} km/h{tail}.")

    if _ok(rel):
        years = " and ".join(str(y) for y in rel["test_seasons"])
        verdict = "beat" if rel["mae"] < rel["mae_persistence"] else "did not beat"
        s.append(f"On the {years} test seasons the model's average error here was {_f(rel['mae'])} µg/m³, against "
                 f"{_f(rel['mae_persistence'])} for persistence, so it {verdict} that simple baseline.")

    for pad in PADDING:  # only when the optional sections were missing
        if sum(len(x.split()) for x in s) + 30 >= MIN_WORDS:
            break
        s.append(pad)

    cite = lim[0]["chunk_id"] if lim else None
    s.append("Limitation: satellites detect heat anomalies, not stubble burning itself, and miss fires under cloud, "
             "so counts are lower bounds; PM2.5 here is modelled, not measured." + (f" [{cite}]" if cite else ""))

    d = _d(snap["date"]).rsplit(" ", 1)[0]
    return {"headline": f"{snap['district']}: {snap['fire_count']} fire detections on {d}",
            "briefing": " ".join(s), "citations": [cite] if cite else [], "label": LABEL}
