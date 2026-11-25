"""Verifier: deterministic checks on an Analyst briefing. No LLM.

It enforces "the LLM never does the maths":
  * every number in the text must match a number in the tool results or the retrieved chunks (after rounding),
  * every citation must be a chunk that was actually retrieved, and there must be at least one,
  * length limits, and no health advice or personal names.

Known limit: it checks that a number *exists* in the evidence, not that the sentence uses it correctly
(for example, it cannot tell "rose 68%" from "fell 68%"). Signs are ignored when matching.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date as Date
from decimal import ROUND_HALF_UP, Decimal

from psw import config

ISO_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
CITATION = re.compile(r"\[?\bK\d{3}\b\]?")
INLINE_CITATION = re.compile(r"\[(K\d{3})\]")
# digits not glued to a letter, digit or '.', so PM2.5, PM10, p10, K012, NO2 and N95 are not numbers
NUMBER = re.compile(r"(?<![A-Za-z\d.])(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)")


@dataclass(frozen=True)
class Num:
    value: float
    decimals: int
    raw: str


def _half_up(x: float, decimals: int) -> Decimal:
    return Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-decimals), rounding=ROUND_HALF_UP)


def extract_numbers(text: str) -> list[Num]:
    """Numbers in free text. ISO dates contribute their year, month and day. Signs are dropped (we compare magnitudes)."""
    out: list[Num] = []
    for m in ISO_DATE.finditer(text):
        for part in m.groups():
            out.append(Num(float(int(part)), 0, part))
    rest = CITATION.sub(" ", ISO_DATE.sub(" ", text))
    for m in NUMBER.finditer(rest):
        raw = m.group(1)
        clean = raw.replace(",", "")
        out.append(Num(float(clean), len(clean.split(".")[1]) if "." in clean else 0, raw))
    return out


def _walk(obj):
    """Yield every leaf (number or string) and every dict key 'chunk_id' value found in a nested result."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _walk(v)
    else:
        yield obj


def _chunk_ids(obj) -> set[str]:
    ids: set[str] = set()
    if isinstance(obj, dict):
        if isinstance(obj.get("chunk_id"), str):
            ids.add(obj["chunk_id"])
        for v in obj.values():
            ids |= _chunk_ids(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            ids |= _chunk_ids(v)
    return ids


@dataclass
class Evidence:
    """What the briefing is allowed to quote: numbers and retrieved chunk ids."""
    allowed_numbers: list[float] = field(default_factory=list)
    retrieved_ids: set[str] = field(default_factory=set)

    @classmethod
    def from_tool_results(cls, tool_results: list[dict], dates=()) -> "Evidence":
        nums: list[float] = [float(c) for c in config.VERIFY_CONSTANTS]
        ids: set[str] = set()
        for d in dates:
            d = Date.fromisoformat(d) if isinstance(d, str) else d
            nums += [float(d.year), float(d.month), float(d.day)]
        for tr in tool_results:
            result = tr.get("result", {})
            if isinstance(result, dict) and "error" in result:
                continue
            ids |= _chunk_ids(result)
            for leaf in _walk(result):
                if isinstance(leaf, bool) or leaf is None:
                    continue
                if isinstance(leaf, (int, float)):
                    nums.append(abs(float(leaf)))
                elif isinstance(leaf, str):
                    nums += [n.value for n in extract_numbers(leaf)]  # dates, "p50 above 60", chunk text
        return cls(allowed_numbers=sorted(set(nums)), retrieved_ids=ids)

    def matches(self, value: float, decimals: int) -> bool:
        """True if some allowed number a satisfies: round(a, decimals) == value, or (integers >= 10) |value - a| <= 0.5,
        or (a is a ratio in (0, 1]) value == round(a * 100)."""
        n = Decimal(repr(float(value)))
        for a in self.allowed_numbers:
            if _half_up(a, decimals) == n:
                return True
            if decimals == 0 and value >= 10 and abs(value - a) <= 0.5:
                return True
            if decimals == 0 and 0 < a <= 1 and _half_up(a * 100, 0) == n:
                return True
        return False


@dataclass
class VerifyResult:
    ok: bool
    errors: list[str]
    checked_numbers: list[float]


def _words(text: str) -> list[str]:
    return re.findall(r"\S+", CITATION.sub(" ", text))


def _is_text(x) -> bool:
    return isinstance(x, str) and x.strip() != ""


def verify(briefing: dict, evidence: Evidence) -> VerifyResult:
    """Check a briefing {headline, briefing, citations}. Returns every error found, not just the first."""
    errors: list[str] = []
    checked: list[float] = []
    headline = briefing.get("headline") if isinstance(briefing, dict) else None
    text = briefing.get("briefing") if isinstance(briefing, dict) else None
    cites = briefing.get("citations") if isinstance(briefing, dict) else None

    if not _is_text(headline):
        errors.append("headline is missing")
    elif len(_words(headline)) > config.HEADLINE_MAX_WORDS:
        errors.append(f"headline has {len(_words(headline))} words (max {config.HEADLINE_MAX_WORDS})")
    if not _is_text(text):
        errors.append("briefing text is missing")

    for label, s in (("headline", headline), ("briefing", text)):
        if not _is_text(s):
            continue
        for n in extract_numbers(s):
            checked.append(n.value)
            shown = str(int(n.raw)) if n.raw.isdigit() else n.raw  # "05" from a date shows as 5
            msg = f"number {shown} not found in evidence"
            if not evidence.matches(n.value, n.decimals) and msg not in errors:
                errors.append(msg)
        for pattern, why in config.BANNED_PATTERNS:
            m = re.search(pattern, s, re.I)
            if m:
                errors.append(f"banned content in {label} ({why}): '{m.group(0)}'")
        m = re.search(config.BANNED_NAME_PATTERN, s)
        if m:
            errors.append(f"banned content in {label} (personal name): '{m.group(0)}'")

    if _is_text(text):
        n_words = len(_words(text))
        lo, hi = config.BRIEFING_WORDS
        if not lo <= n_words <= hi:
            errors.append(f"word count {n_words} outside {lo}-{hi}")

    if not isinstance(cites, list) or not all(isinstance(c, str) for c in cites):
        errors.append("citations must be a list of chunk ids")
        cites = []
    if not cites:
        errors.append("at least 1 citation is required")
    for c in cites:
        if c not in evidence.retrieved_ids:
            errors.append(f"citation {c} was not retrieved")
    if _is_text(text):
        for c in sorted(set(INLINE_CITATION.findall(text))):
            if c not in evidence.retrieved_ids:
                errors.append(f"inline citation [{c}] was not retrieved")
            elif c not in cites:
                errors.append(f"inline citation [{c}] is not listed in citations")

    return VerifyResult(ok=not errors, errors=errors, checked_numbers=checked)
