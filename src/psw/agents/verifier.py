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
FIELD_NAME = re.compile(r"\b[a-z]+(?:_[a-z0-9]+)+\b")  # raw tool field names such as rank_in_punjab or upwind_frp_l0
URL = re.compile(r"https?://\S+")  # digits inside an address are not facts anyone may quote
SATELLITE_NAME = re.compile(r"\bNOAA[- ]?2[01](?:\s*/\s*(?:NOAA[- ]?)?2[01])?\b|\bINSAT-3DS?\b")  # names, not numbers
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
    rest = SATELLITE_NAME.sub(" ", CITATION.sub(" ", ISO_DATE.sub(" ", text)))
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
                    nums += [n.value for n in extract_numbers(URL.sub(" ", leaf))]  # dates, "p50 above 60", chunk text
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


def verify(briefing: dict, evidence: Evidence, word_range: tuple[int, int] | None = None) -> VerifyResult:
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
        for name in dict.fromkeys(FIELD_NAME.findall(s)):
            errors.append(f"raw field name in {label}: '{name}' (write it in plain words)")
        m = re.search(config.BANNED_NAME_PATTERN, s)
        if m:
            errors.append(f"banned content in {label} (personal name): '{m.group(0)}'")

    if _is_text(text):
        n_words = len(_words(text))
        lo, hi = word_range or config.BRIEFING_WORDS
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


def sentences(text: str) -> list[str]:
    """Sentences, with a citation marker written after the full stop kept with the sentence it follows."""
    out: list[str] = []
    for part in re.split(r"(?<=[.!?])\s+|\n+", text):
        lead = re.match(r"^(?:\[K\d{3}\][\s.]*)+", part)
        if lead and out:
            out[-1] += " " + lead.group(0).strip()
            part = part[lead.end():]
        if part.strip():
            out.append(part.strip())
    return out


# ---------- audit briefs: the same checks, plus what every audit must say ----------

# the range must be tied to the burned-area product's error, in one sentence
BURNED_ERROR = re.compile(r"(?=[^.]*(burned[- ]area|MCD64A1))(?=[^.]*(error|uncertain))", re.I)


def _has_burned_area_caveat(text: str) -> bool:
    return any(BURNED_ERROR.match(sentence) for sentence in sentences(text))


def _other_season_errors(text: str, audit_seasons: dict[str, int]) -> list[str]:
    """An earlier season's audit document may only support a sentence that is about that season."""
    errors = []
    for sentence in sentences(text):
        said = INLINE_CITATION.sub(" ", sentence)
        for cid in INLINE_CITATION.findall(sentence):
            year = audit_seasons.get(cid)
            if year and str(year) not in said:
                errors.append(f"[{cid}] is the {year} audit: it cannot support a sentence that is not about {year}")
    return list(dict.fromkeys(errors))


MEASURE = re.compile(r"per burned (km|area|square)", re.I)  # what a gap range is a range of
ABOUT_PRODUCT = re.compile(r"cropland|MCD64A1|burned[- ]area (is|has|product)", re.I)
PASS_TIME = re.compile(r"(outside|between)[^.]{0,40}\b(over)?pass", re.I)  # the first of D23's three explanations
DIRECTION_WORDS = {"direction unclear": "unclear", "no sign of a detection gap": "no sign"}


def _gap_errors(text: str, gap: dict) -> list[str]:
    """What a brief must say about this district-season's get_gap_range result, whatever else it says."""
    if gap.get("gap_low_pct") is None:
        return [] if "insufficient evidence" in text.lower() else ["say there is insufficient evidence for a range"]
    errors = []
    if not _has_burned_area_caveat(text):
        errors.append("say in one sentence that the range does not include the error of the burned-area product")
    if not MEASURE.search(text):
        errors.append("say what the range measures: the fall in counts per burned km²")
    conf = gap["confidence"]
    stated = re.compile(rf"confidence[^.]{{0,30}}\b{conf}\b|\b{conf}\b[^.]{{0,15}}confidence", re.I)
    # "burned area in cropland is low confidence" is about the satellite product and says nothing about the range
    if not any(stated.search(s) and not ABOUT_PRODUCT.search(s) for s in sentences(text)):
        errors.append(f"state that the confidence of the range is {conf}")
    if gap.get("direction") == "counts fell more than burned area" and not PASS_TIME.search(text):
        errors.append("say that one explanation is burning moving outside satellite pass times (it is not proven)")
    word = DIRECTION_WORDS.get(gap.get("direction"))
    if word and word not in text.lower():
        errors.append(f"say plainly: {gap['direction']}")
    return errors


def verify_audit(briefing: dict, evidence: Evidence, gap: dict | None = None,
                 word_range: tuple[int, int] = config.AUDIT_WORDS,
                 audit_seasons: dict[str, int] | None = None, night_too_few: bool = False) -> VerifyResult:
    """`gap`: the get_gap_range result for this district and season (None or an error for the base season).
    `audit_seasons`: chunk id -> season for every retrieved chunk that is another season's audit document.
    `night_too_few`: the place is a district, so any mention of night detections must say they are too few."""
    base = verify(briefing, evidence, word_range=word_range)
    errors = list(base.errors)
    headline = briefing.get("headline") if isinstance(briefing, dict) else None
    text = briefing.get("briefing") if isinstance(briefing, dict) else None
    if _is_text(text):
        for label in config.AUDIT_LABELS:
            if f"{label.lower()}:" not in text.lower():
                errors.append(f"missing the part labelled '{label}:'")
        if "triangulation" not in text.lower():
            errors.append("say that this is triangulation, not field verification")
        if gap and "error" not in gap:
            errors += _gap_errors(text, gap)
        errors += _other_season_errors(text, audit_seasons or {})
        if night_too_few and re.search(r"\bnight", text, re.I) and "too few" not in text.lower():
            errors.append("say that a district's night detections are too few to show a change in timing")
    for s in (headline, text):
        if not _is_text(s):
            continue
        for pattern, why in config.AUDIT_BANNED_PATTERNS:
            m = re.search(pattern, s, re.I)
            if m:
                errors.append(f"banned content ({why}): '{m.group(0)}'")
    return VerifyResult(ok=not errors, errors=errors, checked_numbers=base.checked_numbers)


# ---------- dispatch briefs: where help goes, so allocation figures and names fail closed ----------

# The only ways a unit figure may be written. {d} is a district, matched just before.
BRACKET_FORM = re.compile(r"\s*\(\s*(\d[\d,]*)\s+units?\s*\)", re.I)
MOVE_FORM = re.compile(r"\s+from\s+(\d[\d,]*)\s+units?\s+to\s+(\d[\d,]*)\s+units?\b", re.I)
_N = r"(\d[\d,]*)"
TOTAL_FORMS = [  # (which total, pattern). Each states a figure for the whole plan, never for one district.
    ("cap", rf"limit per district(?:\s+is|\s+of|\s+set at|,)?\s+{_N}\s+units?\b"),
    ("cap", rf"limit of\s+{_N}\s+units? per district\b"),
    ("cap", rf"no district (?:may|can|will) (?:take|get|receive) more than\s+{_N}\s+units?\b"),
    ("fleet", rf"\bof\s+{_N}\s+units?,"),
    ("fleet", rf"\bof the\s+{_N}\s+units?\b"),
    ("fleet", rf"\bnone of the\s+{_N}\s+units?\b"),
    ("fleet", rf"{_N}\s+units? (?:are|were) available\b"),
    ("fleet", rf"{_N}\s+units? across (?:Punjab|the state)\b"),
    ("unplaced", rf"{_N}\s+units? (?:are|were|is) (?:not placed|unplaced)\b"),
    ("placed", rf"{_N}\s+units? (?:are|were) placed\b"),
    ("others", rf"other(?:\s+\d+)? districts[^.;]{{0,30}}?\bshare\s+{_N}\s+units?\b"),
]
TOTAL_FORMS = [(name, re.compile(p, re.I)) for name, p in TOTAL_FORMS]
UNIT_WORD = re.compile(r"\bunits?\b", re.I)
ALLOCATING = re.compile(r"\b(gets?|receives?|allocat\w*|assign\w*|deploy\w*|sends?|sent|moves?|moved|shift\w*|shares?"
                        r"|takes?|machines?|teams?)\b", re.I)
SPELLED = re.compile(r"\b(two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|twenty|thirty|forty|fifty|sixty|seventy"
                     r"|eighty|ninety|hundred|dozen)\b[-\w ]{0,12}?\b(units?|machines?|teams?)\b", re.I)
# numbers that say what they are, so they cannot be read as an allocation
MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
LABELLED = [re.compile(p, re.I) for p in (
    r"\bcell\s+\d+", r"\d[\d,.]*\s*(?:%|percent\b|mm\b)", r"\d+(?:\s*(?:to|-|–)\s*\d+)?[\s-]*days?\b",
    r"\d[\d,]*\s+(?:forecast\s+|satellite\s+|fire\s+)?detections?\b", rf"\b\d{{1,2}}\s+(?:{MONTHS})(?:\s+\d{{4}})?",
    r"\b\d{4}-\d{2}-\d{2}\b")]
NOT_A_COUNT = re.compile(r"not\s+(a\s+)?(count|measure|number)", re.I)
UNCERTAIN = re.compile(r"uncertain|\bbetween\b|80%", re.I)  # not "range": the gap range is something else


@dataclass
class Plan:
    """What the code decided, for checking what the brief says about it."""
    units: dict[str, int]  # district -> units
    detected_units: dict[str, int]  # district -> units without the likely-burning adjustment
    totals: dict[str, int]  # fleet, placed, unplaced, cap (limit per district), others (the rest share these)
    use_likely: bool
    most_moved: str | None = None  # the district the adjustment moves most, if it moves any
    confidence: str | None = None  # of the gap range behind the biggest move (likely planning only)
    words: frozenset[str] = frozenset()  # words seen in lower case in our own texts: capitalised, they are not names


def _names(district: str) -> list[str]:
    short = district.split(" (")[0]
    return [district] if short == district else [district, short]


def _mentions(text: str, districts) -> list[tuple[int, int, str]]:
    """(start, end, district) for every district named in the text."""
    found = {}
    for d in districts:
        for name in _names(d):
            for m in re.finditer(rf"\b{re.escape(name)}(?![A-Za-z])", text):
                if m.start() not in found or m.end() > found[m.start()][0]:
                    found[m.start()] = (m.end(), d)
    return sorted((start, end, d) for start, (end, d) in found.items())


def _number(raw: str) -> int:
    return int(raw.replace(",", ""))


def _approved_units(sentence: str, plan: Plan) -> tuple[list[tuple[int, int]], list[str]]:
    """(spans of the unit figures written in an approved form, errors for forms whose figure is wrong)."""
    spans, errors = [], []
    for _, end, d in _mentions(sentence, plan.units):
        if m := BRACKET_FORM.match(sentence, end):
            spans.append(m.span())
            if _number(m.group(1)) != plan.units[d]:
                errors.append(f"{d} gets {plan.units[d]} units, not {_number(m.group(1))}")
        elif m := MOVE_FORM.match(sentence, end):
            spans.append(m.span())
            said = (_number(m.group(1)), _number(m.group(2)))
            right = (plan.detected_units[d], plan.units[d])
            if said != right:
                errors.append(f"{d} moves from {right[0]} units to {right[1]} units, not from {said[0]} to {said[1]}")
    for name, pattern in TOTAL_FORMS:
        for m in pattern.finditer(sentence):
            if any(a <= m.start(1) < b for a, b in spans):
                continue
            spans.append(m.span())
            if _number(m.group(1)) != plan.totals[name]:
                errors.append(f"'{m.group(0)}' is wrong: that figure is {plan.totals[name]}")
    return spans, errors


def _bare_numbers(sentence: str, covered: list[tuple[int, int]]) -> list[str]:
    """Numbers that are neither part of an approved unit form nor labelled as something else."""
    covered = covered + [m.span() for p in LABELLED for m in p.finditer(sentence)]
    return [m.group(1) for m in NUMBER.finditer(sentence) if not any(a <= m.start(1) < b for a, b in covered)]


# a figure written straight after a district ("Sangrur, 57", "Sangrur at 57") reads as its allocation
BESIDE = r"[\s,:;=–-]*(?:(?:at|with|gets?)\s+)?"


def unit_errors(text: str, plan: Plan, where_part: str = "") -> list[str]:
    """Allocation figures fail closed. A unit figure may be written only as "District (N units)", "District from X units
    to Y units", or one of the TOTAL_FORMS, and each is checked. In a sentence that talks about units, machines, teams
    or who gets what (and everywhere in `where_part`), any other unlabelled number is rejected, and so is one written
    straight after a district's name.

    What is left: a bare figure outside `where_part`, away from any district name, in a sentence with no such word.
    """
    errors = []
    where = {" ".join(s.split()) for s in sentences(CITATION.sub(" ", where_part))}
    for sentence in sentences(CITATION.sub(" ", text)):
        sentence = " ".join(sentence.split())
        spans, wrong = _approved_units(sentence, plan)
        errors += wrong
        if m := SPELLED.search(sentence):
            errors.append(f"write unit figures in digits, in an approved form: '{m.group(0)}'")
        in_where = any(sentence in w or w in sentence for w in where)
        suspect = []
        if in_where or UNIT_WORD.search(sentence) or ALLOCATING.search(sentence):
            suspect = _bare_numbers(sentence, spans)
        else:
            bare = _bare_numbers(sentence, spans)
            for _, end, _d in _mentions(sentence, plan.units):
                m = re.compile(BESIDE + r"(\d[\d,.]*\d|\d)").match(sentence, end)
                if m and m.group(1) in bare:
                    suspect.append(m.group(1))
        for raw in suspect:
            errors.append(f"the figure {raw} could be read as an allocation: move it to a sentence that says what it counts, "
                          "or write units as 'District (N units)', 'District from N units to M units' or 'The limit per "
                          "district is N units'")
    return list(dict.fromkeys(errors))


# ---------- names: nothing below a district (D18) ----------

BELOW_DISTRICT = re.compile(r"\b(villages?|tehsils?|panchayats?|sarpanch\w*|gaon|pind|khasra|mohalla)\b", re.I)
PROPER_NOUNS = frozenset(("punjab india delhi viirs modis firms noaa nasa npp snpp caqm ppcb crm grap cams naaqs national "
                          "green tribunal crop residue management custom hiring centre centres happy super seeder "
                          "mohali ropar muktsar nawanshahr january february march april may june july august september "
                          "october november december").split())
# ordinary words that can open a sentence, so the check does not depend on them appearing in lower case elsewhere
ORDINARY = frozenset(("a an the of in on at to for from by with without and or but if as it its this that these those they "
                      "their there here what where why how who when which each every all no none not some most more less "
                      "fewer other others only also then than so while after before during both either since because "
                      "is are was were be been being has have had do does did can could may might should would will must "
                      "we our you one dispatch brief plan units unit district districts check recompute awareness teams "
                      "machines machine rain likely forecast these overall however meanwhile finally first second next "
                      "today daily recheck confirm start use keep send note watch limit almost visit go focus prioritise "
                  "prioritize begin place move stage monitor ensure coordinate").split())
WORD = re.compile(r"[A-Za-z]+")


def lowercase_words(texts) -> frozenset[str]:
    """Words that appear written in lower case. A place or person's name does not."""
    found: set[str] = set()
    for text in texts:
        found.update(w for w in WORD.findall(URL.sub(" ", str(text))) if w.islower())
    return frozenset(found)


def name_errors(text: str, plan: Plan) -> list[str]:
    """No village, field or farmer names. A capitalised word is accepted only if it is part of a district's name, one of
    the PROPER_NOUNS, or an ordinary word (ORDINARY, or seen in lower case in `plan.words`).

    What is left: a name that is also an ordinary word, or one written in lower case without "village" or the like.
    """
    errors = [f"name nothing below a district: '{m.group(0)}'" for m in BELOW_DISTRICT.finditer(text)]
    if any(c.isalpha() and not c.isascii() for c in text):
        errors.append("write in English letters only: names in other scripts cannot be checked")
    rest = CITATION.sub(" ", text)
    for district in plan.units:
        for name in _names(district) + [district.replace("(", "").replace(")", "")]:
            rest = rest.replace(name, " ")
    known = ORDINARY | PROPER_NOUNS | plan.words
    for word in WORD.findall(rest):
        if len(word) > 1 and word[0].isupper() and word.lower() not in known:
            errors.append(f"'{word}' looks like a name: name only districts and cells (no village, field or farmer names)")
    return list(dict.fromkeys(errors))


def _dispatch_banned(label: str, s: str) -> list[str]:
    errors = []
    for patterns, flags in ((config.DISPATCH_BANNED_PATTERNS, re.I), (config.DISPATCH_BANNED_CASE_SENSITIVE, 0)):
        for pattern, why in patterns:
            m = re.search(pattern, s, flags)
            if m:
                errors.append(f"banned content in {label} (no {why}; this is support, not enforcement): '{m.group(0)}'")
    return errors


RAIN_AHEAD = re.compile(r"(?=.*(\brain|\bmm\b))(?=.*\bnext\b)", re.I | re.S)


def _archive_rain_errors(text: str) -> list[str]:
    """Rain "over the next 3 days" is observed archive data standing in for a forecast: the sentence must say so."""
    for sentence in sentences(text):
        if RAIN_AHEAD.match(sentence) and not re.search(r"archive|stand(s|ing)? in", sentence, re.I):
            return ["say in the same sentence that rain for the next 3 days comes from the weather archive, standing in "
                    "for a forecast"]
    return []


def _where_part(text: str) -> str:
    """The "Where the units go" part: everything before "Why:"."""
    return re.split(r"\bwhy\s*:", text, flags=re.I)[0] if re.search(r"where the units go\s*:", text, re.I) else ""


def _dispatch_content_errors(text: str, plan: Plan) -> list[str]:
    errors = [f"missing the part labelled '{label}:'" for label in config.DISPATCH_LABELS
              if f"{label.lower()}:" not in text.lower()]
    placed = {d: n for d, n in plan.units.items() if n > 0}
    if placed:
        top = max(placed, key=lambda d: (placed[d], d))
        leaders = [d for d, n in placed.items() if n == placed[top]]
        if not _mentions(text, leaders):
            errors.append(f"name the district that gets the most units: {top}")
        watch = re.split(r"what to watch\s*:", text, flags=re.I)
        if len(watch) > 1 and not _mentions(watch[-1], leaders):
            errors.append(f"'What to watch' must be about {top}, the district with the most units: its rain and cells")
    if "detection gap" not in text.lower():
        errors.append("state the detection-gap caveat, using the words 'detection gap'")
    if plan.use_likely and not NOT_A_COUNT.search(text):
        errors.append("say that likely burning is not a count of fires")
    if plan.use_likely and plan.most_moved and not _mentions(text, [plan.most_moved]):
        errors.append(f"name the district the adjustment moves most: {plan.most_moved}")
    if plan.use_likely and plan.confidence:
        conf = plan.confidence
        stated = re.compile(rf"confidence[^.]{{0,40}}\b{conf}\b|\b{conf}\b[^.]{{0,15}}confidence", re.I)
        if not any(stated.search(s) for s in sentences(text)) or re.search(r"high confidence", text, re.I):
            about = plan.most_moved or "the leading district"
            errors.append(f"state that the gap range behind {about} has {conf} confidence")
    if plan.use_likely and re.search(r"expected burning", text, re.I):
        errors.append("do not write 'expected burning': call the adjusted figure 'likely burning'")
    if not re.search(r"\brain", text, re.I):
        errors.append("say what to watch, including rain")
    if not UNCERTAIN.search(text):
        errors.append("say how uncertain the forecast is")
    return errors


def verify_dispatch(briefing: dict, evidence: Evidence, plan: Plan, district: str | None = None,
                    word_range: tuple[int, int] = config.DISPATCH_WORDS) -> VerifyResult:
    """The shared checks, plus: the four parts, no enforcement talk, units beside the right district, the detection-gap
    caveat, rain and uncertainty. `district`: a paragraph about it is then required (`district_paragraph`)."""
    base = verify(briefing, evidence, word_range=word_range)
    errors, checked = list(base.errors), list(base.checked_numbers)
    d = briefing if isinstance(briefing, dict) else {}
    headline, text, paragraph = d.get("headline"), d.get("briefing"), d.get("district_paragraph")
    if _is_text(headline):
        errors += _dispatch_banned("headline", headline) + unit_errors(headline, plan, headline) + name_errors(headline, plan)
    if _is_text(text):
        errors += _dispatch_banned("briefing", text) + _dispatch_content_errors(text, plan)
        errors += unit_errors(text, plan, _where_part(text)) + name_errors(text, plan) + _archive_rain_errors(text)
    if district:
        if not _is_text(paragraph):
            errors.append(f"district_paragraph about {district} is missing")
        else:
            part = verify({"headline": district, "briefing": paragraph, "citations": d.get("citations")}, evidence,
                          word_range=config.DISTRICT_PARAGRAPH_WORDS)
            checked += part.checked_numbers
            found = [e for e in part.errors if "citation" not in e]  # the brief's citations are checked above
            found += _dispatch_banned("paragraph", paragraph) + unit_errors(paragraph, plan)
            found += name_errors(paragraph, plan) + _archive_rain_errors(paragraph)
            errors += [f"district paragraph: {e}" for e in found]
    elif _is_text(paragraph):
        errors.append("no district_paragraph was asked for")
    return VerifyResult(ok=not errors, errors=list(dict.fromkeys(errors)), checked_numbers=checked)
