"""The Critic (D21): a second reading of a brief that has already passed the Verifier.

An `AgentSpec` with no tools. It gets the brief, its type and the text of every chunk it cites, so it can judge what the
Verifier cannot: whether a citation supports its sentence, overclaims, missing caveats, clarity and tone.

It never adds numbers. Its rewrite may only delete, reorder or reword, and is used only if it adds no number, sign or
citation AND passes the Verifier again. Whatever the Critic does, including failing, the verified original is never lost.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Callable

from psw.agents.runtime import AgentSpec, run_agent
from psw.agents.verifier import (CITATION, INLINE_CITATION, ISO_DATE, SATELLITE_NAME, URL, Evidence, VerifyResult,
                                  extract_numbers, sentences)

ISSUE_TYPES = ("unsupported_citation", "overclaim", "missing_caveat", "unclear", "tone")
CITATION_WARNING = "⚠️ Critic: citation may not support sentence"
MAX_ISSUES, MAX_CHARS = 8, 300
NEGATIVE = re.compile(r"(?<![\w.])[-−](\d+(?:\.\d+)?)")  # a minus sign, not a hyphen in "30-50" or a name

SYSTEM_PROMPT = """You review a short brief about stubble fires and air quality in Punjab, India. The brief has already passed an automatic check that every number in it appears in the data. You do not recheck numbers, and you cannot call tools.

You get the brief type, the brief, and the full text of every knowledge chunk it cites. Figures without a citation come from data tools and need no citation.

Report only real problems, each with one of these types:
- unsupported_citation: a sentence carries a [K012] citation, but the text of that chunk does not say what the sentence says, or is about another season or place. Go through every [K012] marker one at a time and compare its sentence with that chunk's text. The chunk must state the claim itself: a chunk about a method or a limitation does not support a count, percentage or share, and a chunk that is only on a related topic does not count.
- overclaim: the sentence says more than the signals can show.
- missing_caveat: a limitation needed to read a claim correctly is absent.
- unclear: a reader could easily misread the sentence.
- tone: alarming, blaming or promotional wording.

For audit briefs: the signals are satellite and official data with no field verification. The detection gap is a fall in counts per burned km², not a count of unseen fires. Treat as overclaims: that farmers are hiding fires; that official numbers are wrong; stating how much burning really happened as fact; that the signals prove burning moved to other hours; any wording that implies field verification or ground truth. Suggesting field checks as future work is not a problem.

If deleting, reordering or rewording can fix the problems, return the whole corrected brief text in "suggested_text". In it:
- never add a number, a date, a citation id or a new fact, and never change a minus sign;
- fix an unsupported citation by removing that [K012] marker. Never move a marker to another sentence. Keep at least one citation;
- keep any part labels, the sentences about triangulation and about what the range leaves out, the range, and its confidence and direction;
- keep the length inside the word range you are given.
"suggested_text" is the brief text only: do not put the headline in it. Otherwise set "suggested_text" to null.

Return only this JSON object, with no other text:
{"ok": true or false, "issues": [{"type": "...", "sentence": "the sentence, copied", "note": "what is wrong, briefly"}], "suggested_text": "..." or null}"""


def build_critic() -> AgentSpec:
    return AgentSpec(name="critic", system_prompt=SYSTEM_PROMPT, tools=[], max_turns=2, json_output=True)


def base_mode(mode: str) -> str:
    """'agent_revised+critic' -> 'agent_revised'."""
    return str(mode).split("+")[0]


def cited_ids(draft: dict) -> list[str]:
    listed = [c for c in draft.get("citations", []) if isinstance(c, str)]
    return list(dict.fromkeys(INLINE_CITATION.findall(draft.get("briefing", "")) + listed))


def cited_chunks(draft: dict, kb) -> list[dict]:
    """[{chunk_id, title, text}] for every chunk the brief cites, from the index."""
    table = kb.chunks.set_index("chunk_id")
    return [{"chunk_id": c, "title": str(table.loc[c, "title"]), "text": str(table.loc[c, "text"])}
            for c in cited_ids(draft) if c in table.index]


def figure_citations(text: str, chunks: list[dict]) -> list[dict]:
    """Found by code, not by the model: a cited sentence with a figure that the cited chunk does not contain.

    Years are ignored. This is certain evidence that the chunk does not support the sentence's figure; it says nothing
    about sentences without figures, which only the Critic can judge.
    """
    numbers = {c["chunk_id"]: Evidence(allowed_numbers=[n.value for n in extract_numbers(URL.sub(" ", c["text"]))])
               for c in chunks}
    found = []
    for sentence in sentences(text):
        figures = [n for n in extract_numbers(sentence) if not (n.decimals == 0 and 1900 <= n.value <= 2100)]
        for cid in dict.fromkeys(INLINE_CITATION.findall(sentence)):
            missing = [n.raw for n in figures if cid in numbers and not numbers[cid].matches(n.value, n.decimals)]
            if missing:
                found.append({"type": "unsupported_citation", "sentence": sentence[:MAX_CHARS],
                              "note": f"{cid} does not contain the figure(s) {', '.join(missing)}", "by": "check"})
    return found


# the three directions get_gap_range can report: results for one place, which a general method note cannot establish
RESULT_STATEMENTS = ("counts fell more than burned area", "no sign of a detection gap", "direction unclear")


def _plain(text: str) -> str:
    return " ".join(text.lower().replace("direction is unclear", "direction unclear").split())


def result_citations(text: str, chunks: list[dict]) -> list[dict]:
    """Found by code: a sentence stating this place's result, citing a chunk that does not state that result.

    Narrow on purpose: it knows only the gap direction wording. A reworded result is left to the Critic.
    """
    known = {c["chunk_id"]: _plain(c["text"]) for c in chunks}
    found = []
    for sentence in sentences(text):
        said = [r for r in RESULT_STATEMENTS if r in _plain(sentence)]
        for cid in dict.fromkeys(INLINE_CITATION.findall(sentence)):
            missing = [r for r in said if cid in known and r not in known[cid]]
            if missing:
                found.append({"type": "unsupported_citation", "sentence": sentence[:MAX_CHARS],
                              "note": f"{cid} does not state the result '{missing[0]}'", "by": "check"})
    return found


def code_findings(text: str, chunks: list[dict]) -> list[dict]:
    """Everything the code can prove about citations, one entry per sentence."""
    out = figure_citations(text, chunks)
    return out + _not_already_in(result_citations(text, chunks), out)


def user_message(draft: dict, brief_type: str, chunks: list[dict], word_range: tuple[int, int]) -> str:
    cited = "\n\n".join(f"[{c['chunk_id']}] {c['title']}\n{c['text']}" for c in chunks) or "none"
    message = (f"Brief type: {brief_type}\nWord range for the text: {word_range[0]} to {word_range[1]}\n"
               f"Headline: {draft.get('headline', '')}\n\nBrief:\n<<<\n{draft.get('briefing', '')}\n>>>\n\nCited chunks:\n{cited}")
    known = code_findings(draft.get("briefing", ""), chunks)
    if known:
        listed = "\n".join(f"- {i['note']}: \"{i['sentence']}\"" for i in known)
        message += f"\n\nAn automatic check already found these unsupported citations. Report them and remove those markers:\n{listed}"
    return message


def clean_review(raw) -> tuple[list[dict], str | None] | None:
    """(issues, suggested text or None), or None if the answer is not a usable review."""
    if not isinstance(raw, dict) or not isinstance(raw.get("issues"), list) or not isinstance(raw.get("ok"), bool):
        return None
    issues = [{"type": i["type"], "sentence": str(i.get("sentence", ""))[:MAX_CHARS], "note": str(i.get("note", ""))[:MAX_CHARS]}
              for i in raw["issues"] if isinstance(i, dict) and i.get("type") in ISSUE_TYPES]
    if not raw["ok"] and not issues:  # "not ok" with nothing readable behind it tells us nothing
        return None
    text = raw.get("suggested_text")
    return issues[:MAX_ISSUES], text.strip() if isinstance(text, str) and text.strip() else None


def _signed_numbers(text: str) -> set[tuple[float, bool]]:
    """Every number as (value, is negative)."""
    rest = SATELLITE_NAME.sub(" ", CITATION.sub(" ", ISO_DATE.sub(" ", text)))
    negative = {float(m) for m in NEGATIVE.findall(rest.replace(",", ""))}
    return {(n.value, n.value in negative) for n in extract_numbers(text)}


def _words(sentence: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", CITATION.sub(" ", sentence).lower()))


def moved_citations(old_text: str, new_text: str) -> list[str]:
    """Citation ids the rewrite attached to a different sentence. Each new sentence is matched to the old sentence
    that shares most of its words, and may only carry markers that old sentence had."""
    old = [(_words(s), set(INLINE_CITATION.findall(s))) for s in sentences(old_text)]
    moved = []
    for s in sentences(new_text):
        marks = INLINE_CITATION.findall(s)
        if not marks or not old:
            continue
        words = _words(s)
        _, had = max(old, key=lambda o: len(words & o[0]) / max(1, len(words | o[0])))
        moved += [c for c in marks if c not in had]
    return sorted(set(moved))


def apply_rewrite(draft: dict, new_text: str, check: Callable[[dict], VerifyResult]
                  ) -> tuple[dict | None, VerifyResult | None, list[str]]:
    """(the rewritten draft, its verdict, []) if the rewrite is acceptable, else (None, None, why it was rejected)."""
    old_text = draft["briefing"]
    reasons = []
    added = _signed_numbers(new_text) - _signed_numbers(old_text)
    if added:
        shown = ", ".join(f"{'-' if neg else ''}{value:g}" for value, neg in sorted(added))
        reasons.append(f"it adds or changes a number: {shown}")
    old_inline, new_inline = set(INLINE_CITATION.findall(old_text)), set(INLINE_CITATION.findall(new_text))
    if new_inline - old_inline:
        reasons.append(f"it adds a citation: {', '.join(sorted(new_inline - old_inline))}")
    elif moved := moved_citations(old_text, new_text):
        reasons.append(f"it moves a citation to another sentence: {', '.join(moved)}")
    citations = [c for c in draft.get("citations", []) if c in new_inline or c not in old_inline]
    candidate = {**draft, "briefing": new_text, "citations": citations}
    verdict = check(candidate)
    reasons += [f"it fails the Verifier: {e}" for e in verdict.errors]
    return (None, None, reasons) if reasons else (candidate, verdict, [])


@dataclass
class Review:
    draft: dict  # the rewrite if it was accepted, otherwise the original
    verdict: VerifyResult | None  # the rewrite's verdict, None if the original stands
    info: dict  # stored on the record as `critic`
    entry: dict  # trace line


def _same(a: str, b: str) -> bool:
    """The model and the code may quote one sentence differently (with or without its part label)."""
    x, y = " ".join(a.split()), " ".join(b.split())
    return bool(x and y) and (x in y or y in x)


def _not_already_in(extra: list[dict], issues: list[dict]) -> list[dict]:
    seen = [i["sentence"] for i in issues if i["type"] == "unsupported_citation"]
    return [e for e in extra if not any(_same(e["sentence"], s) for s in seen)]


def drop_headline(text: str, headline: str) -> str:
    """The model sometimes repeats the headline on top of the rewritten text; that is an addition, not a rewrite."""
    first, _, rest = text.strip().partition("\n")
    rest = rest.strip()

    def plain(s: str) -> str:
        return re.sub(r"\W+", " ", s).strip().lower()

    title = plain(first) == plain(headline) or (rest.startswith("What detections say:") and not first.rstrip().endswith((".", "]")))
    return rest if rest and title else text.strip()


def _info(issues=(), rewrite_used=False, error=None, rejected=(), original=None, remaining=()) -> dict:
    """`remaining`: what the code check still finds in the text that will be shown."""
    issues = list(issues)
    issues += _not_already_in(list(remaining), issues)
    unsupported = any(i["type"] == "unsupported_citation" for i in issues)
    return {"ok": error is None and not issues, "issues": issues, "rewrite_used": rewrite_used, "error": error,
            "rewrite_rejected": list(rejected), "original_briefing": original,
            "warning": CITATION_WARNING if remaining or (unsupported and not rewrite_used) else None}


def review(draft: dict, brief_type: str, chunks: list[dict], llm, check: Callable[[dict], VerifyResult],
           word_range: tuple[int, int]) -> Review:
    """One Critic pass over a verified draft. Never raises because of the LLM or its answer."""
    t0 = time.perf_counter()
    kept, verdict = draft, None
    known: list[dict] = []
    try:
        known = code_findings(draft["briefing"], chunks)
        res = run_agent(build_critic(), user_message(draft, brief_type, chunks, word_range), llm)
        parsed = None if res.error else clean_review(res.output)
        if parsed is None:
            info = _info(error=res.error or "the critic did not return a usable review", remaining=known)
        else:
            issues, text = parsed
            issues = [{**i, "by": "critic"} for i in issues]
            text = drop_headline(text, draft.get("headline", "")) if text else None
            if (issues or known) and text and text != draft["briefing"]:
                candidate, new_verdict, rejected = apply_rewrite(draft, text, check)
                if candidate:
                    kept, verdict = candidate, new_verdict
                    left = code_findings(text, chunks)
                    fixed = [i for i in known if not any(_same(i["sentence"], x["sentence"]) for x in left)]
                    info = _info(issues + _not_already_in(fixed, issues), rewrite_used=True,
                                 original=draft["briefing"], remaining=left)
                else:
                    info = _info(issues, rejected=rejected, remaining=known)
            else:
                info = _info(issues, remaining=known)
    except Exception as e:  # a Critic bug or odd answer must not cost us a verified brief
        kept, verdict, info = draft, None, _info(error=f"{type(e).__name__}: {e}", remaining=known)
    summary = info["error"] or (f"{len(info['issues'])} issue(s)" + (", rewrite used" if info["rewrite_used"] else "")
                                + (", rewrite rejected" if info["rewrite_rejected"] else ""))
    entry = {"step": 1, "agent": "critic", "tool": "review",
             "args": {"brief_type": brief_type, "chunks": [c["chunk_id"] for c in chunks]},
             "result_summary": summary[:180], "ms": int((time.perf_counter() - t0) * 1000)}
    return Review(kept, verdict, info, entry)
