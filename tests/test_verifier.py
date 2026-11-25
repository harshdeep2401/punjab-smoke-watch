"""The Verifier is the heart of the 'the LLM never does the maths' claim. These tests came first."""
import pytest

from psw import config
from psw.agents.verifier import Evidence, extract_numbers, verify

SNAP = {"district": "Sangrur", "date": "2025-11-05", "fire_count": 87, "frp_sum_mw": 1234.5, "fire_count_7d_mean": 40.2,
        "change_vs_7d_pct": -68.3, "rank_in_punjab": 2, "punjab_total": 410,
        "last_7_days": [{"date": "2025-10-30", "count": 12}, {"date": "2025-10-31", "count": 72}]}
FC = {"city": "Bathinda", "issue_date": "2025-11-05", "target_date": "2025-11-06", "p10": 61.2, "p50": 95.7, "p90": 140.4,
      "pm25_today": 88.8, "band_width": 79.2, "exceeds_naaqs_60_prob_hint": "p50 above 60"}
KB = {"results": [{"chunk_id": "K012", "title": "Limitations", "text": "Satellites detect heat anomalies. Overpass is about 01:30 and 13:30; the pixel is 375 m."}]}


def results():
    return [{"tool": "get_district_snapshot", "args": {}, "result": SNAP},
            {"tool": "get_city_forecast", "args": {}, "result": FC},
            {"tool": "search_knowledge", "args": {}, "result": KB}]


def evidence():
    return Evidence.from_tool_results(results(), dates=["2025-11-05"])


FILLER = ("This is plain filler language that carries no figures at all. " * 10).strip()  # 110 words


def briefing(text=FILLER, headline="Sangrur fire activity stays high", citations=("K012",)):
    return {"headline": headline, "briefing": text + " [K012]" if citations else text, "citations": list(citations)}


def test_clean_text_passes():
    r = verify(briefing(), evidence())
    assert r.ok and r.errors == []


def test_quoted_numbers_from_tool_results_pass():
    text = (FILLER + " Sangrur recorded 87 detections on 5 November 2025, 1,234.5 MW of fire power, "
            "against a 7-day mean of 40.2. The 80% interval is 61.2 to 140.4 and the median is 95.7.")
    r = verify(briefing(text), evidence())
    assert r.ok, r.errors
    assert 87 in r.checked_numbers and 1234.5 in r.checked_numbers


def test_invented_number_is_caught():
    r = verify(briefing(FILLER + " Detections reached 93 today."), evidence())
    assert not r.ok and "number 93 not found in evidence" in r.errors


def test_all_errors_are_returned_not_just_the_first():
    r = verify(briefing(FILLER + " We saw 93 and 77 detections."), evidence())
    assert "number 93 not found in evidence" in r.errors and "number 77 not found in evidence" in r.errors


def test_rounding_matches_but_wrong_rounding_does_not():
    ok = verify(briefing(FILLER + " About 87 detections and 96 ug/m3 and 1,235 MW."), evidence())
    assert ok.ok, ok.errors  # 87 vs 87; 95.7 -> 96; 1234.5 -> 1235 (half up)
    bad = verify(briefing(FILLER + " About 41 detections on average."), evidence())  # 40.2 rounds to 40, not 41
    assert not bad.ok


def test_decimals_must_match_after_rounding():
    assert verify(briefing(FILLER + " The mean was 40.2 per day."), evidence()).ok
    assert not verify(briefing(FILLER + " The mean was 40.7 per day."), evidence()).ok


def test_percent_and_sign_variants_are_accepted():
    # evidence has -68.3; the text may say "68%", "68.3%", "-68.3%" or use a real minus sign
    for phrase in ("a fall of 68%", "a fall of 68.3%", "a change of -68.3%", "a change of −68.3%"):
        assert verify(briefing(FILLER + " " + phrase + "."), evidence()).ok, phrase


def test_ratio_to_percent_rule():
    ev = Evidence.from_tool_results([{"tool": "x", "args": {}, "result": {"coverage": 0.844}}] + results()[2:], dates=[])
    assert verify(briefing(FILLER + " Coverage was 84% of days."), ev).ok
    assert not verify(briefing(FILLER + " Coverage was 94% of days."), ev).ok


def test_pm25_and_citation_ids_are_not_numbers():
    nums = extract_numbers("PM2.5 and PM10 rose, see [K012] and K007; NO2 too")
    assert nums == []
    assert verify(briefing(FILLER + " PM2.5 is the topic."), evidence()).ok


def test_ranges_and_hyphenated_units():
    assert [n.value for n in extract_numbers("7-day mean, 24-hour limit, 61.2-140.4")] == [7, 24, 61.2, 140.4]


def test_thousands_separator():
    assert [n.value for n in extract_numbers("1,234 and 12,345.6 and 987")] == [1234, 12345.6, 987]


def test_constants_are_allowed_without_evidence():
    ev = Evidence.from_tool_results([], dates=[])
    for c in config.VERIFY_CONSTANTS:
        assert ev.matches(float(c), 0)
    assert not ev.matches(33.0, 0)


def test_tolerance_cannot_be_gamed_with_small_integers():
    ev = Evidence.from_tool_results([{"tool": "x", "args": {}, "result": {"a": 87}}], dates=[])
    for n in (1, 2, 3, 5, 9, 20, 55, 86, 88, 99):
        assert not ev.matches(float(n), 0), n
    assert ev.matches(87.0, 0)


def test_dates_in_iso_form_check_their_parts():
    ok = verify(briefing(FILLER + " Data are as of 2025-11-05."), evidence())
    assert ok.ok, ok.errors
    bad = verify(briefing(FILLER + " Data are as of 2025-11-19."), evidence())
    assert not bad.ok and "number 19 not found in evidence" in bad.errors


def test_numbers_in_retrieved_chunk_text_are_allowed():
    assert verify(briefing(FILLER + " The pixel is 375 m wide."), evidence()).ok


def test_citation_not_retrieved_is_caught():
    b = briefing()
    b["citations"] = ["K012", "K999"]
    r = verify(b, evidence())
    assert not r.ok and any("K999" in e for e in r.errors)


def test_inline_citation_must_be_listed_and_retrieved():
    r = verify({"headline": "Headline words here", "briefing": FILLER + " [K012] and [K003].", "citations": ["K012"]}, evidence())
    assert not r.ok and any("K003" in e for e in r.errors)


def test_at_least_one_citation_is_required():
    r = verify(briefing(citations=()), evidence())
    assert not r.ok and any("citation" in e for e in r.errors)


@pytest.mark.parametrize("n_words,ok", [(60, False), (99, False), (100, True), (150, True), (180, True), (181, False)])
def test_word_count_limits(n_words, ok):
    text = " ".join(["word"] * n_words)
    r = verify({"headline": "A headline", "briefing": text + " [K012]", "citations": ["K012"]}, evidence())
    assert (not any("word count" in e for e in r.errors)) is ok


@pytest.mark.parametrize("phrase", ["You should wear a mask outside.", "People should stay indoors today.",
                                    "Use an N95 respirator.", "Please consult a doctor.", "Wear a face mask."])
def test_health_advice_is_banned(phrase):
    r = verify(briefing(FILLER + " " + phrase), evidence())
    assert not r.ok and any("health advice" in e for e in r.errors)


@pytest.mark.parametrize("phrase", ["The farmer Harpreet burned it.", "Mr Singh reported it.", "Smt Kaur said so."])
def test_personal_names_after_farmer_or_title_are_banned(phrase):
    r = verify(briefing(FILLER + " " + phrase), evidence())
    assert not r.ok and any("personal name" in e for e in r.errors)


def test_plural_farmers_and_ordinary_words_are_fine():
    assert verify(briefing(FILLER + " Farmers in the district may use machinery instead."), evidence()).ok


def test_headline_is_checked_for_length_and_numbers():
    long = verify(briefing(headline=" ".join(["word"] * 13)), evidence())
    assert any("headline" in e for e in long.errors)
    bad_num = verify(briefing(headline="Sangrur hits 93 detections"), evidence())
    assert "number 93 not found in evidence" in bad_num.errors


def test_malformed_output_does_not_crash():
    for bad in ({}, {"headline": "x"}, {"briefing": 5, "headline": "x", "citations": "K012"}):
        r = verify(bad, evidence())
        assert not r.ok and r.errors
