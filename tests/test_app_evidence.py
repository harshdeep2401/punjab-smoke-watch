"""Part 2 of the Evidence page: the bootstrap, the wording that follows the result, and the page on committed data."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from psw import config
from psw.app import audit_view, evidence_view as ev, helpers
from psw.response import intervals


def test_the_interval_is_deterministic_and_collapses_for_a_constant():
    seasons = [np.full((30, 1), 2.0), np.full((20, 1), 2.0)]
    mean = lambda x: x[:, 0].mean()  # noqa: E731
    assert intervals.interval(seasons, mean, reps=200) == (2.0, 2.0)
    noisy = [np.random.default_rng(1).normal(0, 1, (60, 1))]
    low, high = intervals.interval(noisy, mean, reps=500)
    assert low < noisy[0].mean() < high and (low, high) == intervals.interval(noisy, mean, reps=500)


def test_blocks_keep_neighbouring_days_together():
    idx = intervals.block_indices(20, 7, np.random.default_rng(0))
    assert len(idx) == 20 and idx.max() < 20
    assert all(b - a == 1 for a, b in zip(idx[:6], idx[1:7]))
    assert list(intervals.block_indices(3, 7, np.random.default_rng(0))) == [0, 1, 2]


def test_ranking_lead_reports_each_season():
    rows = []
    for season, model_right in ((2024, True), (2025, False)):
        for day in range(10):
            actual = np.arange(10) * 5  # districts 5 to 9 are the true top 5
            for variant in ("model", "persistence"):
                right = model_right and variant == "model"
                p50 = actual if right or not model_right else actual[::-1]
                rows += [{"season": season, "issue_date": day, "variant": variant, "district": f"d{i}", "p50": p,
                          "actual": a} for i, (p, a) in enumerate(zip(p50, actual))]
    lead = intervals.ranking_lead(pd.DataFrame(rows), "model", "persistence")
    assert lead["top5"]["by_season"]["2024"]["points"] == 100 and lead["top5"]["by_season"]["2025"]["points"] == 0
    assert lead["top5"]["points"] == 50 and lead["n_days"] == 20


def _decision(low_2025=-5.0, model=95.0, naive=91.0):
    lead = {"of": "model", "against": "last_season_share", "points": 4.0, "low": 2.0, "high": 7.0,
            "by_season": {"2024": {"points": 4.5, "low": 2.0, "high": 9.0},
                          "2025": {"points": 3.5, "low": 2.2, "high": 6.4}}}
    pct = {"model": model, "last_season_share": naive, "equal_split": 80.0}
    return {"units": 500, "k": 20.0, "max_share": 0.25, "n_days": 148, "pct_of_oracle": pct,
            "by_season": {"2024": pct, "2025": pct}, "score_note": "SCORE", "lead_over_last_season_share": lead,
            "ranking_lead": {"of": "model", "against": "persistence", "n_days": 128,
                             "top5": {"points": 3.9, "low": 1.1, "high": 7.8,
                                      "by_season": {"2024": {"points": 7.9, "low": 4.4, "high": 12.3},
                                                    "2025": {"points": -0.3, "low": low_2025, "high": 5.8}}},
                             "spearman": {"difference": 0.07, "low": 0.04, "high": 0.12}}}


def test_a_pooled_lead_that_one_season_does_not_show_is_called_unproven():
    text = ev.interpret_ranking(_decision())
    assert "does not hold in 2025 (-0.3, -5.4 to +5.8)" not in text  # the interval given is the one in the data
    assert "does not hold in 2025 (-0.3, -5.0 to +5.8)" in text and "unproven" in text
    assert "holds in each test season" in ev.interpret_ranking(_decision(low_2025=0.5))


def test_decision_text_follows_the_result():
    text = " ".join(ev.interpret_decision(_decision()))
    assert "**95.0%**" in text and "91.0%" in text and "+4.0 points" in text and "holds in each test season" in text
    assert "scored higher" not in text
    assert "scored higher" in " ".join(ev.interpret_decision(_decision(model=88.0)))
    chance = _decision()
    chance["lead_over_last_season_share"]["low"] = -1.0
    assert "not clearly more than chance" in " ".join(ev.interpret_decision(chance))
    frame = ev.decision_frame(_decision())
    assert list(frame["key"]) == ["model", "last_season_share", "equal_split"]
    colours = ev.decision_figure(_decision()).data[0].marker.color
    assert colours[0] != colours[1] == colours[2]  # only the allocation in use is coloured


def test_method_lines_quote_the_codes_thresholds():
    text = " ".join(ev.audit_method_lines())
    assert "50 km²" in text and "at least 100" in text and "25%" in text and "1.5 times" in text and "never high" in text
    empty = {name: None for name in audit_view.TABLES}
    assert ev.audit_gap_frame(empty).empty
    assert ev.random_hit(22) == pytest.approx(5 / 22)


def test_reliability_counts_audits_and_dispatches(tmp_path):
    for folder, extra in (("audits", {"manual_edits": ["x"], "critic": {"warning": "w"}}), ("dispatch", {})):
        (tmp_path / folder).mkdir()
        (tmp_path / folder / "a.json").write_text(json.dumps(
            {"mode": "agent_revised+critic", "briefing": "two words", "verification": {"ok": True},
             "first_attempt_errors": ["e"], "model": "m", **extra}))
    rel = helpers.reliability(tmp_path).set_index("set")
    assert rel.loc["audit briefs", "Critic warnings"] == 1 and rel.loc["audit briefs", "edited by hand"] == 1
    assert rel.loc["dispatch briefs", "agent_revised"] == 1 and rel.loc["dispatch briefs", "read by the Critic"] == 0


# ---------- the page ----------

PAGE = str(config.ROOT / "src/psw/app/pages/4_Evidence.py")
needs_data = pytest.mark.skipif(ev.load_decision() is None, reason="processed data not present")


def _run():
    from streamlit.testing.v1 import AppTest
    return AppTest.from_file(PAGE, default_timeout=60).run()


def _text(at):
    return " ".join(x.value for kind in (at.markdown, at.caption, at.info) for x in kind)


@needs_data
def test_evidence_page_shows_the_part_2_results():
    at = _run()
    assert not at.exception, [e.value for e in at.exception]
    text, d = _text(at), ev.load_decision()
    for line in ev.interpret_decision(d) + [ev.interpret_ranking(d)]:
        assert line in text
    assert audit_view.CAPTION in [c.value for c in at.caption] and "not the chance that a new run passes" in text
    assert d["score_note"] in text


def test_evidence_page_says_what_to_run_when_part_2_files_are_missing(monkeypatch, tmp_path):
    import shutil
    for name in ("metrics.json", "retrieval_eval.json"):
        shutil.copy(config.PROCESSED / name, tmp_path)
    monkeypatch.setattr(config, "PROCESSED", tmp_path)
    at = _run()
    assert not at.exception, [e.value for e in at.exception]
    text = _text(at)
    assert "No decision backtest found" in text and "No fire-risk backtest found" in text and "No audit tables" in text


def _forward_report():
    crit = lambda met: [{"name": n, "met": met, "value": 0.6 if "ranking" in n or "band" in n else 0.03 if "persistence" in n else 6.0,  # noqa: E731
                         "rule": "r"} for n in ("ranking holds", "not worse than persistence",
                                                 "allocation beats the naive split", "band is honest")]
    variants = {v: {"top5_hit": 0.6, "spearman": 0.6, "coverage80": 0.8} for v in ("model", "persistence", "climatology")}
    arm = lambda met, end: {"first_issue": "2026-09-15", "last_issue": end, "data_end": end, "variants": variants,  # noqa: E731
                            "allocation_pct_of_oracle": {"model": 95.0, "last_season_share": 89.0},
                            "allocation_lead": {"points": 6.0, "low": 1.0, "high": 9.0},
                            "ranking_lead": {"against": "persistence", "top5": {"points": 1.0, "low": -3.0, "high": 4.0}},
                            "criteria": crit(met), "detections_same_dates_by_season": {"2025": 100, "2026": 80},
                            "post_hoc_backtest_same_window": {"2025": {"model": {"spearman": 0.7}}}}
    return {"season": 2026, "chosen_variant": "model", "frozen_code_commit": "abc", "trained_on": [2022, 2025],
            "noaa20_scale": 0.93, "notes": [], "arms": {"snpp": arm(True, "2026-10-29"), "noaa20": arm(False, "2026-11-23")}}


def test_forward_text_follows_the_report():
    r = _forward_report()
    text = " ".join(ev.forward_summary(r))
    assert "S-NPP arm met 4 of 4, the NOAA-20 arm 0 of 4" in text and "+6.0 points" in text
    assert "lower than every backtest season" in text and "added after seeing the result" in text
    lines = ev.criteria_lines(r, "noaa20")
    assert all(line.startswith("❌ not met") for line in lines) and "Spearman 0.60" in lines[0]
    frame = ev.forward_frame(r, None, None)
    assert list(frame.columns) == ["measure", "2026, S-NPP (15 Sep to 29 Oct)", "2026, NOAA-20 stand-in (15 Sep to 23 Nov)"]
    assert "80 in 2026 against 100 in 2025" in ev.detection_note(r)
    assert ev.load_forward(Path("/nonexistent")) is None


@pytest.mark.skipif(ev.load_forward() is None, reason="forward test not run")
def test_evidence_page_shows_the_forward_test():
    at = _run()
    assert not at.exception, [e.value for e in at.exception]
    text = _text(at)
    for line in ev.forward_summary(ev.load_forward()):
        assert line in text
