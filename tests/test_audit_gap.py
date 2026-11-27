"""Gap ranges and the other audit tables, on small synthetic data. No network."""
import json

import pandas as pd
import pytest

from psw.audit import gap
from psw.audit.sensors import AFTERNOON, MORNING, NIGHT

D = "Sangrur"
SEASONS = (2022, 2023, 2024)


def fires(spec: dict) -> pd.DataFrame:
    """spec: {(district or None, sensor, season): (day detections, night detections)}"""
    rows = []
    for (district, sensor, season), (day, night) in spec.items():
        label = MORNING if sensor == "modis" else AFTERNOON
        rows += [{"district": district, "sensor": sensor, "season": season, "pass_label": label}] * day
        rows += [{"district": district, "sensor": sensor, "season": season, "pass_label": NIGHT}] * night
    return pd.DataFrame(rows)


def area(spec: dict) -> pd.DataFrame:
    return pd.DataFrame([{"district": d, "season": s, "burned_km2": float(v)} for (d, s), v in spec.items()])


def official(spec: dict) -> pd.DataFrame:
    return pd.DataFrame([{"district": d, "season": s, "count": v, "source_url": f"https://example.org/{s}",
                          "method_note": "satellite"} for (d, s), v in spec.items()])


def scenario(snpp=(1000, 1000, 100), burned=(1000, 1000, 700), off=None, night=(2, 2, 40), modis=(100, 100, 40),
             district=D):
    """One district over three seasons, so the Punjab rows equal the district's."""
    ms = {}
    for season, n, nt, m in zip(SEASONS, snpp, night, modis):
        ms[(district, "snpp", season)] = (n - nt, nt)
        ms[(district, "modis", season)] = (m, 0)
    ba = {(district, s): v for s, v in zip(SEASONS, burned)}
    off = snpp if off is None else off
    return fires(ms), area(ba), official({(district, s): v for s, v in zip(SEASONS, off)})


def ranges(*data, bases=(2022, 2023)) -> pd.DataFrame:
    ms, ba, off = data
    return gap.gap_range(ms, ba, off, gap.timing_indicators(ms), bases).set_index(["district", "season"])


def row(table, district="Punjab", season=2024):
    return table.loc[(district, season)]


def test_gap_is_the_fall_in_detections_per_burned_km2():
    r = row(ranges(*scenario()))  # detections -90%, burned area -30%
    assert r["gap_low"] == pytest.approx(1 - 0.1 / 0.7) and r["gap_high"] == pytest.approx(0.857, abs=0.001)
    assert r["detected_change_pct"] == pytest.approx(-90) and r["reference_change_pct"] == pytest.approx(-30)
    assert r["detected_basis"] == "S-NPP detections against 2022"
    assert r["methods"].split("; ") == ["snpp / burned_area, base 2022", "official / burned_area, base 2022",
                                        "snpp / burned_area, base 2023", "official / burned_area, base 2023"]


def test_burned_area_falling_more_than_detections_gives_a_negative_unclipped_gap():
    r = row(ranges(*scenario(snpp=(1000, 1000, 900), burned=(1000, 1000, 300))))
    assert r["gap_high"] == pytest.approx(-2.0) and r["gap_low"] == pytest.approx(-2.0)
    assert r["direction"] == "no sign of a detection gap" and r["confidence"] == "low"


def test_a_range_that_straddles_zero_is_low_and_unclear():
    r = row(ranges(*scenario(snpp=(1000, 1000, 100), off=(1000, 1000, 900))))
    assert r["gap_low"] < 0 < r["gap_high"]
    assert r["confidence"] == "low" and r["direction"] == "direction unclear"


@pytest.mark.parametrize("kwargs, reason", [
    ({"burned": (1000, 1000, 49)}, "burned area under 50 km² in 2024 (49 km²)"),
    ({"burned": (49, 49, 700)}, "burned area under 50 km² in 2022 (49 km²)"),
    ({"snpp": (99, 99, 20), "night": (2, 2, 4)}, "base count under 100 in 2022 (99)"),
])
def test_small_numbers_give_no_range_and_a_reason(kwargs, reason):
    r = row(ranges(*scenario(**kwargs)))
    assert r["confidence"] == "insufficient evidence" and pd.isna(r["gap_low"]) and pd.isna(r["gap_high"])
    assert reason in r["reason"] and pd.isna(r["methods"]) and pd.isna(r["detected_change_pct"])


def test_the_thresholds_themselves_pass():
    r = row(ranges(*scenario(snpp=(100, 100, 20), burned=(50, 50, 50), night=(2, 2, 4))))
    assert r["gap_low"] == pytest.approx(0.8)


def test_only_the_usable_pairs_enter_the_range():
    r = row(ranges(*scenario(burned=(10, 1000, 700))))  # 2022 is too small, 2023 is fine
    assert r["methods"] == "snpp / burned_area, base 2023; official / burned_area, base 2023"
    assert r["detected_basis"] == "S-NPP detections against 2023"


def test_the_first_season_has_no_row_because_it_has_no_earlier_base():
    t = ranges(*scenario())
    assert sorted(t.loc["Punjab"].index) == [2023, 2024]


def test_missing_official_counts_leave_the_snpp_estimates():
    ms, ba, off = scenario()
    r = row(ranges(ms, ba, off[off["season"] != 2024]))
    assert r["methods"] == "snpp / burned_area, base 2022; snpp / burned_area, base 2023"


def test_punjab_is_medium_only_with_both_timing_indicators_and_a_large_enough_gap():
    t = ranges(*scenario())
    assert row(t)["confidence"] == "medium"
    assert row(t, D)["confidence"] == "low"  # the same numbers, but a district is never above low
    assert row(ranges(*scenario(night=(2, 2, 0))))["confidence"] == "low"  # no night support
    assert row(ranges(*scenario(modis=(100, 100, 12))))["confidence"] == "low"  # no MODIS support
    small = row(ranges(*scenario(snpp=(1000, 1000, 560), burned=(1000, 1000, 700), night=(2, 2, 60), modis=(100, 100, 90))))
    assert 0 < small["gap_low"] < 0.25 and small["confidence"] == "low"


def test_nothing_is_ever_high_and_every_row_carries_the_caveat():
    t = ranges(*scenario(snpp=(10000, 10000, 100), night=(2, 2, 90), modis=(1000, 1000, 900)))
    assert set(t["confidence"]) <= {"medium", "low", "insufficient evidence"}
    assert (t["caveat"] == gap.GAP_CAVEAT).all() and "does not include" in gap.GAP_CAVEAT


def test_noaa_satellites_never_change_the_range():
    ms, ba, off = scenario()
    extra = fires({(D, s, y): (5000, 0) for s in ("noaa20", "noaa21") for y in SEASONS})
    a, b = ranges(ms, ba, off), ranges(pd.concat([ms, extra]), ba, off)
    assert a["gap_low"].equals(b["gap_low"]) and a["gap_high"].equals(b["gap_high"])


def test_wilson_interval():
    lo, hi = gap.wilson_interval(10, 100)
    assert lo == pytest.approx(0.0552, abs=1e-4) and hi == pytest.approx(0.1744, abs=1e-4)
    assert gap.wilson_interval(0, 50)[0] == pytest.approx(0, abs=1e-12)
    assert all(pd.isna(v) for v in gap.wilson_interval(0, 0))


def timing(night, modis=(100, 100, 100), snpp=(1000, 1000, 1000)):
    ms, _, _ = scenario(snpp=snpp, night=night, modis=modis)
    return gap.timing_indicators(ms).set_index("season")


def test_night_support_needs_the_whole_interval_above_the_base_interval():
    assert not timing((10, 10, 20)).loc[2024, "supports_night"]  # higher share, but the intervals overlap
    t = timing((10, 10, 40))
    assert t.loc[2024, "ci_low"] > t.loc[2022, "ci_high"] and t.loc[2024, "supports_night"]
    assert not t.loc[2022, "supports_night"] and not t.loc[2023, "supports_night"]


def test_modis_support_starts_at_a_ratio_of_one_and_a_half():
    # S-NPP halves; MODIS keeps 75% (ratio exactly 1.5) or 74% (just below)
    at = timing((2, 2, 2), modis=(100, 100, 75), snpp=(1000, 1000, 500))
    below = timing((2, 2, 2), modis=(100, 100, 74), snpp=(1000, 1000, 500))
    assert at.loc[2024, "modis_vs_viirs"] == pytest.approx(1.5) and at.loc[2024, "supports_modis"]
    assert not below.loc[2024, "supports_modis"] and not at.loc[2022, "supports_modis"]


def test_tables_count_punjab_only_and_fill_zeros():
    ms = fires({("Sangrur", "snpp", 2022): (8, 2), ("Moga", "snpp", 2022): (5, 0), (None, "snpp", 2022): (100, 100),
                ("Sangrur", "snpp", 2023): (4, 0), ("Sangrur", "noaa21", 2023): (6, 0), ("Moga", "modis", 2022): (3, 7)})
    trend = gap.sensor_trend(ms).set_index(["district", "sensor", "season"])
    assert trend.loc[("Punjab", "snpp", 2022), "count"] == 15  # fires outside the districts are left out
    assert trend.loc[("Moga", "snpp", 2023), "count"] == 0 and trend.loc[("Moga", "snpp", 2023), "change_pct"] == -100
    assert trend.loc[("Punjab", "noaa21", 2023), "base_season"] == 2023
    assert pd.isna(trend.loc[("Moga", "noaa21", 2023), "change_pct"])  # nothing to compare with

    prof = gap.diurnal_profile(ms).set_index(["district", "sensor", "season", "pass_label"])
    assert prof.loc[("Sangrur", "snpp", 2022, NIGHT), "share"] == pytest.approx(0.2)
    assert prof.loc[("Sangrur", "snpp", 2022, MORNING), "count"] == 0

    night = gap.night_share(ms).set_index(["district", "season"])
    assert night.loc[("Punjab", 2022), "night_share"] == pytest.approx(2 / 15)  # MODIS is not part of it
    assert night.loc[("Moga", 2023), "total"] == 0 and pd.isna(night.loc[("Moga", 2023), "night_share"])


def test_burned_vs_detected_and_official_totals():
    ms = fires({("Sangrur", "snpp", 2022): (200, 0), ("Moga", "snpp", 2022): (100, 0),
                ("Sangrur", "snpp", 2023): (50, 0), ("Moga", "snpp", 2023): (50, 0)})
    ba = area({("Sangrur", 2022): 100, ("Moga", 2022): 100, ("Sangrur", 2023): 100, ("Moga", 2023): 50})
    bvd = gap.burned_vs_detected(ms, ba).set_index(["district", "season"])
    assert bvd.loc[("Punjab", 2023), "burned_km2"] == 150
    assert bvd.loc[("Punjab", 2023), "detections_per_km2"] == pytest.approx(100 / 150)
    assert bvd.loc[("Sangrur", 2023), "detections_change_pct"] == pytest.approx(-75)

    off = official({("Sangrur", 2022): 300, ("Moga", 2022): 150, ("Sangrur", 2023): 90})  # Moga is missing in 2023
    t = gap.official_vs_satellite(ms, off).set_index(["district", "season"])
    assert t.loc[("Punjab", 2022), "official"] == 450 and t.loc[("Punjab", 2022), "ratio"] == pytest.approx(1.5)
    assert ("Punjab", 2023) not in t.index  # never a partial total


def test_summary_is_strict_json_with_the_headline_numbers():
    ms, ba, off = scenario(burned=(1000, 1000, 700))
    summary = gap.audit_summary(gap.audit_tables(ms, ba, off))
    json.dumps(summary, allow_nan=False)
    s = summary["seasons"]["2024"]
    assert s["sensors"]["snpp"]["change_pct"] == -90.0 and s["burned_change_pct"] == -30.0
    assert s["gap"]["low_pct"] == 85.7 and s["gap"]["confidence"] == "medium" and s["gap"]["reason"] is None
    assert summary["seasons"]["2022"]["gap"] is None
    assert summary["caveat"] == gap.GAP_CAVEAT and "three explanations" in summary["explanations"]
