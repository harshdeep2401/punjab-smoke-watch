import pandas as pd
import pytest

from psw import config
from psw.response import machines

RAW = config.DATA / "manual" / "crm_machines.csv"


def test_committed_table_matches_the_totals_printed_in_the_report():
    raw = pd.read_csv(RAW)
    assert list(raw.columns) == machines.COLUMNS
    assert len(raw) == 23 * 17 and raw["count"].sum() == 146_540
    by_type = raw.groupby("machine_type")["count"].sum()
    printed = {"happy_seeder": 13719, "super_seeder": 61951, "paddy_straw_chopper": 10202, "mulcher": 7608,
               "rmb_plough": 9640, "zero_till_drill": 23843, "super_sms": 6334, "rotary_slasher_shrub_master": 2322,
               "shrub_master": 83, "cutter_cum_spreader": 970, "rotavator": 3835, "baler": 2183, "rake": 2039,
               "crop_reaper_binder": 86, "smart_seeder": 78, "surface_seeder": 1045, "tractor": 602}
    assert by_type.to_dict() == printed
    by_district = raw.groupby("district")["count"].sum()
    assert by_district["Sangrur"] == 12520 and by_district["Malerkotla"] == 380 and by_district["Amritsar"] == 5336


def test_loading_folds_malerkotla_into_sangrur():
    stock = machines.load_machine_stock()
    assert stock["district"].nunique() == 22 and "Malerkotla" not in set(stock["district"])
    assert stock["count"].sum() == 146_540
    sangrur = stock[stock["district"] == "Sangrur"]
    assert sangrur["count"].sum() == 12520 + 380
    assert sangrur.set_index("machine_type").loc["super_seeder", "count"] == 6363 + 241
    assert sangrur["method_note"].str.endswith("includes Malerkotla (D10)").all()
    assert not stock.loc[stock["district"] != "Sangrur", "method_note"].str.contains("includes").any()


@pytest.mark.parametrize("change, message", [
    ({"count": "-1"}, "whole number"),
    ({"district": "Nowhere"}, "unknown district"),
    ({"source_url": ""}, "blank source_url"),
    ({"source_url": "https://a.example/x https://b.example/y"}, "exactly one web address"),
    ({"machine_type": "super_seeder"}, "duplicate"),
])
def test_a_bad_hand_edit_is_refused(tmp_path, change, message):
    raw = pd.read_csv(RAW, dtype=str)
    for column, value in change.items():
        raw.loc[0, column] = value
    path = tmp_path / "stock.csv"
    raw.to_csv(path, index=False)
    with pytest.raises(ValueError, match=message):
        machines.load_machine_stock(path)
