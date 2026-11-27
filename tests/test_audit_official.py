import pytest

from psw import config
from psw.audit import official

DISTRICTS = {"Sangrur", "Moga", "Malerkotla"}
URL = "https://example.org/report.pdf"
HEADER = "season,district,count,source_url,method_note"


def load(tmp_path, monkeypatch, *rows):
    monkeypatch.setattr(official, "known_districts", lambda _geo=None: DISTRICTS)
    path = tmp_path / "official_counts.csv"
    path.write_text("\n".join([HEADER, *rows]) + "\n")
    return official.load_official(path)


def row(season="2023", district="Moga", count="12", url=URL, note="satellite"):
    return f"{season},{district},{count},{url},{note}"


def test_malerkotla_is_added_to_sangrur(tmp_path, monkeypatch):
    out = load(tmp_path, monkeypatch, row(district="Sangrur", count="100"), row(district="Malerkotla", count="7"),
               row(count="0"))
    assert out[["district", "count"]].values.tolist() == [["Moga", 0], ["Sangrur", 107]]
    assert out.loc[1, "method_note"] == "satellite; includes Malerkotla (D10)"
    assert out["season"].tolist() == [2023, 2023]


@pytest.mark.parametrize("bad, message", [
    (row(district="Mogga"), r"unknown district \['Mogga'\] \(line 2\)"),
    (row(count=""), "blank count"),
    (row(note=""), "blank method_note"),
    (row(url=""), "blank source_url"),
    (row(count="1.5"), "whole number"),
    (row(count="-3"), "whole number"),
    (row(season="23"), "not a year"),
    (row(url="the 2023 report"), "exactly one web address"),
    (row(url=f"{URL};{URL}"), "exactly one web address"),
    (row(district="Malerkotla"), "Malerkotla has no Sangrur row"),
])
def test_bad_rows_fail(tmp_path, monkeypatch, bad, message):
    with pytest.raises(ValueError, match=message):
        load(tmp_path, monkeypatch, bad)


def test_duplicate_rows_fail(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match=r"duplicate season and district \(line 2, 3\)"):
        load(tmp_path, monkeypatch, row(), row(count="13"))


def test_empty_line_fails(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match=r"blank season \(line 3\)"):
        load(tmp_path, monkeypatch, row(), "", row(district="Sangrur"))


def test_merge_with_different_sources_fails(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="different sources"):
        load(tmp_path, monkeypatch, row(district="Sangrur"), row(district="Malerkotla", url="https://example.org/other"))


def test_committed_file_matches_the_state_totals_in_the_reports():
    out = official.load_official()
    assert out.groupby("season")["count"].sum().to_dict() == {2022: 49922, 2023: 36663, 2024: 10909, 2025: 5114}
    assert out.groupby("season").size().to_dict() == {s: 22 for s in config.SEASONS}
    assert set(out["district"]) == official.known_districts() - set(official.MERGE_INTO)
