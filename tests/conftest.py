import pytest

from psw import config


@pytest.fixture(autouse=True)
def isolated_raw(tmp_path, monkeypatch):
    """Every test writes its caches to a temp dir and never sleeps."""
    monkeypatch.setattr(config, "RAW", tmp_path / "raw")
    from psw.ingest import firms
    monkeypatch.setattr(firms, "SLEEP", 0)
