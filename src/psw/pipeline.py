"""CLI entry point: uv run python -m psw.pipeline <command>. I/O lives here."""
from __future__ import annotations

import argparse
import logging

from psw import config


def cmd_ingest(_args) -> None:
    from psw.ingest import airquality, firms, weather

    if not config.FIRMS_MAP_KEY:
        raise SystemExit("FIRMS_MAP_KEY is not set in .env")
    fires = firms.load_all_fires(config.SEASONS)
    print(f"fires: {len(fires)} rows; per season: {fires.groupby('season').size().to_dict()}")
    aq = airquality.daily_pm25(config.SEASONS)
    print(f"pm25 daily: {len(aq)} rows, {int(aq['pm25_mean'].isna().sum())} NaN")
    wx = weather.daily_weather(config.SEASONS)
    print(f"weather daily: {len(wx)} rows, {int(wx.isna().any(axis=1).sum())} rows with NaN")


def cmd_process(_args) -> None:
    import geopandas as gpd

    from psw.ingest import airquality, firms, weather
    from psw.processing import districts

    geo = config.GEO / "punjab_districts.geojson"
    names = sorted(gpd.read_file(geo)["district"])
    fires = districts.assign_districts(firms.load_all_fires(config.SEASONS), geo)
    for c in ("latitude", "longitude"):
        fires[c] = fires[c].round(4)
    config.PROCESSED.mkdir(parents=True, exist_ok=True)
    fires.to_parquet(config.PROCESSED / "fires.parquet", index=False)
    daily = districts.district_daily(fires, names, config.SEASONS)
    daily.to_parquet(config.PROCESSED / "district_daily.parquet", index=False)
    airquality.daily_pm25(config.SEASONS).to_parquet(config.PROCESSED / "pm25_daily.parquet", index=False)
    weather.daily_weather(config.SEASONS).to_parquet(config.PROCESSED / "weather_daily.parquet", index=False)
    print(f"fires: {len(fires)} rows, {int(fires['in_punjab'].sum())} inside Punjab; "
          f"district_daily: {len(daily)} rows for {len(names)} districts")


def cmd_train(_args) -> None:
    import pandas as pd

    from psw.models import evaluate
    from psw.processing import features as feat

    P = config.PROCESSED
    fires = pd.read_parquet(P / "fires.parquet")
    table = feat.build_features(fires, pd.read_parquet(P / "pm25_daily.parquet"), pd.read_parquet(P / "weather_daily.parquet"))
    table.to_parquet(P / "features.parquet", index=False)
    print(f"features: {len(table)} rows, {table.shape[1]} columns")
    m = evaluate.run_all(table)
    print(f"test seasons {m['seasons_test']}, conformal applied: {m['conformal_applied']}")
    print(f"{'variant':12s} {'MAE':>7s} {'pinball':>8s} {'cov80':>7s} {'width':>7s}")
    for name in evaluate.VARIANTS:
        o = m["variants"][name]["overall"]
        print(f"{name:12s} {o['mae']:7.2f} {o['pinball']:8.2f} {o['coverage80']:7.1%} {o['band_width']:7.1f}")
    print("top features (gain):", {k: round(v) for k, v in list(m["feature_importance"].items())[:6]})


def _later(phase: int):
    def run(_args) -> None:
        raise SystemExit(f"Not implemented yet (Phase {phase}).")
    return run


COMMANDS = {
    "ingest": cmd_ingest,
    "process": cmd_process,
    "train": cmd_train,
    "kb": _later(4),
    "brief": _later(5),
    "demo-cache": _later(7),
}


def cmd_all(args) -> None:
    for name in ("ingest", "process", "train"):  # extended as later phases land
        COMMANDS[name](args)


COMMANDS["all"] = cmd_all


def main() -> None:
    p = argparse.ArgumentParser(prog="psw.pipeline")
    p.add_argument("command", choices=sorted(COMMANDS))
    p.add_argument("--date")
    p.add_argument("--district")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    COMMANDS[args.command](args)


if __name__ == "__main__":
    main()
