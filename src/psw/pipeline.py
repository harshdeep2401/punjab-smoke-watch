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


COMMANDS = {"ingest": cmd_ingest}


def main() -> None:
    p = argparse.ArgumentParser(prog="psw.pipeline")
    p.add_argument("command", choices=sorted(COMMANDS))
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    COMMANDS[args.command](args)


if __name__ == "__main__":
    main()
