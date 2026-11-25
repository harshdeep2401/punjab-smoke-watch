"""Wind maths in the meteorological 'from' convention (degrees the wind blows FROM, clockwise from north)."""
from __future__ import annotations

import numpy as np


def uv_from_dir_speed(dir_deg, speed):
    """Return (u, v): u eastward, v northward components of the air motion.

    A north wind (from 0 deg) blows toward the south, so v < 0.
    """
    rad = np.deg2rad(dir_deg)
    return -np.asarray(speed) * np.sin(rad), -np.asarray(speed) * np.cos(rad)


def dir_from_uv(u, v):
    """Meteorological 'from' direction in [0, 360) for components (u, v)."""
    return np.mod(np.rad2deg(np.arctan2(-np.asarray(u), -np.asarray(v))), 360.0)
