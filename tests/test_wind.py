import numpy as np

from psw.processing.wind import dir_from_uv, uv_from_dir_speed


def test_north_wind_blows_south():
    u, v = uv_from_dir_speed(0.0, 5.0)
    assert v < 0 and abs(u) < 1e-9


def test_west_wind_blows_east():
    u, v = uv_from_dir_speed(270.0, 5.0)
    assert u > 0 and abs(v) < 1e-9


def test_roundtrip():
    for d in (0, 45, 90, 180, 270, 359):
        u, v = uv_from_dir_speed(d, 3.0)
        assert abs(((dir_from_uv(u, v) - d + 180) % 360) - 180) < 1e-6


def test_vector_mean_of_350_and_10_is_north():
    u, v = uv_from_dir_speed(np.array([350.0, 10.0]), np.array([4.0, 4.0]))
    d = float(dir_from_uv(u.mean(), v.mean()))
    assert min(d, 360 - d) < 1e-6
