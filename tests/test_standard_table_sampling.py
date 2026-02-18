import numpy as np

from deep3 import PATConverterApp


def test_horizontal_table_sampling_uses_fixed_5deg_grid():
    ang = np.arange(-180.0, 181.0, 1.0, dtype=float)
    val = 0.6 + 0.4 * np.cos(np.deg2rad(ang)) ** 2
    t_ang, t_val = PATConverterApp._table_points_horizontal(None, ang, val)

    assert t_ang[0] == -180.0
    assert t_ang[-1] == 180.0
    assert len(t_ang) == 73
    assert np.allclose(np.diff(t_ang), 5.0)
    assert t_val.shape == t_ang.shape


def test_vertical_table_sampling_uses_fixed_1deg_grid():
    ang = np.arange(-90.0, 90.0 + 0.1, 0.1, dtype=float)
    val = 0.5 + 0.5 * np.cos(np.deg2rad(ang)) ** 2
    t_ang, t_val = PATConverterApp._table_points_vertical(None, ang, val)

    assert t_ang[0] == -90.0
    assert t_ang[-1] == 90.0
    assert len(t_ang) == 181
    assert np.allclose(np.diff(t_ang), 1.0)
    assert t_val.shape == t_ang.shape
