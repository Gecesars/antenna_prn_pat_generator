import numpy as np
import pytest

from eftx_aedt_live.farfield import (
    _coerce_cut_values,
    _coerce_grid_values,
    _dedupe_cut_points,
    _extract_solution_xy,
    theta_to_elev,
)


def test_coerce_cut_values_accepts_2xN_matrix():
    x = np.linspace(0.0, 360.0, 361)
    y = np.vstack([np.linspace(-10.0, 0.0, 361), np.linspace(-20.0, -5.0, 361)])
    out = _coerce_cut_values(x, y, "dB(GainTotal)")
    assert out.shape == (361,)
    assert np.isclose(out[0], -10.0)
    assert np.isclose(out[-1], 0.0)


def test_coerce_cut_values_accepts_Nx2_matrix():
    x = np.linspace(0.0, 180.0, 181)
    y = np.column_stack([np.linspace(1.0, 2.0, 181), np.linspace(3.0, 4.0, 181)])
    out = _coerce_cut_values(x, y, "dB(GainTotal)")
    assert out.shape == (181,)
    assert np.isclose(out[0], 1.0)
    assert np.isclose(out[-1], 2.0)


def test_coerce_cut_values_prefers_pair_with_axis_row():
    x = np.linspace(-180.0, 180.0, 361)
    v = np.linspace(-5.0, 0.0, 361)
    y = np.vstack([x, v])  # common [x, y] payload from PyAEDT
    out = _coerce_cut_values(x, y, "dB(GainTotal)")
    assert np.allclose(out, v)


def test_coerce_cut_values_from_flattened_traces():
    x = np.linspace(-180.0, 180.0, 361)
    first = np.linspace(-3.0, 0.0, 361)
    second = np.linspace(-10.0, -7.0, 361)
    y = np.concatenate([first, second])
    out = _coerce_cut_values(x, y, "dB(GainTotal)")
    assert out.shape == (361,)
    assert np.allclose(out, first)


def test_dedupe_cut_points_uses_mean_per_angle():
    ang = np.array([-180.0, -180.0, -179.0, -179.0, 0.0])
    val = np.array([-10.0, -3.0, -5.0, -7.0, -1.0])
    out_a, out_v = _dedupe_cut_points(ang, val)
    assert np.allclose(out_a, [-180.0, -179.0, 0.0])
    assert np.allclose(out_v, [-6.5, -6.0, -1.0])


def test_coerce_grid_values_accepts_transposed_matrix():
    t = np.linspace(0.0, 180.0, 5)
    p = np.array([0.0, 90.0])
    z_tp = np.arange(10, dtype=float).reshape(5, 2)
    z_pt = z_tp.T
    out = _coerce_grid_values(t, p, z_pt)
    assert out is not None
    assert out.shape == (5, 2)
    assert np.allclose(out, z_tp)


def test_coerce_grid_values_rejects_pair_like_matrix():
    t = np.linspace(-180.0, 180.0, 361)
    p = np.array([0.0, 90.0])
    pair_like = np.vstack([t, np.linspace(-3.0, 0.0, 361)])
    out = _coerce_grid_values(t, p, pair_like)
    assert out is None


def test_theta_to_elev_supports_signed_theta():
    th = np.array([-180.0, -90.0, 0.0, 90.0, 180.0])
    ev = theta_to_elev(th)
    assert np.all(ev >= -90.0 - 1e-9)
    assert np.all(ev <= 90.0 + 1e-9)
    assert np.isclose(ev[2], 90.0)


def test_theta_to_elev_keeps_elevation_like_input():
    th = np.linspace(-90.0, 90.0, 181)
    ev = theta_to_elev(th)
    assert np.allclose(ev, th)


class _DummySolutionData:
    def __init__(self, x, y, expr="dB(GainTotal)"):
        self.primary_sweep_values = np.asarray(x, dtype=float)
        self._y = np.asarray(y, dtype=float)
        self.expressions = [expr]

    def get_expression_data(self, expr):
        return self._y


def test_extract_solution_xy_accepts_solution_like_object():
    x = np.linspace(-90.0, 90.0, 181)
    y = np.linspace(-20.0, 0.0, 181)
    sx, sy = _extract_solution_xy(_DummySolutionData(x, y), "dB(GainTotal)")
    assert np.allclose(sx, x)
    assert np.allclose(sy, y)


def test_extract_solution_xy_rejects_bool_result():
    with pytest.raises(TypeError):
        _extract_solution_xy(False, "dB(GainTotal)")
