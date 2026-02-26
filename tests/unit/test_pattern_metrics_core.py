from __future__ import annotations

import numpy as np

from core.analysis.pattern_metrics import (
    directivity_2d_cut,
    hpbw_cut_1d,
    integrate_power_numpy,
    metrics_cut_1d,
    smart_decimate_indices,
)


def _gaussian_cut(n: int = 721, sigma_deg: float = 18.0):
    ang = np.linspace(-180.0, 180.0, n, dtype=float)
    e = np.exp(-0.5 * (ang / float(sigma_deg)) ** 2)
    return ang, e


def test_metrics_cut_1d_basic_invariants():
    ang, e = _gaussian_cut()
    m = metrics_cut_1d(ang, e, span_mode=1, xdb=3.0, use_numba=False)

    assert int(m["points"]) == int(ang.size)
    assert float(m["peak_db"]) <= 0.1
    assert abs(float(m["peak_angle_deg"])) <= 1.0
    assert float(m["hpbw_deg"]) > 0.0
    assert np.isfinite(float(m["d2d_lin"]))
    assert np.isfinite(float(m["d2d_db"]))


def test_hpbw_and_directivity_wrappers():
    ang, e = _gaussian_cut()
    hpbw = hpbw_cut_1d(ang, e, xdb=3.0, use_numba=False)
    d2d = directivity_2d_cut(ang, e, span_mode=1, use_numba=False)
    assert np.isfinite(float(hpbw))
    assert float(hpbw) > 0.0
    assert np.isfinite(float(d2d))
    assert float(d2d) > 0.0


def test_integrate_power_numpy_returns_finite():
    ang, e = _gaussian_cut()
    p = integrate_power_numpy(ang, e)
    assert np.isfinite(float(p))
    assert float(p) > 0.0


def test_smart_decimate_indices_bounds_and_order():
    ang, e = _gaussian_cut(n=1801, sigma_deg=22.0)
    idx = smart_decimate_indices(ang, e, target_rows=73, use_numba=False)
    assert idx.ndim == 1
    assert idx.size <= 73
    assert idx.size >= 8
    assert np.all(idx >= 0)
    assert np.all(idx < ang.size)
    assert np.all(np.diff(idx) >= 0)

