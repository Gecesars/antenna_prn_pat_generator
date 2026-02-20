from __future__ import annotations

import math

import numpy as np

from core.analysis.pattern_metrics import directivity_2d_cut, metrics_cut_1d


def _d2d_reference(angles_deg: np.ndarray, e_lin: np.ndarray, span_mode: int) -> float:
    e = np.asarray(e_lin, dtype=np.float64).reshape(-1)
    a = np.asarray(angles_deg, dtype=np.float64).reshape(-1)
    e = np.abs(e)
    emax = float(np.max(e))
    if emax <= 1e-30:
        return float("nan")
    p = (e / emax) ** 2
    integ = float(np.trapezoid(p, np.deg2rad(a)))
    if integ <= 1e-30:
        return float("nan")
    span = 2.0 * math.pi if int(span_mode) == 0 else math.pi
    return float(span / integ)


def test_hpbw_and_d2d_vrp_against_reference():
    ang = np.arange(-90.0, 90.0 + 1e-9, 0.1, dtype=np.float64)
    e = np.clip(np.cos(np.deg2rad(ang)), 0.0, None)
    m = metrics_cut_1d(ang, e, span_mode=1, xdb=3.0, use_numba=True)

    assert abs(float(m["hpbw_deg"]) - 90.0) <= 0.2
    ref_d2d = _d2d_reference(ang, e, span_mode=1)
    assert math.isfinite(ref_d2d)
    assert abs(float(m["d2d_lin"]) - ref_d2d) / ref_d2d <= 1e-6
    assert abs(float(directivity_2d_cut(ang, e, span_mode=1)) - ref_d2d) / ref_d2d <= 1e-6


def test_hrp_front_to_back_and_d2d():
    ang = np.arange(-180.0, 180.0 + 1e-9, 1.0, dtype=np.float64)
    e = 1.0 + 0.5 * np.cos(np.deg2rad(ang))
    m = metrics_cut_1d(ang, e, span_mode=0, xdb=3.0, use_numba=True)

    expected_fb = 20.0 * math.log10(1.5 / 0.5)
    assert abs(float(m["fb_db"]) - expected_fb) <= 1e-2
    ref_d2d = _d2d_reference(ang, e, span_mode=0)
    assert abs(float(m["d2d_lin"]) - ref_d2d) / ref_d2d <= 1e-6


def test_first_null_detected_for_sinc_like_pattern():
    ang = np.arange(-90.0, 90.0 + 1e-9, 0.1, dtype=np.float64)
    e = np.abs(np.sinc(ang / 30.0))
    m = metrics_cut_1d(ang, e, span_mode=1, xdb=3.0, use_numba=True)
    assert math.isfinite(float(m["first_null_db"]))
    assert float(m["first_null_db"]) < -10.0


def test_numba_and_numpy_paths_are_close():
    ang = np.arange(-180.0, 180.0 + 1e-9, 1.0, dtype=np.float64)
    e = np.abs(np.cos(np.deg2rad(ang))) + 0.05
    mn = metrics_cut_1d(ang, e, span_mode=0, use_numba=False)
    mj = metrics_cut_1d(ang, e, span_mode=0, use_numba=True)

    for key in ("hpbw_deg", "first_null_db", "fb_db", "d2d_lin", "d2d_db", "peak_angle_deg"):
        av = float(mn[key])
        bv = float(mj[key])
        if math.isfinite(av) and math.isfinite(bv):
            assert abs(av - bv) <= 1e-2

