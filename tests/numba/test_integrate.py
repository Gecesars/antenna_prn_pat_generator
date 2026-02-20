from __future__ import annotations

import math

import numpy as np

from core.numba_kernels.integrate import deg2rad_inplace_numba, trapz_numba


def test_deg2rad_inplace_matches_numpy():
    deg = np.array([-180.0, -90.0, 0.0, 90.0, 180.0], dtype=np.float64)
    out = np.empty_like(deg)
    deg2rad_inplace_numba(deg, out)
    assert np.allclose(out, np.deg2rad(deg), atol=1e-12, rtol=0.0)


def test_trapz_matches_numpy():
    x = np.linspace(0.0, math.pi, 2049, dtype=np.float64)
    y = np.sin(x)
    ref = float(np.trapezoid(y, x))
    got = float(trapz_numba(x, y))
    assert abs(got - ref) <= 1e-10

