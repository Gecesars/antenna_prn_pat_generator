from __future__ import annotations

import math

import numpy as np

from .utils import njit_if_available


def deg2rad_inplace_python(deg: np.ndarray, out_rad: np.ndarray) -> None:
    n = int(min(len(deg), len(out_rad)))
    k = math.pi / 180.0
    for i in range(n):
        out_rad[i] = float(deg[i]) * k


def trapz_python(x_rad: np.ndarray, y: np.ndarray) -> float:
    n = int(min(len(x_rad), len(y)))
    if n < 2:
        return 0.0
    total = 0.0
    for i in range(n - 1):
        dx = float(x_rad[i + 1]) - float(x_rad[i])
        total += 0.5 * (float(y[i]) + float(y[i + 1])) * dx
    return float(total)


deg2rad_inplace_numba = njit_if_available(parallel=False, fastmath=False)(deg2rad_inplace_python)
trapz_numba = njit_if_available(parallel=False, fastmath=False)(trapz_python)

