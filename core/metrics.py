from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np

from core.angles import ang_dist_deg


def _prepare_series(angles_deg: np.ndarray, mag_db: np.ndarray):
    a = np.asarray(angles_deg, dtype=float).reshape(-1)
    y = np.asarray(mag_db, dtype=float).reshape(-1)
    if a.size == 0 or y.size == 0 or a.size != y.size:
        raise ValueError("Invalid series.")
    idx = np.argsort(a)
    return a[idx], y[idx]


def _cross_x(x1: float, y1: float, x2: float, y2: float, yth: float) -> float:
    if abs(y2 - y1) < 1e-12:
        return float(x1)
    t = (yth - y1) / (y2 - y1)
    t = max(0.0, min(1.0, t))
    return float(x1 + t * (x2 - x1))


def beamwidth_xdb(
    angles_deg: np.ndarray,
    mag_db: np.ndarray,
    xdb: float,
    around_peak: bool = True,
    peak_ang: Optional[float] = None,
    wrap: bool = False,
) -> Dict[str, float]:
    """
    Compute X dB beamwidth around peak.
    Returns dict with width_deg, left_deg, right_deg, peak_deg, peak_db, threshold_db.
    """
    a, y = _prepare_series(angles_deg, mag_db)
    if a.size < 3:
        return {
            "width_deg": float("nan"),
            "left_deg": float("nan"),
            "right_deg": float("nan"),
            "peak_deg": float("nan"),
            "peak_db": float("nan"),
            "threshold_db": float("nan"),
        }

    xdb = float(abs(xdb))

    if wrap:
        a = np.concatenate([a, a + 360.0])
        y = np.concatenate([y, y])

    if peak_ang is not None:
        pa = float(peak_ang)
        if wrap:
            pa = ((pa + 180.0) % 360.0) - 180.0
            pa = pa if pa >= a[0] else pa + 360.0
        ipk = int(np.argmin(np.abs(a - pa)))
    else:
        ipk = int(np.argmax(y))

    peak_db = float(y[ipk])
    peak_deg = float(a[ipk])
    thr = peak_db - xdb

    # Search left crossing
    left = float("nan")
    i = ipk
    while i > 0:
        if y[i - 1] < thr <= y[i] or y[i - 1] > thr >= y[i] or (y[i] >= thr and y[i - 1] < thr):
            left = _cross_x(a[i - 1], y[i - 1], a[i], y[i], thr)
            break
        i -= 1

    # Search right crossing
    right = float("nan")
    i = ipk
    while i < len(a) - 1:
        if y[i] >= thr > y[i + 1] or y[i] <= thr < y[i + 1] or (y[i] >= thr and y[i + 1] < thr):
            right = _cross_x(a[i], y[i], a[i + 1], y[i + 1], thr)
            break
        i += 1

    width = float("nan")
    if math.isfinite(left) and math.isfinite(right):
        width = float(right - left)
        if wrap:
            width = width % 360.0
            if width > 180.0:
                width = 360.0 - width

    if wrap and math.isfinite(left):
        left = ((left + 180.0) % 360.0) - 180.0
    if wrap and math.isfinite(right):
        right = ((right + 180.0) % 360.0) - 180.0
    if wrap:
        peak_deg = ((peak_deg + 180.0) % 360.0) - 180.0

    return {
        "width_deg": float(width),
        "left_deg": float(left),
        "right_deg": float(right),
        "peak_deg": float(peak_deg),
        "peak_db": float(peak_db),
        "threshold_db": float(thr),
    }
