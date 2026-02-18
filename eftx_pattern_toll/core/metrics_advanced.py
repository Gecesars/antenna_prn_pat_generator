from __future__ import annotations

import math
from typing import Dict

import numpy as np

from core.metrics import beamwidth_xdb


def max_db(mag_db: np.ndarray) -> float:
    y = np.asarray(mag_db, dtype=float).reshape(-1)
    if y.size == 0:
        return float("nan")
    return float(np.max(y))


def min_db(mag_db: np.ndarray) -> float:
    y = np.asarray(mag_db, dtype=float).reshape(-1)
    if y.size == 0:
        return float("nan")
    return float(np.min(y))


def avg_db(mag_db: np.ndarray) -> float:
    y = np.asarray(mag_db, dtype=float).reshape(-1)
    if y.size == 0:
        return float("nan")
    return float(np.mean(y))


def avg_lin(mag_lin: np.ndarray) -> float:
    y = np.asarray(mag_lin, dtype=float).reshape(-1)
    if y.size == 0:
        return float("nan")
    return float(np.mean(y))


def pk2pk_db(mag_db: np.ndarray) -> float:
    mx = max_db(mag_db)
    mn = min_db(mag_db)
    if not (math.isfinite(mx) and math.isfinite(mn)):
        return float("nan")
    return float(mx - mn)


def summarize_advanced_metrics(
    angles_deg: np.ndarray,
    mag_lin: np.ndarray,
    xdb: float = 10.0,
    wrap: bool = False,
) -> Dict[str, float]:
    v = np.asarray(mag_lin, dtype=float).reshape(-1)
    v = np.clip(v, 1e-12, None)
    db = 20.0 * np.log10(v)
    bw = beamwidth_xdb(angles_deg, db, xdb=xdb, wrap=wrap)
    return {
        "max_db": max_db(db),
        "avg_db": avg_db(db),
        "avg_lin": avg_lin(v),
        "pk2pk_db": pk2pk_db(db),
        "bw_xdb": float(bw.get("width_deg", float("nan"))),
        "bw_left_deg": float(bw.get("left_deg", float("nan"))),
        "bw_right_deg": float(bw.get("right_deg", float("nan"))),
        "bw_threshold_db": float(bw.get("threshold_db", float("nan"))),
        "peak_deg": float(bw.get("peak_deg", float("nan"))),
        "peak_db": float(bw.get("peak_db", float("nan")),),
    }
