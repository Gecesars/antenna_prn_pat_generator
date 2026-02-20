from __future__ import annotations

import time
from typing import Callable, Optional

import numpy as np

from core.config.perf import USE_NUMBA

from .integrate import deg2rad_inplace_numba, trapz_numba
from .metrics_1d import metrics_cut_1d_numba
from .resample import smart_decimate_indices_numba
from .utils import NUMBA_AVAILABLE


ProgressCallback = Optional[Callable[[int, int, str], None]]


def warmup_all_kernels(progress_cb: ProgressCallback = None) -> dict:
    t0 = time.perf_counter()
    if (not NUMBA_AVAILABLE) or (not USE_NUMBA):
        return {
            "status": "skipped",
            "numba_available": bool(NUMBA_AVAILABLE),
            "use_numba": bool(USE_NUMBA),
            "elapsed_s": 0.0,
        }

    a = np.linspace(-180.0, 180.0, 361, dtype=np.float64)
    e = np.abs(np.cos(np.deg2rad(a))).astype(np.float64)
    r = np.empty_like(a)
    y = np.ones_like(a)

    steps = [
        "deg2rad_inplace",
        "trapz",
        "metrics_1d",
        "resample_decimate",
    ]

    for idx, label in enumerate(steps, start=1):
        if callable(progress_cb):
            progress_cb(idx, len(steps), f"Numba warm-up: {label}")
        if label == "deg2rad_inplace":
            deg2rad_inplace_numba(a, r)
        elif label == "trapz":
            _ = trapz_numba(r, y)
        elif label == "metrics_1d":
            _ = metrics_cut_1d_numba(a, e, 0, 3.0)
        else:
            _ = smart_decimate_indices_numba(a, e, 73)

    dt = max(0.0, time.perf_counter() - t0)
    return {
        "status": "ok",
        "numba_available": bool(NUMBA_AVAILABLE),
        "use_numba": bool(USE_NUMBA),
        "elapsed_s": float(dt),
    }

