from __future__ import annotations

import logging
import math
import threading
from typing import Callable, Dict, Optional, Tuple

import numpy as np

from core.config.perf import NUMBA_DTYPE, USE_NUMBA
from core.numba_kernels.integrate import trapz_python
from core.numba_kernels.metrics_1d import metrics_cut_1d_numba, metrics_cut_1d_python
from core.numba_kernels.resample import smart_decimate_indices_numba, smart_decimate_indices_python
from core.numba_kernels.utils import NUMBA_AVAILABLE
from core.numba_kernels.warmup import warmup_all_kernels


LOGGER = logging.getLogger("eftx")
_WARMUP_STARTED = False
_WARMUP_LOCK = threading.Lock()


def _dtype_from_config(default=np.float64):
    if str(NUMBA_DTYPE).lower() == "float32":
        return np.float32
    return default


def _prepare_inputs(angles_deg, e_lin, dtype=None) -> Tuple[np.ndarray, np.ndarray]:
    dt = _dtype_from_config(np.float64) if dtype is None else np.dtype(dtype)
    a = np.ascontiguousarray(np.asarray(angles_deg, dtype=dt).reshape(-1))
    e = np.ascontiguousarray(np.asarray(e_lin, dtype=dt).reshape(-1))
    if a.ndim != 1 or e.ndim != 1 or a.size != e.size:
        raise ValueError("angles_deg and e_lin must be 1D with same length.")
    if a.size < 3:
        raise ValueError("At least 3 samples are required.")

    mask = np.isfinite(a) & np.isfinite(e)
    if not np.any(mask):
        raise ValueError("No finite samples.")
    a = a[mask]
    e = e[mask]
    if a.size < 3:
        raise ValueError("At least 3 finite samples are required.")

    order = np.argsort(a)
    a = a[order]
    e = e[order]

    au, inv = np.unique(a, return_inverse=True)
    if au.size != a.size:
        acc = np.zeros(au.shape[0], dtype=np.float64)
        cnt = np.zeros(au.shape[0], dtype=np.float64)
        np.add.at(acc, inv, e.astype(np.float64))
        np.add.at(cnt, inv, 1.0)
        e = (acc / np.maximum(cnt, 1.0)).astype(a.dtype, copy=False)
        a = au.astype(a.dtype, copy=False)
    if a.size < 3:
        raise ValueError("At least 3 unique angle samples are required.")
    return np.ascontiguousarray(a), np.ascontiguousarray(e)


def _metrics_tuple_to_dict(out: tuple, points: int, span_mode: int) -> Dict[str, float]:
    (
        peak_idx,
        peak_angle_deg,
        peak_db,
        hpbw_deg,
        hpbw_left_deg,
        hpbw_right_deg,
        first_null_db,
        fb_db,
        back_angle_deg,
        back_db,
        d2d_lin,
        d2d_db,
    ) = out
    return {
        "peak_idx": int(peak_idx),
        "peak_angle_deg": float(peak_angle_deg),
        "peak_db": float(peak_db),
        "hpbw_deg": float(hpbw_deg),
        "hpbw_left_deg": float(hpbw_left_deg),
        "hpbw_right_deg": float(hpbw_right_deg),
        "first_null_db": float(first_null_db),
        "fb_db": float(fb_db),
        "back_angle_deg": float(back_angle_deg),
        "back_db": float(back_db),
        "d2d_lin": float(d2d_lin),
        "d2d_db": float(d2d_db),
        "span_mode": int(span_mode),
        "points": int(points),
    }


def metrics_cut_1d(
    angles_deg,
    e_lin,
    span_mode: int,
    xdb: float = 3.0,
    dtype=None,
    use_numba: Optional[bool] = None,
) -> Dict[str, float]:
    a, e = _prepare_inputs(angles_deg, e_lin, dtype=dtype)
    use_nb = bool(USE_NUMBA and NUMBA_AVAILABLE) if use_numba is None else bool(use_numba and NUMBA_AVAILABLE)
    if use_nb:
        out = metrics_cut_1d_numba(a, e, int(span_mode), float(xdb))
    else:
        out = metrics_cut_1d_python(a, e, int(span_mode), float(xdb))
    return _metrics_tuple_to_dict(out, points=int(a.size), span_mode=int(span_mode))


def hpbw_cut_1d(angles_deg, e_lin, xdb: float = 3.0, dtype=None, use_numba: Optional[bool] = None) -> float:
    m = metrics_cut_1d(angles_deg, e_lin, span_mode=1, xdb=xdb, dtype=dtype, use_numba=use_numba)
    return float(m.get("hpbw_deg", float("nan")))


def directivity_2d_cut(angles_deg, e_lin, span_mode: int, dtype=None, use_numba: Optional[bool] = None) -> float:
    m = metrics_cut_1d(angles_deg, e_lin, span_mode=span_mode, xdb=3.0, dtype=dtype, use_numba=use_numba)
    return float(m.get("d2d_lin", float("nan")))


def smart_decimate_indices(angles_deg, values, target_rows: int, use_numba: Optional[bool] = None) -> np.ndarray:
    a = np.ascontiguousarray(np.asarray(angles_deg, dtype=np.float64).reshape(-1))
    v = np.ascontiguousarray(np.asarray(values, dtype=np.float64).reshape(-1))
    if a.size != v.size:
        raise ValueError("angles_deg and values must have same length.")
    use_nb = bool(USE_NUMBA and NUMBA_AVAILABLE) if use_numba is None else bool(use_numba and NUMBA_AVAILABLE)
    if use_nb:
        idx = smart_decimate_indices_numba(a, v, int(target_rows))
    else:
        idx = smart_decimate_indices_python(a, v, int(target_rows))
    return np.asarray(idx, dtype=np.int32)


def integrate_power_numpy(angles_deg, e_lin) -> float:
    a, e = _prepare_inputs(angles_deg, e_lin, dtype=np.float64)
    emax = float(np.max(np.abs(e)))
    if emax <= 1e-30:
        return float("nan")
    p = (np.abs(e) / emax) ** 2
    rad = np.deg2rad(a.astype(np.float64))
    return float(trapz_python(rad, p.astype(np.float64)))


def start_numba_warmup_thread(
    logger: Optional[logging.Logger] = None,
    status_cb: Optional[Callable[[str], None]] = None,
) -> bool:
    global _WARMUP_STARTED
    with _WARMUP_LOCK:
        if _WARMUP_STARTED:
            return False
        _WARMUP_STARTED = True

    log = logger or LOGGER

    def _run():
        try:
            info = warmup_all_kernels()
            msg = (
                f"Numba warm-up {info.get('status')} | "
                f"available={info.get('numba_available')} use_numba={info.get('use_numba')} "
                f"elapsed={float(info.get('elapsed_s', 0.0)):.3f}s"
            )
            log.info(msg)
            if callable(status_cb):
                status_cb(msg)
        except Exception:
            log.exception("Numba warm-up failed.")

    threading.Thread(target=_run, daemon=True, name="eftx-numba-warmup").start()
    return True

