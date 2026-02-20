from __future__ import annotations

from typing import Callable

from core.config.perf import NUMBA_FASTMATH, NUMBA_PARALLEL, USE_NUMBA


try:
    from numba import njit  # type: ignore

    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover
    njit = None  # type: ignore
    NUMBA_AVAILABLE = False


def _identity_decorator(fn: Callable):
    return fn


def njit_if_available(**kwargs):
    if (not NUMBA_AVAILABLE) or (not USE_NUMBA):
        return _identity_decorator

    opts = {"cache": True}
    opts.update(kwargs)
    if "fastmath" not in opts:
        opts["fastmath"] = bool(NUMBA_FASTMATH)
    if "parallel" not in opts:
        opts["parallel"] = bool(NUMBA_PARALLEL)

    def _decorator(fn: Callable):
        return njit(**opts)(fn)

    return _decorator

