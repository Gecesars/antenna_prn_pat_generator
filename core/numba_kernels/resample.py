from __future__ import annotations

import numpy as np

from .utils import njit_if_available


def smart_decimate_indices_python(angles_deg: np.ndarray, values: np.ndarray, target_rows: int) -> np.ndarray:
    _ = angles_deg
    n = int(min(len(angles_deg), len(values)))
    target = int(target_rows)
    if target <= 0 or n <= 0:
        return np.zeros((0,), dtype=np.int32)
    if n <= target:
        return np.arange(n, dtype=np.int32)

    selected = np.zeros(n, dtype=np.uint8)
    count = 0

    def _try_mark(i: int):
        nonlocal count
        if i < 0 or i >= n:
            return
        if count >= target:
            return
        if selected[i] == 0:
            selected[i] = 1
            count += 1

    peak_idx = 0
    min_idx = 0
    vmax = float(values[0])
    vmin = float(values[0])
    for i in range(1, n):
        v = float(values[i])
        if v > vmax:
            vmax = v
            peak_idx = i
        if v < vmin:
            vmin = v
            min_idx = i

    _try_mark(0)
    _try_mark(n - 1)
    _try_mark(peak_idx)
    _try_mark(min_idx)

    if count < target:
        slots = target - count
        for k in range(1, slots + 1):
            idx = int(round((k * (n - 1)) / float(slots + 1)))
            _try_mark(idx)

    if count < target:
        stride = max(1, n // max(1, target))
        i = stride
        while i < (n - 1) and count < target:
            _try_mark(i)
            i += stride

    if count < target:
        for i in range(1, n - 1):
            if count >= target:
                break
            _try_mark(i)

    out = np.empty(count, dtype=np.int32)
    w = 0
    for i in range(n):
        if selected[i] != 0:
            out[w] = np.int32(i)
            w += 1
    return out


smart_decimate_indices_numba = njit_if_available(parallel=False, fastmath=False)(smart_decimate_indices_python)

