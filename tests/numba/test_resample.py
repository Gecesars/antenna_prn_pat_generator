from __future__ import annotations

import numpy as np

from core.analysis.pattern_metrics import smart_decimate_indices


def test_smart_decimation_preserves_critical_indices():
    ang = np.linspace(-180.0, 180.0, 2001, dtype=np.float64)
    val = np.cos(np.deg2rad(ang)) ** 2
    target = 73
    idx = smart_decimate_indices(ang, val, target_rows=target, use_numba=True)

    assert idx.dtype == np.int32
    assert idx.size <= target
    assert np.all(np.diff(idx) > 0)
    assert int(idx[0]) == 0
    assert int(idx[-1]) == len(ang) - 1
    assert int(np.argmax(val)) in set(int(x) for x in idx.tolist())
    assert int(np.argmin(val)) in set(int(x) for x in idx.tolist())


def test_smart_decimation_handles_small_target():
    ang = np.linspace(-90.0, 90.0, 181, dtype=np.float64)
    val = 0.5 + 0.5 * np.cos(np.deg2rad(ang))
    idx = smart_decimate_indices(ang, val, target_rows=2, use_numba=True)
    assert idx.size <= 2
    assert np.all(np.diff(idx) >= 0)

