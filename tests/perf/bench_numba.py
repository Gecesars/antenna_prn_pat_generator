from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from core.analysis.pattern_metrics import metrics_cut_1d, smart_decimate_indices


HAS_NUMBA = importlib.util.find_spec("numba") is not None


@pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")
def test_bench_metrics_cut_numba_vs_numpy(benchmark):
    ang = np.arange(-180.0, 180.0 + 1e-9, 1.0, dtype=np.float64)
    e = np.abs(np.cos(np.deg2rad(ang))) + 0.1

    # Warm-up JIT outside benchmark.
    _ = metrics_cut_1d(ang, e, span_mode=0, use_numba=True)

    def _run_numba():
        return metrics_cut_1d(ang, e, span_mode=0, use_numba=True)

    result = benchmark(_run_numba)
    assert "d2d_lin" in result


@pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")
def test_bench_decimation_numba(benchmark):
    ang = np.linspace(-180.0, 180.0, 20001, dtype=np.float64)
    val = np.cos(np.deg2rad(ang)) ** 2

    _ = smart_decimate_indices(ang, val, target_rows=73, use_numba=True)

    def _run():
        return smart_decimate_indices(ang, val, target_rows=73, use_numba=True)

    out = benchmark(_run)
    assert out.size <= 73

