from __future__ import annotations

import os
import numpy as np
import pytest


def _cuda_available() -> bool:
    try:
        import torch  # type: ignore

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def pytest_runtest_setup(item):
    if "cuda" in item.keywords and not _cuda_available():
        pytest.skip("CUDA is not available.")


@pytest.fixture
def sample_rgb_u8():
    rng = np.random.default_rng(123)
    return rng.integers(0, 256, size=(96, 128, 3), dtype=np.uint8)
