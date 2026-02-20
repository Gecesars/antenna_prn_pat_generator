from __future__ import annotations

import pytest

from core.image_tensor.backend import get_backend


def test_backend_detect_auto():
    b = get_backend("auto")
    assert b.device in {"cpu", "cuda"}
    text = b.log_env()
    assert "device=" in text


@pytest.mark.cuda
def test_backend_detect_cuda_explicit():
    b = get_backend("cuda")
    assert b.device == "cuda"
