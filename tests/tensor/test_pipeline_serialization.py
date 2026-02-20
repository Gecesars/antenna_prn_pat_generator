from __future__ import annotations

import os
import time

import numpy as np

from core.image_tensor.pipeline.pipeline import Pipeline
from core.image_tensor.ops.color import ContrastOp
from core.image_tensor.ops.filters import GaussianBlurOp
from core.image_tensor.tensor_image import TensorImage


def test_pipeline_json_and_cache(tmp_path, sample_rgb_u8):
    cache_dir = tmp_path / "cache"
    pipe = Pipeline([GaussianBlurOp(0.6), ContrastOp(1.05)], name="p1")
    payload = pipe.to_json()
    pipe2 = Pipeline.from_json(payload)
    assert pipe2.signature() == pipe.signature()

    img = TensorImage.from_numpy(sample_rgb_u8)
    t0 = time.perf_counter()
    o1 = pipe2.execute(img, device="cpu", use_cache=True, cache_dir=str(cache_dir))
    dt1 = time.perf_counter() - t0

    t1 = time.perf_counter()
    o2 = pipe2.execute(img, device="cpu", use_cache=True, cache_dir=str(cache_dir))
    dt2 = time.perf_counter() - t1

    a1 = o1.to_numpy(dtype="uint8")
    a2 = o2.to_numpy(dtype="uint8")
    assert np.array_equal(a1, a2)
    # On small images, disk cache I/O can be slower than recomputation.
    assert dt2 <= dt1 * 8.0
    assert any((cache_dir).rglob("meta.json"))
    assert any((cache_dir).rglob("output.png"))
