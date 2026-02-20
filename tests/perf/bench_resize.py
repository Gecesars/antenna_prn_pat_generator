from __future__ import annotations

import numpy as np

from core.image_tensor.pipeline.pipeline import Pipeline
from core.image_tensor.ops.geometry import ResizeOp
from core.image_tensor.tensor_image import TensorImage


def test_bench_resize_2k(benchmark):
    rng = np.random.default_rng(123)
    arr = rng.integers(0, 256, size=(2048, 2048, 3), dtype=np.uint8)
    img = TensorImage.from_numpy(arr)
    pipe = Pipeline([ResizeOp(1536, 1536, "bilinear")], name="bench_resize")

    def _run():
        out = pipe.execute(img, device="cpu", use_cache=False)
        return out.shape

    shape = benchmark(_run)
    assert shape == (1536, 1536, 3)
