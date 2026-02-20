from __future__ import annotations

import numpy as np

from core.image_tensor.pipeline.pipeline import Pipeline
from core.image_tensor.ops.filters import GaussianBlurOp
from core.image_tensor.ops.color import ContrastOp
from core.image_tensor.tensor_image import TensorImage


def test_batch_streaming_sanity(tmp_path):
    pipe = Pipeline([GaussianBlurOp(0.5), ContrastOp(1.04)], name="batch")
    cache_dir = tmp_path / "cache"
    rng = np.random.default_rng(42)
    for _ in range(24):
        arr = rng.integers(0, 256, size=(128, 128, 3), dtype=np.uint8)
        img = TensorImage.from_numpy(arr)
        out = pipe.execute(img, device="cpu", use_cache=True, cache_dir=str(cache_dir))
        assert out.shape == img.shape
