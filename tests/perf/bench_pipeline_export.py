from __future__ import annotations

import numpy as np

from core.image_tensor.pipeline.pipeline import Pipeline
from core.image_tensor.ops.filters import GaussianBlurOp, UnsharpMaskOp
from core.image_tensor.ops.color import ContrastOp, BrightnessOp
from core.image_tensor.ops.annotate import MetricBoxOp
from core.image_tensor.tensor_image import TensorImage


def test_bench_pipeline_export(benchmark):
    rng = np.random.default_rng(99)
    arr = rng.integers(0, 256, size=(1800, 1800, 3), dtype=np.uint8)
    img = TensorImage.from_numpy(arr)
    pipe = Pipeline(
        [
            GaussianBlurOp(0.6),
            UnsharpMaskOp(1.2, 130, 1),
            ContrastOp(1.08),
            BrightnessOp(1.02),
            MetricBoxOp(lines=["Gain: 7.3 dBd", "HPBW: 73 deg"]),
        ],
        name="bench_export",
    )

    def _run():
        out = pipe.execute(img, device="cpu", use_cache=False)
        return out.shape

    shape = benchmark(_run)
    assert shape[0] == 1800 and shape[1] == 1800 and shape[2] in (3, 4)
