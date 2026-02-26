from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image

from core.image_tensor.io.save import export_png
from core.image_tensor.ops.color import ContrastOp
from core.image_tensor.ops.filters import GaussianBlurOp
from core.image_tensor.pipeline.pipeline import Pipeline
from core.image_tensor.tensor_image import TensorImage


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    mse = float(np.mean((a - b) ** 2))
    if mse <= 1e-12:
        return 99.0
    return 20.0 * math.log10(255.0 / math.sqrt(mse))


def test_tensor_pipeline_golden_regression(tmp_path):
    base = Path(__file__).resolve().parents[1] / "golden"
    in_path = base / "inputs" / "sample_input.png"
    exp_path = base / "expected" / "sample_expected.png"
    assert in_path.is_file()
    assert exp_path.is_file()

    img = TensorImage.from_file(str(in_path), device="cpu")
    pipe = Pipeline([GaussianBlurOp(0.7), ContrastOp(1.06)], name="golden-regression")
    out = pipe.execute(img, device="cpu", use_cache=False)

    actual_path = tmp_path / "golden_actual.png"
    export_png(out.to("cpu"), str(actual_path), dpi=300)

    actual = np.asarray(Image.open(actual_path).convert("RGB"), dtype=np.uint8)
    expected = np.asarray(Image.open(exp_path).convert("RGB"), dtype=np.uint8)
    assert actual.shape == expected.shape
    assert _psnr(actual, expected) >= 45.0

