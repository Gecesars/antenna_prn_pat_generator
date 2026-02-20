from __future__ import annotations

import numpy as np

from core.image_tensor.backend import get_backend
from core.image_tensor.ops.annotate import MarkerCrosshairOp, TextLabelOp, MetricBoxOp
from core.image_tensor.ops.blend import AlphaBlendOp
from core.image_tensor.tensor_image import TensorImage


def test_blend_and_annotate_changes_pixels(sample_rgb_u8):
    b = get_backend("cpu")
    img = TensorImage.from_numpy(sample_rgb_u8)
    out = AlphaBlendOp(alpha=0.5, overlay_color=(255, 0, 0, 255)).apply(img, b)
    out = MarkerCrosshairOp(20, 20, size=10, label="C").apply(out, b)
    out = TextLabelOp(10, 10, text="hello").apply(out, b)
    out = MetricBoxOp(lines=["Gain: 7.2 dBd", "HPBW: 70 deg"]).apply(out, b)
    a0 = img.ensure_rgba().to_numpy(dtype="uint8")
    a1 = out.ensure_rgba().to_numpy(dtype="uint8")
    assert a0.shape == a1.shape
    assert np.mean(np.abs(a1.astype(float) - a0.astype(float))) > 0.1
