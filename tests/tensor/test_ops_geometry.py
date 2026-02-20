from __future__ import annotations

from core.image_tensor.backend import get_backend
from core.image_tensor.ops.geometry import ResizeOp, CropOp, PadOp, RotateOp
from core.image_tensor.tensor_image import TensorImage


def test_ops_geometry_shapes(sample_rgb_u8):
    b = get_backend("cpu")
    img = TensorImage.from_numpy(sample_rgb_u8)
    out = ResizeOp(80, 50, "bilinear").apply(img, b)
    assert out.shape[:2] == (50, 80)
    out = CropOp(10, 5, 30, 20).apply(out, b)
    assert out.shape[:2] == (20, 30)
    out = PadOp(left=2, top=3, right=4, bottom=5).apply(out, b)
    assert out.shape[:2] == (28, 36)
    out = RotateOp(15.0, expand=False).apply(out, b)
    assert out.shape[:2] == (28, 36)
