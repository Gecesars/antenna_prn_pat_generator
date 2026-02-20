from __future__ import annotations

from core.image_tensor.backend import get_backend
from core.image_tensor.ops.filters import GaussianBlurOp, MedianBlurOp, BilateralBlurOp, UnsharpMaskOp
from core.image_tensor.tensor_image import TensorImage


def test_filters_preserve_shape(sample_rgb_u8):
    b = get_backend("cpu")
    img = TensorImage.from_numpy(sample_rgb_u8)
    for op in [
        GaussianBlurOp(1.2),
        MedianBlurOp(3),
        BilateralBlurOp(7, 20.0, 20.0),
        UnsharpMaskOp(1.0, 120, 1),
    ]:
        out = op.apply(img, b)
        assert out.shape == img.shape
        assert out.to_numpy(dtype="uint8").dtype.name == "uint8"
