from __future__ import annotations

import numpy as np

from core.image_tensor.tensor_image import TensorImage


def test_roundtrip_numpy(sample_rgb_u8):
    img = TensorImage.from_numpy(sample_rgb_u8, device="cpu")
    out = img.to_numpy()
    assert out.shape == sample_rgb_u8.shape
    assert out.dtype == sample_rgb_u8.dtype
    assert np.array_equal(out, sample_rgb_u8)


def test_ensure_rgba_rgb(sample_rgb_u8):
    img = TensorImage.from_numpy(sample_rgb_u8)
    rgba = img.ensure_rgba()
    assert rgba.shape[2] == 4
    rgb = rgba.ensure_rgb()
    assert rgb.shape[2] == 3
