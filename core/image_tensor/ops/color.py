from __future__ import annotations

import numpy as np
from PIL import ImageEnhance

from ..backend import Backend
from ..tensor_image import TensorImage
from .base import ImageOp, register_op


@register_op
class GammaOp(ImageOp):
    op_name = "gamma"

    def __init__(self, gamma: float = 1.0):
        self.gamma = float(gamma)

    def validate(self, img: TensorImage) -> None:
        _ = img
        if self.gamma <= 0.0:
            raise ValueError("gamma must be > 0")

    def apply(self, img: TensorImage, backend: Backend) -> TensorImage:
        _ = backend
        g = float(self.gamma)
        arr = img.to_numpy(dtype="float32")
        rgb = np.power(np.clip(arr[:, :, :3], 0.0, 1.0), 1.0 / g)
        if arr.shape[2] == 4:
            out = np.concatenate([rgb, arr[:, :, 3:4]], axis=2)
        else:
            out = rgb
        return TensorImage.from_numpy(out.astype(np.float32), device=img.device)


@register_op
class LevelsOp(ImageOp):
    op_name = "levels"

    def __init__(self, in_min: float = 0.0, in_max: float = 1.0, out_min: float = 0.0, out_max: float = 1.0):
        self.in_min = float(in_min)
        self.in_max = float(in_max)
        self.out_min = float(out_min)
        self.out_max = float(out_max)

    def validate(self, img: TensorImage) -> None:
        _ = img
        if self.in_max <= self.in_min:
            raise ValueError("in_max must be > in_min")
        if self.out_max < self.out_min:
            raise ValueError("out_max must be >= out_min")

    def apply(self, img: TensorImage, backend: Backend) -> TensorImage:
        _ = backend
        arr = img.to_numpy(dtype="float32")
        rgb = arr[:, :, :3]
        rgb = (rgb - self.in_min) / (self.in_max - self.in_min)
        rgb = np.clip(rgb, 0.0, 1.0)
        rgb = self.out_min + rgb * (self.out_max - self.out_min)
        rgb = np.clip(rgb, 0.0, 1.0)
        if arr.shape[2] == 4:
            out = np.concatenate([rgb, arr[:, :, 3:4]], axis=2)
        else:
            out = rgb
        return TensorImage.from_numpy(out.astype(np.float32), device=img.device)


@register_op
class ContrastOp(ImageOp):
    op_name = "contrast"

    def __init__(self, factor: float = 1.0):
        self.factor = float(factor)

    def apply(self, img: TensorImage, backend: Backend) -> TensorImage:
        _ = backend
        pim = img.to_pil()
        out = ImageEnhance.Contrast(pim).enhance(float(self.factor))
        return TensorImage.from_pil(out, device=img.device)


@register_op
class BrightnessOp(ImageOp):
    op_name = "brightness"

    def __init__(self, factor: float = 1.0):
        self.factor = float(factor)

    def apply(self, img: TensorImage, backend: Backend) -> TensorImage:
        _ = backend
        pim = img.to_pil()
        out = ImageEnhance.Brightness(pim).enhance(float(self.factor))
        return TensorImage.from_pil(out, device=img.device)
