from __future__ import annotations

import numpy as np
from PIL import ImageFilter

from ..backend import Backend
from ..tensor_image import TensorImage
from .base import ImageOp, register_op


@register_op
class GaussianBlurOp(ImageOp):
    op_name = "gaussian_blur"

    def __init__(self, radius: float = 1.0):
        self.radius = float(radius)

    def apply(self, img: TensorImage, backend: Backend) -> TensorImage:
        _ = backend
        pim = img.to_pil().filter(ImageFilter.GaussianBlur(radius=max(0.0, float(self.radius))))
        return TensorImage.from_pil(pim, device=img.device)


@register_op
class MedianBlurOp(ImageOp):
    op_name = "median_blur"

    def __init__(self, size: int = 3):
        self.size = int(size)

    def apply(self, img: TensorImage, backend: Backend) -> TensorImage:
        _ = backend
        s = int(max(3, self.size))
        if s % 2 == 0:
            s += 1
        pim = img.to_pil().filter(ImageFilter.MedianFilter(size=s))
        return TensorImage.from_pil(pim, device=img.device)


@register_op
class BilateralBlurOp(ImageOp):
    op_name = "bilateral_blur"

    def __init__(self, d: int = 7, sigma_color: float = 40.0, sigma_space: float = 40.0):
        self.d = int(d)
        self.sigma_color = float(sigma_color)
        self.sigma_space = float(sigma_space)

    def apply(self, img: TensorImage, backend: Backend) -> TensorImage:
        _ = backend
        arr = img.to_numpy(dtype="uint8")
        try:
            import cv2  # type: ignore

            out = cv2.bilateralFilter(arr, d=int(max(1, self.d)), sigmaColor=float(self.sigma_color), sigmaSpace=float(self.sigma_space))
            return TensorImage.from_numpy(out, device=img.device)
        except Exception:
            # Fallback keeps deterministic behavior.
            pim = img.to_pil().filter(ImageFilter.GaussianBlur(radius=1.0))
            return TensorImage.from_pil(pim, device=img.device)


@register_op
class UnsharpMaskOp(ImageOp):
    op_name = "unsharp_mask"

    def __init__(self, radius: float = 1.0, amount: int = 120, threshold: int = 2):
        self.radius = float(radius)
        self.amount = int(amount)
        self.threshold = int(threshold)

    def apply(self, img: TensorImage, backend: Backend) -> TensorImage:
        _ = backend
        pim = img.to_pil().filter(
            ImageFilter.UnsharpMask(
                radius=max(0.0, float(self.radius)),
                percent=max(0, int(self.amount)),
                threshold=max(0, int(self.threshold)),
            )
        )
        return TensorImage.from_pil(pim, device=img.device)
