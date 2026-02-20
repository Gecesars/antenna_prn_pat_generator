from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from ..backend import Backend
from ..tensor_image import TensorImage
from .base import ImageOp, register_op


_RESAMPLE = {
    "nearest": Image.Resampling.NEAREST,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
}


def _resample(mode: str):
    token = str(mode or "bilinear").strip().lower()
    return _RESAMPLE.get(token, Image.Resampling.BILINEAR)


@register_op
class ResizeOp(ImageOp):
    op_name = "resize"

    def __init__(self, width: int, height: int, method: str = "bilinear"):
        self.width = int(width)
        self.height = int(height)
        self.method = str(method)

    def validate(self, img: TensorImage) -> None:
        _ = img
        if self.width <= 0 or self.height <= 0:
            raise ValueError("resize width/height must be > 0")

    def apply(self, img: TensorImage, backend: Backend) -> TensorImage:
        _ = backend
        pim = img.to_pil().resize((self.width, self.height), resample=_resample(self.method))
        return TensorImage.from_pil(pim, device=img.device)


@register_op
class CropOp(ImageOp):
    op_name = "crop"

    def __init__(self, x: int, y: int, width: int, height: int):
        self.x = int(x)
        self.y = int(y)
        self.width = int(width)
        self.height = int(height)

    def validate(self, img: TensorImage) -> None:
        _ = img
        if self.width <= 0 or self.height <= 0:
            raise ValueError("crop width/height must be > 0")

    def apply(self, img: TensorImage, backend: Backend) -> TensorImage:
        _ = backend
        pim = img.to_pil()
        box = (self.x, self.y, self.x + self.width, self.y + self.height)
        out = pim.crop(box)
        return TensorImage.from_pil(out, device=img.device)


@register_op
class PadOp(ImageOp):
    op_name = "pad"

    def __init__(self, left: int = 0, top: int = 0, right: int = 0, bottom: int = 0, fill: Sequence[int] = (0, 0, 0, 0)):
        self.left = int(left)
        self.top = int(top)
        self.right = int(right)
        self.bottom = int(bottom)
        self.fill = [int(x) for x in list(fill)]

    def validate(self, img: TensorImage) -> None:
        _ = img
        if min(self.left, self.top, self.right, self.bottom) < 0:
            raise ValueError("pad values must be >= 0")

    def apply(self, img: TensorImage, backend: Backend) -> TensorImage:
        _ = backend
        src = img.to_numpy(dtype="uint8")
        h, w, c = src.shape
        out_h = h + self.top + self.bottom
        out_w = w + self.left + self.right
        if out_h <= 0 or out_w <= 0:
            raise ValueError("invalid padded shape")
        fill = list(self.fill)
        if c == 3 and len(fill) >= 4:
            fill = fill[:3]
        if c == 4 and len(fill) == 3:
            fill = fill + [255]
        if len(fill) < c:
            fill = fill + [0] * (c - len(fill))
        out = np.zeros((out_h, out_w, c), dtype=np.uint8)
        out[:, :, :] = np.asarray(fill[:c], dtype=np.uint8)
        out[self.top : self.top + h, self.left : self.left + w, :] = src
        return TensorImage.from_numpy(out, device=img.device)


@register_op
class RotateOp(ImageOp):
    op_name = "rotate"

    def __init__(self, angle_deg: float, expand: bool = False, method: str = "bilinear"):
        self.angle_deg = float(angle_deg)
        self.expand = bool(expand)
        self.method = str(method)

    def apply(self, img: TensorImage, backend: Backend) -> TensorImage:
        _ = backend
        pim = img.to_pil()
        out = pim.rotate(self.angle_deg, resample=_resample(self.method), expand=bool(self.expand))
        return TensorImage.from_pil(out, device=img.device)


@register_op
class AffineOp(ImageOp):
    op_name = "affine"

    def __init__(
        self,
        matrix: Sequence[float],
        out_width: Optional[int] = None,
        out_height: Optional[int] = None,
        method: str = "bilinear",
    ):
        m = list(matrix)
        if len(m) != 6:
            raise ValueError("affine matrix must have 6 values")
        self.matrix = [float(x) for x in m]
        self.out_width = None if out_width is None else int(out_width)
        self.out_height = None if out_height is None else int(out_height)
        self.method = str(method)

    def apply(self, img: TensorImage, backend: Backend) -> TensorImage:
        _ = backend
        pim = img.to_pil()
        w = int(self.out_width) if self.out_width else pim.size[0]
        h = int(self.out_height) if self.out_height else pim.size[1]
        out = pim.transform((w, h), Image.Transform.AFFINE, data=tuple(self.matrix), resample=_resample(self.method))
        return TensorImage.from_pil(out, device=img.device)


@register_op
class PerspectiveOp(ImageOp):
    op_name = "perspective"

    def __init__(
        self,
        coeffs: Sequence[float],
        out_width: Optional[int] = None,
        out_height: Optional[int] = None,
        method: str = "bilinear",
    ):
        c = list(coeffs)
        if len(c) != 8:
            raise ValueError("perspective coeffs must have 8 values")
        self.coeffs = [float(x) for x in c]
        self.out_width = None if out_width is None else int(out_width)
        self.out_height = None if out_height is None else int(out_height)
        self.method = str(method)

    def apply(self, img: TensorImage, backend: Backend) -> TensorImage:
        _ = backend
        pim = img.to_pil()
        w = int(self.out_width) if self.out_width else pim.size[0]
        h = int(self.out_height) if self.out_height else pim.size[1]
        out = pim.transform((w, h), Image.Transform.PERSPECTIVE, data=tuple(self.coeffs), resample=_resample(self.method))
        return TensorImage.from_pil(out, device=img.device)
