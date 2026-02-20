from __future__ import annotations

import os
from typing import Sequence

import numpy as np
from PIL import Image

from ..backend import Backend
from ..tensor_image import TensorImage
from .base import ImageOp, register_op


def _normalize_color(v: Sequence[int]) -> np.ndarray:
    raw = [int(x) for x in list(v)]
    if len(raw) < 4:
        raw = raw + [255] * (4 - len(raw))
    raw = [max(0, min(255, int(x))) for x in raw[:4]]
    return np.asarray(raw, dtype=np.float32)


@register_op
class AlphaBlendOp(ImageOp):
    op_name = "alpha_blend"

    def __init__(
        self,
        alpha: float = 0.4,
        overlay_color: Sequence[int] = (255, 0, 0, 128),
        overlay_path: str = "",
    ):
        self.alpha = float(alpha)
        self.overlay_color = [int(x) for x in list(overlay_color)]
        self.overlay_path = str(overlay_path or "")

    def validate(self, img: TensorImage) -> None:
        _ = img
        if self.alpha < 0.0 or self.alpha > 1.0:
            raise ValueError("alpha must be in [0, 1]")

    def apply(self, img: TensorImage, backend: Backend) -> TensorImage:
        _ = backend
        base = img.ensure_rgba().to_numpy(dtype="float32")
        h, w, _ = base.shape

        if self.overlay_path and os.path.isfile(self.overlay_path):
            ov = Image.open(self.overlay_path).convert("RGBA").resize((w, h), resample=Image.Resampling.BILINEAR)
            overlay = np.asarray(ov, dtype=np.float32) / 255.0
        else:
            c = _normalize_color(self.overlay_color) / 255.0
            overlay = np.zeros_like(base, dtype=np.float32)
            overlay[:, :, :] = c

        a_global = float(max(0.0, min(1.0, self.alpha)))
        a_overlay = overlay[:, :, 3:4] * a_global
        out_rgb = base[:, :, :3] * (1.0 - a_overlay) + overlay[:, :, :3] * a_overlay
        out_a = np.clip(base[:, :, 3:4] + a_overlay * (1.0 - base[:, :, 3:4]), 0.0, 1.0)
        out = np.concatenate([out_rgb, out_a], axis=2)
        return TensorImage.from_numpy(out.astype(np.float32), device=img.device)
