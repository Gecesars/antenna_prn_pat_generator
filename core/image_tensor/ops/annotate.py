from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..backend import Backend
from ..tensor_image import TensorImage
from .base import ImageOp, register_op


def _clamp_color(c: Sequence[int], default: Sequence[int]) -> Tuple[int, int, int, int]:
    raw = [int(x) for x in (list(c) if c is not None else list(default))]
    if len(raw) < 4:
        raw = raw + [255] * (4 - len(raw))
    raw = [max(0, min(255, int(x))) for x in raw[:4]]
    return int(raw[0]), int(raw[1]), int(raw[2]), int(raw[3])


def _load_font(size: int):
    token = int(max(6, size))
    candidates = [
        "segoeui.ttf",
        "SegoeUI.ttf",
        "DejaVuSans.ttf",
        "arial.ttf",
    ]
    for name in candidates:
        try:
            return ImageFont.truetype(name, token)
        except Exception:
            continue
    return ImageFont.load_default()


def _to_pil_rgba(img: TensorImage) -> Image.Image:
    return img.ensure_rgba().to_pil().convert("RGBA")


@register_op
class MarkerPointOp(ImageOp):
    op_name = "marker_point"

    def __init__(
        self,
        x: float,
        y: float,
        radius: int = 4,
        color: Sequence[int] = (255, 0, 0, 255),
        label: str = "",
        font_size: int = 12,
    ):
        self.x = float(x)
        self.y = float(y)
        self.radius = int(radius)
        self.color = [int(v) for v in list(color)]
        self.label = str(label)
        self.font_size = int(font_size)

    def apply(self, img: TensorImage, backend: Backend) -> TensorImage:
        _ = backend
        pim = _to_pil_rgba(img)
        d = ImageDraw.Draw(pim, "RGBA")
        c = _clamp_color(self.color, (255, 0, 0, 255))
        r = max(1, int(self.radius))
        x = float(self.x)
        y = float(self.y)
        d.ellipse((x - r, y - r, x + r, y + r), outline=c, fill=c)
        if self.label:
            f = _load_font(self.font_size)
            d.text((x + r + 3, y - r - 3), self.label, fill=c, font=f)
        return TensorImage.from_pil(pim, device=img.device)


@register_op
class MarkerLineOp(ImageOp):
    op_name = "marker_line"

    def __init__(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        width: int = 2,
        color: Sequence[int] = (255, 0, 0, 255),
        label: str = "",
        font_size: int = 12,
    ):
        self.x1 = float(x1)
        self.y1 = float(y1)
        self.x2 = float(x2)
        self.y2 = float(y2)
        self.width = int(width)
        self.color = [int(v) for v in list(color)]
        self.label = str(label)
        self.font_size = int(font_size)

    def apply(self, img: TensorImage, backend: Backend) -> TensorImage:
        _ = backend
        pim = _to_pil_rgba(img)
        d = ImageDraw.Draw(pim, "RGBA")
        c = _clamp_color(self.color, (255, 0, 0, 255))
        w = max(1, int(self.width))
        d.line((self.x1, self.y1, self.x2, self.y2), fill=c, width=w)
        if self.label:
            f = _load_font(self.font_size)
            xm = 0.5 * (self.x1 + self.x2)
            ym = 0.5 * (self.y1 + self.y2)
            d.text((xm + 4, ym + 4), self.label, fill=c, font=f)
        return TensorImage.from_pil(pim, device=img.device)


@register_op
class MarkerCrosshairOp(ImageOp):
    op_name = "marker_crosshair"

    def __init__(
        self,
        x: float,
        y: float,
        size: int = 16,
        width: int = 1,
        color: Sequence[int] = (255, 255, 0, 255),
        label: str = "",
        font_size: int = 12,
    ):
        self.x = float(x)
        self.y = float(y)
        self.size = int(size)
        self.width = int(width)
        self.color = [int(v) for v in list(color)]
        self.label = str(label)
        self.font_size = int(font_size)

    def apply(self, img: TensorImage, backend: Backend) -> TensorImage:
        _ = backend
        pim = _to_pil_rgba(img)
        d = ImageDraw.Draw(pim, "RGBA")
        c = _clamp_color(self.color, (255, 255, 0, 255))
        s = max(2, int(self.size))
        w = max(1, int(self.width))
        x = float(self.x)
        y = float(self.y)
        d.line((x - s, y, x + s, y), fill=c, width=w)
        d.line((x, y - s, x, y + s), fill=c, width=w)
        if self.label:
            f = _load_font(self.font_size)
            d.text((x + s + 3, y + 2), self.label, fill=c, font=f)
        return TensorImage.from_pil(pim, device=img.device)


@register_op
class TextLabelOp(ImageOp):
    op_name = "text_label"

    def __init__(
        self,
        x: float,
        y: float,
        text: str,
        color: Sequence[int] = (255, 255, 255, 255),
        bg_color: Sequence[int] = (0, 0, 0, 128),
        font_size: int = 12,
        padding: int = 3,
    ):
        self.x = float(x)
        self.y = float(y)
        self.text = str(text)
        self.color = [int(v) for v in list(color)]
        self.bg_color = [int(v) for v in list(bg_color)]
        self.font_size = int(font_size)
        self.padding = int(padding)

    def apply(self, img: TensorImage, backend: Backend) -> TensorImage:
        _ = backend
        pim = _to_pil_rgba(img)
        d = ImageDraw.Draw(pim, "RGBA")
        f = _load_font(self.font_size)
        c = _clamp_color(self.color, (255, 255, 255, 255))
        bg = _clamp_color(self.bg_color, (0, 0, 0, 128))
        x = float(self.x)
        y = float(self.y)
        bb = d.textbbox((x, y), self.text, font=f)
        pad = max(0, int(self.padding))
        d.rectangle((bb[0] - pad, bb[1] - pad, bb[2] + pad, bb[3] + pad), fill=bg)
        d.text((x, y), self.text, fill=c, font=f)
        return TensorImage.from_pil(pim, device=img.device)


@register_op
class MetricBoxOp(ImageOp):
    op_name = "metric_box"

    def __init__(
        self,
        lines: Sequence[str],
        anchor: str = "top_right",
        margin: int = 10,
        padding: int = 6,
        font_size: int = 12,
        fg_color: Sequence[int] = (255, 255, 255, 255),
        bg_color: Sequence[int] = (17, 24, 39, 210),
        border_color: Sequence[int] = (148, 163, 184, 220),
    ):
        self.lines = [str(x) for x in list(lines)]
        self.anchor = str(anchor)
        self.margin = int(margin)
        self.padding = int(padding)
        self.font_size = int(font_size)
        self.fg_color = [int(v) for v in list(fg_color)]
        self.bg_color = [int(v) for v in list(bg_color)]
        self.border_color = [int(v) for v in list(border_color)]

    def apply(self, img: TensorImage, backend: Backend) -> TensorImage:
        _ = backend
        pim = _to_pil_rgba(img)
        d = ImageDraw.Draw(pim, "RGBA")
        f = _load_font(self.font_size)
        lines = self.lines or ["-"]
        widths = []
        heights = []
        for ln in lines:
            bb = d.textbbox((0, 0), ln, font=f)
            widths.append(max(1, bb[2] - bb[0]))
            heights.append(max(1, bb[3] - bb[1]))
        w_txt = max(widths)
        h_txt = int(sum(heights) + max(0, len(lines) - 1) * 2)
        pad = max(0, int(self.padding))
        box_w = w_txt + 2 * pad
        box_h = h_txt + 2 * pad
        margin = max(0, int(self.margin))
        W, H = pim.size

        a = self.anchor.strip().lower()
        if a == "top_left":
            x0 = margin
            y0 = margin
        elif a == "bottom_left":
            x0 = margin
            y0 = max(0, H - margin - box_h)
        elif a == "bottom_right":
            x0 = max(0, W - margin - box_w)
            y0 = max(0, H - margin - box_h)
        else:  # top_right default
            x0 = max(0, W - margin - box_w)
            y0 = margin
        x1 = min(W - 1, x0 + box_w)
        y1 = min(H - 1, y0 + box_h)

        fg = _clamp_color(self.fg_color, (255, 255, 255, 255))
        bg = _clamp_color(self.bg_color, (17, 24, 39, 210))
        bd = _clamp_color(self.border_color, (148, 163, 184, 220))
        d.rectangle((x0, y0, x1, y1), fill=bg, outline=bd, width=1)
        cy = y0 + pad
        for ln, lh in zip(lines, heights):
            d.text((x0 + pad, cy), ln, fill=fg, font=f)
            cy += lh + 2
        return TensorImage.from_pil(pim, device=img.device)
