"""Layout constants for PDF report generation."""

from __future__ import annotations

from dataclasses import dataclass

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm

PAGE_SIZE = A4

# Safe area in millimeters (requested in relatorio.md).
SAFE_LEFT_MM = 24.0
SAFE_RIGHT_MM = 18.0
SAFE_TOP_MM = 38.0
SAFE_BOTTOM_MM = 30.0

# Typography.
TITLE_FONT_SIZE = 15
SUBTITLE_FONT_SIZE = 8.8
BODY_FONT_SIZE = 7.6
TABLE_FONT_SIZE = 6.8
METRICS_FONT_SIZE = 8.0

# Spacing (points).
TITLE_SPACING_PT = 7.0
SUBTITLE_SPACING_PT = 6.0
BLOCK_SPACING_PT = 8.0

# Table rendering limits.
MIN_DISPLAY_ROWS = 60
MAX_DISPLAY_ROWS = 90
DEFAULT_DISPLAY_ROWS = 80

# Diagram rendering.
MIN_IMAGE_WIDTH_PX = 1200
MIN_IMAGE_HEIGHT_PX = 700
DEFAULT_IMAGE_DPI = 300
SUPPORTED_DPI = (200, 300)

# Diagram + metrics block.
TOP_BLOCK_COL_RATIOS = (0.72, 0.28)
TOP_BLOCK_MAX_HEIGHT_MM = 88.0


def mm_to_pt(value_mm: float) -> float:
    return float(value_mm) * mm


@dataclass(frozen=True)
class SafeBox:
    left: float
    bottom: float
    width: float
    height: float

    @property
    def right(self) -> float:
        return self.left + self.width

    @property
    def top(self) -> float:
        return self.bottom + self.height


def get_safe_box(page_size=PAGE_SIZE) -> SafeBox:
    page_w, page_h = page_size
    left = mm_to_pt(SAFE_LEFT_MM)
    right_margin = mm_to_pt(SAFE_RIGHT_MM)
    top_margin = mm_to_pt(SAFE_TOP_MM)
    bottom = mm_to_pt(SAFE_BOTTOM_MM)
    width = page_w - left - right_margin
    height = page_h - top_margin - bottom
    return SafeBox(left=left, bottom=bottom, width=width, height=height)
