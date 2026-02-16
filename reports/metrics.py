"""Metrics formatting helpers for PDF report pages."""

from __future__ import annotations

import math
from typing import Iterable, List, Mapping, Tuple


MetricItem = Tuple[str, str]


def _is_finite_number(value) -> bool:
    try:
        v = float(value)
    except Exception:
        return False
    return math.isfinite(v)


def _fmt(value, digits: int = 2, suffix: str = "") -> str:
    if not _is_finite_number(value):
        return "-"
    return f"{float(value):.{digits}f}{suffix}"


def metrics_items_from_dict(metrics: Mapping[str, object]) -> List[MetricItem]:
    if not isinstance(metrics, Mapping) or not metrics:
        return [("Metrics", "Unavailable")]

    items: List[MetricItem] = []
    items.append(("Peak", f"{_fmt(metrics.get('peak_db'), 2)} dB"))
    items.append(("Peak angle", f"{_fmt(metrics.get('peak_angle_deg'), 2)} deg"))
    items.append(("HPBW", f"{_fmt(metrics.get('hpbw_deg'), 2)} deg"))
    items.append(("D2D", _fmt(metrics.get("d2d"), 3)))
    items.append(("D2D (dB)", f"{_fmt(metrics.get('d2d_db'), 2)} dB"))
    items.append(("1st null", f"{_fmt(metrics.get('first_null_db'), 2)} dB"))

    kind = str(metrics.get("kind", "")).upper()
    if kind == "H":
        items.append(("F/B", f"{_fmt(metrics.get('fb_db'), 2)} dB"))

    if _is_finite_number(metrics.get("angle_min")) and _is_finite_number(metrics.get("angle_max")):
        items.append(
            (
                "Range",
                f"{_fmt(metrics.get('angle_min'), 1)} .. {_fmt(metrics.get('angle_max'), 1)} deg",
            )
        )
    items.append(("Step", f"{_fmt(metrics.get('step_deg'), 3)} deg"))
    points = int(metrics.get("points", 0) or 0)
    items.append(("Points", f"{points:d}"))

    return [(k, v) for (k, v) in items if v and v != "-"]


def split_metrics_columns(items: Iterable[MetricItem], max_items: int = 14) -> List[List[str]]:
    data = list(items)[: max(1, int(max_items))]
    if not data:
        data = [("Metrics", "Unavailable")]

    half = (len(data) + 1) // 2
    left = data[:half]
    right = data[half:]
    rows: List[List[str]] = []
    for i in range(half):
        lk, lv = left[i]
        if i < len(right):
            rk, rv = right[i]
        else:
            rk, rv = "", ""
        rows.append([lk, lv, rk, rv])
    return rows

