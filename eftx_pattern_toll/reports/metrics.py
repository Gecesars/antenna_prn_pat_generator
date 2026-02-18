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

    key_meta = {
        "peak_db": ("Peak", "dB", 2),
        "peak_angle_deg": ("Peak angle", "deg", 2),
        "hpbw_deg": ("HPBW", "deg", 2),
        "d2d": ("D2D", "", 3),
        "d2d_db": ("D2D (dB)", "dB", 2),
        "first_null_db": ("1st null", "dB", 2),
        "fb_db": ("F/B", "dB", 2),
        "angle_min": ("Range min", "deg", 1),
        "angle_max": ("Range max", "deg", 1),
        "step_deg": ("Step", "deg", 3),
        "points": ("Points", "", 0),
    }

    items: List[MetricItem] = []
    used_keys = set()

    kind = str(metrics.get("kind", "")).upper()
    if kind:
        items.append(("Kind", kind))
        used_keys.add("kind")

    for key in (
        "peak_db",
        "peak_angle_deg",
        "hpbw_deg",
        "d2d",
        "d2d_db",
        "first_null_db",
        "fb_db",
        "angle_min",
        "angle_max",
        "step_deg",
        "points",
    ):
        if key == "fb_db" and kind != "H":
            continue
        if key not in metrics:
            continue
        label, unit, digits = key_meta[key]
        raw = metrics.get(key)
        if key == "points":
            try:
                value_txt = f"{int(raw):d}"
            except Exception:
                value_txt = "-"
        elif _is_finite_number(raw):
            value_txt = _fmt(raw, digits)
        else:
            value_txt = str(raw).strip() if raw is not None else "-"
        if unit and value_txt != "-":
            value_txt = f"{value_txt} {unit}"
        if value_txt and value_txt != "-":
            items.append((label, value_txt))
        used_keys.add(key)

    # Include any additional metrics keys not covered above.
    for key in sorted(metrics.keys()):
        if key in used_keys:
            continue
        raw = metrics.get(key)
        if isinstance(raw, bool):
            value_txt = "Yes" if raw else "No"
        elif isinstance(raw, int):
            value_txt = str(raw)
        elif _is_finite_number(raw):
            value_txt = _fmt(raw, 3)
        else:
            value_txt = str(raw).strip() if raw is not None else "-"
        if not value_txt or value_txt == "-":
            continue
        label = str(key).replace("_", " ").strip().title()
        items.append((label, value_txt))

    if not items:
        return [("Metrics", "Unavailable")]
    return items


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
