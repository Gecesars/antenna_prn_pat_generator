"""Table compaction and styling helpers for PDF reporting."""

from __future__ import annotations

import re
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from reportlab.lib import colors
from reportlab.platypus import Table, TableStyle


NUM_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


def _to_float(value) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip().replace(",", ".")
    if not s or not NUM_RE.match(s):
        return float("nan")
    try:
        return float(s)
    except Exception:
        return float("nan")


def _is_numeric_text(value) -> bool:
    return np.isfinite(_to_float(value))


def _row_to_text(row: Sequence[object], width: int) -> List[str]:
    r = [str(x) if x is not None else "" for x in row]
    if len(r) < width:
        r = r + ([""] * (width - len(r)))
    return r[:width]


def compact_table_rows(
    rows: Sequence[Sequence[object]],
    max_rows: int,
    metric_angles: Iterable[float] | None = None,
) -> Tuple[List[List[object]], bool]:
    all_rows = [list(r) for r in rows]
    n = len(all_rows)
    max_rows = max(10, int(max_rows))
    if n <= max_rows:
        return all_rows, False

    keep = {0, n - 1}

    # Angle and value columns (first two columns, when numeric).
    ang = np.asarray([_to_float(r[0] if r else None) for r in all_rows], dtype=float)
    val = np.asarray([_to_float(r[1] if len(r) > 1 else None) for r in all_rows], dtype=float)

    if np.isfinite(val).any():
        keep.add(int(np.nanargmax(val)))
        keep.add(int(np.nanargmin(val)))
    if metric_angles is not None and np.isfinite(ang).any():
        for a_ref in metric_angles:
            try:
                a0 = float(a_ref)
            except Exception:
                continue
            if not np.isfinite(a0):
                continue
            idx = int(np.nanargmin(np.abs(ang - a0)))
            keep.add(idx)

    slots = max_rows - len(keep)
    if slots > 0:
        candidates = [i for i in range(n) if i not in keep]
        if len(candidates) <= slots:
            keep.update(candidates)
        else:
            picks = np.linspace(0, len(candidates) - 1, num=slots, dtype=int)
            keep.update(candidates[int(i)] for i in picks.tolist())

    selected = sorted(keep)
    return [all_rows[i] for i in selected], True


def fold_two_column_table(
    columns: Sequence[str],
    rows: Sequence[Sequence[object]],
) -> Tuple[List[str], List[List[object]]]:
    cols = [str(c) for c in columns]
    rows_list = [list(r) for r in rows]
    if len(cols) != 2 or not rows_list:
        return cols, rows_list

    mid = (len(rows_list) + 1) // 2
    left = rows_list[:mid]
    right = rows_list[mid:]
    out_rows: List[List[object]] = []
    for i in range(mid):
        l = left[i]
        r = right[i] if i < len(right) else ["", ""]
        out_rows.append([l[0], l[1], r[0], r[1]])
    out_cols = [cols[0], cols[1], cols[0], cols[1]]
    return out_cols, out_rows


def build_table_for_pdf(
    columns: Sequence[str],
    rows: Sequence[Sequence[object]],
    *,
    available_width: float,
    font_size: float = 6.8,
) -> Table:
    if not columns:
        columns = ["Column", "Value"]
    width = len(columns)
    data: List[List[str]] = [_row_to_text(columns, width)]
    data.extend(_row_to_text(r, width) for r in rows)

    col_width = float(available_width) / max(width, 1)
    table = Table(data, colWidths=[col_width] * width, repeatRows=1, hAlign="LEFT")

    style = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dfe8f5")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#1f2933")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), float(font_size)),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cfd6df")),
        ("TOPPADDING", (0, 0), (-1, -1), 0.2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0.2),
        ("LEFTPADDING", (0, 0), (-1, -1), 1.5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 1.5),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]

    for r in range(1, len(data)):
        bg = "#f8fafc" if r % 2 == 0 else "#ffffff"
        style.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor(bg)))

    # Keep table visually uniform for technical reading: center all cells.
    style.append(("ALIGN", (0, 0), (-1, -1), "CENTER"))

    table.setStyle(TableStyle(style))
    return table
