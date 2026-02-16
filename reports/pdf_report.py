"""Public API for professional multipage PDF report export."""

from __future__ import annotations

import csv
import logging
import math
import os
import re
import tempfile
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.platypus import Paragraph, Table, TableStyle

from . import layout
from .merge_template import apply_template_underlay
from .metrics import metrics_items_from_dict
from .tables import build_table_for_pdf, compact_table_rows, fold_two_column_table


ProgressCallback = Optional[Callable[[int, int, str], None]]
CancelCheck = Optional[Callable[[], bool]]


class ReportExportError(RuntimeError):
    """Raised when report export cannot continue."""


class ReportCancelled(RuntimeError):
    """Raised when a user cancellation token is triggered."""


def _slug(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return "page"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("._") or "page"


def _as_columns_rows(table_obj: Mapping[str, object]) -> Tuple[List[str], List[List[object]], str]:
    if not isinstance(table_obj, Mapping):
        raise ReportExportError("Invalid table payload for report page.")
    columns = table_obj.get("columns", [])
    rows = table_obj.get("rows", [])
    note = str(table_obj.get("note", "") or "").strip()
    if not isinstance(columns, Sequence) or not columns:
        raise ReportExportError("Table columns missing in report page.")
    if not isinstance(rows, Sequence) or not rows:
        raise ReportExportError("Table rows missing in report page.")
    cols = [str(c) for c in columns]
    rows_out = [list(r) if isinstance(r, Sequence) else [r] for r in rows]
    return cols, rows_out, note


def _parse_metric_angles(metrics: Mapping[str, object]) -> List[float]:
    angles: List[float] = []
    if not isinstance(metrics, Mapping):
        return angles
    for key in ("peak_angle_deg", "hpbw_left_deg", "hpbw_right_deg"):
        try:
            v = float(metrics.get(key))
        except Exception:
            continue
        if math.isfinite(v):
            angles.append(v)
    return angles


def _validate_image(path: str, warnings: List[str], page_label: str) -> None:
    if not os.path.isfile(path):
        raise ReportExportError(f"Plot image missing: {path}")
    with PILImage.open(path) as img:
        w, h = img.size
    if w < layout.MIN_IMAGE_WIDTH_PX or h < layout.MIN_IMAGE_HEIGHT_PX:
        warnings.append(
            f"{page_label}: low image resolution ({w}x{h}px). Recommended >= "
            f"{layout.MIN_IMAGE_WIDTH_PX}x{layout.MIN_IMAGE_HEIGHT_PX}px."
        )


def _write_full_csv(path: str, columns: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        wr = csv.writer(f)
        wr.writerow([str(c) for c in columns])
        for row in rows:
            wr.writerow(list(row))


def _to_float(value) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if value is None:
        return float("nan")
    try:
        return float(str(value).strip().replace(",", "."))
    except Exception:
        return float("nan")


def _estimate_step(values: Sequence[float]) -> float:
    arr = [float(v) for v in values if math.isfinite(float(v))]
    if len(arr) < 2:
        return float("nan")
    arr_sorted = sorted(set(arr))
    if len(arr_sorted) < 2:
        return float("nan")
    diffs = [abs(arr_sorted[i + 1] - arr_sorted[i]) for i in range(len(arr_sorted) - 1)]
    diffs = [d for d in diffs if d > 1e-12]
    if not diffs:
        return float("nan")
    return float(np.median(np.asarray(diffs, dtype=float)))  # type: ignore[name-defined]


def _format_table_rows(columns: Sequence[str], rows: Sequence[Sequence[object]]) -> List[List[str]]:
    cols = [str(c) for c in columns]
    rows_in = [list(r) for r in rows]
    if not rows_in:
        return []

    angle_cols: List[int] = []
    for i, c in enumerate(cols):
        lc = c.lower()
        if ("ang" in lc) or ("theta" in lc) or ("phi" in lc) or ("grau" in lc) or ("deg" in lc):
            angle_cols.append(i)
    if not angle_cols:
        # Fallback: assume Ang/Val paired columns.
        angle_cols = [i for i in range(0, len(cols), 2)]

    value_cols = {i + 1 for i in angle_cols if i + 1 < len(cols)}
    angle_digits: Dict[int, int] = {}
    for ci in angle_cols:
        cvals = [_to_float(r[ci]) for r in rows_in if len(r) > ci]
        step = _estimate_step(cvals)
        if math.isfinite(step) and step >= 0.999:
            angle_digits[ci] = 0
        else:
            angle_digits[ci] = 1

    out: List[List[str]] = []
    for row in rows_in:
        line: List[str] = []
        for ci, val in enumerate(row):
            fv = _to_float(val)
            if math.isfinite(fv):
                if ci in angle_cols:
                    line.append(f"{fv:.{angle_digits.get(ci, 1)}f}")
                elif ci in value_cols:
                    line.append(f"{fv:.2f}")
                else:
                    line.append(f"{fv:.2f}")
            else:
                line.append(str(val))
        out.append(line)
    return out


def _metrics_table(metrics: Mapping[str, object], width: float) -> Table:
    items = metrics_items_from_dict(metrics)[:14]
    if not items:
        items = [("Metrics", "Unavailable")]

    lbl_style = ParagraphStyle(
        "MetricLabel",
        fontName="Helvetica-Bold",
        fontSize=layout.METRICS_FONT_SIZE,
        leading=layout.METRICS_FONT_SIZE + 1.5,
        textColor=colors.HexColor("#1f2933"),
    )
    val_style = ParagraphStyle(
        "MetricValue",
        fontName="Helvetica",
        fontSize=layout.METRICS_FONT_SIZE,
        leading=layout.METRICS_FONT_SIZE + 1.5,
        textColor=colors.HexColor("#111111"),
    )
    data = [[Paragraph(f"{k}:", lbl_style), Paragraph(str(v), val_style)] for (k, v) in items]

    col1 = float(width) * 0.46
    col2 = float(width) - col1
    tbl = Table(data, colWidths=[col1, col2], hAlign="LEFT")
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f5f8fc")),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#d3dbe5")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4.0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4.0),
                ("TOPPADDING", (0, 0), (-1, -1), 2.0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.0),
                ("ALIGN", (0, 0), (0, -1), "LEFT"),
                ("ALIGN", (1, 0), (1, -1), "RIGHT"),
            ]
        )
    )
    return tbl


def _draw_centered_paragraph(
    canv: rl_canvas.Canvas,
    para: Paragraph,
    left: float,
    width: float,
    y_top: float,
    max_height: float,
) -> float:
    w, h = para.wrap(width, max_height)
    x = left + (width - w) * 0.5
    y = y_top - h
    para.drawOn(canv, x, y)
    return h


def _draw_plot_fitted(
    canv: rl_canvas.Canvas,
    image_path: str,
    box_left: float,
    box_bottom: float,
    box_width: float,
    box_height: float,
) -> None:
    with PILImage.open(image_path) as img:
        iw, ih = img.size
    if iw <= 0 or ih <= 0:
        return
    scale = min(float(box_width) / float(iw), float(box_height) / float(ih))
    dw = max(1.0, float(iw) * scale)
    dh = max(1.0, float(ih) * scale)
    dx = box_left + (float(box_width) - dw) * 0.5
    dy = box_bottom + (float(box_height) - dh) * 0.5
    canv.drawImage(ImageReader(image_path), dx, dy, width=dw, height=dh, mask="auto")


def export_report_pdf(
    payload: Mapping[str, object],
    output_pdf_path: str,
    template_pdf_path: str,
    *,
    dpi: int = layout.DEFAULT_IMAGE_DPI,
    save_full_csv: bool = True,
    csv_output_dir: Optional[str] = None,
    max_display_rows: int = layout.DEFAULT_DISPLAY_ROWS,
    progress_cb: ProgressCallback = None,
    cancel_check: CancelCheck = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, object]:
    """
    Generate a multipage PDF report and merge template as underlay.
    One page is generated for each cut in payload["pages"].
    """
    log = logger or logging.getLogger(__name__)
    pages = payload.get("pages", []) if isinstance(payload, Mapping) else []
    if not isinstance(pages, Sequence) or not pages:
        raise ReportExportError("Report payload has no pages.")
    if not os.path.isfile(template_pdf_path):
        raise FileNotFoundError(f"Template PDF not found: {template_pdf_path}")

    dpi = int(dpi) if int(dpi) in layout.SUPPORTED_DPI else layout.DEFAULT_IMAGE_DPI
    max_rows = max(layout.MIN_DISPLAY_ROWS, min(layout.MAX_DISPLAY_ROWS, int(max_display_rows)))

    safe = layout.get_safe_box(layout.PAGE_SIZE)
    warnings: List[str] = []
    csv_files: List[str] = []
    per_page: List[Dict[str, object]] = []

    report_dir = os.path.dirname(os.path.abspath(output_pdf_path))
    os.makedirs(report_dir, exist_ok=True)
    if save_full_csv:
        csv_root = csv_output_dir or os.path.join(report_dir, "report_tables")
        os.makedirs(csv_root, exist_ok=True)
    else:
        csv_root = ""

    tmp_root = tempfile.mkdtemp(prefix="eftx_report_")
    content_pdf_path = os.path.join(tmp_root, "report_content.pdf")

    styles = getSampleStyleSheet()
    h1 = ParagraphStyle(
        "ReportH1",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=layout.TITLE_FONT_SIZE,
        leading=layout.TITLE_FONT_SIZE + 1.8,
        alignment=1,
    )
    h2 = ParagraphStyle(
        "ReportH2",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=layout.SUBTITLE_FONT_SIZE,
        leading=layout.SUBTITLE_FONT_SIZE + 1.6,
        alignment=1,
        textColor=colors.HexColor("#3f4b5a"),
    )
    note_style = ParagraphStyle(
        "ReportNote",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=layout.BODY_FONT_SIZE,
        leading=layout.BODY_FONT_SIZE + 1.2,
        textColor=colors.HexColor("#5d6b7a"),
    )

    canv = rl_canvas.Canvas(content_pdf_path, pagesize=layout.PAGE_SIZE)
    total_pages = len(pages)

    for idx, page in enumerate(pages, start=1):
        if callable(cancel_check) and cancel_check():
            raise ReportCancelled("Report export cancelled by user.")
        if callable(progress_cb):
            progress_cb(idx, total_pages, f"Building report page {idx}/{total_pages}")

        if not isinstance(page, Mapping):
            raise ReportExportError(f"Invalid page payload at index {idx}.")
        page_title = str(page.get("page_title", "") or "").strip()
        page_subtitle = str(page.get("page_subtitle", "") or "").strip()
        plot_path = str(page.get("plot_image_path", "") or "").strip()
        metrics = page.get("metrics", {})
        columns, rows_full, table_note = _as_columns_rows(page.get("table", {}))

        if not page_title:
            raise ReportExportError(f"Missing page_title in page {idx}.")
        if not plot_path:
            raise ReportExportError(f"Missing plot_image_path in page {idx}.")

        _validate_image(plot_path, warnings, page_title)

        if save_full_csv:
            csv_name = f"{idx:02d}_{_slug(page_title)}.csv"
            csv_path = os.path.join(csv_root, csv_name)
            _write_full_csv(csv_path, columns, rows_full)
            csv_files.append(csv_path)

        title_para = Paragraph(page_title, h1)
        subtitle_para = Paragraph(page_subtitle, h2) if page_subtitle else None
        h_title = title_para.wrap(safe.width, safe.height)[1]
        h_subtitle = subtitle_para.wrap(safe.width, safe.height)[1] if subtitle_para is not None else 0.0

        top_h_base = layout.mm_to_pt(layout.TOP_BLOCK_MAX_HEIGHT_MM)
        left_w = safe.width * layout.TOP_BLOCK_COL_RATIOS[0]
        right_w = safe.width - left_w
        metrics_tbl = _metrics_table(metrics if isinstance(metrics, Mapping) else {}, width=right_w)
        h_metrics = metrics_tbl.wrapOn(canv, right_w, top_h_base)[1]
        top_h = min(max(top_h_base, h_metrics), safe.height * 0.62)

        metric_angles = _parse_metric_angles(metrics if isinstance(metrics, Mapping) else {})
        rows_limit = max_rows
        compacted = False
        fit_reduced = False
        rows_disp: List[List[object]] = []
        columns_disp: List[str] = []
        table_for_pdf = None
        note_text = ""
        h_table = 0.0
        h_note = 0.0

        while True:
            rows_candidate, compact_now = compact_table_rows(rows_full, max_rows=rows_limit, metric_angles=metric_angles)
            compacted = bool(compact_now)
            columns_disp, rows_disp = fold_two_column_table(columns, rows_candidate)
            rows_fmt = _format_table_rows(columns_disp, rows_disp)

            note_text = str(table_note or "").strip()
            if compacted:
                compact_note = "Tabela exibida compactada; dados completos no CSV."
                note_text = f"{note_text} {compact_note}".strip()
            elif "compact" in note_text.lower():
                note_text = ""

            h_note = 0.0
            if note_text:
                note_para = Paragraph(note_text, note_style)
                h_note = note_para.wrap(safe.width, safe.height)[1] + 3.5

            available_table_h = safe.height - h_title - h_subtitle - top_h - layout.BLOCK_SPACING_PT - h_note - 4.0
            available_table_h = max(56.0, float(available_table_h))

            table_for_pdf = build_table_for_pdf(
                columns_disp,
                rows_fmt,
                available_width=safe.width,
                font_size=layout.TABLE_FONT_SIZE,
            )
            h_table = table_for_pdf.wrapOn(canv, safe.width, available_table_h)[1]
            if h_table <= available_table_h + 1e-6:
                break
            if rows_limit <= 12:
                break
            rows_limit = max(12, rows_limit - 4)
            compacted = True
            fit_reduced = True

        if compacted:
            warnings.append(f"{page_title}: table compacted for PDF display.")
        if fit_reduced:
            warnings.append(f"{page_title}: table rows reduced to fit one page.")

        per_page.append(
            {
                "title": page_title,
                "compacted": bool(compacted),
                "rows_full": len(rows_full),
                "rows_pdf": len(rows_disp),
            }
        )

        # Draw sequence (single page per cut): header -> plot -> metrics -> table -> showPage.
        y_top = safe.top
        used = _draw_centered_paragraph(canv, title_para, safe.left, safe.width, y_top, safe.height)
        y_top -= used + layout.TITLE_SPACING_PT
        if subtitle_para is not None:
            used = _draw_centered_paragraph(canv, subtitle_para, safe.left, safe.width, y_top, safe.height)
            y_top -= used + layout.SUBTITLE_SPACING_PT
        else:
            y_top -= layout.SUBTITLE_SPACING_PT

        top_bottom = y_top - top_h
        _draw_plot_fitted(canv, plot_path, safe.left, top_bottom, left_w, top_h)
        metrics_h = metrics_tbl.wrapOn(canv, right_w, top_h)[1]
        metrics_tbl.drawOn(canv, safe.left + left_w, top_bottom + top_h - metrics_h)

        y_table_top = top_bottom - layout.BLOCK_SPACING_PT
        if table_for_pdf is not None:
            table_for_pdf.drawOn(canv, safe.left, y_table_top - h_table)
            y_after_table = y_table_top - h_table
        else:
            y_after_table = y_table_top

        if note_text:
            note_para = Paragraph(note_text, note_style)
            note_h = note_para.wrap(safe.width, safe.height)[1]
            note_para.drawOn(canv, safe.left, y_after_table - 3.5 - note_h)

        canv.showPage()

    canv.save()

    if callable(progress_cb):
        progress_cb(0, total_pages, "Applying template underlay")
    apply_template_underlay(content_pdf_path, template_pdf_path, output_pdf_path, progress_cb=progress_cb)

    result = {
        "output_pdf_path": output_pdf_path,
        "content_pdf_path": content_pdf_path,
        "csv_files": csv_files,
        "warnings": warnings,
        "page_results": per_page,
        "pages": total_pages,
        "dpi": dpi,
    }
    log.info("Report PDF exported: %s (pages=%s)", output_pdf_path, total_pages)
    return result
