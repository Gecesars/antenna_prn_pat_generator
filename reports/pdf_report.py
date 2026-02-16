"""Public API for professional multipage PDF report export."""

from __future__ import annotations

import csv
import logging
import os
import re
import tempfile
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    Image as RLImage,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from . import layout
from .merge_template import apply_template_underlay
from .metrics import metrics_items_from_dict, split_metrics_columns
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
        if v == v:  # NaN-safe finite-ish check without importing math.
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


def _metrics_table(metrics: Mapping[str, object], width: float) -> Table:
    rows = split_metrics_columns(metrics_items_from_dict(metrics), max_items=14)
    data = [["Metric", "Value", "Metric", "Value"]] + rows
    col_w = float(width) / 4.0
    tbl = Table(data, colWidths=[col_w] * 4, hAlign="LEFT")
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef7")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#1f2933")),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cfd6df")),
                ("FONTSIZE", (0, 0), (-1, -1), layout.METRICS_FONT_SIZE),
                ("LEFTPADDING", (0, 0), (-1, -1), 3.0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3.0),
                ("TOPPADDING", (0, 0), (-1, -1), 1.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
                ("ALIGN", (0, 1), (0, -1), "LEFT"),
                ("ALIGN", (2, 1), (2, -1), "LEFT"),
                ("ALIGN", (1, 1), (1, -1), "RIGHT"),
                ("ALIGN", (3, 1), (3, -1), "RIGHT"),
            ]
        )
    )
    return tbl


def _write_full_csv(path: str, columns: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        wr = csv.writer(f)
        wr.writerow([str(c) for c in columns])
        for row in rows:
            wr.writerow(list(row))


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

    Payload format:
      {
        "report_title": "...",
        "pages": [
          {
            "page_title": "...",
            "page_subtitle": "...",
            "plot_image_path": "...png",
            "metrics": {...},
            "table": {
              "columns": [...],
              "rows": [...],
              "note": "..."
            }
          }
        ]
      }
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
    doc = None
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
        alignment=1,  # center
        spaceAfter=layout.TITLE_SPACING_PT,
    )
    h2 = ParagraphStyle(
        "ReportH2",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=layout.SUBTITLE_FONT_SIZE,
        leading=layout.SUBTITLE_FONT_SIZE + 1.6,
        alignment=1,  # center
        textColor=colors.HexColor("#3f4b5a"),
        spaceAfter=layout.SUBTITLE_SPACING_PT,
    )
    note_style = ParagraphStyle(
        "ReportNote",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=layout.BODY_FONT_SIZE,
        leading=layout.BODY_FONT_SIZE + 1.2,
        textColor=colors.HexColor("#5d6b7a"),
    )

    story = []
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
        metric_angles = _parse_metric_angles(metrics if isinstance(metrics, Mapping) else {})
        if save_full_csv:
            csv_name = f"{idx:02d}_{_slug(page_title)}.csv"
            csv_path = os.path.join(csv_root, csv_name)
            _write_full_csv(csv_path, columns, rows_full)
            csv_files.append(csv_path)

        max_h = layout.mm_to_pt(layout.TOP_BLOCK_MAX_HEIGHT_MM)
        left_w = safe.width * layout.TOP_BLOCK_COL_RATIOS[0]
        right_w = safe.width - left_w

        plot = RLImage(plot_path)
        plot._restrictSize(left_w, max_h)
        metrics_tbl = _metrics_table(
            metrics if isinstance(metrics, Mapping) else {},
            width=right_w,
        )
        top_block = Table(
            [[plot, metrics_tbl]],
            colWidths=[left_w, right_w],
            hAlign="LEFT",
            style=TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 0.0),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 0.0),
                    ("TOPPADDING", (0, 0), (-1, -1), 0.0),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 0.0),
                ]
            ),
        )

        title_para = Paragraph(page_title, h1)
        subtitle_para = Paragraph(page_subtitle, h2) if page_subtitle else None
        _, h_title = title_para.wrap(safe.width, safe.height)
        if subtitle_para is not None:
            _, h_subtitle = subtitle_para.wrap(safe.width, safe.height)
        else:
            h_subtitle = layout.SUBTITLE_SPACING_PT
        _, h_top = top_block.wrap(safe.width, safe.height)

        rows_limit = max_rows
        rows_compact, compacted = compact_table_rows(rows_full, max_rows=rows_limit, metric_angles=metric_angles)
        fit_reduced = False
        while True:
            columns_disp, rows_disp = fold_two_column_table(columns, rows_compact)
            note_text = table_note
            if compacted:
                compact_note = "Table compacted for PDF display; full data saved in CSV."
                note_text = f"{note_text} {compact_note}".strip()

            h_note = 0.0
            if note_text:
                note_para = Paragraph(note_text, note_style)
                _, h_note = note_para.wrap(safe.width, safe.height)
                h_note += 3.5
            avail_table_h = safe.height - h_title - h_subtitle - h_top - layout.BLOCK_SPACING_PT - h_note - 2.0
            avail_table_h = max(56.0, float(avail_table_h))

            table_for_pdf = build_table_for_pdf(
                columns_disp,
                rows_disp,
                available_width=safe.width,
                font_size=layout.TABLE_FONT_SIZE,
            )
            _, h_table = table_for_pdf.wrap(safe.width, avail_table_h)
            if h_table <= avail_table_h + 1e-6:
                break
            if rows_limit <= 12:
                break
            rows_limit = max(12, rows_limit - 4)
            rows_compact, _ = compact_table_rows(rows_full, max_rows=rows_limit, metric_angles=metric_angles)
            compacted = True
            fit_reduced = True

        if compacted:
            warnings.append(f"{page_title}: table compacted for PDF display.")
        if fit_reduced:
            warnings.append(f"{page_title}: table rows reduced to fit one page.")

        page_result = {
            "title": page_title,
            "compacted": bool(compacted),
            "rows_full": len(rows_full),
            "rows_pdf": len(rows_disp),
        }
        per_page.append(page_result)

        story.append(title_para)
        if subtitle_para is not None:
            story.append(subtitle_para)
        else:
            story.append(Spacer(1, layout.SUBTITLE_SPACING_PT))
        story.append(KeepTogether([top_block]))
        story.append(Spacer(1, layout.BLOCK_SPACING_PT))
        story.append(table_for_pdf)
        if note_text:
            story.append(Spacer(1, 3.5))
            story.append(Paragraph(note_text, note_style))

        if idx < total_pages:
            story.append(PageBreak())

    if callable(progress_cb):
        progress_cb(0, total_pages, "Rendering content PDF")

    doc = SimpleDocTemplate(
        content_pdf_path,
        pagesize=layout.PAGE_SIZE,
        leftMargin=safe.left,
        rightMargin=layout.mm_to_pt(layout.SAFE_RIGHT_MM),
        topMargin=layout.mm_to_pt(layout.SAFE_TOP_MM),
        bottomMargin=safe.bottom,
        title=str(payload.get("report_title", "EFTX Report")),
        author=str(payload.get("author", "EFTX")),
    )
    doc.build(story)

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
