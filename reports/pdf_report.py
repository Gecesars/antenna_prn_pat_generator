"""Public API for professional multipage PDF report export."""

from __future__ import annotations

import csv
import logging
import math
import os
import re
import tempfile
import time
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image as PILImage
try:
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas as rl_canvas
    from reportlab.platypus import Paragraph, Table, TableStyle
    _REPORTLAB_IMPORT_ERROR = None
except Exception as _exc:
    colors = None
    ParagraphStyle = None
    getSampleStyleSheet = None
    ImageReader = None
    rl_canvas = None
    Paragraph = None
    Table = None
    TableStyle = None
    _REPORTLAB_IMPORT_ERROR = _exc

from core.audit import emit_audit
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


def _ensure_reportlab_available() -> None:
    if _REPORTLAB_IMPORT_ERROR is None:
        return
    raise ReportExportError(
        "PDF export dependency missing: reportlab. "
        "Install with 'python -m pip install reportlab' in the same environment used to run the app."
    ) from _REPORTLAB_IMPORT_ERROR


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
    items = metrics_items_from_dict(metrics)
    if not items:
        items = [("Metrics", "Unavailable")]

    lbl_style = ParagraphStyle(
        "MetricLabel",
        fontName="Helvetica-Bold",
        fontSize=layout.METRICS_FONT_SIZE - 0.2,
        leading=layout.METRICS_FONT_SIZE + 1.2,
        textColor=colors.HexColor("#1f2933"),
    )
    val_style = ParagraphStyle(
        "MetricValue",
        fontName="Helvetica",
        fontSize=layout.METRICS_FONT_SIZE - 0.2,
        leading=layout.METRICS_FONT_SIZE + 1.2,
        textColor=colors.HexColor("#111111"),
    )
    rows = split_metrics_columns(items, max_items=max(1, len(items)))
    data = []
    for lk, lv, rk, rv in rows:
        data.append(
            [
                Paragraph(f"{lk}:", lbl_style) if lk else Paragraph("", lbl_style),
                Paragraph(str(lv), val_style) if lv else Paragraph("", val_style),
                Paragraph(f"{rk}:", lbl_style) if rk else Paragraph("", lbl_style),
                Paragraph(str(rv), val_style) if rv else Paragraph("", val_style),
            ]
        )

    c1 = float(width) * 0.18
    c2 = float(width) * 0.32
    c3 = float(width) * 0.18
    c4 = float(width) - c1 - c2 - c3
    tbl = Table(data, colWidths=[c1, c2, c3, c4], hAlign="LEFT")
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f5f8fc")),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#d3dbe5")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 3.0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3.0),
                ("TOPPADDING", (0, 0), (-1, -1), 1.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
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


def _normalize_glossary(glossary_obj: object) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    seen = set()
    if isinstance(glossary_obj, Mapping):
        items = glossary_obj.items()
    elif isinstance(glossary_obj, Sequence) and not isinstance(glossary_obj, (str, bytes)):
        items = []
        for entry in glossary_obj:
            if isinstance(entry, Mapping):
                term = str(entry.get("term", "") or entry.get("name", "")).strip()
                definition = str(entry.get("definition", "") or entry.get("desc", "")).strip()
                items.append((term, definition))
            elif isinstance(entry, Sequence) and not isinstance(entry, (str, bytes)) and len(entry) >= 2:
                term = str(entry[0]).strip()
                definition = str(entry[1]).strip()
                items.append((term, definition))
            elif isinstance(entry, str):
                items.append((entry.strip(), ""))
    else:
        items = []

    for term, definition in items:
        t = str(term or "").strip()
        d = str(definition or "").strip()
        if not t:
            continue
        key = t.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append((t, d))
    return out


def _build_glossary_table(entries: Sequence[Tuple[str, str]], width: float) -> Table:
    header_style = ParagraphStyle(
        "GlossaryHeader",
        fontName="Helvetica-Bold",
        fontSize=layout.TABLE_FONT_SIZE + 0.4,
        leading=layout.TABLE_FONT_SIZE + 2.0,
        textColor=colors.HexColor("#1f2933"),
    )
    term_style = ParagraphStyle(
        "GlossaryTerm",
        fontName="Helvetica-Bold",
        fontSize=layout.TABLE_FONT_SIZE,
        leading=layout.TABLE_FONT_SIZE + 1.8,
        textColor=colors.HexColor("#1f2933"),
    )
    def_style = ParagraphStyle(
        "GlossaryDef",
        fontName="Helvetica",
        fontSize=layout.TABLE_FONT_SIZE,
        leading=layout.TABLE_FONT_SIZE + 1.8,
        textColor=colors.HexColor("#111111"),
    )

    data = [[Paragraph("Termo", header_style), Paragraph("Descricao", header_style)]]
    for term, definition in entries:
        data.append([Paragraph(term, term_style), Paragraph(definition or "-", def_style)])

    col1 = float(width) * 0.22
    col2 = float(width) - col1
    table = Table(data, colWidths=[col1, col2], repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dfe8f5")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#1f2933")),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cfd6df")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 3.0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3.0),
                ("TOPPADDING", (0, 0), (-1, -1), 2.0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.0),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ]
        )
    )
    return table


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
    _ensure_reportlab_available()
    log = logger or logging.getLogger(__name__)
    t_start = time.perf_counter()
    emit_audit(
        "EXPORT_REPORT_PDF_START",
        logger=log,
        output_pdf=output_pdf_path,
        template_pdf=template_pdf_path,
        save_full_csv=int(bool(save_full_csv)),
        max_rows=int(max_display_rows),
    )
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
    desc_style = ParagraphStyle(
        "ReportDescription",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=layout.BODY_FONT_SIZE + 0.2,
        leading=layout.BODY_FONT_SIZE + 2.0,
        alignment=0,
        textColor=colors.HexColor("#2b3a4a"),
    )
    gloss_title_style = ParagraphStyle(
        "GlossaryTitle",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=layout.TITLE_FONT_SIZE - 0.5,
        leading=layout.TITLE_FONT_SIZE + 1.0,
        alignment=1,
    )
    gloss_sub_style = ParagraphStyle(
        "GlossarySubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=layout.SUBTITLE_FONT_SIZE,
        leading=layout.SUBTITLE_FONT_SIZE + 1.6,
        alignment=1,
        textColor=colors.HexColor("#3f4b5a"),
    )

    canv = rl_canvas.Canvas(content_pdf_path, pagesize=layout.PAGE_SIZE)
    total_pages = len(pages)
    glossary_entries = _normalize_glossary(payload.get("glossary", []) if isinstance(payload, Mapping) else [])
    glossary_pages = 0

    for idx, page in enumerate(pages, start=1):
        page_t0 = time.perf_counter()
        if callable(cancel_check) and cancel_check():
            raise ReportCancelled("Report export cancelled by user.")
        if callable(progress_cb):
            progress_cb(idx, total_pages, f"Building report page {idx}/{total_pages}")

        if not isinstance(page, Mapping):
            raise ReportExportError(f"Invalid page payload at index {idx}.")
        page_title = str(page.get("page_title", "") or "").strip()
        page_subtitle = str(page.get("page_subtitle", "") or "").strip()
        page_description = str(page.get("page_description", "") or "").strip()
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
        description_para = Paragraph(page_description, desc_style) if page_description else None
        h_title = title_para.wrap(safe.width, safe.height)[1]
        h_subtitle = subtitle_para.wrap(safe.width, safe.height)[1] if subtitle_para is not None else 0.0
        h_desc = description_para.wrap(safe.width, safe.height)[1] if description_para is not None else 0.0
        h_desc_gap = 4.0 if description_para is not None else 0.0

        metrics_tbl = _metrics_table(metrics if isinstance(metrics, Mapping) else {}, width=safe.width)
        h_metrics = metrics_tbl.wrapOn(canv, safe.width, safe.height)[1]

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
        plot_h = layout.mm_to_pt(72.0)
        min_plot_h = layout.mm_to_pt(48.0)
        pref_plot_h = layout.mm_to_pt(86.0)
        min_table_h = 56.0

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

            available_body_h = safe.height - h_title - h_subtitle - h_desc - h_desc_gap
            remaining_h = available_body_h - h_metrics - (2.0 * layout.BLOCK_SPACING_PT) - h_note - 4.0
            if remaining_h <= (min_plot_h + min_table_h):
                plot_h = min_plot_h
            else:
                plot_h = min(pref_plot_h, max(min_plot_h, remaining_h - min_table_h))
            available_table_h = max(min_table_h, float(remaining_h - plot_h))

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
                # Last fallback: shrink plot area to preserve metrics + table on the same page.
                if pref_plot_h > min_plot_h + 1.0:
                    pref_plot_h = max(min_plot_h, pref_plot_h - 8.0)
                    fit_reduced = True
                    continue
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

        # Draw sequence (single page per cut): header -> plot -> full metrics table -> data table -> showPage.
        y_top = safe.top
        used = _draw_centered_paragraph(canv, title_para, safe.left, safe.width, y_top, safe.height)
        y_top -= used + layout.TITLE_SPACING_PT
        if subtitle_para is not None:
            used = _draw_centered_paragraph(canv, subtitle_para, safe.left, safe.width, y_top, safe.height)
            y_top -= used + layout.SUBTITLE_SPACING_PT
        else:
            y_top -= layout.SUBTITLE_SPACING_PT
        if description_para is not None:
            desc_h = description_para.wrap(safe.width, safe.height)[1]
            description_para.drawOn(canv, safe.left, y_top - desc_h)
            y_top -= desc_h + h_desc_gap

        plot_bottom = y_top - plot_h
        _draw_plot_fitted(canv, plot_path, safe.left, plot_bottom, safe.width, plot_h)

        y_metrics_top = plot_bottom - layout.BLOCK_SPACING_PT
        metrics_h = metrics_tbl.wrapOn(canv, safe.width, safe.height)[1]
        metrics_tbl.drawOn(canv, safe.left, y_metrics_top - metrics_h)

        y_table_top = y_metrics_top - metrics_h - layout.BLOCK_SPACING_PT
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
        emit_audit(
            "REPORT_PAGE_BUILT",
            logger=log,
            index=int(idx),
            title=page_title,
            rows_full=int(len(rows_full)),
            rows_pdf=int(len(rows_disp)),
            compacted=int(bool(compacted)),
            elapsed_ms=round((time.perf_counter() - page_t0) * 1000.0, 3),
        )

    if glossary_entries:
        remaining = list(glossary_entries)
        part_idx = 1
        while remaining:
            if callable(cancel_check) and cancel_check():
                raise ReportCancelled("Report export cancelled by user.")
            if callable(progress_cb):
                progress_cb(min(part_idx, total_pages), total_pages, "Building glossary page")

            title_txt = "Glossario de Termos"
            if part_idx > 1:
                title_txt += f" (continuacao {part_idx})"
            title_para = Paragraph(title_txt, gloss_title_style)
            subtitle_para = Paragraph(
                "Definicoes usadas nos diagramas, tabelas e metricas do projeto.",
                gloss_sub_style,
            )
            y_top = safe.top
            used = _draw_centered_paragraph(canv, title_para, safe.left, safe.width, y_top, safe.height)
            y_top -= used + layout.TITLE_SPACING_PT
            used = _draw_centered_paragraph(canv, subtitle_para, safe.left, safe.width, y_top, safe.height)
            y_top -= used + layout.SUBTITLE_SPACING_PT
            available_h = max(80.0, y_top - safe.bottom)

            take = len(remaining)
            table = None
            while take > 0:
                table = _build_glossary_table(remaining[:take], safe.width)
                h_tbl = table.wrapOn(canv, safe.width, available_h)[1]
                if h_tbl <= available_h + 1e-6:
                    break
                take -= 1
            if take <= 0 or table is None:
                take = 1
                table = _build_glossary_table(remaining[:1], safe.width)
                h_tbl = table.wrapOn(canv, safe.width, available_h)[1]

            table.drawOn(canv, safe.left, y_top - h_tbl)
            canv.showPage()
            glossary_pages += 1
            remaining = remaining[take:]
            part_idx += 1

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
        "glossary_pages": glossary_pages,
        "output_pages": total_pages + glossary_pages,
        "dpi": dpi,
    }
    emit_audit(
        "EXPORT_REPORT_PDF_OK",
        logger=log,
        output_pdf=output_pdf_path,
        pages=int(total_pages),
        glossary_pages=int(glossary_pages),
        elapsed_ms=round((time.perf_counter() - t_start) * 1000.0, 3),
    )
    log.info("Report PDF exported: %s (pages=%s)", output_pdf_path, total_pages)
    return result
