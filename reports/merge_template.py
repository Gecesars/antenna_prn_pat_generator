"""Template underlay merge for generated report content PDFs."""

from __future__ import annotations

import copy
import os
from typing import Callable, Optional

from pypdf import PdfReader, PdfWriter


ProgressCallback = Optional[Callable[[int, int, str], None]]


def apply_template_underlay(
    content_pdf_path: str,
    template_pdf_path: str,
    output_pdf_path: str,
    progress_cb: ProgressCallback = None,
) -> None:
    if not os.path.isfile(content_pdf_path):
        raise FileNotFoundError(f"Content PDF not found: {content_pdf_path}")
    if not os.path.isfile(template_pdf_path):
        raise FileNotFoundError(f"Template PDF not found: {template_pdf_path}")

    content_reader = PdfReader(content_pdf_path)
    template_reader = PdfReader(template_pdf_path)
    if not content_reader.pages:
        raise ValueError("Content PDF has no pages.")
    if not template_reader.pages:
        raise ValueError("Template PDF has no pages.")

    writer = PdfWriter()
    total = len(content_reader.pages)

    for idx, page in enumerate(content_reader.pages, start=1):
        template_idx = min(idx - 1, len(template_reader.pages) - 1)
        base_page = copy.copy(template_reader.pages[template_idx])
        base_page.merge_page(page)
        writer.add_page(base_page)
        if callable(progress_cb):
            progress_cb(idx, total, f"Merging template page {idx}/{total}")

    out_dir = os.path.dirname(os.path.abspath(output_pdf_path))
    os.makedirs(out_dir, exist_ok=True)
    with open(output_pdf_path, "wb") as f:
        writer.write(f)

