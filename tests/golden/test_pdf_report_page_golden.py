from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from reports.pdf_report import export_report_pdf

try:
    import pypdfium2 as pdfium
except Exception:  # pragma: no cover
    pdfium = None  # type: ignore


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    mse = float(np.mean((a - b) ** 2))
    if mse <= 1e-12:
        return 99.0
    return 20.0 * math.log10(255.0 / math.sqrt(mse))


def _render_pdf_page(pdf_path: Path, page_index: int = 0, scale: float = 2.0) -> np.ndarray:
    if pdfium is None:
        raise RuntimeError("pypdfium2 is required to render PDF pages.")
    doc = pdfium.PdfDocument(str(pdf_path))
    page = doc.get_page(int(page_index))
    try:
        pil = page.render(scale=float(scale), rotation=0).to_pil()
        return np.asarray(pil.convert("RGB"), dtype=np.uint8)
    finally:
        page.close()
        doc.close()


@pytest.mark.golden
def test_pdf_report_first_page_golden(tmp_path):
    if pdfium is None:
        pytest.skip("pypdfium2 is not available in this environment.")

    base = Path(__file__).resolve().parent
    plot_path = base / "inputs" / "pdf_plot_base.png"
    template_path = base / "inputs" / "pdf_template_base.pdf"
    expected_img_path = base / "expected" / "pdf_report_page0_expected.png"

    assert plot_path.is_file()
    assert template_path.is_file()
    assert expected_img_path.is_file()

    rows = [[float(i - 90), float(-0.04 * abs(i - 90))] for i in range(181)]
    payload = {
        "report_title": "Golden PDF Visual Regression",
        "pages": [
            {
                "page_title": "Diagrama GOLDEN_VRP (Elevacao)",
                "page_subtitle": "Projeto: GOLDEN | Antena: DEMO | Pol: POL1 | Freq: 100.000 MHz | Expr: E/Emax",
                "page_description": "Descricao de regressao visual da pagina tecnica.",
                "plot_image_path": str(plot_path),
                "metrics": {
                    "kind": "V",
                    "peak_db": 0.0,
                    "peak_angle_deg": 0.0,
                    "hpbw_deg": 18.5,
                    "d2d": 12.0,
                    "d2d_db": 10.79,
                    "first_null_db": -19.8,
                    "points": 181,
                    "angle_min": -90.0,
                    "angle_max": 90.0,
                    "step_deg": 1.0,
                },
                "table": {
                    "columns": ["Ang [deg]", "Valor [dB]"],
                    "rows": rows,
                    "note": "Tabela padrao: Elevacao -90..90 deg (passo 1 deg).",
                },
            }
        ],
    }

    output_pdf = tmp_path / "golden_report.pdf"
    result = export_report_pdf(
        payload=payload,
        output_pdf_path=str(output_pdf),
        template_pdf_path=str(template_path),
        save_full_csv=False,
        max_display_rows=80,
    )
    assert output_pdf.exists()
    assert int(result.get("pages", 0)) == 1

    actual = _render_pdf_page(output_pdf, page_index=0, scale=2.0)
    expected = np.asarray(Image.open(expected_img_path).convert("RGB"), dtype=np.uint8)
    assert actual.shape == expected.shape

    psnr = _psnr(actual, expected)
    if psnr < 50.0:
        diff = np.abs(actual.astype(np.int16) - expected.astype(np.int16)).astype(np.uint8)
        diff_path = tmp_path / "pdf_page_diff.png"
        Image.fromarray(diff).save(diff_path)
        raise AssertionError(f"PDF golden mismatch PSNR={psnr:.2f} dB | diff={diff_path}")

