import csv
import os

from pypdf import PdfReader
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from PIL import Image

from reports.pdf_report import export_report_pdf


def _make_template_pdf(path: str):
    c = canvas.Canvas(path, pagesize=A4)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, 810, "EFTX TEMPLATE HEADER")
    c.setFont("Helvetica", 8)
    c.drawString(50, 25, "EFTX TEMPLATE FOOTER")
    c.showPage()
    c.save()


def _make_plot(path: str):
    img = Image.new("RGB", (1800, 1000), color=(255, 255, 255))
    img.save(path, format="PNG")


def _page_payload(idx: int, plot_path: str, rows_count: int = 181):
    rows = [[float(i - 90), float(-0.05 * abs(i - 90))] for i in range(rows_count)]
    return {
        "page_title": f"Diagrama CUT_{idx:02d} (Elevacao)",
        "page_subtitle": "Projeto: TESTE | Antena: DEMO | Pol: POL1 | Freq: 100.000 MHz | Expr: E/Emax",
        "page_description": "Descricao tecnica do corte para validacao automatica.",
        "plot_image_path": plot_path,
        "metrics": {
            "kind": "V",
            "peak_db": 0.0,
            "peak_angle_deg": 0.0,
            "hpbw_deg": 18.5,
            "d2d": 12.0,
            "d2d_db": 10.79,
            "first_null_db": -19.8,
            "points": rows_count,
            "angle_min": -90.0,
            "angle_max": 90.0,
            "step_deg": 1.0,
        },
        "table": {
            "columns": ["Ang [deg]", "Valor [dB]"],
            "rows": rows,
            "note": "Teste automatico",
        },
    }


def _pdf_text(path: str) -> str:
    return "\n".join((p.extract_text() or "") for p in PdfReader(path).pages)


def test_pdf_report_single_page(tmp_path):
    template = tmp_path / "template.pdf"
    plot = tmp_path / "plot.png"
    out_pdf = tmp_path / "relatorio.pdf"
    _make_template_pdf(str(template))
    _make_plot(str(plot))

    payload = {"report_title": "Relatorio 1 pagina", "pages": [_page_payload(1, str(plot))]}
    res = export_report_pdf(
        payload=payload,
        output_pdf_path=str(out_pdf),
        template_pdf_path=str(template),
        save_full_csv=True,
    )

    assert out_pdf.exists()
    assert res["pages"] == 1
    assert len(res["csv_files"]) == 1
    assert len(PdfReader(str(out_pdf)).pages) == 1


def test_pdf_report_eight_pages(tmp_path):
    template = tmp_path / "template.pdf"
    plot = tmp_path / "plot.png"
    out_pdf = tmp_path / "relatorio_lote.pdf"
    _make_template_pdf(str(template))
    _make_plot(str(plot))

    pages = [_page_payload(i + 1, str(plot)) for i in range(8)]
    payload = {"report_title": "Relatorio lote", "pages": pages}
    res = export_report_pdf(
        payload=payload,
        output_pdf_path=str(out_pdf),
        template_pdf_path=str(template),
        save_full_csv=False,
    )

    assert out_pdf.exists()
    assert res["pages"] == 8
    reader = PdfReader(str(out_pdf))
    assert len(reader.pages) == 8
    # Regression: after template merge, each page must preserve its own cut title.
    for i, page in enumerate(reader.pages, start=1):
        txt = page.extract_text() or ""
        assert f"CUT_{i:02d}" in txt


def test_pdf_report_large_table_compaction_and_full_csv(tmp_path):
    template = tmp_path / "template.pdf"
    plot = tmp_path / "plot.png"
    out_pdf = tmp_path / "relatorio_grande.pdf"
    _make_template_pdf(str(template))
    _make_plot(str(plot))

    rows_count = 1801
    payload = {"report_title": "Relatorio tabela grande", "pages": [_page_payload(1, str(plot), rows_count=rows_count)]}
    res = export_report_pdf(
        payload=payload,
        output_pdf_path=str(out_pdf),
        template_pdf_path=str(template),
        save_full_csv=True,
        max_display_rows=80,
    )

    assert out_pdf.exists()
    assert res["page_results"][0]["compacted"] is True
    assert len(res["csv_files"]) == 1
    csv_path = res["csv_files"][0]
    assert os.path.exists(csv_path)

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        rd = csv.reader(f)
        lines = list(rd)
    assert len(lines) == rows_count + 1


def test_pdf_report_no_compaction_note_for_small_table(tmp_path):
    template = tmp_path / "template.pdf"
    plot = tmp_path / "plot.png"
    out_pdf = tmp_path / "relatorio_small.pdf"
    _make_template_pdf(str(template))
    _make_plot(str(plot))

    payload = {"report_title": "Relatorio pequeno", "pages": [_page_payload(1, str(plot), rows_count=10)]}
    res = export_report_pdf(
        payload=payload,
        output_pdf_path=str(out_pdf),
        template_pdf_path=str(template),
        save_full_csv=False,
    )

    assert out_pdf.exists()
    assert res["page_results"][0]["compacted"] is False
    txt = _pdf_text(str(out_pdf)).lower()
    assert "compactada" not in txt
    assert "compacted" not in txt


def test_pdf_report_with_glossary_page(tmp_path):
    template = tmp_path / "template.pdf"
    plot = tmp_path / "plot.png"
    out_pdf = tmp_path / "relatorio_glossario.pdf"
    _make_template_pdf(str(template))
    _make_plot(str(plot))

    payload = {
        "report_title": "Relatorio glossario",
        "pages": [_page_payload(1, str(plot), rows_count=50)],
        "glossary": [
            {"term": "HRP", "definition": "Corte horizontal de azimute."},
            {"term": "VRP", "definition": "Corte vertical de elevacao."},
        ],
    }
    res = export_report_pdf(
        payload=payload,
        output_pdf_path=str(out_pdf),
        template_pdf_path=str(template),
        save_full_csv=False,
    )

    assert out_pdf.exists()
    assert res["pages"] == 1
    assert res["glossary_pages"] == 1
    assert res["output_pages"] == 2
    reader = PdfReader(str(out_pdf))
    assert len(reader.pages) == 2
    txt = (reader.pages[1].extract_text() or "").lower()
    assert "glossario" in txt
    assert "hrp" in txt
