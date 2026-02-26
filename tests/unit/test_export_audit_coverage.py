from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict


ROOT = Path(__file__).resolve().parents[2]
DEEP3_PATH = ROOT / "deep3.py"


def _pat_converter_methods_source() -> Dict[str, str]:
    source = DEEP3_PATH.read_text(encoding="utf-8", errors="ignore").lstrip("\ufeff")
    module = ast.parse(source)
    cls = None
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == "PATConverterApp":
            cls = node
            break
    if cls is None:
        raise AssertionError("Class PATConverterApp not found in deep3.py")

    lines = source.splitlines()
    methods: Dict[str, str] = {}
    for node in cls.body:
        if isinstance(node, ast.FunctionDef):
            start = max(int(getattr(node, "lineno", 1)) - 1, 0)
            end = int(getattr(node, "end_lineno", node.lineno))
            methods[node.name] = "\n".join(lines[start:end])
    return methods


EXPORT_ROUTES_REQUIRING_AUDIT_START = [
    "export_study_complete",
    "export_recorded_files_bundle",
    "generate_all_project_artifacts",
    "export_project_graph_images",
    "export_project_table_images",
    "export_project_pat_files",
    "export_project_adt_files",
    "export_project_prn_files",
    "export_single_pat",
    "export_plot_img",
    "export_table_img",
    "export_all_pat",
    "export_all_prn",
    "export_vertical_harness",
    "export_vertical_array_pat",
    "export_vertical_array_prn",
    "export_horizontal_array_pat",
    "export_horizontal_array_rfs",
    "export_horizontal_array_prn",
    "_run_report_export_async",
    "_run_export_wizard_async",
]


EXPORT_ROUTES_REQUIRING_AUDIT_END = [
    "_finalize_report_export",
    "_finalize_export_wizard",
]


def test_export_routes_have_audit_start_call():
    methods = _pat_converter_methods_source()
    missing = [name for name in EXPORT_ROUTES_REQUIRING_AUDIT_START if "_audit_export_start(" not in methods.get(name, "")]
    assert not missing, f"Export routes without _audit_export_start: {missing}"


def test_async_finalize_routes_have_audit_end_call():
    methods = _pat_converter_methods_source()
    missing = [name for name in EXPORT_ROUTES_REQUIRING_AUDIT_END if "_audit_export_end(" not in methods.get(name, "")]
    assert not missing, f"Finalize routes without _audit_export_end: {missing}"
