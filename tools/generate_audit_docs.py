from __future__ import annotations

import argparse
import ast
import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = REPO_ROOT / "docs"


@dataclass(frozen=True)
class SymbolRow:
    file: str
    owner: str
    name: str
    kind: str
    line: int


def _iter_symbols(py_path: Path) -> Iterable[SymbolRow]:
    source = py_path.read_text(encoding="utf-8", errors="ignore").lstrip("\ufeff")
    mod = ast.parse(source)
    rel = str(py_path.relative_to(REPO_ROOT)).replace("\\", "/")
    for node in mod.body:
        if isinstance(node, ast.FunctionDef):
            yield SymbolRow(file=rel, owner="module", name=node.name, kind="function", line=int(node.lineno))
        elif isinstance(node, ast.ClassDef):
            yield SymbolRow(file=rel, owner=node.name, name=node.name, kind="class", line=int(node.lineno))
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    yield SymbolRow(
                        file=rel,
                        owner=node.name,
                        name=item.name,
                        kind="method",
                        line=int(item.lineno),
                    )


def _feature_area(owner: str, name: str, file: str) -> str:
    low = name.lower()
    f = file.lower()
    if "aedt_live" in f:
        return "AEDT Live"
    if "report" in low or "pdf" in low:
        return "Relatorio"
    if low.startswith("export_") or "export" in low:
        return "Exportacao"
    if low.startswith("load_") or "import" in low:
        return "Importacao"
    if low.startswith("project_") or "project" in low:
        return "Projeto"
    if "wizard" in low:
        return "Wizard"
    if "mechanical" in low or "mech" in f:
        return "Mecanica"
    if owner == "PATConverterApp":
        return "UI Principal"
    return "Core"


def _build_function_map(rows: List[SymbolRow]) -> str:
    stamp = dt.date.today().isoformat()
    out = [
        "# Function Map (auto)",
        "",
        f"Gerado automaticamente em: {stamp}",
        "",
        "| Arquivo | Owner | Simbolo | Tipo | Linha |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        out.append(f"| `{row.file}` | `{row.owner}` | `{row.name}` | {row.kind} | {row.line} |")
    out.append("")
    return "\n".join(out)


def _build_feature_inventory(rows: List[SymbolRow]) -> str:
    stamp = dt.date.today().isoformat()
    out = [
        "# Feature Inventory (auto)",
        "",
        f"Gerado automaticamente em: {stamp}",
        "",
        "| Feature ID | Area | Simbolo | Local |",
        "|---|---|---|---|",
    ]
    idx = 1
    for row in rows:
        if row.kind != "method":
            continue
        if row.owner not in {"PATConverterApp", "AedtLiveTab"} and "eftx_aedt_live" not in row.file:
            continue
        area = _feature_area(row.owner, row.name, row.file)
        fid = f"AUTO-{idx:03d}"
        out.append(f"| {fid} | {area} | `{row.owner}.{row.name}` | `{row.file}:{row.line}` |")
        idx += 1
    out.append("")
    return "\n".join(out)


def _write_or_check(path: Path, content: str, check: bool) -> int:
    if check:
        if not path.exists():
            return 1
        current = path.read_text(encoding="utf-8", errors="ignore")
        return 0 if current == content else 1
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate automatic audit docs.")
    parser.add_argument("--check", action="store_true", help="Do not write files; fail if out of date.")
    args = parser.parse_args(argv)

    targets = [
        REPO_ROOT / "deep3.py",
        REPO_ROOT / "eftx_aedt_live" / "session.py",
        REPO_ROOT / "eftx_aedt_live" / "ui_tab.py",
        REPO_ROOT / "eftx_aedt_live" / "farfield.py",
        REPO_ROOT / "reports" / "pdf_report.py",
        REPO_ROOT / "core" / "analysis" / "pattern_metrics.py",
    ]
    rows: List[SymbolRow] = []
    for target in targets:
        if target.exists():
            rows.extend(_iter_symbols(target))
    rows = sorted(rows, key=lambda r: (r.file, r.owner, r.line, r.name))

    function_text = _build_function_map(rows)
    feature_text = _build_feature_inventory(rows)
    code = 0
    code |= _write_or_check(DOCS_DIR / "function_map_auto.md", function_text, args.check)
    code |= _write_or_check(DOCS_DIR / "feature_inventory_auto.md", feature_text, args.check)
    return 1 if code else 0


if __name__ == "__main__":
    raise SystemExit(main())
