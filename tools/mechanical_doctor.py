from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from mech.mechanical import collect_mechanical_diagnostics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Mechanical backend doctor check")
    parser.add_argument("--json", default="", help="Optional path to save JSON report")
    parser.add_argument("--strict", action="store_true", help="Fail when FreeCAD in-process is unavailable")
    args = parser.parse_args(argv)

    report = collect_mechanical_diagnostics().to_dict()
    summary = str(report.get("summary", ""))
    print(f"[MECH-DOCTOR] {summary}")

    if args.json:
        out = os.path.abspath(args.json)
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w", encoding="utf-8", newline="\n") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"[MECH-DOCTOR] report saved: {out}")

    if args.strict and not bool(report.get("freecad", {}).get("inprocess_available", False)):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
