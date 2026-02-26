from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any, Dict

from ..diagnostics import collect_mechanical_diagnostics
from ..providers.freecad_provider import FreeCADKernelProvider


LOG = logging.getLogger("mech.freecad_headless_worker")


def _ok(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "ok": True,
        "payload": payload,
    }


def _err(message: str, *, code: str = "worker_error") -> Dict[str, Any]:
    return {
        "ok": False,
        "error": {
            "code": str(code),
            "message": str(message),
        },
    }


def _handle_request(req: Dict[str, Any]) -> Dict[str, Any]:
    cmd = str(req.get("command", "")).strip().lower()
    params = dict(req.get("params", {}) or {})

    if cmd == "doctor":
        return _ok(collect_mechanical_diagnostics(logger=LOG).to_dict())

    if cmd == "create_triangulate":
        kind = str(params.get("kind", "box") or "box")
        primitive_params = dict(params.get("primitive", {}) or {})
        quality = dict(params.get("quality", {}) or {})
        provider = FreeCADKernelProvider(logger=LOG)
        try:
            oid = provider.create_primitive(kind, primitive_params)
            mesh = provider.triangulate(oid, quality=quality)
            return _ok({"obj_id": oid, "mesh": mesh, "capabilities": provider.capabilities.to_dict()})
        finally:
            provider.close()

    if cmd == "import_export":
        src = str(params.get("input_path", "") or "")
        dst = str(params.get("output_path", "") or "")
        in_fmt = str(params.get("input_format", "") or "")
        out_fmt = str(params.get("output_format", "") or "")
        provider = FreeCADKernelProvider(logger=LOG)
        try:
            ids = provider.import_model(src, fmt=in_fmt)
            out = provider.export_model(ids, dst, fmt=out_fmt)
            return _ok({"imported_ids": list(ids), "output_path": out})
        finally:
            provider.close()

    return _err(f"Unsupported command: {cmd}", code="unsupported_command")


def run_worker(request: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return _handle_request(dict(request or {}))
    except Exception as exc:
        LOG.exception("Worker request failed: %s", exc)
        return _err(str(exc), code="worker_exception")


def _read_json_from_stdin() -> Dict[str, Any]:
    raw = sys.stdin.read()
    if not str(raw).strip():
        return {}
    return dict(json.loads(raw))


def _write_json_to_stdout(payload: Dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False))
    sys.stdout.write("\n")
    sys.stdout.flush()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Headless FreeCAD mechanical worker")
    parser.add_argument("--request-file", default="", help="Read JSON request from file instead of stdin")
    parser.add_argument("--response-file", default="", help="Write JSON response to file instead of stdout")
    args = parser.parse_args(argv)

    if args.request_file:
        with open(args.request_file, "r", encoding="utf-8", errors="ignore") as f:
            req = dict(json.load(f))
    else:
        req = _read_json_from_stdin()

    response = run_worker(req)

    if args.response_file:
        with open(args.response_file, "w", encoding="utf-8", newline="\n") as f:
            json.dump(response, f, ensure_ascii=False, indent=2)
    else:
        _write_json_to_stdout(response)
    return 0 if bool(response.get("ok", False)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
