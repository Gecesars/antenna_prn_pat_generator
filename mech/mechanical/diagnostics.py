from __future__ import annotations

import importlib.util
import importlib
import logging
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .interfaces import MechanicalKernel
from .models import MechanicalDiagnostics
from .providers.null_provider import NullMechanicalProvider


def _which_any(*names: str) -> str:
    for name in names:
        path = shutil.which(name)
        if path:
            return str(path)
    return ""


def _common_freecadcmd_paths() -> Tuple[str, ...]:
    roots = [
        os.environ.get("ProgramFiles", ""),
        os.environ.get("ProgramFiles(x86)", ""),
        os.environ.get("LOCALAPPDATA", ""),
    ]
    candidates = []
    for root in [r for r in roots if r]:
        base = Path(root)
        for pattern in ("FreeCAD*", "FreeCAD *"):
            for folder in base.glob(pattern):
                cmd = folder / "bin" / "FreeCADCmd.exe"
                if cmd.is_file():
                    candidates.append(str(cmd))
    return tuple(dict.fromkeys(candidates))


def _detect_freecadcmd() -> str:
    from_path = _which_any("FreeCADCmd", "FreeCADCmd.exe")
    if from_path:
        return from_path
    for candidate in _common_freecadcmd_paths():
        if Path(candidate).is_file():
            return str(candidate)
    return ""


def _find_spec(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def _probe_module_import(name: str) -> Tuple[bool, str]:
    if not _find_spec(name):
        return False, "module_not_found"
    try:
        importlib.import_module(name)
        return True, ""
    except Exception as exc:
        return False, f"{exc.__class__.__name__}: {exc}"


def _detect_freecadcmd_python(freecad_cmd: str) -> str:
    cmd = str(freecad_cmd or "").strip()
    if not cmd:
        return ""
    try:
        proc = subprocess.run(
            [cmd, "-c", "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')"],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
    except Exception:
        return ""
    if proc.returncode != 0:
        return ""
    return str(proc.stdout or "").strip().splitlines()[0].strip() if str(proc.stdout or "").strip() else ""


def collect_mechanical_diagnostics(logger: Optional[logging.Logger] = None) -> MechanicalDiagnostics:
    log = logger or logging.getLogger(__name__)

    freecad_ok, freecad_err = _probe_module_import("FreeCAD")
    part_ok, part_err = _probe_module_import("Part")
    fem_ok, fem_err = _probe_module_import("Fem")
    gmsh_bin = _which_any("gmsh", "gmsh.exe")
    ccx_bin = _which_any("ccx", "ccx.exe", "calculix", "calculix.exe")
    freecad_cmd = _detect_freecadcmd()
    host_py_minor = f"{sys.version_info[0]}.{sys.version_info[1]}"
    freecad_cmd_py_minor = _detect_freecadcmd_python(freecad_cmd)
    py_minor_abi_match = bool(
        freecad_cmd_py_minor
        and host_py_minor
        and (str(freecad_cmd_py_minor) == str(host_py_minor))
    )

    import_errors: Dict[str, str] = {}
    if freecad_err:
        import_errors["FreeCAD"] = str(freecad_err)
    if part_err:
        import_errors["Part"] = str(part_err)
    if fem_err:
        import_errors["Fem"] = str(fem_err)

    report: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": {
            "os_name": os.name,
            "system": platform.system(),
            "release": platform.release(),
            "python": platform.python_version(),
        },
        "freecad": {
            "import_freecad": bool(freecad_ok),
            "import_part": bool(part_ok),
            "inprocess_available": bool(freecad_ok and part_ok),
            "freecad_cmd": str(freecad_cmd),
            "headless_available": bool(freecad_cmd),
            "host_python_minor": str(host_py_minor),
            "freecad_cmd_python_minor": str(freecad_cmd_py_minor),
            "python_abi_match": bool(py_minor_abi_match),
            "import_errors": dict(import_errors),
        },
        "fem": {
            "python_fem_module": bool(fem_ok),
            "gmsh_bin": str(gmsh_bin),
            "calculix_bin": str(ccx_bin),
            "fem_available": bool(fem_ok and gmsh_bin and ccx_bin),
        },
        "viewport": {
            "opengl_check": "not_run",
            "note": "Runtime OpenGL check is deferred to UI session.",
        },
    }

    status_bits = []
    if report["freecad"]["inprocess_available"]:
        status_bits.append("freecad:inprocess")
    if report["freecad"]["headless_available"]:
        status_bits.append("freecad:headless")
    if report["freecad"]["headless_available"] and not report["freecad"]["python_abi_match"]:
        status_bits.append("freecad:abi-mismatch")
    if report["fem"]["fem_available"]:
        status_bits.append("fem:available")
    report["summary"] = ", ".join(status_bits) if status_bits else "freecad unavailable (fallback mode)"

    try:
        log.info("Mechanical doctor summary: %s", report["summary"])
    except Exception:
        pass

    return MechanicalDiagnostics(report=report)


def build_default_kernel(logger: Optional[logging.Logger] = None) -> Tuple[MechanicalKernel, Dict[str, Any]]:
    """Build mechanical kernel provider with safe fallback.

    Returns `(provider, diagnostics_report)` and never raises.
    """
    log = logger or logging.getLogger(__name__)
    diag = collect_mechanical_diagnostics(logger=log).to_dict()

    if bool(diag.get("freecad", {}).get("inprocess_available", False)):
        try:
            from .providers.freecad_provider import FreeCADKernelProvider

            provider = FreeCADKernelProvider(logger=log)
            return provider, diag
        except Exception as exc:
            try:
                log.exception("Failed to initialize FreeCAD provider; using null provider: %s", exc)
            except Exception:
                pass

    reason = str(diag.get("summary", "FreeCAD unavailable"))
    return NullMechanicalProvider(reason=reason), diag
