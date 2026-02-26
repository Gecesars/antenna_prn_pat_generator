from __future__ import annotations

import ctypes
import importlib
import importlib.util
from dataclasses import dataclass
import sys
from pathlib import Path
from typing import Optional
import os
import re
import platform
import traceback

import threading
import time

from core.audit import audit_span, emit_audit
from .psutil_shim import create_psutil_shim_module


_DLL_DIRS_ADDED: set[str] = set()
_PATH_DIRS_ADDED: set[str] = set()
_PSUTIL_BACKEND: str = "uninitialized"


def _sanitize_sys_path_for_pyvista() -> None:
    """Drop shadowing paths that can hijack ``import pyvista``."""
    bad_suffixes = (
        os.path.normcase(os.path.normpath(r"ansys\tools\visualization_interface\backends")),
        os.path.normcase(os.path.normpath(r"ansys\tools\visualization_interface\backends\pyvista")),
    )
    cleaned: list[str] = []
    changed = False
    for entry in list(sys.path):
        try:
            norm = os.path.normcase(os.path.normpath(str(entry or "")))
        except Exception:
            norm = ""
        if any(norm.endswith(sfx) for sfx in bad_suffixes):
            changed = True
            continue
        cleaned.append(entry)
    if changed:
        sys.path[:] = cleaned


def _is_bad_pyvista_module(mod) -> bool:
    """Detect broken or shadowed ``pyvista`` imports."""
    if mod is None:
        return True
    try:
        mod_file = str(getattr(mod, "__file__", "") or "").replace("/", "\\").lower()
    except Exception:
        mod_file = ""
    if "ansys\\tools\\visualization_interface\\backends\\pyvista\\pyvista.py" in mod_file:
        return True
    if not hasattr(mod, "_plot"):
        return True
    return False


def _reset_pyvista_modules() -> None:
    """Remove pyvista-related modules from import cache."""
    for key in list(sys.modules.keys()):
        k = str(key or "")
        if k == "pyvista" or k.startswith("pyvista."):
            sys.modules.pop(k, None)


def _ensure_pyvista_sane() -> None:
    """Ensure pyvista points to the real package, not a shadowed module."""
    _sanitize_sys_path_for_pyvista()
    try:
        import pyvista as pv  # type: ignore
        if _is_bad_pyvista_module(pv):
            raise RuntimeError("bad_pyvista_state")
    except Exception:
        _reset_pyvista_modules()
        _sanitize_sys_path_for_pyvista()
        try:
            import pyvista as pv  # type: ignore # noqa: F401
            if _is_bad_pyvista_module(pv):
                _reset_pyvista_modules()
        except Exception:
            _reset_pyvista_modules()


def _version_to_token(version_hint: Optional[str]) -> Optional[str]:
    if version_hint is None:
        return None
    txt = str(version_hint).strip()
    if not txt:
        return None
    txt = txt.replace(",", ".")
    if txt.isdigit() and len(txt) == 3:
        return txt
    m = re.match(r"^(\d{4})\.(\d+)$", txt)
    if m:
        yy = int(m.group(1)) % 100
        rel = int(m.group(2))
        return f"{yy}{rel}"
    return None


def _discover_aedt_roots(version_hint: Optional[str]) -> list[Path]:
    roots: list[Path] = []
    seen: set[str] = set()

    def _add(p: Path):
        try:
            rp = p.resolve()
        except Exception:
            rp = p
        key = str(rp).lower()
        if key in seen:
            return
        if p.exists() and p.is_dir():
            seen.add(key)
            roots.append(rp)

    token = _version_to_token(version_hint)
    env_keys = []
    if token:
        env_keys.append(f"ANSYSEM_ROOT{token}")
    env_keys.extend([k for k in os.environ.keys() if str(k).upper().startswith("ANSYSEM_ROOT")])
    for k in env_keys:
        try:
            v = os.environ.get(k, "")
            if v:
                _add(Path(v))
        except Exception:
            continue

    program_files = Path(r"C:\Program Files\ANSYS Inc")
    if token:
        _add(program_files / f"v{token}" / "AnsysEM")
    try:
        for p in sorted(program_files.glob("v*/AnsysEM"), reverse=True):
            _add(p)
    except Exception:
        pass
    return roots


def _collect_runtime_dirs(version_hint: Optional[str]) -> list[Path]:
    dirs: list[Path] = []
    seen: set[str] = set()

    def _push(p: Path):
        try:
            rp = p.resolve()
        except Exception:
            rp = p
        key = str(rp).lower()
        if key in seen:
            return
        if p.exists() and p.is_dir():
            seen.add(key)
            dirs.append(rp)

    for root in _discover_aedt_roots(version_hint):
        _push(root)
        for name in ("common", "commonfiles", "Delcross", "gRPCFiles", "platforms", "PythonFiles", "syslib", "tp"):
            _push(root / name)

        # Sibling commonfiles tree (outside AnsysEM) contains CPython runtimes.
        ver_root = root.parent
        commonfiles = ver_root / "commonfiles"
        _push(commonfiles)
        _push(commonfiles / "CPython")
        _push(commonfiles / "IronPython")
        try:
            for py_root in commonfiles.glob("CPython/*/winx64/Release/python"):
                _push(py_root)
                _push(py_root / "DLLs")
        except Exception:
            pass
    return dirs


def _prepend_path_dirs(dirs: list[Path]) -> None:
    if os.name != "nt":
        return
    cur = os.environ.get("PATH", "")
    cur_parts = cur.split(os.pathsep) if cur else []
    cur_keys = {os.path.normcase(os.path.normpath(p)) for p in cur_parts if p}
    add_parts: list[str] = []
    for d in dirs:
        ds = str(d)
        k = os.path.normcase(os.path.normpath(ds))
        if (not ds) or (k in cur_keys) or (k in _PATH_DIRS_ADDED):
            continue
        add_parts.append(ds)
        _PATH_DIRS_ADDED.add(k)
    if add_parts:
        os.environ["PATH"] = os.pathsep.join(add_parts + cur_parts)


def _prepare_windows_dll_resolution(version_hint: Optional[str] = None) -> None:
    """Improve DLL resolution for frozen Windows builds.

    In some installations, extension modules (for example ``psutil._psutil_windows``)
    may fail to load if runtime DLL lookup paths are incomplete.
    """
    if os.name != "nt":
        return

    dirs: list[Path] = []
    try:
        exe_dir = Path(sys.executable).resolve().parent
        dirs.extend([exe_dir, exe_dir / "lib", exe_dir / "lib" / "psutil"])
    except Exception:
        pass
    try:
        pkg_root = Path(__file__).resolve().parents[1]
        dirs.extend([pkg_root, pkg_root / "lib", pkg_root / "lib" / "psutil"])
    except Exception:
        pass
    dirs.extend(_collect_runtime_dirs(version_hint))

    _prepend_path_dirs(dirs)
    for d in dirs:
        try:
            ds = str(d)
        except Exception:
            continue
        key = os.path.normcase(os.path.normpath(ds))
        if (not ds) or (key in _DLL_DIRS_ADDED) or (not os.path.isdir(ds)):
            continue
        _DLL_DIRS_ADDED.add(key)
        try:
            os.add_dll_directory(ds)
        except Exception:
            pass

    # Opportunistically preload common runtime DLLs when available.
    preload_names = (
        "python3.dll",
        "python312.dll",
        "vcruntime140.dll",
        "vcruntime140_1.dll",
        "msvcp140.dll",
        "concrt140.dll",
    )
    for name in preload_names:
        for d in dirs:
            candidate = d / name
            if not candidate.exists():
                continue
            try:
                ctypes.WinDLL(str(candidate))
                break
            except Exception:
                continue


def _patch_pyaedt_process_discovery_no_powershell() -> None:
    """Disable PowerShell-based process discovery in PyAEDT on Windows.

    Some PyAEDT builds use PowerShell/WMIC for process discovery, which can
    create visible console windows in frozen GUI apps. This patch replaces that
    path with a psutil-based implementation.
    """
    if os.name != "nt":
        return
    try:
        import psutil as _psutil  # type: ignore
    except Exception:
        _psutil = None
    if _psutil is None:
        return

    try:
        gm = importlib.import_module("ansys.aedt.core.generic.general_methods")
    except Exception:
        return
    if getattr(gm, "_eftx_no_ps_discovery_patch", False):
        return

    def _safe_get_target_processes(target_name: list[str]) -> list[tuple[int, list[str]]]:
        if platform.system() != "Windows":
            return []
        targets = {str(t or "").strip().lower() for t in (target_name or []) if str(t or "").strip()}
        if not targets:
            return []
        found: list[tuple[int, list[str]]] = []
        try:
            for proc in _psutil.process_iter(attrs=["pid", "name", "cmdline"]):
                info = getattr(proc, "info", {}) or {}
                name = str(info.get("name") or "").strip().lower()
                if name not in targets:
                    continue
                try:
                    pid = int(info.get("pid") or getattr(proc, "pid", 0))
                except Exception:
                    continue
                cmdline_raw = info.get("cmdline")
                if isinstance(cmdline_raw, str):
                    cmdline = [x for x in cmdline_raw.split() if x]
                elif isinstance(cmdline_raw, (list, tuple)):
                    cmdline = [str(x) for x in cmdline_raw if str(x)]
                else:
                    cmdline = []
                found.append((pid, cmdline))
        except Exception:
            return []
        return found

    def _parse_grpc_port(cmdline: list[str]) -> int:
        cmd = [str(x) for x in (cmdline or []) if str(x)]
        if "-grpcsrv" not in cmd:
            return -1
        try:
            raw = str(cmd[cmd.index("-grpcsrv") + 1]).strip()
        except Exception:
            return -1
        if not raw:
            return -1
        if raw.isdigit():
            try:
                return int(raw)
            except Exception:
                return -1
        # Formats seen in AEDT: "host:port:mode" or "port".
        for part in reversed(raw.split(":")):
            part = str(part).strip()
            if part.isdigit():
                try:
                    return int(part)
                except Exception:
                    return -1
        return -1

    def _normalize_version_token(version: Optional[str]) -> Optional[str]:
        if version is None:
            return None
        txt = str(version).strip()
        if not txt:
            return None
        if "." in txt:
            # Keep compatibility with PyAEDT token matching logic.
            txt = txt[-4:].replace(".", "")
        if txt.isdigit() and len(txt) == 3:
            return txt
        return None

    def _safe_active_sessions(
        version: Optional[str] = None,
        student_version: bool = False,
        non_graphical: Optional[bool] = None,
    ) -> dict[int, int]:
        if platform.system() != "Windows":
            return {}
        target_names = {"ansysedtsv.exe"} if bool(student_version) else {"ansysedt.exe"}
        target_names_noext = {x.replace(".exe", "") for x in target_names}
        version_tok = _normalize_version_token(version)
        out: dict[int, int] = {}
        try:
            for proc in _psutil.process_iter(attrs=["pid", "name", "cmdline"]):
                info = getattr(proc, "info", {}) or {}
                name = str(info.get("name") or "").strip().lower()
                if name not in target_names and name not in target_names_noext:
                    continue
                cmdline_raw = info.get("cmdline")
                if isinstance(cmdline_raw, str):
                    cmdline = [x for x in cmdline_raw.split() if x]
                elif isinstance(cmdline_raw, (list, tuple)):
                    cmdline = [str(x) for x in cmdline_raw if str(x)]
                else:
                    cmdline = []
                cmd_join = " ".join(cmdline).lower()
                if version_tok and version_tok not in cmd_join:
                    # Keep behavior permissive: if token is missing, include anyway.
                    pass
                is_ng = "-ng" in [x.lower() for x in cmdline]
                if non_graphical is True and (not is_ng):
                    continue
                if non_graphical is False and is_ng:
                    continue
                try:
                    pid = int(info.get("pid") or getattr(proc, "pid", 0))
                except Exception:
                    continue
                parsed_port = _parse_grpc_port(cmdline)

                listen_ports: set[int] = set()
                try:
                    for c in proc.net_connections(kind="tcp"):
                        if str(getattr(c, "status", "")).upper() != "LISTEN":
                            continue
                        laddr = getattr(c, "laddr", None)
                        pval = int(getattr(laddr, "port", 0) or 0)
                        if pval > 0:
                            listen_ports.add(pval)
                except Exception:
                    listen_ports = set()

                # Ignore stale cmdline ports not currently bound by the process.
                if parsed_port > 0 and parsed_port not in listen_ports:
                    parsed_port = -1

                # Fallback: infer a single non-reserved listener when cmdline is stale.
                if parsed_port <= 0 and listen_ports:
                    candidates = sorted([p for p in listen_ports if p not in (2001, 2002)])
                    if len(candidates) == 1:
                        parsed_port = int(candidates[0])

                out[pid] = parsed_port
        except Exception:
            return {}
        return out

    def _safe_grpc_active_sessions(
        version: Optional[str] = None,
        student_version: bool = False,
        non_graphical: Optional[bool] = None,
    ) -> list[int]:
        data = _safe_active_sessions(version=version, student_version=student_version, non_graphical=non_graphical)
        return sorted({int(p) for p in data.values() if int(p) > -1})

    def _safe_com_active_sessions(
        version: Optional[str] = None,
        student_version: bool = False,
        non_graphical: Optional[bool] = None,
    ) -> list[int]:
        data = _safe_active_sessions(version=version, student_version=student_version, non_graphical=non_graphical)
        return [int(pid) for pid, port in data.items() if int(port) <= -1]

    def _safe_is_grpc_session_active(port: int) -> bool:
        try:
            port_i = int(port)
        except Exception:
            return False
        if port_i <= 0:
            return False
        # Check across both graphical and non-graphical AEDT sessions.
        return port_i in _safe_grpc_active_sessions(non_graphical=None)

    try:
        gm._get_target_processes = _safe_get_target_processes
        gm.active_sessions = _safe_active_sessions
        gm.grpc_active_sessions = _safe_grpc_active_sessions
        gm.com_active_sessions = _safe_com_active_sessions
        gm.is_grpc_session_active = _safe_is_grpc_session_active
        setattr(gm, "_eftx_no_ps_discovery_patch", True)
    except Exception:
        return

    # In some versions, desktop.py imports this symbol directly.
    try:
        desktop_mod = importlib.import_module("ansys.aedt.core.desktop")
        desktop_mod._get_target_processes = _safe_get_target_processes
        desktop_mod.active_sessions = _safe_active_sessions
        desktop_mod.grpc_active_sessions = _safe_grpc_active_sessions
        desktop_mod.com_active_sessions = _safe_com_active_sessions
        desktop_mod.is_grpc_session_active = _safe_is_grpc_session_active
    except Exception:
        pass


def _import_hfss(version_hint: Optional[str] = None):
    global _PSUTIL_BACKEND
    # Prefer the modern import path used by PyAEDT docs.
    _prepare_windows_dll_resolution(version_hint)
    _ensure_pyvista_sane()
    # Some installed environments fail loading psutil native extension
    # (_psutil_windows). Provide a lightweight fallback shim so AEDT APIs
    # can still be imported and used.
    try:
        import psutil  # noqa: F401
        # In some frozen builds, `import psutil` succeeds but native extension
        # loading fails later. Force a direct probe here.
        try:
            import psutil._psutil_windows  # type: ignore # noqa: F401
            _PSUTIL_BACKEND = "native"
        except Exception:
            try:
                sys.modules.pop("psutil", None)
                sys.modules["psutil"] = create_psutil_shim_module()
                _PSUTIL_BACKEND = "shim(native_probe_failed)"
            except Exception:
                _PSUTIL_BACKEND = "native_probe_failed(no_shim)"
    except Exception:
        try:
            sys.modules.pop("psutil", None)
            sys.modules["psutil"] = create_psutil_shim_module()
            _PSUTIL_BACKEND = "shim(import_failed)"
        except Exception:
            _PSUTIL_BACKEND = "import_failed(no_shim)"
    _patch_pyaedt_process_discovery_no_powershell()
    modern_errors: list[str] = []

    def _fmt_exc(exc: Exception) -> str:
        msg = f"{type(exc).__name__}: {exc}"
        fn = getattr(exc, "filename", None)
        if fn:
            msg += f" | file={fn}"
        wn = getattr(exc, "winerror", None)
        if wn is not None:
            msg += f" | winerror={wn}"
        return msg

    def _try_modern_imports() -> Optional[type]:
        # Prefer direct module import first. Importing from package root can
        # pull optional visualization stack eagerly in frozen builds.
        attempts = (
            ("ansys.aedt.core.hfss", lambda: importlib.import_module("ansys.aedt.core.hfss").Hfss),
            ("ansys.aedt.core", lambda: importlib.import_module("ansys.aedt.core").Hfss),
        )
        for label, loader in attempts:
            try:
                cls = loader()
                return cls
            except Exception as e_try:
                modern_errors.append(f"{label}: {_fmt_exc(e_try)}")
        return None

    hfss_cls = _try_modern_imports()
    if hfss_cls is not None:
        return hfss_cls

    # Retry once after pyvista sanitation when modern import hints at graphics shadowing.
    combined = " | ".join(modern_errors).lower()
    if ("pyvista" in combined) or ("_plot" in combined):
        _ensure_pyvista_sane()
        hfss_cls = _try_modern_imports()
        if hfss_cls is not None:
            return hfss_cls

    # Legacy fallback: only try when the compatibility package is present.
    # This prevents masking the real modern import error with
    # "No module named 'pyaedt'" when pyaedt is intentionally absent.
    try:
        if importlib.util.find_spec("pyaedt") is None:
            raise RuntimeError(
                "Legacy fallback package 'pyaedt' is not available."
            )
        from pyaedt import Hfss  # type: ignore
        return Hfss
    except Exception as e_legacy:
        detail = " | ".join(modern_errors[-6:]) if modern_errors else "no-modern-details"
        tb = traceback.format_exc(limit=6).strip().replace("\n", " || ")
        raise RuntimeError(
            "Unable to import HFSS API. "
            f"Modern import attempts: {detail} | "
            f"Legacy fallback failed: {_fmt_exc(e_legacy)} | "
            f"trace={tb}"
        ) from e_legacy


def _force_pyaedt_single_desktop_mode() -> None:
    """Avoid multi-desktop branch that is unstable in some PyAEDT builds."""
    try:
        from ansys.aedt.core.generic.settings import settings as aedt_settings  # type: ignore
        aedt_settings.use_multi_desktop = False
    except Exception:
        try:
            from pyaedt.generic.settings import settings as aedt_settings  # type: ignore
            aedt_settings.use_multi_desktop = False
        except Exception:
            pass


def _patch_desktop_compat(hfss_obj) -> None:
    """Patch Desktop attribute aliases for mixed PyAEDT versions.

    Some builds reference `grpc_plugin` while others use `grpc_plug_in`.
    """
    def _patch_one(desk_obj) -> None:
        if desk_obj is None:
            return
        gp = getattr(desk_obj, "grpc_plugin", None)
        gpi = getattr(desk_obj, "grpc_plug_in", None)
        if gp is None and gpi is not None:
            try:
                setattr(desk_obj, "grpc_plugin", gpi)
                gp = gpi
            except Exception:
                pass
        if gpi is None and gp is not None:
            try:
                setattr(desk_obj, "grpc_plug_in", gp)
            except Exception:
                pass
        if not hasattr(desk_obj, "_Desktop__aedt_version_id"):
            try:
                aid = getattr(desk_obj, "aedt_version_id", None)
                if aid is not None:
                    setattr(desk_obj, "_Desktop__aedt_version_id", aid)
            except Exception:
                pass

    try:
        # Newer PyAEDT typically exposes desktop_class.
        _patch_one(getattr(hfss_obj, "desktop_class", None))
        # Defensive aliases for other shapes.
        _patch_one(getattr(hfss_obj, "_desktop", None))
        _patch_one(getattr(hfss_obj, "desktop", None))
    except Exception:
        pass


def _patch_pyaedt_solution_constants() -> None:
    """Work around PyAEDT builds that expect `.default_solution` on design constants.

    Some versions expose only `solution_default` (e.g. `HfssConstants`), but internal
    paths still access `default_solution`, causing runtime failures during report pulls.
    """
    for mod_name in ("ansys.aedt.core.generic.aedt_constants", "pyaedt.generic.aedt_constants"):
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            continue
        try:
            for name in dir(mod):
                obj = getattr(mod, name, None)
                if not isinstance(obj, type):
                    continue
                if hasattr(obj, "solution_default") and (not hasattr(obj, "default_solution")):
                    try:
                        setattr(obj, "default_solution", getattr(obj, "solution_default"))
                    except Exception:
                        pass
        except Exception:
            continue


@dataclass
class AedtConnectionConfig:
    """Connection parameters for AEDT/HFSS via PyAEDT.

    Notes:
      - `version` accepts values like "2025.2" or 252 (per PyAEDT docs).
      - If you want to attach to an already running AEDT session, set new_desktop=False
        and (optionally) provide aedt_process_id.
    """
    version: str = "2025.2"
    non_graphical: bool = False
    new_desktop: bool = False
    close_on_exit: bool = False
    student_version: bool = False
    machine: str = ""
    port: int = 0
    aedt_process_id: Optional[int] = None
    remove_lock: bool = False


class AedtHfssSession:
    """Manages a single HFSS session through PyAEDT.

    Design goals:
      - Avoid side effects on import.
      - Keep one HFSS handle alive and reuse it for post-processing.
      - Provide deterministic disconnect semantics.

    Typical usage:
      session = AedtHfssSession(AedtConnectionConfig(version="2025.2"))
      session.connect(project=r"C:\\...\\proj.aedt", design="HFSSDesign1")
      ... use session.hfss ...
      session.disconnect()
    """

    def __init__(self, cfg: AedtConnectionConfig):
        self.cfg = cfg
        self._hfss = None
        self._lock = threading.RLock()
        self._connected_at: Optional[float] = None
        self._last_project: Optional[str] = None
        self._last_design: Optional[str] = None
        self._ever_connected = False

    @staticmethod
    def _norm_project(value: Optional[str]) -> Optional[str]:
        if not value:
            return None
        txt = str(value).strip()
        if not txt:
            return None
        p = Path(txt)
        if p.suffix.lower() == ".aedt":
            try:
                return str(p.resolve())
            except Exception:
                return str(p)
        return txt

    @staticmethod
    def _project_lock_file(project: Optional[str]) -> Optional[Path]:
        ptxt = AedtHfssSession._norm_project(project)
        if not ptxt:
            return None
        p = Path(ptxt)
        if p.suffix.lower() != ".aedt":
            return None
        lock = Path(str(p) + ".lock")
        if lock.exists():
            return lock
        return None

    @staticmethod
    def _project_matches(current: Optional[str], requested: Optional[str]) -> bool:
        if not requested:
            return True
        if not current:
            return False
        c = str(current).strip()
        r = str(requested).strip()
        if not c or not r:
            return False
        c_norm = os.path.normcase(os.path.normpath(c))
        r_norm = os.path.normcase(os.path.normpath(r))
        if c_norm == r_norm:
            return True
        # Compare by basename/stem when one side is a path and the other is project token.
        c_stem = Path(c).stem.lower()
        r_stem = Path(r).stem.lower()
        return c_stem == r_stem

    @staticmethod
    def _pick_attach_pid(student_version: bool, non_graphical: Optional[bool]) -> Optional[int]:
        """Pick an existing AEDT process PID for attach mode."""
        tgt = AedtHfssSession._pick_attach_target(student_version=student_version, non_graphical=non_graphical)
        return None if tgt is None else int(tgt.get("pid") or 0) or None

    @staticmethod
    def _pick_attach_target(
        student_version: bool,
        non_graphical: Optional[bool],
        allow_fallback: bool = True,
    ) -> Optional[dict]:
        """Pick best AEDT attach target and return {'pid','port','is_ng','ctime'}.

        Selection policy:
          1) Prefer sessions matching requested graphical mode.
          2) Fallback to any session when strict match is unavailable.
          3) Prefer valid listening gRPC ports and most recent process.
        """
        target_name = "ansysedtsv.exe" if bool(student_version) else "ansysedt.exe"
        target_noext = target_name.replace(".exe", "")
        ps_mod = None
        try:
            import psutil as _psutil  # type: ignore
            ps_mod = _psutil
        except Exception:
            try:
                ps_mod = create_psutil_shim_module()
            except Exception:
                return None
        if ps_mod is None:
            return None

        candidates: list[dict] = []
        try:
            for proc in ps_mod.process_iter(attrs=["pid", "name", "cmdline", "create_time"]):
                info = getattr(proc, "info", {}) or {}
                name = str(info.get("name") or "").strip().lower()
                if name not in (target_name, target_noext):
                    continue
                cmdline_raw = info.get("cmdline")
                if isinstance(cmdline_raw, str):
                    cmdline = [x for x in cmdline_raw.split() if x]
                elif isinstance(cmdline_raw, (list, tuple)):
                    cmdline = [str(x) for x in cmdline_raw if str(x)]
                else:
                    cmdline = []
                args = [x.lower() for x in cmdline]
                is_ng = "-ng" in args

                try:
                    pid = int(info.get("pid") or getattr(proc, "pid", 0))
                except Exception:
                    continue
                if pid <= 0:
                    continue
                try:
                    ctime = float(info.get("create_time") or getattr(proc, "create_time", lambda: 0.0)())
                except Exception:
                    ctime = 0.0
                parsed_port = -1
                if "-grpcsrv" in cmdline:
                    try:
                        raw = str(cmdline[cmdline.index("-grpcsrv") + 1]).strip()
                        if raw.isdigit():
                            parsed_port = int(raw)
                        else:
                            for part in reversed(raw.split(":")):
                                part = str(part).strip()
                                if part.isdigit():
                                    parsed_port = int(part)
                                    break
                    except Exception:
                        parsed_port = -1
                listen_ports: set[int] = set()
                try:
                    for c in proc.net_connections(kind="tcp"):
                        if str(getattr(c, "status", "")).upper() != "LISTEN":
                            continue
                        laddr = getattr(c, "laddr", None)
                        pval = int(getattr(laddr, "port", 0) or 0)
                        if pval > 0:
                            listen_ports.add(pval)
                except Exception:
                    listen_ports = set()

                if parsed_port > 0 and parsed_port not in listen_ports:
                    parsed_port = -1
                if parsed_port <= 0 and listen_ports:
                    candidates_ports = sorted([p for p in listen_ports if p not in (2001, 2002)])
                    if len(candidates_ports) == 1:
                        parsed_port = int(candidates_ports[0])

                candidates.append(
                    {
                        "pid": pid,
                        "port": int(parsed_port) if int(parsed_port) > 0 else 0,
                        "is_ng": bool(is_ng),
                        "ctime": float(ctime),
                    }
                )
        except Exception:
            return None
        if not candidates:
            return None

        def _pick(pool: list[dict]) -> Optional[dict]:
            if not pool:
                return None
            ranked = sorted(
                pool,
                key=lambda x: (
                    1 if int(x.get("port", 0)) > 0 else 0,
                    float(x.get("ctime", 0.0)),
                ),
                reverse=True,
            )
            return ranked[0]

        preferred: list[dict]
        if non_graphical is True:
            preferred = [c for c in candidates if bool(c.get("is_ng"))]
        elif non_graphical is False:
            preferred = [c for c in candidates if (not bool(c.get("is_ng")))]
        else:
            preferred = list(candidates)

        picked = _pick(preferred)
        if picked is None and allow_fallback:
            picked = _pick(candidates)
        return picked

    @staticmethod
    def _open_project_context(hfss_obj, project: Optional[str], design: Optional[str]) -> None:
        """Open/activate the requested project and optional design explicitly."""
        proj = AedtHfssSession._norm_project(project)
        if not proj:
            return
        p = Path(proj)
        proj_is_file = p.suffix.lower() == ".aedt"
        proj_file = str(p) if proj_is_file else ""
        proj_name = p.stem if proj_is_file else str(proj).strip()
        last_error: Optional[Exception] = None

        if proj_is_file and p.exists() and hasattr(hfss_obj, "load_project"):
            try:
                loaded = hfss_obj.load_project(proj_file, design=design, close_active=False, set_active=False)
                if loaded:
                    if design and hasattr(hfss_obj, "set_active_design"):
                        try:
                            hfss_obj.set_active_design(design)
                        except Exception:
                            pass
                    return
            except Exception as e_load:
                last_error = e_load

        desk = getattr(hfss_obj, "odesktop", None)
        if desk is None:
            desk_cls = getattr(hfss_obj, "desktop_class", None)
            if desk_cls is not None:
                desk = getattr(desk_cls, "odesktop", None)
        if desk is None:
            if last_error:
                raise RuntimeError(f"Unable to access AEDT desktop object: {last_error}") from last_error
            raise RuntimeError("Unable to access AEDT desktop object.")

        if proj_is_file and p.exists() and hasattr(desk, "OpenProject"):
            try:
                desk.OpenProject(proj_file)
            except Exception as e_open:
                last_error = e_open

        pobj = None
        for token in [proj_file, proj_name, proj]:
            token = str(token or "").strip()
            if not token:
                continue
            try:
                pobj = desk.SetActiveProject(token)
                if pobj is not None:
                    break
            except Exception as e_set:
                last_error = e_set
                continue

        if pobj is None:
            if last_error:
                raise RuntimeError(f"Unable to activate project '{proj}': {last_error}") from last_error
            raise RuntimeError(f"Unable to activate project '{proj}'.")

        if design:
            try:
                pobj.SetActiveDesign(design)
            except Exception:
                try:
                    if hasattr(hfss_obj, "set_active_design"):
                        hfss_obj.set_active_design(design)
                except Exception:
                    pass

        active_project = AedtHfssSession._norm_project(
            getattr(hfss_obj, "project_file", None)
        ) or AedtHfssSession._norm_project(getattr(hfss_obj, "project_name", None))
        # Some PyAEDT builds can leave project_file/project_name blank right
        # after SetActiveProject/load_project. Only fail on explicit mismatch.
        if active_project and (not AedtHfssSession._project_matches(active_project, proj)):
            raise RuntimeError(
                f"Project context mismatch. Requested='{proj}' Active='{active_project or ''}'"
            )

    @property
    def hfss(self):
        if self._hfss is None:
            raise RuntimeError("HFSS session not connected")
        return self._hfss

    @property
    def is_connected(self) -> bool:
        return self._hfss is not None

    @property
    def connected_seconds(self) -> float:
        if not self._connected_at:
            return 0.0
        return time.time() - self._connected_at

    def connect(
        self,
        project: Optional[str] = None,
        design: Optional[str] = None,
        setup: Optional[str] = None,
        force: bool = False,
        remove_lock_override: Optional[bool] = None,
    ):
        """Connect to AEDT/HFSS (launch or attach), and open a project/design.

        Parameters:
          project: project name or full path to .aedt (per PyAEDT Hfss docs)
          design: design name within the project (optional)
          setup: nominal setup to select (optional)
          force: force context rebind even if project/design appear unchanged.

        Raises:
          RuntimeError on connection failures.
        """
        with self._lock:
            with audit_span(
                "HFSS_SESSION_CONNECT",
                version=self.cfg.version,
                project=project or "",
                design=design or "",
                setup=setup or "",
                force=int(bool(force)),
            ):
                project = self._norm_project(project)
                design = str(design).strip() if design else None
                remove_lock = bool(self.cfg.remove_lock) if remove_lock_override is None else bool(remove_lock_override)

                if self._hfss is not None and not force:
                    current_project = self._norm_project(getattr(self._hfss, "project_file", None)) or self._norm_project(getattr(self._hfss, "project_name", None)) or self._norm_project(self._last_project)
                    current_design = str(getattr(self._hfss, "design_name", "") or self._last_design or "").strip() or None
                    same_project = self._project_matches(current_project, project)
                    same_design = (not design) or (current_design == design)
                    if same_project and same_design:
                        _patch_desktop_compat(self._hfss)
                        return

                if self._hfss is not None:
                    try:
                        self._hfss.release_desktop(close_projects=False, close_desktop=False)
                    except Exception:
                        try:
                            self._hfss.close_project(save_project=False)
                        except Exception:
                            pass
                    finally:
                        self._hfss = None

                _prepare_windows_dll_resolution(self.cfg.version)
                Hfss = _import_hfss(self.cfg.version)
                _force_pyaedt_single_desktop_mode()
                _patch_pyaedt_solution_constants()

                # Normalize paths (optional).
                if project:
                    p = Path(project)
                    if p.suffix.lower() == ".aedt" and p.exists():
                        project = str(p)

                # New desktop only on first connect when requested.
                new_desktop = bool(self.cfg.new_desktop and not self._ever_connected)
                attach_pid = self.cfg.aedt_process_id
                attach_port = int(self.cfg.port or 0)
                call_non_graphical = bool(self.cfg.non_graphical)
                if not new_desktop:
                    if attach_pid:
                        picked = self._pick_attach_target(
                            student_version=self.cfg.student_version,
                            non_graphical=self.cfg.non_graphical,
                            allow_fallback=True,
                        )
                        if picked and int(picked.get("pid", 0)) == int(attach_pid) and int(picked.get("port", 0)) > 0 and attach_port <= 0:
                            attach_port = int(picked.get("port", 0))
                    else:
                        picked = self._pick_attach_target(
                            student_version=self.cfg.student_version,
                            non_graphical=self.cfg.non_graphical,
                            allow_fallback=False,
                        )
                        if picked is None:
                            # No strict mode match available. Fall back to any running
                            # AEDT session and align non_graphical with the chosen target.
                            picked = self._pick_attach_target(
                                student_version=self.cfg.student_version,
                                non_graphical=None,
                                allow_fallback=True,
                            )
                            if picked is not None:
                                call_non_graphical = bool(picked.get("is_ng"))
                        if picked:
                            attach_pid = int(picked.get("pid", 0)) or None
                            if attach_port <= 0:
                                attach_port = int(picked.get("port", 0) or 0)
                port_to_use = int(attach_port if (not new_desktop and attach_port > 0) else int(self.cfg.port or 0))

                def _open(version_value):
                    return Hfss(
                        project=project,
                        design=design,
                        setup=setup,
                        version=version_value,
                        non_graphical=call_non_graphical,
                        new_desktop=new_desktop,
                        close_on_exit=self.cfg.close_on_exit,
                        student_version=self.cfg.student_version,
                        machine=self.cfg.machine,
                        port=port_to_use,
                        aedt_process_id=attach_pid,
                        remove_lock=remove_lock,
                    )

                def _open_no_project(version_value):
                    return Hfss(
                        project=None,
                        design=None,
                        setup=setup,
                        version=version_value,
                        non_graphical=call_non_graphical,
                        new_desktop=new_desktop,
                        close_on_exit=self.cfg.close_on_exit,
                        student_version=self.cfg.student_version,
                        machine=self.cfg.machine,
                        port=port_to_use,
                        aedt_process_id=attach_pid,
                        remove_lock=remove_lock,
                    )

                opened_via_fallback = False
                try:
                    self._hfss = _open(self.cfg.version)
                    _patch_desktop_compat(self._hfss)
                except Exception as e_first:
                    msg_first = str(e_first)

                    if project and ("OpenProject" in msg_first or "project" in msg_first.lower()):
                        try:
                            self._hfss = _open_no_project(self.cfg.version)
                            _patch_desktop_compat(self._hfss)
                            self._open_project_context(self._hfss, project=project, design=design)
                            opened_via_fallback = True
                        except Exception:
                            try:
                                if self._hfss is not None:
                                    self._hfss.release_desktop(close_projects=False, close_desktop=False)
                            except Exception:
                                pass
                            finally:
                                self._hfss = None

                    if (not opened_via_fallback) and ("PyDesktopPlugin.dll" not in msg_first):
                        try:
                            if self._hfss is not None:
                                self._hfss.release_desktop(close_projects=False, close_desktop=False)
                        except Exception:
                            pass
                        finally:
                            self._hfss = None
                        raise RuntimeError(
                            f"Failed to connect to AEDT/HFSS via PyAEDT: {e_first} "
                            f"| psutil_backend={_PSUTIL_BACKEND}"
                        ) from e_first

                    if not opened_via_fallback:
                        retry_errors: list[str] = []
                        try_versions = []
                        if new_desktop:
                            token = _version_to_token(self.cfg.version)
                            if token and token != str(self.cfg.version).strip():
                                try_versions.append(token)
                            try_versions.append(None)

                        for v_try in try_versions:
                            try:
                                _prepare_windows_dll_resolution(self.cfg.version if v_try is None else str(v_try))
                                self._hfss = _open(v_try)
                                _patch_desktop_compat(self._hfss)
                                break
                            except Exception as e_retry:
                                self._hfss = None
                                retry_errors.append(f"version={v_try!r}: {e_retry}")

                        if self._hfss is None:
                            roots = [str(p) for p in _discover_aedt_roots(self.cfg.version)]
                            help_msg = (
                                "Falha ao carregar dependencias nativas do AEDT (PyDesktopPlugin.dll). "
                                f"Versao solicitada: {self.cfg.version!r}. "
                                f"Raizes AEDT detectadas: {roots or ['<nenhuma>']}."
                            )
                            if retry_errors:
                                help_msg += " Retries: " + " | ".join(retry_errors)
                            elif not new_desktop:
                                help_msg += (
                                    " Attach mode: retries were skipped intentionally "
                                    "to avoid opening multiple AEDT instances."
                                )
                            raise RuntimeError(
                                f"Failed to connect to AEDT/HFSS via PyAEDT: {e_first}. {help_msg} "
                                f"| psutil_backend={_PSUTIL_BACKEND}"
                            ) from e_first

                self._connected_at = time.time()
                self._last_project = project
                self._last_design = design
                self._ever_connected = True
                emit_audit(
                    "HFSS_SESSION_CONNECT_OK",
                    version=self.cfg.version,
                    project=project or "",
                    design=design or "",
                    setup=setup or "",
                )

    def reconnect_last(self):
        """Reconnect using the last project/design."""
        with self._lock:
            if self._hfss is not None:
                return
            self.connect(project=self._last_project, design=self._last_design)

    def disconnect(self):
        """Release HFSS session.

        Important:
          - In some environments, calling release/close may be necessary to avoid orphaned AEDT sessions.
          - We do not forcibly kill AEDT processes.
        """
        with self._lock:
            if self._hfss is None:
                return
            try:
                emit_audit("HFSS_SESSION_DISCONNECT_START", project=self._last_project or "", design=self._last_design or "")
                self._hfss.release_desktop(close_projects=False, close_desktop=self.cfg.close_on_exit)
            except Exception:
                # Fallback: older PyAEDT versions may use different shutdown semantics.
                try:
                    self._hfss.close_project(save_project=False)
                except Exception:
                    pass
            finally:
                self._hfss = None
                self._connected_at = None
                emit_audit("HFSS_SESSION_DISCONNECT_OK", project=self._last_project or "", design=self._last_design or "")
