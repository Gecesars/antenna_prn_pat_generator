from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path
from typing import Optional
import os

import threading
import time


def _import_hfss():
    # Prefer the modern import path used by PyAEDT docs.
    try:
        from ansys.aedt.core import Hfss  # type: ignore
        return Hfss
    except Exception:
        # Fallback for older installs.
        try:
            from pyaedt import Hfss  # type: ignore
            return Hfss
        except Exception as e:
            raise RuntimeError(f"Unable to import PyAEDT Hfss API: {e}") from e


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

            Hfss = _import_hfss()
            _force_pyaedt_single_desktop_mode()
            _patch_pyaedt_solution_constants()

            # Normalize paths (optional).
            if project:
                p = Path(project)
                if p.suffix.lower() == ".aedt" and p.exists():
                    project = str(p)

            try:
                # New desktop only on first connect when requested.
                new_desktop = bool(self.cfg.new_desktop and not self._ever_connected)
                self._hfss = Hfss(
                    project=project,
                    design=design,
                    setup=setup,
                    version=self.cfg.version,
                    non_graphical=self.cfg.non_graphical,
                    new_desktop=new_desktop,
                    close_on_exit=self.cfg.close_on_exit,
                    student_version=self.cfg.student_version,
                    machine=self.cfg.machine,
                    port=self.cfg.port,
                    aedt_process_id=self.cfg.aedt_process_id,
                    remove_lock=remove_lock,
                )
                _patch_desktop_compat(self._hfss)
            except Exception as e:
                self._hfss = None
                raise RuntimeError(f"Failed to connect to AEDT/HFSS via PyAEDT: {e}") from e

            self._connected_at = time.time()
            self._last_project = project
            self._last_design = design
            self._ever_connected = True

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
