from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class AppBridge:
    """Bridges AEDT-extracted patterns into the existing application.

    This class is intentionally reflection-based to avoid tight coupling.
    It will:
      1) Try to push cuts into the app's *current workspace* (project state)
      2) Try to store entries into the app's *library* (DB), if present
      3) Otherwise, do nothing (caller should at least export to disk)

    Expected/known integration points (best effort):
      - app.load_from_library(pattern_dict)
      - app.v_angles / app.v_mags / app.h_angles / app.h_mags + app.plot_diagrams()
      - app.db_manager.add_diagram(name, type, angles, values, meta)  (or any object with add_diagram)
    """

    app: Any

    def push_payload_to_project(self, payload: Dict) -> bool:
        if hasattr(self.app, "import_pattern_into_project") and callable(getattr(self.app, "import_pattern_into_project")):
            self.app.import_pattern_into_project(payload)
            return True

        cuts = payload.get("cuts_2d", {}) if isinstance(payload, dict) else {}
        ok = False
        if isinstance(cuts, dict):
            for mode, ptype in (("VRP", "VRP"), ("HRP", "HRP")):
                cut = cuts.get(mode)
                if not isinstance(cut, dict):
                    continue
                ok = self.push_to_workspace(
                    name=str(cut.get("name") or f"AEDT_{mode}"),
                    ptype=ptype,
                    angles_deg=cut.get("angles_deg", []),
                    values=cut.get("mag_lin", cut.get("values", [])),
                    meta=cut.get("meta", {}),
                ) or ok
        return ok

    def push_payload_to_library(self, payload: Dict) -> bool:
        if hasattr(self.app, "add_diagram_entry") and callable(getattr(self.app, "add_diagram_entry")):
            self.app.add_diagram_entry(payload)
            return True

        cuts = payload.get("cuts_2d", {}) if isinstance(payload, dict) else {}
        ok = False
        if isinstance(cuts, dict):
            for mode in ("VRP", "HRP"):
                cut = cuts.get(mode)
                if not isinstance(cut, dict):
                    continue
                ok = self.add_to_library(
                    name=str(cut.get("name") or f"AEDT_{mode}"),
                    ptype=mode,
                    angles_deg=cut.get("angles_deg", []),
                    values=cut.get("mag_lin", cut.get("values", [])),
                    meta=cut.get("meta", {}),
                ) or ok
        return ok

    def push_to_workspace(self, name: str, ptype: str, angles_deg, values, meta: Optional[Dict] = None) -> bool:
        meta = meta or {}
        pattern = {
            "name": name,
            "type": ptype,
            "angles": list(map(float, angles_deg)),
            "values": list(map(float, values)),
            "meta": meta,
        }

        if hasattr(self.app, "import_pattern_into_project") and callable(getattr(self.app, "import_pattern_into_project")):
            mode = "VRP" if str(ptype).upper().startswith("V") else "HRP"
            payload = {
                "cuts_2d": {
                    mode: {
                        "name": str(name),
                        "angles_deg": pattern["angles"],
                        "mag_lin": pattern["values"],
                        "meta": meta,
                    }
                },
                "meta": dict(meta),
            }
            try:
                self.app.import_pattern_into_project(payload)
                return True
            except Exception:
                pass

        # Preferred: reuse the app's existing loader
        if hasattr(self.app, "load_from_library") and callable(getattr(self.app, "load_from_library")):
            try:
                self.app.load_from_library(pattern)
                return True
            except Exception:
                pass

        # Fallback: set arrays directly
        try:
            if ptype.upper().startswith("VRP"):
                setattr(self.app, "v_angles", pattern["angles"])
                setattr(self.app, "v_mags", pattern["values"])
            elif ptype.upper().startswith("HRP"):
                setattr(self.app, "h_angles", pattern["angles"])
                setattr(self.app, "h_mags", pattern["values"])
            if hasattr(self.app, "plot_diagrams") and callable(getattr(self.app, "plot_diagrams")):
                self.app.plot_diagrams()
                return True
        except Exception:
            pass

        return False

    def add_to_library(self, name: str, ptype: str, angles_deg, values, meta: Optional[Dict] = None) -> bool:
        meta = meta or {}
        if hasattr(self.app, "add_diagram_entry") and callable(getattr(self.app, "add_diagram_entry")):
            mode = "VRP" if str(ptype).upper().startswith("V") else "HRP"
            payload = {
                "cuts_2d": {
                    mode: {
                        "name": str(name),
                        "angles_deg": list(map(float, angles_deg)),
                        "mag_lin": list(map(float, values)),
                        "meta": dict(meta),
                    }
                },
                "meta": dict(meta),
            }
            try:
                self.app.add_diagram_entry(payload)
                return True
            except Exception:
                pass

        # Try common attribute names for DB manager
        for attr in ("db_manager", "dbm", "db", "library_db", "database"):
            obj = getattr(self.app, attr, None)
            if obj is None:
                continue
            if hasattr(obj, "add_diagram") and callable(getattr(obj, "add_diagram")):
                try:
                    obj.add_diagram(name=name, type_=ptype, angles=list(map(float, angles_deg)), values=list(map(float, values)), meta=meta)
                    return True
                except TypeError:
                    # different signature: add_diagram(name, type, angles, values, meta)
                    try:
                        obj.add_diagram(name, ptype, list(map(float, angles_deg)), list(map(float, values)), meta)
                        return True
                    except Exception:
                        pass
                except Exception:
                    pass

        # Some apps expose a direct method
        for fn in ("add_diagram_to_library", "save_diagram_to_library", "library_add"):
            if hasattr(self.app, fn) and callable(getattr(self.app, fn)):
                try:
                    getattr(self.app, fn)(name, ptype, angles_deg, values, meta)
                    return True
                except Exception:
                    pass

        return False
