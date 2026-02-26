from __future__ import annotations

import importlib
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from ..models import CapabilityReport, MechanicalError


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _fmt_token(fmt: str | None, path: str = "") -> str:
    token = str(fmt or "").strip().lower().lstrip(".")
    if token:
        return token
    ext = Path(path).suffix.lower().lstrip(".")
    return ext


def _vec_components(vec_obj: Any) -> Tuple[float, float, float]:
    for names in (("x", "y", "z"), ("X", "Y", "Z")):
        try:
            return float(getattr(vec_obj, names[0])), float(getattr(vec_obj, names[1])), float(getattr(vec_obj, names[2]))
        except Exception:
            continue
    try:
        arr = list(vec_obj)
        return float(arr[0]), float(arr[1]), float(arr[2])
    except Exception:
        return 0.0, 0.0, 0.0


class FreeCADKernelProvider:
    """FreeCAD/OCCT in-process provider.

    This class keeps FreeCAD imports lazy and safe so the app can run
    in fallback mode when FreeCAD is not installed.
    """

    def __init__(self, logger: logging.Logger | None = None):
        self.logger = logger or logging.getLogger(__name__)
        self._fc = None
        self._part = None
        self._mesh_mod = None
        self._doc = None
        self._shapes: Dict[str, Any] = {}
        self._meta: Dict[str, Dict[str, Any]] = {}
        self._caps = CapabilityReport(provider="freecad")
        self._init_backend()

    @property
    def capabilities(self) -> CapabilityReport:
        return self._caps

    def _init_backend(self) -> None:
        try:
            self._fc = importlib.import_module("FreeCAD")
            self._part = importlib.import_module("Part")
            try:
                self._mesh_mod = importlib.import_module("Mesh")
            except Exception:
                self._mesh_mod = None
        except Exception as exc:
            raise MechanicalError(
                f"Unable to import FreeCAD modules: {exc}",
                code="freecad_import_error",
            ) from exc

        freecad_cmd = ""
        try:
            from ..diagnostics import _detect_freecadcmd  # local import to avoid cycles

            freecad_cmd = _detect_freecadcmd()
        except Exception:
            freecad_cmd = ""

        if hasattr(self._fc, "newDocument"):
            try:
                self._doc = self._fc.newDocument(f"EFTX_Mech_{uuid.uuid4().hex[:8]}")
            except Exception:
                self._doc = None

        self._caps = CapabilityReport(
            provider="freecad",
            freecad_available=True,
            freecad_headless_available=bool(freecad_cmd),
            fem_available=bool(importlib.util.find_spec("Fem") is not None),
            validate_available=True,
            heal_available=True,
            import_formats=("step", "stp", "stl", "iges", "igs"),
            export_formats=("step", "stp", "stl", "iges", "igs"),
            primitive_kinds=("box", "cylinder", "sphere", "cone", "tube"),
            boolean_ops=("union", "cut", "common"),
            notes=("in-process FreeCAD provider",),
            extra={"freecad_cmd": str(freecad_cmd)},
        )

    def _new_id(self) -> str:
        return str(uuid.uuid4())

    def _vec(self, xyz: Sequence[float]) -> Any:
        x, y, z = [float(v) for v in xyz]
        return self._fc.Vector(x, y, z)

    def _get_shape(self, obj_id: str) -> Any:
        key = str(obj_id)
        if key not in self._shapes:
            raise MechanicalError(f"Shape id not found: {key}", code="shape_not_found", details={"obj_id": key})
        return self._shapes[key]

    def _store_shape(self, shape: Any, *, name: str, kind: str, source_path: str = "") -> str:
        obj_id = self._new_id()
        self._shapes[obj_id] = shape
        self._meta[obj_id] = {
            "name": str(name or kind),
            "kind": str(kind),
            "source_path": str(source_path or ""),
        }
        return obj_id

    def _center_xyz(self, params: Mapping[str, Any]) -> Tuple[float, float, float]:
        center = params.get("center", (0.0, 0.0, 0.0))
        try:
            vals = list(center)
        except Exception:
            vals = [0.0, 0.0, 0.0]
        if len(vals) != 3:
            vals = [0.0, 0.0, 0.0]
        return float(vals[0]), float(vals[1]), float(vals[2])

    def create_primitive(self, kind: str, params: Mapping[str, Any]) -> str:
        token = str(kind or "").strip().lower()
        if token not in self._caps.primitive_kinds:
            raise MechanicalError(
                f"Unsupported primitive kind for FreeCAD provider: {token}",
                code="primitive_not_supported",
                details={"kind": token},
            )

        cx, cy, cz = self._center_xyz(params)
        p = dict(params or {})

        if token == "box":
            w = max(1e-9, _safe_float(p.get("width", 1.0), 1.0))
            d = max(1e-9, _safe_float(p.get("depth", 1.0), 1.0))
            h = max(1e-9, _safe_float(p.get("height", 1.0), 1.0))
            shape = self._part.makeBox(w, d, h)
            shape.translate(self._vec((cx - w * 0.5, cy - d * 0.5, cz - h * 0.5)))
        elif token == "cylinder":
            r = max(1e-9, _safe_float(p.get("radius", 0.5), 0.5))
            h = max(1e-9, _safe_float(p.get("height", 1.0), 1.0))
            shape = self._part.makeCylinder(r, h, self._vec((cx, cy, cz - h * 0.5)), self._vec((0.0, 0.0, 1.0)), 360.0)
        elif token == "sphere":
            r = max(1e-9, _safe_float(p.get("radius", 0.5), 0.5))
            shape = self._part.makeSphere(r, self._vec((cx, cy, cz)))
        elif token == "cone":
            r = max(1e-9, _safe_float(p.get("radius", 0.5), 0.5))
            h = max(1e-9, _safe_float(p.get("height", 1.0), 1.0))
            shape = self._part.makeCone(r, 0.0, h, self._vec((cx, cy, cz - h * 0.5)), self._vec((0.0, 0.0, 1.0)), 360.0)
        elif token == "tube":
            ro = max(1e-9, _safe_float(p.get("outer_radius", p.get("radius", 0.6)), 0.6))
            ri = max(1e-9, _safe_float(p.get("inner_radius", ro * 0.5), ro * 0.5))
            if ri >= ro:
                ri = ro * 0.5
            h = max(1e-9, _safe_float(p.get("height", 1.0), 1.0))
            outer = self._part.makeCylinder(ro, h, self._vec((cx, cy, cz - h * 0.5)), self._vec((0.0, 0.0, 1.0)), 360.0)
            inner = self._part.makeCylinder(ri, h, self._vec((cx, cy, cz - h * 0.5)), self._vec((0.0, 0.0, 1.0)), 360.0)
            shape = outer.cut(inner)
        else:
            raise MechanicalError(f"Unhandled primitive kind: {token}", code="primitive_not_supported")

        return self._store_shape(shape, name=token, kind=token)

    def transform(self, obj_id: str, transform: Mapping[str, Any]) -> None:
        shape = self._get_shape(obj_id).copy()
        payload = dict(transform or {})

        tx = _safe_float(payload.get("tx", payload.get("x", 0.0)), 0.0)
        ty = _safe_float(payload.get("ty", payload.get("y", 0.0)), 0.0)
        tz = _safe_float(payload.get("tz", payload.get("z", 0.0)), 0.0)
        rx = _safe_float(payload.get("rx_deg", payload.get("rx", 0.0)), 0.0)
        ry = _safe_float(payload.get("ry_deg", payload.get("ry", 0.0)), 0.0)
        rz = _safe_float(payload.get("rz_deg", payload.get("rz", 0.0)), 0.0)
        scale = max(1e-9, _safe_float(payload.get("scale", 1.0), 1.0))

        bb = shape.BoundBox
        center = self._vec(((bb.XMin + bb.XMax) * 0.5, (bb.YMin + bb.YMax) * 0.5, (bb.ZMin + bb.ZMax) * 0.5))

        if abs(scale - 1.0) > 1e-12:
            try:
                m = self._fc.Matrix()
                m.A11 = scale
                m.A22 = scale
                m.A33 = scale
                m.A44 = 1.0
                shape = shape.transformGeometry(m)
            except Exception as exc:
                raise MechanicalError(
                    f"FreeCAD scale transform failed: {exc}",
                    code="transform_failed",
                    details={"obj_id": str(obj_id), "scale": float(scale)},
                ) from exc

        try:
            if abs(rx) > 1e-12:
                shape.rotate(center, self._vec((1.0, 0.0, 0.0)), rx)
            if abs(ry) > 1e-12:
                shape.rotate(center, self._vec((0.0, 1.0, 0.0)), ry)
            if abs(rz) > 1e-12:
                shape.rotate(center, self._vec((0.0, 0.0, 1.0)), rz)
            if abs(tx) > 1e-12 or abs(ty) > 1e-12 or abs(tz) > 1e-12:
                shape.translate(self._vec((tx, ty, tz)))
        except Exception as exc:
            raise MechanicalError(
                f"FreeCAD transform failed: {exc}",
                code="transform_failed",
                details={"obj_id": str(obj_id)},
            ) from exc

        self._shapes[str(obj_id)] = shape

    def boolean(self, op: str, a_id: str, b_id: str | Sequence[str]) -> str:
        token = str(op or "").strip().lower()
        if token not in self._caps.boolean_ops:
            raise MechanicalError(f"Unsupported boolean op: {token}", code="boolean_not_supported")

        shape_a = self._get_shape(a_id)
        targets: List[str]
        if isinstance(b_id, str):
            targets = [str(b_id)]
        else:
            targets = [str(x) for x in b_id]
        targets = [t for t in targets if t and t != str(a_id)]
        if not targets:
            raise MechanicalError("Boolean operation requires at least one target shape", code="invalid_boolean_args")

        out = shape_a
        for tid in targets:
            bshape = self._get_shape(tid)
            if token == "union":
                out = out.fuse(bshape)
            elif token == "cut":
                out = out.cut(bshape)
            else:
                out = out.common(bshape)

        return self._store_shape(out, name=f"{token}_result", kind=f"boolean:{token}")

    def import_model(self, path: str, fmt: str | None = None) -> Sequence[str]:
        src = str(path or "").strip()
        if not src or not os.path.isfile(src):
            raise MechanicalError(f"Input path not found: {src}", code="path_not_found")

        token = _fmt_token(fmt, src)
        if token not in self._caps.import_formats:
            raise MechanicalError(f"Import format not supported: {token}", code="import_format_not_supported")

        try:
            if token in {"step", "stp", "iges", "igs"}:
                shape = self._part.read(src)
            elif token == "stl":
                if self._mesh_mod is None:
                    raise MechanicalError("FreeCAD Mesh module is unavailable for STL import", code="mesh_module_missing")
                mesh = self._mesh_mod.Mesh(src)
                shape = self._part.Shape()
                shape.makeShapeFromMesh(mesh.Topology, 0.1)
            else:
                raise MechanicalError(f"Import format not supported: {token}", code="import_format_not_supported")
        except MechanicalError:
            raise
        except Exception as exc:
            raise MechanicalError(
                f"FreeCAD import failed: {exc}",
                code="import_failed",
                details={"path": src, "format": token},
            ) from exc

        sub_shapes: List[Any] = []
        try:
            solids = list(getattr(shape, "Solids", []) or [])
        except Exception:
            solids = []
        if len(solids) > 1:
            sub_shapes = solids
        elif len(solids) == 1:
            sub_shapes = [solids[0]]
        else:
            try:
                shells = list(getattr(shape, "Shells", []) or [])
            except Exception:
                shells = []
            if len(shells) > 1:
                sub_shapes = shells

        if not sub_shapes:
            sub_shapes = [shape]

        out_ids: List[str] = []
        total = len(sub_shapes)
        for idx, shp in enumerate(sub_shapes, start=1):
            try:
                s = shp.copy() if hasattr(shp, "copy") else shp
            except Exception:
                s = shp
            if total > 1:
                name = f"{Path(src).stem}_body_{idx:02d}"
            else:
                name = Path(src).stem
            oid = self._store_shape(s, name=name, kind=f"import:{token}", source_path=src)
            try:
                self._meta[str(oid)]["extra"] = {"body_index": int(idx), "body_count": int(total)}
            except Exception:
                pass
            out_ids.append(oid)
        return out_ids

    def export_model(self, obj_ids: Sequence[str], path: str, fmt: str | None = None) -> str:
        out_path = str(path or "").strip()
        if not out_path:
            raise MechanicalError("Output path is empty", code="invalid_output_path")

        token = _fmt_token(fmt, out_path)
        if token not in self._caps.export_formats:
            raise MechanicalError(f"Export format not supported: {token}", code="export_format_not_supported")

        ids = [str(x) for x in obj_ids if str(x) in self._shapes]
        if not ids:
            raise MechanicalError("No valid objects selected for export", code="empty_selection")

        shapes = [self._get_shape(oid) for oid in ids]
        if len(shapes) == 1:
            out_shape = shapes[0]
        else:
            try:
                out_shape = self._part.makeCompound(shapes)
            except Exception:
                out_shape = shapes[0]
                for extra in shapes[1:]:
                    out_shape = out_shape.fuse(extra)

        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)

        try:
            if token in {"step", "stp"}:
                out_shape.exportStep(out_path)
            elif token == "stl":
                out_shape.exportStl(out_path)
            elif token in {"iges", "igs"}:
                out_shape.exportIges(out_path)
            else:
                raise MechanicalError(f"Export format not supported: {token}", code="export_format_not_supported")
        except MechanicalError:
            raise
        except Exception as exc:
            raise MechanicalError(
                f"FreeCAD export failed: {exc}",
                code="export_failed",
                details={"path": out_path, "format": token},
            ) from exc

        return out_path

    def get_properties(self, obj_id: str) -> Mapping[str, Any]:
        shape = self._get_shape(obj_id)
        bb = shape.BoundBox
        cm = _vec_components(getattr(shape, "CenterOfMass", (0.0, 0.0, 0.0)))
        props: Dict[str, Any] = {
            "id": str(obj_id),
            "name": self._meta.get(str(obj_id), {}).get("name", ""),
            "kind": self._meta.get(str(obj_id), {}).get("kind", ""),
            "volume": float(getattr(shape, "Volume", 0.0) or 0.0),
            "area": float(getattr(shape, "Area", 0.0) or 0.0),
            "center_of_mass": [float(cm[0]), float(cm[1]), float(cm[2])],
            "bbox": {
                "xmin": float(bb.XMin),
                "xmax": float(bb.XMax),
                "ymin": float(bb.YMin),
                "ymax": float(bb.YMax),
                "zmin": float(bb.ZMin),
                "zmax": float(bb.ZMax),
            },
            "is_valid": bool(shape.isValid()),
        }
        extra = self._meta.get(str(obj_id), {}).get("extra", {})
        if isinstance(extra, dict):
            props["extra"] = dict(extra)
        return props

    def set_properties(self, obj_id: str, props: Mapping[str, Any]) -> None:
        key = str(obj_id)
        _ = self._get_shape(key)
        row = self._meta.setdefault(key, {})
        for k, v in dict(props or {}).items():
            row[str(k)] = v

    def triangulate(self, obj_id: str, quality: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
        shape = self._get_shape(obj_id)
        q = dict(quality or {})
        deflection = max(1e-6, _safe_float(q.get("deflection", q.get("tolerance", 0.2)), 0.2))

        try:
            points, facets = shape.tessellate(deflection)
        except Exception as exc:
            raise MechanicalError(
                f"Triangulation failed: {exc}",
                code="triangulation_failed",
                details={"obj_id": str(obj_id), "deflection": float(deflection)},
            ) from exc

        verts = []
        for p in list(points or []):
            x, y, z = _vec_components(p)
            verts.append((float(x), float(y), float(z)))

        faces = []
        for tri in list(facets or []):
            try:
                a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
                faces.append((a, b, c))
            except Exception:
                continue

        return {
            "vertices": [list(v) for v in verts],
            "faces": [list(f) for f in faces],
            "meta": {
                "obj_id": str(obj_id),
                "deflection": float(deflection),
            },
        }

    def validate(self, obj_id: str) -> Mapping[str, Any]:
        shape = self._get_shape(obj_id)
        messages: List[str] = []
        ok = bool(shape.isValid())
        if not ok:
            messages.append("Shape is not valid according to OCCT validation")
        bb = shape.BoundBox
        return {
            "ok": bool(ok),
            "provider": "freecad",
            "messages": messages,
            "bbox": {
                "dx": float(bb.XLength),
                "dy": float(bb.YLength),
                "dz": float(bb.ZLength),
            },
        }

    def heal(self, obj_id: str) -> Mapping[str, Any]:
        shape = self._get_shape(obj_id).copy()
        fixed = False
        messages: List[str] = []
        try:
            if hasattr(shape, "fix"):
                fixed = bool(shape.fix(1e-6, 1e-7, 1e-5))
            if hasattr(shape, "removeSplitter"):
                shape = shape.removeSplitter()
            self._shapes[str(obj_id)] = shape
        except Exception as exc:
            messages.append(f"Heal failed: {exc}")

        report = self.validate(obj_id)
        report = dict(report)
        report["healed"] = bool(fixed)
        if messages:
            report.setdefault("messages", [])
            report["messages"].extend(messages)
            report["ok"] = False
        return report

    def delete(self, obj_id: str) -> None:
        key = str(obj_id)
        self._shapes.pop(key, None)
        self._meta.pop(key, None)

    def close(self) -> None:
        if self._doc is None:
            return
        try:
            if hasattr(self._fc, "closeDocument") and getattr(self._doc, "Name", ""):
                self._fc.closeDocument(self._doc.Name)
        except Exception:
            pass
        finally:
            self._doc = None
