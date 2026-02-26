from __future__ import annotations

import ast
import copy
import json
import math
import os
import logging
import uuid
from datetime import datetime, timezone
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .commands import (
    CommandStack,
    SnapshotCommand,
    build_command,
)
from .fem_study import FEMStudyManager
from .geometry_ops import (
    boolean_intersect,
    boolean_subtract,
    boolean_union,
    create_box,
    create_cone,
    create_cylinder,
    create_plane,
    create_sphere,
    create_tube,
    transform_mesh,
)
from .measures import angle, bbox, centroid, distance, object_metrics
from .scene_object import Marker, MeshData, SceneObject
from .validators import ensure_boolean_ready, validate_mesh
from mech.mechanical import build_default_kernel, collect_mechanical_diagnostics


ALLOWED_FUNCS = {
    "sqrt": math.sqrt,
    "log10": math.log10,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "abs": abs,
    "min": min,
    "max": max,
}

ALLOWED_AST_NODES = {
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Constant,
    ast.Name,
    ast.Load,
    ast.Call,
    ast.Pow,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.USub,
    ast.UAdd,
}

LOGGER = logging.getLogger(__name__)


class SafeExpressionEvaluator:
    def __init__(self):
        self._funcs = dict(ALLOWED_FUNCS)

    def evaluate(self, expr: str, variables: Dict[str, float]) -> float:
        text = str(expr or "").strip()
        if not text:
            raise RuntimeError("Empty expression.")
        tree = ast.parse(text, mode="eval")
        for node in ast.walk(tree):
            if type(node) not in ALLOWED_AST_NODES:
                raise RuntimeError(f"Unsupported token in expression: {type(node).__name__}")
            if isinstance(node, ast.Call):
                if not isinstance(node.func, ast.Name):
                    raise RuntimeError("Only direct function calls are allowed.")
                if node.func.id not in self._funcs:
                    raise RuntimeError(f"Function not allowed: {node.func.id}")
        env = {k: float(v) for k, v in variables.items()}
        env.update({k: v for k, v in self._funcs.items()})
        try:
            out = eval(compile(tree, "<expr>", "eval"), {"__builtins__": {}}, env)
        except Exception as e:
            raise RuntimeError(f"Expression evaluation failed: {e}") from e
        try:
            return float(out)
        except Exception as e:
            raise RuntimeError("Expression result is not numeric.") from e


class SceneEngine:
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or LOGGER
        self.objects: Dict[str, SceneObject] = {}
        self.selection: List[str] = []
        self.markers: Dict[str, Marker] = {}
        self.measurements: List[dict] = []
        self.boundaries: Dict[str, dict] = {}
        self.layers: Dict[str, dict] = {}
        self._default_mesh_quality: Dict[str, float] = {"deflection": 0.2}
        self.tool_mode: str = "Select"
        self.command_stack = CommandStack()
        self._listeners: List[Callable[[str, dict], None]] = []
        self.evaluator = SafeExpressionEvaluator()
        self.kernel, self._backend_diag = build_default_kernel(logger=self.logger)
        self._kernel_provider_name = str(getattr(self.kernel.capabilities, "provider", "null") or "null")
        self.fem = FEMStudyManager(object_provider=lambda: self.objects, event_emitter=self.emit)
        self._ensure_layer("Default", color="#86b6f6", visible=True, locked=False)

    # ---------------- events ----------------
    def add_listener(self, callback: Callable[[str, dict], None]) -> None:
        if callback not in self._listeners:
            self._listeners.append(callback)

    def remove_listener(self, callback: Callable[[str, dict], None]) -> None:
        if callback in self._listeners:
            self._listeners.remove(callback)

    def emit(self, event: str, payload: Optional[dict] = None) -> None:
        p = payload or {}
        for cb in list(self._listeners):
            try:
                cb(str(event), p)
            except Exception:
                continue

    def backend_diagnostics(self) -> dict:
        try:
            self._backend_diag = collect_mechanical_diagnostics(logger=self.logger).to_dict()
        except Exception:
            pass
        return dict(self._backend_diag or {})

    def backend_capabilities(self) -> dict:
        try:
            return dict(self.kernel.capabilities.to_dict())
        except Exception:
            return {"provider": self._kernel_provider_name}

    def backend_has(self, capability: str) -> bool:
        token = str(capability or "").strip().lower()
        caps = self.backend_capabilities()
        if token == "freecad_available":
            return bool(caps.get("freecad_available", False))
        if token == "freecad_headless_available":
            return bool(caps.get("freecad_headless_available", False))
        if token == "fem_available":
            return bool(caps.get("fem_available", False))
        if token in {"heal_available", "validate_available"}:
            return bool(caps.get(token, False))
        if token.startswith("primitive:"):
            kind = token.split(":", 1)[1]
            return kind in set(caps.get("primitive_kinds", []) or [])
        if token.startswith("import:"):
            fmt = token.split(":", 1)[1]
            return fmt in set(caps.get("import_formats", []) or [])
        if token.startswith("export:"):
            fmt = token.split(":", 1)[1]
            return fmt in set(caps.get("export_formats", []) or [])
        return bool(caps.get(token, False))

    def reconnect_backend(self) -> dict:
        try:
            close_fn = getattr(self.kernel, "close", None)
            if callable(close_fn):
                close_fn()
        except Exception:
            pass
        self.kernel, self._backend_diag = build_default_kernel(logger=self.logger)
        self._kernel_provider_name = str(getattr(self.kernel.capabilities, "provider", "null") or "null")
        for obj in self.objects.values():
            obj.meta.pop("kernel_obj_id", None)
            obj.meta["kernel_provider"] = str(self._kernel_provider_name)
        self.emit("scene_changed", {"reason": "backend_reconnect"})
        return self.backend_capabilities()

    # ---------------- quality/layers ----------------
    @staticmethod
    def _clean_layer_name(name: str) -> str:
        token = str(name or "").strip()
        if not token:
            token = "Default"
        safe = "".join(ch for ch in token if ch.isalnum() or ch in {"_", "-", " ", "."}).strip()
        return safe or "Default"

    @staticmethod
    def _coerce_quality(quality: Optional[dict]) -> Dict[str, float]:
        row = dict(quality or {})
        try:
            dfl = float(row.get("deflection", row.get("tolerance", 0.2)))
        except Exception:
            dfl = 0.2
        dfl = max(1e-6, dfl)
        return {"deflection": float(dfl)}

    def mesh_quality(self) -> Dict[str, float]:
        return dict(self._default_mesh_quality)

    def set_default_mesh_quality(self, quality: Optional[dict]) -> None:
        self._default_mesh_quality = self._coerce_quality(quality)

    def _ensure_layer(self, name: str, color: str = "#86b6f6", visible: bool = True, locked: bool = False) -> dict:
        key = self._clean_layer_name(name)
        row = dict(self.layers.get(key, {})) if key in self.layers else {}
        row["name"] = key
        row["color"] = str(row.get("color", color or "#86b6f6"))
        row["visible"] = bool(row.get("visible", visible))
        row["locked"] = bool(row.get("locked", locked))
        self.layers[key] = row
        return row

    def _apply_layer_defaults(self, obj: SceneObject, preferred: str = "") -> None:
        token = self._clean_layer_name(preferred or str(obj.meta.get("layer", "") or "Default"))
        row = self._ensure_layer(token)
        obj.meta["layer"] = str(row["name"])
        obj.meta.setdefault("layer_color", str(row.get("color", "#86b6f6")))
        self._ensure_object_metadata(obj)

    def _ensure_object_metadata(self, obj: SceneObject) -> None:
        if not isinstance(obj.meta, dict):
            obj.meta = {}
        meta = obj.meta
        material_name = str(meta.get("material", meta.get("obj_material", "")) or "")
        transform = dict(meta.get("transform", {})) if isinstance(meta.get("transform", {}), dict) else {}
        transform.setdefault("tx", 0.0)
        transform.setdefault("ty", 0.0)
        transform.setdefault("tz", 0.0)
        transform.setdefault("rx_deg", 0.0)
        transform.setdefault("ry_deg", 0.0)
        transform.setdefault("rz_deg", 0.0)
        transform.setdefault("sx", 1.0)
        transform.setdefault("sy", 1.0)
        transform.setdefault("sz", 1.0)

        tags = meta.get("tags", [])
        if not isinstance(tags, list):
            tags = [str(tags)]

        meta["uuid"] = str(meta.get("uuid", obj.id) or obj.id)
        meta["name"] = str(obj.name)
        meta["type"] = str(meta.get("type", "object") or "object")
        meta["source"] = str(obj.source)
        meta["layer"] = self._clean_layer_name(str(meta.get("layer", "Default") or "Default"))
        meta["visible"] = bool(obj.visible)
        meta["locked"] = bool(obj.locked)
        meta["material"] = str(material_name)
        meta["color"] = str(obj.color)
        meta["transform"] = transform
        meta["tags"] = [str(x) for x in tags if str(x).strip()]
        meta["fem_role"] = str(meta.get("fem_role", "solid") or "solid")
        meta["exclude_from_solve"] = bool(meta.get("exclude_from_solve", False))
        meta["notes"] = str(meta.get("notes", "") or "")

    def _normalize_layers(self) -> None:
        if not isinstance(self.layers, dict):
            self.layers = {}
        self._ensure_layer("Default", color="#86b6f6", visible=True, locked=False)
        for obj in self.objects.values():
            self._apply_layer_defaults(obj)

    # ---------------- state ----------------
    def serialize_state(self) -> dict:
        return {
            "objects": {oid: obj.to_dict() for oid, obj in self.objects.items()},
            "selection": list(self.selection),
            "markers": {mid: mk.to_dict() for mid, mk in self.markers.items()},
            "measurements": copy.deepcopy(self.measurements),
            "boundaries": copy.deepcopy(self.boundaries),
            "layers": copy.deepcopy(self.layers),
            "mesh_quality": dict(self._default_mesh_quality),
            "tool_mode": str(self.tool_mode),
            "fem_state": self.fem.serialize(),
        }

    def restore_state(self, state: dict, emit: bool = True) -> None:
        self.objects = {str(k): SceneObject.from_dict(v) for k, v in dict(state.get("objects", {})).items()}
        self.selection = [str(x) for x in state.get("selection", []) if str(x) in self.objects]
        self.markers = {str(k): Marker.from_dict(v) for k, v in dict(state.get("markers", {})).items()}
        self.measurements = copy.deepcopy(state.get("measurements", []))
        self.boundaries = {str(k): dict(v) for k, v in dict(state.get("boundaries", {})).items()}
        self.layers = copy.deepcopy(state.get("layers", {})) if isinstance(state.get("layers", {}), dict) else {}
        self._default_mesh_quality = self._coerce_quality(state.get("mesh_quality", self._default_mesh_quality))
        self._normalize_layers()
        try:
            self.fem.restore(state.get("fem_state", {}))
        except Exception:
            self.fem.restore({})
        self.tool_mode = str(state.get("tool_mode", "Select"))
        if emit:
            self.emit("scene_changed", {"reason": "restore"})

    def execute(self, label: str, action: Callable[["SceneEngine"], None], kind: str = "") -> None:
        cmd = build_command(kind=kind, label=str(label), action=action) if kind else SnapshotCommand(label=str(label), action=action)
        self.command_stack.execute(cmd, self)
        self.emit("scene_changed", {"reason": "command", "label": str(label)})

    def undo(self) -> bool:
        ok = self.command_stack.undo(self)
        if ok:
            self.emit("scene_changed", {"reason": "undo"})
        return ok

    def redo(self) -> bool:
        ok = self.command_stack.redo(self)
        if ok:
            self.emit("scene_changed", {"reason": "redo"})
        return ok

    # ---------------- object operations ----------------
    def add_object(self, obj: SceneObject, undoable: bool = True) -> str:
        oid = str(obj.id)

        def _add(_eng: "SceneEngine"):
            item = obj.clone()
            _eng._apply_layer_defaults(item)
            _eng.objects[oid] = item
            _eng.selection = [oid]

        if undoable:
            self.execute(f"Add {obj.name}", _add)
        else:
            _add(self)
            self.emit("scene_changed", {"reason": "add"})
        return oid

    def add_mesh_object(
        self,
        mesh: MeshData,
        name: str = "Object",
        source: str = "Local",
        color: str = "#86b6f6",
        opacity: float = 0.85,
        layer: str = "Default",
        undoable: bool = True,
    ) -> str:
        obj = SceneObject.create(name=name, mesh=mesh, source=source)
        obj.color = str(color)
        obj.opacity = float(max(0.05, min(1.0, opacity)))
        obj.meta["layer"] = self._clean_layer_name(layer)
        return self.add_object(obj, undoable=undoable)

    def delete_objects(self, ids: Sequence[str]) -> None:
        ids = [str(i) for i in ids if str(i) in self.objects]
        if not ids:
            return

        def _delete(_eng: "SceneEngine"):
            for oid in ids:
                _eng.objects.pop(oid, None)
            _eng.selection = [x for x in _eng.selection if x in _eng.objects]
            for mk in _eng.markers.values():
                if mk.target_object_id in ids:
                    mk.target_object_id = None
            removed = []
            for bid, row in list(_eng.boundaries.items()):
                targets = [str(x) for x in row.get("targets", []) if str(x) not in ids and str(x) in _eng.objects]
                if targets:
                    row["targets"] = targets
                else:
                    removed.append(str(bid))
            for bid in removed:
                _eng.boundaries.pop(str(bid), None)

        self.execute(f"Delete {len(ids)} object(s)", _delete, kind="delete")

    def duplicate_objects(self, ids: Sequence[str]) -> List[str]:
        src_ids = [str(i) for i in ids if str(i) in self.objects]
        if not src_ids:
            return []
        created: List[str] = []

        def _dup(_eng: "SceneEngine"):
            used_names = {o.name for o in _eng.objects.values()}
            out = []
            for oid in src_ids:
                src = _eng.objects[oid]
                dup = src.clone()
                dup.id = str(np.random.randint(1, 10**9)) + "_" + src.id[-8:]
                base = f"{src.name}_copy"
                name = base
                idx = 2
                while name in used_names:
                    name = f"{base}_{idx}"
                    idx += 1
                used_names.add(name)
                dup.name = name
                dup.mesh = transform_mesh(dup.mesh, tx=0.03, ty=0.03, tz=0.0)
                dup.meta.pop("boundary_ids", None)
                for key in ("import_asset_id", "import_asset_name", "import_asset_path", "import_body_index", "import_body_count"):
                    dup.meta.pop(key, None)
                _eng._apply_layer_defaults(dup)
                _eng.objects[dup.id] = dup
                out.append(dup.id)
            _eng.selection = list(out)
            created[:] = out

        self.execute(f"Duplicate {len(src_ids)} object(s)", _dup)
        return created

    def rename_object(self, obj_id: str, new_name: str) -> None:
        oid = str(obj_id)
        if oid not in self.objects:
            return
        token = str(new_name or "").strip()
        if not token:
            return

        def _rename(_eng: "SceneEngine"):
            _eng.objects[oid].name = token
            if isinstance(_eng.objects[oid].meta, dict):
                _eng.objects[oid].meta["name"] = str(token)

        self.execute("Rename object", _rename, kind="rename")

    def set_visibility(self, ids: Sequence[str], visible: bool) -> None:
        ids = [str(i) for i in ids if str(i) in self.objects]
        if not ids:
            return

        def _set(_eng: "SceneEngine"):
            for oid in ids:
                _eng.objects[oid].visible = bool(visible)
                if isinstance(_eng.objects[oid].meta, dict):
                    _eng.objects[oid].meta["visible"] = bool(visible)

        self.execute("Set visibility", _set, kind="visibility")

    def set_lock(self, ids: Sequence[str], locked: bool) -> None:
        ids = [str(i) for i in ids if str(i) in self.objects]
        if not ids:
            return

        def _set(_eng: "SceneEngine"):
            for oid in ids:
                _eng.objects[oid].locked = bool(locked)
                if isinstance(_eng.objects[oid].meta, dict):
                    _eng.objects[oid].meta["locked"] = bool(locked)

        self.execute("Set lock", _set)

    def set_color_opacity(self, ids: Sequence[str], color: Optional[str] = None, opacity: Optional[float] = None) -> None:
        ids = [str(i) for i in ids if str(i) in self.objects]
        if not ids:
            return

        def _set(_eng: "SceneEngine"):
            for oid in ids:
                if color is not None:
                    _eng.objects[oid].color = str(color)
                    if isinstance(_eng.objects[oid].meta, dict):
                        _eng.objects[oid].meta["color"] = str(color)
                if opacity is not None:
                    _eng.objects[oid].opacity = float(max(0.05, min(1.0, float(opacity))))

        self.execute("Set object style", _set)

    # ---------------- face/edge adjustments ----------------
    @staticmethod
    def _safe_face_index(mesh: MeshData, face_index: int) -> int:
        try:
            idx = int(face_index)
        except Exception:
            return -1
        faces = np.asarray(mesh.faces, dtype=int)
        if faces.ndim != 2 or faces.shape[1] != 3:
            return -1
        if idx < 0 or idx >= int(faces.shape[0]):
            return -1
        return idx

    @staticmethod
    def _face_normal(mesh: MeshData, face_index: int) -> np.ndarray:
        faces = np.asarray(mesh.faces, dtype=int)
        verts = np.asarray(mesh.vertices, dtype=float)
        tri = faces[int(face_index)]
        p0, p1, p2 = verts[int(tri[0])], verts[int(tri[1])], verts[int(tri[2])]
        n = np.cross(p1 - p0, p2 - p0)
        nn = float(np.linalg.norm(n))
        if nn <= 1e-12:
            return np.array([0.0, 0.0, 1.0], dtype=float)
        return (n / nn).astype(float)

    def face_info(self, obj_id: str, face_index: int) -> dict:
        oid = str(obj_id)
        if oid not in self.objects:
            raise RuntimeError("Object not found.")
        obj = self.objects[oid]
        idx = self._safe_face_index(obj.mesh, face_index)
        if idx < 0:
            raise RuntimeError("Face index out of range.")
        tri = np.asarray(obj.mesh.faces[idx], dtype=int)
        verts = np.asarray(obj.mesh.vertices, dtype=float)
        points = np.asarray([verts[int(tri[0])], verts[int(tri[1])], verts[int(tri[2])]], dtype=float)
        c = np.mean(points, axis=0)
        n = self._face_normal(obj.mesh, idx)
        return {
            "object_id": oid,
            "face_index": int(idx),
            "vertex_ids": [int(tri[0]), int(tri[1]), int(tri[2])],
            "centroid": [float(c[0]), float(c[1]), float(c[2])],
            "normal": [float(n[0]), float(n[1]), float(n[2])],
        }

    def pick_edge_from_face(self, obj_id: str, face_index: int, point: Optional[Sequence[float]] = None) -> Optional[Tuple[int, int]]:
        oid = str(obj_id)
        if oid not in self.objects:
            return None
        obj = self.objects[oid]
        idx = self._safe_face_index(obj.mesh, face_index)
        if idx < 0:
            return None
        tri = np.asarray(obj.mesh.faces[idx], dtype=int)
        edges = [
            (int(tri[0]), int(tri[1])),
            (int(tri[1]), int(tri[2])),
            (int(tri[2]), int(tri[0])),
        ]
        if point is None:
            a, b = edges[0]
            return (min(a, b), max(a, b))
        p = np.asarray(point, dtype=float).reshape(-1)
        if p.size != 3:
            a, b = edges[0]
            return (min(a, b), max(a, b))
        verts = np.asarray(obj.mesh.vertices, dtype=float)
        best = None
        best_d = float("inf")
        for a, b in edges:
            pa = verts[int(a)]
            pb = verts[int(b)]
            ab = pb - pa
            denom = float(np.dot(ab, ab))
            if denom <= 1e-12:
                d = float(np.linalg.norm(p - pa))
            else:
                t = float(np.dot(p - pa, ab) / denom)
                t = max(0.0, min(1.0, t))
                proj = pa + t * ab
                d = float(np.linalg.norm(p - proj))
            if d < best_d:
                best_d = d
                best = (int(a), int(b))
        if best is None:
            return None
        return (min(best[0], best[1]), max(best[0], best[1]))

    def adjust_face_offset(self, obj_id: str, face_index: int, offset: float) -> bool:
        oid = str(obj_id)
        if oid not in self.objects:
            return False
        idx = self._safe_face_index(self.objects[oid].mesh, face_index)
        if idx < 0:
            return False
        delta = float(offset)
        if abs(delta) <= 1e-12:
            return True

        def _apply(_eng: "SceneEngine"):
            obj = _eng.objects.get(oid)
            if obj is None or obj.locked:
                return
            mesh = obj.mesh.clone()
            faces = np.asarray(mesh.faces, dtype=int)
            verts = np.asarray(mesh.vertices, dtype=float)
            tri = faces[int(idx)]
            n = _eng._face_normal(mesh, idx)
            for vid in [int(tri[0]), int(tri[1]), int(tri[2])]:
                verts[vid] = verts[vid] + n * delta
            mesh.vertices = verts
            obj.mesh = mesh

        self.execute("Adjust face offset", _apply, kind="transform")
        return True

    def adjust_edge_offset(self, obj_id: str, edge: Sequence[int], offset: float) -> bool:
        oid = str(obj_id)
        if oid not in self.objects:
            return False
        try:
            a = int(edge[0])
            b = int(edge[1])
        except Exception:
            return False
        if a == b:
            return False
        delta = float(offset)
        if abs(delta) <= 1e-12:
            return True

        obj = self.objects[oid]
        verts0 = np.asarray(obj.mesh.vertices, dtype=float)
        if a < 0 or b < 0 or a >= int(verts0.shape[0]) or b >= int(verts0.shape[0]):
            return False

        def _apply(_eng: "SceneEngine"):
            obj2 = _eng.objects.get(oid)
            if obj2 is None or obj2.locked:
                return
            mesh = obj2.mesh.clone()
            faces = np.asarray(mesh.faces, dtype=int)
            verts = np.asarray(mesh.vertices, dtype=float)
            normals: List[np.ndarray] = []
            for fi, tri in enumerate(faces):
                st = {int(tri[0]), int(tri[1]), int(tri[2])}
                if a in st and b in st:
                    normals.append(_eng._face_normal(mesh, fi))
            if normals:
                n = np.mean(np.asarray(normals, dtype=float), axis=0)
                nn = float(np.linalg.norm(n))
                if nn > 1e-12:
                    n = n / nn
                else:
                    n = np.array([0.0, 0.0, 1.0], dtype=float)
            else:
                p0 = verts[int(a)]
                p1 = verts[int(b)]
                d = p1 - p0
                z = np.array([0.0, 0.0, 1.0], dtype=float)
                n = np.cross(d, z)
                nn = float(np.linalg.norm(n))
                if nn <= 1e-12:
                    n = np.array([1.0, 0.0, 0.0], dtype=float)
                else:
                    n = n / nn
            verts[int(a)] = verts[int(a)] + n * delta
            verts[int(b)] = verts[int(b)] + n * delta
            mesh.vertices = verts
            obj2.mesh = mesh

        self.execute("Adjust edge offset", _apply, kind="transform")
        return True

    # ---------------- layers ----------------
    def layer_rows(self) -> List[dict]:
        self._normalize_layers()
        counts: Dict[str, int] = {}
        for obj in self.objects.values():
            layer = self._clean_layer_name(str(obj.meta.get("layer", "Default") or "Default"))
            counts[layer] = counts.get(layer, 0) + 1
        rows = []
        for key, row in self.layers.items():
            item = dict(row)
            item["name"] = self._clean_layer_name(str(item.get("name", key) or key))
            item["count"] = int(counts.get(item["name"], 0))
            rows.append(item)
        rows.sort(key=lambda r: (0 if str(r.get("name", "")) == "Default" else 1, str(r.get("name", "")).lower()))
        return rows

    def object_layer(self, obj_id: str) -> str:
        oid = str(obj_id)
        if oid not in self.objects:
            return "Default"
        self._apply_layer_defaults(self.objects[oid])
        return str(self.objects[oid].meta.get("layer", "Default") or "Default")

    def set_objects_layer(self, ids: Sequence[str], layer_name: str) -> None:
        targets = [str(i) for i in ids if str(i) in self.objects]
        if not targets:
            return
        token = self._clean_layer_name(layer_name)
        row = self._ensure_layer(token)

        def _apply(_eng: "SceneEngine"):
            _eng._ensure_layer(token, color=str(row.get("color", "#86b6f6")))
            for oid in targets:
                obj = _eng.objects.get(oid)
                if obj is None:
                    continue
                obj.meta["layer"] = str(token)
                obj.meta["layer_color"] = str(_eng.layers[token].get("color", "#86b6f6"))

        self.execute(f"Assign layer {token}", _apply)

    def rename_layer(self, old_name: str, new_name: str) -> bool:
        old_token = self._clean_layer_name(old_name)
        if old_token not in self.layers:
            return False
        new_token = self._clean_layer_name(new_name)
        if not new_token:
            return False
        if new_token == old_token:
            return True
        if new_token in self.layers:
            raise RuntimeError(f"Layer already exists: {new_token}")
        old_row = dict(self.layers.get(old_token, {}))

        def _apply(_eng: "SceneEngine"):
            _eng.layers.pop(old_token, None)
            _eng.layers[new_token] = {
                "name": str(new_token),
                "color": str(old_row.get("color", "#86b6f6")),
                "visible": bool(old_row.get("visible", True)),
                "locked": bool(old_row.get("locked", False)),
            }
            for obj in _eng.objects.values():
                layer = _eng._clean_layer_name(str(obj.meta.get("layer", "Default") or "Default"))
                if layer == old_token:
                    obj.meta["layer"] = str(new_token)

        self.execute(f"Rename layer {old_token}", _apply)
        return True

    def delete_layer(self, layer_name: str, fallback_layer: str = "Default") -> bool:
        token = self._clean_layer_name(layer_name)
        if token == "Default":
            return False
        if token not in self.layers:
            return False
        fallback = self._clean_layer_name(fallback_layer)
        self._ensure_layer(fallback)

        def _apply(_eng: "SceneEngine"):
            _eng.layers.pop(token, None)
            for obj in _eng.objects.values():
                layer = _eng._clean_layer_name(str(obj.meta.get("layer", "Default") or "Default"))
                if layer == token:
                    obj.meta["layer"] = str(fallback)
                    obj.meta["layer_color"] = str(_eng.layers[fallback].get("color", "#86b6f6"))

        self.execute(f"Delete layer {token}", _apply)
        return True

    def set_layer_visibility(self, layer_name: str, visible: bool) -> int:
        token = self._clean_layer_name(layer_name)
        if token not in self.layers:
            return 0
        targets = [oid for oid, obj in self.objects.items() if self._clean_layer_name(str(obj.meta.get("layer", "Default") or "Default")) == token]

        def _apply(_eng: "SceneEngine"):
            row = dict(_eng.layers.get(token, {}))
            row["visible"] = bool(visible)
            row.setdefault("name", token)
            row.setdefault("color", "#86b6f6")
            row.setdefault("locked", False)
            _eng.layers[token] = row
            for oid in targets:
                if oid in _eng.objects:
                    _eng.objects[oid].visible = bool(visible)

        self.execute(f"Layer visibility {token}", _apply, kind="visibility")
        return len(targets)

    def set_layer_color(self, layer_name: str, color: str, apply_to_objects: bool = False) -> int:
        token = self._clean_layer_name(layer_name)
        if token not in self.layers:
            return 0
        out_color = str(color or "#86b6f6")
        targets = [oid for oid, obj in self.objects.items() if self._clean_layer_name(str(obj.meta.get("layer", "Default") or "Default")) == token]

        def _apply(_eng: "SceneEngine"):
            row = dict(_eng.layers.get(token, {}))
            row["color"] = out_color
            row.setdefault("name", token)
            row.setdefault("visible", True)
            row.setdefault("locked", False)
            _eng.layers[token] = row
            for oid in targets:
                obj = _eng.objects.get(oid)
                if obj is None:
                    continue
                obj.meta["layer_color"] = out_color
                if apply_to_objects:
                    obj.color = out_color

        self.execute(f"Layer color {token}", _apply)
        return len(targets)

    # ---------------- boundaries ----------------
    def boundary_rows(self) -> List[dict]:
        rows = [dict(v) for v in self.boundaries.values() if isinstance(v, dict)]
        rows.sort(key=lambda r: (str(r.get("created_utc", "")), str(r.get("id", ""))))
        return rows

    def boundaries_for_object(self, obj_id: str) -> List[dict]:
        oid = str(obj_id)
        out = []
        for row in self.boundary_rows():
            targets = [str(x) for x in row.get("targets", [])]
            if oid in targets:
                out.append(row)
        return out

    def apply_boundary(
        self,
        targets: Sequence[str],
        boundary_type: str,
        params: Optional[dict] = None,
        *,
        name: str = "",
        picked_point: Optional[Sequence[float]] = None,
        source: str = "ui",
    ) -> str:
        ids = [str(i) for i in targets if str(i) in self.objects]
        if not ids:
            raise RuntimeError("Boundary application requires at least one target object.")
        token = str(boundary_type or "").strip().lower()
        if not token:
            raise RuntimeError("Boundary type is required.")
        p3 = None
        if picked_point is not None:
            arr = np.asarray(picked_point, dtype=float).reshape(-1)
            if arr.size == 3:
                p3 = [float(arr[0]), float(arr[1]), float(arr[2])]
        bid = str(uuid.uuid4())
        row = {
            "id": bid,
            "name": str(name or f"{token}_{len(self.boundaries) + 1}"),
            "type": token,
            "targets": list(ids),
            "params": dict(params or {}),
            "picked_point": p3,
            "source": str(source or "ui"),
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }

        def _apply(_eng: "SceneEngine"):
            _eng.boundaries[bid] = dict(row)
            for oid in ids:
                obj = _eng.objects.get(oid)
                if obj is None:
                    continue
                refs = [str(x) for x in obj.meta.get("boundary_ids", [])] if isinstance(obj.meta.get("boundary_ids", []), list) else []
                if bid not in refs:
                    refs.append(bid)
                obj.meta["boundary_ids"] = refs

        self.execute(f"Apply boundary {token}", _apply, kind="boundary")
        return bid

    def remove_boundary(self, boundary_id: str) -> bool:
        bid = str(boundary_id)
        if bid not in self.boundaries:
            return False

        def _apply(_eng: "SceneEngine"):
            _eng.boundaries.pop(bid, None)
            for obj in _eng.objects.values():
                refs = [str(x) for x in obj.meta.get("boundary_ids", [])] if isinstance(obj.meta.get("boundary_ids", []), list) else []
                if bid in refs:
                    obj.meta["boundary_ids"] = [x for x in refs if x != bid]

        self.execute("Remove boundary", _apply, kind="boundary")
        return True

    def clear_boundaries(self, targets: Sequence[str] = ()) -> int:
        ids = [str(i) for i in targets if str(i) in self.objects]
        if ids:
            remove_ids = [str(bid) for bid, row in self.boundaries.items() if set([str(x) for x in row.get("targets", [])]).intersection(set(ids))]
        else:
            remove_ids = [str(bid) for bid in self.boundaries.keys()]
        if not remove_ids:
            return 0

        def _apply(_eng: "SceneEngine"):
            for bid in remove_ids:
                _eng.boundaries.pop(str(bid), None)
            for obj in _eng.objects.values():
                refs = [str(x) for x in obj.meta.get("boundary_ids", [])] if isinstance(obj.meta.get("boundary_ids", []), list) else []
                if refs:
                    obj.meta["boundary_ids"] = [x for x in refs if x not in remove_ids]

        self.execute("Clear boundaries", _apply, kind="boundary")
        return len(remove_ids)

    def transform_objects(
        self,
        ids: Sequence[str],
        tx: float = 0.0,
        ty: float = 0.0,
        tz: float = 0.0,
        rx_deg: float = 0.0,
        ry_deg: float = 0.0,
        rz_deg: float = 0.0,
        scale: float = 1.0,
    ) -> None:
        ids = [str(i) for i in ids if str(i) in self.objects]
        if not ids:
            return

        def _apply(_eng: "SceneEngine"):
            for oid in ids:
                obj = _eng.objects[oid]
                if obj.locked:
                    continue
                kid = self._kernel_obj_id(obj)
                if kid:
                    try:
                        self.kernel.transform(
                            kid,
                            {
                                "tx": tx,
                                "ty": ty,
                                "tz": tz,
                                "rx_deg": rx_deg,
                                "ry_deg": ry_deg,
                                "rz_deg": rz_deg,
                                "scale": scale,
                            },
                        )
                        self._sync_mesh_from_kernel(obj)
                        continue
                    except Exception as exc:
                        try:
                            self.logger.warning("Kernel transform failed for %s; using mesh fallback: %s", oid, exc)
                        except Exception:
                            pass
                obj.mesh = transform_mesh(
                    obj.mesh,
                    tx=tx,
                    ty=ty,
                    tz=tz,
                    rx_deg=rx_deg,
                    ry_deg=ry_deg,
                    rz_deg=rz_deg,
                    scale=scale,
                )
                if isinstance(obj.meta, dict):
                    tf = dict(obj.meta.get("transform", {})) if isinstance(obj.meta.get("transform", {}), dict) else {}
                    tf["tx"] = float(float(tf.get("tx", 0.0)) + float(tx))
                    tf["ty"] = float(float(tf.get("ty", 0.0)) + float(ty))
                    tf["tz"] = float(float(tf.get("tz", 0.0)) + float(tz))
                    tf["rx_deg"] = float(float(tf.get("rx_deg", 0.0)) + float(rx_deg))
                    tf["ry_deg"] = float(float(tf.get("ry_deg", 0.0)) + float(ry_deg))
                    tf["rz_deg"] = float(float(tf.get("rz_deg", 0.0)) + float(rz_deg))
                    tf["sx"] = float(float(tf.get("sx", 1.0)) * float(scale))
                    tf["sy"] = float(float(tf.get("sy", 1.0)) * float(scale))
                    tf["sz"] = float(float(tf.get("sz", 1.0)) * float(scale))
                    obj.meta["transform"] = tf

        self.execute("Transform object(s)", _apply, kind="transform")

    # ---------------- primitives ----------------
    @staticmethod
    def _kernel_obj_id(obj: SceneObject) -> str:
        try:
            return str(obj.meta.get("kernel_obj_id", "") or "")
        except Exception:
            return ""

    def _bind_kernel_object(self, obj: SceneObject, kernel_obj_id: str, kernel_kind: str) -> None:
        obj.meta["kernel_obj_id"] = str(kernel_obj_id)
        obj.meta["kernel_provider"] = str(self._kernel_provider_name)
        obj.meta["kernel_kind"] = str(kernel_kind)

    def _sync_mesh_from_kernel(self, obj: SceneObject, quality: Optional[dict] = None) -> None:
        kid = self._kernel_obj_id(obj)
        if not kid:
            return
        q = self._coerce_quality(quality or obj.meta.get("triangulation_quality", self._default_mesh_quality))
        tri = self.kernel.triangulate(kid, quality=q)
        verts = np.asarray(tri.get("vertices", []), dtype=float)
        faces = np.asarray(tri.get("faces", []), dtype=int)
        if verts.ndim != 2 or verts.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
            raise RuntimeError("Kernel triangulation returned an invalid mesh payload.")
        obj.mesh = MeshData(vertices=verts, faces=faces)
        obj.meta["triangulation_quality"] = dict(q)

    def create_primitive(self, kind: str, params: dict, name: str = "Primitive") -> str:
        token = str(kind or "").strip().lower()
        params = dict(params or {})
        quality = self._coerce_quality(self._default_mesh_quality)
        kernel_candidates = set(self.backend_capabilities().get("primitive_kinds", []) or [])

        # Prefer the geometric kernel when available and compatible.
        if token in kernel_candidates:
            try:
                kernel_id = self.kernel.create_primitive(token, params)
                tri = self.kernel.triangulate(kernel_id, quality=quality)
                mesh = MeshData(
                    vertices=np.asarray(tri.get("vertices", []), dtype=float),
                    faces=np.asarray(tri.get("faces", []), dtype=int),
                )
                obj = SceneObject.create(name=name, mesh=mesh, source=f"Kernel:{self._kernel_provider_name}:{token}")
                self._bind_kernel_object(obj, kernel_id, token)
                obj.meta["triangulation_quality"] = dict(quality)
                self._apply_layer_defaults(obj)

                def _add(_eng: "SceneEngine"):
                    _eng.objects[obj.id] = obj.clone()
                    _eng.selection = [obj.id]

                self.execute(f"Create {token}", _add, kind="create_primitive")
                return obj.id
            except Exception as exc:
                try:
                    self.logger.warning("Kernel primitive failed, using local fallback (%s): %s", token, exc)
                except Exception:
                    pass

        # Local fallback always available.
        if token == "box":
            mesh = create_box(params.get("width", 1.0), params.get("depth", 1.0), params.get("height", 1.0), params.get("center", (0, 0, 0)))
        elif token == "cylinder":
            mesh = create_cylinder(params.get("radius", 0.5), params.get("height", 1.0), params.get("segments", 32), params.get("center", (0, 0, 0)))
        elif token == "sphere":
            mesh = create_sphere(params.get("radius", 0.5), params.get("theta_res", 24), params.get("phi_res", 24), params.get("center", (0, 0, 0)))
        elif token == "cone":
            mesh = create_cone(params.get("radius", 0.5), params.get("height", 1.0), params.get("segments", 32), params.get("center", (0, 0, 0)))
        elif token == "tube":
            mesh = create_tube(
                params.get("outer_radius", params.get("radius", 0.6)),
                params.get("inner_radius", params.get("inner", 0.3)),
                params.get("height", 1.0),
                params.get("segments", 48),
                params.get("center", (0, 0, 0)),
            )
        elif token == "plane":
            mesh = create_plane(params.get("width", 1.0), params.get("depth", 1.0), params.get("center", (0, 0, 0)), params.get("axis", "z"))
        else:
            raise RuntimeError(f"Unknown primitive kind: {kind}")
        obj = SceneObject.create(name=name, mesh=mesh, source=f"Primitive:{token}")
        self._apply_layer_defaults(obj)

        def _add_local(_eng: "SceneEngine"):
            _eng.objects[obj.id] = obj.clone()
            _eng.selection = [obj.id]

        self.execute(f"Create {token}", _add_local, kind="create_primitive")
        return obj.id

    def import_model(
        self,
        path: str,
        fmt: Optional[str] = None,
        name_hint: str = "",
        *,
        layer_name: str = "",
        triangulation_quality: Optional[dict] = None,
    ) -> List[str]:
        src = str(path or "").strip()
        if not src or not os.path.isfile(src):
            raise RuntimeError(f"Input path not found: {src}")

        token = str(fmt or "").strip().lower().lstrip(".")
        if not token:
            token = os.path.splitext(src)[1].lower().lstrip(".")
        imported_ids: List[str] = []
        use_kernel = token in set(self.backend_capabilities().get("import_formats", []) or [])
        base_name = str(name_hint or os.path.splitext(os.path.basename(src))[0]).strip() or "imported_model"
        quality = self._coerce_quality(triangulation_quality or self._default_mesh_quality)
        target_layer = self._clean_layer_name(layer_name or base_name)
        self._ensure_layer(target_layer)
        asset_id = f"{base_name}_{uuid.uuid4().hex[:8]}"

        def _apply(_eng: "SceneEngine"):
            created: List[str] = []
            if use_kernel:
                kernel_ids = self.kernel.import_model(src, fmt=token)
                total = int(len(kernel_ids))
                for idx, kid in enumerate(kernel_ids, start=1):
                    tri = self.kernel.triangulate(kid, quality=quality)
                    mesh = MeshData(
                        vertices=np.asarray(tri.get("vertices", []), dtype=float),
                        faces=np.asarray(tri.get("faces", []), dtype=int),
                    )
                    body_name = base_name if total <= 1 else f"{base_name}_body_{idx:02d}"
                    obj = SceneObject.create(
                        name=body_name,
                        mesh=mesh,
                        source=f"KernelImport:{token}",
                    )
                    self._bind_kernel_object(obj, kid, f"import:{token}")
                    obj.meta["import_asset_id"] = str(asset_id)
                    obj.meta["import_asset_name"] = str(base_name)
                    obj.meta["import_asset_path"] = str(src)
                    obj.meta["import_body_index"] = int(idx)
                    obj.meta["import_body_count"] = int(total)
                    obj.meta["triangulation_quality"] = dict(quality)
                    _eng._apply_layer_defaults(obj, preferred=target_layer)
                    _eng.objects[obj.id] = obj
                    created.append(obj.id)
            else:
                try:
                    import pyvista as pv
                except Exception as exc:  # pragma: no cover - dependency/environment gate
                    raise RuntimeError(f"PyVista required for fallback import: {exc}") from exc
                raw = pv.read(src)
                meshes = []
                if isinstance(raw, pv.MultiBlock):
                    for block in raw:
                        if block is None:
                            continue
                        try:
                            poly = block.extract_surface().triangulate().clean()
                        except Exception:
                            continue
                        if int(getattr(poly, "n_points", 0)) <= 0 or int(getattr(poly, "n_cells", 0)) <= 0:
                            continue
                        meshes.append(poly)
                else:
                    poly = raw.triangulate().clean()
                    if int(getattr(poly, "n_points", 0)) > 0 and int(getattr(poly, "n_cells", 0)) > 0:
                        meshes.append(poly)

                if not meshes:
                    raise RuntimeError("Imported mesh is empty.")

                total = int(len(meshes))
                for idx, poly in enumerate(meshes, start=1):
                    verts = np.asarray(poly.points, dtype=float)
                    faces_raw = np.asarray(poly.faces)
                    if faces_raw.size == 0:
                        continue
                    faces = faces_raw.reshape(-1, 4)[:, 1:4].astype(int)
                    if verts.size == 0 or faces.size == 0:
                        continue
                    mesh = MeshData(vertices=verts, faces=faces)
                    body_name = base_name if total <= 1 else f"{base_name}_body_{idx:02d}"
                    obj = SceneObject.create(
                        name=body_name,
                        mesh=mesh,
                        source=f"Import:{token or 'mesh'}",
                    )
                    obj.meta["import_asset_id"] = str(asset_id)
                    obj.meta["import_asset_name"] = str(base_name)
                    obj.meta["import_asset_path"] = str(src)
                    obj.meta["import_body_index"] = int(idx)
                    obj.meta["import_body_count"] = int(total)
                    obj.meta["triangulation_quality"] = dict(quality)
                    _eng._apply_layer_defaults(obj, preferred=target_layer)
                    _eng.objects[obj.id] = obj
                    created.append(obj.id)

                if not created:
                    raise RuntimeError("Imported mesh is empty after triangulation.")

            _eng.selection = list(created)
            imported_ids[:] = list(created)

        self.execute(f"Import {os.path.basename(src)}", _apply, kind="import")
        return imported_ids

    def retessellate_objects(self, ids: Sequence[str], quality: Optional[dict] = None) -> int:
        targets = [str(i) for i in ids if str(i) in self.objects and bool(self._kernel_obj_id(self.objects[str(i)]))]
        if not targets:
            return 0
        q = self._coerce_quality(quality or self._default_mesh_quality)
        changed: List[str] = []

        def _apply(_eng: "SceneEngine"):
            for oid in targets:
                obj = _eng.objects.get(oid)
                if obj is None:
                    continue
                try:
                    _eng._sync_mesh_from_kernel(obj, quality=q)
                    changed.append(oid)
                except Exception:
                    continue

        self.execute(f"Retessellate {len(targets)} object(s)", _apply)
        return len(changed)

    def export_model(self, obj_ids: Sequence[str], path: str, fmt: Optional[str] = None) -> str:
        ids = [str(i) for i in obj_ids if str(i) in self.objects]
        if not ids:
            raise RuntimeError("No objects selected for export.")

        out_path = str(path or "").strip()
        if not out_path:
            raise RuntimeError("Output path is empty.")

        token = str(fmt or "").strip().lower().lstrip(".")
        if not token:
            token = os.path.splitext(out_path)[1].lower().lstrip(".")
        if not token:
            raise RuntimeError("Unable to infer export format.")

        kernel_ok = token in set(self.backend_capabilities().get("export_formats", []) or [])
        kernel_ids = [self._kernel_obj_id(self.objects[oid]) for oid in ids]
        if kernel_ok and all(bool(kid) for kid in kernel_ids):
            return self.kernel.export_model(kernel_ids, out_path, fmt=token)

        try:
            import pyvista as pv
        except Exception as exc:  # pragma: no cover - dependency/environment gate
            raise RuntimeError(f"PyVista required for fallback export: {exc}") from exc

        if len(ids) == 1:
            obj = self.objects[ids[0]]
            faces = np.asarray(obj.mesh.faces, dtype=int)
            verts = np.asarray(obj.mesh.vertices, dtype=float)
            ff = np.empty((faces.shape[0], 4), dtype=np.int64)
            ff[:, 0] = 3
            ff[:, 1:] = faces.astype(np.int64)
            poly = pv.PolyData(verts, ff.reshape(-1))
            poly.save(out_path)
            return out_path

        out_dir = os.path.abspath(out_path)
        os.makedirs(out_dir, exist_ok=True)
        for oid in ids:
            obj = self.objects[oid]
            faces = np.asarray(obj.mesh.faces, dtype=int)
            verts = np.asarray(obj.mesh.vertices, dtype=float)
            ff = np.empty((faces.shape[0], 4), dtype=np.int64)
            ff[:, 0] = 3
            ff[:, 1:] = faces.astype(np.int64)
            poly = pv.PolyData(verts, ff.reshape(-1))
            poly.save(os.path.join(out_dir, f"{obj.name}.{token}"))
        return out_dir

    # ---------------- booleans ----------------
    def boolean_union(self, ids: Sequence[str], tolerance: float = 1e-6) -> Optional[str]:
        sel = [str(i) for i in ids if str(i) in self.objects]
        if len(sel) < 2:
            raise RuntimeError("Union requires at least 2 selected objects.")
        for oid in sel:
            ensure_boolean_ready(self.objects[oid].mesh)

        def _apply(_eng: "SceneEngine"):
            first = self.objects[sel[0]]
            kernel_ids = [self._kernel_obj_id(self.objects[oid]) for oid in sel]
            kernel_ready = all(bool(k) for k in kernel_ids)
            if kernel_ready:
                try:
                    out_kernel_id = self.kernel.boolean("union", kernel_ids[0], kernel_ids[1:])
                    tri = self.kernel.triangulate(out_kernel_id, quality={"deflection": 0.2})
                    out = MeshData(vertices=np.asarray(tri.get("vertices", []), dtype=float), faces=np.asarray(tri.get("faces", []), dtype=int))
                    out_obj = SceneObject.create(name=f"{first.name}_union", mesh=out, source=f"Kernel:Boolean:Union")
                    self._bind_kernel_object(out_obj, out_kernel_id, "boolean:union")
                    out_obj.color = first.color
                    out_obj.opacity = first.opacity
                    _eng._apply_layer_defaults(out_obj, preferred=str(first.meta.get("layer", "Default") or "Default"))
                    for oid in sel:
                        _eng.objects.pop(oid, None)
                    _eng.objects[out_obj.id] = out_obj
                    _eng.selection = [out_obj.id]
                    return
                except Exception as exc:
                    try:
                        self.logger.warning("Kernel union failed; using mesh fallback: %s", exc)
                    except Exception:
                        pass

            meshes = [self.objects[oid].mesh for oid in sel]
            out = boolean_union(meshes, tolerance=tolerance)
            out_obj = SceneObject.create(name=f"{first.name}_union", mesh=out, source="Boolean:Union")
            out_obj.color = first.color
            out_obj.opacity = first.opacity
            _eng._apply_layer_defaults(out_obj, preferred=str(first.meta.get("layer", "Default") or "Default"))
            for oid in sel:
                _eng.objects.pop(oid, None)
            _eng.objects[out_obj.id] = out_obj
            _eng.selection = [out_obj.id]

        self.execute("Boolean union", _apply, kind="boolean")
        return self.selection[0] if self.selection else None

    def boolean_subtract(self, primary_id: str, cutter_ids: Sequence[str], tolerance: float = 1e-6) -> Optional[str]:
        pid = str(primary_id)
        cutters = [str(i) for i in cutter_ids if str(i) in self.objects and str(i) != pid]
        if pid not in self.objects or not cutters:
            raise RuntimeError("Subtract requires a primary and at least one cutter.")
        ensure_boolean_ready(self.objects[pid].mesh)
        for oid in cutters:
            ensure_boolean_ready(self.objects[oid].mesh)

        def _apply(_eng: "SceneEngine"):
            primary = _eng.objects[pid]
            kid_primary = self._kernel_obj_id(primary)
            kid_cutters = [self._kernel_obj_id(self.objects[c]) for c in cutters]
            kernel_ready = bool(kid_primary) and all(bool(k) for k in kid_cutters)
            if kernel_ready:
                try:
                    out_kernel_id = self.kernel.boolean("cut", kid_primary, kid_cutters)
                    tri = self.kernel.triangulate(out_kernel_id, quality={"deflection": 0.2})
                    out = MeshData(vertices=np.asarray(tri.get("vertices", []), dtype=float), faces=np.asarray(tri.get("faces", []), dtype=int))
                    out_obj = SceneObject.create(name=f"{primary.name}_sub", mesh=out, source="Kernel:Boolean:Subtract")
                    self._bind_kernel_object(out_obj, out_kernel_id, "boolean:cut")
                    out_obj.color = primary.color
                    out_obj.opacity = primary.opacity
                    _eng._apply_layer_defaults(out_obj, preferred=str(primary.meta.get("layer", "Default") or "Default"))
                    _eng.objects[pid] = out_obj
                    for cid in cutters:
                        _eng.objects.pop(cid, None)
                    _eng.selection = [out_obj.id]
                    return
                except Exception as exc:
                    try:
                        self.logger.warning("Kernel subtract failed; using mesh fallback: %s", exc)
                    except Exception:
                        pass

            out = boolean_subtract(primary.mesh, [self.objects[c].mesh for c in cutters], tolerance=tolerance)
            out_obj = SceneObject.create(name=f"{primary.name}_sub", mesh=out, source="Boolean:Subtract")
            out_obj.color = primary.color
            out_obj.opacity = primary.opacity
            _eng._apply_layer_defaults(out_obj, preferred=str(primary.meta.get("layer", "Default") or "Default"))
            _eng.objects[pid] = out_obj
            for cid in cutters:
                _eng.objects.pop(cid, None)
            _eng.selection = [out_obj.id]

        self.execute("Boolean subtract", _apply, kind="boolean")
        return self.selection[0] if self.selection else None

    def boolean_intersect(self, ids: Sequence[str], tolerance: float = 1e-6) -> Optional[str]:
        sel = [str(i) for i in ids if str(i) in self.objects]
        if len(sel) < 2:
            raise RuntimeError("Intersect requires at least 2 selected objects.")
        for oid in sel:
            ensure_boolean_ready(self.objects[oid].mesh)

        def _apply(_eng: "SceneEngine"):
            first = self.objects[sel[0]]
            kernel_ids = [self._kernel_obj_id(self.objects[oid]) for oid in sel]
            kernel_ready = all(bool(k) for k in kernel_ids)
            if kernel_ready:
                try:
                    out_kernel_id = self.kernel.boolean("common", kernel_ids[0], kernel_ids[1:])
                    tri = self.kernel.triangulate(out_kernel_id, quality={"deflection": 0.2})
                    out = MeshData(vertices=np.asarray(tri.get("vertices", []), dtype=float), faces=np.asarray(tri.get("faces", []), dtype=int))
                    out_obj = SceneObject.create(name=f"{first.name}_inter", mesh=out, source="Kernel:Boolean:Intersect")
                    self._bind_kernel_object(out_obj, out_kernel_id, "boolean:common")
                    out_obj.color = first.color
                    out_obj.opacity = first.opacity
                    _eng._apply_layer_defaults(out_obj, preferred=str(first.meta.get("layer", "Default") or "Default"))
                    for oid in sel:
                        _eng.objects.pop(oid, None)
                    _eng.objects[out_obj.id] = out_obj
                    _eng.selection = [out_obj.id]
                    return
                except Exception as exc:
                    try:
                        self.logger.warning("Kernel intersect failed; using mesh fallback: %s", exc)
                    except Exception:
                        pass

            out = boolean_intersect([self.objects[oid].mesh for oid in sel], tolerance=tolerance)
            out_obj = SceneObject.create(name=f"{first.name}_inter", mesh=out, source="Boolean:Intersect")
            out_obj.color = first.color
            out_obj.opacity = first.opacity
            _eng._apply_layer_defaults(out_obj, preferred=str(first.meta.get("layer", "Default") or "Default"))
            for oid in sel:
                _eng.objects.pop(oid, None)
            _eng.objects[out_obj.id] = out_obj
            _eng.selection = [out_obj.id]

        self.execute("Boolean intersect", _apply, kind="boolean")
        return self.selection[0] if self.selection else None

    # ---------------- selection ----------------
    def select(self, ids: Sequence[str], mode: str = "replace") -> None:
        mode = str(mode or "replace").lower()
        incoming = [str(i) for i in ids if str(i) in self.objects]
        if mode == "add":
            out = list(dict.fromkeys(self.selection + incoming))
        elif mode == "remove":
            rem = set(incoming)
            out = [x for x in self.selection if x not in rem]
        elif mode == "invert":
            rem = set(self.selection)
            out = [oid for oid in self.objects.keys() if oid not in rem]
        elif mode == "all":
            out = list(self.objects.keys())
        elif mode == "none":
            out = []
        else:
            out = list(dict.fromkeys(incoming))
        self.selection = out
        self.emit("selection_changed", {"selection": list(self.selection)})

    # ---------------- measurements ----------------
    def compute_object_measure(self, obj_id: str) -> dict:
        oid = str(obj_id)
        if oid not in self.objects:
            raise RuntimeError("Object not found.")
        obj = self.objects[oid]
        m = object_metrics(obj)
        row = {"type": "object", "object_id": oid, "name": obj.name, **m}
        self.measurements.append(row)
        self.emit("measurements_changed", {"count": len(self.measurements)})
        return row

    def compute_distance(self, p1, p2, name: str = "distance") -> dict:
        d = distance(p1, p2)
        row = {
            "type": "distance",
            "name": str(name or "distance"),
            "p1": [float(x) for x in np.asarray(p1, dtype=float).reshape(3).tolist()],
            "p2": [float(x) for x in np.asarray(p2, dtype=float).reshape(3).tolist()],
            "value": float(d),
        }
        self.measurements.append(row)
        self.emit("measurements_changed", {"count": len(self.measurements)})
        return row

    def compute_angle(self, p1, p2, p3, name: str = "angle") -> dict:
        a = angle(p1, p2, p3)
        row = {
            "type": "angle",
            "name": str(name or "angle"),
            "p1": [float(x) for x in np.asarray(p1, dtype=float).reshape(3).tolist()],
            "p2": [float(x) for x in np.asarray(p2, dtype=float).reshape(3).tolist()],
            "p3": [float(x) for x in np.asarray(p3, dtype=float).reshape(3).tolist()],
            "value_deg": float(a),
        }
        self.measurements.append(row)
        self.emit("measurements_changed", {"count": len(self.measurements)})
        return row

    def clear_measurements(self) -> None:
        self.measurements = []
        self.emit("measurements_changed", {"count": 0})

    # ---------------- markers ----------------
    def marker_variables(self, marker: Marker) -> Dict[str, float]:
        vars_out = {
            "x": float(marker.position[0]),
            "y": float(marker.position[1]),
            "z": float(marker.position[2]),
        }
        target_id = marker.target_object_id
        if target_id and target_id in self.objects:
            vars_out.update(object_metrics(self.objects[target_id]))
            vars_out["cx"], vars_out["cy"], vars_out["cz"] = centroid(self.objects[target_id].mesh)
        return vars_out

    def evaluate_marker(self, marker_id: str) -> Marker:
        mid = str(marker_id)
        if mid not in self.markers:
            raise RuntimeError("Marker not found.")
        mk = self.markers[mid]
        vars_ctx = self.marker_variables(mk)
        out: Dict[str, float] = {}
        valid = True
        err = ""
        for key, expr in mk.expressions.items():
            k = str(key or "").strip() or "value"
            try:
                out[k] = float(self.evaluator.evaluate(expr, vars_ctx))
            except Exception as e:
                valid = False
                err = str(e)
                out[k] = float("nan")
        mk.last_values = out
        mk.valid = bool(valid)
        mk.error = str(err)
        self.emit("markers_changed", {"marker_id": mid})
        return mk

    def evaluate_all_markers(self) -> None:
        for mid in list(self.markers.keys()):
            try:
                self.evaluate_marker(mid)
            except Exception:
                continue

    def add_marker(
        self,
        position,
        name: str = "Marker",
        target_object_id: Optional[str] = None,
        expressions: Optional[Dict[str, str]] = None,
    ) -> str:
        p = np.asarray(position, dtype=float).reshape(-1)
        if p.size != 3:
            p = np.zeros(3, dtype=float)
        mk = Marker.create(name=name, position=p, target_object_id=target_object_id)
        if expressions:
            mk.expressions = {str(k): str(v) for k, v in dict(expressions).items()}

        def _add(_eng: "SceneEngine"):
            _eng.markers[mk.id] = mk.clone()

        self.execute("Add marker", _add)
        self.evaluate_marker(mk.id)
        return mk.id

    def update_marker(
        self,
        marker_id: str,
        name: Optional[str] = None,
        position=None,
        expressions: Optional[Dict[str, str]] = None,
        style: Optional[Dict[str, object]] = None,
        visible: Optional[bool] = None,
    ) -> None:
        mid = str(marker_id)
        if mid not in self.markers:
            return

        def _update(_eng: "SceneEngine"):
            mk = _eng.markers[mid]
            if name is not None:
                mk.name = str(name)
            if position is not None:
                p = np.asarray(position, dtype=float).reshape(-1)
                if p.size == 3:
                    mk.position = p.astype(float)
            if expressions is not None:
                mk.expressions = {str(k): str(v) for k, v in dict(expressions).items()}
            if style is not None:
                base = dict(mk.style)
                base.update(dict(style))
                mk.style = base
            if visible is not None:
                mk.visible = bool(visible)

        self.execute("Update marker", _update)
        self.evaluate_marker(mid)

    def delete_marker(self, marker_id: str) -> None:
        mid = str(marker_id)
        if mid not in self.markers:
            return

        def _del(_eng: "SceneEngine"):
            _eng.markers.pop(mid, None)

        self.execute("Delete marker", _del)

    def save_markers(self, path: str) -> None:
        out = [mk.to_dict() for mk in self.markers.values()]
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

    def load_markers(self, path: str) -> int:
        if not os.path.isfile(path):
            return 0
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            data = json.load(f)
        count = 0

        def _load(_eng: "SceneEngine"):
            nonlocal count
            _eng.markers = {}
            for row in data if isinstance(data, list) else []:
                mk = Marker.from_dict(row)
                _eng.markers[mk.id] = mk
                count += 1

        self.execute("Load markers", _load)
        self.evaluate_all_markers()
        return int(count)

    # ---------------- diagnostics ----------------
    def object_validation(self, obj_id: str) -> dict:
        oid = str(obj_id)
        if oid not in self.objects:
            raise RuntimeError("Object not found.")
        obj = self.objects[oid]
        kid = self._kernel_obj_id(obj)
        if kid:
            try:
                return dict(self.kernel.validate(kid))
            except Exception:
                pass
        return validate_mesh(obj.mesh)

    def object_heal(self, obj_id: str) -> dict:
        oid = str(obj_id)
        if oid not in self.objects:
            raise RuntimeError("Object not found.")
        obj = self.objects[oid]
        kid = self._kernel_obj_id(obj)
        if not kid:
            return {
                "ok": False,
                "messages": ["Heal is available only for kernel-backed objects."],
            }
        report = dict(self.kernel.heal(kid))
        try:
            self._sync_mesh_from_kernel(obj)
        except Exception:
            pass
        self.emit("scene_changed", {"reason": "heal", "object_id": oid})
        return report
