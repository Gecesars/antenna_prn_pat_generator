from __future__ import annotations

import ast
import copy
import json
import math
import os
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .commands import CommandStack, SnapshotCommand
from .geometry_ops import (
    boolean_intersect,
    boolean_subtract,
    boolean_union,
    create_box,
    create_cone,
    create_cylinder,
    create_plane,
    create_sphere,
    transform_mesh,
)
from .measures import angle, bbox, centroid, distance, object_metrics
from .scene_object import Marker, MeshData, SceneObject
from .validators import ensure_boolean_ready, validate_mesh


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
    def __init__(self):
        self.objects: Dict[str, SceneObject] = {}
        self.selection: List[str] = []
        self.markers: Dict[str, Marker] = {}
        self.measurements: List[dict] = []
        self.tool_mode: str = "Select"
        self.command_stack = CommandStack()
        self._listeners: List[Callable[[str, dict], None]] = []
        self.evaluator = SafeExpressionEvaluator()

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

    # ---------------- state ----------------
    def serialize_state(self) -> dict:
        return {
            "objects": {oid: obj.to_dict() for oid, obj in self.objects.items()},
            "selection": list(self.selection),
            "markers": {mid: mk.to_dict() for mid, mk in self.markers.items()},
            "measurements": copy.deepcopy(self.measurements),
            "tool_mode": str(self.tool_mode),
        }

    def restore_state(self, state: dict, emit: bool = True) -> None:
        self.objects = {str(k): SceneObject.from_dict(v) for k, v in dict(state.get("objects", {})).items()}
        self.selection = [str(x) for x in state.get("selection", []) if str(x) in self.objects]
        self.markers = {str(k): Marker.from_dict(v) for k, v in dict(state.get("markers", {})).items()}
        self.measurements = copy.deepcopy(state.get("measurements", []))
        self.tool_mode = str(state.get("tool_mode", "Select"))
        if emit:
            self.emit("scene_changed", {"reason": "restore"})

    def execute(self, label: str, action: Callable[["SceneEngine"], None]) -> None:
        cmd = SnapshotCommand(label=str(label), action=action)
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
            _eng.objects[oid] = obj.clone()
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
        undoable: bool = True,
    ) -> str:
        obj = SceneObject.create(name=name, mesh=mesh, source=source)
        obj.color = str(color)
        obj.opacity = float(max(0.05, min(1.0, opacity)))
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

        self.execute(f"Delete {len(ids)} object(s)", _delete)

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

        self.execute("Rename object", _rename)

    def set_visibility(self, ids: Sequence[str], visible: bool) -> None:
        ids = [str(i) for i in ids if str(i) in self.objects]
        if not ids:
            return

        def _set(_eng: "SceneEngine"):
            for oid in ids:
                _eng.objects[oid].visible = bool(visible)

        self.execute("Set visibility", _set)

    def set_lock(self, ids: Sequence[str], locked: bool) -> None:
        ids = [str(i) for i in ids if str(i) in self.objects]
        if not ids:
            return

        def _set(_eng: "SceneEngine"):
            for oid in ids:
                _eng.objects[oid].locked = bool(locked)

        self.execute("Set lock", _set)

    def set_color_opacity(self, ids: Sequence[str], color: Optional[str] = None, opacity: Optional[float] = None) -> None:
        ids = [str(i) for i in ids if str(i) in self.objects]
        if not ids:
            return

        def _set(_eng: "SceneEngine"):
            for oid in ids:
                if color is not None:
                    _eng.objects[oid].color = str(color)
                if opacity is not None:
                    _eng.objects[oid].opacity = float(max(0.05, min(1.0, float(opacity))))

        self.execute("Set object style", _set)

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

        self.execute("Transform object(s)", _apply)

    # ---------------- primitives ----------------
    def create_primitive(self, kind: str, params: dict, name: str = "Primitive") -> str:
        token = str(kind or "").strip().lower()
        if token == "box":
            mesh = create_box(params.get("width", 1.0), params.get("depth", 1.0), params.get("height", 1.0), params.get("center", (0, 0, 0)))
        elif token == "cylinder":
            mesh = create_cylinder(params.get("radius", 0.5), params.get("height", 1.0), params.get("segments", 32), params.get("center", (0, 0, 0)))
        elif token == "sphere":
            mesh = create_sphere(params.get("radius", 0.5), params.get("theta_res", 24), params.get("phi_res", 24), params.get("center", (0, 0, 0)))
        elif token == "cone":
            mesh = create_cone(params.get("radius", 0.5), params.get("height", 1.0), params.get("segments", 32), params.get("center", (0, 0, 0)))
        elif token == "plane":
            mesh = create_plane(params.get("width", 1.0), params.get("depth", 1.0), params.get("center", (0, 0, 0)), params.get("axis", "z"))
        else:
            raise RuntimeError(f"Unknown primitive kind: {kind}")
        return self.add_mesh_object(mesh, name=name, source=f"Primitive:{token}")

    # ---------------- booleans ----------------
    def boolean_union(self, ids: Sequence[str], tolerance: float = 1e-6) -> Optional[str]:
        sel = [str(i) for i in ids if str(i) in self.objects]
        if len(sel) < 2:
            raise RuntimeError("Union requires at least 2 selected objects.")
        for oid in sel:
            ensure_boolean_ready(self.objects[oid].mesh)

        def _apply(_eng: "SceneEngine"):
            meshes = [self.objects[oid].mesh for oid in sel]
            out = boolean_union(meshes, tolerance=tolerance)
            first = self.objects[sel[0]]
            out_obj = SceneObject.create(name=f"{first.name}_union", mesh=out, source="Boolean:Union")
            out_obj.color = first.color
            out_obj.opacity = first.opacity
            for oid in sel:
                _eng.objects.pop(oid, None)
            _eng.objects[out_obj.id] = out_obj
            _eng.selection = [out_obj.id]

        self.execute("Boolean union", _apply)
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
            out = boolean_subtract(primary.mesh, [self.objects[c].mesh for c in cutters], tolerance=tolerance)
            out_obj = SceneObject.create(name=f"{primary.name}_sub", mesh=out, source="Boolean:Subtract")
            out_obj.color = primary.color
            out_obj.opacity = primary.opacity
            _eng.objects[pid] = out_obj
            for cid in cutters:
                _eng.objects.pop(cid, None)
            _eng.selection = [out_obj.id]

        self.execute("Boolean subtract", _apply)
        return self.selection[0] if self.selection else None

    def boolean_intersect(self, ids: Sequence[str], tolerance: float = 1e-6) -> Optional[str]:
        sel = [str(i) for i in ids if str(i) in self.objects]
        if len(sel) < 2:
            raise RuntimeError("Intersect requires at least 2 selected objects.")
        for oid in sel:
            ensure_boolean_ready(self.objects[oid].mesh)

        def _apply(_eng: "SceneEngine"):
            out = boolean_intersect([self.objects[oid].mesh for oid in sel], tolerance=tolerance)
            first = self.objects[sel[0]]
            out_obj = SceneObject.create(name=f"{first.name}_inter", mesh=out, source="Boolean:Intersect")
            out_obj.color = first.color
            out_obj.opacity = first.opacity
            for oid in sel:
                _eng.objects.pop(oid, None)
            _eng.objects[out_obj.id] = out_obj
            _eng.selection = [out_obj.id]

        self.execute("Boolean intersect", _apply)
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
        return validate_mesh(self.objects[oid].mesh)
