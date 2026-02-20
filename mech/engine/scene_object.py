from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional
import copy
import uuid

import numpy as np


@dataclass
class MeshData:
    vertices: np.ndarray
    faces: np.ndarray

    def clone(self) -> "MeshData":
        return MeshData(
            vertices=np.array(self.vertices, dtype=float, copy=True),
            faces=np.array(self.faces, dtype=int, copy=True),
        )

    def to_dict(self) -> dict:
        return {
            "vertices": np.asarray(self.vertices, dtype=float).tolist(),
            "faces": np.asarray(self.faces, dtype=int).tolist(),
        }

    @staticmethod
    def from_dict(data: dict) -> "MeshData":
        v = np.asarray(data.get("vertices", []), dtype=float)
        f = np.asarray(data.get("faces", []), dtype=int)
        if v.ndim != 2 or v.shape[1] != 3:
            v = np.zeros((0, 3), dtype=float)
        if f.ndim != 2 or f.shape[1] != 3:
            f = np.zeros((0, 3), dtype=int)
        return MeshData(vertices=v, faces=f)


@dataclass
class SceneObject:
    id: str
    name: str
    mesh: MeshData
    visible: bool = True
    locked: bool = False
    color: str = "#86b6f6"
    opacity: float = 0.85
    source: str = "Local"
    meta: Dict[str, object] = field(default_factory=dict)

    @staticmethod
    def create(name: str, mesh: MeshData, source: str = "Local") -> "SceneObject":
        return SceneObject(
            id=str(uuid.uuid4()),
            name=str(name or "Object"),
            mesh=mesh.clone(),
            source=str(source or "Local"),
        )

    def clone(self) -> "SceneObject":
        return SceneObject(
            id=str(self.id),
            name=str(self.name),
            mesh=self.mesh.clone(),
            visible=bool(self.visible),
            locked=bool(self.locked),
            color=str(self.color),
            opacity=float(self.opacity),
            source=str(self.source),
            meta=copy.deepcopy(self.meta),
        )

    def to_dict(self) -> dict:
        return {
            "id": str(self.id),
            "name": str(self.name),
            "mesh": self.mesh.to_dict(),
            "visible": bool(self.visible),
            "locked": bool(self.locked),
            "color": str(self.color),
            "opacity": float(self.opacity),
            "source": str(self.source),
            "meta": copy.deepcopy(self.meta),
        }

    @staticmethod
    def from_dict(data: dict) -> "SceneObject":
        mesh = MeshData.from_dict(data.get("mesh", {}))
        return SceneObject(
            id=str(data.get("id", str(uuid.uuid4()))),
            name=str(data.get("name", "Object")),
            mesh=mesh,
            visible=bool(data.get("visible", True)),
            locked=bool(data.get("locked", False)),
            color=str(data.get("color", "#86b6f6")),
            opacity=float(data.get("opacity", 0.85)),
            source=str(data.get("source", "Local")),
            meta=copy.deepcopy(data.get("meta", {})),
        )


@dataclass
class Marker:
    id: str
    name: str
    position: np.ndarray
    expressions: Dict[str, str] = field(default_factory=dict)
    last_values: Dict[str, float] = field(default_factory=dict)
    style: Dict[str, object] = field(default_factory=dict)
    visible: bool = True
    valid: bool = True
    error: str = ""
    target_object_id: Optional[str] = None

    @staticmethod
    def create(name: str, position: np.ndarray, target_object_id: Optional[str] = None) -> "Marker":
        pos = np.asarray(position, dtype=float).reshape(-1)
        if pos.size != 3:
            pos = np.zeros(3, dtype=float)
        return Marker(
            id=str(uuid.uuid4()),
            name=str(name or "Marker"),
            position=pos.astype(float),
            target_object_id=target_object_id,
            style={"color": "#ffd166", "size": 10.0},
        )

    def clone(self) -> "Marker":
        return Marker(
            id=str(self.id),
            name=str(self.name),
            position=np.array(self.position, dtype=float, copy=True),
            expressions=copy.deepcopy(self.expressions),
            last_values=copy.deepcopy(self.last_values),
            style=copy.deepcopy(self.style),
            visible=bool(self.visible),
            valid=bool(self.valid),
            error=str(self.error),
            target_object_id=self.target_object_id,
        )

    def to_dict(self) -> dict:
        return {
            "id": str(self.id),
            "name": str(self.name),
            "position": np.asarray(self.position, dtype=float).tolist(),
            "expressions": copy.deepcopy(self.expressions),
            "last_values": copy.deepcopy(self.last_values),
            "style": copy.deepcopy(self.style),
            "visible": bool(self.visible),
            "valid": bool(self.valid),
            "error": str(self.error),
            "target_object_id": self.target_object_id,
        }

    @staticmethod
    def from_dict(data: dict) -> "Marker":
        pos = np.asarray(data.get("position", [0.0, 0.0, 0.0]), dtype=float).reshape(-1)
        if pos.size != 3:
            pos = np.zeros(3, dtype=float)
        return Marker(
            id=str(data.get("id", str(uuid.uuid4()))),
            name=str(data.get("name", "Marker")),
            position=pos.astype(float),
            expressions=copy.deepcopy(data.get("expressions", {})),
            last_values=copy.deepcopy(data.get("last_values", {})),
            style=copy.deepcopy(data.get("style", {"color": "#ffd166", "size": 10.0})),
            visible=bool(data.get("visible", True)),
            valid=bool(data.get("valid", True)),
            error=str(data.get("error", "")),
            target_object_id=data.get("target_object_id", None),
        )
