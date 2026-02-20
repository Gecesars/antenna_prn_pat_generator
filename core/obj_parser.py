from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class ObjTriangle:
    v: Tuple[int, int, int]
    vt: Tuple[Optional[int], Optional[int], Optional[int]]
    vn: Tuple[Optional[int], Optional[int], Optional[int]]
    object_name: str
    group_name: str
    material: str
    smoothing: str
    line_no: int
    raw: str


@dataclass
class ObjModel:
    path: str
    vertices: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=float))
    texcoords: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=float))
    normals: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=float))
    triangles: List[ObjTriangle] = field(default_factory=list)
    mtllib: List[str] = field(default_factory=list)
    comments: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def sha256(self) -> str:
        try:
            h = hashlib.sha256()
            with open(self.path, "rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
            return h.hexdigest()
        except Exception:
            return ""

    def summary(self) -> dict:
        object_names = sorted({t.object_name for t in self.triangles if t.object_name})
        group_names = sorted({t.group_name for t in self.triangles if t.group_name})
        materials = sorted({t.material for t in self.triangles if t.material})
        return {
            "path": self.path,
            "sha256": self.sha256(),
            "vertex_count": int(self.vertices.shape[0]),
            "texcoord_count": int(self.texcoords.shape[0]),
            "normal_count": int(self.normals.shape[0]),
            "triangle_count": int(len(self.triangles)),
            "object_names": object_names,
            "group_names": group_names,
            "materials": materials,
            "mtllib": list(self.mtllib),
            "warnings": list(self.warnings),
        }

    def mesh_chunks(self, split_mode: str = "object_group_material") -> List[dict]:
        """Return triangulated mesh chunks preserving OBJ semantic partitions.

        split_mode:
          - object
          - object_group
          - object_group_material (default)
          - material
          - single
        """
        mode = str(split_mode or "object_group_material").strip().lower()
        groups: Dict[Tuple[str, str, str], List[ObjTriangle]] = {}
        for t in self.triangles:
            if mode == "object":
                key = (t.object_name or "Object", "", "")
            elif mode == "object_group":
                key = (t.object_name or "Object", t.group_name or "", "")
            elif mode == "material":
                key = ("", "", t.material or "default")
            elif mode == "single":
                key = ("Scene", "", "")
            else:
                key = (t.object_name or "Object", t.group_name or "", t.material or "default")
            groups.setdefault(key, []).append(t)

        out: List[dict] = []
        for key, tris in groups.items():
            obj_name, grp_name, mat = key
            used = sorted({i for tri in tris for i in tri.v})
            remap = {old: idx for idx, old in enumerate(used)}
            vertices = self.vertices[np.asarray(used, dtype=int)] if used else np.zeros((0, 3), dtype=float)
            faces = np.asarray([[remap[x] for x in tri.v] for tri in tris], dtype=int) if tris else np.zeros((0, 3), dtype=int)
            corner_indices = []
            for tri in tris:
                corner_indices.append(
                    {
                        "v": list(tri.v),
                        "vt": [None if x is None else int(x) for x in tri.vt],
                        "vn": [None if x is None else int(x) for x in tri.vn],
                        "line_no": int(tri.line_no),
                        "smoothing": str(tri.smoothing),
                    }
                )
            name_parts = [p for p in [obj_name, grp_name, mat] if p]
            name = "_".join(name_parts) if name_parts else "Object"
            out.append(
                {
                    "name": name,
                    "object_name": obj_name,
                    "group_name": grp_name,
                    "material": mat,
                    "vertices": vertices,
                    "faces": faces,
                    "corner_indices": corner_indices,
                    "triangle_count": int(faces.shape[0]),
                }
            )
        return out


def _resolve_index(idx_raw: str, n: int) -> Optional[int]:
    token = str(idx_raw or "").strip()
    if not token:
        return None
    try:
        idx = int(token)
    except Exception:
        return None
    if idx > 0:
        out = idx - 1
    elif idx < 0:
        out = n + idx
    else:
        return None
    if out < 0 or out >= n:
        return None
    return int(out)


def _parse_float_tuple(parts: Sequence[str], fallback_len: int = 3) -> Tuple[float, ...]:
    vals: List[float] = []
    for x in parts:
        try:
            vals.append(float(str(x).replace(",", ".")))
        except Exception:
            vals.append(0.0)
    if len(vals) < fallback_len:
        vals.extend([0.0] * (fallback_len - len(vals)))
    return tuple(vals[:fallback_len])


def parse_obj_file(path: str) -> ObjModel:
    p = os.path.abspath(str(path))
    if not os.path.isfile(p):
        raise FileNotFoundError(p)
    model = ObjModel(path=p)
    vertices: List[Tuple[float, float, float]] = []
    texcoords: List[Tuple[float, float, float]] = []
    normals: List[Tuple[float, float, float]] = []

    current_object = Path(p).stem or "Object"
    current_group = ""
    current_material = "default"
    current_smoothing = "off"

    with open(p, "r", encoding="utf-8", errors="ignore") as f:
        for ln, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#"):
                model.comments.append(raw.rstrip("\n"))
                continue
            parts = line.split()
            if not parts:
                continue
            cmd = parts[0].lower()
            args = parts[1:]

            if cmd == "mtllib":
                if args:
                    model.mtllib.append(" ".join(args).strip())
                continue
            if cmd == "o":
                current_object = " ".join(args).strip() or current_object
                continue
            if cmd == "g":
                current_group = " ".join(args).strip()
                continue
            if cmd == "usemtl":
                current_material = " ".join(args).strip() or current_material
                continue
            if cmd == "s":
                current_smoothing = " ".join(args).strip() or "off"
                continue
            if cmd == "v":
                x, y, z = _parse_float_tuple(args[:3], 3)
                vertices.append((x, y, z))
                continue
            if cmd == "vt":
                u, v, w = _parse_float_tuple(args[:3], 3)
                texcoords.append((u, v, w))
                continue
            if cmd == "vn":
                x, y, z = _parse_float_tuple(args[:3], 3)
                normals.append((x, y, z))
                continue
            if cmd == "f":
                if len(args) < 3:
                    model.warnings.append(f"Line {ln}: face with <3 vertices ignored.")
                    continue
                corners = []
                for token in args:
                    raw_idx = str(token).split("/")
                    v_idx = _resolve_index(raw_idx[0] if len(raw_idx) >= 1 else "", len(vertices))
                    vt_idx = _resolve_index(raw_idx[1] if len(raw_idx) >= 2 else "", len(texcoords))
                    vn_idx = _resolve_index(raw_idx[2] if len(raw_idx) >= 3 else "", len(normals))
                    if v_idx is None:
                        model.warnings.append(f"Line {ln}: invalid vertex index '{token}'.")
                        corners = []
                        break
                    corners.append((v_idx, vt_idx, vn_idx))
                if len(corners) < 3:
                    continue
                c0 = corners[0]
                for k in range(1, len(corners) - 1):
                    c1 = corners[k]
                    c2 = corners[k + 1]
                    model.triangles.append(
                        ObjTriangle(
                            v=(int(c0[0]), int(c1[0]), int(c2[0])),
                            vt=(c0[1], c1[1], c2[1]),
                            vn=(c0[2], c1[2], c2[2]),
                            object_name=current_object,
                            group_name=current_group,
                            material=current_material,
                            smoothing=current_smoothing,
                            line_no=int(ln),
                            raw=line,
                        )
                    )
                continue
            # Commands not used for geometry are preserved as warnings only.
            if cmd not in {"l", "p"}:
                model.warnings.append(f"Line {ln}: unsupported token '{cmd}' ignored.")

    model.vertices = np.asarray(vertices, dtype=float) if vertices else np.zeros((0, 3), dtype=float)
    model.texcoords = np.asarray(texcoords, dtype=float) if texcoords else np.zeros((0, 3), dtype=float)
    model.normals = np.asarray(normals, dtype=float) if normals else np.zeros((0, 3), dtype=float)
    return model


def parse_obj_many(paths: Sequence[str]) -> List[ObjModel]:
    out: List[ObjModel] = []
    for p in paths:
        try:
            out.append(parse_obj_file(p))
        except Exception:
            continue
    return out
