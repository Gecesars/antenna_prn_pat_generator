from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import numpy as np

from .scene_object import MeshData


def _safe_float(value, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def create_box(width: float, depth: float, height: float, center=(0.0, 0.0, 0.0)) -> MeshData:
    w = max(1e-9, _safe_float(width, 1.0)) * 0.5
    d = max(1e-9, _safe_float(depth, 1.0)) * 0.5
    h = max(1e-9, _safe_float(height, 1.0)) * 0.5
    cx, cy, cz = [float(x) for x in center]
    vertices = np.array(
        [
            [cx - w, cy - d, cz - h],
            [cx + w, cy - d, cz - h],
            [cx + w, cy + d, cz - h],
            [cx - w, cy + d, cz - h],
            [cx - w, cy - d, cz + h],
            [cx + w, cy - d, cz + h],
            [cx + w, cy + d, cz + h],
            [cx - w, cy + d, cz + h],
        ],
        dtype=float,
    )
    faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 6, 5],
            [4, 7, 6],
            [0, 4, 5],
            [0, 5, 1],
            [1, 5, 6],
            [1, 6, 2],
            [2, 6, 7],
            [2, 7, 3],
            [3, 7, 4],
            [3, 4, 0],
        ],
        dtype=int,
    )
    return MeshData(vertices=vertices, faces=faces)


def create_plane(width: float, depth: float, center=(0.0, 0.0, 0.0), axis: str = "z") -> MeshData:
    w = max(1e-9, _safe_float(width, 1.0)) * 0.5
    d = max(1e-9, _safe_float(depth, 1.0)) * 0.5
    cx, cy, cz = [float(x) for x in center]
    axis = str(axis or "z").lower()
    if axis == "x":
        vertices = np.array(
            [
                [cx, cy - w, cz - d],
                [cx, cy + w, cz - d],
                [cx, cy + w, cz + d],
                [cx, cy - w, cz + d],
            ],
            dtype=float,
        )
    elif axis == "y":
        vertices = np.array(
            [
                [cx - w, cy, cz - d],
                [cx + w, cy, cz - d],
                [cx + w, cy, cz + d],
                [cx - w, cy, cz + d],
            ],
            dtype=float,
        )
    else:
        vertices = np.array(
            [
                [cx - w, cy - d, cz],
                [cx + w, cy - d, cz],
                [cx + w, cy + d, cz],
                [cx - w, cy + d, cz],
            ],
            dtype=float,
        )
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=int)
    return MeshData(vertices=vertices, faces=faces)


def create_cylinder(radius: float, height: float, segments: int = 32, center=(0.0, 0.0, 0.0)) -> MeshData:
    r = max(1e-9, _safe_float(radius, 0.5))
    h = max(1e-9, _safe_float(height, 1.0))
    n = max(8, int(segments))
    cx, cy, cz = [float(x) for x in center]
    top_z = cz + h * 0.5
    bot_z = cz - h * 0.5
    ang = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    top = np.column_stack([cx + r * np.cos(ang), cy + r * np.sin(ang), np.full(n, top_z)])
    bot = np.column_stack([cx + r * np.cos(ang), cy + r * np.sin(ang), np.full(n, bot_z)])
    vertices = np.vstack([top, bot, [[cx, cy, top_z], [cx, cy, bot_z]]]).astype(float)
    top_center = 2 * n
    bot_center = 2 * n + 1
    faces = []
    for i in range(n):
        j = (i + 1) % n
        ti, tj = i, j
        bi, bj = n + i, n + j
        faces.append([ti, tj, bj])
        faces.append([ti, bj, bi])
        faces.append([top_center, tj, ti])
        faces.append([bot_center, bi, bj])
    return MeshData(vertices=vertices, faces=np.asarray(faces, dtype=int))


def create_cone(radius: float, height: float, segments: int = 32, center=(0.0, 0.0, 0.0)) -> MeshData:
    r = max(1e-9, _safe_float(radius, 0.5))
    h = max(1e-9, _safe_float(height, 1.0))
    n = max(8, int(segments))
    cx, cy, cz = [float(x) for x in center]
    base_z = cz - h * 0.5
    apex = np.array([[cx, cy, cz + h * 0.5]], dtype=float)
    ang = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    base = np.column_stack([cx + r * np.cos(ang), cy + r * np.sin(ang), np.full(n, base_z)])
    vertices = np.vstack([base, apex, [[cx, cy, base_z]]]).astype(float)
    apex_idx = n
    base_center = n + 1
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces.append([apex_idx, i, j])
        faces.append([base_center, j, i])
    return MeshData(vertices=vertices, faces=np.asarray(faces, dtype=int))


def create_sphere(radius: float, theta_res: int = 24, phi_res: int = 24, center=(0.0, 0.0, 0.0)) -> MeshData:
    r = max(1e-9, _safe_float(radius, 0.5))
    nt = max(8, int(theta_res))
    np_ = max(8, int(phi_res))
    cx, cy, cz = [float(x) for x in center]
    vertices = []
    faces = []
    for i in range(nt + 1):
        th = math.pi * i / nt
        st = math.sin(th)
        ct = math.cos(th)
        for j in range(np_):
            ph = 2.0 * math.pi * j / np_
            vertices.append([cx + r * st * math.cos(ph), cy + r * st * math.sin(ph), cz + r * ct])
    vertices = np.asarray(vertices, dtype=float)

    def vid(i, j):
        return i * np_ + (j % np_)

    for i in range(nt):
        for j in range(np_):
            a = vid(i, j)
            b = vid(i + 1, j)
            c = vid(i + 1, j + 1)
            d = vid(i, j + 1)
            if i != 0:
                faces.append([a, b, d])
            if i != nt - 1:
                faces.append([d, b, c])
    return MeshData(vertices=vertices, faces=np.asarray(faces, dtype=int))


def _rotation_matrix(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    rx = math.radians(float(rx_deg))
    ry = math.radians(float(ry_deg))
    rz = math.radians(float(rz_deg))
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    rx_m = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=float)
    ry_m = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=float)
    rz_m = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    return rz_m @ ry_m @ rx_m


def transform_mesh(
    mesh: MeshData,
    tx: float = 0.0,
    ty: float = 0.0,
    tz: float = 0.0,
    rx_deg: float = 0.0,
    ry_deg: float = 0.0,
    rz_deg: float = 0.0,
    scale: float = 1.0,
    pivot: np.ndarray | None = None,
) -> MeshData:
    out = mesh.clone()
    v = np.asarray(out.vertices, dtype=float)
    if v.size == 0:
        return out
    if pivot is None:
        pivot = np.mean(v, axis=0)
    pivot = np.asarray(pivot, dtype=float).reshape(3)
    s = max(1e-9, float(scale))
    v = (v - pivot) * s + pivot
    if abs(rx_deg) > 1e-12 or abs(ry_deg) > 1e-12 or abs(rz_deg) > 1e-12:
        r = _rotation_matrix(rx_deg, ry_deg, rz_deg)
        v = (v - pivot) @ r.T + pivot
    if abs(tx) > 1e-12 or abs(ty) > 1e-12 or abs(tz) > 1e-12:
        v = v + np.array([float(tx), float(ty), float(tz)], dtype=float)
    out.vertices = v
    return out


def _to_polydata(mesh: MeshData):
    try:
        import pyvista as pv
    except Exception as e:  # pragma: no cover
        raise RuntimeError("pyvista is required for boolean operations.") from e

    faces = np.asarray(mesh.faces, dtype=int)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise RuntimeError("Mesh faces must be triangulated (Mx3).")
    flat = np.empty((faces.shape[0], 4), dtype=np.int64)
    flat[:, 0] = 3
    flat[:, 1:] = faces.astype(np.int64)
    poly = pv.PolyData(np.asarray(mesh.vertices, dtype=float), flat.reshape(-1))
    return poly


def _from_polydata(poly) -> MeshData:
    tris = np.asarray(poly.faces).reshape(-1, 4)[:, 1:4]
    verts = np.asarray(poly.points, dtype=float)
    return MeshData(vertices=verts, faces=tris.astype(int))


def preprocess_for_boolean(mesh: MeshData):
    poly = _to_polydata(mesh)
    poly = poly.triangulate().clean()
    try:
        poly = poly.compute_normals(auto_orient_normals=True, consistent_normals=True, splitting=False)
    except Exception:
        pass
    return poly


def boolean_union(meshes: Sequence[MeshData], tolerance: float = 1e-6) -> MeshData:
    if len(meshes) < 2:
        raise RuntimeError("Union requires at least two meshes.")
    result = preprocess_for_boolean(meshes[0])
    for m in meshes[1:]:
        b = preprocess_for_boolean(m)
        result = result.boolean_union(b, tolerance=float(tolerance))
        result = result.triangulate().clean()
    return _from_polydata(result)


def boolean_subtract(primary: MeshData, cutters: Sequence[MeshData], tolerance: float = 1e-6) -> MeshData:
    if not cutters:
        raise RuntimeError("Subtract requires at least one cutter mesh.")
    result = preprocess_for_boolean(primary)
    for c in cutters:
        b = preprocess_for_boolean(c)
        result = result.boolean_difference(b, tolerance=float(tolerance))
        result = result.triangulate().clean()
    return _from_polydata(result)


def boolean_intersect(meshes: Sequence[MeshData], tolerance: float = 1e-6) -> MeshData:
    if len(meshes) < 2:
        raise RuntimeError("Intersect requires at least two meshes.")
    result = preprocess_for_boolean(meshes[0])
    for m in meshes[1:]:
        b = preprocess_for_boolean(m)
        result = result.boolean_intersection(b, tolerance=float(tolerance))
        result = result.triangulate().clean()
    return _from_polydata(result)


def export_mesh(mesh: MeshData, path: str) -> None:
    poly = _to_polydata(mesh)
    poly.save(str(path))
