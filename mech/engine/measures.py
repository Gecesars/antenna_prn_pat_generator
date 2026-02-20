from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np

from .scene_object import MeshData, SceneObject


def bbox(mesh: MeshData) -> Dict[str, float]:
    v = np.asarray(mesh.vertices, dtype=float)
    if v.size == 0:
        return {
            "xmin": 0.0,
            "xmax": 0.0,
            "ymin": 0.0,
            "ymax": 0.0,
            "zmin": 0.0,
            "zmax": 0.0,
            "dx": 0.0,
            "dy": 0.0,
            "dz": 0.0,
        }
    mins = np.min(v, axis=0)
    maxs = np.max(v, axis=0)
    d = maxs - mins
    return {
        "xmin": float(mins[0]),
        "xmax": float(maxs[0]),
        "ymin": float(mins[1]),
        "ymax": float(maxs[1]),
        "zmin": float(mins[2]),
        "zmax": float(maxs[2]),
        "dx": float(d[0]),
        "dy": float(d[1]),
        "dz": float(d[2]),
    }


def centroid(mesh: MeshData) -> Tuple[float, float, float]:
    v = np.asarray(mesh.vertices, dtype=float)
    if v.size == 0:
        return (0.0, 0.0, 0.0)
    c = np.mean(v, axis=0)
    return (float(c[0]), float(c[1]), float(c[2]))


def area(mesh: MeshData) -> float:
    f = np.asarray(mesh.faces, dtype=int)
    v = np.asarray(mesh.vertices, dtype=float)
    if f.size == 0 or v.size == 0:
        return 0.0
    tri = v[f]
    a = tri[:, 1] - tri[:, 0]
    b = tri[:, 2] - tri[:, 0]
    cr = np.cross(a, b)
    return float(0.5 * np.sum(np.linalg.norm(cr, axis=1)))


def volume(mesh: MeshData) -> float:
    """Signed volume via tetrahedra wrt origin (absolute returned)."""
    f = np.asarray(mesh.faces, dtype=int)
    v = np.asarray(mesh.vertices, dtype=float)
    if f.size == 0 or v.size == 0:
        return 0.0
    tri = v[f]
    vol = np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])) / 6.0
    return float(abs(np.sum(vol)))


def distance(p1, p2) -> float:
    a = np.asarray(p1, dtype=float).reshape(-1)
    b = np.asarray(p2, dtype=float).reshape(-1)
    if a.size != 3 or b.size != 3:
        return 0.0
    return float(np.linalg.norm(a - b))


def angle(p1, p2, p3) -> float:
    a = np.asarray(p1, dtype=float).reshape(-1)
    b = np.asarray(p2, dtype=float).reshape(-1)
    c = np.asarray(p3, dtype=float).reshape(-1)
    if a.size != 3 or b.size != 3 or c.size != 3:
        return 0.0
    u = a - b
    v = c - b
    nu = float(np.linalg.norm(u))
    nv = float(np.linalg.norm(v))
    if nu <= 1e-12 or nv <= 1e-12:
        return 0.0
    cosv = float(np.dot(u, v) / (nu * nv))
    cosv = max(-1.0, min(1.0, cosv))
    return float(math.degrees(math.acos(cosv)))


def object_metrics(obj: SceneObject) -> Dict[str, float]:
    b = bbox(obj.mesh)
    c = centroid(obj.mesh)
    a = area(obj.mesh)
    v = volume(obj.mesh)
    return {
        "bbox_dx": float(b["dx"]),
        "bbox_dy": float(b["dy"]),
        "bbox_dz": float(b["dz"]),
        "bbox_xmin": float(b["xmin"]),
        "bbox_xmax": float(b["xmax"]),
        "bbox_ymin": float(b["ymin"]),
        "bbox_ymax": float(b["ymax"]),
        "bbox_zmin": float(b["zmin"]),
        "bbox_zmax": float(b["zmax"]),
        "centroid_x": float(c[0]),
        "centroid_y": float(c[1]),
        "centroid_z": float(c[2]),
        "area": float(a),
        "volume": float(v),
    }
