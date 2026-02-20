from __future__ import annotations

from typing import Dict
import numpy as np

from .scene_object import MeshData


def validate_mesh(mesh: MeshData) -> Dict[str, object]:
    v = np.asarray(mesh.vertices, dtype=float)
    f = np.asarray(mesh.faces, dtype=int)
    out: Dict[str, object] = {
        "ok": True,
        "vertices": int(v.shape[0]) if v.ndim == 2 else 0,
        "faces": int(f.shape[0]) if f.ndim == 2 else 0,
        "degenerate_faces": 0,
        "nan_vertices": False,
        "manifold": None,
        "messages": [],
    }

    if v.ndim != 2 or v.shape[1] != 3:
        out["ok"] = False
        out["messages"].append("Invalid vertices shape. Expected Nx3.")
        return out
    if f.ndim != 2 or f.shape[1] != 3:
        out["ok"] = False
        out["messages"].append("Invalid faces shape. Expected Mx3 triangulated mesh.")
        return out

    if np.isnan(v).any() or np.isinf(v).any():
        out["ok"] = False
        out["nan_vertices"] = True
        out["messages"].append("Vertices contain NaN or Inf.")

    if f.size > 0:
        tri = v[f]
        cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        area2 = np.linalg.norm(cross, axis=1)
        deg = int(np.sum(area2 <= 1e-16))
        out["degenerate_faces"] = deg
        if deg > 0:
            out["messages"].append(f"Degenerate faces: {deg}")

    try:
        import pyvista as pv

        ff = np.empty((f.shape[0], 4), dtype=np.int64)
        ff[:, 0] = 3
        ff[:, 1:] = f.astype(np.int64)
        poly = pv.PolyData(v, ff.reshape(-1))
        out["manifold"] = bool(getattr(poly, "is_manifold", False))
        if out["manifold"] is False:
            out["messages"].append("Mesh is not manifold.")
    except Exception:
        out["messages"].append("PyVista manifold check unavailable.")

    if out["messages"]:
        out["ok"] = bool(out["ok"] and out["degenerate_faces"] == 0 and (out["nan_vertices"] is False))
    return out


def ensure_boolean_ready(mesh: MeshData) -> None:
    info = validate_mesh(mesh)
    if not bool(info.get("ok", False)):
        raise RuntimeError("Mesh validation failed before boolean: " + " | ".join(info.get("messages", [])))
