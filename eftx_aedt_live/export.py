from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import json
import numpy as np

from core.audit import audit_span, emit_audit


@dataclass
class PatternExport:
    """Helpers to persist extracted patterns.

    Files:
      - 2D cuts: CSV or JSON
      - 3D grids: NPZ (compact) + optional OBJ mesh export
    """

    base_dir: Path

    def ensure(self) -> Path:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        return self.base_dir

    def save_cut_csv(self, name: str, angles_deg, values, meta: Optional[Dict] = None) -> Path:
        with audit_span("EXPORT_CUT_CSV", name=name, base_dir=str(self.base_dir)):
            self.ensure()
            out = self.base_dir / f"{name}.csv"
            arr = np.column_stack([np.asarray(angles_deg, float), np.asarray(values, float)])
            header = "angle_deg,value"
            np.savetxt(out, arr, delimiter=",", header=header, comments="")
            if meta:
                (self.base_dir / f"{name}.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            emit_audit("EXPORT_CUT_CSV_OK", path=str(out), rows=int(arr.shape[0]))
            return out

    def save_cut_json(self, name: str, angles_deg, values, meta: Optional[Dict] = None) -> Path:
        with audit_span("EXPORT_CUT_JSON", name=name, base_dir=str(self.base_dir)):
            self.ensure()
            out = self.base_dir / f"{name}.json"
            payload = {
                "angles_deg": list(map(float, angles_deg)),
                "values": list(map(float, values)),
                "meta": meta or {},
            }
            tmp = self.base_dir / f".{name}.json.tmp"
            tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            tmp.replace(out)
            emit_audit("EXPORT_CUT_JSON_OK", path=str(out), rows=int(len(payload["angles_deg"])))
            return out

    def save_grid_npz(self, name: str, theta_deg, phi_deg, values, meta: Optional[Dict] = None) -> Path:
        with audit_span("EXPORT_GRID_NPZ", name=name, base_dir=str(self.base_dir)):
            self.ensure()
            out = self.base_dir / f"{name}.npz"
            tmp = self.base_dir / f".{name}.tmp.npz"
            t = np.asarray(theta_deg, float)
            p = np.asarray(phi_deg, float)
            z = np.asarray(values, float)
            np.savez_compressed(
                tmp,
                theta_deg=t,
                phi_deg=p,
                values=z,
                meta=json.dumps(meta or {}, ensure_ascii=False),
            )
            tmp.replace(out)
            emit_audit(
                "EXPORT_GRID_NPZ_OK",
                path=str(out),
                theta_pts=int(t.size),
                phi_pts=int(p.size),
                value_shape=tuple(int(x) for x in z.shape),
            )
            return out

    def export_obj_from_db_grid(
        self,
        name: str,
        theta_deg,
        phi_deg,
        values_db,
        db_min: float = -40.0,
        db_max: float = 0.0,
        gamma: float = 1.0,
        scale: float = 1.0,
    ) -> Tuple[Path, Path]:
        """Export an OBJ mesh from a (theta, phi) grid in dB.

        Parameters:
          db_min/db_max: clip range used to avoid huge spikes.
          gamma: nonlinear shaping exponent; >1 emphasizes main lobe.
          scale: overall scale multiplier.

        Outputs:
          (obj_path, mtl_path)
        """
        with audit_span("EXPORT_OBJ_FROM_DB_GRID", name=name, base_dir=str(self.base_dir)):
            self.ensure()
            obj_path = self.base_dir / f"{name}.obj"
            mtl_path = self.base_dir / f"{name}.mtl"

            th = np.deg2rad(np.asarray(theta_deg, float))
            ph = np.deg2rad(np.asarray(phi_deg, float))
            zdb = np.asarray(values_db, float)

            # Normalize and map dB -> radius in [0,1] (relative).
            zdb = np.clip(zdb, db_min, db_max)
            zn = (zdb - db_min) / max(1e-12, (db_max - db_min))
            r = (zn ** gamma) * scale

            # Build mesh vertices on a structured grid.
            # Theta index i, Phi index j.
            T, P = np.meshgrid(th, ph, indexing="ij")
            R = r

            X = R * np.sin(T) * np.cos(P)
            Y = R * np.sin(T) * np.sin(P)
            Z = R * np.cos(T)

            # Write OBJ
            lines = []
            lines.append(f"mtllib {mtl_path.name}")
            lines.append(f"o {name}")
            lines.append("usemtl default")

            # Vertices
            # OBJ uses 1-indexed vertices; we flatten in row-major order.
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    lines.append(f"v {X[i,j]:.6e} {Y[i,j]:.6e} {Z[i,j]:.6e}")

            n_theta, n_phi = X.shape

            def vid(i, j) -> int:
                return i * n_phi + j + 1

            # Faces (two triangles per quad)
            for i in range(n_theta - 1):
                for j in range(n_phi - 1):
                    v00 = vid(i, j)
                    v10 = vid(i + 1, j)
                    v11 = vid(i + 1, j + 1)
                    v01 = vid(i, j + 1)
                    lines.append(f"f {v00} {v10} {v11}")
                    lines.append(f"f {v00} {v11} {v01}")

            obj_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

            # Simple MTL
            mtl_path.write_text(
                "\n".join([
                    "newmtl default",
                    "Ka 0.2 0.2 0.2",
                    "Kd 0.8 0.8 0.8",
                    "Ks 0.0 0.0 0.0",
                    "d 1.0",
                ]) + "\n",
                encoding="utf-8",
            )

            emit_audit(
                "EXPORT_OBJ_FROM_DB_GRID_OK",
                obj=str(obj_path),
                mtl=str(mtl_path),
                theta=int(n_theta),
                phi=int(n_phi),
            )
            return obj_path, mtl_path
