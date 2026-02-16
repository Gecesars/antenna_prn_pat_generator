from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eftx_aedt_live.export import PatternExport
from eftx_aedt_live.farfield import CutRequest, FarFieldExtractor, GridRequest
from eftx_aedt_live.session import AedtConnectionConfig, AedtHfssSession


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AEDT/HFSS smoke post-processing test.")
    p.add_argument("--aedt", required=True, help="Path to .aedt project or project name.")
    p.add_argument("--design", required=True, help="HFSS design name.")
    p.add_argument("--setup", required=True, help="Setup sweep (e.g. 'Setup1 : LastAdaptive').")
    p.add_argument("--sphere", default="3D_Sphere", help="Infinite sphere setup name.")
    p.add_argument("--expr", default="dB(GainTotal)", help="Far-field expression.")
    p.add_argument("--freq", default="", help="Optional frequency variation (e.g. 0.8GHz).")
    p.add_argument("--out", default="out", help="Output directory.")
    p.add_argument("--version", default="2025.2", help="AEDT version.")
    p.add_argument("--new-session", action="store_true", help="Start a new AEDT desktop session.")
    p.add_argument("--non-graphical", action="store_true", help="Run AEDT in non-graphical mode.")
    p.add_argument("--theta-points", type=int, default=181, help="Theta points for 3D grid.")
    p.add_argument("--phi-points", type=int, default=361, help="Phi points for 3D grid.")
    return p.parse_args()


def is_db_expression(expr: str) -> bool:
    text = str(expr).strip().lower().replace(" ", "")
    return ("db(" in text) or text.startswith("db")


def to_mag_arrays(values: np.ndarray, expr: str) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(values, dtype=float)
    arr = np.nan_to_num(arr, nan=-300.0, neginf=-300.0, posinf=0.0)
    if is_db_expression(expr):
        arr_db = arr - float(np.max(arr))
        arr_lin = np.power(10.0, arr_db / 20.0)
        arr_lin = np.clip(arr_lin, 0.0, None)
        return arr_lin, arr_db
    arr_lin = np.clip(arr, 0.0, None)
    vmax = float(np.max(arr_lin)) if arr_lin.size else 0.0
    if vmax > 0:
        arr_lin = arr_lin / vmax
    arr_db = 20.0 * np.log10(np.maximum(arr_lin, 1e-12))
    return arr_lin, arr_db


def save_cut_json(path: Path, angles: np.ndarray, mag_lin: np.ndarray, mag_db: np.ndarray, meta: Dict) -> None:
    payload = {
        "angles_deg": angles.tolist(),
        "mag_lin": mag_lin.tolist(),
        "mag_db": mag_db.tolist(),
        "meta": meta,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_log = out_dir / "run_log.txt"
    events = []

    def mark(tag: str, t0: float, ok: bool, extra: str = ""):
        dt_ms = (time.perf_counter() - t0) * 1000.0
        status = "OK" if ok else "FAIL"
        line = f"{tag} {status} dt={dt_ms:.2f}ms {extra}".strip()
        print(line)
        events.append(line)

    cfg = AedtConnectionConfig(
        version=args.version,
        new_desktop=bool(args.new_session),
        non_graphical=bool(args.non_graphical),
        close_on_exit=False,
    )
    session = AedtHfssSession(cfg)
    extractor = FarFieldExtractor(session)
    exporter = PatternExport(out_dir)

    try:
        t0 = time.perf_counter()
        session.connect(project=args.aedt, design=args.design, setup=None)
        mark("connect", t0, True, f"version={args.version}")

        base_meta = {
            "project": args.aedt,
            "design": args.design,
            "setup": args.setup,
            "sphere": args.sphere,
            "expr": args.expr,
            "freq": args.freq,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

        t0 = time.perf_counter()
        hrp = extractor.extract_cut(
            CutRequest(
                setup_sweep=args.setup,
                sphere_name=args.sphere,
                expression=args.expr,
                primary_sweep="Theta",
                fixed={"Phi": "90deg", **({"Freq": args.freq} if args.freq else {})},
                convert_theta_to_elevation=False,
            )
        )
        h_ang = np.asarray(hrp.angles_deg, dtype=float)
        h_lin, h_db = to_mag_arrays(np.asarray(hrp.values, dtype=float), args.expr)
        save_cut_json(out_dir / "hrp.json", h_ang, h_lin, h_db, {**base_meta, **dict(hrp.meta), "mode": "HRP"})
        mark("pull_hrp", t0, True, f"points={h_ang.size}")

        t0 = time.perf_counter()
        vrp = extractor.extract_cut(
            CutRequest(
                setup_sweep=args.setup,
                sphere_name=args.sphere,
                expression=args.expr,
                primary_sweep="Theta",
                fixed={"Phi": "0deg", **({"Freq": args.freq} if args.freq else {})},
                convert_theta_to_elevation=True,
            )
        )
        v_ang = np.asarray(vrp.angles_deg, dtype=float)
        v_lin, v_db = to_mag_arrays(np.asarray(vrp.values, dtype=float), args.expr)
        save_cut_json(out_dir / "vrp.json", v_ang, v_lin, v_db, {**base_meta, **dict(vrp.meta), "mode": "VRP"})
        mark("pull_vrp", t0, True, f"points={v_ang.size}")

        t0 = time.perf_counter()
        grid = extractor.extract_grid(
            GridRequest(
                setup_sweep=args.setup,
                sphere_name=args.sphere,
                expression=args.expr,
                theta_points=max(3, int(args.theta_points)),
                phi_points=max(3, int(args.phi_points)),
                freq=args.freq or None,
                convert_theta_to_elevation=False,
            )
        )
        theta = np.asarray(grid.theta_deg, dtype=float)
        phi = np.asarray(grid.phi_deg, dtype=float)
        values = np.asarray(grid.values, dtype=float)
        _, grid_db = to_mag_arrays(values, args.expr)
        npz_path = exporter.save_grid_npz("ff_3d", theta, phi, grid_db, {**base_meta, **dict(grid.meta), "mode": "3D"})
        mark("pull_3d", t0, True, f"shape={theta.size}x{phi.size}")

        checks = []
        checks.append(("hrp", bool(h_ang.size and h_lin.size and np.all(np.isfinite(h_lin)))))
        checks.append(("vrp", bool(v_ang.size and v_lin.size and np.all(np.isfinite(v_lin)))))
        checks.append(("ff_3d", bool(theta.size and phi.size and grid_db.size and np.all(np.isfinite(grid_db)))))
        failed = [name for name, ok in checks if not ok]
        if failed:
            raise RuntimeError(f"Output validation failed for: {', '.join(failed)}")

        (out_dir / "run_log.txt").write_text("\n".join(events) + f"\nnpz={npz_path}\n", encoding="utf-8")
        print(f"Artifacts written to: {out_dir}")
        return 0
    except Exception as e:
        events.append(f"ERROR {e}")
        run_log.write_text("\n".join(events) + "\n", encoding="utf-8")
        print(f"FAILED: {e}", file=sys.stderr)
        return 1
    finally:
        t0 = time.perf_counter()
        try:
            session.disconnect()
            mark("disconnect", t0, True, "")
        except Exception as e:
            mark("disconnect", t0, False, str(e))
            run_log.write_text("\n".join(events) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
