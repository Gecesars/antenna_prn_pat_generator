"""Coaxial divider geometry + AEDT/HFSS model generation helpers.

This module mirrors the geometry logic from ``exemplo_divisor.md`` while
keeping the UI layer independent from AEDT runtime details.
"""

from __future__ import annotations

import math
import re
import inspect
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple

import numpy as np


SUBSTRATE_MATERIALS = {
    "Ar": (1.0006, 0.0),
    "Teflon (PTFE)": (2.1, 0.0002),
    "FR-4": (4.4, 0.02),
    "Rogers RO4003C": (3.55, 0.0027),
}


class DividerGeometryError(RuntimeError):
    """Raised when divider geometry inputs are invalid."""


def _as_float(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    raw = row.get(key, default)
    try:
        return float(raw)
    except Exception as exc:
        raise DividerGeometryError(f"Invalid numeric value for '{key}': {raw}") from exc


def _as_int(row: Mapping[str, Any], key: str, default: int = 0) -> int:
    raw = row.get(key, default)
    try:
        return int(round(float(raw)))
    except Exception as exc:
        raise DividerGeometryError(f"Invalid integer value for '{key}': {raw}") from exc


def sanitize_material_name(name: str) -> str:
    token = str(name or "material").strip()
    token = token.replace(" ", "_").replace("(", "").replace(")", "")
    token = re.sub(r"[^A-Za-z0-9_]+", "_", token)
    return token or "material"


def _output_diameters(d_int_tube: float, d_out_50: float, n_outputs: int) -> Tuple[float, float]:
    if int(n_outputs) > 6:
        dia_diel = float(d_int_tube) / 3.0
        dia_cond = dia_diel / 2.3
        return dia_diel, dia_cond
    if int(n_outputs) > 4:
        dia_diel = float(d_int_tube) / 2.2
        dia_cond = dia_diel / 2.3
        return dia_diel, dia_cond
    return float(d_int_tube), float(d_out_50)


def compute_coaxial_divider_geometry(params: Mapping[str, Any]) -> dict:
    """Compute coaxial divider geometry values using the example logic."""
    p = dict(params or {})
    f_start = _as_float(p, "f_start")
    f_stop = _as_float(p, "f_stop")
    d_ext = _as_float(p, "d_ext")
    wall_thick = _as_float(p, "wall_thick")
    n_sections = _as_int(p, "n_sections")
    n_outputs = _as_int(p, "n_outputs")
    diel_material = str(p.get("diel_material", "Ar") or "Ar").strip()
    if diel_material not in SUBSTRATE_MATERIALS:
        raise DividerGeometryError(
            f"Unknown dielectric material '{diel_material}'. "
            f"Available: {', '.join(SUBSTRATE_MATERIALS.keys())}"
        )

    if f_start <= 0.0 or f_stop <= 0.0 or f_stop <= f_start:
        raise DividerGeometryError("Frequency bounds must satisfy 0 < f_start < f_stop.")
    if d_ext <= 0.0 or wall_thick <= 0.0:
        raise DividerGeometryError("d_ext and wall_thick must be > 0.")
    if d_ext <= (2.0 * wall_thick):
        raise DividerGeometryError("d_ext must be greater than 2 * wall_thick.")
    if n_sections < 1:
        raise DividerGeometryError("n_sections must be >= 1.")
    if n_outputs < 1:
        raise DividerGeometryError("n_outputs must be >= 1.")

    f0_mhz = (f_start + f_stop) * 0.5
    er_val, tan_delta = SUBSTRATE_MATERIALS[diel_material]
    wavelength_mm = 299792.458 / (f0_mhz * math.sqrt(er_val))
    sec_len_mm = wavelength_mm / 4.0
    len_inner_mm = float(n_sections) * sec_len_mm
    len_outer_mm = len_inner_mm * 1.05
    d_int_tube = d_ext - (2.0 * wall_thick)

    z0 = 50.0
    z_eff = z0 / float(n_outputs)
    z_sects = [
        z0 * (z_eff / z0) ** ((2.0 * i - 1.0) / (2.0 * float(n_sections)))
        for i in range(1, n_sections + 1)
    ]
    main_diams = [d_int_tube / math.exp(z_i * math.sqrt(er_val) / 59.952) for z_i in z_sects]
    d_out_50ohm = d_int_tube / math.exp(z0 * math.sqrt(er_val) / 59.952)
    dia_saida_diel, dia_saida_cond = _output_diameters(d_int_tube, d_out_50ohm, n_outputs)

    out = dict(p)
    out.update(
        {
            "f0_mhz": float(f0_mhz),
            "sec_len_mm": float(sec_len_mm),
            "len_inner_mm": float(len_inner_mm),
            "len_outer_mm": float(len_outer_mm),
            "d_int_tube": float(d_int_tube),
            "main_diams": [float(x) for x in main_diams],
            "z_sects": [float(x) for x in z_sects],
            "d_out_50ohm": float(d_out_50ohm),
            "dia_saida_diel": float(dia_saida_diel),
            "dia_saida_cond": float(dia_saida_cond),
            "comp_saida_mm": float(len_inner_mm * 0.1),
            "diel_er": float(er_val),
            "diel_tand": float(tan_delta),
        }
    )
    return out


def _status_emit(status_cb: Optional[Callable[[str], None]], text: str) -> None:
    msg = str(text or "").strip()
    if not msg:
        return
    if callable(status_cb):
        try:
            status_cb(msg)
        except Exception:
            pass


def _best_face_id_by_point(faces: Sequence[Any], point_xyz: Sequence[float], max_dist: float = 1e-3) -> Optional[int]:
    target = np.asarray(point_xyz, dtype=float).reshape(3)
    best = None
    best_d = float("inf")
    for face in faces:
        try:
            c = np.asarray(face.center, dtype=float).reshape(3)
            d = float(np.linalg.norm(c - target))
            if d < best_d:
                best = face
                best_d = d
        except Exception:
            continue
    if best is None or best_d > float(max_dist):
        return None
    try:
        return int(best.id)
    except Exception:
        return None


def _default_project_path(prefix: str = "Divisor_Coaxial") -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    project_dir = Path.cwd() / "HFSS_Projects"
    project_dir.mkdir(parents=True, exist_ok=True)
    return str(project_dir / f"{prefix}_{ts}.aedt")


def _build_hfss_constructor_kwargs(
    hfss_cls: Any,
    *,
    project_path: str,
    design_name: str,
    solution_type: str,
    new_desktop: bool,
    close_on_exit: bool,
) -> dict:
    """Build constructor kwargs compatible with multiple PyAEDT signatures."""
    try:
        sig = inspect.signature(hfss_cls.__init__)
        params = set(sig.parameters.keys())
    except Exception:
        params = set()

    kwargs: dict = {}
    if "projectname" in params:
        kwargs["projectname"] = str(project_path)
    elif "project" in params:
        kwargs["project"] = str(project_path)

    if "designname" in params:
        kwargs["designname"] = str(design_name)
    elif "design" in params:
        kwargs["design"] = str(design_name)

    if "solution_type" in params:
        kwargs["solution_type"] = str(solution_type)
    if "new_desktop" in params:
        kwargs["new_desktop"] = bool(new_desktop)
    if "close_on_exit" in params:
        kwargs["close_on_exit"] = bool(close_on_exit)
    return kwargs


def _alt_hfss_constructor_kwargs(base_kwargs: Mapping[str, Any]) -> Optional[dict]:
    """Build alternate keyword names for cross-version constructor fallback."""
    out = dict(base_kwargs)
    changed = False

    if "project" in out and "projectname" not in out:
        out["projectname"] = out.pop("project")
        changed = True
    elif "projectname" in out and "project" not in out:
        out["project"] = out.pop("projectname")
        changed = True

    if "design" in out and "designname" not in out:
        out["designname"] = out.pop("design")
        changed = True
    elif "designname" in out and "design" not in out:
        out["design"] = out.pop("designname")
        changed = True

    if not changed:
        return None
    return out


def _build_divider_geometry_in_hfss(
    hfss: Any,
    geom: Mapping[str, Any],
    *,
    status_cb: Optional[Callable[[str], None]] = None,
    create_setup: bool = True,
) -> None:
    mdl = hfss.modeler
    vm = hfss.variable_manager

    vm["f_start"] = f"{float(geom['f_start'])}MHz"
    vm["f_stop"] = f"{float(geom['f_stop'])}MHz"
    vm["f0"] = "(f_start+f_stop)/2"
    vm["d_ext"] = f"{float(geom['d_ext'])}mm"
    vm["wall_thick"] = f"{float(geom['wall_thick'])}mm"
    vm["n_sections"] = str(int(geom["n_sections"]))
    vm["n_outputs"] = str(int(geom["n_outputs"]))
    vm["comp_total"] = f"{float(geom['len_outer_mm'])}mm"
    vm["comp_secoes"] = f"{float(geom['len_inner_mm'])}mm"
    vm["dia_int_tubo"] = "d_ext-(2*wall_thick)"
    vm["dia_saida_diel"] = f"{geom['dia_saida_diel']}mm"
    vm["dia_saida_cond"] = f"{geom['dia_saida_cond']}mm"
    vm["comp_saida"] = f"{geom['comp_saida_mm']}mm"
    for i, d in enumerate(geom["main_diams"], start=1):
        vm[f"dia_sc{i}"] = f"{float(d)}mm"
    _status_emit(
        status_cb,
        "Design inputs loaded: "
        f"f_start={float(geom['f_start']):.6f}MHz, "
        f"f_stop={float(geom['f_stop']):.6f}MHz, "
        f"f0={float(geom['f0_mhz']):.6f}MHz",
    )

    _status_emit(status_cb, "Creating core bodies...")
    diel_principal = mdl.create_cylinder(
        "Z",
        [0, 0, 0],
        "dia_int_tubo/2",
        "comp_total",
        name="Diel_Principal",
        num_sides=0,
    )
    inner_parts = []
    for i in range(int(geom["n_sections"])):
        c = mdl.create_cylinder(
            "Z",
            [0, 0, f"{float(geom['sec_len_mm']) * i}"],
            f"dia_sc{i + 1}/2",
            f"{geom['sec_len_mm']}mm",
            name=f"Cond_Sec{i + 1}",
            num_sides=0,
        )
        inner_parts.append(c)

    output_outer_1 = mdl.create_cylinder(
        "X",
        [0, 0, "comp_secoes"],
        "dia_saida_diel/2",
        "comp_saida",
        num_sides=0,
        name="Output_Outer_1",
    )
    output_inner_1 = mdl.create_cylinder(
        "X",
        [0, 0, "comp_secoes"],
        "dia_saida_cond/2",
        "comp_saida",
        num_sides=0,
        name="Output_Inner_1",
    )

    if int(geom["n_outputs"]) > 1:
        try:
            mdl.duplicate_around_axis(
                [output_outer_1.name, output_inner_1.name],
                axis="Z",
                angle=f"360deg/{int(geom['n_outputs'])}",
                clones=int(geom["n_outputs"]),
            )
        except TypeError:
            mdl.duplicate_around_axis(
                [output_outer_1.name, output_inner_1.name],
                "Z",
                f"360deg/{int(geom['n_outputs'])}",
                int(geom["n_outputs"]),
            )

    outer_names = [diel_principal.name] + [
        str(name)
        for name in getattr(mdl, "object_names", [])
        if str(name).startswith("Output_Outer_")
    ]
    united_diel = mdl.unite(outer_names)
    united_diel_name = str(united_diel[0] if isinstance(united_diel, (list, tuple)) else united_diel)
    final_diel_obj = mdl[united_diel_name]
    final_diel_obj.name = "Volume_Diel"

    inner_names = [part.name for part in inner_parts] + [
        str(name)
        for name in getattr(mdl, "object_names", [])
        if str(name).startswith("Output_Inner_") or str(name).startswith("Cond_Saida_")
    ]
    united_cond = mdl.unite(inner_names)
    united_cond_name = str(united_cond[0] if isinstance(united_cond, (list, tuple)) else united_cond)
    final_cond_obj = mdl[united_cond_name]
    final_cond_obj.name = "Condutor_Unico"

    _status_emit(status_cb, "Assigning dielectric and conductor materials...")
    mat_name = sanitize_material_name(str(geom["diel_material"]))
    if mat_name not in hfss.materials:
        m = hfss.materials.add_material(mat_name)
        m.permittivity = float(geom["diel_er"])
        m.dielectric_loss_tangent = float(geom["diel_tand"])
    final_diel_obj.material_name = mat_name
    final_cond_obj.material_name = "pec"

    _status_emit(status_cb, "Creating wave ports...")
    p1_face = None
    for face in final_diel_obj.faces:
        try:
            if abs(float(face.center[2])) < 1e-6:
                p1_face = int(face.id)
                break
        except Exception:
            continue
    if p1_face is not None:
        hfss.wave_port(assignment=p1_face, name="P1", renormalize=True, impedance="50")

    output_len = float(geom["comp_saida_mm"])
    for k in range(int(geom["n_outputs"])):
        angle_rad = math.radians((360.0 * float(k)) / float(geom["n_outputs"]))
        px = output_len * math.cos(angle_rad)
        py = output_len * math.sin(angle_rad)
        pz = float(geom["len_inner_mm"])
        fid = _best_face_id_by_point(final_diel_obj.faces, (px, py, pz), max_dist=1e-3)
        if fid is not None:
            hfss.wave_port(assignment=fid, name=f"P{k + 2}", renormalize=True, impedance="50")

    if bool(create_setup):
        _status_emit(status_cb, "Creating HFSS setup and frequency sweep...")
        setup_name = "Setup1"
        if setup_name in getattr(hfss, "setup_names", []):
            setup = hfss.get_setup(setup_name)
        else:
            setup = hfss.create_setup(setup_name)

        setup.props["Frequency"] = "f0"
        setup.props["MaximumPasses"] = 8
        setup.props["MaxDeltaS"] = 0.02
        try:
            setup.update()
        except Exception:
            pass

        sweep_name = "Sweep1"
        try:
            sweep_names = set(setup.get_sweep_names())
            if sweep_name in sweep_names:
                setup.delete_sweep(sweep_name)
        except Exception:
            pass

        setup.create_frequency_sweep(
            unit="MHz",
            start_frequency=float(geom["f_start"]),
            stop_frequency=float(geom["f_stop"]),
            num_of_freq_points=201,
            name=sweep_name,
            sweep_type="Fast",
            save_fields=False,
        )
        _status_emit(
            status_cb,
            "Sweep configured: "
            f"{float(geom['f_start']):.6f}MHz -> {float(geom['f_stop']):.6f}MHz, 201 points (Fast).",
        )

    hfss.save_project()


def generate_hfss_divider_model(
    params: Mapping[str, Any],
    *,
    status_cb: Optional[Callable[[str], None]] = None,
    project_path: Optional[str] = None,
    create_setup: bool = True,
    new_desktop: bool = True,
    close_on_exit: bool = False,
) -> Optional[str]:
    """Generate the divider geometry in AEDT/HFSS using PyAEDT."""
    try:
        from ansys.aedt.core import Hfss  # type: ignore
    except Exception as exc:
        raise DividerGeometryError(
            "ansys-aedt-core (PyAEDT) is not available. "
            "Install requirements and ensure AEDT is installed."
        ) from exc

    p = compute_coaxial_divider_geometry(params)
    target_path = str(project_path or _default_project_path())
    _status_emit(status_cb, "Starting HFSS divider geometry generation...")
    _status_emit(status_cb, f"Target AEDT project: {target_path}")

    hfss_kwargs = _build_hfss_constructor_kwargs(
        Hfss,
        project_path=target_path,
        design_name="DivisorCoaxial",
        solution_type="Modal",
        new_desktop=bool(new_desktop),
        close_on_exit=bool(close_on_exit),
    )
    hfss_kwargs_alt = _alt_hfss_constructor_kwargs(hfss_kwargs)

    ctor_attempts = [hfss_kwargs]
    if hfss_kwargs_alt:
        ctor_attempts.append(hfss_kwargs_alt)
    last_exc: Optional[Exception] = None

    for idx, kwargs in enumerate(ctor_attempts, start=1):
        _status_emit(status_cb, f"HFSS constructor attempt {idx}: {sorted(kwargs.keys())}")
        try:
            with Hfss(**kwargs) as hfss:
                _build_divider_geometry_in_hfss(
                    hfss,
                    p,
                    status_cb=status_cb,
                    create_setup=bool(create_setup),
                )
                _status_emit(status_cb, f"HFSS divider geometry saved: {target_path}")
                return target_path
        except TypeError as exc:
            last_exc = exc
            _status_emit(status_cb, f"HFSS constructor TypeError: {exc}")
            continue

    if last_exc is not None:
        raise DividerGeometryError(
            "Failed to construct Hfss session with compatible argument names. "
            f"Last error: {last_exc}"
        ) from last_exc
    raise DividerGeometryError("Failed to construct Hfss session.")
