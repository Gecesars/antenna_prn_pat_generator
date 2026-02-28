"""Coaxial divider geometry + AEDT/HFSS model generation helpers.

This module mirrors the geometry logic from ``exemplo_divisor.md`` while
keeping the UI layer independent from AEDT runtime details.
"""

from __future__ import annotations

import math
import os
import re
import inspect
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

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


def _output_diel_diameter_mm(
    d_int_tube_mm: float,
    n_outputs: int,
    len_inner_mm: Optional[float] = None,
    min_hf_scale: float = 0.70,
    ref_len_to_diam_ratio: float = 10.0,
) -> float:
    """Output dielectric diameter logic mirrored from Calc_Div_EFTX.py."""
    if int(n_outputs) > 6:
        base_fraction = 0.25
    elif int(n_outputs) > 4:
        base_fraction = 0.32
    else:
        base_fraction = 0.40

    if len_inner_mm is None:
        hf_scale = 1.0
    else:
        length_ratio = float(len_inner_mm) / max(float(d_int_tube_mm), 1e-9)
        normalized_ratio = max(min(length_ratio / float(ref_len_to_diam_ratio), 1.0), 0.0)
        hf_scale = float(min_hf_scale) + (1.0 - float(min_hf_scale)) * math.sqrt(normalized_ratio)
    return float(d_int_tube_mm) * float(base_fraction) * float(hf_scale)


def _output_length_mm(
    len_inner_mm: float,
    d_int_tube_mm: float,
    base_ratio: float = 0.1,
    min_tube_diam_factor: float = 1.0,
) -> float:
    """Output branch length logic mirrored from Calc_Div_EFTX.py."""
    return max(float(len_inner_mm) * float(base_ratio), float(d_int_tube_mm) * float(min_tube_diam_factor))


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
    len_outer_mm = (len_inner_mm * 1.05) * 1.10
    d_int_tube = d_ext - (2.0 * wall_thick)

    z0 = 50.0
    z_eff = z0 / float(n_outputs)
    z_sects = [
        z0 * (z_eff / z0) ** ((2.0 * i - 1.0) / (2.0 * float(n_sections)))
        for i in range(1, n_sections + 1)
    ]
    main_diams = [d_int_tube / math.exp(z_i * math.sqrt(er_val) / 59.952) for z_i in z_sects]
    d_out_50ohm = d_int_tube / math.exp(z0 * math.sqrt(er_val) / 59.952)
    dia_saida_diel = _output_diel_diameter_mm(
        d_int_tube_mm=d_int_tube,
        n_outputs=n_outputs,
        len_inner_mm=len_inner_mm,
    )
    ratio_inner_outer = float(d_out_50ohm) / max(float(d_int_tube), 1e-9)
    dia_saida_cond = float(dia_saida_diel) * float(ratio_inner_outer)
    comp_saida_mm = _output_length_mm(
        len_inner_mm=len_inner_mm,
        d_int_tube_mm=d_int_tube,
    )

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
            "comp_saida_mm": float(comp_saida_mm),
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
    for project_dir in _candidate_project_dirs():
        if _is_writable_dir(project_dir):
            return str(project_dir / f"{prefix}_{ts}.aedt")
    raise DividerGeometryError(
        "Unable to find a writable folder for HFSS projects. "
        "Set EFTX_HFSS_PROJECT_DIR or choose a writable path in 'Save AEDT as...'."
    )


def _candidate_project_dirs() -> Sequence[Path]:
    candidates: list[Path] = []

    env_path = str(os.getenv("EFTX_HFSS_PROJECT_DIR") or "").strip()
    if env_path:
        candidates.append(Path(env_path).expanduser())

    local_app_data = str(os.getenv("LOCALAPPDATA") or "").strip()
    if local_app_data:
        candidates.append(Path(local_app_data) / "EFTX" / "DiagramSuite" / "HFSS_Projects")

    home = Path.home()
    candidates.append(home / "Documents" / "EFTX" / "DiagramSuite" / "HFSS_Projects")
    candidates.append(home / "AppData" / "Local" / "EFTX" / "DiagramSuite" / "HFSS_Projects")
    candidates.append(Path.cwd() / "HFSS_Projects")

    unique: list[Path] = []
    seen: set[str] = set()
    for p in candidates:
        key = str(p).lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(p)
    return unique


def _is_writable_dir(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception:
        return False
    probe = path / ".eftx_write_test.tmp"
    try:
        with probe.open("w", encoding="utf-8") as f:
            f.write("ok")
        try:
            probe.unlink()
        except Exception:
            pass
        return True
    except Exception:
        try:
            if probe.exists():
                probe.unlink()
        except Exception:
            pass
        return False


def _normalize_project_target(project_path: Optional[str]) -> tuple[str, str]:
    """Return (project_file_path, project_directory_path)."""
    raw = str(project_path or "").strip()
    if not raw:
        return "", ""

    candidate = Path(raw).expanduser()
    if candidate.suffix.lower() == ".aedt":
        return str(candidate), str(candidate.parent)

    if candidate.exists() and candidate.is_dir():
        return "", str(candidate)

    return str(candidate), str(candidate.parent)


def _safe_current_project_file(hfss: Any) -> str:
    for attr in ("project_file", "_project_file"):
        try:
            value = str(getattr(hfss, attr, "") or "").strip()
            if value:
                return value
        except Exception:
            continue

    try:
        pdir = str(getattr(hfss, "project_path", "") or "").strip()
        pname = str(getattr(hfss, "project_name", "") or "").strip()
        if pdir and pname:
            if pdir.lower().endswith(".aedt"):
                return pdir
            return str(Path(pdir) / f"{pname}.aedt")
    except Exception:
        pass
    return ""


def _paths_equal(left: str, right: str) -> bool:
    a = str(left or "").strip()
    b = str(right or "").strip()
    if not a or not b:
        return False
    try:
        pa = Path(a).expanduser().resolve(strict=False)
        pb = Path(b).expanduser().resolve(strict=False)
        return str(pa).lower() == str(pb).lower()
    except Exception:
        return a.lower() == b.lower()


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


def _build_hfss_constructor_attempts(
    hfss_cls: Any,
    *,
    project_path: str,
    design_name: str = "DivisorCoaxial",
    solution_type: str = "Modal",
    new_desktop: bool = True,
    close_on_exit: bool = False,
) -> list[dict]:
    kwargs = _build_hfss_constructor_kwargs(
        hfss_cls,
        project_path=project_path,
        design_name=design_name,
        solution_type=solution_type,
        new_desktop=bool(new_desktop),
        close_on_exit=bool(close_on_exit),
    )
    out = [kwargs]
    alt = _alt_hfss_constructor_kwargs(kwargs)
    if alt:
        out.append(alt)
    return out


def _safe_setup_names(hfss: Any) -> list[str]:
    try:
        raw = getattr(hfss, "setup_names", None)
        if not raw:
            return []
        return [str(x) for x in list(raw)]
    except Exception:
        return []


def _release_hfss_reference(hfss: Any, status_cb: Optional[Callable[[str], None]] = None) -> None:
    # Keep session alive to avoid destabilizing concurrent AEDT connections.
    # PyAEDT 0.25.0 may close the shared desktop even with close_desktop=False.
    _ = hfss
    _ = status_cb
    return


def _run_setup_with_fallback(
    hfss: Any,
    setup_name: str,
    *,
    status_cb: Optional[Callable[[str], None]] = None,
) -> bool:
    token = str(setup_name or "Setup1")
    errors: list[str] = []

    try:
        out = hfss.analyze_setup(token, blocking=True)
        if bool(out):
            return True
    except Exception as exc:
        errors.append(f"analyze_setup('{token}') -> {exc}")

    try:
        # Fallback for environments where setup_names is None internally.
        out = hfss.odesign.Analyze(token)
        if out is None or bool(out):
            return True
    except Exception as exc:
        errors.append(f"odesign.Analyze('{token}') -> {exc}")

    try:
        out = hfss.odesign.AnalyzeAll()
        if out is None or bool(out):
            return True
    except Exception as exc:
        errors.append(f"odesign.AnalyzeAll() -> {exc}")

    _status_emit(status_cb, "Setup execution failed on all fallback paths.")
    for item in errors:
        _status_emit(status_cb, item)
    return False


def _ensure_project_design_context(
    hfss: Any,
    *,
    target_path: Optional[str] = None,
    design_name: Optional[str] = None,
    status_cb: Optional[Callable[[str], None]] = None,
) -> None:
    target_file, target_dir = _normalize_project_target(target_path)
    dname = str(design_name or "").strip()

    try:
        current_file = _safe_current_project_file(hfss)
        needs_load = (
            bool(target_file)
            and Path(target_file).exists()
            and (not current_file or not _paths_equal(current_file, target_file))
        )
        if needs_load:
            hfss.load_project(target_file, design=dname or None, close_active=False, set_active=True)
            _status_emit(status_cb, f"Project context restored from: {target_file}")
        elif target_file and (getattr(hfss, "oproject", None) is None) and Path(target_file).exists():
            hfss.load_project(target_file, design=dname or None, close_active=False, set_active=True)
            _status_emit(status_cb, f"Project context restored from: {target_file}")
    except Exception as exc:
        _status_emit(status_cb, f"Project context restore warning: {exc}")

    try:
        if dname and str(getattr(hfss, "design_name", "") or "").strip() != dname:
            hfss.set_active_design(dname)
    except Exception:
        pass

    if target_dir:
        try:
            hfss._project_path = target_dir
        except Exception:
            pass
    if target_file:
        try:
            hfss._project_file = target_file
        except Exception:
            pass


def _check_setup_sweep_exists(hfss: Any, setup_name: str, sweep_name: str) -> tuple[bool, bool]:
    setup_ok = False
    sweep_ok = False
    setup_names = _safe_setup_names(hfss)
    if str(setup_name) in setup_names:
        setup_ok = True
        try:
            sw = hfss.get_setup(str(setup_name))
            sweep_ok = str(sweep_name) in set(sw.get_sweep_names())
        except Exception:
            sweep_ok = False
    return setup_ok, sweep_ok


def _cleanup_existing_divider_entities(hfss: Any, status_cb: Optional[Callable[[str], None]] = None) -> None:
    mdl = hfss.modeler
    prefixes = (
        "Diel_Principal",
        "Cond_Sec",
        "Output_Outer_",
        "Output_Inner_",
        "Cond_Saida_",
    )
    explicit_names = {"Volume_Diel", "Condutor_Unico"}
    to_delete = []
    for name in list(getattr(mdl, "object_names", [])):
        token = str(name)
        if token in explicit_names or token.startswith(prefixes):
            to_delete.append(token)
    if to_delete:
        try:
            mdl.delete(to_delete)
            _status_emit(status_cb, f"Removed existing divider solids: {len(to_delete)}")
        except Exception:
            pass

    for b in list(getattr(hfss, "boundaries", []) or []):
        try:
            bname = str(getattr(b, "name", "") or "")
            if re.match(r"^P\d+$", bname):
                b.delete()
        except Exception:
            continue


def compute_s11_metrics(s11_complex: Sequence[complex], z0: float = 50.0) -> dict:
    s11 = np.asarray(s11_complex, dtype=complex).reshape(-1)
    mag = np.clip(np.abs(s11), 1e-15, None)
    rl_db = -20.0 * np.log10(mag)
    den = 1.0 - s11
    den = np.where(np.abs(den) < 1e-12, 1e-12 + 0j, den)
    zin = float(z0) * (1.0 + s11) / den
    return {
        "return_loss_db": rl_db,
        "impedance_ohm": np.abs(zin),
        "impedance_real_ohm": np.real(zin),
        "impedance_imag_ohm": np.imag(zin),
        "phase_deg": np.degrees(np.angle(s11)),
    }


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def analyze_impedance_adjustments(
    rf_data: Mapping[str, Any],
    params: Mapping[str, Any],
    *,
    z0: float = 50.0,
) -> list[dict]:
    """Analyze complex impedance and suggest viable variable adjustments.

    Returns dictionaries with:
    ``variable``, ``current``, ``suggested``, ``delta``, ``priority``,
    ``confidence``, ``rationale``.

    Automatic adjustment is intentionally restricted to geometry variables
    safe for impedance auto-tune: ``d_ext_mm`` and ``wall_thick_mm``.
    """
    freq = np.asarray(rf_data.get("frequency", []), dtype=float).reshape(-1)
    zr = np.asarray(rf_data.get("impedance_real_ohm", []), dtype=float).reshape(-1)
    zi = np.asarray(rf_data.get("impedance_imag_ohm", []), dtype=float).reshape(-1)

    if len(freq) < 2 or len(zr) < 2 or len(zi) < 2:
        return []

    n = min(len(freq), len(zr), len(zi))
    freq = freq[:n]
    zr = zr[:n]

    f_start = float(params.get("f_start", float(freq[0])))
    f_stop = float(params.get("f_stop", float(freq[-1])))
    f0 = 0.5 * (f_start + f_stop)
    i0 = int(np.argmin(np.abs(freq - f0)))

    z_real_0 = float(zr[i0])
    mismatch_real = z_real_0 - float(z0)

    d_ext = float(params.get("d_ext", 20.0))
    wall = float(params.get("wall_thick", 1.5))
    step_d = _clamp(abs(mismatch_real) * 0.01, 0.05, 0.8)
    step_wall = _clamp(abs(mismatch_real) * 0.005, 0.03, 0.4)

    suggestions: list[dict] = []

    # d_ext: larger d_ext tends to increase characteristic impedance.
    d_ext_new = d_ext - step_d if mismatch_real > 0 else d_ext + step_d
    d_ext_new = _clamp(d_ext_new, (2.0 * wall) + 0.25, 500.0)
    suggestions.append(
        {
            "variable": "d_ext_mm",
            "current": d_ext,
            "suggested": d_ext_new,
            "delta": d_ext_new - d_ext,
            "priority": 1,
            "confidence": "medium",
            "rationale": (
                f"Zreal@f0={z_real_0:.3f}Ohm vs target {float(z0):.1f}Ohm. "
                "Adjusting outer diameter helps move real impedance toward target."
            ),
        }
    )

    # wall_thick: thicker wall reduces inner diameter, generally reducing impedance.
    wall_new = wall + step_wall if mismatch_real > 0 else wall - step_wall
    wall_new = _clamp(wall_new, 0.1, max(0.1, (d_ext / 2.0) - 0.2))
    suggestions.append(
        {
            "variable": "wall_thick_mm",
            "current": wall,
            "suggested": wall_new,
            "delta": wall_new - wall,
            "priority": 1,
            "confidence": "medium",
            "rationale": "Wall thickness adjustment complements d_ext to tune real impedance matching.",
        }
    )

    suggestions.sort(key=lambda item: (int(item.get("priority", 99)), str(item.get("variable", ""))))
    return suggestions
def _extract_s11_metrics_from_hfss(
    hfss: Any,
    *,
    setup_name: str = "Setup1",
    sweep_name: str = "Sweep1",
    port_name: str = "P1",
    project_path: Optional[str] = None,
    status_cb: Optional[Callable[[str], None]] = None,
) -> dict:
    target_file, target_dir = _normalize_project_target(project_path)
    _ensure_project_design_context(
        hfss,
        target_path=target_file or project_path,
        design_name=str(getattr(hfss, "design_name", "") or ""),
        status_cb=status_cb,
    )
    proj = str(getattr(hfss, "project_path", "") or "").strip()

    # PyAEDT post-processing expects project_path to be a folder, not *.aedt file.
    if proj.lower().endswith(".aedt"):
        fixed_dir = str(Path(proj).expanduser().parent)
        try:
            hfss._project_path = fixed_dir
            proj = fixed_dir
            _status_emit(status_cb, f"Adjusted HFSS project_path to folder context: {fixed_dir}")
        except Exception:
            pass

    if not proj and target_file:
        try:
            hfss.save_project(file_name=target_file, overwrite=True)
            try:
                if target_dir:
                    hfss._project_path = target_dir
                hfss._project_file = target_file
            except Exception:
                pass
            _status_emit(status_cb, f"Project path was empty; forced save to {target_file}.")
        except Exception as exc:
            _status_emit(status_cb, f"Could not force-save project path before post extraction: {exc}")
    elif target_dir and (not _paths_equal(proj, target_dir)):
        try:
            hfss._project_path = target_dir
        except Exception:
            pass

    setup_candidates = [f"{setup_name} : {sweep_name}", f"{setup_name} : LastAdaptive", setup_name]
    token = str(port_name or "P1").strip() or "P1"
    expr_candidates: list[str] = [f"S({token},{token})"]
    match = re.match(r"^[Pp](\d+)$", token)
    if match:
        idx = match.group(1)
        expr_candidates.append(f"S({idx},{idx})")
    for fallback_expr in ("S(P1,P1)", "S(1,1)"):
        if fallback_expr not in expr_candidates:
            expr_candidates.append(fallback_expr)
    last_exc: Optional[Exception] = None

    for expr in expr_candidates:
        for setup_sweep_name in setup_candidates:
            try:
                sol = hfss.post.get_solution_data(
                    expressions=[expr],
                    setup_sweep_name=setup_sweep_name,
                    domain="Sweep",
                    report_category="Modal Solution Data",
                )
                if not sol:
                    continue

                # PyAEDT get_expression_data returns (x_values, y_values).
                x_real, y_real = sol.get_expression_data(expression=expr, formula="real")
                x_imag, y_imag = sol.get_expression_data(expression=expr, formula="imag")

                if x_real is None or y_real is None or x_imag is None or y_imag is None:
                    continue

                freq = np.asarray(x_real, dtype=float).reshape(-1)
                real = np.asarray(y_real, dtype=float).reshape(-1)
                imag = np.asarray(y_imag, dtype=float).reshape(-1)

                if len(freq) == 0:
                    freq = np.asarray(sol.primary_sweep_values, dtype=float).reshape(-1)
                if len(freq) == 0 or len(real) == 0 or len(imag) == 0:
                    continue
                n = min(len(freq), len(real), len(imag))
                freq = freq[:n]
                real = real[:n]
                imag = imag[:n]
                s11 = real + 1j * imag
                metrics = compute_s11_metrics(s11, z0=50.0)
                sweep_var = str(getattr(sol, "primary_sweep", "") or "")
                sweep_unit = str(getattr(sol, "units_sweeps", {}).get(sweep_var, "") or "MHz")
                _status_emit(status_cb, f"RF curves extracted with {expr} on '{setup_sweep_name}'.")
                return {
                    "expression": expr,
                    "setup_sweep_name": setup_sweep_name,
                    "frequency": freq.tolist(),
                    "frequency_unit": sweep_unit,
                    "s11_real": real.tolist(),
                    "s11_imag": imag.tolist(),
                    "return_loss_db": metrics["return_loss_db"].tolist(),
                    "impedance_ohm": metrics["impedance_ohm"].tolist(),
                    "impedance_real_ohm": metrics["impedance_real_ohm"].tolist(),
                    "impedance_imag_ohm": metrics["impedance_imag_ohm"].tolist(),
                    "phase_deg": metrics["phase_deg"].tolist(),
                }
            except Exception as exc:
                last_exc = exc
                continue

    if last_exc is not None:
        raise DividerGeometryError(
            "Could not extract S11 solution data from HFSS. "
            f"Last error: {last_exc}"
        ) from last_exc
    raise DividerGeometryError("Could not extract S11 solution data from HFSS.")


def _build_divider_geometry_in_hfss(
    hfss: Any,
    geom: Mapping[str, Any],
    *,
    status_cb: Optional[Callable[[str], None]] = None,
    create_setup: bool = True,
    rebuild_geometry: bool = True,
    setup_name: str = "Setup1",
    sweep_name: str = "Sweep1",
    save_project_path: Optional[str] = None,
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
    vm["sec_len"] = f"{float(geom['sec_len_mm'])}mm"
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

    names_now = {str(n) for n in getattr(mdl, "object_names", [])}
    has_base_geometry = {"Volume_Diel", "Condutor_Unico"}.issubset(names_now)
    do_rebuild = bool(rebuild_geometry)
    if (not do_rebuild) and (not has_base_geometry):
        _status_emit(status_cb, "Existing divider geometry not found. Switching to rebuild mode.")
        do_rebuild = True

    if do_rebuild:
        _cleanup_existing_divider_entities(hfss, status_cb=status_cb)
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
                [0, 0, f"{i}*sec_len"],
                f"dia_sc{i + 1}/2",
                "sec_len",
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
    else:
        _status_emit(status_cb, "Applying variable-only update (no geometry rebuild).")
        final_diel_obj = mdl["Volume_Diel"]
        final_cond_obj = mdl["Condutor_Unico"]

    _status_emit(status_cb, "Assigning dielectric and conductor materials...")
    mat_name = sanitize_material_name(str(geom["diel_material"]))
    if mat_name not in hfss.materials:
        m = hfss.materials.add_material(mat_name)
        m.permittivity = float(geom["diel_er"])
        m.dielectric_loss_tangent = float(geom["diel_tand"])
    final_diel_obj.material_name = mat_name
    final_cond_obj.material_name = "pec"

    if do_rebuild:
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
    else:
        _status_emit(status_cb, "Keeping existing wave ports (variable-only mode).")

    if bool(create_setup):
        _status_emit(status_cb, "Creating HFSS setup and frequency sweep...")
        setup_names = _safe_setup_names(hfss)
        if setup_name in setup_names:
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

    if str(save_project_path or "").strip():
        hfss.save_project(file_name=str(save_project_path), overwrite=True)


def generate_hfss_divider_model(
    params: Mapping[str, Any],
    *,
    status_cb: Optional[Callable[[str], None]] = None,
    project_path: Optional[str] = None,
    design_name: str = "DivisorCoaxial",
    setup_name: str = "Setup1",
    sweep_name: str = "Sweep1",
    create_setup: bool = True,
    rebuild_geometry: bool = True,
    new_desktop: bool = False,
    close_on_exit: bool = False,
) -> Optional[str]:
    """Generate the divider geometry in AEDT/HFSS using PyAEDT."""
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r".*Graphics dependencies are required.*",
                category=UserWarning,
            )
            from ansys.aedt.core import Hfss  # type: ignore
    except Exception as exc:
        raise DividerGeometryError(
            "ansys-aedt-core (PyAEDT) is not available. "
            "Install requirements and ensure AEDT is installed."
        ) from exc

    p = compute_coaxial_divider_geometry(params)
    if project_path:
        target = Path(str(project_path)).expanduser()
        parent_dir = target.parent if str(target.parent).strip() else Path.cwd()
        if not _is_writable_dir(parent_dir):
            raise DividerGeometryError(
                f"Selected project folder is not writable: {parent_dir}. "
                "Choose another location or run with elevated permissions."
            )
        target_path = str(target)
    else:
        target_path = str(_default_project_path())
    _status_emit(status_cb, "Starting HFSS divider geometry generation...")
    _status_emit(status_cb, f"Target AEDT project: {target_path}")
    _status_emit(status_cb, f"Target HFSS design: {design_name}")

    ctor_attempts = _build_hfss_constructor_attempts(
        Hfss,
        project_path=target_path,
        design_name=str(design_name),
        solution_type="Modal",
        new_desktop=bool(new_desktop),
        close_on_exit=bool(close_on_exit),
    )
    last_exc: Optional[Exception] = None

    for idx, kwargs in enumerate(ctor_attempts, start=1):
        _status_emit(status_cb, f"HFSS constructor attempt {idx}: {sorted(kwargs.keys())}")
        hfss = None
        try:
            hfss = Hfss(**kwargs)
        except TypeError as exc:
            last_exc = exc
            _status_emit(status_cb, f"HFSS constructor TypeError: {exc}")
            continue
        try:
            _build_divider_geometry_in_hfss(
                hfss,
                p,
                status_cb=status_cb,
                create_setup=bool(create_setup),
                rebuild_geometry=bool(rebuild_geometry),
                setup_name=str(setup_name),
                sweep_name=str(sweep_name),
                save_project_path=target_path,
            )
            _status_emit(status_cb, f"HFSS divider geometry saved: {target_path}")
            return target_path
        finally:
            _release_hfss_reference(hfss, status_cb=status_cb)

    if last_exc is not None:
        raise DividerGeometryError(
            "Failed to construct Hfss session with compatible argument names. "
            f"Last error: {last_exc}"
        ) from last_exc
    raise DividerGeometryError("Failed to construct Hfss session.")


def run_hfss_divider_analysis(
    params: Mapping[str, Any],
    *,
    status_cb: Optional[Callable[[str], None]] = None,
    project_path: Optional[str] = None,
    design_name: str = "DivisorCoaxial",
    setup_name: str = "Setup1",
    sweep_name: str = "Sweep1",
    port_name: str = "P1",
    rebuild_geometry: bool = True,
    new_desktop: bool = False,
    close_on_exit: bool = False,
) -> dict:
    """Create/update geometry, run setup, and return RF curves for plotting."""
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r".*Graphics dependencies are required.*",
                category=UserWarning,
            )
            from ansys.aedt.core import Hfss  # type: ignore
    except Exception as exc:
        raise DividerGeometryError(
            "ansys-aedt-core (PyAEDT) is not available. "
            "Install requirements and ensure AEDT is installed."
        ) from exc

    p = compute_coaxial_divider_geometry(params)
    if project_path:
        target = Path(str(project_path)).expanduser()
        parent_dir = target.parent if str(target.parent).strip() else Path.cwd()
        if not _is_writable_dir(parent_dir):
            raise DividerGeometryError(
                f"Selected project folder is not writable: {parent_dir}. "
                "Choose another location or run with elevated permissions."
            )
        target_path = str(target)
    else:
        target_path = str(_default_project_path())

    _status_emit(status_cb, "Starting HFSS analysis pipeline for divider...")
    _status_emit(status_cb, f"Target AEDT project: {target_path}")
    _status_emit(status_cb, f"Target HFSS design: {design_name}")
    ctor_attempts = _build_hfss_constructor_attempts(
        Hfss,
        project_path=target_path,
        design_name=str(design_name),
        solution_type="Modal",
        new_desktop=bool(new_desktop),
        close_on_exit=bool(close_on_exit),
    )

    last_exc: Optional[Exception] = None
    for idx, kwargs in enumerate(ctor_attempts, start=1):
        _status_emit(status_cb, f"HFSS constructor attempt {idx}: {sorted(kwargs.keys())}")
        hfss = None
        try:
            hfss = Hfss(**kwargs)
        except TypeError as exc:
            last_exc = exc
            _status_emit(status_cb, f"HFSS constructor TypeError: {exc}")
            continue
        try:
            _ensure_project_design_context(
                hfss,
                target_path=target_path,
                design_name=str(design_name),
                status_cb=status_cb,
            )
            setup_exists, sweep_exists = _check_setup_sweep_exists(
                hfss,
                str(setup_name),
                str(sweep_name),
            )
            create_or_refresh_setup = bool(rebuild_geometry) or (not setup_exists) or (not sweep_exists)
            if create_or_refresh_setup:
                _status_emit(
                    status_cb,
                    "Setup/Sweep will be refreshed: "
                    f"setup_exists={setup_exists}, sweep_exists={sweep_exists}, rebuild={bool(rebuild_geometry)}",
                )
            else:
                _status_emit(status_cb, "Reusing existing Setup/Sweep (no rebuild).")

            _build_divider_geometry_in_hfss(
                hfss,
                p,
                status_cb=status_cb,
                create_setup=bool(create_or_refresh_setup),
                rebuild_geometry=bool(rebuild_geometry),
                setup_name=str(setup_name),
                sweep_name=str(sweep_name),
                save_project_path=target_path,
            )
            _status_emit(status_cb, f"Running setup '{setup_name}'...")
            _ensure_project_design_context(
                hfss,
                target_path=target_path,
                design_name=str(design_name),
                status_cb=status_cb,
            )
            solved = _run_setup_with_fallback(
                hfss,
                str(setup_name),
                status_cb=status_cb,
            )
            if not solved:
                raise DividerGeometryError(f"HFSS setup '{setup_name}' did not complete successfully.")

            rf = _extract_s11_metrics_from_hfss(
                hfss,
                setup_name=setup_name,
                sweep_name=sweep_name,
                port_name=port_name,
                project_path=target_path,
                status_cb=status_cb,
            )
            hfss.save_project(file_name=target_path, overwrite=True)
            _status_emit(status_cb, f"HFSS analysis completed and saved: {target_path}")
            out = dict(rf)
            out["project_path"] = target_path
            out["design_name"] = str(design_name)
            out["port_name"] = str(port_name)
            out["f_start"] = float(p["f_start"])
            out["f_stop"] = float(p["f_stop"])
            out["f0_mhz"] = float(p["f0_mhz"])
            return out
        finally:
            _release_hfss_reference(hfss, status_cb=status_cb)

    if last_exc is not None:
        raise DividerGeometryError(
            "Failed to construct Hfss session with compatible argument names. "
            f"Last error: {last_exc}"
        ) from last_exc
    raise DividerGeometryError("Failed to run HFSS divider analysis.")
