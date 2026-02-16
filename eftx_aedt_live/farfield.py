from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import importlib
import numpy as np

from .session import AedtHfssSession


AngleList = List[float]
FloatList = List[float]


def _setup_name_from_sweep(setup_or_sweep: str) -> str:
    txt = str(setup_or_sweep or "").strip().strip('"')
    if not txt:
        return ""
    if ":" in txt:
        return txt.split(":", 1)[0].strip()
    return txt


def _unique_str(values: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for raw in values:
        v = str(raw or "").strip()
        if not v:
            continue
        k = v.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(v)
    return out


def _patch_solution_constants_alias() -> None:
    """Compatibility patch for PyAEDT versions missing `default_solution` constants."""
    for mod_name in ("ansys.aedt.core.generic.aedt_constants", "pyaedt.generic.aedt_constants"):
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            continue
        try:
            for name in dir(mod):
                obj = getattr(mod, name, None)
                if not isinstance(obj, type):
                    continue
                if hasattr(obj, "solution_default") and (not hasattr(obj, "default_solution")):
                    try:
                        setattr(obj, "default_solution", getattr(obj, "solution_default"))
                    except Exception:
                        pass
        except Exception:
            continue


def _as_deg_str(v: Union[float, int, str]) -> str:
    """Normalize angle-like values to AEDT-friendly strings when needed."""
    if isinstance(v, str):
        return v
    return f"{float(v)}deg"


def theta_to_elev(theta_deg: np.ndarray) -> np.ndarray:
    """Convert HFSS Theta into elevation domain [-90, +90] robustly.

    Important:
      - If theta already looks like elevation (-90..90), keep it as-is.
      - If theta is classical HFSS (0..180), use elev = 90 - theta.
      - Otherwise, choose the candidate that best fits [-90..90].
    """
    th = np.asarray(theta_deg, dtype=float)
    if th.size == 0:
        return th

    finite = np.isfinite(th)
    if not np.any(finite):
        return np.asarray(th, dtype=float)
    mn = float(np.min(th[finite]))
    mx = float(np.max(th[finite]))

    # Common case in this app: pull already comes in elevation-like theta.
    if mn >= -95.0 and mx <= 95.0:
        return np.asarray(th, dtype=float)

    # Classic HFSS theta convention.
    if mn >= -5.0 and mx <= 185.0:
        return 90.0 - th

    # Fallback to legacy robust conversion for ambiguous theta domains.
    out = 90.0 - th
    out = (out + 180.0) % 360.0 - 180.0
    hi = out > 90.0
    lo = out < -90.0
    if np.any(hi):
        out[hi] = 180.0 - out[hi]
    if np.any(lo):
        out[lo] = -180.0 - out[lo]
    return out


def unwrap_phi(phi_deg: np.ndarray, domain: str = "-180..180") -> np.ndarray:
    """Normalize phi to a consistent domain."""
    ph = np.asarray(phi_deg, dtype=float)
    if domain == "0..360":
        ph = np.mod(ph, 360.0)
        ph[ph < 0] += 360.0
    else:
        # -180..180
        ph = (ph + 180.0) % 360.0 - 180.0
    return ph


def _coerce_cut_values(x_vals: np.ndarray, y_vals: np.ndarray, expression: str = "") -> np.ndarray:
    """Coerce solution data returned by AEDT into a 1D vector aligned with `x_vals`.

    HFSS/PyAEDT can return cut values as:
      - (N,)
      - (N, 1) / (1, N)
      - (2, N) or (N, 2) (common when extra channels are returned)
      - flattened arrays with repeated traces.
    """
    x = np.asarray(x_vals, dtype=float).reshape(-1)
    y = np.asarray(y_vals, dtype=float)
    n = int(x.size)
    if n <= 0:
        return np.asarray([], dtype=float)

    def _is_axis_like(vec: np.ndarray) -> bool:
        vv = np.asarray(vec, dtype=float).reshape(-1)
        if vv.size != n:
            return False
        return bool(np.allclose(vv, x, rtol=1e-6, atol=1e-6))

    if y.ndim == 0:
        return np.full((n,), float(y), dtype=float)

    if y.ndim == 1:
        if y.size == n:
            return y.astype(float, copy=False)
        if y.size > n:
            # If multiple traces are flattened, keep the first complete trace.
            return y[:n].astype(float, copy=False)
        if y.size > 1:
            src_x = np.linspace(0.0, 1.0, int(y.size))
            dst_x = np.linspace(0.0, 1.0, n)
            return np.interp(dst_x, src_x, y).astype(float, copy=False)
        return np.full((n,), float(y[0]) if y.size else 0.0, dtype=float)

    # Common PyAEDT case: matrix packs [x, y] (or [y, x]).
    if y.ndim == 2:
        if y.shape == (2, n):
            r0 = np.asarray(y[0], dtype=float)
            r1 = np.asarray(y[1], dtype=float)
            if _is_axis_like(r0):
                return r1
            if _is_axis_like(r1):
                return r0
        if y.shape == (n, 2):
            c0 = np.asarray(y[:, 0], dtype=float)
            c1 = np.asarray(y[:, 1], dtype=float)
            if _is_axis_like(c0):
                return c1
            if _is_axis_like(c1):
                return c0

    # Try to find the axis that matches x-size and project remaining dims.
    match_axes = [ax for ax, sz in enumerate(y.shape) if int(sz) == n]
    if match_axes:
        ax = match_axes[0]
        aligned = np.moveaxis(y, ax, 0)  # shape: (n, ...)
        tail = aligned.reshape(n, -1)
        if tail.shape[1] == 1:
            out = tail[:, 0]
        elif tail.shape[1] == 2 and ("db(" not in str(expression).lower()):
            # Likely real/imag channels for non-dB expressions.
            out = np.hypot(tail[:, 0], tail[:, 1])
        else:
            # Keep the first channel/trace deterministically.
            out = tail[:, 0]
        return np.asarray(out, dtype=float)

    # Fallback: flatten and align length.
    flat = y.reshape(-1)
    if flat.size >= n:
        return np.asarray(flat[:n], dtype=float)
    if flat.size > 1:
        src_x = np.linspace(0.0, 1.0, int(flat.size))
        dst_x = np.linspace(0.0, 1.0, n)
        return np.interp(dst_x, src_x, flat).astype(float, copy=False)
    return np.full((n,), float(flat[0]) if flat.size else 0.0, dtype=float)


def _dedupe_cut_points(angles: np.ndarray, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Remove duplicate angle samples deterministically (use mean per angle)."""
    ang = np.asarray(angles, dtype=float).reshape(-1)
    val = np.asarray(values, dtype=float).reshape(-1)
    if ang.size != val.size or ang.size <= 1:
        return ang, val
    # Round to avoid float noise creating fake unique bins.
    key = np.round(ang, 6)
    uniq, inv = np.unique(key, return_inverse=True)
    if uniq.size == key.size:
        return ang, val
    out = np.zeros((uniq.size,), dtype=float)
    cnt = np.zeros((uniq.size,), dtype=float)
    np.add.at(out, inv, val)
    np.add.at(cnt, inv, 1.0)
    out = out / np.maximum(cnt, 1.0)
    return uniq, out


def _coerce_grid_values(theta_vals: np.ndarray, phi_vals: np.ndarray, z_vals: np.ndarray) -> Optional[np.ndarray]:
    """Coerce HFSS grid values to shape (len(theta_vals), len(phi_vals))."""
    t = np.asarray(theta_vals, dtype=float).reshape(-1)
    p = np.asarray(phi_vals, dtype=float).reshape(-1)
    z = np.asarray(z_vals, dtype=float)
    tn = int(t.size)
    pn = int(p.size)
    if tn <= 0 or pn <= 0:
        return None

    if z.ndim == 1:
        if z.size == tn * pn:
            return z.reshape((tn, pn), order="C")
        return None

    if z.ndim == 2:
        # Reject pair-like matrices [theta, values] to force loop fallback.
        if z.shape == (2, tn):
            if np.allclose(z[0], t, rtol=1e-6, atol=1e-6) or np.allclose(z[1], t, rtol=1e-6, atol=1e-6):
                return None
        if z.shape == (tn, 2):
            if np.allclose(z[:, 0], t, rtol=1e-6, atol=1e-6) or np.allclose(z[:, 1], t, rtol=1e-6, atol=1e-6):
                return None
        if z.shape == (tn, pn):
            return z
        if z.shape == (pn, tn):
            return z.T
        if 1 in z.shape:
            flat = z.reshape(-1)
            if flat.size == tn * pn:
                return flat.reshape((tn, pn), order="C")
        return None

    # Higher rank not expected for this extraction path.
    return None


def _extract_solution_xy(sol, expression: str) -> Tuple[np.ndarray, np.ndarray]:
    """Extract (x, y) arrays from a PyAEDT SolutionData-like object."""
    if sol is None or isinstance(sol, bool):
        raise TypeError(f"invalid solution object: {type(sol).__name__}")
    if not hasattr(sol, "primary_sweep_values"):
        raise TypeError(f"solution object missing primary_sweep_values: {type(sol).__name__}")
    x = np.asarray(getattr(sol, "primary_sweep_values"), dtype=float).reshape(-1)
    if x.size == 0:
        raise ValueError("empty primary_sweep_values")

    y_raw = None
    try:
        y_raw = sol.get_expression_data(expression)
    except Exception:
        exprs = list(getattr(sol, "expressions", []) or [])
        if exprs:
            y_raw = sol.get_expression_data(exprs[0])
        else:
            raise
    y = _coerce_cut_values(x, np.asarray(y_raw, dtype=float), expression)
    if y.size == 0:
        raise ValueError("empty expression data")
    return x, y


@dataclass(frozen=True)
class CutRequest:
    setup_sweep: str                   # e.g. "Setup1 : LastAdaptive"
    sphere_name: str                   # e.g. "3D_Sphere"
    expression: str                    # e.g. "dB(GainTotal)" or "db(RealizedGainTotal)"
    primary_sweep: str                 # "Theta" or "Phi"
    fixed: Dict[str, Union[str, float, int]]  # e.g. {"Phi": "0deg", "Freq": "0.8GHz"}
    convert_theta_to_elevation: bool = False
    phi_domain: str = "-180..180"
    unwrap: bool = True


@dataclass(frozen=True)
class CutResult:
    angles_deg: AngleList
    values: FloatList
    meta: Dict[str, Union[str, float, int]]


@dataclass(frozen=True)
class GridRequest:
    setup_sweep: str
    sphere_name: str
    expression: str
    theta_points: int = 181
    phi_points: int = 361
    freq: Optional[str] = None
    phi_domain: str = "-180..180"
    convert_theta_to_elevation: bool = False


@dataclass(frozen=True)
class GridResult:
    theta_deg: AngleList
    phi_deg: AngleList
    values: List[List[float]]     # shape (len(theta), len(phi)) or transposed; see meta
    meta: Dict[str, Union[str, float, int]]


class FarFieldExtractor:
    """Extract far-field cuts and 3D grids from HFSS via PyAEDT.

    Key objective:
      - Provide robust retrieval across AEDT/PyAEDT versions by trying multiple strategies:
        (A) PostProcessor.get_solution_data (preferred) and (B) report_templates.far_field fallback.

    References:
      - HFSS far-field report template supports variations like Theta/Phi/Freq and sphere_name.  (PyAEDT docs)
    """

    def __init__(self, session: AedtHfssSession):
        self.session = session

    # ---------------------------
    # Public API
    # ---------------------------

    def extract_cut(self, req: CutRequest) -> CutResult:
        sol = self._get_solution_data_cut(req)
        ang = np.asarray(sol[0], dtype=float)
        val = _coerce_cut_values(ang, np.asarray(sol[1], dtype=float), req.expression)

        if req.primary_sweep.lower() == "theta" and req.convert_theta_to_elevation:
            ang = theta_to_elev(ang)
        if req.primary_sweep.lower() == "phi" and req.unwrap:
            ang = unwrap_phi(ang, domain=req.phi_domain)

        # Ensure ascending X for consistent downstream tools
        order = np.argsort(ang)
        ang = ang[order]
        val = val[order]
        ang, val = _dedupe_cut_points(ang, val)

        return CutResult(
            angles_deg=ang.tolist(),
            values=val.tolist(),
            meta={
                "setup_sweep": req.setup_sweep,
                "sphere_name": req.sphere_name,
                "expression": req.expression,
                "primary_sweep": req.primary_sweep,
                **{k: (v if isinstance(v, str) else float(v)) for k, v in req.fixed.items()},
                "theta_to_elev": int(req.convert_theta_to_elevation),
                "phi_domain": req.phi_domain,
            },
        )

    def extract_grid(self, req: GridRequest) -> GridResult:
        # Attempt direct multi-sweep extraction using get_solution_data + variations
        sol = self._get_solution_data_grid(req)
        theta = np.asarray(sol[0], dtype=float)
        phi = np.asarray(sol[1], dtype=float)
        grid = np.asarray(sol[2], dtype=float)  # expected shape (len(theta), len(phi))

        if req.convert_theta_to_elevation:
            theta = theta_to_elev(theta)
        phi = unwrap_phi(phi, domain=req.phi_domain)

        # Sort axes + reorder grid accordingly
        t_order = np.argsort(theta)
        p_order = np.argsort(phi)
        theta = theta[t_order]
        phi = phi[p_order]
        grid = grid[np.ix_(t_order, p_order)]

        return GridResult(
            theta_deg=theta.tolist(),
            phi_deg=phi.tolist(),
            values=grid.tolist(),
            meta={
                "setup_sweep": req.setup_sweep,
                "sphere_name": req.sphere_name,
                "expression": req.expression,
                "theta_points": int(req.theta_points),
                "phi_points": int(req.phi_points),
                "freq": req.freq or "",
                "phi_domain": req.phi_domain,
                "theta_to_elev": int(req.convert_theta_to_elevation),
                "grid_shape": f"{grid.shape[0]}x{grid.shape[1]}",
                "grid_layout": "values[theta_index][phi_index]",
            },
        )

    # ---------------------------
    # Internal strategies
    # ---------------------------

    def _setup_sweep_candidates(self, setup_sweep: str) -> List[str]:
        """Return candidate setup/sweep strings compatible with current HFSS context."""
        req = str(setup_sweep or "").strip()
        setup_name = _setup_name_from_sweep(req)
        hfss = self.session.hfss

        solved: List[str] = []
        try:
            raw = list(getattr(hfss, "existing_analysis_sweeps", []) or [])
            solved = [str(x).strip() for x in raw if str(x).strip()]
        except Exception:
            solved = []

        out: List[str] = []
        if setup_name:
            out.extend([sw for sw in solved if _setup_name_from_sweep(sw).lower() == setup_name.lower()])
        if req:
            out.append(req)
        if setup_name:
            out.append(setup_name)
            out.append(f"{setup_name} : LastAdaptive")
        out.extend(solved[:2])
        return _unique_str(out)

    def _nominal_freq_candidates(self) -> List[str]:
        hfss = self.session.hfss
        out: List[str] = []
        av = getattr(hfss, "available_variations", None)
        if av is None:
            return out
        for attr in ("nominal_values", "nominal"):
            try:
                values = getattr(av, attr)
            except Exception:
                continue
            if not isinstance(values, dict):
                continue
            for key, raw in values.items():
                if str(key).strip().lower() != "freq":
                    continue
                if isinstance(raw, (list, tuple)):
                    for item in raw:
                        txt = str(item).strip()
                        if txt:
                            out.append(txt)
                else:
                    txt = str(raw).strip()
                    if txt:
                        out.append(txt)
        return _unique_str(out)

    def _variation_candidates(self, req: CutRequest) -> List[Dict[str, List[str]]]:
        base = {k: [(_as_deg_str(v) if k in ("Phi", "Theta") else str(v))] for k, v in req.fixed.items()}
        out: List[Dict[str, List[str]]] = []
        v_all = dict(base)
        if req.primary_sweep not in v_all:
            v_all[req.primary_sweep] = ["All"]
        out.append(v_all)
        out.append(dict(base))

        if "Freq" not in base:
            for f in self._nominal_freq_candidates():
                v = dict(v_all)
                v["Freq"] = [f]
                out.append(v)
                v2 = dict(base)
                v2["Freq"] = [f]
                out.append(v2)

        uniq: List[Dict[str, List[str]]] = []
        seen = set()
        for item in out:
            key = tuple(sorted((str(k), tuple(map(str, item.get(k, [])))) for k in item.keys()))
            if key in seen:
                continue
            seen.add(key)
            uniq.append(item)
        return uniq

    def _get_solution_data_cut_report(self, req: CutRequest, setup_name: str, variations: Dict[str, List[str]]) -> Tuple[np.ndarray, np.ndarray]:
        hfss = self.session.hfss
        rep_factory = getattr(getattr(hfss, "post", None), "reports_by_category", None)
        if rep_factory is None or (not hasattr(rep_factory, "far_field")):
            raise RuntimeError("reports_by_category.far_field not available")
        rep = rep_factory.far_field(
            expressions=req.expression,
            setup=setup_name,
            sphere_name=req.sphere_name,
            **variations,
        )
        if not rep:
            raise RuntimeError("far_field report object was not created")
        try:
            rep.primary_sweep = req.primary_sweep
        except Exception:
            pass
        try:
            for k, vals in variations.items():
                rep.variations[k] = vals
        except Exception:
            pass
        sol = rep.get_solution_data()
        return _extract_solution_xy(sol, req.expression)

    def _get_solution_data_cut_direct(self, req: CutRequest, setup_name: str, variations: Dict[str, List[str]]) -> Tuple[np.ndarray, np.ndarray]:
        hfss = self.session.hfss
        sweeps = {}
        for k, vals in variations.items():
            if isinstance(vals, list) and vals:
                sweeps[k] = vals[0]
            else:
                sweeps[k] = vals
        if req.primary_sweep not in sweeps:
            sweeps[req.primary_sweep] = "All"
        sol = hfss.post.get_solution_data_per_variation(
            solution_type="Far Fields",
            setup_sweep_name=setup_name,
            context=["Context:=", req.sphere_name],
            sweeps=sweeps,
            expressions=req.expression,
        )
        if hasattr(sol, "primary_sweep"):
            try:
                sol.primary_sweep = req.primary_sweep
            except Exception:
                pass
        return _extract_solution_xy(sol, req.expression)

    def _get_solution_data_cut(self, req: CutRequest) -> Tuple[np.ndarray, np.ndarray]:
        hfss = self.session.hfss

        # Strategy A: PostProcessor3D.get_solution_data.
        # Strategy B: reports_by_category.far_field().get_solution_data().
        # Strategy C: direct get_solution_data_per_variation.
        errors = []
        candidates = self._variation_candidates(req)
        setup_candidates = self._setup_sweep_candidates(req.setup_sweep)

        for setup_name in setup_candidates:
            for variations in candidates:
                try:
                    sol = hfss.post.get_solution_data(
                        expressions=req.expression,
                        setup_sweep_name=setup_name,
                        report_category="Far Fields",
                        context=req.sphere_name,
                        variations=variations,
                        primary_sweep_variable=req.primary_sweep,
                    )
                    x, y = _extract_solution_xy(sol, req.expression)
                    return x, y
                except Exception as e:
                    msg_err = str(e)
                    errors.append(f"{setup_name}: {msg_err}")
                    # Compatibility workaround for PyAEDT constant mismatch.
                    if "default_solution" in msg_err and "HfssConstants" in msg_err:
                        _patch_solution_constants_alias()
                try:
                    x, y = self._get_solution_data_cut_report(req, setup_name, variations)
                    return x, y
                except Exception as e:
                    errors.append(f"{setup_name}: report_fallback: {e}")
                try:
                    x, y = self._get_solution_data_cut_direct(req, setup_name, variations)
                    return x, y
                except Exception as e:
                    errors.append(f"{setup_name}: direct_fallback: {e}")

        msg = " | ".join(errors[-3:]) if errors else "unknown error"
        raise RuntimeError(f"Failed to extract far-field cut from HFSS. Last errors: {msg}")

    def _get_solution_data_grid(self, req: GridRequest) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        hfss = self.session.hfss
        setup_candidates = self._setup_sweep_candidates(req.setup_sweep)
        setup_for_grid = setup_candidates[0] if setup_candidates else str(req.setup_sweep or "").strip()

        theta = np.linspace(0.0, 180.0, int(req.theta_points))
        phi = np.linspace(-180.0, 180.0, int(req.phi_points))

        # Strategy A: ask HFSS for a 2-sweep dataset (Theta primary, Phi secondary).
        # Many setups return a matrix suitable for SolutionData.plot_3d(primary='Theta', secondary='Phi').
        phi_from_solution = None
        try:
            variations = {
                "Theta": ["All"],
                "Phi": ["All"],
            }
            if req.freq:
                variations["Freq"] = [req.freq]
            sol = hfss.post.get_solution_data(
                expressions=req.expression,
                setup_sweep_name=setup_for_grid,
                report_category="Far Fields",
                context=req.sphere_name,
                variations=variations,
                primary_sweep_variable="Theta",
            )

            # If SolutionData contains both sweeps, we can reconstruct the grid from the full matrix.
            # Common: sol.primary_sweep_values gives Theta; sol.variation_values('Phi') gives Phi.
            t_vals = np.asarray(sol.primary_sweep_values, dtype=float)
            try:
                p_vals = np.asarray(sol.variation_values("Phi"), dtype=float)
                phi_from_solution = p_vals
            except Exception:
                # fallback: assume phi sweep is the second axis in the full matrix
                p_vals = phi
            # full_matrix_real_imag shape is not standardized across report types.
            # Prefer get_expression_data which returns the curve data; for matrices it may return flattened.
            z = np.asarray(sol.get_expression_data(req.expression), dtype=float)
            z2 = _coerce_grid_values(t_vals, p_vals, z)
            if z2 is not None:
                return t_vals, p_vals, z2

            # Else fallback to loop.
        except Exception:
            pass

        # Strategy B: loop over phi, extracting theta cuts. This is slower, but deterministic.
        phi_loop = np.asarray(phi_from_solution, dtype=float) if phi_from_solution is not None else phi
        if phi_loop.ndim != 1 or phi_loop.size == 0:
            phi_loop = phi
        grid = np.empty((theta.size, phi_loop.size), dtype=float)
        for j, ph in enumerate(phi_loop):
            cut = self.extract_cut(CutRequest(
                setup_sweep=setup_for_grid,
                sphere_name=req.sphere_name,
                expression=req.expression,
                primary_sweep="Theta",
                fixed={"Phi": _as_deg_str(ph), **({"Freq": req.freq} if req.freq else {})},
                convert_theta_to_elevation=False,
                phi_domain=req.phi_domain,
                unwrap=False,
            ))
            # Interpolate to our theta grid in case HFSS uses a different sampling.
            v = np.interp(theta, np.asarray(cut.angles_deg, dtype=float), np.asarray(cut.values, dtype=float))
            grid[:, j] = v

        return theta, np.asarray(phi_loop, dtype=float), grid
