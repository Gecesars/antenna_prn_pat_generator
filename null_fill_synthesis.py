from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

C0 = 299_792_458.0


def _build_A(
    f_hz: float,
    z_m: np.ndarray,
    eps_deg: np.ndarray,
) -> Tuple[np.ndarray, float, float]:
    lam0 = C0 / max(float(f_hz), 1.0)
    k = 2.0 * math.pi / lam0
    eps_rad = np.deg2rad(np.asarray(eps_deg, dtype=float))
    z = np.asarray(z_m, dtype=float).reshape(-1)
    A = np.exp(1j * k * z[None, :] * np.sin(eps_rad)[:, None])
    return A, k, lam0


def _sample_elem_pattern(
    elem_pattern: Optional[Callable[[np.ndarray], np.ndarray]],
    eps_deg: np.ndarray,
) -> np.ndarray:
    eps = np.asarray(eps_deg, dtype=float)
    if elem_pattern is None:
        return np.ones(eps.size, dtype=complex)

    try:
        out = elem_pattern(eps)
    except Exception:
        out = np.array([elem_pattern(float(x)) for x in eps], dtype=complex)
    out = np.asarray(out, dtype=complex).reshape(-1)
    if out.size != eps.size:
        raise ValueError("elem_pattern retornou tamanho invalido.")
    out[~np.isfinite(out)] = 0.0 + 0.0j
    return out


def _build_floor_target(
    eps_deg: np.ndarray,
    fill_bands: List[dict],
) -> Tuple[np.ndarray, np.ndarray]:
    eps = np.asarray(eps_deg, dtype=float)
    floor_lin = np.zeros(eps.size, dtype=float)
    fit_weight = np.ones(eps.size, dtype=float)

    for b in fill_bands or []:
        e0 = float(b.get("eps_min_deg", 0.0))
        e1 = float(b.get("eps_max_deg", 0.0))
        lo, hi = (e0, e1) if e0 <= e1 else (e1, e0)
        floor_db = float(b.get("floor_db", -999.0))
        w = float(b.get("weight", 1.0))

        mask = (eps >= lo) & (eps <= hi)
        if not np.any(mask):
            continue

        # floor_db e em dB de campo (20*log10|E|).
        floor = 10.0 ** (floor_db / 20.0)
        floor = max(0.0, min(1.0, floor))
        floor_lin[mask] = np.maximum(floor_lin[mask], floor)
        fit_weight[mask] = np.maximum(fit_weight[mask], max(0.1, w))

    return floor_lin, fit_weight


def _initial_w(
    z_m: np.ndarray,
    k: float,
    tilt_deg: Optional[float],
    progressive_phase_deg_per_elem: float = 0.0,
    amp_init: Optional[np.ndarray] = None,
) -> np.ndarray:
    z = np.asarray(z_m, dtype=float).reshape(-1)
    n = np.arange(z.size, dtype=float)
    w = np.ones(z.size, dtype=complex)

    beta = math.radians(float(progressive_phase_deg_per_elem))
    if abs(beta) > 1e-14:
        w = w * np.exp(1j * (n * beta))

    if tilt_deg is not None:
        eps0 = math.radians(float(tilt_deg))
        w = w * np.exp(-1j * k * z * math.sin(eps0))

    if amp_init is not None:
        a = np.asarray(amp_init, dtype=float).reshape(-1)
        if a.size != z.size:
            raise ValueError("amp_init com tamanho invalido.")
        w = np.abs(a) * np.exp(1j * np.angle(w))
    return w


def _solve_tikhonov(
    A: np.ndarray,
    d: np.ndarray,
    reg_lambda: float,
    fit_weight: Optional[np.ndarray] = None,
) -> np.ndarray:
    if fit_weight is None:
        Aw = A
        dw = d
    else:
        fw = np.asarray(fit_weight, dtype=float).reshape(-1)
        Aw = A * fw[:, None]
        dw = d * fw

    AH = Aw.conj().T
    M = AH @ Aw + float(reg_lambda) * np.eye(A.shape[1], dtype=complex)
    b = AH @ dw
    return np.linalg.solve(M, b)


def _project_mode(
    w: np.ndarray,
    mode: str,
    z_m: np.ndarray,
    k: float,
    tilt_deg: Optional[float],
    progressive_phase_deg_per_elem: float = 0.0,
    amp_fixed: Optional[np.ndarray] = None,
) -> np.ndarray:
    mode_l = str(mode).strip().lower()
    if mode_l == "both":
        return w

    z = np.asarray(z_m, dtype=float).reshape(-1)
    n = np.arange(z.size, dtype=float)

    if mode_l == "amplitude":
        beta = math.radians(float(progressive_phase_deg_per_elem))
        phi = n * beta
        if tilt_deg is not None:
            phi = phi - k * z * math.sin(math.radians(float(tilt_deg)))
        a = np.abs(w)
        return a * np.exp(1j * phi)

    if mode_l == "phase":
        if amp_fixed is None:
            a = np.ones(z.size, dtype=float)
        else:
            a = np.asarray(amp_fixed, dtype=float).reshape(-1)
            if a.size != z.size:
                raise ValueError("amp_fixed com tamanho invalido.")
        return np.abs(a) * np.exp(1j * np.angle(w))

    raise ValueError("mode invalido. Use 'amplitude', 'phase' ou 'both'.")


def _apply_amp_limits_db(w: np.ndarray, amp_limits_db: Optional[Tuple[float, float]]) -> np.ndarray:
    if amp_limits_db is None:
        return w
    lo_db, hi_db = amp_limits_db
    lo_db = float(lo_db)
    hi_db = float(hi_db)
    lo_db, hi_db = (lo_db, hi_db) if lo_db <= hi_db else (hi_db, lo_db)

    a = np.abs(w)
    ref = float(np.max(a))
    if ref <= 1e-15:
        return w
    att_db = -20.0 * np.log10(np.maximum(a / ref, 1e-12))
    att_db = np.clip(att_db, lo_db, hi_db)
    a_new = ref * (10.0 ** (-att_db / 20.0))
    return a_new * np.exp(1j * np.angle(w))


def _apply_phase_limits_deg(w: np.ndarray, phase_limits_deg: Optional[float]) -> np.ndarray:
    if phase_limits_deg is None:
        return w
    lim = abs(float(phase_limits_deg))
    if lim <= 0:
        return w

    a = np.abs(w)
    phi = np.unwrap(np.angle(w))
    phi_ref = float(phi[0])
    rel = phi - phi_ref
    rel = np.clip(rel, -math.radians(lim), math.radians(lim))
    phi_new = phi_ref + rel
    return a * np.exp(1j * phi_new)


def _normalize_weights(w: np.ndarray, norm: str) -> np.ndarray:
    mode = str(norm).strip().lower()
    if mode == "max_1":
        m = float(np.max(np.abs(w)))
        return w / (m if m > 1e-15 else 1.0)
    # default: sum |w|^2 = 1
    p = float(np.sum(np.abs(w) ** 2))
    return w / (math.sqrt(p) if p > 1e-15 else 1.0)


def _band_floor_metrics(
    eps_deg: np.ndarray,
    e_mag_norm: np.ndarray,
    fill_bands: List[dict],
) -> List[dict]:
    out: List[dict] = []
    for b in fill_bands or []:
        e0 = float(b.get("eps_min_deg", 0.0))
        e1 = float(b.get("eps_max_deg", 0.0))
        lo, hi = (e0, e1) if e0 <= e1 else (e1, e0)
        floor_db = float(b.get("floor_db", -999.0))
        mask = (eps_deg >= lo) & (eps_deg <= hi)
        if not np.any(mask):
            continue
        min_db = float(np.min(20.0 * np.log10(np.maximum(e_mag_norm[mask], 1e-12))))
        out.append(
            {
                "eps_min_deg": lo,
                "eps_max_deg": hi,
                "floor_db": floor_db,
                "min_db": min_db,
                "margin_db": min_db - floor_db,
            }
        )
    return out


def _local_minima_indices(mag: np.ndarray) -> np.ndarray:
    v = np.asarray(mag, dtype=float).reshape(-1)
    if v.size < 3:
        return np.array([], dtype=int)
    m = (v[1:-1] <= v[:-2]) & (v[1:-1] <= v[2:])
    return np.where(m)[0] + 1


def _null_pair_by_order(
    eps_deg: np.ndarray,
    mag_norm: np.ndarray,
    peak_idx: int,
    order: int,
) -> Dict[str, Optional[int]]:
    mins = _local_minima_indices(mag_norm)
    mins = np.array([i for i in mins if 1 <= i < len(mag_norm) - 1], dtype=int)
    left = [i for i in mins if i < peak_idx]
    right = [i for i in mins if i > peak_idx]

    left_sorted = sorted(left, key=lambda i: abs(i - peak_idx))
    right_sorted = sorted(right, key=lambda i: abs(i - peak_idx))

    k = max(1, int(order)) - 1
    idx_l = left_sorted[k] if k < len(left_sorted) else None
    idx_r = right_sorted[k] if k < len(right_sorted) else None
    return {"left": idx_l, "right": idx_r}


def _null_window_mask(
    eps_deg: np.ndarray,
    peak_idx: int,
    null_idx: int,
) -> Tuple[np.ndarray, float]:
    eps = np.asarray(eps_deg, dtype=float)
    dpk = abs(float(eps[null_idx] - eps[peak_idx]))
    width = max(0.35, min(3.5, 0.22 * dpk))
    mask = np.abs(eps - float(eps[null_idx])) <= width
    return mask, width


def _local_maxima_indices(mag: np.ndarray) -> np.ndarray:
    v = np.asarray(mag, dtype=float).reshape(-1)
    if v.size < 3:
        return np.array([], dtype=int)
    m = (v[1:-1] >= v[:-2]) & (v[1:-1] >= v[2:])
    return np.where(m)[0] + 1


def _mainlobe_mask_from_first_nulls(
    eps_deg: np.ndarray,
    mag_norm: np.ndarray,
    peak_idx: int,
) -> np.ndarray:
    pair_main = _null_pair_by_order(eps_deg, mag_norm, peak_idx, 1)
    out = np.zeros_like(np.asarray(eps_deg, dtype=float), dtype=bool)
    il = pair_main.get("left")
    ir = pair_main.get("right")
    if il is not None and ir is not None:
        lo = min(int(il), int(ir))
        hi = max(int(il), int(ir))
        out[lo : hi + 1] = True
        return out
    eps = np.asarray(eps_deg, dtype=float)
    out[np.abs(eps - float(eps[peak_idx])) <= 2.0] = True
    return out


def _peak_idx_near_reference(
    eps_deg: np.ndarray,
    mag: np.ndarray,
    ref_peak_idx: int,
    max_offset_deg: float = 6.0,
) -> int:
    eps = np.asarray(eps_deg, dtype=float)
    v = np.asarray(mag, dtype=float)
    if eps.size == 0:
        return 0
    ref_idx = int(max(0, min(ref_peak_idx, eps.size - 1)))
    mask = np.abs(eps - float(eps[ref_idx])) <= abs(float(max_offset_deg))
    if np.any(mask):
        local = np.where(mask)[0]
        return int(local[int(np.argmax(v[local]))])
    return int(np.argmax(v))


def _track_null_idx_near_reference(
    eps_deg: np.ndarray,
    mag: np.ndarray,
    ref_null_idx: int,
    max_offset_deg: float,
) -> int:
    eps = np.asarray(eps_deg, dtype=float)
    v = np.asarray(mag, dtype=float)
    if eps.size == 0:
        return 0
    idx_ref = int(max(0, min(ref_null_idx, eps.size - 1)))
    mask = np.abs(eps - float(eps[idx_ref])) <= abs(float(max_offset_deg))
    if np.any(mask):
        local = np.where(mask)[0]
        sub = v[local]
        mins_sub = _local_minima_indices(sub)
        if mins_sub.size:
            candidates = local[mins_sub]
            best = min(candidates.tolist(), key=lambda i: (abs(int(i) - idx_ref), float(v[int(i)])))
            return int(best)
        return int(local[int(np.argmin(sub))])
    return idx_ref


def synth_null_fill_by_order(
    f_hz: float,
    z_m: np.ndarray,
    eps_grid_deg: np.ndarray,
    null_order: int,
    null_fill_percent: float,
    mode: str,
    mainlobe_tilt_deg: Optional[float] = None,
    elem_pattern: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    reg_lambda: float = 1e-5,
    max_iters: int = 20,
    preserve_mainlobe_weight: float = 12.0,
    fill_weight: float = 32.0,
    phase_limits_deg: Optional[float] = None,
    progressive_phase_deg_per_elem: float = 0.0,
    amp_fixed: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    eps = np.asarray(eps_grid_deg, dtype=float).reshape(-1)
    if eps.size < 8:
        raise ValueError("eps_grid_deg deve ter ao menos 8 amostras.")
    z = np.asarray(z_m, dtype=float).reshape(-1)
    if z.size < 1:
        raise ValueError("z_m invalido.")

    mode_l = str(mode).strip().lower()
    if mode_l not in ("amplitude", "phase", "both"):
        raise ValueError("mode invalido. Use 'amplitude', 'phase' ou 'both'.")

    A, k, lam0 = _build_A(f_hz, z, eps)
    elem = _sample_elem_pattern(elem_pattern, eps)
    B = elem[:, None] * A

    w0 = _initial_w(
        z,
        k,
        mainlobe_tilt_deg,
        progressive_phase_deg_per_elem=progressive_phase_deg_per_elem,
        amp_init=amp_fixed,
    )
    w0 = _normalize_weights(w0, "sum_abs2_1")

    E_ref = B @ w0
    mag_ref = np.abs(E_ref)
    peak_ref = float(np.max(mag_ref)) if mag_ref.size else 1.0
    peak_ref = peak_ref if peak_ref > 1e-12 else 1.0
    mag_ref_norm = mag_ref / peak_ref
    peak_idx_ref = int(np.argmax(mag_ref))

    pair_target_ref = _null_pair_by_order(eps, mag_ref_norm, peak_idx_ref, null_order)
    ref_nulls: Dict[str, Dict[str, float]] = {}
    for side in ("left", "right"):
        idx = pair_target_ref.get(side)
        if idx is None:
            continue
        i = int(idx)
        ref_nulls[side] = {
            "idx": i,
            "eps_deg": float(eps[i]),
            "amp": float(mag_ref[i]),
            "db": float(20.0 * np.log10(max(mag_ref[i] / peak_ref, 1e-12))),
        }

    if not ref_nulls:
        mins = _local_minima_indices(mag_ref_norm)
        if mins.size == 0:
            raise ValueError("Nao foi possivel detectar nulos no diagrama inicial.")
        nearest = sorted(mins.tolist(), key=lambda i: abs(int(i) - peak_idx_ref))[0]
        side = "left" if int(nearest) < peak_idx_ref else "right"
        i = int(nearest)
        ref_nulls[side] = {
            "idx": i,
            "eps_deg": float(eps[i]),
            "amp": float(mag_ref[i]),
            "db": float(20.0 * np.log10(max(mag_ref[i] / peak_ref, 1e-12))),
        }

    main_mask_ref = _mainlobe_mask_from_first_nulls(eps, mag_ref_norm, peak_idx_ref)

    pct = max(0.0, min(100.0, float(null_fill_percent)))
    target_frac = pct / 100.0
    preserve_w_eff = max(float(preserve_mainlobe_weight), 2.0)
    fill_w_eff = max(float(fill_weight), 2.0)
    if pct > 0.0:
        preserve_w_eff = max(preserve_w_eff, 16.0)
        fill_w_eff = max(fill_w_eff, 24.0)

    phase_limit_eff = phase_limits_deg
    if phase_limit_eff is None and mode_l == "both" and pct > 0.0:
        # Limite automatico de fase para reduzir cancelamentos profundos
        # e preservar o lobo principal durante o preenchimento.
        phase_limit_eff = float(np.clip(55.0 - 0.6 * pct, 25.0, 55.0))

    target_abs_by_side: Dict[str, float] = {}
    track_span_deg_by_side: Dict[str, float] = {}
    for side, info in ref_nulls.items():
        amp0 = float(info["amp"])
        target_abs_by_side[side] = amp0 + target_frac * (peak_ref - amp0)
        dpk = abs(float(info["eps_deg"]) - float(eps[peak_idx_ref]))
        track_span_deg_by_side[side] = max(1.0, min(5.0, 0.22 * dpk + 0.5))

    w = w0.copy()
    reg = max(float(reg_lambda), 1e-12)
    iters = max(1, int(max_iters))

    for _ in range(iters):
        E_now = B @ w
        mag_now = np.abs(E_now)
        peak_now = float(np.max(mag_now)) if mag_now.size else 1.0
        peak_now = peak_now if peak_now > 1e-12 else 1.0
        peak_idx_now = _peak_idx_near_reference(eps, mag_now, peak_idx_ref, max_offset_deg=6.0)
        phase_now = np.angle(E_now)

        d = E_ref.copy()
        W = np.full_like(mag_now, 0.25, dtype=float)
        W[main_mask_ref] = np.maximum(W[main_mask_ref], max(6.0, 4.0 * preserve_w_eff))

        # Trava o pico do lobo principal para evitar deslocamentos.
        peak_lock = np.abs(eps - float(eps[peak_idx_ref])) <= 1.0
        d[peak_lock] = E_ref[peak_lock]
        W[peak_lock] = np.maximum(W[peak_lock], max(40.0, 6.0 * preserve_w_eff))

        for side in ("left", "right"):
            if side not in target_abs_by_side:
                continue
            idx_now = _track_null_idx_near_reference(
                eps,
                mag_now,
                int(ref_nulls[side]["idx"]),
                track_span_deg_by_side.get(side, 3.0),
            )
            center_deg = float(eps[idx_now])

            dpk = abs(center_deg - float(eps[peak_idx_now]))
            half_width = max(0.45, min(5.0, 0.24 * dpk + 0.35))
            mask = np.abs(eps - center_deg) <= half_width
            if not np.any(mask):
                continue

            sigma = max(0.15, 0.45 * half_width)
            x = (eps[mask] - center_deg) / sigma
            g = np.exp(-0.5 * (x ** 2))

            target_abs = float(target_abs_by_side[side])
            side_def = max(target_abs - float(mag_now[idx_now]), 0.0)
            boost = 1.0 + 1.8 * side_def / max(target_abs, 1e-9)
            target_abs_eff = min(peak_ref, target_abs * boost)
            desired_mag = np.maximum(
                np.abs(E_ref[mask]),
                (1.0 - g) * np.abs(E_ref[mask]) + g * target_abs_eff,
            )
            d[mask] = desired_mag * np.exp(1j * phase_now[mask])

            deficit = np.maximum(desired_mag - mag_now[mask], 0.0)
            rel_def = deficit / max(target_abs_eff, 1e-9)
            w_local = fill_w_eff * (8.0 + 20.0 * g + 60.0 * rel_def)
            W[mask] = np.maximum(W[mask], w_local)

        BW = B * W[:, None]
        dW = d * W
        BH = BW.conj().T
        reg_prior = 6.0 * reg
        reg_ref = 2.0 * reg
        M = BH @ BW + (reg_prior + reg_ref) * np.eye(B.shape[1], dtype=complex)
        b = BH @ dW + reg_prior * w + reg_ref * w0
        w_free = np.linalg.solve(M, b)

        w_proj = _project_mode(
            w_free,
            mode_l,
            z,
            k,
            mainlobe_tilt_deg,
            progressive_phase_deg_per_elem=progressive_phase_deg_per_elem,
            amp_fixed=amp_fixed,
        )
        w_proj = _apply_phase_limits_deg(w_proj, phase_limit_eff)
        w = _normalize_weights(w_proj, "sum_abs2_1")

    E_final = B @ w
    E_ini = E_ref
    mag_ini = np.abs(E_ini)
    mag_fin = np.abs(E_final)
    mag_ini_norm = mag_ini / (np.max(mag_ini) if np.max(mag_ini) > 0 else 1.0)
    mag_fin_norm = mag_fin / (np.max(mag_fin) if np.max(mag_fin) > 0 else 1.0)

    peak_idx_fin = _peak_idx_near_reference(eps, mag_fin, peak_idx_ref, max_offset_deg=8.0)

    null_regions: List[dict] = []
    null_levels: List[dict] = []
    for side, info in ref_nulls.items():
        idx_fin = _track_null_idx_near_reference(
            eps,
            mag_fin,
            int(info["idx"]),
            track_span_deg_by_side.get(side, 3.0),
        )

        _, hw = _null_window_mask(eps, peak_idx_fin, idx_fin)
        null_regions.append(
            {
                "side": side,
                "idx": idx_fin,
                "eps_deg": float(eps[idx_fin]),
                "half_width_deg": float(hw),
            }
        )

        amp0 = float(info["amp"])
        ampf = float(mag_fin[idx_fin])
        tgt = float(target_abs_by_side.get(side, amp0))
        gap = max(peak_ref - amp0, 1e-12)
        achieved_pct = 100.0 * (ampf - amp0) / gap
        target_pct_side = 100.0 * (tgt - amp0) / gap
        null_levels.append(
            {
                "side": side,
                "idx": idx_fin,
                "eps_deg": float(eps[idx_fin]),
                "initial_db": float(20.0 * np.log10(max(amp0 / peak_ref, 1e-12))),
                "target_db": float(20.0 * np.log10(max(tgt / peak_ref, 1e-12))),
                "final_db": float(20.0 * np.log10(max(ampf / peak_ref, 1e-12))),
                "target_percent": float(target_pct_side),
                "achieved_percent": float(achieved_pct),
            }
        )

    cond = float(np.linalg.cond(B.conj().T @ B + reg * np.eye(B.shape[1], dtype=complex)))
    achieved_percent_avg = float(
        np.mean([x.get("achieved_percent", 0.0) for x in null_levels]) if null_levels else 0.0
    )

    return {
        "w": w,
        "AF_initial": A @ w0,
        "AF_final": A @ w,
        "E_initial": E_ini,
        "E_final": E_final,
        "eps_deg": eps,
        "lambda0_m": lam0,
        "k_rad_m": k,
        "mode": mode_l,
        "condition_number": cond,
        "peak_eps_deg": float(eps[peak_idx_fin]),
        "null_order": int(max(1, null_order)),
        "target_percent": pct,
        "achieved_percent": achieved_percent_avg,
        "phase_limit_deg": None if phase_limit_eff is None else float(phase_limit_eff),
        "null_regions": null_regions,
        "null_levels": null_levels,
    }


def synth_null_fill_vertical(
    f_hz: float,
    z_m: np.ndarray,
    eps_grid_deg: np.ndarray,
    fill_bands: List[dict],
    mode: str,
    mainlobe_tilt_deg: Optional[float] = None,
    elem_pattern: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    reg_lambda: float = 1e-3,
    max_iters: int = 8,
    amp_limits_db: Optional[Tuple[float, float]] = None,
    phase_limits_deg: Optional[float] = None,
    norm: str = "sum_abs2_1",
    progressive_phase_deg_per_elem: float = 0.0,
    amp_fixed: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """
    Sintese de pesos complexos para preenchimento de nulos em composicao vertical.

    fill_bands:
      [{"eps_min_deg": 2, "eps_max_deg": 8, "floor_db": -14, "weight": 1.0}, ...]
    """
    eps = np.asarray(eps_grid_deg, dtype=float).reshape(-1)
    if eps.size < 4:
        raise ValueError("eps_grid_deg deve ter ao menos 4 amostras.")
    z = np.asarray(z_m, dtype=float).reshape(-1)
    if z.size < 1:
        raise ValueError("z_m invalido.")

    mode_l = str(mode).strip().lower()
    if mode_l not in ("amplitude", "phase", "both"):
        raise ValueError("mode invalido. Use 'amplitude', 'phase' ou 'both'.")

    A, k, lam0 = _build_A(f_hz, z, eps)
    elem = _sample_elem_pattern(elem_pattern, eps)

    floor_lin, fit_weight = _build_floor_target(eps, fill_bands)
    w = _initial_w(
        z,
        k,
        mainlobe_tilt_deg,
        progressive_phase_deg_per_elem=progressive_phase_deg_per_elem,
        amp_init=amp_fixed,
    )
    w = _normalize_weights(w, norm)

    AF_initial = A @ w
    E_initial = elem * AF_initial

    iters = max(1, int(max_iters))
    reg = max(float(reg_lambda), 1e-12)

    for _ in range(iters):
        AF_now = A @ w
        E_now = elem * AF_now
        mag_now = np.abs(E_now)
        peak_now = float(np.max(mag_now)) if mag_now.size else 1.0
        peak_now = peak_now if peak_now > 1e-12 else 1.0

        # Piso relativo ao pico atual.
        floor_abs = floor_lin * peak_now
        target_mag = mag_now.copy()
        band_mask = floor_lin > 0.0
        if np.any(band_mask):
            target_mag[band_mask] = np.maximum(mag_now[band_mask], floor_abs[band_mask])

        # Reforca explicitamente pontos deficitarios da banda para atacar nulos.
        deficit = np.maximum(floor_abs - mag_now, 0.0)
        dyn_weight = np.ones_like(mag_now, dtype=float)
        dyn_weight[band_mask] += 4.0
        if np.any(band_mask):
            rel_def = np.zeros_like(mag_now, dtype=float)
            rel_def[band_mask] = deficit[band_mask] / np.maximum(floor_abs[band_mask], 1e-9)
            dyn_weight += 35.0 * rel_def
        dyn_weight *= fit_weight

        # Evita pedir AF irrealista quando |E_elem| e muito baixo.
        elem_abs_now = np.abs(elem)
        elem_floor = max(float(np.percentile(elem_abs_now, 20)), 0.12 * float(np.max(elem_abs_now)), 1e-4)
        elem_mag_eff = np.maximum(elem_abs_now, elem_floor)

        target_af_mag = target_mag / elem_mag_eff
        phase_af = np.angle(AF_now)
        d_af = target_af_mag * np.exp(1j * phase_af)

        w_free = _solve_tikhonov(A, d_af, reg_lambda=reg, fit_weight=dyn_weight)
        w_proj = _project_mode(
            w_free,
            mode_l,
            z,
            k,
            mainlobe_tilt_deg,
            progressive_phase_deg_per_elem=progressive_phase_deg_per_elem,
            amp_fixed=amp_fixed,
        )
        w_proj = _apply_amp_limits_db(w_proj, amp_limits_db)
        w_proj = _apply_phase_limits_deg(w_proj, phase_limits_deg)
        w = _normalize_weights(w_proj, norm)

    AF_final = A @ w
    E_final = elem * AF_final
    e_mag = np.abs(E_final)
    peak = float(np.max(e_mag)) if e_mag.size else 1.0
    peak = peak if peak > 1e-12 else 1.0
    e_norm = e_mag / peak
    e_db = 20.0 * np.log10(np.maximum(e_norm, 1e-12))

    idx_pk = int(np.argmax(e_mag)) if e_mag.size else 0
    cond = float(np.linalg.cond(A.conj().T @ A + reg * np.eye(A.shape[1], dtype=complex)))

    return {
        "w": w,
        "AF_initial": AF_initial,
        "AF_final": AF_final,
        "E_initial": E_initial,
        "E_final": E_final,
        "E_final_norm": e_norm,
        "E_final_db": e_db,
        "eps_deg": eps,
        "lambda0_m": lam0,
        "k_rad_m": k,
        "condition_number": cond,
        "peak_eps_deg": float(eps[idx_pk]) if eps.size else 0.0,
        "band_metrics": _band_floor_metrics(eps, e_norm, fill_bands),
        "fill_bands": list(fill_bands or []),
        "mode": mode_l,
    }


def weights_to_harness(
    w: np.ndarray,
    f_hz: float,
    vf: float,
    ref_index: int = 0,
) -> Dict[str, np.ndarray]:
    wv = np.asarray(w, dtype=complex).reshape(-1)
    if wv.size == 0:
        raise ValueError("w vazio.")
    if not (0 <= int(ref_index) < wv.size):
        raise ValueError("ref_index fora do intervalo.")

    lam0 = C0 / max(float(f_hz), 1.0)
    lamg = float(vf) * lam0

    a = np.abs(wv)
    p = a ** 2
    p_sum = float(np.sum(p))
    p_frac = p / (p_sum if p_sum > 1e-15 else 1.0)

    phi = np.angle(wv)
    phi_ref = float(phi[int(ref_index)])
    dphi = (phi - phi_ref) % (2.0 * math.pi)
    phase_deg = np.rad2deg(dphi)
    delta_len_m = (dphi / (2.0 * math.pi)) * lamg

    p_ref = float(p_frac[int(ref_index)]) if p_frac[int(ref_index)] > 1e-15 else float(np.max(p_frac))
    p_ref = p_ref if p_ref > 1e-15 else 1.0
    att_db = -10.0 * np.log10(np.maximum(p_frac, 1e-15) / p_ref)

    return {
        "amp": a,
        "phase_deg": phase_deg,
        "p_frac": p_frac,
        "att_db_ref": att_db,
        "delta_len_m": delta_len_m,
        "lambda0_m": np.array([lam0], dtype=float),
        "lambda_g_m": np.array([lamg], dtype=float),
    }
