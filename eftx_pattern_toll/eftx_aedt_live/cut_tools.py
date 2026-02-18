from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def wrap_to_180(angles_deg: np.ndarray) -> np.ndarray:
    a = np.asarray(angles_deg, dtype=float)
    return (a + 180.0) % 360.0 - 180.0


def _wrap_periodic(angles_deg: np.ndarray, start_deg: float, period_deg: float) -> np.ndarray:
    a = np.asarray(angles_deg, dtype=float)
    return (a - start_deg) % period_deg + start_deg


def _dedupe_mean(angles_deg: np.ndarray, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    a = np.asarray(angles_deg, dtype=float).reshape(-1)
    v = np.asarray(values, dtype=float).reshape(-1)
    if a.size != v.size:
        n = min(a.size, v.size)
        a = a[:n]
        v = v[:n]
    if a.size == 0:
        return a, v
    order = np.argsort(a)
    a = a[order]
    v = v[order]
    uniq, inv = np.unique(np.round(a, 6), return_inverse=True)
    if uniq.size == a.size:
        return a, v
    acc = np.zeros((uniq.size,), dtype=float)
    cnt = np.zeros((uniq.size,), dtype=float)
    np.add.at(acc, inv, v)
    np.add.at(cnt, inv, 1.0)
    out = acc / np.maximum(cnt, 1.0)
    return uniq, out


def _interp_periodic(x_deg: np.ndarray, y: np.ndarray, q_deg: np.ndarray, start_deg: float, period_deg: float) -> np.ndarray:
    x = _wrap_periodic(np.asarray(x_deg, dtype=float), start_deg, period_deg)
    yv = np.asarray(y, dtype=float)
    q = _wrap_periodic(np.asarray(q_deg, dtype=float), start_deg, period_deg)
    x, yv = _dedupe_mean(x, yv)
    if x.size == 0:
        return np.zeros_like(q)
    if x.size == 1:
        return np.full_like(q, yv[0], dtype=float)
    x_ext = np.concatenate([x, x[:1] + period_deg])
    y_ext = np.concatenate([yv, yv[:1]])
    q_adj = q.copy()
    q_adj[q_adj < x_ext[0]] += period_deg
    return np.interp(q_adj, x_ext, y_ext)


def _interp_circular(x_deg: np.ndarray, y: np.ndarray, q_deg: np.ndarray) -> np.ndarray:
    return _interp_periodic(x_deg, y, q_deg, start_deg=-180.0, period_deg=360.0)


def _prepare_vrp_angles(angles_deg: np.ndarray, values_lin: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    a = np.asarray(angles_deg, dtype=float).reshape(-1)
    v = np.asarray(values_lin, dtype=float).reshape(-1)
    n = min(a.size, v.size)
    a = a[:n]
    v = v[:n]
    if n == 0:
        return a, v

    candidates = []

    def add_candidate(arr: np.ndarray):
        aa = np.asarray(arr, dtype=float).reshape(-1)
        if aa.size != n:
            return
        mask = np.isfinite(aa) & np.isfinite(v)
        if not np.any(mask):
            return
        aa = aa[mask]
        vv = v[mask]
        in_dom = (aa >= -90.0) & (aa <= 90.0)
        if not np.any(in_dom):
            return
        aa = aa[in_dom]
        vv = vv[in_dom]
        span = float(np.max(aa) - np.min(aa)) if aa.size > 1 else 0.0
        score = (int(aa.size), span)
        candidates.append((score, aa, vv))

    aw = wrap_to_180(a)
    add_candidate(a)
    add_candidate(90.0 - a)
    add_candidate(a - 90.0)
    add_candidate(aw)
    add_candidate(90.0 - aw)

    if not candidates:
        return np.clip(a, -90.0, 90.0), v

    # Higher sample count first, then larger angular span.
    candidates.sort(key=lambda item: item[0], reverse=True)
    best_a = np.asarray(candidates[0][1], dtype=float)
    best_v = np.asarray(candidates[0][2], dtype=float)
    return best_a, best_v


def mag_db_from_linear(values_lin: np.ndarray) -> np.ndarray:
    v = np.asarray(values_lin, dtype=float)
    vmax = float(np.max(v)) if v.size else 0.0
    if vmax <= 0.0:
        return np.full(v.shape, -300.0, dtype=float)
    vn = np.clip(v / vmax, 1e-12, None)
    return 20.0 * np.log10(vn)


def wrap_to_vrp(angles_deg: np.ndarray) -> np.ndarray:
    """Wrap to VRP domain [-90, +90], preserving +90 samples."""
    a = np.asarray(angles_deg, dtype=float)
    out = (a + 90.0) % 180.0 - 90.0
    # Keep +90 on the positive side to avoid collapsing +90 into -90.
    mask = np.isclose(out, -90.0, atol=1e-9) & (a > 0.0)
    out[mask] = 90.0
    return out


def shift_cut_no_interp(
    angles_deg: np.ndarray,
    values_lin: np.ndarray,
    mode: str,
    target_peak_deg: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Shift a cut to a target peak angle without interpolation.

    This preserves real pulled samples from AEDT:
      - same values
      - same sample count
      - only angular displacement + domain wrap
    """
    kind = str(mode).strip().upper()
    if kind not in ("HRP", "VRP"):
        raise ValueError("mode must be HRP or VRP")

    a = np.asarray(angles_deg, dtype=float).reshape(-1)
    v = np.clip(np.asarray(values_lin, dtype=float).reshape(-1), 0.0, None)
    n = min(a.size, v.size)
    a = a[:n]
    v = v[:n]
    if n == 0:
        return (
            np.asarray([], dtype=float),
            np.asarray([], dtype=float),
            {"shift_deg": 0.0, "peak_before_deg": 0.0, "peak_after_deg": 0.0},
        )

    mask = np.isfinite(a) & np.isfinite(v)
    a = a[mask]
    v = v[mask]
    if a.size == 0:
        return (
            np.asarray([], dtype=float),
            np.asarray([], dtype=float),
            {"shift_deg": 0.0, "peak_before_deg": 0.0, "peak_after_deg": 0.0},
        )

    if kind == "HRP":
        wrap_fn = wrap_to_180
        a_work = wrap_fn(a)
        v_work = v
        a_work, v_work = _dedupe_mean(a_work, v_work)
    else:
        wrap_fn = wrap_to_vrp
        # For VRP, if source spans full theta (-180..180), keep a contiguous
        # 180-deg window centered at the real peak before any shift.
        a180 = wrap_to_180(a)
        span = float(np.max(a180) - np.min(a180)) if a180.size else 0.0
        if a180.size >= 250 and span > 220.0:
            p = float(a180[int(np.argmax(v))])
            rel = (a180 - p + 180.0) % 360.0 - 180.0
            mask = np.isfinite(rel) & np.isfinite(v) & (np.abs(rel) <= 90.0 + 1e-9)
            a_work = rel[mask]
            v_work = v[mask]
        else:
            # Otherwise select the best in-domain branch directly.
            a_work, v_work = _prepare_vrp_angles(a, v)
        a_work = wrap_fn(a_work)
        a_work, v_work = _dedupe_mean(a_work, v_work)
        # Deterministic order before index-shift attempt.
        order = np.argsort(a_work)
        a_work = a_work[order]
        v_work = v_work[order]

    if a_work.size == 0 or v_work.size == 0:
        return (
            np.asarray([], dtype=float),
            np.asarray([], dtype=float),
            {"shift_deg": 0.0, "peak_before_deg": 0.0, "peak_after_deg": 0.0},
        )

    target = float(wrap_fn(np.asarray([target_peak_deg], dtype=float))[0])
    peak_before = float(a_work[int(np.argmax(v_work))])

    if kind == "VRP" and a_work.size >= 3:
        # Prefer integer sample roll on near-uniform grids.
        # This keeps every pulled sample intact (no interpolation, no angle overlap).
        dif = np.diff(a_work)
        dif = dif[np.isfinite(dif)]
        if dif.size:
            step = float(np.median(np.abs(dif)))
            if step > 1e-9:
                uniform = bool(np.allclose(np.abs(dif), step, rtol=0.0, atol=1e-6))
                if uniform:
                    shift_steps = int(round((target - peak_before) / step))
                    a_out = a_work.copy()
                    v_out = np.roll(v_work, shift_steps)
                    peak_after = float(a_out[int(np.argmax(v_out))]) if a_out.size else 0.0
                    return a_out, v_out, {
                        "shift_deg": float(shift_steps * step),
                        "peak_before_deg": peak_before,
                        "peak_after_deg": peak_after,
                    }

    shift = float(target - peak_before)
    a_out = wrap_fn(a_work + shift)
    a_out, v_out = _dedupe_mean(a_out, v_work)

    # Keep values intact; only reorder by angle for deterministic plotting.
    order = np.argsort(a_out)
    a_out = a_out[order]
    v_out = v_out[order]

    peak_after = float(a_out[int(np.argmax(v_out))]) if a_out.size else 0.0
    return a_out, v_out, {"shift_deg": shift, "peak_before_deg": peak_before, "peak_after_deg": peak_after}


def transform_cut(
    angles_deg: np.ndarray,
    values_lin: np.ndarray,
    mode: str,
    rotation_deg: float = 0.0,
    align_peak_zero: bool = False,
    target_points: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Re-sample and rotate 2D cut for AEDT import review.

    - `mode='HRP'`: circular domain [-180, 180]
    - `mode='VRP'`: vertical domain [-90, 90]
    """
    kind = str(mode).strip().upper()
    if kind not in ("HRP", "VRP"):
        raise ValueError("mode must be HRP or VRP")

    a = np.asarray(angles_deg, dtype=float).reshape(-1)
    v = np.asarray(values_lin, dtype=float).reshape(-1)
    n = min(a.size, v.size)
    a = a[:n]
    v = np.clip(v[:n], 0.0, None)
    if n == 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float), {"shift_deg": 0.0, "peak_before_deg": 0.0, "peak_after_deg": 0.0}

    if kind == "HRP":
        if target_points is None:
            target_points = 361
        target_points = max(3, int(target_points))
        a_work = wrap_to_180(a)
        a_work, v_work = _dedupe_mean(a_work, v)
        peak_before = float(a_work[int(np.argmax(v_work))]) if a_work.size else 0.0
        shift = float(rotation_deg) + ((-peak_before) if align_peak_zero else 0.0)
        tgt = np.linspace(-180.0, 180.0, target_points)
        query = tgt - shift
        v_out = _interp_circular(a_work, v_work, query)
        peak_after = float(tgt[int(np.argmax(v_out))]) if tgt.size else 0.0
        return tgt, v_out, {"shift_deg": float(shift), "peak_before_deg": float(peak_before), "peak_after_deg": float(peak_after)}

    # VRP
    if target_points is None:
        target_points = 181
    target_points = max(3, int(target_points))
    a_work, v_work = _prepare_vrp_angles(a, v)
    a_work, v_work = _dedupe_mean(a_work, v_work)
    peak_before = float(a_work[int(np.argmax(v_work))]) if a_work.size else 0.0
    shift = float(rotation_deg) + ((-peak_before) if align_peak_zero else 0.0)
    tgt = np.linspace(-90.0, 90.0, target_points)
    query = tgt - shift
    if a_work.size == 0:
        v_out = np.zeros_like(tgt)
    elif a_work.size == 1:
        v_out = np.full_like(tgt, v_work[0], dtype=float)
    else:
        # Use periodic interpolation in 180-deg domain to preserve energy while rotating.
        v_out = _interp_periodic(a_work, v_work, query, start_deg=-90.0, period_deg=180.0)
    peak_after = float(tgt[int(np.argmax(v_out))]) if tgt.size else 0.0
    return tgt, v_out, {"shift_deg": float(shift), "peak_before_deg": float(peak_before), "peak_after_deg": float(peak_after)}
