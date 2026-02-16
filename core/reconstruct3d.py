from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

from core.angles import elev_to_theta_deg, theta_to_elev_deg, wrap_phi_deg


@dataclass(frozen=True)
class PatternCut:
    type: Literal["H", "V"]
    angles_deg: np.ndarray
    values_lin: np.ndarray
    meta: dict


@dataclass(frozen=True)
class SphericalPattern:
    theta_deg: np.ndarray
    phi_deg: np.ndarray
    mag_lin: np.ndarray
    meta: dict


def _sorted_unique_mean(angles_deg: np.ndarray, values_lin: np.ndarray):
    a = np.asarray(angles_deg, dtype=float).reshape(-1)
    v = np.asarray(values_lin, dtype=float).reshape(-1)
    if a.size == 0 or v.size == 0 or a.size != v.size:
        raise ValueError("Invalid cut arrays.")
    idx = np.argsort(a)
    a = a[idx]
    v = v[idx]
    au, inv = np.unique(a, return_inverse=True)
    acc = np.zeros_like(au, dtype=float)
    cnt = np.zeros_like(au, dtype=float)
    for i, val in zip(inv, v):
        acc[i] += float(val)
        cnt[i] += 1.0
    vu = acc / np.maximum(cnt, 1.0)
    return au, vu


def _sample_h_phi(cut_h: PatternCut, phi_deg: np.ndarray) -> np.ndarray:
    a = wrap_phi_deg(np.asarray(cut_h.angles_deg, dtype=float).reshape(-1))
    v = np.asarray(cut_h.values_lin, dtype=float).reshape(-1)
    au, vu = _sorted_unique_mean(a, v)
    a_ext = np.concatenate([au, au[:1] + 360.0])
    v_ext = np.concatenate([vu, vu[:1]])
    p = wrap_phi_deg(np.asarray(phi_deg, dtype=float))
    p_adj = p.copy()
    p_adj[p_adj < a_ext[0]] += 360.0
    out = np.interp(p_adj, a_ext, v_ext)
    return np.clip(out, 0.0, None)


def _sample_v_theta(cut_v: PatternCut, theta_deg: np.ndarray) -> np.ndarray:
    elev = theta_to_elev_deg(np.asarray(theta_deg, dtype=float))
    a = np.asarray(cut_v.angles_deg, dtype=float).reshape(-1)
    v = np.asarray(cut_v.values_lin, dtype=float).reshape(-1)

    # If V cut already appears to be theta-like [0, 180], convert to elevation.
    if float(np.nanmin(a)) >= -1e-9 and float(np.nanmax(a)) <= 180.0 + 1e-9:
        a = theta_to_elev_deg(a)

    au, vu = _sorted_unique_mean(a, v)
    out = np.interp(elev, au, vu, left=vu[0], right=vu[-1])
    return np.clip(out, 0.0, None)


def reconstruct_spherical(
    cut_v: Optional[PatternCut],
    cut_h: Optional[PatternCut],
    mode: str,
    theta_deg: np.ndarray,
    phi_deg: np.ndarray,
    alpha: float = 1.0,
    beta: float = 1.0,
    eps: float = 1e-12,
    separable_mode: str = "direct",
) -> SphericalPattern:
    """
    Build a 3D spherical pattern from 2D cuts.

    mode:
      - omni: V-only, omnidirectional over phi
      - separable: V x H with selectable combination mode
      - harmonic: reserved for phase 2 (not implemented)
    separable_mode:
      - direct
      - preserve_peak
    """
    th = np.asarray(theta_deg, dtype=float).reshape(-1)
    ph = wrap_phi_deg(np.asarray(phi_deg, dtype=float).reshape(-1))
    if th.size == 0 or ph.size == 0:
        raise ValueError("theta_deg and phi_deg must be non-empty.")

    m = str(mode).strip().lower()
    if m == "harmonic":
        raise NotImplementedError("harmonic mode is planned (phase 2).")

    if m == "omni":
        if cut_v is None:
            raise ValueError("omni mode requires a vertical cut (cut_v).")
        vth = _sample_v_theta(cut_v, th)
        mag = np.repeat(vth[:, None], ph.size, axis=1)
    elif m == "separable":
        if cut_v is None or cut_h is None:
            raise ValueError("separable mode requires both cut_v and cut_h.")
        vth = _sample_v_theta(cut_v, th)
        hph = _sample_h_phi(cut_h, ph)

        va = np.power(np.maximum(vth, eps), float(alpha))
        hb = np.power(np.maximum(hph, eps), float(beta))

        sm = str(separable_mode).strip().lower()
        if sm == "direct":
            den = float(np.max(hb)) if hb.size else 1.0
            den = den if den > eps else 1.0
            mag = np.outer(va, hb) / den
        elif sm == "preserve_peak":
            v1 = va / max(float(np.max(va)), eps)
            h1 = hb / max(float(np.max(hb)), eps)
            mag = np.sqrt(np.outer(v1, h1))
        else:
            raise ValueError("Invalid separable_mode. Use 'direct' or 'preserve_peak'.")
    else:
        raise ValueError("Invalid mode. Use 'omni' | 'separable' | 'harmonic'.")

    mmax = float(np.max(mag)) if mag.size else 1.0
    if mmax > eps:
        mag = mag / mmax
    mag = np.clip(mag, 0.0, 1.0)

    return SphericalPattern(
        theta_deg=th,
        phi_deg=ph,
        mag_lin=mag,
        meta={
            "mode": m,
            "separable_mode": separable_mode,
            "alpha": float(alpha),
            "beta": float(beta),
            "theta_points": int(th.size),
            "phi_points": int(ph.size),
        },
    )


def sample_spherical(pattern: SphericalPattern, theta_deg: float, phi_deg: float) -> float:
    """Nearest-neighbor sample from a spherical grid."""
    th = np.asarray(pattern.theta_deg, dtype=float)
    ph = wrap_phi_deg(np.asarray(pattern.phi_deg, dtype=float))
    m = np.asarray(pattern.mag_lin, dtype=float)
    if m.shape != (th.size, ph.size):
        raise ValueError("Invalid spherical grid shape.")
    i = int(np.argmin(np.abs(th - float(theta_deg))))
    phi_wr = float(wrap_phi_deg(float(phi_deg)))
    # circular nearest in phi
    d = np.abs(((ph - phi_wr + 180.0) % 360.0) - 180.0)
    j = int(np.argmin(d))
    return float(m[i, j])


def cut_from_arrays(cut_type: str, angles_deg: np.ndarray, values_lin: np.ndarray, meta: Optional[dict] = None) -> PatternCut:
    t = str(cut_type).strip().upper()
    if t not in ("H", "V"):
        raise ValueError("cut_type must be H or V")
    a = np.asarray(angles_deg, dtype=float).reshape(-1)
    v = np.asarray(values_lin, dtype=float).reshape(-1)
    if a.size == 0 or v.size == 0 or a.size != v.size:
        raise ValueError("Invalid cut arrays.")
    v = np.clip(np.asarray(v, dtype=float), 0.0, None)
    vmax = float(np.max(v))
    if vmax > 1e-12:
        v = v / vmax
    return PatternCut(type=t, angles_deg=a, values_lin=v, meta=dict(meta or {}))
