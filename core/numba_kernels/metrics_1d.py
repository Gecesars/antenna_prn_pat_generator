from __future__ import annotations

import math

import numpy as np

from .utils import njit_if_available


@njit_if_available(parallel=False, fastmath=False)
def _safe_log10_db_power(p: float) -> float:
    eps = 1e-300
    x = p if p > eps else eps
    return 10.0 * math.log10(x)


@njit_if_available(parallel=False, fastmath=False)
def _interp_linear_x(x1: float, y1: float, x2: float, y2: float, yth: float) -> float:
    dy = y2 - y1
    if abs(dy) <= 1e-15:
        return float(x1)
    t = (yth - y1) / dy
    if t < 0.0:
        t = 0.0
    if t > 1.0:
        t = 1.0
    return float(x1 + t * (x2 - x1))


@njit_if_available(parallel=False, fastmath=False)
def _interp_periodic(angles: np.ndarray, values: np.ndarray, xq: float, period: float) -> float:
    n = int(len(angles))
    if n <= 0:
        return float("nan")
    if n == 1:
        return float(values[0])

    a0 = float(angles[0])
    an = float(angles[n - 1])
    x = float(xq)
    while x < a0:
        x += period
    while x >= (a0 + period):
        x -= period

    if x <= an:
        for i in range(n - 1):
            x1 = float(angles[i])
            x2 = float(angles[i + 1])
            if x1 <= x <= x2:
                y1 = float(values[i])
                y2 = float(values[i + 1])
                if abs(x2 - x1) <= 1e-15:
                    return y1
                t = (x - x1) / (x2 - x1)
                return y1 + t * (y2 - y1)
        return float(values[n - 1])

    x1 = an
    y1 = float(values[n - 1])
    x2 = float(angles[0]) + period
    y2 = float(values[0])
    if abs(x2 - x1) <= 1e-15:
        return y1
    t = (x - x1) / (x2 - x1)
    return y1 + t * (y2 - y1)


def metrics_cut_1d_python(
    angles_deg: np.ndarray,
    e_lin: np.ndarray,
    span_mode: int,
    xdb: float,
):
    n = int(min(len(angles_deg), len(e_lin)))
    nan = float("nan")
    if n < 3:
        return (-1, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan)

    emax = 0.0
    for i in range(n):
        v = abs(float(e_lin[i]))
        if v > emax:
            emax = v
    if emax <= 1e-30:
        return (-1, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan)

    p = np.empty(n, dtype=np.float64)
    p_db = np.empty(n, dtype=np.float64)
    peak_idx = 0
    peak_p = -1.0
    for i in range(n):
        amp = abs(float(e_lin[i])) / emax
        pi = amp * amp
        p[i] = pi
        p_db[i] = _safe_log10_db_power(pi)
        if pi > peak_p:
            peak_p = pi
            peak_idx = i

    peak_angle = float(angles_deg[peak_idx])
    peak_db = float(p_db[peak_idx])
    thr = peak_db - abs(float(xdb))

    left = nan
    for i in range(peak_idx, 0, -1):
        y1 = float(p_db[i - 1])
        y2 = float(p_db[i])
        if (y1 < thr <= y2) or (y1 > thr >= y2):
            left = _interp_linear_x(float(angles_deg[i - 1]), y1, float(angles_deg[i]), y2, thr)
            break

    right = nan
    for i in range(peak_idx, n - 1):
        y1 = float(p_db[i])
        y2 = float(p_db[i + 1])
        if (y2 < thr <= y1) or (y2 > thr >= y1):
            right = _interp_linear_x(float(angles_deg[i]), y1, float(angles_deg[i + 1]), y2, thr)
            break

    hpbw = nan
    if math.isfinite(left) and math.isfinite(right):
        hpbw = float(abs(right - left))

    left_min_idx = -1
    right_min_idx = -1
    for i in range(1, n - 1):
        if (p[i] <= p[i - 1]) and (p[i] <= p[i + 1]):
            if i < peak_idx:
                left_min_idx = i
            elif (i > peak_idx) and (right_min_idx < 0):
                right_min_idx = i
    first_null_db = nan
    if (left_min_idx >= 0) and (right_min_idx >= 0):
        lv = float(p_db[left_min_idx])
        rv = float(p_db[right_min_idx])
        first_null_db = lv if lv > rv else rv
    elif left_min_idx >= 0:
        first_null_db = float(p_db[left_min_idx])
    elif right_min_idx >= 0:
        first_null_db = float(p_db[right_min_idx])

    back_angle = nan
    back_db = nan
    fb_db = nan
    if int(span_mode) == 0:
        back_angle = peak_angle + 180.0
        amin = float(angles_deg[0])
        amax = float(angles_deg[n - 1])
        while back_angle < amin:
            back_angle += 360.0
        while back_angle > amax:
            back_angle -= 360.0
        back_p = _interp_periodic(angles_deg, p, back_angle, 360.0)
        back_db = _safe_log10_db_power(back_p)
        fb_db = peak_db - back_db
    else:
        far_idx = 0
        far_dist = -1.0
        for i in range(n):
            d = abs(float(angles_deg[i]) - peak_angle)
            if d > far_dist:
                far_dist = d
                far_idx = i
        back_angle = float(angles_deg[far_idx])
        back_db = float(p_db[far_idx])
        fb_db = peak_db - back_db

    integ = 0.0
    k = math.pi / 180.0
    for i in range(n - 1):
        x1 = float(angles_deg[i]) * k
        x2 = float(angles_deg[i + 1]) * k
        dx = x2 - x1
        integ += 0.5 * (float(p[i]) + float(p[i + 1])) * dx

    d2d_lin = nan
    d2d_db = nan
    if integ > 1e-30:
        span = (2.0 * math.pi) if int(span_mode) == 0 else math.pi
        d2d_lin = float(span / integ)
        if d2d_lin > 1e-30:
            d2d_db = 10.0 * math.log10(d2d_lin)

    return (
        int(peak_idx),
        float(peak_angle),
        float(peak_db),
        float(hpbw),
        float(left),
        float(right),
        float(first_null_db),
        float(fb_db),
        float(back_angle),
        float(back_db),
        float(d2d_lin),
        float(d2d_db),
    )


metrics_cut_1d_numba = njit_if_available(parallel=False, fastmath=False)(metrics_cut_1d_python)
