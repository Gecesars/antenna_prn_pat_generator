import math

import numpy as np

from null_fill_synthesis import synth_null_fill_by_order, synth_null_fill_vertical, weights_to_harness


def _build_case():
    f_hz = 100e6
    lam = 299_792_458.0 / f_hz
    n = 4
    spacing = 0.8 * lam
    z_m = np.arange(n, dtype=float) * spacing
    eps = np.arange(0.0, 30.0 + 1e-9, 0.1)
    fill_bands = [{"eps_min_deg": 2.0, "eps_max_deg": 8.0, "floor_db": -14.0, "weight": 1.0}]
    return f_hz, z_m, eps, fill_bands


def test_base_case_modes():
    f_hz, z_m, eps, fill_bands = _build_case()
    res_amp = synth_null_fill_vertical(f_hz, z_m, eps, fill_bands, mode="amplitude", max_iters=8)
    res_phase = synth_null_fill_vertical(f_hz, z_m, eps, fill_bands, mode="phase", max_iters=8)
    res_both = synth_null_fill_vertical(f_hz, z_m, eps, fill_bands, mode="both", max_iters=8)

    m_amp = res_amp["band_metrics"][0]["min_db"]
    m_phase = res_phase["band_metrics"][0]["min_db"]
    m_both = res_both["band_metrics"][0]["min_db"]

    assert math.isfinite(m_amp)
    assert math.isfinite(m_phase)
    assert math.isfinite(m_both)
    # Em geral "both" e o mais capaz; tolerancia larga para robustez numerica.
    assert m_both >= max(m_amp, m_phase) - 2.0


def test_normalization_sum_abs2():
    f_hz, z_m, eps, fill_bands = _build_case()
    res = synth_null_fill_vertical(f_hz, z_m, eps, fill_bands, mode="both", max_iters=6, norm="sum_abs2_1")
    w = np.asarray(res["w"], dtype=complex)
    assert abs(float(np.sum(np.abs(w) ** 2)) - 1.0) < 1e-6


def test_robustness_reg_lambda():
    f_hz, z_m, eps, fill_bands = _build_case()
    for reg in (1e-4, 1e-3, 1e-2, 1e-1):
        res = synth_null_fill_vertical(f_hz, z_m, eps, fill_bands, mode="both", reg_lambda=reg, max_iters=6)
        w = np.asarray(res["w"], dtype=complex)
        assert np.isfinite(np.real(w)).all()
        assert np.isfinite(np.imag(w)).all()
        assert np.isfinite(float(res["condition_number"]))


def test_weights_to_harness_shapes():
    f_hz, z_m, eps, fill_bands = _build_case()
    res = synth_null_fill_vertical(f_hz, z_m, eps, fill_bands, mode="both", max_iters=5)
    harness = weights_to_harness(np.asarray(res["w"], dtype=complex), f_hz=f_hz, vf=0.78, ref_index=0)
    n = len(z_m)
    assert len(harness["amp"]) == n
    assert len(harness["p_frac"]) == n
    assert len(harness["phase_deg"]) == n
    assert len(harness["delta_len_m"]) == n
    assert abs(float(np.sum(harness["p_frac"])) - 1.0) < 1e-9


def test_null_fill_by_order_first_null_effective_and_stable():
    f_hz = 900e6
    lam = 299_792_458.0 / f_hz
    n = 6
    z_m = np.arange(n, dtype=float) * (0.65 * lam)
    eps = np.arange(-90.0, 90.0 + 1e-9, 0.1)

    res = synth_null_fill_by_order(
        f_hz=f_hz,
        z_m=z_m,
        eps_grid_deg=eps,
        null_order=1,
        null_fill_percent=20.0,
        mode="both",
        mainlobe_tilt_deg=0.0,
        reg_lambda=1e-5,
        max_iters=30,
        preserve_mainlobe_weight=30.0,
        fill_weight=32.0,
    )

    achieved = float(res.get("achieved_percent", float("nan")))
    peak_eps = abs(float(res.get("peak_eps_deg", 0.0)))

    assert math.isfinite(achieved)
    assert achieved >= 10.0
    assert peak_eps <= 2.5


def test_null_fill_by_order_changes_target_region():
    f_hz = 900e6
    lam = 299_792_458.0 / f_hz
    n = 8
    z_m = np.arange(n, dtype=float) * (0.65 * lam)
    eps = np.arange(-90.0, 90.0 + 1e-9, 0.1)

    r1 = synth_null_fill_by_order(
        f_hz=f_hz,
        z_m=z_m,
        eps_grid_deg=eps,
        null_order=1,
        null_fill_percent=10.0,
        mode="both",
        mainlobe_tilt_deg=0.0,
        reg_lambda=1e-5,
        max_iters=20,
    )
    r2 = synth_null_fill_by_order(
        f_hz=f_hz,
        z_m=z_m,
        eps_grid_deg=eps,
        null_order=2,
        null_fill_percent=10.0,
        mode="both",
        mainlobe_tilt_deg=0.0,
        reg_lambda=1e-5,
        max_iters=20,
    )

    c1 = sorted(abs(float(x.get("eps_deg", 0.0))) for x in (r1.get("null_regions", []) or []))
    c2 = sorted(abs(float(x.get("eps_deg", 0.0))) for x in (r2.get("null_regions", []) or []))
    assert len(c1) >= 1 and len(c2) >= 1
    assert max(c2) > max(c1)
