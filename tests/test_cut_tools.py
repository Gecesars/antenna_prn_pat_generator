import numpy as np

from eftx_aedt_live.cut_tools import shift_cut_no_interp, transform_cut, wrap_to_180


def test_wrap_to_180_domain():
    a = np.array([-540.0, -190.0, -180.0, 0.0, 180.0, 190.0, 540.0])
    out = wrap_to_180(a)
    assert np.all(out >= -180.0)
    assert np.all(out < 180.0 + 1e-12)


def test_transform_cut_hrp_align_peak_zero():
    a = np.linspace(-180.0, 180.0, 361)
    # Peak intentionally off-center near +35 deg
    v = np.exp(-0.5 * ((a - 35.0) / 20.0) ** 2)
    out_a, out_v, meta = transform_cut(a, v, mode="HRP", rotation_deg=0.0, align_peak_zero=True, target_points=361)
    peak = float(out_a[int(np.argmax(out_v))])
    assert out_a.size == 361
    assert np.isclose(float(np.min(out_a)), -180.0)
    assert np.isclose(float(np.max(out_a)), 180.0)
    assert abs(peak) <= 1.0
    assert abs(float(meta["peak_after_deg"])) <= 1.0


def test_transform_cut_hrp_align_peak_to_90():
    a = np.linspace(-180.0, 180.0, 361)
    # Peak intentionally off-center near -42 deg
    v = np.exp(-0.5 * ((a + 42.0) / 16.0) ** 2)
    out_a, out_v, meta = transform_cut(a, v, mode="HRP", rotation_deg=90.0, align_peak_zero=True, target_points=361)
    peak = float(out_a[int(np.argmax(out_v))])
    assert out_a.size == 361
    assert abs(peak - 90.0) <= 1.0
    assert abs(float(meta["peak_after_deg"]) - 90.0) <= 1.0


def test_shift_cut_no_interp_hrp_preserves_values_and_count():
    a = np.linspace(-179.0, 179.0, 359)
    v = np.exp(-0.5 * ((a - 47.0) / 18.0) ** 2)
    out_a, out_v, meta = shift_cut_no_interp(a, v, mode="HRP", target_peak_deg=90.0)
    assert out_a.size == a.size
    assert out_v.size == v.size
    assert abs(float(out_a[int(np.argmax(out_v))]) - 90.0) <= 1.0
    np.testing.assert_allclose(np.sort(out_v), np.sort(v), rtol=0.0, atol=1e-12)
    assert abs(float(meta["peak_after_deg"]) - 90.0) <= 1.0


def test_shift_cut_no_interp_vrp_preserves_values_and_count():
    a = np.linspace(-90.0, 90.0, 181)
    v = np.exp(-0.5 * ((a + 23.0) / 14.0) ** 2)
    out_a, out_v, meta = shift_cut_no_interp(a, v, mode="VRP", target_peak_deg=0.0)
    assert out_a.size == a.size
    assert out_v.size == v.size
    assert abs(float(out_a[int(np.argmax(out_v))])) <= 1.0
    np.testing.assert_allclose(np.sort(out_v), np.sort(v), rtol=0.0, atol=1e-12)
    assert abs(float(meta["peak_after_deg"])) <= 1.0


def test_shift_cut_no_interp_vrp_full_theta_domain_avoids_overlap():
    # Simulate a full theta sweep where only one branch should define VRP.
    a = np.linspace(-180.0, 180.0, 361)
    v = np.exp(-0.5 * ((a - 90.0) / 12.0) ** 2)
    out_a, out_v, meta = shift_cut_no_interp(a, v, mode="VRP", target_peak_deg=0.0)
    assert np.all(out_a >= -90.0 - 1e-9)
    assert np.all(out_a <= 90.0 + 1e-9)
    # After selecting a coherent branch and shifting, angles should be unique.
    assert out_a.size == np.unique(np.round(out_a, 6)).size
    assert abs(float(out_a[int(np.argmax(out_v))])) <= 1.0
    assert abs(float(meta["peak_after_deg"])) <= 1.0
    # Peak neighborhood should remain coherent (no artificial center notch).
    i0 = int(np.argmin(np.abs(out_a - 0.0)))
    assert out_v[i0] > 0.95
    if i0 > 0:
        assert out_v[i0 - 1] > 0.90
    if i0 + 1 < out_v.size:
        assert out_v[i0 + 1] > 0.90


def test_shift_cut_no_interp_vrp_full_theta_domain_peak_at_zero_kept_centered():
    a = np.linspace(-180.0, 180.0, 361)
    v = np.exp(-0.5 * (a / 18.0) ** 2)
    out_a, out_v, meta = shift_cut_no_interp(a, v, mode="VRP", target_peak_deg=0.0)
    i0 = int(np.argmin(np.abs(out_a - 0.0)))
    assert abs(float(out_a[int(np.argmax(out_v))])) <= 1.0
    assert out_v[i0] > 0.99
    if i0 > 0:
        assert out_v[i0 - 1] > 0.95
    if i0 + 1 < out_v.size:
        assert out_v[i0 + 1] > 0.95
    assert abs(float(meta["peak_after_deg"])) <= 1.0


def test_transform_cut_vrp_rotate_and_resample():
    a = np.linspace(-90.0, 90.0, 181)
    v = np.exp(-0.5 * ((a + 10.0) / 15.0) ** 2)
    out_a, out_v, meta = transform_cut(a, v, mode="VRP", rotation_deg=10.0, align_peak_zero=False, target_points=181)
    peak = float(out_a[int(np.argmax(out_v))])
    assert out_a.size == 181
    assert np.isclose(float(np.min(out_a)), -90.0)
    assert np.isclose(float(np.max(out_a)), 90.0)
    # Peak was at -10 deg; with +10 deg rotation it should move near 0 deg.
    assert abs(peak) <= 1.5
    assert abs(float(meta["peak_after_deg"])) <= 1.5


def test_transform_cut_vrp_align_peak_zero_from_boundary():
    a = np.linspace(-90.0, 90.0, 181)
    # Peak at boundary
    v = np.exp(-0.5 * ((a + 90.0) / 8.0) ** 2)
    out_a, out_v, meta = transform_cut(a, v, mode="VRP", rotation_deg=0.0, align_peak_zero=True, target_points=181)
    peak = float(out_a[int(np.argmax(out_v))])
    assert abs(peak) <= 1.5
    assert abs(float(meta["peak_after_deg"])) <= 1.5


def test_transform_cut_vrp_converts_theta_0_180_to_elevation():
    theta = np.linspace(0.0, 180.0, 181)
    # Peak at theta=90 -> elev=0
    v = np.exp(-0.5 * ((theta - 90.0) / 12.0) ** 2)
    out_a, out_v, _ = transform_cut(theta, v, mode="VRP", rotation_deg=0.0, align_peak_zero=False, target_points=181)
    peak = float(out_a[int(np.argmax(out_v))])
    assert np.isclose(float(np.min(out_a)), -90.0)
    assert np.isclose(float(np.max(out_a)), 90.0)
    assert abs(peak) <= 1.0
