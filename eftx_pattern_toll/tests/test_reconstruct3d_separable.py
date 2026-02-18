import numpy as np

from core.reconstruct3d import cut_from_arrays, reconstruct_spherical


def test_reconstruct3d_separable_direct_outer_product_behavior():
    elev = np.array([-90.0, 0.0, 90.0])
    v = np.array([0.3, 1.0, 0.3])
    phi_cut = np.array([-180.0, 0.0, 180.0])
    h = np.array([0.5, 1.0, 0.5])

    cut_v = cut_from_arrays("V", elev, v)
    cut_h = cut_from_arrays("H", phi_cut, h)

    theta = np.array([0.0, 90.0, 180.0])
    phi = np.array([-180.0, 0.0, 180.0])

    sp = reconstruct_spherical(
        cut_v=cut_v,
        cut_h=cut_h,
        mode="separable",
        theta_deg=theta,
        phi_deg=phi,
        alpha=1.0,
        beta=1.0,
        separable_mode="direct",
    )

    assert sp.mag_lin.shape == (3, 3)
    assert np.isclose(float(np.max(sp.mag_lin)), 1.0)
    # Center should be peak
    assert sp.mag_lin[1, 1] >= sp.mag_lin[0, 1]
    assert sp.mag_lin[1, 1] >= sp.mag_lin[1, 0]
