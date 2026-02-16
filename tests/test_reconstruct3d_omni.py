import numpy as np

from core.reconstruct3d import cut_from_arrays, reconstruct_spherical


def test_reconstruct3d_omni_phi_invariant():
    elev = np.linspace(-90.0, 90.0, 181)
    v = np.clip(np.cos(np.deg2rad(elev)), 0.0, 1.0)
    cut_v = cut_from_arrays("V", elev, v)

    theta = np.linspace(0.0, 180.0, 181)
    phi = np.linspace(-180.0, 180.0, 73)
    sp = reconstruct_spherical(cut_v=cut_v, cut_h=None, mode="omni", theta_deg=theta, phi_deg=phi)

    assert sp.mag_lin.shape == (theta.size, phi.size)
    # Omni: each theta row must be constant over phi
    row_std = np.std(sp.mag_lin, axis=1)
    assert float(np.max(row_std)) < 1e-10
