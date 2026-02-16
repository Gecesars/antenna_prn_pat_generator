import numpy as np

from core.angles import ang_dist_deg, elev_to_theta_deg, theta_to_elev_deg, wrap_phi_deg


def test_theta_elevation_roundtrip():
    eps = np.array([-90.0, -45.0, 0.0, 45.0, 90.0])
    theta = elev_to_theta_deg(eps)
    assert np.allclose(theta, [180.0, 135.0, 90.0, 45.0, 0.0])
    eps2 = theta_to_elev_deg(theta)
    assert np.allclose(eps2, eps)


def test_wrap_phi_deg():
    vals = np.array([-540.0, -181.0, -180.0, -179.0, 0.0, 179.0, 180.0, 181.0, 540.0])
    out = wrap_phi_deg(vals)
    assert np.all(out >= -180.0)
    assert np.all(out < 180.0)
    assert np.isclose(out[2], -180.0)
    assert np.isclose(out[6], -180.0)


def test_ang_dist_deg_wrap():
    assert np.isclose(float(ang_dist_deg(179.0, -179.0)), 2.0)
    assert np.isclose(float(ang_dist_deg(-170.0, 170.0)), 20.0)
