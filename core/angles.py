from __future__ import annotations

import numpy as np


def elev_to_theta_deg(elev_deg):
    """
    Convert elevation eps [-90, 90] to HFSS polar theta [0, 180].
    theta = 90 - eps
    """
    return 90.0 - np.asarray(elev_deg, dtype=float)


def theta_to_elev_deg(theta_deg):
    """
    Convert HFSS polar theta [0, 180] to elevation eps [-90, 90].
    eps = 90 - theta
    """
    return 90.0 - np.asarray(theta_deg, dtype=float)


def wrap_phi_deg(phi_deg):
    """Wrap azimuth to [-180, 180)."""
    x = np.asarray(phi_deg, dtype=float)
    return ((x + 180.0) % 360.0) - 180.0


def ang_dist_deg(a_deg, b_deg):
    """Minimum circular distance between two azimuth angles, in degrees."""
    a = np.asarray(a_deg, dtype=float)
    b = np.asarray(b_deg, dtype=float)
    return np.abs(((a - b + 180.0) % 360.0) - 180.0)
