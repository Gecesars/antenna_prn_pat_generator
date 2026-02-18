import numpy as np

from core.metrics import beamwidth_xdb


def test_beamwidth_xdb_linear_profile():
    angles = np.arange(-90.0, 91.0, 1.0)
    mag_db = -np.abs(angles)

    bw = beamwidth_xdb(angles, mag_db, xdb=10.0, wrap=False)
    assert np.isfinite(bw["width_deg"])
    assert abs(bw["width_deg"] - 20.0) < 0.5
    assert abs(bw["left_deg"] + 10.0) < 0.6
    assert abs(bw["right_deg"] - 10.0) < 0.6
