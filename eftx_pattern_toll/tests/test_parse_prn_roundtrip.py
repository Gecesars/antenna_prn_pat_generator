import os
import tempfile

import numpy as np

from deep3 import parse_prn, write_prn_file


def test_prn_roundtrip_basic():
    h_angles = np.arange(-180.0, 181.0, 1.0)
    h_values = np.clip(np.cos(np.deg2rad(h_angles)), 0.0, 1.0)
    v_angles = np.arange(-90.0, 90.0 + 1e-9, 0.1)
    v_values = np.clip(np.cos(np.deg2rad(v_angles)), 0.0, 1.0)

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "roundtrip.prn")
        write_prn_file(
            path=path,
            name="ROUNDTRIP",
            make="EFTX",
            frequency=100.0,
            freq_unit="MHz",
            h_width=65.0,
            v_width=12.0,
            front_to_back=20.0,
            gain=5.0,
            h_angles=h_angles,
            h_values=h_values,
            v_angles=v_angles,
            v_values=v_values,
        )

        out = parse_prn(path)
        assert len(out["h_angles"]) == 360
        assert len(out["v_angles"]) == 360
        assert np.isfinite(out["h_values"]).all()
        assert np.isfinite(out["v_values"]).all()
        assert float(np.max(out["h_values"])) <= 1.0 + 1e-9
        assert float(np.max(out["v_values"])) <= 1.0 + 1e-9

