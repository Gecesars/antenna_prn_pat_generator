import os
import tempfile

import numpy as np

from deep3 import (
    write_pat_conventional_combined,
    write_pat_horizontal_new_format,
    write_pat_vertical_new_format,
)


def test_pat_horizontal_roundtrip_basic():
    angles = np.arange(-180.0, 181.0, 1.0)
    values = np.clip(np.cos(np.deg2rad(angles)), 0.0, 1.0)

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "h.pat")
        write_pat_horizontal_new_format(path, "H ROUNDTRIP", 0.0, 1, angles, values, step=1)
        arr = np.loadtxt(path, delimiter=",", skiprows=1)
        ang_out = arr[:, 0]
        val_out = arr[:, 1]

        assert len(ang_out) >= 360
        assert np.isfinite(val_out).all()
        assert float(np.max(ang_out)) <= 360.0 + 1e-9


def test_pat_vertical_roundtrip_basic():
    angles = np.arange(-90.0, 90.0 + 1e-9, 0.1)
    values = np.clip(np.cos(np.deg2rad(angles)), 0.0, 1.0)

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "v.pat")
        write_pat_vertical_new_format(path, "V ROUNDTRIP", 0.0, 1, angles, values, step=1)
        arr = np.loadtxt(path, delimiter=",", skiprows=1)
        ang_out = arr[:, 0]
        val_out = arr[:, 1]

        assert len(ang_out) >= 360
        assert np.isfinite(val_out).all()
        assert float(np.max(ang_out)) <= 360.0 + 1e-9


def test_pat_conventional_combined_matches_edx_shape():
    h_angles = np.arange(-180.0, 181.0, 1.0)
    h_values = np.clip(np.cos(np.deg2rad(h_angles)) ** 2, 0.0, 1.0)
    v_angles = np.arange(-90.0, 90.0 + 1e-9, 0.1)
    v_values = np.clip(np.cos(np.deg2rad(v_angles)) ** 2, 0.0, 1.0)

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "combined.pat")
        write_pat_conventional_combined(
            path=path,
            description="EDX STYLE",
            gain=16.65,
            num_antennas=1,
            h_angles=h_angles,
            h_values=h_values,
            v_angles=v_angles,
            v_values=v_values,
        )
        lines = open(path, "r", encoding="utf-8").read().splitlines()
        # Header + 360 HRP + "999" + "1, 91" + "269," + 91 VRP
        assert len(lines) == 1 + 360 + 1 + 1 + 1 + 91
        assert lines[361] == "999"
        assert lines[362].startswith("1, ")
        assert lines[363].endswith(",")
