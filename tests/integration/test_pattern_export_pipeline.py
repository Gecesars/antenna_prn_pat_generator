from __future__ import annotations

import json

import numpy as np

from eftx_aedt_live.export import PatternExport


def test_pattern_export_pipeline_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("DEBUG_AUDIT", "1")

    exporter = PatternExport(tmp_path)
    ang = np.linspace(-180.0, 180.0, 361, dtype=float)
    val = np.clip(np.cos(np.deg2rad(ang)), 0.0, None)

    p_csv = exporter.save_cut_csv("hrp_case", ang, val, {"mode": "HRP"})
    p_json = exporter.save_cut_json("hrp_case", ang, val, {"mode": "HRP"})
    assert p_csv.is_file()
    assert p_json.is_file()

    payload = json.loads(p_json.read_text(encoding="utf-8"))
    assert len(payload["angles_deg"]) == 361
    assert len(payload["values"]) == 361
    assert payload["meta"]["mode"] == "HRP"

    theta = np.linspace(-90.0, 90.0, 181, dtype=float)
    phi = np.linspace(-180.0, 180.0, 361, dtype=float)
    thg, phg = np.meshgrid(theta, phi, indexing="ij")
    z = -20.0 + 10.0 * np.cos(np.deg2rad(thg)) * np.cos(np.deg2rad(phg))

    p_npz = exporter.save_grid_npz("grid_case", theta, phi, z, {"mode": "3D"})
    assert p_npz.is_file()
    with np.load(p_npz) as npz:
        assert npz["theta_deg"].shape[0] == 181
        assert npz["phi_deg"].shape[0] == 361
        assert tuple(npz["values"].shape) == (181, 361)

    p_obj, p_mtl = exporter.export_obj_from_db_grid(
        "grid_mesh",
        theta_deg=np.array([-90.0, 0.0, 90.0], dtype=float),
        phi_deg=np.array([-180.0, -90.0, 0.0, 90.0], dtype=float),
        values_db=np.array(
            [
                [-30.0, -20.0, -10.0, -20.0],
                [-15.0, -5.0, 0.0, -5.0],
                [-30.0, -20.0, -10.0, -20.0],
            ],
            dtype=float,
        ),
        db_min=-40.0,
        db_max=0.0,
        gamma=1.0,
    )
    assert p_obj.is_file()
    assert p_mtl.is_file()
    assert "mtllib grid_mesh.mtl" in p_obj.read_text(encoding="utf-8")

