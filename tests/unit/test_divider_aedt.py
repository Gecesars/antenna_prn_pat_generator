from __future__ import annotations

import pytest

from core.divider_aedt import (
    DividerGeometryError,
    _extract_s11_metrics_from_hfss,
    _alt_hfss_constructor_kwargs,
    _build_hfss_constructor_kwargs,
    _normalize_project_target,
    analyze_impedance_adjustments,
    compute_s11_metrics,
    compute_coaxial_divider_geometry,
)


def _base_params() -> dict:
    return {
        "f_start": 800.0,
        "f_stop": 1200.0,
        "d_ext": 20.0,
        "wall_thick": 1.5,
        "n_sections": 4,
        "n_outputs": 4,
        "diel_material": "Ar",
    }


def test_compute_coaxial_divider_geometry_shape_and_positive_values():
    out = compute_coaxial_divider_geometry(_base_params())
    assert out["f0_mhz"] == pytest.approx(1000.0)
    assert len(out["main_diams"]) == 4
    assert len(out["z_sects"]) == 4
    assert out["sec_len_mm"] > 0.0
    assert out["len_inner_mm"] > 0.0
    assert out["len_outer_mm"] > out["len_inner_mm"]
    assert out["d_int_tube"] > 0.0
    assert out["dia_saida_diel"] > 0.0
    assert out["dia_saida_cond"] > 0.0


def test_compute_coaxial_divider_geometry_output_diameter_rules():
    p4 = _base_params()
    p4["n_outputs"] = 4
    o4 = compute_coaxial_divider_geometry(p4)
    assert o4["dia_saida_diel"] == pytest.approx(o4["d_int_tube"])

    p5 = _base_params()
    p5["n_outputs"] = 5
    o5 = compute_coaxial_divider_geometry(p5)
    assert o5["dia_saida_diel"] == pytest.approx(o5["d_int_tube"] / 2.2)
    assert o5["dia_saida_cond"] == pytest.approx(o5["dia_saida_diel"] / 2.3)

    p7 = _base_params()
    p7["n_outputs"] = 7
    o7 = compute_coaxial_divider_geometry(p7)
    assert o7["dia_saida_diel"] == pytest.approx(o7["d_int_tube"] / 3.0)
    assert o7["dia_saida_cond"] == pytest.approx(o7["dia_saida_diel"] / 2.3)


def test_compute_coaxial_divider_geometry_invalid_inputs_raise():
    with pytest.raises(DividerGeometryError):
        compute_coaxial_divider_geometry({**_base_params(), "f_stop": 700.0})
    with pytest.raises(DividerGeometryError):
        compute_coaxial_divider_geometry({**_base_params(), "n_sections": 0})
    with pytest.raises(DividerGeometryError):
        compute_coaxial_divider_geometry({**_base_params(), "diel_material": "Unknown"})


def test_build_hfss_constructor_kwargs_supports_new_and_legacy_signatures():
    class _LegacyHfss:
        def __init__(self, projectname=None, designname=None, solution_type=None, new_desktop=False, close_on_exit=False):
            pass

    class _NewHfss:
        def __init__(self, project=None, design=None, solution_type=None, new_desktop=False, close_on_exit=False):
            pass

    kw_old = _build_hfss_constructor_kwargs(
        _LegacyHfss,
        project_path="x.aedt",
        design_name="D",
        solution_type="Modal",
        new_desktop=True,
        close_on_exit=False,
    )
    assert kw_old["projectname"] == "x.aedt"
    assert kw_old["designname"] == "D"
    assert kw_old["solution_type"] == "Modal"

    kw_new = _build_hfss_constructor_kwargs(
        _NewHfss,
        project_path="x.aedt",
        design_name="D",
        solution_type="Modal",
        new_desktop=True,
        close_on_exit=False,
    )
    assert kw_new["project"] == "x.aedt"
    assert kw_new["design"] == "D"
    assert kw_new["solution_type"] == "Modal"


def test_alt_hfss_constructor_kwargs_switches_project_design_names():
    base_new = {
        "project": "x.aedt",
        "design": "D",
        "solution_type": "Modal",
        "new_desktop": True,
        "close_on_exit": False,
    }
    alt_old = _alt_hfss_constructor_kwargs(base_new)
    assert alt_old is not None
    assert alt_old["projectname"] == "x.aedt"
    assert alt_old["designname"] == "D"
    assert "project" not in alt_old
    assert "design" not in alt_old

    base_old = {
        "projectname": "x.aedt",
        "designname": "D",
        "solution_type": "Modal",
    }
    alt_new = _alt_hfss_constructor_kwargs(base_old)
    assert alt_new is not None
    assert alt_new["project"] == "x.aedt"
    assert alt_new["design"] == "D"
    assert "projectname" not in alt_new
    assert "designname" not in alt_new


def test_compute_s11_metrics_returns_expected_arrays():
    s11 = [0.1 + 0.0j, 0.2 + 0.1j, 0.0 - 0.5j]
    out = compute_s11_metrics(s11, z0=50.0)
    assert len(out["return_loss_db"]) == 3
    assert len(out["impedance_ohm"]) == 3
    assert len(out["phase_deg"]) == 3
    assert out["return_loss_db"][0] > 0.0
    assert out["impedance_ohm"][0] > 0.0


def test_normalize_project_target_handles_aedt_file_and_directory(tmp_path):
    aedt_file = tmp_path / "Divisor_Coaxial.aedt"
    pf, pd = _normalize_project_target(str(aedt_file))
    assert pf == str(aedt_file)
    assert pd == str(tmp_path)

    folder = tmp_path / "projects"
    folder.mkdir()
    pf2, pd2 = _normalize_project_target(str(folder))
    assert pf2 == ""
    assert pd2 == str(folder)


def test_extract_s11_metrics_reads_y_vector_from_solution_data():
    class _FakeSolution:
        def __init__(self):
            self.primary_sweep = "Freq"
            self.units_sweeps = {"Freq": "MHz"}
            self.primary_sweep_values = [600.0, 700.0]

        def get_expression_data(self, expression=None, formula="real", **kwargs):
            x = [600.0, 700.0]
            if formula == "real":
                return x, [0.1, 0.2]
            if formula == "imag":
                return x, [0.0, 0.0]
            return x, [0.0, 0.0]

    class _FakePost:
        def get_solution_data(self, **kwargs):
            return _FakeSolution()

    class _FakeHfss:
        def __init__(self):
            self.design_name = "DivisorCoaxial"
            self.project_path = ""
            self.post = _FakePost()
            self.oproject = object()
            self._project_path = ""

        def set_active_design(self, *args, **kwargs):
            return True

    out = _extract_s11_metrics_from_hfss(
        _FakeHfss(),
        setup_name="Setup1",
        sweep_name="Sweep1",
        port_name="P1",
        project_path=r"C:\tmp\Divisor_Coaxial.aedt",
    )
    assert out["expression"] == "S(P1,P1)"
    assert out["frequency"] == pytest.approx([600.0, 700.0])
    assert out["return_loss_db"][0] == pytest.approx(20.0, rel=1e-6)
    assert out["return_loss_db"][1] == pytest.approx(13.9794000867, rel=1e-6)


def test_analyze_impedance_adjustments_returns_viable_suggestions():
    rf_data = {
        "frequency": [600.0, 650.0, 700.0, 750.0, 800.0],
        "return_loss_db": [8.0, 9.5, 10.0, 9.0, 8.5],
        "impedance_real_ohm": [58.0, 56.0, 55.0, 56.0, 57.0],
        "impedance_imag_ohm": [8.0, 5.0, 3.0, 4.0, 6.0],
    }
    params = {
        "f_start": 600.0,
        "f_stop": 800.0,
        "d_ext": 33.0,
        "wall_thick": 1.5,
        "n_sections": 4,
        "n_outputs": 4,
    }
    out = analyze_impedance_adjustments(rf_data, params)
    assert len(out) == 2
    keys = {row["variable"] for row in out}
    assert "d_ext_mm" in keys
    assert "wall_thick_mm" in keys
    assert "f_start_mhz" not in keys
    assert "f_stop_mhz" not in keys
    assert "n_sections" not in keys
    assert "outputs" not in keys
    assert "dielectric" not in keys

    d_ext_row = next(row for row in out if row["variable"] == "d_ext_mm")
    wall_row = next(row for row in out if row["variable"] == "wall_thick_mm")
    assert d_ext_row["suggested"] < params["d_ext"]  # Zreal high => reduce d_ext
    assert wall_row["suggested"] > params["wall_thick"]  # Zreal high => increase wall
