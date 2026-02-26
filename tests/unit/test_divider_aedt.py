from __future__ import annotations

import pytest

from core.divider_aedt import (
    DividerGeometryError,
    _alt_hfss_constructor_kwargs,
    _build_hfss_constructor_kwargs,
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
