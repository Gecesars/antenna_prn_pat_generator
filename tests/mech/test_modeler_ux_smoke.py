from __future__ import annotations

import importlib.util

import pytest


HAS_PYSIDE6 = importlib.util.find_spec("PySide6") is not None
pytestmark = pytest.mark.mechanical


@pytest.mark.skipif(not HAS_PYSIDE6, reason="PySide6 not available")
def test_mechanics_page_builds(monkeypatch):
    from mech.ui.page_mechanics import MechanicsPage

    _ = monkeypatch
    required = [
        "_default_preferences",
        "_load_preferences",
        "_apply_preferences",
        "_build_ops_tabs",
        "load_layout",
        "save_layout",
        "reset_layout_default",
        "_apply_boolean_from_tab",
        "action_boolean_diagnose",
        "action_backend_diagnostics",
        "action_save_backend_report",
        "action_validate_selected",
        "action_heal_selected",
        "action_apply_boundary_from_tab",
        "action_apply_boundary_quick",
        "action_show_boundary_summary",
        "action_assign_selected_layer",
        "action_apply_grid_settings",
        "action_retessellate_selected",
        "action_reconnect_backend",
        "action_set_subselection_mode",
        "action_adjust_selected_face",
        "action_adjust_selected_edge",
        "action_set_visual_preset",
        "action_open_selected_analysis_tab",
        "action_toggle_left_panel",
        "action_toggle_right_panel",
        "action_toggle_bottom_panel",
        "action_toggle_side_panels",
        "action_cycle_selection_mode",
        "action_toggle_navigation_lod",
        "action_apply_right_transform",
        "action_components_insert_primitive",
        "action_components_create_from_selection",
        "action_components_insert_instance",
        "action_components_insert_template",
        "action_fem_new_study",
        "action_fem_remove_study",
        "action_fem_set_active_from_list",
        "action_fem_include_selected",
        "action_fem_assign_material_selected",
        "action_fem_add_bc",
        "action_fem_add_load",
        "action_fem_apply_mesh_config",
        "action_fem_generate_mesh",
        "action_fem_apply_solver_config",
        "action_fem_validate_study",
        "action_fem_run_solve",
        "action_fem_refresh_results",
    ]
    for name in required:
        assert hasattr(MechanicsPage, name), f"Missing method: {name}"
