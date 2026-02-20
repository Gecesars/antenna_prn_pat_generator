from __future__ import annotations

import importlib.util

import pytest


HAS_PYSIDE6 = importlib.util.find_spec("PySide6") is not None


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
    ]
    for name in required:
        assert hasattr(MechanicsPage, name), f"Missing method: {name}"
