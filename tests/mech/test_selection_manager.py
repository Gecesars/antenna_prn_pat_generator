from __future__ import annotations

import importlib.util

import pytest


HAS_PYSIDE6 = importlib.util.find_spec("PySide6") is not None
pytestmark = pytest.mark.mechanical


class _Obj:
    def __init__(self, name: str, *, locked: bool = False, visible: bool = True, layer: str = "Default"):
        self.name = name
        self.locked = locked
        self.visible = visible
        self.meta = {"layer": layer}
        self.source = "Local"


@pytest.mark.skipif(not HAS_PYSIDE6, reason="PySide6 not available")
def test_selection_manager_modes_and_toggle():
    from mech.ui.selection_manager import SelectionManager

    mgr = SelectionManager()
    objects = {"a": _Obj("A"), "b": _Obj("B")}

    mgr.set_from_ids(["a"], objects, op="replace")
    assert mgr.selected_ids() == ["a"]

    mgr.set_from_ids(["b"], objects, op="add")
    assert mgr.selected_ids() == ["a", "b"]

    mgr.set_from_ids(["a"], objects, op="toggle")
    assert mgr.selected_ids() == ["b"]


@pytest.mark.skipif(not HAS_PYSIDE6, reason="PySide6 not available")
def test_selection_manager_face_pick_maps_parent_object():
    from mech.ui.selection_manager import SelectionManager

    mgr = SelectionManager()
    mgr.set_mode("face")
    out = mgr.set_from_viewport_pick(
        {
            "object_id": "obj_1",
            "display_name": "Body-1",
            "picked_cell_id": 12,
            "visible": True,
            "locked": False,
            "layer": "L1",
        },
        op="replace",
    )
    assert out is not None
    assert out.entity_type == "face"
    assert out.parent_object_id == "obj_1"
    assert mgr.selected_ids() == ["obj_1"]


@pytest.mark.skipif(not HAS_PYSIDE6, reason="PySide6 not available")
def test_selection_manager_locked_filter_blocks_selection():
    from mech.ui.selection_manager import SelectionManager

    mgr = SelectionManager()
    objects = {"lock": _Obj("Locked", locked=True), "free": _Obj("Free", locked=False)}

    mgr.set_from_ids(["lock"], objects, op="replace")
    assert mgr.selected_ids() == []

    mgr.set_from_ids(["free"], objects, op="replace")
    assert mgr.selected_ids() == ["free"]
