from __future__ import annotations

from pathlib import Path

import pytest

from mech.engine.scene_engine import SceneEngine


pytestmark = pytest.mark.mechanical


def _last_command_name(engine: SceneEngine) -> str:
    return engine.command_stack._undo[-1].__class__.__name__


def test_scene_engine_uses_named_command_classes(tmp_path: Path):
    eng = SceneEngine()

    box_id = eng.create_primitive("box", {"width": 1.0, "depth": 1.0, "height": 1.0}, name="BoxA")
    assert box_id in eng.objects
    assert _last_command_name(eng) == "CreatePrimitiveCommand"

    eng.transform_objects([box_id], tx=1.0, ty=0.0, tz=0.0)
    assert _last_command_name(eng) == "TransformCommand"

    eng.rename_object(box_id, "BoxA_renamed")
    assert _last_command_name(eng) == "RenameCommand"

    eng.set_visibility([box_id], False)
    assert _last_command_name(eng) == "VisibilityCommand"

    eng.delete_objects([box_id])
    assert _last_command_name(eng) == "DeleteCommand"


def test_scene_engine_boolean_command_class():
    eng = SceneEngine()

    a = eng.create_primitive("box", {"width": 1.0, "depth": 1.0, "height": 1.0, "center": (0.0, 0.0, 0.0)}, name="A")
    b = eng.create_primitive("box", {"width": 1.0, "depth": 1.0, "height": 1.0, "center": (0.4, 0.0, 0.0)}, name="B")

    out = eng.boolean_union([a, b])
    assert out
    assert _last_command_name(eng) == "BooleanCommand"


def test_scene_engine_import_command_class(tmp_path: Path):
    eng = SceneEngine()

    obj_path = tmp_path / "tri.obj"
    obj_path.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")

    imported = eng.import_model(str(obj_path), fmt="obj")
    assert imported
    assert _last_command_name(eng) == "ImportCommand"


def test_scene_engine_boundary_command_class():
    eng = SceneEngine()
    box_id = eng.create_primitive("box", {"width": 1.0, "depth": 1.0, "height": 1.0}, name="BoundaryTarget")

    bid = eng.apply_boundary([box_id], "fixed", params={"value": 0.0, "direction": "normal"})
    assert bid in eng.boundaries
    assert _last_command_name(eng) == "BoundaryCommand"

    assert eng.remove_boundary(bid) is True
    assert _last_command_name(eng) == "BoundaryCommand"


def test_scene_engine_layer_flow():
    eng = SceneEngine()
    a = eng.create_primitive("box", {"width": 1.0, "depth": 1.0, "height": 1.0}, name="LayerA")
    b = eng.create_primitive("box", {"width": 1.0, "depth": 1.0, "height": 1.0}, name="LayerB")

    eng.set_objects_layer([a, b], "AssemblyLayer")
    assert eng.object_layer(a) == "AssemblyLayer"
    assert eng.object_layer(b) == "AssemblyLayer"

    rows = {str(r.get("name", "")): dict(r) for r in eng.layer_rows()}
    assert rows["AssemblyLayer"]["count"] == 2

    changed = eng.set_layer_visibility("AssemblyLayer", False)
    assert changed == 2
    assert eng.objects[a].visible is False
    assert eng.objects[b].visible is False

    touched = eng.set_layer_color("AssemblyLayer", "#ff3355", apply_to_objects=True)
    assert touched == 2
    assert eng.objects[a].color == "#ff3355"
    assert eng.objects[b].color == "#ff3355"

    assert eng.rename_layer("AssemblyLayer", "AssemblyLayerRenamed") is True
    assert eng.object_layer(a) == "AssemblyLayerRenamed"
    assert eng.object_layer(b) == "AssemblyLayerRenamed"

    assert eng.delete_layer("AssemblyLayerRenamed") is True
    assert eng.object_layer(a) == "Default"
    assert eng.object_layer(b) == "Default"


def test_scene_engine_face_and_edge_adjust():
    eng = SceneEngine()
    oid = eng.create_primitive("box", {"width": 1.0, "depth": 1.0, "height": 1.0}, name="Geom")
    obj = eng.objects[oid]
    base_vertices = obj.mesh.vertices.copy()
    face_idx = 0

    info = eng.face_info(oid, face_idx)
    assert info["face_index"] == face_idx
    assert len(info["vertex_ids"]) == 3

    assert eng.adjust_face_offset(oid, face_idx, 0.2) is True
    moved_vertices = eng.objects[oid].mesh.vertices
    assert (moved_vertices != base_vertices).any()

    edge = eng.pick_edge_from_face(oid, face_idx, point=info["centroid"])
    assert edge is not None
    assert eng.adjust_edge_offset(oid, edge, 0.1) is True
