from __future__ import annotations

from pathlib import Path

import pytest

from mech.engine.scene_engine import SceneEngine


pytestmark = pytest.mark.mechanical


class _FakeCaps:
    def to_dict(self):
        return {
            "provider": "fake_kernel",
            "primitive_kinds": [],
            "import_formats": ["step", "stp"],
            "export_formats": [],
        }


class _FakeKernel:
    def __init__(self):
        self.capabilities = _FakeCaps()
        self.triangulate_calls = []

    def import_model(self, path: str, fmt: str | None = None):
        _ = (path, fmt)
        return ["kid_1", "kid_2", "kid_3"]

    def triangulate(self, obj_id: str, quality=None):
        self.triangulate_calls.append(dict(quality or {}))
        shift = float(obj_id.split("_")[-1])
        return {
            "vertices": [
                [0.0 + shift, 0.0, 0.0],
                [1.0 + shift, 0.0, 0.0],
                [0.0 + shift, 1.0, 0.0],
            ],
            "faces": [[0, 1, 2]],
        }


def test_import_model_multibody_assigns_asset_metadata(tmp_path: Path):
    eng = SceneEngine()
    fake = _FakeKernel()
    eng.kernel = fake
    eng._kernel_provider_name = "fake_kernel"

    src = tmp_path / "assembly.step"
    src.write_text("dummy", encoding="utf-8")

    ids = eng.import_model(
        str(src),
        fmt="step",
        name_hint="assembly",
        layer_name="cad_layer",
        triangulation_quality={"deflection": 0.05},
    )
    assert len(ids) == 3

    asset_ids = {str(eng.objects[oid].meta.get("import_asset_id", "")) for oid in ids}
    assert len(asset_ids) == 1
    assert all(asset_ids)

    counts = {int(eng.objects[oid].meta.get("import_body_count", 0)) for oid in ids}
    assert counts == {3}
    indexes = sorted(int(eng.objects[oid].meta.get("import_body_index", 0)) for oid in ids)
    assert indexes == [1, 2, 3]
    layers = {str(eng.objects[oid].meta.get("layer", "")) for oid in ids}
    assert layers == {"cad_layer"}
    assert len(fake.triangulate_calls) == 3
    assert {float(row.get("deflection", 0.0)) for row in fake.triangulate_calls} == {0.05}

    names = [eng.objects[oid].name for oid in ids]
    assert names == ["assembly_body_01", "assembly_body_02", "assembly_body_03"]


def test_clear_boundaries_only_selected_targets():
    eng = SceneEngine()
    a = eng.create_primitive("box", {"width": 1.0, "depth": 1.0, "height": 1.0}, name="A")
    b = eng.create_primitive("box", {"width": 1.0, "depth": 1.0, "height": 1.0}, name="B")

    b1 = eng.apply_boundary([a], "fixed", params={"value": 0.0})
    b2 = eng.apply_boundary([b], "force", params={"value": 5.0})
    assert b1 in eng.boundaries
    assert b2 in eng.boundaries

    removed = eng.clear_boundaries([a])
    assert removed == 1
    assert b1 not in eng.boundaries
    assert b2 in eng.boundaries


def test_retessellate_objects_uses_quality(tmp_path: Path):
    eng = SceneEngine()
    fake = _FakeKernel()
    eng.kernel = fake
    eng._kernel_provider_name = "fake_kernel"

    src = tmp_path / "retess.step"
    src.write_text("dummy", encoding="utf-8")
    ids = eng.import_model(str(src), fmt="step", name_hint="retess", triangulation_quality={"deflection": 0.1})
    assert len(ids) == 3
    assert len(fake.triangulate_calls) == 3

    changed = eng.retessellate_objects(ids, quality={"deflection": 0.02})
    assert changed == 3
    assert len(fake.triangulate_calls) == 6
    assert {float(row.get("deflection", 0.0)) for row in fake.triangulate_calls[-3:]} == {0.02}
