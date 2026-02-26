from __future__ import annotations

import pytest

from mech.engine.scene_engine import SceneEngine


pytestmark = pytest.mark.mechanical


def test_fem_study_validation_and_solve_flow():
    eng = SceneEngine()
    a = eng.create_primitive("box", {"width": 1.0, "depth": 1.0, "height": 1.0}, name="A")
    b = eng.create_primitive("box", {"width": 1.0, "depth": 1.0, "height": 1.0}, name="B")

    sid = eng.fem.new_study(name="Static A", study_type="Structural Static", units="mm")
    eng.fem.include_bodies([a, b], fem_role="solid", study_id=sid)
    eng.fem.assign_material([a], "Steel", {"E": 210e9, "nu": 0.3}, study_id=sid)

    checks = eng.fem.validate(sid)
    assert checks
    assert any(str(row.get("status", "")).lower() == "error" for row in checks)
    assert any(str(row.get("code", "")) == "materials_complete" for row in checks)

    eng.fem.assign_material([b], "Steel", {"E": 210e9, "nu": 0.3}, study_id=sid)
    eng.fem.add_boundary_condition([a], "fixed support", {"value": 0.0}, study_id=sid)
    eng.fem.add_load([b], "force", {"value": 100.0, "direction": "z"}, study_id=sid)
    eng.fem.configure_mesh(study_id=sid, global_size=10.0, growth_rate=1.25, quality_target=0.72)
    eng.fem.mark_mesh_generated(True, quality_avg=0.74, study_id=sid)
    eng.fem.configure_solver(study_id=sid, type="direct", tolerance=1e-7, max_iterations=800, threads=0)

    checks_ok = eng.fem.validate(sid)
    assert not any(str(row.get("status", "")).lower() == "error" for row in checks_ok)

    result = eng.fem.run_solve(sid)
    assert "displacement_max_mm" in result
    assert "stress_von_mises_mpa" in result
    study = eng.fem.get(sid)
    assert study is not None
    assert str(study.solver.get("status", "")) == "completed"


def test_fem_state_roundtrip_serialize_restore():
    eng = SceneEngine()
    oid = eng.create_primitive("box", {"width": 1.0, "depth": 1.0, "height": 1.0}, name="Obj")
    sid = eng.fem.new_study(name="Roundtrip", study_type="Structural Static", units="mm")
    eng.fem.include_bodies([oid], study_id=sid)
    eng.fem.assign_material([oid], "Aluminum", {"E": 69e9, "nu": 0.33}, study_id=sid)

    state = eng.serialize_state()
    eng2 = SceneEngine()
    eng2.restore_state(state, emit=False)

    assert eng2.fem.active_study_id() == sid
    study = eng2.fem.get(sid)
    assert study is not None
    assert oid in study.bodies
    assert oid in study.materials
