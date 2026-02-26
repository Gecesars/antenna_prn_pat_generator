from __future__ import annotations

from pathlib import Path

import pytest

from mech.mechanical.io import SCENE_SCHEMA_VERSION, load_scene_payload, save_scene_payload
from mech.mechanical.validators import validate_scene_payload


pytestmark = pytest.mark.mechanical


def test_scene_payload_save_load_roundtrip(tmp_path: Path):
    payload = {
        "schema": SCENE_SCHEMA_VERSION,
        "objects": [{"id": "1", "name": "obj"}],
        "groups": [],
        "selection_state": {"selected": ["1"]},
        "view_state": {"camera": [0, 0, 1]},
        "backend": {"provider": "null"},
        "boundaries": [],
        "dirty": False,
    }
    path = tmp_path / "scene.json"
    saved = save_scene_payload(str(path), payload)
    loaded = load_scene_payload(saved)
    report = validate_scene_payload(loaded)

    assert Path(saved).exists()
    assert loaded["schema"] == SCENE_SCHEMA_VERSION
    assert report["ok"] is True
    assert report["boundary_count"] == 0
