from __future__ import annotations

import json
import os
from typing import Any, Dict, Mapping


SCENE_SCHEMA_VERSION = "mechanical_scene_v1"


def serialize_scene_payload(scene_payload: Mapping[str, Any]) -> Dict[str, Any]:
    data = dict(scene_payload or {})
    data.setdefault("schema", SCENE_SCHEMA_VERSION)
    data.setdefault("objects", [])
    data.setdefault("groups", [])
    data.setdefault("selection_state", {})
    data.setdefault("view_state", {})
    data.setdefault("backend", {})
    data.setdefault("boundaries", [])
    data.setdefault("dirty", False)
    return data


def save_scene_payload(path: str, scene_payload: Mapping[str, Any]) -> str:
    out = os.path.abspath(str(path or "mechanical_scene_v1.json"))
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    data = serialize_scene_payload(scene_payload)
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return out


def load_scene_payload(path: str) -> Dict[str, Any]:
    src = os.path.abspath(str(path or ""))
    if not src or not os.path.isfile(src):
        raise FileNotFoundError(src)
    with open(src, "r", encoding="utf-8", errors="ignore") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise RuntimeError("Invalid scene payload format")
    return serialize_scene_payload(data)
