from __future__ import annotations

from typing import Any, Dict, Mapping

from .io import SCENE_SCHEMA_VERSION


def validate_scene_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    data = dict(payload or {})
    messages = []

    schema = str(data.get("schema", "")).strip()
    if schema != SCENE_SCHEMA_VERSION:
        messages.append(f"Unexpected schema: {schema or '<missing>'}")

    objects = data.get("objects", [])
    if not isinstance(objects, list):
        messages.append("objects must be a list")

    groups = data.get("groups", [])
    if not isinstance(groups, list):
        messages.append("groups must be a list")

    backend = data.get("backend", {})
    if not isinstance(backend, dict):
        messages.append("backend must be an object")
    boundaries = data.get("boundaries", [])
    if not isinstance(boundaries, list):
        messages.append("boundaries must be a list")

    return {
        "ok": len(messages) == 0,
        "messages": messages,
        "schema": schema,
        "object_count": len(objects) if isinstance(objects, list) else 0,
        "group_count": len(groups) if isinstance(groups, list) else 0,
        "boundary_count": len(boundaries) if isinstance(boundaries, list) else 0,
    }
