from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Tuple


class MechanicalError(RuntimeError):
    """Controlled error type for mechanical provider operations."""

    def __init__(self, message: str, *, code: str = "mechanical_error", details: Mapping[str, Any] | None = None):
        super().__init__(str(message))
        self.code = str(code or "mechanical_error")
        self.details: Dict[str, Any] = dict(details or {})

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "message": str(self),
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class CapabilityReport:
    provider: str
    freecad_available: bool = False
    freecad_headless_available: bool = False
    fem_available: bool = False
    validate_available: bool = False
    heal_available: bool = False
    import_formats: Tuple[str, ...] = ()
    export_formats: Tuple[str, ...] = ()
    primitive_kinds: Tuple[str, ...] = ()
    boolean_ops: Tuple[str, ...] = ()
    notes: Tuple[str, ...] = ()
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "freecad_available": bool(self.freecad_available),
            "freecad_headless_available": bool(self.freecad_headless_available),
            "fem_available": bool(self.fem_available),
            "validate_available": bool(self.validate_available),
            "heal_available": bool(self.heal_available),
            "import_formats": list(self.import_formats),
            "export_formats": list(self.export_formats),
            "primitive_kinds": list(self.primitive_kinds),
            "boolean_ops": list(self.boolean_ops),
            "notes": list(self.notes),
            "extra": dict(self.extra),
        }


@dataclass(frozen=True)
class MeshPayload:
    vertices: Tuple[Tuple[float, float, float], ...]
    faces: Tuple[Tuple[int, int, int], ...]
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vertices": [list(v) for v in self.vertices],
            "faces": [list(f) for f in self.faces],
            "meta": dict(self.meta),
        }


@dataclass(frozen=True)
class MechanicalDiagnostics:
    report: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.report)
