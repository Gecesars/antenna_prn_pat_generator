from __future__ import annotations

from typing import Any, Mapping, Sequence

from ..models import CapabilityReport, MechanicalError


class NullMechanicalProvider:
    """Fallback provider used when FreeCAD kernel is unavailable."""

    def __init__(self, reason: str = "FreeCAD is not available in this environment"):
        self._reason = str(reason or "FreeCAD is not available in this environment")
        self._capabilities = CapabilityReport(
            provider="null",
            freecad_available=False,
            freecad_headless_available=False,
            fem_available=False,
            validate_available=False,
            heal_available=False,
            import_formats=(),
            export_formats=(),
            primitive_kinds=(),
            boolean_ops=(),
            notes=(self._reason,),
        )

    @property
    def capabilities(self) -> CapabilityReport:
        return self._capabilities

    def _not_available(self, operation: str) -> MechanicalError:
        return MechanicalError(
            f"{operation} not available: {self._reason}",
            code="backend_unavailable",
            details={"provider": "null", "operation": str(operation)},
        )

    def create_primitive(self, kind: str, params: Mapping[str, Any]) -> str:
        raise self._not_available(f"create_primitive({kind})")

    def transform(self, obj_id: str, transform: Mapping[str, Any]) -> None:
        raise self._not_available("transform")

    def boolean(self, op: str, a_id: str, b_id: str | Sequence[str]) -> str:
        raise self._not_available(f"boolean({op})")

    def import_model(self, path: str, fmt: str | None = None) -> Sequence[str]:
        raise self._not_available(f"import_model({fmt or ''})")

    def export_model(self, obj_ids: Sequence[str], path: str, fmt: str | None = None) -> str:
        raise self._not_available(f"export_model({fmt or ''})")

    def get_properties(self, obj_id: str) -> Mapping[str, Any]:
        raise self._not_available("get_properties")

    def set_properties(self, obj_id: str, props: Mapping[str, Any]) -> None:
        raise self._not_available("set_properties")

    def triangulate(self, obj_id: str, quality: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
        raise self._not_available("triangulate")

    def validate(self, obj_id: str) -> Mapping[str, Any]:
        return {
            "ok": False,
            "provider": "null",
            "messages": [self._reason],
        }

    def heal(self, obj_id: str) -> Mapping[str, Any]:
        return {
            "ok": False,
            "provider": "null",
            "messages": [self._reason],
        }

    def delete(self, obj_id: str) -> None:
        return None

    def close(self) -> None:
        return None
