from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence

from .models import CapabilityReport


class MechanicalKernel(Protocol):
    @property
    def capabilities(self) -> CapabilityReport:
        ...

    def create_primitive(self, kind: str, params: Mapping[str, Any]) -> str:
        ...

    def transform(self, obj_id: str, transform: Mapping[str, Any]) -> None:
        ...

    def boolean(self, op: str, a_id: str, b_id: str | Sequence[str]) -> str:
        ...

    def import_model(self, path: str, fmt: str | None = None) -> Sequence[str]:
        ...

    def export_model(self, obj_ids: Sequence[str], path: str, fmt: str | None = None) -> str:
        ...

    def get_properties(self, obj_id: str) -> Mapping[str, Any]:
        ...

    def set_properties(self, obj_id: str, props: Mapping[str, Any]) -> None:
        ...

    def triangulate(self, obj_id: str, quality: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
        ...

    def validate(self, obj_id: str) -> Mapping[str, Any]:
        ...

    def heal(self, obj_id: str) -> Mapping[str, Any]:
        ...

    def delete(self, obj_id: str) -> None:
        ...

    def close(self) -> None:
        ...
