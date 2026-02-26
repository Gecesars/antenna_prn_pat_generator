from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from PySide6.QtCore import QObject, Signal


class SelectionMode(str, Enum):
    OBJECT = "object"
    FACE = "face"
    EDGE = "edge"
    VERTEX = "vertex"
    BODY = "body"
    COMPONENT = "component"

    @classmethod
    def from_value(cls, value: object) -> "SelectionMode":
        token = str(value or "").strip().lower()
        for mode in cls:
            if mode.value == token:
                return mode
        return cls.OBJECT


@dataclass
class SelectionItem:
    entity_id: str
    entity_type: str
    parent_object_id: str | None = None
    sub_index: int | None = None
    display_name: str = ""
    layer: str | None = None
    locked: bool = False
    visible: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def key(self) -> Tuple[str, str, str, int]:
        return (
            str(self.entity_type or "object"),
            str(self.entity_id or ""),
            str(self.parent_object_id or ""),
            int(self.sub_index) if self.sub_index is not None else -1,
        )


def _dedupe_items(items: Iterable[SelectionItem]) -> List[SelectionItem]:
    out: List[SelectionItem] = []
    seen = set()
    for item in items:
        key = item.key()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


class SelectionManager(QObject):
    selection_changed = Signal(list)
    hover_changed = Signal(object)
    active_item_changed = Signal(object)
    mode_changed = Signal(str)
    filters_changed = Signal(dict)

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self.mode = SelectionMode.OBJECT
        self._selected: List[SelectionItem] = []
        self._hover: SelectionItem | None = None
        self._active: SelectionItem | None = None
        self._filters = {
            "allow_hidden": False,
            "allow_locked": False,
            "layers": None,
            "types": None,
        }

    # ---------------- state getters ----------------
    def current_selection(self) -> List[SelectionItem]:
        return list(self._selected)

    def selected_ids(self) -> List[str]:
        ids: List[str] = []
        for item in self._selected:
            if item.entity_type in {"object", "body", "component"}:
                ids.append(str(item.entity_id))
            elif item.parent_object_id:
                ids.append(str(item.parent_object_id))
        out: List[str] = []
        seen = set()
        for oid in ids:
            if oid in seen:
                continue
            seen.add(oid)
            out.append(oid)
        return out

    def active_item(self) -> SelectionItem | None:
        return self._active

    def hover_item(self) -> SelectionItem | None:
        return self._hover

    def has_selection(self) -> bool:
        return bool(self._selected)

    # ---------------- mode / filters ----------------
    def set_mode(self, mode: SelectionMode | str) -> None:
        out = SelectionMode.from_value(mode)
        if out == self.mode:
            return
        self.mode = out
        self.mode_changed.emit(self.mode.value)

    def set_filters(
        self,
        *,
        allow_hidden: Optional[bool] = None,
        allow_locked: Optional[bool] = None,
        layers: Optional[Sequence[str]] = None,
        types: Optional[Sequence[str]] = None,
    ) -> None:
        changed = False
        if allow_hidden is not None:
            v = bool(allow_hidden)
            if v != bool(self._filters["allow_hidden"]):
                self._filters["allow_hidden"] = v
                changed = True
        if allow_locked is not None:
            v = bool(allow_locked)
            if v != bool(self._filters["allow_locked"]):
                self._filters["allow_locked"] = v
                changed = True
        if layers is not None:
            layer_set = {str(x).strip() for x in layers if str(x).strip()}
            v = layer_set or None
            if v != self._filters["layers"]:
                self._filters["layers"] = v
                changed = True
        if types is not None:
            type_set = {str(x).strip().lower() for x in types if str(x).strip()}
            v = type_set or None
            if v != self._filters["types"]:
                self._filters["types"] = v
                changed = True
        if changed:
            self.filters_changed.emit(dict(self._filters))
            self._compact_by_filters()

    def _accepts(self, item: SelectionItem) -> bool:
        if not bool(self._filters.get("allow_hidden", False)) and not bool(item.visible):
            return False
        if not bool(self._filters.get("allow_locked", False)) and bool(item.locked):
            return False
        layers = self._filters.get("layers")
        if layers is not None and str(item.layer or "") not in layers:
            return False
        types = self._filters.get("types")
        if types is not None and str(item.entity_type or "").lower() not in types:
            return False
        return True

    # ---------------- selection mutations ----------------
    def clear(self) -> None:
        self._set_selection([])
        self._set_active(None)

    def set_single(self, item: SelectionItem) -> None:
        if not self._accepts(item):
            self.clear()
            return
        self._set_selection([item])
        self._set_active(item)

    def add(self, item: SelectionItem) -> None:
        if not self._accepts(item):
            return
        self._set_selection(_dedupe_items([*self._selected, item]))
        self._set_active(item)

    def toggle(self, item: SelectionItem) -> None:
        if not self._accepts(item):
            return
        key = item.key()
        if any(row.key() == key for row in self._selected):
            out = [row for row in self._selected if row.key() != key]
            self._set_selection(out)
            self._set_active(out[0] if out else None)
            return
        self.add(item)

    def remove(self, item: SelectionItem) -> None:
        key = item.key()
        out = [row for row in self._selected if row.key() != key]
        self._set_selection(out)
        self._set_active(out[0] if out else None)

    def apply_mode(self, item: SelectionItem | None, op: str = "replace") -> SelectionItem | None:
        mode = str(op or "replace").strip().lower()
        if item is None:
            if mode == "replace":
                self.clear()
            return None
        if mode == "add":
            self.add(item)
        elif mode == "toggle":
            self.toggle(item)
        elif mode in {"remove", "subtract"}:
            self.remove(item)
        else:
            self.set_single(item)
        return item

    def set_from_tree(self, items: Sequence[SelectionItem], op: str = "replace") -> None:
        rows = [row for row in items if isinstance(row, SelectionItem) and self._accepts(row)]
        rows = _dedupe_items(rows)
        mode = str(op or "replace").strip().lower()
        if mode == "add":
            self._set_selection(_dedupe_items([*self._selected, *rows]))
        elif mode == "toggle":
            out = list(self._selected)
            by_key = {row.key(): row for row in out}
            for row in rows:
                key = row.key()
                if key in by_key:
                    by_key.pop(key, None)
                else:
                    by_key[key] = row
            out2 = list(by_key.values())
            self._set_selection(_dedupe_items(out2))
        elif mode in {"remove", "subtract"}:
            remove_keys = {row.key() for row in rows}
            self._set_selection([row for row in self._selected if row.key() not in remove_keys])
        else:
            self._set_selection(rows)
        self._set_active(rows[0] if rows else (self._selected[0] if self._selected else None))

    def set_from_ids(self, ids: Sequence[str], objects: Optional[Mapping[str, Any]] = None, op: str = "replace") -> None:
        rows: List[SelectionItem] = []
        refs = dict(objects or {})
        for oid in [str(x) for x in ids if str(x).strip()]:
            obj = refs.get(oid)
            meta = getattr(obj, "meta", {}) if obj is not None else {}
            if not isinstance(meta, dict):
                meta = {}
            rows.append(
                SelectionItem(
                    entity_id=str(oid),
                    entity_type="object",
                    parent_object_id=None,
                    display_name=str(getattr(obj, "name", oid)),
                    layer=str(meta.get("layer", "Default") or "Default"),
                    locked=bool(getattr(obj, "locked", False)),
                    visible=bool(getattr(obj, "visible", True)),
                    metadata={"source": str(getattr(obj, "source", ""))},
                )
            )
        self.set_from_tree(rows, op=op)

    def set_from_viewport_pick(self, pick_result: Mapping[str, Any], op: str = "replace") -> SelectionItem | None:
        row = dict(pick_result or {})
        oid = str(row.get("object_id", "")).strip()
        if not oid:
            if str(op or "replace").strip().lower() == "replace":
                self.clear()
            return None
        display = str(row.get("display_name", oid) or oid)
        layer = str(row.get("layer", "") or "")
        locked = bool(row.get("locked", False))
        visible = bool(row.get("visible", True))
        picked_cell = row.get("picked_cell_id")
        sub_index = int(picked_cell) if isinstance(picked_cell, (int, float)) and int(picked_cell) >= 0 else None

        entity_type = self.mode.value
        entity_id = oid
        parent = None
        if self.mode in {SelectionMode.FACE, SelectionMode.EDGE, SelectionMode.VERTEX}:
            if sub_index is None:
                if str(op or "replace").strip().lower() == "replace":
                    self.clear()
                return None
            parent = oid
            entity_id = f"{oid}:{entity_type}:{int(sub_index)}"
            display = f"{display} [{entity_type} {int(sub_index)}]"
        elif self.mode == SelectionMode.BODY:
            entity_type = "body"
        elif self.mode == SelectionMode.COMPONENT:
            entity_type = "component"
        else:
            entity_type = "object"

        item = SelectionItem(
            entity_id=str(entity_id),
            entity_type=str(entity_type),
            parent_object_id=parent,
            sub_index=sub_index if parent is not None else None,
            display_name=display,
            layer=layer or None,
            locked=locked,
            visible=visible,
            metadata={
                "picked_point": row.get("picked_point"),
                "modifiers": row.get("modifiers"),
            },
        )
        return self.apply_mode(item, op=op)

    def box_select(self, items: Sequence[SelectionItem], op: str = "replace") -> None:
        rows = [x for x in items if isinstance(x, SelectionItem) and self._accepts(x)]
        self.set_from_tree(rows, op=op)

    def lasso_select(self, items: Sequence[SelectionItem], op: str = "replace") -> None:
        rows = [x for x in items if isinstance(x, SelectionItem) and self._accepts(x)]
        self.set_from_tree(rows, op=op)

    # ---------------- hover ----------------
    def set_hover(self, item: SelectionItem | None) -> None:
        if item is not None and not self._accepts(item):
            item = None
        old = self._hover.key() if self._hover is not None else None
        new = item.key() if item is not None else None
        if old == new:
            return
        self._hover = item
        self.hover_changed.emit(item)

    # ---------------- scene sync ----------------
    def sync_scene(self, objects: Mapping[str, Any]) -> None:
        refs = dict(objects or {})
        out: List[SelectionItem] = []
        for item in self._selected:
            oid = str(item.entity_id)
            if item.parent_object_id:
                oid = str(item.parent_object_id)
            if oid not in refs:
                continue
            obj = refs[oid]
            meta = getattr(obj, "meta", {})
            if not isinstance(meta, dict):
                meta = {}
            refreshed = SelectionItem(
                entity_id=str(item.entity_id),
                entity_type=str(item.entity_type),
                parent_object_id=str(item.parent_object_id) if item.parent_object_id else None,
                sub_index=item.sub_index,
                display_name=str(item.display_name or getattr(obj, "name", oid)),
                layer=str(meta.get("layer", "Default") or "Default"),
                locked=bool(getattr(obj, "locked", False)),
                visible=bool(getattr(obj, "visible", True)),
                metadata=dict(item.metadata),
            )
            if self._accepts(refreshed):
                out.append(refreshed)
        self._set_selection(out)
        if self._active is not None and not any(x.key() == self._active.key() for x in out):
            self._set_active(out[0] if out else None)
        if self._hover is not None:
            oid = str(self._hover.entity_id)
            if self._hover.parent_object_id:
                oid = str(self._hover.parent_object_id)
            if oid not in refs:
                self.set_hover(None)

    # ---------------- internals ----------------
    def _compact_by_filters(self) -> None:
        out = [x for x in self._selected if self._accepts(x)]
        self._set_selection(out)
        if self._active is not None and not any(x.key() == self._active.key() for x in out):
            self._set_active(out[0] if out else None)
        if self._hover is not None and not self._accepts(self._hover):
            self.set_hover(None)

    def _set_selection(self, items: Sequence[SelectionItem]) -> None:
        out = _dedupe_items(items)
        old_keys = [x.key() for x in self._selected]
        new_keys = [x.key() for x in out]
        if old_keys == new_keys:
            self._selected = list(out)
            return
        self._selected = list(out)
        self.selection_changed.emit(list(self._selected))

    def _set_active(self, item: SelectionItem | None) -> None:
        old = self._active.key() if self._active is not None else None
        new = item.key() if item is not None else None
        if old == new:
            self._active = item
            return
        self._active = item
        self.active_item_changed.emit(item)
