from __future__ import annotations

from typing import Dict, List

from PySide6.QtCore import Qt, Signal, QPoint
from PySide6.QtWidgets import QWidget, QVBoxLayout, QTreeWidget, QTreeWidgetItem, QLabel, QLineEdit

from .context_menu import ContextInfo


class SceneTreeWidget(QWidget):
    selectionRequested = Signal(list, str)  # ids, mode
    contextRequested = Signal(object)  # ContextInfo

    def __init__(self, parent=None):
        super().__init__(parent)
        self._objects: Dict[str, object] = {}
        self._selection: List[str] = []
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        layout.addWidget(QLabel("Scene Tree"))
        self.search_edit = QLineEdit(self)
        self.search_edit.setPlaceholderText("Search by name/tag...")
        self.search_edit.textChanged.connect(self._apply_filter)
        layout.addWidget(self.search_edit)
        self.tree = QTreeWidget(self)
        self.tree.setHeaderLabels(["Name", "Visible", "Locked", "Source"])
        self.tree.setSelectionMode(QTreeWidget.ExtendedSelection)
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._on_context_menu)
        self.tree.itemSelectionChanged.connect(self._on_selection_changed)
        layout.addWidget(self.tree)

    def refresh(self, objects: Dict[str, object], selection: List[str]) -> None:
        self._objects = dict(objects)
        self._selection = list(selection)
        self._apply_filter()

    def _apply_filter(self):
        selected = set(self._selection)
        token = str(self.search_edit.text() or "").strip().lower()
        self.tree.blockSignals(True)
        self.tree.clear()
        for oid, obj in self._objects.items():
            name = str(getattr(obj, "name", oid))
            source = str(getattr(obj, "source", ""))
            meta = getattr(obj, "meta", {})
            group = str(meta.get("group", "")) if isinstance(meta, dict) else ""
            hay = " | ".join([name, source, group]).lower()
            if token and token not in hay:
                continue
            item = QTreeWidgetItem(
                [
                    name,
                    "Yes" if bool(getattr(obj, "visible", True)) else "No",
                    "Yes" if bool(getattr(obj, "locked", False)) else "No",
                    source,
                ]
            )
            item.setData(0, Qt.UserRole, str(oid))
            self.tree.addTopLevelItem(item)
            if oid in selected:
                item.setSelected(True)
        self.tree.blockSignals(False)

    def _on_selection_changed(self):
        ids = []
        for it in self.tree.selectedItems():
            oid = str(it.data(0, Qt.UserRole))
            if oid:
                ids.append(oid)
        self.selectionRequested.emit(ids, "replace")

    def _on_context_menu(self, pos: QPoint):
        item = self.tree.itemAt(pos)
        picked = None
        if item is not None:
            picked = str(item.data(0, Qt.UserRole) or "")
            if picked:
                item.setSelected(True)
        selected_ids = []
        for it in self.tree.selectedItems():
            oid = str(it.data(0, Qt.UserRole))
            if oid:
                selected_ids.append(oid)
        ctx = ContextInfo(
            widget="scene_tree",
            mouse_pos=pos,
            global_pos=self.tree.viewport().mapToGlobal(pos),
            picked_object_id=picked or None,
            selected_ids=selected_ids,
            tool_mode="Select",
        )
        self.contextRequested.emit(ctx)
