from __future__ import annotations

from typing import Dict, List

from PySide6.QtCore import Qt, Signal, QPoint
from PySide6.QtWidgets import QWidget, QVBoxLayout, QTreeWidget, QTreeWidgetItem, QLabel, QLineEdit

from .context_menu import ContextInfo


class SceneTreeWidget(QWidget):
    selectionRequested = Signal(list, str)  # ids, mode
    visibilityRequested = Signal(list, bool)  # ids, visible
    contextRequested = Signal(object)  # ContextInfo
    ROLE_NODE_TYPE = int(Qt.UserRole + 1)
    ROLE_GROUP_IDS = int(Qt.UserRole + 2)
    COL_NAME = 0
    COL_TYPE = 1
    COL_VISIBLE = 2
    COL_LOCKED = 3
    COL_LAYER = 4
    COL_MATERIAL = 5
    COL_FEM_ROLE = 6
    COL_SOLVE = 7

    def __init__(self, parent=None):
        super().__init__(parent)
        self._objects: Dict[str, object] = {}
        self._selection: List[str] = []
        self._updating = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        layout.addWidget(QLabel("Scene Tree"))
        self.search_edit = QLineEdit(self)
        self.search_edit.setPlaceholderText("Search by name/type/layer/material/tag...")
        self.search_edit.textChanged.connect(self._apply_filter)
        layout.addWidget(self.search_edit)
        self.tree = QTreeWidget(self)
        self.tree.setHeaderLabels(["Name", "Type", "Visible", "Locked", "Layer", "Material", "FEM Role", "Solve"])
        self.tree.setSelectionMode(QTreeWidget.ExtendedSelection)
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._on_context_menu)
        self.tree.itemSelectionChanged.connect(self._on_selection_changed)
        self.tree.itemChanged.connect(self._on_item_changed)
        layout.addWidget(self.tree)

    def refresh(self, objects: Dict[str, object], selection: List[str]) -> None:
        self._objects = dict(objects)
        self._selection = list(selection)
        self._apply_filter()

    def _apply_filter(self):
        selected = set(self._selection)
        token = str(self.search_edit.text() or "").strip().lower()
        self._updating = True
        self.tree.blockSignals(True)
        self.tree.clear()

        grouped: Dict[str, dict] = {}
        singles: List[str] = []
        for oid, obj in self._objects.items():
            name = str(getattr(obj, "name", oid))
            source = str(getattr(obj, "source", ""))
            meta = getattr(obj, "meta", {})
            if not isinstance(meta, dict):
                meta = {}
            group = str(meta.get("group", ""))
            layer = str(meta.get("layer", "Default") or "Default")
            obj_type = str(meta.get("type", "") or "")
            if not obj_type:
                src_l = source.lower()
                if "import" in src_l:
                    obj_type = "imported"
                elif "primitive" in src_l:
                    obj_type = "primitive"
                else:
                    obj_type = "object"
            material = str(meta.get("material", meta.get("obj_material", "")) or "")
            fem_role = str(meta.get("fem_role", "solid") or "solid")
            tags = meta.get("tags", [])
            if isinstance(tags, list):
                tags_txt = ",".join([str(x) for x in tags if str(x).strip()])
            else:
                tags_txt = str(tags or "")
            import_asset_name = str(meta.get("import_asset_name", ""))
            import_asset_path = str(meta.get("import_asset_path", ""))
            hay = " | ".join([name, source, group, layer, obj_type, material, fem_role, tags_txt]).lower()
            import_hay = " | ".join([import_asset_name, import_asset_path]).lower()
            if token and token not in hay and token not in import_hay:
                continue
            import_asset_id = str(meta.get("import_asset_id", "")).strip()
            try:
                body_count = int(meta.get("import_body_count", 0) or 0)
            except Exception:
                body_count = 0
            if import_asset_id and body_count > 1:
                row = grouped.setdefault(
                    import_asset_id,
                    {
                        "name": str(import_asset_name or "Imported assembly"),
                        "source": str(import_asset_path or source),
                        "ids": [],
                    },
                )
                row["ids"].append(str(oid))
            else:
                singles.append(str(oid))

        def _vis_state(ids: List[str]) -> Qt.CheckState:
            vis = [bool(getattr(self._objects.get(oid), "visible", True)) for oid in ids if oid in self._objects]
            if not vis:
                return Qt.Unchecked
            if all(vis):
                return Qt.Checked
            if any(vis):
                return Qt.PartiallyChecked
            return Qt.Unchecked

        def _add_object_item(parent, oid: str):
            if oid not in self._objects:
                return
            obj = self._objects[oid]
            name = str(getattr(obj, "name", oid))
            meta = getattr(obj, "meta", {})
            if not isinstance(meta, dict):
                meta = {}
            source = str(getattr(obj, "source", ""))
            obj_type = str(meta.get("type", "") or "")
            if not obj_type:
                src_l = source.lower()
                if "import" in src_l:
                    obj_type = "imported"
                elif "primitive" in src_l:
                    obj_type = "primitive"
                else:
                    obj_type = "object"
            layer = str(meta.get("layer", "Default") or "Default")
            material = str(meta.get("material", meta.get("obj_material", "")) or "")
            fem_role = str(meta.get("fem_role", "solid") or "solid")
            solve_flag = "No" if bool(meta.get("exclude_from_solve", False)) else "Yes"
            item = QTreeWidgetItem(
                [
                    name,
                    obj_type,
                    "Yes" if bool(getattr(obj, "visible", True)) else "No",
                    "Yes" if bool(getattr(obj, "locked", False)) else "No",
                    layer,
                    material,
                    fem_role,
                    solve_flag,
                ]
            )
            item.setData(self.COL_NAME, Qt.UserRole, str(oid))
            item.setData(self.COL_NAME, self.ROLE_NODE_TYPE, "object")
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(self.COL_VISIBLE, Qt.Checked if bool(getattr(obj, "visible", True)) else Qt.Unchecked)
            if parent is None:
                self.tree.addTopLevelItem(item)
            else:
                parent.addChild(item)
            if oid in selected:
                item.setSelected(True)

        for asset_id, row in grouped.items():
            ids = [str(x) for x in row.get("ids", []) if str(x) in self._objects]
            if not ids:
                continue
            label = str(row.get("name", "Imported assembly") or "Imported assembly")
            source = str(row.get("source", "") or "")
            parent = QTreeWidgetItem(
                [
                    f"{label} ({len(ids)} bodies)",
                    "assembly",
                    "Mixed" if _vis_state(ids) == Qt.PartiallyChecked else ("Yes" if _vis_state(ids) == Qt.Checked else "No"),
                    "-",
                    "-",
                    "-",
                    "-",
                    "-",
                ]
            )
            parent.setData(self.COL_NAME, Qt.UserRole, "")
            parent.setData(self.COL_NAME, self.ROLE_NODE_TYPE, "group")
            parent.setData(self.COL_NAME, self.ROLE_GROUP_IDS, list(ids))
            parent.setFlags((parent.flags() | Qt.ItemIsUserCheckable) & ~Qt.ItemIsSelectable)
            parent.setCheckState(self.COL_VISIBLE, _vis_state(ids))
            self.tree.addTopLevelItem(parent)
            for oid in sorted(ids, key=lambda x: str(getattr(self._objects.get(x), "name", x)).lower()):
                _add_object_item(parent, oid)

        for oid in sorted(singles, key=lambda x: str(getattr(self._objects.get(x), "name", x)).lower()):
            _add_object_item(None, oid)

        self.tree.expandAll()
        self.tree.blockSignals(False)
        self._updating = False

    def _on_selection_changed(self):
        ids = []
        for it in self.tree.selectedItems():
            node_type = str(it.data(self.COL_NAME, self.ROLE_NODE_TYPE) or "object").strip().lower()
            if node_type != "object":
                continue
            oid = str(it.data(self.COL_NAME, Qt.UserRole))
            if oid:
                ids.append(oid)
        self.selectionRequested.emit(ids, "replace")

    def _on_item_changed(self, item: QTreeWidgetItem, column: int):
        if self._updating:
            return
        if int(column) != self.COL_VISIBLE:
            return
        node_type = str(item.data(self.COL_NAME, self.ROLE_NODE_TYPE) or "object").strip().lower()
        state = item.checkState(self.COL_VISIBLE)
        visible = bool(state != Qt.Unchecked)
        if node_type == "group":
            ids = [str(x) for x in item.data(self.COL_NAME, self.ROLE_GROUP_IDS) or [] if str(x)]
            item.setText(self.COL_VISIBLE, "Mixed" if state == Qt.PartiallyChecked else ("Yes" if visible else "No"))
            if ids:
                self.visibilityRequested.emit(ids, visible)
            return
        oid = str(item.data(self.COL_NAME, Qt.UserRole) or "")
        item.setText(self.COL_VISIBLE, "Yes" if visible else "No")
        if oid:
            self.visibilityRequested.emit([oid], visible)

    def _on_context_menu(self, pos: QPoint):
        item = self.tree.itemAt(pos)
        picked = None
        picked_is_group = False
        group_ids = []
        if item is not None:
            node_type = str(item.data(self.COL_NAME, self.ROLE_NODE_TYPE) or "object").strip().lower()
            picked = str(item.data(self.COL_NAME, Qt.UserRole) or "")
            picked_is_group = node_type == "group"
            if picked_is_group:
                group_ids = [str(x) for x in item.data(self.COL_NAME, self.ROLE_GROUP_IDS) or [] if str(x)]
            if picked:
                item.setSelected(True)
        selected_ids = []
        for it in self.tree.selectedItems():
            node_type = str(it.data(self.COL_NAME, self.ROLE_NODE_TYPE) or "object").strip().lower()
            if node_type != "object":
                continue
            oid = str(it.data(self.COL_NAME, Qt.UserRole))
            if oid:
                selected_ids.append(oid)
        if not selected_ids and picked_is_group and group_ids:
            selected_ids = list(group_ids)
        ctx = ContextInfo(
            widget="scene_tree",
            mouse_pos=pos,
            global_pos=self.tree.viewport().mapToGlobal(pos),
            picked_object_id=picked or None,
            selected_ids=selected_ids,
            tool_mode="Select",
        )
        self.contextRequested.emit(ctx)
