from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
from PySide6.QtCore import Qt, QPoint, QByteArray
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QApplication,
    QColorDialog,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QComboBox,
    QCheckBox,
    QLineEdit,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from mech.engine.scene_engine import SceneEngine
from mech.engine.scene_object import MeshData
from mech.engine.geometry_ops import transform_mesh
from mech.engine.measures import object_metrics

from .context_menu import ContextInfo, ContextMenuDispatcher
from .measurements_panel import MeasurementsPanel
from .properties_panel import PropertiesPanel
from .scene_tree import SceneTreeWidget
from .viewport_pyvista import ViewportPyVista


class MechanicsPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.engine = SceneEngine()
        self.dispatcher = ContextMenuDispatcher(self)
        self._snap_enabled = False
        self._snap_step = 1.0
        self._angle_snap_step = 5.0
        self._boolean_tolerance = 1e-6
        self._measure_kind = ""
        self._measure_points: List[tuple] = []
        self._last_cursor_ctx: Optional[ContextInfo] = None
        self._copied_transform: Optional[dict] = None
        self._boolean_primary_id: str = ""
        self._boolean_tool_ids: List[str] = []
        self._boolean_keep_originals = False
        self._prefs = self._default_preferences()
        self._config_dir = Path.home() / ".eftx_converter" / "mech"
        self._layouts_dir = self._config_dir / "layouts"
        self._build_ui()
        self._connect()
        self._load_preferences()
        self._apply_preferences()
        self.load_layout("layout_default", silent=True)
        self.engine.add_listener(self._on_engine_event)
        self.refresh_all()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        top = QHBoxLayout()
        top.addWidget(QLabel("Analises Mecanicas - 3D Modeler (PySide6 + PyVistaQt)"))
        self.btn_import = QPushButton("Import Mesh")
        self.btn_import.clicked.connect(self.action_import_mesh)
        top.addWidget(self.btn_import)
        self.btn_snapshot = QPushButton("Screenshot")
        self.btn_snapshot.clicked.connect(self.action_screenshot)
        top.addWidget(self.btn_snapshot)
        self.btn_markers = QPushButton("Save Markers")
        self.btn_markers.clicked.connect(self._save_markers)
        top.addWidget(self.btn_markers)
        self.btn_load_markers = QPushButton("Load Markers")
        self.btn_load_markers.clicked.connect(self._load_markers)
        top.addWidget(self.btn_load_markers)
        self.btn_undo = QPushButton("Undo")
        self.btn_redo = QPushButton("Redo")
        self.btn_undo.clicked.connect(self.action_undo)
        self.btn_redo.clicked.connect(self.action_redo)
        top.addWidget(self.btn_undo)
        top.addWidget(self.btn_redo)
        self.profile_combo = QComboBox(self)
        self.profile_combo.addItems(["CAD Classic", "Minimal", "Analysis"])
        self.profile_combo.currentTextChanged.connect(self._on_profile_changed)
        top.addWidget(QLabel("Profile"))
        top.addWidget(self.profile_combo)
        self.units_combo = QComboBox(self)
        self.units_combo.addItems(["mm", "cm", "m", "in"])
        self.units_combo.currentTextChanged.connect(self._on_units_changed)
        top.addWidget(QLabel("Units"))
        top.addWidget(self.units_combo)
        self.chk_snap = QCheckBox("Snap Grid", self)
        self.chk_snap.toggled.connect(lambda v: self._set_snap_enabled(v))
        top.addWidget(self.chk_snap)
        self.btn_pref = QPushButton("Preferences")
        self.btn_pref.clicked.connect(self.action_open_preferences)
        top.addWidget(self.btn_pref)
        self.layout_combo = QComboBox(self)
        self.layout_combo.addItems(["layout_default", "layout_modeling", "layout_analysis"])
        top.addWidget(QLabel("Layout"))
        top.addWidget(self.layout_combo)
        self.btn_layout_save = QPushButton("Save Layout")
        self.btn_layout_save.clicked.connect(self._save_layout_from_combo)
        top.addWidget(self.btn_layout_save)
        self.btn_layout_load = QPushButton("Load Layout")
        self.btn_layout_load.clicked.connect(self._load_layout_from_combo)
        top.addWidget(self.btn_layout_load)
        self.btn_layout_reset = QPushButton("Reset Layout")
        self.btn_layout_reset.clicked.connect(self.reset_layout_default)
        top.addWidget(self.btn_layout_reset)
        root.addLayout(top)

        self.ops_tabs = self._build_ops_tabs()
        root.addWidget(self.ops_tabs)

        self.main_split = QSplitter(Qt.Horizontal, self)
        self.scene_tree = SceneTreeWidget(self)
        self.viewport = ViewportPyVista(self)

        right_wrap = QWidget(self)
        right_layout = QVBoxLayout(right_wrap)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(6)
        self.properties = PropertiesPanel(self)
        self.log_box = QTextEdit(self)
        self.log_box.setReadOnly(True)
        self.log_box.setMinimumHeight(120)
        self.log_box.setContextMenuPolicy(Qt.CustomContextMenu)
        self.log_box.customContextMenuRequested.connect(self._context_log)
        self.inline_wizard_box = QTextEdit(self)
        self.inline_wizard_box.setReadOnly(True)
        self.inline_wizard_box.setMaximumHeight(120)
        self.inline_wizard_box.setPlaceholderText("Inline Wizard")
        right_layout.addWidget(self.properties, 3)
        right_layout.addWidget(QLabel("Inline Wizard"))
        right_layout.addWidget(self.inline_wizard_box, 1)
        right_layout.addWidget(QLabel("Log"))
        right_layout.addWidget(self.log_box, 2)

        self.main_split.addWidget(self.scene_tree)
        self.main_split.addWidget(self.viewport)
        self.main_split.addWidget(right_wrap)
        self.main_split.setStretchFactor(0, 1)
        self.main_split.setStretchFactor(1, 4)
        self.main_split.setStretchFactor(2, 2)

        self.measurements = MeasurementsPanel(self)

        self.outer_split = QSplitter(Qt.Vertical, self)
        self.outer_split.addWidget(self.main_split)
        self.outer_split.addWidget(self.measurements)
        self.outer_split.setStretchFactor(0, 5)
        self.outer_split.setStretchFactor(1, 2)
        root.addWidget(self.outer_split, 1)

        self.hint_label = QLabel("Hint: Click to select. Shift adds selection. RMB opens full context menu.")
        root.addWidget(self.hint_label)

    def _connect(self):
        self.scene_tree.selectionRequested.connect(self._on_tree_selection)
        self.scene_tree.contextRequested.connect(self._show_context_menu)
        self.viewport.selectionRequested.connect(self._on_viewport_selection)
        self.viewport.contextRequested.connect(self._show_context_menu)
        self.viewport.statusMessage.connect(self._log)
        self.viewport.measurePointPicked.connect(self._on_measure_point)
        self.properties.applyTransformRequested.connect(self._on_apply_transform)
        self.properties.contextRequested.connect(self._show_context_menu)
        self.measurements.contextRequested.connect(self._show_context_menu)

    def _set_hint(self, text: str, wizard_lines: Optional[Sequence[str]] = None):
        msg = str(text or "").strip()
        if not msg:
            msg = "Click to select. Shift adds selection. RMB opens full context menu."
        self.hint_label.setText(f"Hint: {msg}")
        if wizard_lines is not None:
            lines = [str(x) for x in wizard_lines if str(x).strip()]
            self.inline_wizard_box.setPlainText("\n".join(lines))

    def _default_preferences(self) -> dict:
        return {
            "profile": "CAD Classic",
            "units": "mm",
            "grid": {"enabled": True, "step_mm": 5.0, "size_mm": 2000.0, "color": "#2b2b2b"},
            "snap": {"grid": True, "step_mm": 5.0, "angle": True, "step_deg": 5.0},
            "render": {"mode": "solid_edges", "background": "dark", "aa": 4},
            "shortcuts": {"focus": "F", "undo": "Ctrl+Z", "redo": "Ctrl+Y", "delete": "Del"},
            "boolean": {"tolerance": 1e-6, "keep_originals": False, "clean": True, "triangulate": True, "normals": True},
        }

    @staticmethod
    def _merge_dict(base: dict, patch: dict) -> dict:
        out = dict(base)
        for key, value in dict(patch or {}).items():
            if isinstance(value, dict) and isinstance(out.get(key), dict):
                out[key] = MechanicsPage._merge_dict(out[key], value)
            else:
                out[key] = value
        return out

    def _prefs_path(self) -> Path:
        return self._config_dir / "preferences.json"

    def _layout_path(self, name: str) -> Path:
        token = str(name or "layout_default").strip().lower().replace(" ", "_")
        if not token.startswith("layout_"):
            token = f"layout_{token}"
        safe = "".join(ch for ch in token if ch.isalnum() or ch in {"_", "-"})
        return self._layouts_dir / f"{safe}.json"

    def _save_preferences(self):
        try:
            self._config_dir.mkdir(parents=True, exist_ok=True)
            with open(self._prefs_path(), "w", encoding="utf-8", newline="\n") as f:
                json.dump(self._prefs, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self._log(f"Failed to save preferences: {e}")

    def _load_preferences(self):
        self._prefs = self._default_preferences()
        try:
            path = self._prefs_path()
            if path.is_file():
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    raw = json.load(f)
                if isinstance(raw, dict):
                    self._prefs = self._merge_dict(self._prefs, raw)
        except Exception as e:
            self._log(f"Failed to load preferences, using defaults: {e}")
        snap_cfg = self._prefs.get("snap", {})
        self._snap_enabled = bool(snap_cfg.get("grid", True))
        self._snap_step = float(snap_cfg.get("step_mm", 5.0))
        self._angle_snap_step = float(snap_cfg.get("step_deg", 5.0))
        bool_cfg = self._prefs.get("boolean", {})
        self._boolean_tolerance = float(bool_cfg.get("tolerance", 1e-6))
        self._boolean_keep_originals = bool(bool_cfg.get("keep_originals", False))

    def _apply_preferences(self):
        unit = str(self._prefs.get("units", "mm"))
        profile = str(self._prefs.get("profile", "CAD Classic"))
        render_cfg = self._prefs.get("render", {})
        snap_cfg = self._prefs.get("snap", {})
        bool_cfg = self._prefs.get("boolean", {})

        for combo, value in ((self.units_combo, unit), (self.profile_combo, profile)):
            combo.blockSignals(True)
            idx = combo.findText(value)
            if idx >= 0:
                combo.setCurrentIndex(idx)
            combo.blockSignals(False)

        self.chk_snap.blockSignals(True)
        self.chk_snap.setChecked(bool(self._snap_enabled))
        self.chk_snap.blockSignals(False)

        self._snap_step = float(snap_cfg.get("step_mm", self._snap_step))
        self._angle_snap_step = float(snap_cfg.get("step_deg", self._angle_snap_step))
        self._boolean_tolerance = float(bool_cfg.get("tolerance", self._boolean_tolerance))
        self._boolean_keep_originals = bool(bool_cfg.get("keep_originals", self._boolean_keep_originals))

        if hasattr(self, "tr_step_edit"):
            self.tr_step_edit.setText(f"{self._snap_step:.6g}")
        if hasattr(self, "tr_angle_step_edit"):
            self.tr_angle_step_edit.setText(f"{self._angle_snap_step:.6g}")
        if hasattr(self, "bool_tol_edit"):
            self.bool_tol_edit.setText(f"{self._boolean_tolerance:.9g}")
        if hasattr(self, "bool_keep_orig_chk"):
            self.bool_keep_orig_chk.setChecked(bool(self._boolean_keep_originals))
        if hasattr(self, "bool_clean_chk"):
            self.bool_clean_chk.setChecked(bool(bool_cfg.get("clean", True)))
        if hasattr(self, "bool_triang_chk"):
            self.bool_triang_chk.setChecked(bool(bool_cfg.get("triangulate", True)))
        if hasattr(self, "bool_normals_chk"):
            self.bool_normals_chk.setChecked(bool(bool_cfg.get("normals", True)))

        self.action_set_background(str(render_cfg.get("background", "dark")))
        self.action_set_render_mode(str(render_cfg.get("mode", "solid_edges")))
        self._set_hint("Click to select. Shift adds selection. RMB opens full context menu.")

    def _profile_defaults(self, profile: str) -> dict:
        token = str(profile or "CAD Classic").strip().lower()
        if token == "minimal":
            return {"layout": "layout_modeling", "render_mode": "solid", "background": "dark", "snap": False}
        if token == "analysis":
            return {"layout": "layout_analysis", "render_mode": "solid_edges", "background": "light", "snap": True}
        return {"layout": "layout_default", "render_mode": "solid_edges", "background": "dark", "snap": True}

    def _on_profile_changed(self, profile: str):
        cfg = self._profile_defaults(profile)
        self._prefs["profile"] = str(profile)
        self._prefs.setdefault("render", {})
        self._prefs["render"]["mode"] = str(cfg["render_mode"])
        self._prefs["render"]["background"] = str(cfg["background"])
        self._set_snap_enabled(bool(cfg["snap"]), persist=False)
        self.action_set_render_mode(str(cfg["render_mode"]))
        self.action_set_background(str(cfg["background"]))
        self.load_layout(str(cfg["layout"]), silent=True)
        self._save_preferences()
        self._set_hint("Profile loaded. Use RMB in viewport/tree/properties for full operation set.")

    def _on_units_changed(self, unit: str):
        token = str(unit or "mm").strip()
        self._prefs["units"] = token
        self._save_preferences()
        self._set_hint(f"Units set to {token}.")

    def _set_snap_enabled(self, enabled: bool, persist: bool = True):
        self._snap_enabled = bool(enabled)
        if bool(self.chk_snap.isChecked()) != self._snap_enabled:
            self.chk_snap.blockSignals(True)
            self.chk_snap.setChecked(self._snap_enabled)
            self.chk_snap.blockSignals(False)
        self._prefs.setdefault("snap", {})
        self._prefs["snap"]["grid"] = bool(self._snap_enabled)
        if persist:
            self._save_preferences()
        self._set_hint(f"Snap to grid {'enabled' if self._snap_enabled else 'disabled'}.")

    def _save_layout_from_combo(self):
        self.save_layout(self.layout_combo.currentText())

    def _load_layout_from_combo(self):
        self.load_layout(self.layout_combo.currentText())

    def save_layout(self, name: str):
        try:
            self._layouts_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "name": str(name),
                "main_split_sizes": [int(x) for x in self.main_split.sizes()],
                "outer_split_sizes": [int(x) for x in self.outer_split.sizes()],
                "main_split_state": bytes(self.main_split.saveState().toBase64()).decode("ascii"),
                "outer_split_state": bytes(self.outer_split.saveState().toBase64()).decode("ascii"),
                "ops_tab_index": int(self.ops_tabs.currentIndex()),
                "scene_search": str(self.scene_tree.search_edit.text() or ""),
                "profile": str(self.profile_combo.currentText() or ""),
            }
            with open(self._layout_path(name), "w", encoding="utf-8", newline="\n") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            self._log(f"Layout saved: {self._layout_path(name)}")
        except Exception as e:
            QMessageBox.warning(self, "Save layout", str(e))

    def load_layout(self, name: str, silent: bool = False):
        path = self._layout_path(name)
        if not path.is_file():
            if str(name).strip().lower() == "layout_default":
                self.reset_layout_default(save=False)
                return
            if not silent:
                QMessageBox.warning(self, "Load layout", f"Layout not found: {path}")
            return
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                payload = json.load(f)
            if not isinstance(payload, dict):
                raise RuntimeError("Invalid layout format.")
            main_state = str(payload.get("main_split_state", "") or "")
            outer_state = str(payload.get("outer_split_state", "") or "")
            if main_state:
                qba = QByteArray.fromBase64(main_state.encode("ascii"))
                if not qba.isEmpty():
                    self.main_split.restoreState(qba)
            if outer_state:
                qba = QByteArray.fromBase64(outer_state.encode("ascii"))
                if not qba.isEmpty():
                    self.outer_split.restoreState(qba)
            main_sizes = payload.get("main_split_sizes", [])
            outer_sizes = payload.get("outer_split_sizes", [])
            if isinstance(main_sizes, list) and len(main_sizes) == 3:
                self.main_split.setSizes([int(x) for x in main_sizes])
            if isinstance(outer_sizes, list) and len(outer_sizes) == 2:
                self.outer_split.setSizes([int(x) for x in outer_sizes])
            idx = int(payload.get("ops_tab_index", 0))
            if 0 <= idx < self.ops_tabs.count():
                self.ops_tabs.setCurrentIndex(idx)
            self.scene_tree.search_edit.setText(str(payload.get("scene_search", "") or ""))
            if not silent:
                self._log(f"Layout loaded: {path}")
        except Exception as e:
            if not silent:
                QMessageBox.warning(self, "Load layout", str(e))

    def reset_layout_default(self, save: bool = True):
        self.main_split.setSizes([320, 980, 420])
        self.outer_split.setSizes([720, 260])
        self.ops_tabs.setCurrentIndex(0)
        self.scene_tree.search_edit.setText("")
        if save:
            self.save_layout("layout_default")
        self._set_hint("Default layout restored.")

    def _parse_float(self, widget: QLineEdit, default: float) -> float:
        try:
            return float(str(widget.text()).strip().replace(",", "."))
        except Exception:
            return float(default)

    def _parse_int(self, widget: QLineEdit, default: int) -> int:
        try:
            return int(round(float(str(widget.text()).strip().replace(",", "."))))
        except Exception:
            return int(default)

    def _build_ops_tabs(self) -> QTabWidget:
        tabs = QTabWidget(self)
        tabs.addTab(self._build_tab_scene(), "Scene")
        tabs.addTab(self._build_tab_create(), "Create")
        tabs.addTab(self._build_tab_transform(), "Transform")
        tabs.addTab(self._build_tab_boolean(), "Boolean")
        tabs.addTab(self._build_tab_analyze(), "Analyze")
        tabs.addTab(self._build_tab_markers(), "Markers")
        tabs.addTab(self._build_tab_export(), "Export")
        return tabs

    def _build_tab_scene(self) -> QWidget:
        w = QWidget(self)
        g = QGridLayout(w)
        g.setContentsMargins(6, 4, 6, 4)
        g.setHorizontalSpacing(6)
        g.setVerticalSpacing(6)
        g.addWidget(QLabel("Scene organization and quick selection controls."), 0, 0, 1, 8)
        actions = [
            ("Create Group", self.action_group_selected),
            ("Ungroup", self.action_ungroup_selected),
            ("Rename", self.action_rename_selected),
            ("Duplicate", lambda: self.action_duplicate(self._pick_selected())),
            ("Delete", self.action_delete_selected),
            ("Hide", self.action_hide_selected),
            ("Show", self.action_show_selected),
            ("Isolate", self.action_show_only_selected),
            ("Show All", self.action_show_all),
            ("Select All", self.action_select_all),
            ("Select None", self.action_select_none),
            ("Invert", self.action_select_invert),
            ("Focus", lambda: self.action_focus_object(self.engine.selection[0]) if self.engine.selection else None),
        ]
        for idx, (text, slot) in enumerate(actions):
            r = 1 + (idx // 7)
            c = idx % 7
            btn = QPushButton(text, w)
            btn.clicked.connect(slot)
            g.addWidget(btn, r, c)
        g.addWidget(QLabel("Use search in Scene Tree to filter by name/source/group."), 3, 0, 1, 8)
        return w

    def _build_tab_create(self) -> QWidget:
        w = QWidget(self)
        g = QGridLayout(w)
        g.setContentsMargins(6, 4, 6, 4)
        g.setHorizontalSpacing(6)
        g.setVerticalSpacing(6)

        self.create_kind_combo = QComboBox(w)
        self.create_kind_combo.addItems(["box", "cylinder", "sphere", "cone", "plane"])
        self.create_name_edit = QLineEdit("primitive_1", w)
        self.create_center_combo = QComboBox(w)
        self.create_center_combo.addItems(["center on origin", "center on selection"])
        self.create_w_edit = QLineEdit("1.0", w)
        self.create_d_edit = QLineEdit("1.0", w)
        self.create_h_edit = QLineEdit("1.0", w)
        self.create_r_edit = QLineEdit("0.5", w)
        self.create_seg_edit = QLineEdit("32", w)

        g.addWidget(QLabel("Primitive"), 0, 0)
        g.addWidget(self.create_kind_combo, 0, 1)
        g.addWidget(QLabel("Name"), 0, 2)
        g.addWidget(self.create_name_edit, 0, 3)
        g.addWidget(QLabel("Center"), 0, 4)
        g.addWidget(self.create_center_combo, 0, 5)

        g.addWidget(QLabel("W"), 1, 0)
        g.addWidget(self.create_w_edit, 1, 1)
        g.addWidget(QLabel("D"), 1, 2)
        g.addWidget(self.create_d_edit, 1, 3)
        g.addWidget(QLabel("H"), 1, 4)
        g.addWidget(self.create_h_edit, 1, 5)
        g.addWidget(QLabel("R"), 2, 0)
        g.addWidget(self.create_r_edit, 2, 1)
        g.addWidget(QLabel("Segments"), 2, 2)
        g.addWidget(self.create_seg_edit, 2, 3)

        btn_create = QPushButton("Create", w)
        btn_create_dup = QPushButton("Create && Duplicate", w)
        btn_cancel_preview = QPushButton("Cancel Preview", w)
        btn_create.clicked.connect(self._create_from_tab)
        btn_create_dup.clicked.connect(lambda: self._create_from_tab(duplicate=True))
        btn_cancel_preview.clicked.connect(lambda: self._set_hint("Preview canceled."))
        g.addWidget(btn_create, 3, 0, 1, 2)
        g.addWidget(btn_create_dup, 3, 2, 1, 2)
        g.addWidget(btn_cancel_preview, 3, 4, 1, 2)
        return w

    def _build_tab_transform(self) -> QWidget:
        w = QWidget(self)
        g = QGridLayout(w)
        g.setContentsMargins(6, 4, 6, 4)
        g.setHorizontalSpacing(6)
        g.setVerticalSpacing(6)

        self.tr_mode_combo = QComboBox(w)
        self.tr_mode_combo.addItems(["Select", "Move", "Rotate", "Scale", "Measure"])
        self.tr_mode_combo.currentTextChanged.connect(self.action_set_tool_mode)
        g.addWidget(QLabel("Tool Mode"), 0, 0)
        g.addWidget(self.tr_mode_combo, 0, 1)

        self.tr_tx_edit = QLineEdit("0.0", w)
        self.tr_ty_edit = QLineEdit("0.0", w)
        self.tr_tz_edit = QLineEdit("0.0", w)
        self.tr_rx_edit = QLineEdit("0.0", w)
        self.tr_ry_edit = QLineEdit("0.0", w)
        self.tr_rz_edit = QLineEdit("0.0", w)
        self.tr_scale_edit = QLineEdit("1.0", w)

        g.addWidget(QLabel("Tx"), 1, 0)
        g.addWidget(self.tr_tx_edit, 1, 1)
        g.addWidget(QLabel("Ty"), 1, 2)
        g.addWidget(self.tr_ty_edit, 1, 3)
        g.addWidget(QLabel("Tz"), 1, 4)
        g.addWidget(self.tr_tz_edit, 1, 5)
        g.addWidget(QLabel("Rx deg"), 2, 0)
        g.addWidget(self.tr_rx_edit, 2, 1)
        g.addWidget(QLabel("Ry deg"), 2, 2)
        g.addWidget(self.tr_ry_edit, 2, 3)
        g.addWidget(QLabel("Rz deg"), 2, 4)
        g.addWidget(self.tr_rz_edit, 2, 5)
        g.addWidget(QLabel("Scale"), 3, 0)
        g.addWidget(self.tr_scale_edit, 3, 1)

        self.tr_snap_chk = QCheckBox("Snap to grid", w)
        self.tr_snap_chk.setChecked(bool(self._snap_enabled))
        self.tr_snap_chk.toggled.connect(self._set_snap_enabled)
        g.addWidget(self.tr_snap_chk, 3, 2)
        g.addWidget(QLabel("Grid step"), 3, 3)
        self.tr_step_edit = QLineEdit(f"{self._snap_step:.6g}", w)
        g.addWidget(self.tr_step_edit, 3, 4)
        g.addWidget(QLabel("Angle step"), 3, 5)
        self.tr_angle_step_edit = QLineEdit(f"{self._angle_snap_step:.6g}", w)
        g.addWidget(self.tr_angle_step_edit, 3, 6)

        btn_apply = QPushButton("Apply", w)
        btn_reset = QPushButton("Reset", w)
        btn_apply.clicked.connect(self._apply_transform_from_tab)
        btn_reset.clicked.connect(self._reset_transform_tab_fields)
        g.addWidget(btn_apply, 4, 0, 1, 2)
        g.addWidget(btn_reset, 4, 2, 1, 2)
        return w

    def _build_tab_boolean(self) -> QWidget:
        w = QWidget(self)
        g = QGridLayout(w)
        g.setContentsMargins(6, 4, 6, 4)
        g.setHorizontalSpacing(6)
        g.setVerticalSpacing(6)

        self.bool_a_label = QLabel("A: -", w)
        self.bool_b_label = QLabel("B: -", w)
        btn_set_a = QPushButton("Use selected as A", w)
        btn_set_b = QPushButton("Use selected as B", w)
        btn_swap = QPushButton("Swap A/B", w)
        btn_set_a.clicked.connect(self._set_boolean_a_from_selection)
        btn_set_b.clicked.connect(self._set_boolean_b_from_selection)
        btn_swap.clicked.connect(self._swap_boolean_slots)
        g.addWidget(self.bool_a_label, 0, 0, 1, 2)
        g.addWidget(self.bool_b_label, 0, 2, 1, 2)
        g.addWidget(btn_set_a, 1, 0)
        g.addWidget(btn_set_b, 1, 1)
        g.addWidget(btn_swap, 1, 2)

        self.bool_clean_chk = QCheckBox("Clean before boolean", w)
        self.bool_triang_chk = QCheckBox("Triangulate", w)
        self.bool_normals_chk = QCheckBox("Normals consistent", w)
        self.bool_keep_orig_chk = QCheckBox("Keep originals", w)
        self.bool_clean_chk.setChecked(True)
        self.bool_triang_chk.setChecked(True)
        self.bool_normals_chk.setChecked(True)
        self.bool_keep_orig_chk.setChecked(bool(self._boolean_keep_originals))
        self.bool_keep_orig_chk.toggled.connect(self._on_boolean_keep_changed)
        g.addWidget(self.bool_clean_chk, 2, 0)
        g.addWidget(self.bool_triang_chk, 2, 1)
        g.addWidget(self.bool_normals_chk, 2, 2)
        g.addWidget(self.bool_keep_orig_chk, 2, 3)

        g.addWidget(QLabel("Tolerance"), 3, 0)
        self.bool_tol_edit = QLineEdit(f"{self._boolean_tolerance:.9g}", w)
        g.addWidget(self.bool_tol_edit, 3, 1)
        btn_diagnose = QPushButton("Diagnose", w)
        btn_diagnose.clicked.connect(self.action_boolean_diagnose)
        g.addWidget(btn_diagnose, 3, 2)

        btn_union = QPushButton("Union", w)
        btn_sub = QPushButton("Subtract A-B", w)
        btn_inter = QPushButton("Intersect", w)
        btn_union.clicked.connect(lambda: self._apply_boolean_from_tab("union"))
        btn_sub.clicked.connect(lambda: self._apply_boolean_from_tab("subtract"))
        btn_inter.clicked.connect(lambda: self._apply_boolean_from_tab("intersect"))
        g.addWidget(btn_union, 4, 0)
        g.addWidget(btn_sub, 4, 1)
        g.addWidget(btn_inter, 4, 2)
        self._refresh_boolean_selection_ui()
        return w

    def _build_tab_analyze(self) -> QWidget:
        w = QWidget(self)
        g = QGridLayout(w)
        g.setContentsMargins(6, 4, 6, 4)
        g.setHorizontalSpacing(6)
        g.setVerticalSpacing(6)

        g.addWidget(QLabel("Object metrics"), 0, 0, 1, 4)
        btn_bbox = QPushButton("Bounding Box", w)
        btn_volume = QPushButton("Volume", w)
        btn_area = QPushButton("Area", w)
        btn_centroid = QPushButton("Centroid", w)
        btn_bbox.clicked.connect(lambda: self.action_measure_bbox(self._pick_selected()))
        btn_volume.clicked.connect(lambda: self.action_measure_volume(self._pick_selected()))
        btn_area.clicked.connect(lambda: self.action_measure_area(self._pick_selected()))
        btn_centroid.clicked.connect(lambda: self.action_measure_centroid(self._pick_selected()))
        g.addWidget(btn_bbox, 1, 0)
        g.addWidget(btn_volume, 1, 1)
        g.addWidget(btn_area, 1, 2)
        g.addWidget(btn_centroid, 1, 3)

        g.addWidget(QLabel("Measure tools"), 2, 0, 1, 4)
        btn_dist = QPushButton("Distance (2 points)", w)
        btn_angle = QPushButton("Angle (3 points)", w)
        btn_clear = QPushButton("Clear current", w)
        btn_save = QPushButton("Save measure as marker", w)
        btn_dist.clicked.connect(lambda: self.action_enter_measure_mode("distance"))
        btn_angle.clicked.connect(lambda: self.action_enter_measure_mode("angle"))
        btn_clear.clicked.connect(self.action_clear_current_measure)
        btn_save.clicked.connect(self.action_save_measure_as_marker)
        g.addWidget(btn_dist, 3, 0)
        g.addWidget(btn_angle, 3, 1)
        g.addWidget(btn_clear, 3, 2)
        g.addWidget(btn_save, 3, 3)

        g.addWidget(QLabel("Sections / clipping"), 4, 0, 1, 4)
        btn_xy = QPushButton("Clip XY", w)
        btn_xz = QPushButton("Clip XZ", w)
        btn_yz = QPushButton("Clip YZ", w)
        btn_clr = QPushButton("Clear clips", w)
        btn_xy.clicked.connect(lambda: self.action_add_clipping_plane("xy"))
        btn_xz.clicked.connect(lambda: self.action_add_clipping_plane("xz"))
        btn_yz.clicked.connect(lambda: self.action_add_clipping_plane("yz"))
        btn_clr.clicked.connect(self.action_clear_clipping)
        g.addWidget(btn_xy, 5, 0)
        g.addWidget(btn_xz, 5, 1)
        g.addWidget(btn_yz, 5, 2)
        g.addWidget(btn_clr, 5, 3)
        return w

    def _build_tab_markers(self) -> QWidget:
        w = QWidget(self)
        g = QGridLayout(w)
        g.setContentsMargins(6, 4, 6, 4)
        g.setHorizontalSpacing(6)
        g.setVerticalSpacing(6)

        btn_add = QPushButton("Add marker (cursor/origin)", w)
        btn_cent = QPushButton("Add marker at centroid", w)
        btn_math = QPushButton("Add math marker", w)
        btn_add.clicked.connect(lambda: self.action_add_marker_at_cursor(self._last_cursor_ctx or ContextInfo(widget="viewport")))
        btn_cent.clicked.connect(self._add_marker_centroid_from_selection)
        btn_math.clicked.connect(lambda: self.action_add_custom_math_marker(self._last_cursor_ctx or ContextInfo(widget="viewport")))
        g.addWidget(btn_add, 0, 0)
        g.addWidget(btn_cent, 0, 1)
        g.addWidget(btn_math, 0, 2)

        g.addWidget(QLabel("Marker"), 1, 0)
        self.marker_combo = QComboBox(w)
        self.marker_combo.currentTextChanged.connect(self._on_marker_combo_changed)
        g.addWidget(self.marker_combo, 1, 1, 1, 2)
        g.addWidget(QLabel("Label"), 2, 0)
        self.marker_label_edit = QLineEdit("", w)
        g.addWidget(self.marker_label_edit, 2, 1, 1, 2)
        g.addWidget(QLabel("Color"), 3, 0)
        self.marker_color_edit = QLineEdit("#ffd166", w)
        g.addWidget(self.marker_color_edit, 3, 1)
        g.addWidget(QLabel("Size"), 3, 2)
        self.marker_size_edit = QLineEdit("10.0", w)
        g.addWidget(self.marker_size_edit, 3, 3)
        g.addWidget(QLabel("Expr (key=expr)"), 4, 0)
        self.marker_expr_edit = QLineEdit("", w)
        g.addWidget(self.marker_expr_edit, 4, 1, 1, 3)

        btn_apply = QPushButton("Apply marker changes", w)
        btn_copy = QPushButton("Copy marker JSON", w)
        btn_del = QPushButton("Delete marker", w)
        btn_save = QPushButton("Export markers JSON", w)
        btn_load = QPushButton("Import markers JSON", w)
        btn_apply.clicked.connect(self._apply_marker_changes_from_tab)
        btn_copy.clicked.connect(self._copy_marker_json_from_tab)
        btn_del.clicked.connect(self._delete_marker_from_tab)
        btn_save.clicked.connect(self._save_markers)
        btn_load.clicked.connect(self._load_markers)
        g.addWidget(btn_apply, 5, 0)
        g.addWidget(btn_copy, 5, 1)
        g.addWidget(btn_del, 5, 2)
        g.addWidget(btn_save, 6, 0)
        g.addWidget(btn_load, 6, 1)
        self._refresh_marker_controls()
        return w

    def _build_tab_export(self) -> QWidget:
        w = QWidget(self)
        g = QGridLayout(w)
        g.setContentsMargins(6, 4, 6, 4)
        g.setHorizontalSpacing(6)
        g.setVerticalSpacing(6)
        g.addWidget(QLabel("Geometry export"), 0, 0, 1, 4)
        self.export_fmt_combo = QComboBox(w)
        self.export_fmt_combo.addItems(["STL", "OBJ"])
        g.addWidget(QLabel("Format"), 1, 0)
        g.addWidget(self.export_fmt_combo, 1, 1)
        btn_sel = QPushButton("Export selection", w)
        btn_scene = QPushButton("Export scene", w)
        btn_sel.clicked.connect(self._export_selected_from_tab)
        btn_scene.clicked.connect(self._export_scene_from_tab)
        g.addWidget(btn_sel, 1, 2)
        g.addWidget(btn_scene, 1, 3)
        g.addWidget(QLabel("Viewport capture"), 2, 0, 1, 4)
        btn_snap = QPushButton("Screenshot PNG", w)
        btn_snap.clicked.connect(self.action_screenshot)
        g.addWidget(btn_snap, 3, 0)
        btn_bg_dark = QPushButton("Background Dark", w)
        btn_bg_light = QPushButton("Background Light", w)
        btn_bg_dark.clicked.connect(lambda: self.action_set_background("dark"))
        btn_bg_light.clicked.connect(lambda: self.action_set_background("light"))
        g.addWidget(btn_bg_dark, 3, 1)
        g.addWidget(btn_bg_light, 3, 2)
        return w

    def _create_from_tab(self, duplicate: bool = False):
        token = str(self.create_kind_combo.currentText() or "box").strip().lower()
        name = str(self.create_name_edit.text() or token).strip() or token
        center = np.zeros(3, dtype=float)
        if str(self.create_center_combo.currentText()).lower().endswith("selection"):
            ids = self._pick_selected()
            if ids:
                pts = [np.mean(self.engine.objects[oid].mesh.vertices, axis=0) for oid in ids if oid in self.engine.objects]
                if pts:
                    center = np.mean(np.asarray(pts, dtype=float), axis=0)
        params = {"center": tuple(float(x) for x in center.tolist())}
        if token in {"box", "plane"}:
            params["width"] = self._parse_float(self.create_w_edit, 1.0)
            params["depth"] = self._parse_float(self.create_d_edit, 1.0)
            if token == "box":
                params["height"] = self._parse_float(self.create_h_edit, 1.0)
        elif token in {"cylinder", "cone"}:
            params["radius"] = self._parse_float(self.create_r_edit, 0.5)
            params["height"] = self._parse_float(self.create_h_edit, 1.0)
            params["segments"] = max(8, self._parse_int(self.create_seg_edit, 32))
        elif token == "sphere":
            params["radius"] = self._parse_float(self.create_r_edit, 0.5)
            seg = max(8, self._parse_int(self.create_seg_edit, 24))
            params["theta_res"] = seg
            params["phi_res"] = seg
        new_id = self.engine.create_primitive(token, params=params, name=name)
        self.engine.select([new_id], mode="replace")
        if duplicate:
            self.engine.duplicate_objects([new_id])
        self._set_hint("Primitive created. Use Transform tab or RMB for edits.")

    def _apply_transform_from_tab(self):
        ids = self._pick_selected()
        if not ids:
            QMessageBox.warning(self, "Transform", "Select at least one object.")
            return
        self._snap_step = max(1e-9, self._parse_float(self.tr_step_edit, self._snap_step))
        self._angle_snap_step = max(1e-9, self._parse_float(self.tr_angle_step_edit, self._angle_snap_step))
        self._prefs.setdefault("snap", {})
        self._prefs["snap"]["step_mm"] = float(self._snap_step)
        self._prefs["snap"]["step_deg"] = float(self._angle_snap_step)
        self._save_preferences()

        tx = self._parse_float(self.tr_tx_edit, 0.0)
        ty = self._parse_float(self.tr_ty_edit, 0.0)
        tz = self._parse_float(self.tr_tz_edit, 0.0)
        rx = self._parse_float(self.tr_rx_edit, 0.0)
        ry = self._parse_float(self.tr_ry_edit, 0.0)
        rz = self._parse_float(self.tr_rz_edit, 0.0)
        sc = max(1e-9, self._parse_float(self.tr_scale_edit, 1.0))
        if self._snap_enabled:
            tx = round(tx / self._snap_step) * self._snap_step
            ty = round(ty / self._snap_step) * self._snap_step
            tz = round(tz / self._snap_step) * self._snap_step
            rx = round(rx / self._angle_snap_step) * self._angle_snap_step
            ry = round(ry / self._angle_snap_step) * self._angle_snap_step
            rz = round(rz / self._angle_snap_step) * self._angle_snap_step
        self.engine.transform_objects(ids, tx=tx, ty=ty, tz=tz, rx_deg=rx, ry_deg=ry, rz_deg=rz, scale=sc)
        self._set_hint("Transform applied to selected objects.")

    def _reset_transform_tab_fields(self):
        for edit in (self.tr_tx_edit, self.tr_ty_edit, self.tr_tz_edit, self.tr_rx_edit, self.tr_ry_edit, self.tr_rz_edit):
            edit.setText("0.0")
        self.tr_scale_edit.setText("1.0")

    def _refresh_boolean_selection_ui(self):
        a = self._boolean_primary_id if self._boolean_primary_id in self.engine.objects else ""
        bs = [oid for oid in self._boolean_tool_ids if oid in self.engine.objects]
        self._boolean_primary_id = a
        self._boolean_tool_ids = bs
        if hasattr(self, "bool_a_label"):
            self.bool_a_label.setText(f"A: {self.engine.objects[a].name if a else '-'}")
        if hasattr(self, "bool_b_label"):
            labels = [self.engine.objects[oid].name for oid in bs]
            self.bool_b_label.setText("B: " + (", ".join(labels) if labels else "-"))

    def _set_boolean_a_from_selection(self):
        ids = self._pick_selected()
        if not ids:
            return
        self._boolean_primary_id = ids[0]
        self._refresh_boolean_selection_ui()

    def _set_boolean_b_from_selection(self):
        ids = self._pick_selected()
        if not ids:
            return
        if self._boolean_primary_id and self._boolean_primary_id in ids:
            ids = [x for x in ids if x != self._boolean_primary_id]
        self._boolean_tool_ids = list(ids)
        self._refresh_boolean_selection_ui()

    def _swap_boolean_slots(self):
        if not self._boolean_tool_ids:
            return
        old_a = self._boolean_primary_id
        self._boolean_primary_id = self._boolean_tool_ids[0]
        rest = self._boolean_tool_ids[1:]
        if old_a:
            rest.append(old_a)
        self._boolean_tool_ids = rest
        self._refresh_boolean_selection_ui()

    def _on_boolean_keep_changed(self, value: bool):
        self._boolean_keep_originals = bool(value)
        self._prefs.setdefault("boolean", {})
        self._prefs["boolean"]["keep_originals"] = bool(value)
        self._save_preferences()

    def _boolean_target_ids(self) -> List[str]:
        out = []
        if self._boolean_primary_id and self._boolean_primary_id in self.engine.objects:
            out.append(self._boolean_primary_id)
        out.extend([oid for oid in self._boolean_tool_ids if oid in self.engine.objects and oid not in out])
        if len(out) < 2:
            out = self._pick_selected()
        return out

    def _apply_boolean_from_tab(self, mode: str):
        ids = self._boolean_target_ids()
        if len(ids) < 2:
            QMessageBox.warning(self, "Boolean", "Select at least 2 objects (A and B).")
            return
        tol = max(1e-12, self._parse_float(self.bool_tol_edit, self._boolean_tolerance))
        self._boolean_tolerance = tol
        self._prefs.setdefault("boolean", {})
        self._prefs["boolean"]["tolerance"] = float(tol)
        self._prefs["boolean"]["clean"] = bool(self.bool_clean_chk.isChecked())
        self._prefs["boolean"]["triangulate"] = bool(self.bool_triang_chk.isChecked())
        self._prefs["boolean"]["normals"] = bool(self.bool_normals_chk.isChecked())
        self._prefs["boolean"]["keep_originals"] = bool(self.bool_keep_orig_chk.isChecked())
        self._save_preferences()

        if self.bool_keep_orig_chk.isChecked():
            ids = self.engine.duplicate_objects(ids)
        self.engine.select(ids, mode="replace")
        try:
            token = str(mode).lower()
            if token == "union":
                self.engine.boolean_union(ids, tolerance=tol)
            elif token == "subtract":
                self.engine.boolean_subtract(ids[0], ids[1:], tolerance=tol)
            else:
                self.engine.boolean_intersect(ids, tolerance=tol)
            self._set_hint(
                f"Boolean {token} completed.",
                [
                    "Boolean wizard",
                    "1) Select A and B or use selected objects.",
                    f"2) Tolerance = {tol:.3g}",
                    "3) Use Diagnose if result is unexpected.",
                ],
            )
        except Exception as e:
            QMessageBox.warning(self, "Boolean failed", str(e))

    def action_boolean_diagnose(self):
        ids = self._boolean_target_ids() or self._pick_selected()
        if not ids:
            QMessageBox.information(self, "Diagnose", "No object selected.")
            return
        lines = []
        for oid in ids:
            if oid not in self.engine.objects:
                continue
            info = self.engine.object_validation(oid)
            tag = self.engine.objects[oid].name
            msg = ", ".join([str(x) for x in info.get("messages", [])]) or "ok"
            lines.append(
                f"{tag}: ok={info.get('ok')} faces={info.get('faces')} deg={info.get('degenerate_faces')} "
                f"manifold={info.get('manifold')} :: {msg}"
            )
        text = "\n".join(lines) if lines else "No diagnostics available."
        self._log(text)
        self._set_hint("Boolean diagnostics complete.", ["Boolean diagnostics"] + lines[:6])

    def _add_marker_centroid_from_selection(self):
        ids = self._pick_selected()
        if not ids:
            QMessageBox.information(self, "Markers", "Select one object to place centroid marker.")
            return
        self.action_marker_centroid(ids[0])
        self._refresh_marker_controls()

    def _refresh_marker_controls(self):
        if not hasattr(self, "marker_combo"):
            return
        current = str(self.marker_combo.currentData() or "")
        self.marker_combo.blockSignals(True)
        self.marker_combo.clear()
        for mid, mk in self.engine.markers.items():
            self.marker_combo.addItem(f"{mk.name} ({mid[:8]})", mid)
        self.marker_combo.blockSignals(False)
        if current:
            for idx in range(self.marker_combo.count()):
                if str(self.marker_combo.itemData(idx)) == current:
                    self.marker_combo.setCurrentIndex(idx)
                    break
        self._on_marker_combo_changed(self.marker_combo.currentText())

    def _current_marker_id(self) -> str:
        if not hasattr(self, "marker_combo"):
            return ""
        return str(self.marker_combo.currentData() or "")

    def _on_marker_combo_changed(self, _text: str):
        mid = self._current_marker_id()
        if not mid or mid not in self.engine.markers:
            self.marker_label_edit.setText("")
            self.marker_color_edit.setText("#ffd166")
            self.marker_size_edit.setText("10.0")
            self.marker_expr_edit.setText("")
            return
        mk = self.engine.markers[mid]
        self.marker_label_edit.setText(str(mk.name))
        self.marker_color_edit.setText(str(mk.style.get("color", "#ffd166")))
        self.marker_size_edit.setText(f"{float(mk.style.get('size', 10.0)):.6g}")
        if mk.expressions:
            key = sorted(mk.expressions.keys())[0]
            self.marker_expr_edit.setText(f"{key}={mk.expressions[key]}")
        else:
            self.marker_expr_edit.setText("")

    def _apply_marker_changes_from_tab(self):
        mid = self._current_marker_id()
        if not mid or mid not in self.engine.markers:
            return
        style = {
            "color": str(self.marker_color_edit.text() or "#ffd166").strip() or "#ffd166",
            "size": max(1.0, self._parse_float(self.marker_size_edit, 10.0)),
        }
        expr_text = str(self.marker_expr_edit.text() or "").strip()
        exprs = None
        if expr_text:
            if "=" not in expr_text:
                QMessageBox.warning(self, "Marker expression", "Use format key=expression.")
                return
            k, v = expr_text.split("=", 1)
            exprs = {str(k).strip() or "value": str(v).strip()}
        self.engine.update_marker(mid, name=str(self.marker_label_edit.text() or "Marker"), style=style, expressions=exprs)
        self._refresh_marker_controls()
        self._set_hint("Marker updated.")

    def _copy_marker_json_from_tab(self):
        mid = self._current_marker_id()
        if not mid or mid not in self.engine.markers:
            return
        QApplication.clipboard().setText(json.dumps(self.engine.markers[mid].to_dict(), ensure_ascii=False, indent=2))

    def _delete_marker_from_tab(self):
        mid = self._current_marker_id()
        if not mid or mid not in self.engine.markers:
            return
        self.engine.delete_marker(mid)
        self._refresh_marker_controls()

    def _export_selected_from_tab(self):
        fmt = str(self.export_fmt_combo.currentText() or "STL").strip().lower()
        self.action_export_selected("stl" if fmt == "stl" else "obj")

    def _export_scene_from_tab(self):
        ids = list(self.engine.objects.keys())
        if not ids:
            QMessageBox.information(self, "Export scene", "Scene is empty.")
            return
        previous = list(self.engine.selection)
        self.engine.select(ids, mode="replace")
        try:
            self._export_selected_from_tab()
        finally:
            self.engine.select(previous, mode="replace")

    def _on_engine_event(self, event: str, _payload: dict):
        if event in {"scene_changed", "selection_changed", "markers_changed", "measurements_changed"}:
            self.refresh_all()
            self._update_undo_redo_buttons()

    def _update_undo_redo_buttons(self):
        self.btn_undo.setEnabled(self.engine.command_stack.can_undo)
        self.btn_redo.setEnabled(self.engine.command_stack.can_redo)

    def refresh_all(self):
        self.scene_tree.refresh(self.engine.objects, self.engine.selection)
        self.viewport.refresh_scene(self.engine.objects, self.engine.selection, self.engine.markers)
        self.measurements.set_rows(self._compose_measure_rows())
        self._refresh_properties()
        self._refresh_boolean_selection_ui()
        self._refresh_marker_controls()
        self._update_undo_redo_buttons()

    def _compose_measure_rows(self):
        rows = [dict(r) for r in self.engine.measurements]
        for mk in self.engine.markers.values():
            vals = ", ".join([f"{k}={v:.5g}" for k, v in mk.last_values.items()]) if mk.last_values else "-"
            rows.append(
                {
                    "type": "marker",
                    "name": mk.name,
                    "value": vals,
                    "info": "valid" if mk.valid else f"invalid: {mk.error}",
                    "object_id": mk.target_object_id or "",
                    "marker_id": mk.id,
                    "p1": [float(x) for x in mk.position.tolist()],
                }
            )
        return rows

    def _refresh_properties(self):
        oid = self.engine.selection[0] if self.engine.selection else ""
        if not oid or oid not in self.engine.objects:
            self.properties.set_object_info("", None)
            return
        obj = self.engine.objects[oid]
        data = {
            "name": obj.name,
            "source": obj.source,
            "visible": obj.visible,
            "locked": obj.locked,
            "metrics": object_metrics(obj),
        }
        self.properties.set_object_info(oid, data)

    def _on_tree_selection(self, ids: list, mode: str):
        self.engine.select(ids, mode=mode)

    def _on_viewport_selection(self, ids: list, mode: str):
        self.engine.select(ids, mode=mode)

    def _on_apply_transform(self, payload: dict):
        ids = self.engine.selection
        if not ids:
            return
        self.engine.transform_objects(ids=ids, **payload)

    def _show_context_menu(self, ctx: ContextInfo):
        self._last_cursor_ctx = ctx
        menu = self.dispatcher.build_menu(ctx)
        gp = ctx.global_pos or QPoint(0, 0)
        menu.exec(gp)

    def _context_log(self, pos: QPoint):
        ctx = ContextInfo(widget="misc", mouse_pos=pos, global_pos=self.log_box.mapToGlobal(pos), selected_ids=list(self.engine.selection))
        self._show_context_menu(ctx)

    def _log(self, text: str):
        self.log_box.append(str(text))

    def _pick_selected(self, fallback_all: bool = False) -> List[str]:
        ids = [i for i in self.engine.selection if i in self.engine.objects]
        if ids:
            return ids
        if fallback_all:
            return list(self.engine.objects.keys())
        return []

    def _ask_confirm(self, title: str, text: str) -> bool:
        return QMessageBox.question(self, title, text, QMessageBox.Yes | QMessageBox.No, QMessageBox.No) == QMessageBox.Yes

    # ---------------- generic/fallback ----------------
    def action_not_implemented(self, name: str):
        self._log(f"[TODO] {name} not implemented.")

    def action_show_shortcuts(self):
        QMessageBox.information(
            self,
            "Atalhos",
            "Del: delete selected\nCtrl+Z/Ctrl+Y: undo/redo\nF: fit selected\nEsc: select mode\nShift+Click: add selection",
        )

    # ---------------- undo/redo ----------------
    def action_undo(self):
        self.engine.undo()

    def action_redo(self):
        self.engine.redo()

    # ---------------- view ----------------
    def action_reset_view(self):
        self.viewport.reset_view()

    def action_fit_all(self):
        self.viewport.fit_all()

    def action_toggle_grid(self):
        self.viewport.toggle_grid()

    def action_toggle_axes(self):
        self.viewport.toggle_axes()

    def action_set_render_mode(self, mode: str):
        self.viewport.set_render_mode(mode)

    def action_set_background(self, mode: str):
        self.viewport.set_background(mode)

    def action_screenshot(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save screenshot", "viewport.png", "PNG (*.png)")
        if not path:
            return
        self.viewport.screenshot(path)
        self._log(f"Screenshot saved: {path}")

    # ---------------- selection ----------------
    def action_select_all(self):
        self.engine.select([], mode="all")

    def action_select_none(self):
        self.engine.select([], mode="none")

    def action_select_invert(self):
        self.engine.select([], mode="invert")

    def action_select_this(self, oid: str):
        self.engine.select([oid], mode="replace")

    def action_add_to_selection(self, oid: str):
        self.engine.select([oid], mode="add")

    def action_remove_from_selection(self, oid: str):
        self.engine.select([oid], mode="remove")

    def action_isolate_selection(self, ids: Sequence[str]):
        keep = set([str(i) for i in ids if i in self.engine.objects])
        if not keep:
            return
        all_ids = list(self.engine.objects.keys())
        hide = [i for i in all_ids if i not in keep]
        if hide:
            self.engine.set_visibility(hide, False)
        self.engine.set_visibility(list(keep), True)
        self.engine.select(list(keep), mode="replace")

    def action_show_only_selected(self):
        ids = self._pick_selected()
        if ids:
            self.action_isolate_selection(ids)

    def action_show_all(self):
        self.engine.set_visibility(list(self.engine.objects.keys()), True)

    def action_hide_selected(self):
        ids = self._pick_selected()
        if ids:
            self.engine.set_visibility(ids, False)

    def action_show_selected(self):
        ids = self._pick_selected()
        if ids:
            self.engine.set_visibility(ids, True)

    # ---------------- tool mode ----------------
    def action_set_tool_mode(self, mode: str):
        self.engine.tool_mode = str(mode)
        self.viewport.set_tool_mode(str(mode))
        if str(mode).lower() != "measure":
            self._measure_kind = ""
            self._measure_points = []
            self._set_hint("Select mode active. Click to select. Shift adds. RMB opens full context menu.")
        else:
            self._set_hint(
                "Measure mode active. Click points in viewport.",
                [
                    "Measure wizard",
                    "Distance: click 2 points.",
                    "Angle: click 3 points.",
                    "Esc or Select mode to exit.",
                ],
            )
        self._log(f"Tool mode set: {mode}")

    def action_enter_measure_mode(self, kind: str):
        self._measure_kind = str(kind or "distance").lower()
        self._measure_points = []
        self.action_set_tool_mode("Measure")
        self._log(f"Measure mode: {self._measure_kind}")
        if self._measure_kind == "angle":
            self._set_hint("Measure angle: click 3 points.", ["Measure wizard", "Step 1: first point", "Step 2: vertex", "Step 3: last point"])
        else:
            self._set_hint("Measure distance: click 2 points.", ["Measure wizard", "Step 1: first point", "Step 2: second point"])

    def _on_measure_point(self, p: tuple):
        if str(self.engine.tool_mode).lower() != "measure":
            return
        self._measure_points.append(tuple(float(x) for x in p))
        if self._measure_kind == "distance" and len(self._measure_points) >= 2:
            self.engine.compute_distance(self._measure_points[-2], self._measure_points[-1], name="distance")
            self._log("Distance measured.")
            self._set_hint("Distance captured. Click two new points for another measurement.")
            self._measure_points = []
        elif self._measure_kind == "angle" and len(self._measure_points) >= 3:
            self.engine.compute_angle(self._measure_points[-3], self._measure_points[-2], self._measure_points[-1], name="angle")
            self._log("Angle measured.")
            self._set_hint("Angle captured. Click three new points for another measurement.")
            self._measure_points = []
        else:
            need = 3 if self._measure_kind == "angle" else 2
            self._set_hint(f"Measurement progress: {len(self._measure_points)}/{need} points.")

    def action_clear_current_measure(self):
        self._measure_points = []
        self._log("Current measure points cleared.")

    def action_copy_last_measure(self):
        if not self.engine.measurements:
            return
        row = self.engine.measurements[-1]
        text = json.dumps(row, ensure_ascii=False)
        QApplication.clipboard().setText(text)

    def action_save_measure_as_marker(self):
        if not self.engine.measurements:
            return
        last = self.engine.measurements[-1]
        if last.get("type") == "distance":
            p = np.mean(np.asarray([last["p1"], last["p2"]], dtype=float), axis=0)
        elif last.get("type") == "angle":
            p = np.asarray(last["p2"], dtype=float)
        else:
            p = np.zeros(3, dtype=float)
        self.engine.add_marker(position=p, name=f"Measure_{last.get('type', 'marker')}")

    # ---------------- create/import ----------------
    def action_import_mesh(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Import mesh", "", "OBJ/STL/PLY (*.obj *.stl *.ply)")
        if not paths:
            return
        imported = 0
        try:
            import pyvista as pv
        except Exception as e:
            QMessageBox.critical(self, "Import mesh", f"PyVista required: {e}")
            return
        for p in paths:
            try:
                poly = pv.read(p).triangulate().clean()
                verts = np.asarray(poly.points, dtype=float)
                faces = np.asarray(poly.faces).reshape(-1, 4)[:, 1:4].astype(int)
                if verts.size == 0 or faces.size == 0:
                    continue
                mesh = MeshData(vertices=verts, faces=faces)
                self.engine.add_mesh_object(mesh, name=os.path.splitext(os.path.basename(p))[0], source="Import")
                imported += 1
            except Exception as e:
                self._log(f"Import failed {p}: {e}")
        self._log(f"Imported meshes: {imported}")

    def action_create_primitive_dialog(self, kind: str):
        token = str(kind).lower()
        name, ok = QInputDialog.getText(self, "Primitive", "Name:", text=f"{token}_1")
        if not ok:
            return
        params = {"center": (0.0, 0.0, 0.0)}
        if token in {"box", "plane"}:
            w, ok = QInputDialog.getDouble(self, "Primitive", "Width:", 1.0, 1e-6, 1e9, 6)
            if not ok:
                return
            d, ok = QInputDialog.getDouble(self, "Primitive", "Depth:", 1.0, 1e-6, 1e9, 6)
            if not ok:
                return
            params["width"] = w
            params["depth"] = d
            if token == "box":
                h, ok = QInputDialog.getDouble(self, "Primitive", "Height:", 1.0, 1e-6, 1e9, 6)
                if not ok:
                    return
                params["height"] = h
        elif token in {"cylinder", "cone"}:
            r, ok = QInputDialog.getDouble(self, "Primitive", "Radius:", 0.5, 1e-6, 1e9, 6)
            if not ok:
                return
            h, ok = QInputDialog.getDouble(self, "Primitive", "Height:", 1.0, 1e-6, 1e9, 6)
            if not ok:
                return
            params["radius"] = r
            params["height"] = h
            params["segments"] = 32
        elif token == "sphere":
            r, ok = QInputDialog.getDouble(self, "Primitive", "Radius:", 0.5, 1e-6, 1e9, 6)
            if not ok:
                return
            params["radius"] = r
        else:
            return
        self.engine.create_primitive(token, params=params, name=name or token)

    def action_add_marker_at_cursor(self, ctx: ContextInfo):
        p = ctx.picked_point_3d or (0.0, 0.0, 0.0)
        self.engine.add_marker(position=p, name="Marker")

    def action_add_label_at_cursor(self, ctx: ContextInfo):
        p = ctx.picked_point_3d or (0.0, 0.0, 0.0)
        self.engine.add_marker(position=p, name="Label")

    def action_add_custom_math_marker(self, ctx: ContextInfo):
        p = ctx.picked_point_3d or (0.0, 0.0, 0.0)
        expr, ok = QInputDialog.getText(self, "Math marker", "Expression (ex.: mass=density*volume):", text="mass=volume")
        if not ok:
            return
        expr = str(expr).strip()
        if "=" not in expr:
            QMessageBox.warning(self, "Math marker", "Use format key=expression.")
            return
        key, val = expr.split("=", 1)
        mid = self.engine.add_marker(position=p, name="MathMarker", expressions={key.strip(): val.strip()})
        mk = self.engine.markers.get(mid)
        if mk and not mk.valid:
            QMessageBox.warning(self, "Math marker", f"Expression invalid: {mk.error}")

    # ---------------- clipping ----------------
    def action_add_clipping_plane(self, orientation: str):
        self.viewport.add_clipping_plane(orientation)

    def action_clear_clipping(self):
        self.viewport.clear_clipping()

    def action_toggle_clipping(self):
        self.viewport.toggle_clipping()

    # ---------------- transform/edit ----------------
    def action_transform_dialog(self, op: str, ids: Sequence[str]):
        ids = [str(i) for i in ids if i in self.engine.objects]
        if not ids:
            ids = self._pick_selected()
        if not ids:
            return
        op = str(op).lower()
        if op == "move":
            tx, ok = QInputDialog.getDouble(self, "Move", "Tx:", 0.0, -1e9, 1e9, 6)
            if not ok:
                return
            ty, ok = QInputDialog.getDouble(self, "Move", "Ty:", 0.0, -1e9, 1e9, 6)
            if not ok:
                return
            tz, ok = QInputDialog.getDouble(self, "Move", "Tz:", 0.0, -1e9, 1e9, 6)
            if not ok:
                return
            if self._snap_enabled:
                tx = round(tx / self._snap_step) * self._snap_step
                ty = round(ty / self._snap_step) * self._snap_step
                tz = round(tz / self._snap_step) * self._snap_step
            self.engine.transform_objects(ids, tx=tx, ty=ty, tz=tz)
        elif op == "rotate":
            rx, ok = QInputDialog.getDouble(self, "Rotate", "Rx (deg):", 0.0, -3600.0, 3600.0, 4)
            if not ok:
                return
            ry, ok = QInputDialog.getDouble(self, "Rotate", "Ry (deg):", 0.0, -3600.0, 3600.0, 4)
            if not ok:
                return
            rz, ok = QInputDialog.getDouble(self, "Rotate", "Rz (deg):", 0.0, -3600.0, 3600.0, 4)
            if not ok:
                return
            self.engine.transform_objects(ids, rx_deg=rx, ry_deg=ry, rz_deg=rz)
        elif op == "scale":
            s, ok = QInputDialog.getDouble(self, "Scale", "Scale factor:", 1.0, 1e-6, 1e6, 6)
            if not ok:
                return
            self.engine.transform_objects(ids, scale=s)

    def action_align_world(self, ids: Sequence[str]):
        ids = [str(i) for i in ids if i in self.engine.objects]
        if not ids:
            return
        self.engine.transform_objects(ids, rx_deg=0.0, ry_deg=0.0, rz_deg=0.0, scale=1.0)

    def action_align_plane_dialog(self, ids: Sequence[str]):
        axis, ok = QInputDialog.getItem(self, "Align plane", "Axis:", ["XY", "XZ", "YZ"], 0, False)
        if not ok:
            return
        self._log(f"Align to plane requested: {axis} on {len(ids)} object(s).")

    def action_toggle_snap(self):
        self._set_snap_enabled(not self._snap_enabled)
        self._log(f"Snap to grid {'enabled' if self._snap_enabled else 'disabled'}.")

    def action_set_snap_step(self):
        val, ok = QInputDialog.getDouble(self, "Snap step", "Step:", self._snap_step, 1e-9, 1e9, 6)
        if not ok:
            return
        self._snap_step = float(val)
        self._prefs.setdefault("snap", {})
        self._prefs["snap"]["step_mm"] = float(self._snap_step)
        self._save_preferences()
        if hasattr(self, "tr_step_edit"):
            self.tr_step_edit.setText(f"{self._snap_step:.6g}")

    def action_duplicate(self, ids: Sequence[str]):
        ids = [str(i) for i in ids if i in self.engine.objects] or self._pick_selected()
        if ids:
            self.engine.duplicate_objects(ids)

    def action_mirror_dialog(self, ids: Sequence[str]):
        ids = [str(i) for i in ids if i in self.engine.objects] or self._pick_selected()
        if not ids:
            return
        axis, ok = QInputDialog.getItem(self, "Mirror", "Axis:", ["X", "Y", "Z"], 0, False)
        if not ok:
            return
        idx = {"X": 0, "Y": 1, "Z": 2}[axis]

        def _apply(eng: SceneEngine):
            for oid in ids:
                obj = eng.objects[oid]
                v = np.asarray(obj.mesh.vertices, dtype=float)
                c = np.mean(v, axis=0)
                v[:, idx] = 2.0 * c[idx] - v[:, idx]
                obj.mesh.vertices = v

        self.engine.execute("Mirror object(s)", _apply)

    def action_array_dialog(self, ids: Sequence[str]):
        ids = [str(i) for i in ids if i in self.engine.objects] or self._pick_selected()
        if not ids:
            return
        mode, ok = QInputDialog.getItem(self, "Array", "Mode:", ["Linear", "Circular"], 0, False)
        if not ok:
            return
        if mode == "Linear":
            n, ok = QInputDialog.getInt(self, "Array linear", "Copies:", 3, 1, 1000)
            if not ok:
                return
            dx, ok = QInputDialog.getDouble(self, "Array linear", "dx:", 1.0, -1e6, 1e6, 6)
            if not ok:
                return
            dy, ok = QInputDialog.getDouble(self, "Array linear", "dy:", 0.0, -1e6, 1e6, 6)
            if not ok:
                return
            dz, ok = QInputDialog.getDouble(self, "Array linear", "dz:", 0.0, -1e6, 1e6, 6)
            if not ok:
                return
            created = []
            for k in range(1, n + 1):
                new_ids = self.engine.duplicate_objects(ids)
                if new_ids:
                    self.engine.transform_objects(new_ids, tx=k * dx, ty=k * dy, tz=k * dz)
                    created.extend(new_ids)
            self.engine.select(created, mode="replace")
        else:
            n, ok = QInputDialog.getInt(self, "Array circular", "Copies:", 6, 1, 1000)
            if not ok:
                return
            step = 360.0 / float(max(1, n))
            created = []
            for k in range(1, n + 1):
                new_ids = self.engine.duplicate_objects(ids)
                if new_ids:
                    self.engine.transform_objects(new_ids, rz_deg=k * step)
                    created.extend(new_ids)
            self.engine.select(created, mode="replace")

    def action_rename_dialog(self, oid: str):
        if oid not in self.engine.objects:
            return
        cur = self.engine.objects[oid].name
        new_name, ok = QInputDialog.getText(self, "Rename", "New name:", text=cur)
        if not ok or not new_name.strip():
            return
        self.engine.rename_object(oid, new_name.strip())

    def action_rename_selected(self):
        ids = self._pick_selected()
        if ids:
            self.action_rename_dialog(ids[0])

    def action_set_color_dialog(self, ids: Sequence[str]):
        ids = [str(i) for i in ids if i in self.engine.objects] or self._pick_selected()
        if not ids:
            return
        col = QColorDialog.getColor(QColor("#86b6f6"), self, "Set color")
        if not col.isValid():
            return
        self.engine.set_color_opacity(ids, color=col.name())

    def action_set_opacity_dialog(self, ids: Sequence[str]):
        ids = [str(i) for i in ids if i in self.engine.objects] or self._pick_selected()
        if not ids:
            return
        val, ok = QInputDialog.getDouble(self, "Opacity", "Opacity [0.05..1.0]:", 0.85, 0.05, 1.0, 3)
        if not ok:
            return
        self.engine.set_color_opacity(ids, opacity=val)

    def action_style_dialog(self, ids: Sequence[str]):
        self.action_set_color_dialog(ids)
        self.action_set_opacity_dialog(ids)

    def action_toggle_lock(self, ids: Sequence[str]):
        ids = [str(i) for i in ids if i in self.engine.objects] or self._pick_selected()
        if not ids:
            return
        base = self.engine.objects[ids[0]].locked
        self.engine.set_lock(ids, not bool(base))

    def action_toggle_lock_selected(self):
        self.action_toggle_lock(self._pick_selected())

    def action_toggle_visibility(self, ids: Sequence[str]):
        ids = [str(i) for i in ids if i in self.engine.objects]
        if not ids:
            return
        base = self.engine.objects[ids[0]].visible
        self.engine.set_visibility(ids, not bool(base))

    def action_delete_selected(self):
        ids = self._pick_selected()
        if not ids:
            return
        if not self._ask_confirm("Delete", f"Delete {len(ids)} selected object(s)?"):
            return
        self.engine.delete_objects(ids)

    def action_reset_transform(self, ids: Sequence[str] = ()):
        ids = [str(i) for i in ids if i in self.engine.objects] or self._pick_selected()
        if not ids:
            return
        self.engine.transform_objects(ids, tx=0.0, ty=0.0, tz=0.0, rx_deg=0.0, ry_deg=0.0, rz_deg=0.0, scale=1.0)

    # ---------------- booleans ----------------
    def action_boolean_settings(self):
        val, ok = QInputDialog.getDouble(self, "Boolean settings", "Tolerance:", self._boolean_tolerance, 1e-9, 1.0, 9)
        if ok:
            self._boolean_tolerance = float(val)
            self._prefs.setdefault("boolean", {})
            self._prefs["boolean"]["tolerance"] = float(self._boolean_tolerance)
            self._save_preferences()
            if hasattr(self, "bool_tol_edit"):
                self.bool_tol_edit.setText(f"{self._boolean_tolerance:.9g}")

    def action_boolean_union(self):
        ids = self._pick_selected()
        if len(ids) >= 2:
            self._boolean_primary_id = ids[0]
            self._boolean_tool_ids = ids[1:]
            self._refresh_boolean_selection_ui()
        self._apply_boolean_from_tab("union")

    def action_boolean_subtract(self):
        ids = self._pick_selected()
        if len(ids) >= 2:
            self._boolean_primary_id = ids[0]
            self._boolean_tool_ids = ids[1:]
            self._refresh_boolean_selection_ui()
        self._apply_boolean_from_tab("subtract")

    def action_boolean_intersect(self):
        ids = self._pick_selected()
        if len(ids) >= 2:
            self._boolean_primary_id = ids[0]
            self._boolean_tool_ids = ids[1:]
            self._refresh_boolean_selection_ui()
        self._apply_boolean_from_tab("intersect")

    # ---------------- measure/analyze ----------------
    def action_measure_bbox(self, ids: Sequence[str]):
        for oid in [str(i) for i in ids if i in self.engine.objects]:
            row = self.engine.compute_object_measure(oid)
            self._log(f"BBox: {row.get('bbox_dx'):.4g} x {row.get('bbox_dy'):.4g} x {row.get('bbox_dz'):.4g}")

    def action_measure_volume(self, ids: Sequence[str]):
        for oid in [str(i) for i in ids if i in self.engine.objects]:
            row = self.engine.compute_object_measure(oid)
            self._log(f"Volume {row.get('name')}: {row.get('volume'):.6g}")

    def action_measure_area(self, ids: Sequence[str]):
        for oid in [str(i) for i in ids if i in self.engine.objects]:
            row = self.engine.compute_object_measure(oid)
            self._log(f"Area {row.get('name')}: {row.get('area'):.6g}")

    def action_measure_centroid(self, ids: Sequence[str]):
        for oid in [str(i) for i in ids if i in self.engine.objects]:
            row = self.engine.compute_object_measure(oid)
            self._log(
                f"Centroid {row.get('name')}: ({row.get('centroid_x'):.5g}, {row.get('centroid_y'):.5g}, {row.get('centroid_z'):.5g})"
            )

    # ---------------- grouping/alignment ----------------
    def action_group_selected(self):
        ids = self._pick_selected()
        if len(ids) < 2:
            return
        group_name, ok = QInputDialog.getText(self, "Group", "Group name:", text="Group1")
        if not ok:
            return

        def _apply(eng: SceneEngine):
            for oid in ids:
                eng.objects[oid].meta["group"] = str(group_name or "Group")

        self.engine.execute("Group selected", _apply)

    def action_ungroup_selected(self):
        ids = self._pick_selected()
        if not ids:
            return

        def _apply(eng: SceneEngine):
            for oid in ids:
                eng.objects[oid].meta.pop("group", None)

        self.engine.execute("Ungroup selected", _apply)

    def action_align_centers(self, axis: str):
        ids = self._pick_selected()
        if len(ids) < 2:
            return
        axis = str(axis).lower()
        idx = {"x": 0, "y": 1, "z": 2}.get(axis, 0)
        c_ref = np.mean(self.engine.objects[ids[0]].mesh.vertices, axis=0)[idx]

        def _apply(eng: SceneEngine):
            for oid in ids[1:]:
                obj = eng.objects[oid]
                c = np.mean(obj.mesh.vertices, axis=0)
                delta = c_ref - c[idx]
                t = [0.0, 0.0, 0.0]
                t[idx] = float(delta)
                obj.mesh = transform_mesh(obj.mesh, tx=t[0], ty=t[1], tz=t[2])

        self.engine.execute("Align centers", _apply)

    def action_distribute(self, axis: str):
        ids = self._pick_selected()
        if len(ids) < 3:
            return
        axis = str(axis).lower()
        idx = {"x": 0, "y": 1, "z": 2}.get(axis, 0)
        pairs = []
        for oid in ids:
            c = np.mean(self.engine.objects[oid].mesh.vertices, axis=0)[idx]
            pairs.append((oid, float(c)))
        pairs.sort(key=lambda x: x[1])
        lo, hi = pairs[0][1], pairs[-1][1]
        step = (hi - lo) / float(max(1, len(pairs) - 1))

        def _apply(eng: SceneEngine):
            for k, (oid, c0) in enumerate(pairs):
                target = lo + k * step
                delta = target - c0
                t = [0.0, 0.0, 0.0]
                t[idx] = float(delta)
                eng.objects[oid].mesh = transform_mesh(eng.objects[oid].mesh, tx=t[0], ty=t[1], tz=t[2])

        self.engine.execute("Distribute objects", _apply)

    # ---------------- export ----------------
    def action_export_selected(self, fmt: str):
        ids = self._pick_selected()
        if not ids:
            return
        ext = "stl" if str(fmt).lower() == "stl" else "obj"
        out_dir = QFileDialog.getExistingDirectory(self, "Export selection")
        if not out_dir:
            return
        try:
            import pyvista as pv
        except Exception as e:
            QMessageBox.warning(self, "Export", f"PyVista required: {e}")
            return
        count = 0
        for oid in ids:
            obj = self.engine.objects[oid]
            faces = np.asarray(obj.mesh.faces, dtype=int)
            verts = np.asarray(obj.mesh.vertices, dtype=float)
            ff = np.empty((faces.shape[0], 4), dtype=np.int64)
            ff[:, 0] = 3
            ff[:, 1:] = faces.astype(np.int64)
            poly = pv.PolyData(verts, ff.reshape(-1))
            path = os.path.join(out_dir, f"{obj.name}.{ext}")
            poly.save(path)
            count += 1
        self._log(f"Exported {count} mesh(es) to {out_dir}")

    # ---------------- marker helpers ----------------
    def action_marker_centroid(self, oid: str):
        if oid not in self.engine.objects:
            return
        c = np.mean(self.engine.objects[oid].mesh.vertices, axis=0)
        self.engine.add_marker(position=c, name=f"{self.engine.objects[oid].name}_centroid", target_object_id=oid)

    def action_label_centroid(self, oid: str):
        self.action_marker_centroid(oid)

    # ---------------- properties context ----------------
    def action_copy_property_value(self, field_name: str):
        field = str(field_name or "").strip().lower()
        text = ""
        mapping = {
            "tx": self.properties.ed_tx.text,
            "ty": self.properties.ed_ty.text,
            "tz": self.properties.ed_tz.text,
            "rx": self.properties.ed_rx.text,
            "ry": self.properties.ed_ry.text,
            "rz": self.properties.ed_rz.text,
            "scale": self.properties.ed_scale.text,
        }
        if field in mapping:
            text = mapping[field]()
        QApplication.clipboard().setText(text)

    def action_copy_all_properties(self):
        ids = self._pick_selected()
        if not ids:
            return
        oid = ids[0]
        obj = self.engine.objects[oid]
        row = {
            "id": oid,
            "name": obj.name,
            "visible": obj.visible,
            "locked": obj.locked,
            "color": obj.color,
            "opacity": obj.opacity,
            "metrics": object_metrics(obj),
        }
        def _f(txt: str, default: float) -> float:
            try:
                return float(str(txt).strip().replace(",", "."))
            except Exception:
                return float(default)
        self._copied_transform = {
            "tx": _f(self.properties.ed_tx.text(), 0.0),
            "ty": _f(self.properties.ed_ty.text(), 0.0),
            "tz": _f(self.properties.ed_tz.text(), 0.0),
            "rx_deg": _f(self.properties.ed_rx.text(), 0.0),
            "ry_deg": _f(self.properties.ed_ry.text(), 0.0),
            "rz_deg": _f(self.properties.ed_rz.text(), 0.0),
            "scale": _f(self.properties.ed_scale.text(), 1.0),
        }
        QApplication.clipboard().setText(json.dumps(row, ensure_ascii=False, indent=2))

    def action_paste_transform(self):
        if not self._copied_transform:
            return
        ids = self._pick_selected()
        if ids:
            self.engine.transform_objects(ids, **self._copied_transform)

    def action_bake_transform(self):
        self._log("Bake transform: already applied directly to mesh.")

    # ---------------- measurements context ----------------
    def action_copy_measurement_row(self, row: Optional[int]):
        if row is None or row < 0:
            return
        data = self.measurements.row_data(int(row))
        QApplication.clipboard().setText(json.dumps(data, ensure_ascii=False))

    def action_copy_measurement_table(self):
        text = "\n".join([json.dumps(r, ensure_ascii=False) for r in self._compose_measure_rows()])
        QApplication.clipboard().setText(text)

    def action_export_measurements_csv(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export measurements", "measurements.csv", "CSV (*.csv)")
        if not path:
            return
        rows = self._compose_measure_rows()
        keys = sorted({k for r in rows for k in r.keys()})
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        self._log(f"Measurements exported: {path}")

    def action_delete_measurement_row(self, row: Optional[int]):
        if row is None or row < 0:
            return
        data = self.measurements.row_data(int(row))
        if not data:
            return
        if str(data.get("type", "")).lower() == "marker":
            mid = str(data.get("marker_id", "")).strip()
            if mid:
                self.engine.delete_marker(mid)
            return
        if row >= len(self.engine.measurements):
            return

        def _apply(eng: SceneEngine):
            eng.measurements.pop(int(row))

        self.engine.execute("Delete measurement", _apply)

    # ---------------- focus / misc ----------------
    def action_focus_object(self, oid: str):
        if oid in self.engine.objects:
            self.engine.select([oid], mode="replace")
        self.viewport.fit_all()

    def action_open_preferences(self):
        units, ok = QInputDialog.getItem(self, "Preferences", "Units:", ["mm", "cm", "m", "in"], self.units_combo.currentIndex(), False)
        if not ok:
            return
        snap_step, ok = QInputDialog.getDouble(self, "Preferences", "Grid snap step:", self._snap_step, 1e-9, 1e9, 6)
        if not ok:
            return
        angle_step, ok = QInputDialog.getDouble(self, "Preferences", "Angle snap step:", self._angle_snap_step, 0.001, 360.0, 4)
        if not ok:
            return
        bg, ok = QInputDialog.getItem(self, "Preferences", "Background:", ["dark", "light"], 0, False)
        if not ok:
            return
        render, ok = QInputDialog.getItem(self, "Preferences", "Render mode:", ["solid_edges", "solid", "wireframe"], 0, False)
        if not ok:
            return
        self._prefs["units"] = str(units)
        self._prefs.setdefault("snap", {})
        self._prefs["snap"]["step_mm"] = float(snap_step)
        self._prefs["snap"]["step_deg"] = float(angle_step)
        self._prefs["snap"]["grid"] = bool(self._snap_enabled)
        self._prefs.setdefault("render", {})
        self._prefs["render"]["background"] = str(bg)
        self._prefs["render"]["mode"] = str(render)
        self._save_preferences()
        self._apply_preferences()
        self._set_hint("Preferences updated.")

    # ---------------- persistence ----------------
    def _save_markers(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save markers", "markers.json", "JSON (*.json)")
        if not path:
            return
        self.engine.save_markers(path)
        self._log(f"Markers saved: {path}")

    def _load_markers(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load markers", "", "JSON (*.json)")
        if not path:
            return
        n = self.engine.load_markers(path)
        self._log(f"Markers loaded: {n}")


def launch_mechanics_window():
    app = QApplication.instance()
    owns = False
    if app is None:
        app = QApplication([])
        owns = True
    win = QMainWindow()
    win.setWindowTitle("EFTX - Analises Mecanicas")
    page = MechanicsPage(win)
    win.setCentralWidget(page)
    win.resize(1600, 980)
    win.show()
    if owns:
        app.exec()
    return win


if __name__ == "__main__":
    launch_mechanics_window()
