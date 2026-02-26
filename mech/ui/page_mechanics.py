from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from PySide6.QtCore import Qt, QPoint, QByteArray, QTimer
from PySide6.QtGui import QAction, QColor, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QColorDialog,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QTabWidget,
    QComboBox,
    QCheckBox,
    QLineEdit,
    QSplitter,
    QStyle,
    QToolButton,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
    QMenu,
)

from mech.engine.scene_engine import SceneEngine
from mech.engine.scene_object import MeshData, SceneObject
from mech.engine.geometry_ops import transform_mesh
from mech.engine.measures import object_metrics

from .context_menu import ContextInfo, ContextMenuDispatcher
from .measurements_panel import MeasurementsPanel
from .properties_panel import PropertiesPanel
from .scene_tree import SceneTreeWidget
from .selection_manager import SelectionItem, SelectionManager, SelectionMode
from .viewport_pyvista import ViewportPyVista


class MechanicsPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.engine = SceneEngine()
        self.selection_manager = SelectionManager(self)
        self.dispatcher = ContextMenuDispatcher(self)
        self._selection_sync_lock = False
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
        self._boundary_overlay_enabled = True
        self._boundary_types = ["fixed", "force", "pressure", "displacement", "symmetry", "contact"]
        self._mesh_deflection = 0.2
        self._grid_enabled = True
        self._grid_step_mm = 5.0
        self._grid_size_mm = 2000.0
        self._grid_color = "#2b2b2b"
        self._subselect_mode = "object"
        self._selected_face: Optional[tuple] = None
        self._selected_edge: Optional[tuple] = None
        self._selected_vertex: Optional[tuple] = None
        self._analysis_tabs: Dict[str, QWidget] = {}
        self._fem_result_cache: Dict[str, Any] = {}
        self._panels_collapsed = False
        self._last_main_sizes = [260, 1340, 320]
        self._last_outer_sizes = [900, 120]
        self._last_left_width = 260
        self._last_right_width = 320
        self._last_bottom_height = 170
        self._max_viewport_state: Optional[dict] = None
        self._panel_shortcuts: List[QShortcut] = []
        self._xray_mode = False
        self._prefs = self._default_preferences()
        self._config_dir = Path.home() / ".eftx_converter" / "mech"
        self._layouts_dir = self._config_dir / "layouts"
        self._build_ui()
        self._connect()
        self._load_preferences()
        self._apply_preferences()
        self.load_layout("layout_default", silent=True)
        self.engine.add_listener(self._on_engine_event)
        self._refresh_backend_status()
        self.refresh_all()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)

        head = QHBoxLayout()
        title = QLabel("Analises Mecanicas - 3D Modeler + FEM")
        title.setStyleSheet("font-weight: 600;")
        head.addWidget(title)
        head.addStretch(1)
        self.btn_toggle_left = QPushButton("Left", self)
        self.btn_toggle_left.clicked.connect(self.action_toggle_left_panel)
        head.addWidget(self.btn_toggle_left)
        self.btn_toggle_right = QPushButton("Right", self)
        self.btn_toggle_right.clicked.connect(self.action_toggle_right_panel)
        head.addWidget(self.btn_toggle_right)
        self.btn_toggle_bottom = QPushButton("Bottom", self)
        self.btn_toggle_bottom.clicked.connect(self.action_toggle_bottom_panel)
        head.addWidget(self.btn_toggle_bottom)
        self.btn_toggle_panels = QPushButton("Max Viewport", self)
        self.btn_toggle_panels.clicked.connect(self.action_toggle_panels)
        head.addWidget(self.btn_toggle_panels)
        self.backend_label = QLabel("Backend: probing...")
        head.addWidget(self.backend_label)
        root.addLayout(head)

        self.ribbon_tabs = self._build_ribbon_tabs()
        self.ribbon_tabs.setMinimumHeight(96)
        self.ribbon_tabs.setMaximumHeight(142)
        root.addWidget(self.ribbon_tabs)

        self.ops_tabs = self._build_ops_tabs()
        self.ops_tabs.setMinimumHeight(110)
        self.ops_tabs.setMaximumHeight(190)
        root.addWidget(self.ops_tabs)

        self.main_split = QSplitter(Qt.Horizontal, self)
        self.main_split.setHandleWidth(8)
        self.scene_tree = SceneTreeWidget(self)
        self.viewport = ViewportPyVista(self)

        self.left_tabs = QTabWidget(self)
        self.left_tabs.setDocumentMode(True)
        self.left_tabs.setStyleSheet(
            "QTabBar::tab{padding:6px 10px; min-height:24px;}"
            "QTabWidget::pane{border:1px solid #314050;}"
        )
        self.left_tabs.addTab(self.scene_tree, "Scene")
        self.left_tabs.addTab(self._build_components_library_tab(), "Components")
        self.left_tabs.addTab(self._build_layers_overview_tab(), "Layers")
        self.left_tabs.addTab(self._build_studies_tab(), "Studies")

        self.right_wrap = QWidget(self)
        right_layout = QVBoxLayout(self.right_wrap)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(6)
        self.properties = PropertiesPanel(self)
        self.right_tabs = QTabWidget(self)
        self.right_tabs.addTab(self.properties, "Properties")
        self.right_tabs.addTab(self._build_right_transform_tab(), "Transform")
        self.right_tabs.addTab(self._build_right_material_tab(), "Material")
        self.right_tabs.addTab(self._build_right_fem_study_tab(), "FEM")
        self.right_tabs.addTab(self._build_right_bcs_tab(), "BCs")
        self.right_tabs.addTab(self._build_right_loads_tab(), "Loads")
        self.right_tabs.addTab(self._build_right_mesh_tab(), "Mesh")
        self.right_tabs.addTab(self._build_right_solve_tab(), "Solve")
        self.right_tabs.addTab(self._build_right_results_tab(), "Results")
        self.inline_wizard_box = QTextEdit(self)
        self.inline_wizard_box.setReadOnly(True)
        self.inline_wizard_box.setMaximumHeight(120)
        self.inline_wizard_box.setPlaceholderText("Inline Wizard")
        right_layout.addWidget(self.right_tabs, 5)
        right_layout.addWidget(QLabel("Inline Wizard"))
        right_layout.addWidget(self.inline_wizard_box, 1)

        self.main_split.addWidget(self.left_tabs)
        self.main_split.addWidget(self.viewport)
        self.main_split.addWidget(self.right_wrap)
        self.main_split.setStretchFactor(0, 1)
        self.main_split.setStretchFactor(1, 7)
        self.main_split.setStretchFactor(2, 2)

        self.log_box = QTextEdit(self)
        self.log_box.setReadOnly(True)
        self.log_box.setMinimumHeight(90)
        self.log_box.setContextMenuPolicy(Qt.CustomContextMenu)
        self.log_box.customContextMenuRequested.connect(self._context_log)
        self.measurements = MeasurementsPanel(self)
        self.selection_info_box = QTextEdit(self)
        self.selection_info_box.setReadOnly(True)
        self.selection_info_box.setMinimumHeight(90)
        self.diagnostics_box = QTextEdit(self)
        self.diagnostics_box.setReadOnly(True)
        self.diagnostics_box.setMinimumHeight(90)
        self.solver_console_box = QTextEdit(self)
        self.solver_console_box.setReadOnly(True)
        self.solver_console_box.setMinimumHeight(90)

        self.bottom_tabs = QTabWidget(self)
        self.bottom_tabs.addTab(self.log_box, "Log")
        self.bottom_tabs.addTab(self.measurements, "Measurements")
        self.bottom_tabs.addTab(self.selection_info_box, "Selection Info")
        self.bottom_tabs.addTab(self.diagnostics_box, "Diagnostics")
        self.bottom_tabs.addTab(self.solver_console_box, "Solver Console")

        self.outer_split = QSplitter(Qt.Vertical, self)
        self.outer_split.setHandleWidth(8)
        self.outer_split.addWidget(self.main_split)
        self.outer_split.addWidget(self.bottom_tabs)
        self.outer_split.setStretchFactor(0, 8)
        self.outer_split.setStretchFactor(1, 1)
        root.addWidget(self.outer_split, 1)
        self.main_split.setSizes([260, 1340, 320])
        self.outer_split.setSizes([900, 120])

        self.hint_label = QLabel("Hint: Click to select. Shift adds selection. RMB opens full context menu.")
        root.addWidget(self.hint_label)
        self._apply_button_icons()
        self._register_shortcuts()

    def _build_ribbon_tabs(self) -> QTabWidget:
        tabs = QTabWidget(self)
        tabs.setDocumentMode(True)

        def _tab() -> tuple[QWidget, QHBoxLayout]:
            w = QWidget(self)
            lay = QHBoxLayout(w)
            lay.setContentsMargins(4, 4, 4, 4)
            lay.setSpacing(6)
            return w, lay

        def _group(title: str, parent: QWidget) -> tuple[QGroupBox, QHBoxLayout]:
            box = QGroupBox(title, parent)
            lay = QHBoxLayout(box)
            lay.setContentsMargins(6, 4, 6, 4)
            lay.setSpacing(4)
            return box, lay

        # Project
        w_proj, lay_proj = _tab()
        g_file, g_file_lay = _group("Projeto", w_proj)
        self.btn_import = QPushButton("Import", g_file)
        self.btn_import.clicked.connect(self.action_import_mesh)
        g_file_lay.addWidget(self.btn_import)
        self.btn_snapshot = QPushButton("Screenshot", g_file)
        self.btn_snapshot.clicked.connect(self.action_screenshot)
        g_file_lay.addWidget(self.btn_snapshot)
        self.btn_markers = QPushButton("Save Markers", g_file)
        self.btn_markers.clicked.connect(self._save_markers)
        g_file_lay.addWidget(self.btn_markers)
        self.btn_load_markers = QPushButton("Load Markers", g_file)
        self.btn_load_markers.clicked.connect(self._load_markers)
        g_file_lay.addWidget(self.btn_load_markers)

        g_layout, g_layout_lay = _group("Layout", w_proj)
        self.layout_combo = QComboBox(g_layout)
        self.layout_combo.addItems(["layout_default", "layout_modeling", "layout_fem", "layout_results", "layout_analysis"])
        self.btn_layout_save = QPushButton("Save Layout", g_layout)
        self.btn_layout_save.clicked.connect(self._save_layout_from_combo)
        self.btn_layout_load = QPushButton("Load Layout", g_layout)
        self.btn_layout_load.clicked.connect(self._load_layout_from_combo)
        self.btn_layout_reset = QPushButton("Reset", g_layout)
        self.btn_layout_reset.clicked.connect(self.reset_layout_default)
        g_layout_lay.addWidget(QLabel("Profile"))
        self.profile_combo = QComboBox(g_layout)
        self.profile_combo.addItems(["CAD Classic", "Minimal", "Analysis"])
        self.profile_combo.currentTextChanged.connect(self._on_profile_changed)
        g_layout_lay.addWidget(self.profile_combo)
        g_layout_lay.addWidget(QLabel("Layout"))
        g_layout_lay.addWidget(self.layout_combo)
        g_layout_lay.addWidget(self.btn_layout_save)
        g_layout_lay.addWidget(self.btn_layout_load)
        g_layout_lay.addWidget(self.btn_layout_reset)

        g_import, g_import_lay = _group("Import Setup", w_proj)
        self.import_deflection_edit = QLineEdit(f"{self._mesh_deflection:.6g}", g_import)
        self.import_deflection_edit.setMaximumWidth(84)
        self.btn_retess = QPushButton("Re-tessellate Mesh", g_import)
        self.btn_retess.clicked.connect(self.action_retessellate_selected)
        g_import_lay.addWidget(QLabel("Mesh defl"))
        g_import_lay.addWidget(self.import_deflection_edit)
        g_import_lay.addWidget(self.btn_retess)

        g_pref, g_pref_lay = _group("Preferences", w_proj)
        g_pref_lay.addWidget(QLabel("Units"))
        self.units_combo = QComboBox(g_pref)
        self.units_combo.addItems(["mm", "cm", "m", "in"])
        self.units_combo.currentTextChanged.connect(self._on_units_changed)
        g_pref_lay.addWidget(self.units_combo)
        self.chk_snap = QCheckBox("Snap Grid", g_pref)
        self.chk_snap.toggled.connect(lambda v: self._set_snap_enabled(v))
        g_pref_lay.addWidget(self.chk_snap)
        self.btn_pref = QPushButton("Preferences", g_pref)
        self.btn_pref.clicked.connect(self.action_open_preferences)
        g_pref_lay.addWidget(self.btn_pref)
        self.btn_doctor = QPushButton("Doctor", g_pref)
        self.btn_doctor.clicked.connect(self.action_backend_diagnostics)
        g_pref_lay.addWidget(self.btn_doctor)
        self.main_menu_btn = QToolButton(g_pref)
        self.main_menu_btn.setText("Menu")
        self.main_menu_btn.setPopupMode(QToolButton.InstantPopup)
        self.main_menu_btn.setMenu(self._build_main_dropdown_menu())
        g_pref_lay.addWidget(self.main_menu_btn)

        lay_proj.addWidget(g_file)
        lay_proj.addWidget(g_layout)
        lay_proj.addWidget(g_import)
        lay_proj.addWidget(g_pref)
        lay_proj.addStretch(1)
        tabs.addTab(w_proj, "Project")

        # Edit
        w_edit, lay_edit = _tab()
        g_hist, g_hist_lay = _group("Edit", w_edit)
        self.btn_undo = QPushButton("Undo", g_hist)
        self.btn_redo = QPushButton("Redo", g_hist)
        self.btn_undo.clicked.connect(self.action_undo)
        self.btn_redo.clicked.connect(self.action_redo)
        btn_dup = QPushButton("Duplicate", g_hist)
        btn_dup.clicked.connect(lambda: self.action_duplicate(self._pick_selected()))
        btn_rename = QPushButton("Rename", g_hist)
        btn_rename.clicked.connect(self.action_rename_selected)
        btn_del = QPushButton("Delete", g_hist)
        btn_del.clicked.connect(self.action_delete_selected)
        btn_group = QPushButton("Group", g_hist)
        btn_group.clicked.connect(self.action_group_selected)
        btn_ungroup = QPushButton("Ungroup", g_hist)
        btn_ungroup.clicked.connect(self.action_ungroup_selected)
        g_hist_lay.addWidget(self.btn_undo)
        g_hist_lay.addWidget(self.btn_redo)
        g_hist_lay.addWidget(btn_dup)
        g_hist_lay.addWidget(btn_rename)
        g_hist_lay.addWidget(btn_del)
        g_hist_lay.addWidget(btn_group)
        g_hist_lay.addWidget(btn_ungroup)
        lay_edit.addWidget(g_hist)
        lay_edit.addStretch(1)
        tabs.addTab(w_edit, "Edit")

        # View
        w_view, lay_view = _tab()
        g_view, g_view_lay = _group("View", w_view)
        self.view_mode_combo = QComboBox(g_view)
        self.view_mode_combo.addItems(["Solid+Edges", "Solid", "Wireframe", "X-Ray"])
        self.view_mode_combo.currentTextChanged.connect(self.action_set_visual_preset)
        g_view_lay.addWidget(QLabel("Render"))
        g_view_lay.addWidget(self.view_mode_combo)
        btn_reset = QPushButton("Reset", g_view)
        btn_reset.clicked.connect(self.action_reset_view)
        btn_fit = QPushButton("Fit All", g_view)
        btn_fit.clicked.connect(self.action_fit_all)
        btn_focus_detail = QPushButton("Detail Focus", g_view)
        btn_focus_detail.clicked.connect(self.action_focus_selection_detail)
        btn_grid = QPushButton("Grid", g_view)
        btn_grid.clicked.connect(self.action_toggle_grid)
        btn_axes = QPushButton("Axes", g_view)
        btn_axes.clicked.connect(self.action_toggle_axes)
        btn_sec = QPushButton("Section XY", g_view)
        btn_sec.clicked.connect(lambda: self.action_add_clipping_plane("xy"))
        g_view_lay.addWidget(btn_reset)
        g_view_lay.addWidget(btn_fit)
        g_view_lay.addWidget(btn_focus_detail)
        g_view_lay.addWidget(btn_grid)
        g_view_lay.addWidget(btn_axes)
        g_view_lay.addWidget(btn_sec)
        lay_view.addWidget(g_view)
        lay_view.addStretch(1)
        tabs.addTab(w_view, "View")

        # Selection
        w_sel, lay_sel = _tab()
        g_sel, g_sel_lay = _group("Selection", w_sel)
        self.subselect_combo = QComboBox(g_sel)
        self.subselect_combo.addItems(["Object", "Face", "Edge", "Vertex", "Body", "Component"])
        self.subselect_combo.currentTextChanged.connect(self.action_set_subselection_mode)
        self.auto_analysis_chk = QCheckBox("Auto analysis tab", g_sel)
        self.auto_analysis_chk.toggled.connect(self._on_auto_analysis_toggle)
        self.btn_analysis_tab = QPushButton("Analyze Selected", g_sel)
        self.btn_analysis_tab.clicked.connect(self.action_open_selected_analysis_tab)
        btn_all = QPushButton("Select All", g_sel)
        btn_none = QPushButton("Select None", g_sel)
        btn_inv = QPushButton("Invert", g_sel)
        btn_iso = QPushButton("Isolate", g_sel)
        btn_show_all = QPushButton("Show All", g_sel)
        btn_all.clicked.connect(self.action_select_all)
        btn_none.clicked.connect(self.action_select_none)
        btn_inv.clicked.connect(self.action_select_invert)
        btn_iso.clicked.connect(self.action_show_only_selected)
        btn_show_all.clicked.connect(self.action_show_all)
        g_sel_lay.addWidget(QLabel("Mode"))
        g_sel_lay.addWidget(self.subselect_combo)
        g_sel_lay.addWidget(btn_all)
        g_sel_lay.addWidget(btn_none)
        g_sel_lay.addWidget(btn_inv)
        g_sel_lay.addWidget(btn_iso)
        g_sel_lay.addWidget(btn_show_all)
        g_sel_lay.addWidget(self.auto_analysis_chk)
        g_sel_lay.addWidget(self.btn_analysis_tab)
        lay_sel.addWidget(g_sel)
        lay_sel.addStretch(1)
        tabs.addTab(w_sel, "Selection")

        # Modeling
        w_model, lay_model = _tab()
        g_prim, g_prim_lay = _group("Modeling", w_model)
        for label, kind in [("Box", "box"), ("Cylinder", "cylinder"), ("Sphere", "sphere"), ("Cone", "cone"), ("Tube", "tube"), ("Plane", "plane")]:
            b = QPushButton(label, g_prim)
            b.clicked.connect(lambda _checked=False, k=kind: self.action_create_primitive_dialog(k))
            g_prim_lay.addWidget(b)
        btn_move = QPushButton("Move", g_prim)
        btn_move.clicked.connect(lambda: self.action_transform_dialog("move", self._pick_selected()))
        btn_rotate = QPushButton("Rotate", g_prim)
        btn_rotate.clicked.connect(lambda: self.action_transform_dialog("rotate", self._pick_selected()))
        btn_scale = QPushButton("Scale", g_prim)
        btn_scale.clicked.connect(lambda: self.action_transform_dialog("scale", self._pick_selected()))
        btn_union = QPushButton("Union", g_prim)
        btn_union.clicked.connect(self.action_boolean_union)
        btn_sub = QPushButton("Subtract", g_prim)
        btn_sub.clicked.connect(self.action_boolean_subtract)
        btn_int = QPushButton("Intersect", g_prim)
        btn_int.clicked.connect(self.action_boolean_intersect)
        g_prim_lay.addWidget(btn_move)
        g_prim_lay.addWidget(btn_rotate)
        g_prim_lay.addWidget(btn_scale)
        g_prim_lay.addWidget(btn_union)
        g_prim_lay.addWidget(btn_sub)
        g_prim_lay.addWidget(btn_int)
        lay_model.addWidget(g_prim)
        lay_model.addStretch(1)
        tabs.addTab(w_model, "Modeling")

        # FEM
        w_fem, lay_fem = _tab()
        g_fem, g_fem_lay = _group("FEM", w_fem)
        btn_new_study = QPushButton("New Study", g_fem)
        btn_new_study.clicked.connect(self.action_fem_new_study)
        btn_include = QPushButton("Include Bodies", g_fem)
        btn_include.clicked.connect(self.action_fem_include_selected)
        btn_mat = QPushButton("Assign Material", g_fem)
        btn_mat.clicked.connect(self.action_fem_assign_material_selected)
        btn_bc = QPushButton("Add Fixed BC", g_fem)
        btn_bc.clicked.connect(lambda: self.action_fem_add_bc("fixed support"))
        btn_load = QPushButton("Add Force Load", g_fem)
        btn_load.clicked.connect(lambda: self.action_fem_add_load("force"))
        btn_mesh = QPushButton("Generate Mesh", g_fem)
        btn_mesh.clicked.connect(self.action_fem_generate_mesh)
        btn_validate = QPushButton("Validate", g_fem)
        btn_validate.clicked.connect(self.action_fem_validate_study)
        btn_run = QPushButton("Run", g_fem)
        btn_run.clicked.connect(self.action_fem_run_solve)
        btn_results = QPushButton("Results", g_fem)
        btn_results.clicked.connect(self.action_fem_refresh_results)
        g_fem_lay.addWidget(btn_new_study)
        g_fem_lay.addWidget(btn_include)
        g_fem_lay.addWidget(btn_mat)
        g_fem_lay.addWidget(btn_bc)
        g_fem_lay.addWidget(btn_load)
        g_fem_lay.addWidget(btn_mesh)
        g_fem_lay.addWidget(btn_validate)
        g_fem_lay.addWidget(btn_run)
        g_fem_lay.addWidget(btn_results)
        lay_fem.addWidget(g_fem)
        lay_fem.addStretch(1)
        tabs.addTab(w_fem, "FEM")

        # Import / Export
        w_io, lay_io = _tab()
        g_io, g_io_lay = _group("Import / Export", w_io)
        b_import = QPushButton("Import STEP/IGES/STL/OBJ", g_io)
        b_import.clicked.connect(self.action_import_mesh)
        b_step = QPushButton("Export STEP", g_io)
        b_stl = QPushButton("Export STL", g_io)
        b_obj = QPushButton("Export OBJ", g_io)
        b_report = QPushButton("Export Report", g_io)
        b_import.clicked.connect(self.action_import_mesh)
        b_step.clicked.connect(lambda: self.action_export_selected("step"))
        b_stl.clicked.connect(lambda: self.action_export_selected("stl"))
        b_obj.clicked.connect(lambda: self.action_export_selected("obj"))
        b_report.clicked.connect(self.action_save_backend_report)
        g_io_lay.addWidget(b_import)
        g_io_lay.addWidget(b_step)
        g_io_lay.addWidget(b_stl)
        g_io_lay.addWidget(b_obj)
        g_io_lay.addWidget(b_report)
        lay_io.addWidget(g_io)
        lay_io.addStretch(1)
        tabs.addTab(w_io, "Import / Export")

        return tabs

    def _build_components_library_tab(self) -> QWidget:
        w = QWidget(self)
        lay = QVBoxLayout(w)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(4)

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(4)
        self.components_search_edit = QLineEdit(w)
        self.components_search_edit.setPlaceholderText("Filter components: name, type, layer, source...")
        self.components_search_edit.textChanged.connect(self._filter_components_library)
        btn_focus = QPushButton("Focus", w)
        btn_focus.clicked.connect(self._on_components_imported_activated)
        btn_clear = QPushButton("Clear", w)
        btn_clear.clicked.connect(self.components_search_edit.clear)
        self.components_status_label = QLabel("Palette")
        self.components_status_label.setStyleSheet("color:#6f8297; font-size:11px;")
        top.addWidget(self.components_search_edit, 1)
        top.addWidget(btn_focus)
        top.addWidget(btn_clear)
        lay.addLayout(top)
        lay.addWidget(self.components_status_label)

        self.components_tabs = QTabWidget(w)
        self.components_tabs.setDocumentMode(True)

        primitives = QWidget(w)
        p_lay = QVBoxLayout(primitives)
        p_lay.setContentsMargins(4, 4, 4, 4)
        p_lay.addWidget(QLabel("Primitives"))
        self.components_primitives_list = QListWidget(primitives)
        self._style_palette_list(self.components_primitives_list)
        for text, kind in [
            ("Box", "box"),
            ("Cylinder", "cylinder"),
            ("Cone", "cone"),
            ("Sphere", "sphere"),
            ("Torus", "torus"),
            ("Plane", "plane"),
            ("Tube/Pipe", "tube"),
            ("Polygon", "polygon"),
            ("Wedge", "wedge"),
        ]:
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, {"kind": kind})
            icon = self._style_icon(QStyle.SP_FileDialogNewFolder)
            if icon is not None:
                item.setIcon(icon)
            self.components_primitives_list.addItem(item)
        self.components_primitives_list.itemDoubleClicked.connect(lambda _it: self.action_components_insert_primitive())
        p_lay.addWidget(self.components_primitives_list)
        btn_primitive_insert = QPushButton("Insert Primitive", primitives)
        btn_primitive_insert.clicked.connect(self.action_components_insert_primitive)
        p_lay.addWidget(btn_primitive_insert)

        sketches = QWidget(w)
        s_lay = QVBoxLayout(sketches)
        s_lay.setContentsMargins(4, 4, 4, 4)
        s_lay.addWidget(QLabel("Sketch / Profiles"))
        self.components_sketch_list = QListWidget(sketches)
        self._style_palette_list(self.components_sketch_list)
        for text in ["Line", "Rectangle", "Circle", "Arc", "Polyline", "Spline"]:
            item = QListWidgetItem(text)
            icon = self._style_icon(QStyle.SP_DialogApplyButton)
            if icon is not None:
                item.setIcon(icon)
            self.components_sketch_list.addItem(item)
        s_lay.addWidget(self.components_sketch_list)
        btn_sketch = QPushButton("Create Sketch Tool", sketches)
        btn_sketch.clicked.connect(lambda: self._set_hint("Sketch tools will be expanded in phase 2."))
        s_lay.addWidget(btn_sketch)

        imported = QWidget(w)
        i_lay = QVBoxLayout(imported)
        i_lay.setContentsMargins(4, 4, 4, 4)
        i_lay.addWidget(QLabel("Imported Components"))
        self.components_imported_list = QListWidget(imported)
        self._style_palette_list(self.components_imported_list)
        self.components_imported_list.itemDoubleClicked.connect(self._on_components_imported_activated)
        i_lay.addWidget(self.components_imported_list)
        row_imp = QHBoxLayout()
        btn_import = QPushButton("Import CAD/Mesh", imported)
        btn_import.clicked.connect(self.action_import_mesh)
        btn_focus_import = QPushButton("Select + Focus", imported)
        btn_focus_import.clicked.connect(self._on_components_imported_activated)
        row_imp.addWidget(btn_import)
        row_imp.addWidget(btn_focus_import)
        i_lay.addLayout(row_imp)

        instances = QWidget(w)
        in_lay = QVBoxLayout(instances)
        in_lay.setContentsMargins(4, 4, 4, 4)
        in_lay.addWidget(QLabel("Instances & Assemblies"))
        self.components_instances_list = QListWidget(instances)
        self._style_palette_list(self.components_instances_list)
        self.components_instances_list.itemDoubleClicked.connect(self._on_components_instance_activated)
        in_lay.addWidget(self.components_instances_list)
        btn_make_comp = QPushButton("Create Component from Selection", instances)
        btn_make_comp.clicked.connect(self.action_components_create_from_selection)
        btn_insert_inst = QPushButton("Insert Instance", instances)
        btn_insert_inst.clicked.connect(self.action_components_insert_instance)
        btn_explode = QPushButton("Explode Assembly", instances)
        btn_explode.clicked.connect(lambda: self._set_hint("Explode assembly: convert instances into editable bodies."))
        btn_focus_inst = QPushButton("Select + Focus", instances)
        btn_focus_inst.clicked.connect(self._on_components_instance_activated)
        in_lay.addWidget(btn_make_comp)
        in_lay.addWidget(btn_insert_inst)
        in_lay.addWidget(btn_explode)
        in_lay.addWidget(btn_focus_inst)

        templates = QWidget(w)
        t_lay = QVBoxLayout(templates)
        t_lay.setContentsMargins(4, 4, 4, 4)
        t_lay.addWidget(QLabel("Project Templates"))
        self.components_templates_list = QListWidget(templates)
        self._style_palette_list(self.components_templates_list)
        for label in [
            "Panel + support",
            "Tower + braces",
            "Cable set",
            "Irradiating system (simplified)",
            "Base + mast + panels",
        ]:
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, {"template": str(label).strip().lower()})
            icon = self._style_icon(QStyle.SP_DirIcon)
            if icon is not None:
                item.setIcon(icon)
            self.components_templates_list.addItem(item)
        self.components_templates_list.itemDoubleClicked.connect(lambda _it: self.action_components_insert_template())
        t_lay.addWidget(self.components_templates_list)
        btn_template = QPushButton("Insert Template", templates)
        btn_template.clicked.connect(self.action_components_insert_template)
        t_lay.addWidget(btn_template)

        self.components_tabs.addTab(primitives, "Primitives")
        self.components_tabs.addTab(sketches, "Sketch")
        self.components_tabs.addTab(imported, "Imported")
        self.components_tabs.addTab(instances, "Instances")
        self.components_tabs.addTab(templates, "Templates")
        lay.addWidget(self.components_tabs, 1)
        self._filter_components_library()
        return w

    def _style_palette_list(self, widget: QListWidget):
        if widget is None:
            return
        widget.setAlternatingRowColors(True)
        widget.setUniformItemSizes(True)
        widget.setSelectionMode(QAbstractItemView.SingleSelection)
        widget.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        widget.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        widget.setSpacing(2)
        widget.setStyleSheet(
            "QListWidget{background:#121923; color:#e6effa; border:1px solid #2f3d4e; border-radius:6px;}"
            "QListWidget::item{padding:6px 6px; border-radius:4px;}"
            "QListWidget::item:selected{background:#295171; color:#f5fbff;}"
            "QListWidget::item:hover{background:#1d2f43;}"
            "QListWidget::item:alternate{background:#16202c;}"
        )

    def _filter_components_library(self):
        token = str(getattr(self, "components_search_edit", None).text() if hasattr(self, "components_search_edit") else "").strip().lower()
        buckets = [
            ("components_primitives_list", "Primitives"),
            ("components_sketch_list", "Sketch"),
            ("components_imported_list", "Imported"),
            ("components_instances_list", "Instances"),
            ("components_templates_list", "Templates"),
        ]
        rows = []
        for attr, label in buckets:
            widget = getattr(self, attr, None)
            if widget is None:
                continue
            total = widget.count()
            visible = 0
            for idx in range(total):
                item = widget.item(idx)
                hay = str(item.text() or "").lower()
                tip = str(item.toolTip() or "").lower()
                data = item.data(Qt.UserRole)
                extra = ""
                if isinstance(data, dict):
                    try:
                        extra = json.dumps(data, ensure_ascii=False).lower()
                    except Exception:
                        extra = str(data).lower()
                elif data is not None:
                    extra = str(data).lower()
                show = (not token) or token in hay or token in tip or token in extra
                item.setHidden(not show)
                if show:
                    visible += 1
            rows.append(f"{label}:{visible}/{total}")
        if hasattr(self, "components_status_label"):
            if token:
                self.components_status_label.setText(f"Filter='{token}'  " + " | ".join(rows))
            else:
                self.components_status_label.setText(" | ".join(rows))

    def _on_components_imported_activated(self, _item: Optional[QListWidgetItem] = None):
        widget = getattr(self, "components_imported_list", None)
        if widget is None:
            return
        row = _item if isinstance(_item, QListWidgetItem) else widget.currentItem()
        if row is None:
            return
        data = row.data(Qt.UserRole) or {}
        ids = [str(x) for x in data.get("ids", []) if str(x) in self.engine.objects]
        if not ids:
            return
        self.engine.select(ids, mode="replace")
        QTimer.singleShot(0, lambda q=list(ids): self.viewport.focus_on_ids(q, detail=(len(q) == 1)))
        self._set_hint(f"Imported selection: {len(ids)} body(s).")

    def _on_components_instance_activated(self, _item: Optional[QListWidgetItem] = None):
        widget = getattr(self, "components_instances_list", None)
        if widget is None:
            return
        row = _item if isinstance(_item, QListWidgetItem) else widget.currentItem()
        if row is None:
            return
        data = row.data(Qt.UserRole) or {}
        ids = [str(x) for x in data.get("ids", []) if str(x) in self.engine.objects]
        if not ids:
            return
        self.engine.select(ids, mode="replace")
        QTimer.singleShot(0, lambda q=list(ids): self.viewport.focus_on_ids(q, detail=(len(q) == 1)))
        self._set_hint(f"Component instances selected: {len(ids)}.")

    def _build_layers_overview_tab(self) -> QWidget:
        w = QWidget(self)
        lay = QVBoxLayout(w)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(4)
        lay.addWidget(QLabel("Layers Overview"))
        self.layers_overview_tree = QTreeWidget(w)
        self.layers_overview_tree.setHeaderLabels(["Layer", "Visible", "Locked", "Objects", "Color"])
        self.layers_overview_tree.setAlternatingRowColors(True)
        self.layers_overview_tree.setRootIsDecorated(False)
        self.layers_overview_tree.setUniformRowHeights(True)
        self.layers_overview_tree.setStyleSheet(
            "QTreeWidget{background:#121923; color:#e6effa; border:1px solid #2f3d4e; border-radius:6px;}"
            "QTreeWidget::item{padding:4px 6px;}"
            "QTreeWidget::item:selected{background:#295171; color:#f5fbff;}"
        )
        lay.addWidget(self.layers_overview_tree, 1)
        row = QHBoxLayout()
        btn_new = QPushButton("New", w)
        btn_new.clicked.connect(self.action_new_layer)
        btn_rename = QPushButton("Rename", w)
        btn_rename.clicked.connect(self.action_rename_layer)
        btn_delete = QPushButton("Delete", w)
        btn_delete.clicked.connect(self.action_delete_layer)
        btn_toggle = QPushButton("Toggle Visibility", w)
        btn_toggle.clicked.connect(self.action_toggle_layer_visibility)
        btn_color = QPushButton("Color", w)
        btn_color.clicked.connect(self.action_set_layer_color)
        row.addWidget(btn_new)
        row.addWidget(btn_rename)
        row.addWidget(btn_delete)
        row.addWidget(btn_toggle)
        row.addWidget(btn_color)
        lay.addLayout(row)
        return w

    def _build_studies_tab(self) -> QWidget:
        w = QWidget(self)
        lay = QVBoxLayout(w)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(4)
        lay.addWidget(QLabel("FEM Studies"))
        self.studies_list = QListWidget(w)
        self._style_palette_list(self.studies_list)
        lay.addWidget(self.studies_list, 1)
        row = QHBoxLayout()
        btn_new = QPushButton("New Study", w)
        btn_new.clicked.connect(self.action_fem_new_study)
        btn_del = QPushButton("Delete", w)
        btn_del.clicked.connect(self.action_fem_remove_study)
        btn_set = QPushButton("Set Active", w)
        btn_set.clicked.connect(self.action_fem_set_active_from_list)
        row.addWidget(btn_new)
        row.addWidget(btn_del)
        row.addWidget(btn_set)
        lay.addLayout(row)
        return w

    def _build_right_transform_tab(self) -> QWidget:
        w = QWidget(self)
        lay = QVBoxLayout(w)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(4)
        lay.addWidget(QLabel("Quick Transform"))
        grid = QGridLayout()
        self.rt_tx = QLineEdit("0.0", w)
        self.rt_ty = QLineEdit("0.0", w)
        self.rt_tz = QLineEdit("0.0", w)
        self.rt_rx = QLineEdit("0.0", w)
        self.rt_ry = QLineEdit("0.0", w)
        self.rt_rz = QLineEdit("0.0", w)
        self.rt_scale = QLineEdit("1.0", w)
        grid.addWidget(QLabel("Tx"), 0, 0)
        grid.addWidget(self.rt_tx, 0, 1)
        grid.addWidget(QLabel("Ty"), 0, 2)
        grid.addWidget(self.rt_ty, 0, 3)
        grid.addWidget(QLabel("Tz"), 0, 4)
        grid.addWidget(self.rt_tz, 0, 5)
        grid.addWidget(QLabel("Rx"), 1, 0)
        grid.addWidget(self.rt_rx, 1, 1)
        grid.addWidget(QLabel("Ry"), 1, 2)
        grid.addWidget(self.rt_ry, 1, 3)
        grid.addWidget(QLabel("Rz"), 1, 4)
        grid.addWidget(self.rt_rz, 1, 5)
        grid.addWidget(QLabel("Scale"), 2, 0)
        grid.addWidget(self.rt_scale, 2, 1)
        lay.addLayout(grid)
        row = QHBoxLayout()
        btn_apply = QPushButton("Apply", w)
        btn_apply.clicked.connect(self.action_apply_right_transform)
        btn_reset = QPushButton("Reset", w)
        btn_reset.clicked.connect(lambda: [self.rt_tx.setText("0.0"), self.rt_ty.setText("0.0"), self.rt_tz.setText("0.0"), self.rt_rx.setText("0.0"), self.rt_ry.setText("0.0"), self.rt_rz.setText("0.0"), self.rt_scale.setText("1.0")])
        row.addWidget(btn_apply)
        row.addWidget(btn_reset)
        lay.addLayout(row)
        lay.addStretch(1)
        return w

    def _build_right_material_tab(self) -> QWidget:
        w = QWidget(self)
        lay = QVBoxLayout(w)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(4)
        self.material_name_combo = QComboBox(w)
        self.material_name_combo.addItems(["Steel", "Aluminum", "Copper", "FR4", "Custom"])
        self.material_density_edit = QLineEdit("7850", w)
        self.material_e_edit = QLineEdit("210e9", w)
        self.material_nu_edit = QLineEdit("0.30", w)
        form = QFormLayout()
        form.addRow("Material", self.material_name_combo)
        form.addRow("Density [kg/m3]", self.material_density_edit)
        form.addRow("E [Pa]", self.material_e_edit)
        form.addRow("nu", self.material_nu_edit)
        lay.addLayout(form)
        btn_assign = QPushButton("Assign to Selection", w)
        btn_assign.clicked.connect(self.action_fem_assign_material_selected)
        lay.addWidget(btn_assign)
        lay.addStretch(1)
        return w

    def _build_right_fem_study_tab(self) -> QWidget:
        w = QWidget(self)
        lay = QVBoxLayout(w)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(4)
        self.fem_study_combo = QComboBox(w)
        self.fem_study_combo.currentIndexChanged.connect(self.action_fem_set_active_from_combo)
        self.fem_study_type_combo = QComboBox(w)
        self.fem_study_type_combo.addItems(["Structural Static", "Modal", "Harmonic", "Buckling", "Thermal"])
        self.fem_role_combo = QComboBox(w)
        self.fem_role_combo.addItems(["solid", "shell", "beam", "reference"])
        self.fem_units_combo = QComboBox(w)
        self.fem_units_combo.addItems(["mm", "cm", "m", "in"])
        form = QFormLayout()
        form.addRow("Study", self.fem_study_combo)
        form.addRow("Type", self.fem_study_type_combo)
        form.addRow("Default FEM Role", self.fem_role_combo)
        form.addRow("Units", self.fem_units_combo)
        lay.addLayout(form)
        row = QHBoxLayout()
        btn_new = QPushButton("New Study", w)
        btn_new.clicked.connect(self.action_fem_new_study)
        btn_include = QPushButton("Include Selection", w)
        btn_include.clicked.connect(self.action_fem_include_selected)
        row.addWidget(btn_new)
        row.addWidget(btn_include)
        lay.addLayout(row)
        self.fem_study_summary = QTextEdit(w)
        self.fem_study_summary.setReadOnly(True)
        self.fem_study_summary.setMinimumHeight(120)
        lay.addWidget(self.fem_study_summary, 1)
        return w

    def _build_right_bcs_tab(self) -> QWidget:
        w = QWidget(self)
        lay = QVBoxLayout(w)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(4)
        self.fem_bc_type_combo = QComboBox(w)
        self.fem_bc_type_combo.addItems(["fixed support", "displacement", "symmetry", "remote support", "pin"])
        self.fem_bc_value_edit = QLineEdit("0.0", w)
        form = QFormLayout()
        form.addRow("BC Type", self.fem_bc_type_combo)
        form.addRow("Value", self.fem_bc_value_edit)
        lay.addLayout(form)
        row = QHBoxLayout()
        btn_add = QPushButton("Add BC", w)
        btn_add.clicked.connect(lambda: self.action_fem_add_bc(str(self.fem_bc_type_combo.currentText() or "fixed support")))
        btn_sync = QPushButton("Sync from Boundaries", w)
        btn_sync.clicked.connect(self.action_fem_sync_boundaries_as_bcs)
        row.addWidget(btn_add)
        row.addWidget(btn_sync)
        lay.addLayout(row)
        self.fem_bcs_box = QTextEdit(w)
        self.fem_bcs_box.setReadOnly(True)
        lay.addWidget(self.fem_bcs_box, 1)
        return w

    def _build_right_loads_tab(self) -> QWidget:
        w = QWidget(self)
        lay = QVBoxLayout(w)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(4)
        self.fem_load_type_combo = QComboBox(w)
        self.fem_load_type_combo.addItems(["force", "pressure", "gravity", "moment", "acceleration", "wind load"])
        self.fem_load_value_edit = QLineEdit("1.0", w)
        self.fem_load_dir_combo = QComboBox(w)
        self.fem_load_dir_combo.addItems(["normal", "x", "y", "z"])
        form = QFormLayout()
        form.addRow("Load Type", self.fem_load_type_combo)
        form.addRow("Value", self.fem_load_value_edit)
        form.addRow("Direction", self.fem_load_dir_combo)
        lay.addLayout(form)
        btn_add = QPushButton("Add Load", w)
        btn_add.clicked.connect(lambda: self.action_fem_add_load(str(self.fem_load_type_combo.currentText() or "force")))
        lay.addWidget(btn_add)
        self.fem_loads_box = QTextEdit(w)
        self.fem_loads_box.setReadOnly(True)
        lay.addWidget(self.fem_loads_box, 1)
        return w

    def _build_right_mesh_tab(self) -> QWidget:
        w = QWidget(self)
        lay = QVBoxLayout(w)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(4)
        self.fem_mesh_size_edit = QLineEdit("20.0", w)
        self.fem_mesh_growth_edit = QLineEdit("1.2", w)
        self.fem_mesh_quality_edit = QLineEdit("0.7", w)
        self.fem_mesh_curvature_chk = QCheckBox("Curvature refinement", w)
        self.fem_mesh_curvature_chk.setChecked(True)
        form = QFormLayout()
        form.addRow("Global size", self.fem_mesh_size_edit)
        form.addRow("Growth rate", self.fem_mesh_growth_edit)
        form.addRow("Quality target", self.fem_mesh_quality_edit)
        lay.addLayout(form)
        lay.addWidget(self.fem_mesh_curvature_chk)
        row = QHBoxLayout()
        btn_cfg = QPushButton("Apply Mesh Config", w)
        btn_cfg.clicked.connect(self.action_fem_apply_mesh_config)
        btn_gen = QPushButton("Generate Mesh", w)
        btn_gen.clicked.connect(self.action_fem_generate_mesh)
        row.addWidget(btn_cfg)
        row.addWidget(btn_gen)
        lay.addLayout(row)
        self.fem_mesh_box = QTextEdit(w)
        self.fem_mesh_box.setReadOnly(True)
        lay.addWidget(self.fem_mesh_box, 1)
        return w

    def _build_right_solve_tab(self) -> QWidget:
        w = QWidget(self)
        lay = QVBoxLayout(w)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(4)
        self.fem_solver_type_combo = QComboBox(w)
        self.fem_solver_type_combo.addItems(["direct", "iterative"])
        self.fem_solver_tol_edit = QLineEdit("1e-6", w)
        self.fem_solver_it_edit = QLineEdit("500", w)
        self.fem_solver_threads_edit = QLineEdit("0", w)
        form = QFormLayout()
        form.addRow("Solver", self.fem_solver_type_combo)
        form.addRow("Tolerance", self.fem_solver_tol_edit)
        form.addRow("Max iterations", self.fem_solver_it_edit)
        form.addRow("Threads", self.fem_solver_threads_edit)
        lay.addLayout(form)
        row = QHBoxLayout()
        btn_cfg = QPushButton("Apply Solver Config", w)
        btn_cfg.clicked.connect(self.action_fem_apply_solver_config)
        btn_validate = QPushButton("Validate", w)
        btn_validate.clicked.connect(self.action_fem_validate_study)
        btn_run = QPushButton("Run Solve", w)
        btn_run.clicked.connect(self.action_fem_run_solve)
        row.addWidget(btn_cfg)
        row.addWidget(btn_validate)
        row.addWidget(btn_run)
        lay.addLayout(row)
        self.fem_validation_box = QTextEdit(w)
        self.fem_validation_box.setReadOnly(True)
        lay.addWidget(self.fem_validation_box, 1)
        return w

    def _build_right_results_tab(self) -> QWidget:
        w = QWidget(self)
        lay = QVBoxLayout(w)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(4)
        row = QHBoxLayout()
        btn_refresh = QPushButton("Refresh Results", w)
        btn_refresh.clicked.connect(self.action_fem_refresh_results)
        row.addWidget(btn_refresh)
        lay.addLayout(row)
        self.fem_results_box = QTextEdit(w)
        self.fem_results_box.setReadOnly(True)
        lay.addWidget(self.fem_results_box, 1)
        return w

    def _style_icon(self, pixmap_name: QStyle.StandardPixmap):
        try:
            return self.style().standardIcon(pixmap_name)
        except Exception:
            return None

    def _apply_button_icons(self):
        icon_map = [
            (getattr(self, "btn_import", None), QStyle.SP_DialogOpenButton),
            (getattr(self, "btn_retess", None), QStyle.SP_BrowserReload),
            (getattr(self, "btn_snapshot", None), QStyle.SP_DialogSaveButton),
            (getattr(self, "btn_markers", None), QStyle.SP_DriveFDIcon),
            (getattr(self, "btn_load_markers", None), QStyle.SP_DialogOpenButton),
            (getattr(self, "btn_undo", None), QStyle.SP_ArrowBack),
            (getattr(self, "btn_redo", None), QStyle.SP_ArrowForward),
            (getattr(self, "btn_pref", None), QStyle.SP_FileDialogDetailedView),
            (getattr(self, "btn_doctor", None), QStyle.SP_MessageBoxInformation),
            (getattr(self, "btn_layout_save", None), QStyle.SP_DialogSaveButton),
            (getattr(self, "btn_layout_load", None), QStyle.SP_DialogOpenButton),
            (getattr(self, "btn_layout_reset", None), QStyle.SP_BrowserReload),
            (getattr(self, "btn_analysis_tab", None), QStyle.SP_FileDialogContentsView),
            (getattr(self, "btn_toggle_left", None), QStyle.SP_ArrowLeft),
            (getattr(self, "btn_toggle_right", None), QStyle.SP_ArrowRight),
            (getattr(self, "btn_toggle_bottom", None), QStyle.SP_ArrowDown),
            (getattr(self, "btn_toggle_panels", None), QStyle.SP_TitleBarMaxButton),
        ]
        for btn, pix in icon_map:
            if btn is None:
                continue
            icon = self._style_icon(pix)
            if icon is not None:
                btn.setIcon(icon)
        if hasattr(self, "main_menu_btn"):
            icon = self._style_icon(QStyle.SP_TitleBarMenuButton)
            if icon is not None:
                self.main_menu_btn.setIcon(icon)

    def _build_main_dropdown_menu(self) -> QMenu:
        menu = QMenu(self)

        def _add(parent: QMenu, text: str, slot, pix: QStyle.StandardPixmap):
            act = QAction(text, parent)
            icon = self._style_icon(pix)
            if icon is not None:
                act.setIcon(icon)
            act.triggered.connect(slot)
            parent.addAction(act)
            return act

        proj_menu = menu.addMenu("Project")
        _add(proj_menu, "Import Mesh", self.action_import_mesh, QStyle.SP_DialogOpenButton)
        _add(proj_menu, "Save Layout", self._save_layout_from_combo, QStyle.SP_DialogSaveButton)
        _add(proj_menu, "Load Layout", self._load_layout_from_combo, QStyle.SP_DialogOpenButton)
        _add(proj_menu, "Reset Layout", self.reset_layout_default, QStyle.SP_BrowserReload)
        _add(proj_menu, "Preferences", self.action_open_preferences, QStyle.SP_FileDialogDetailedView)

        edit_menu = menu.addMenu("Edit")
        _add(edit_menu, "Undo", self.action_undo, QStyle.SP_ArrowBack)
        _add(edit_menu, "Redo", self.action_redo, QStyle.SP_ArrowForward)
        _add(edit_menu, "Duplicate", lambda: self.action_duplicate(self._pick_selected()), QStyle.SP_FileDialogNewFolder)
        _add(edit_menu, "Rename", self.action_rename_selected, QStyle.SP_FileDialogInfoView)
        _add(edit_menu, "Delete", self.action_delete_selected, QStyle.SP_TrashIcon)
        _add(edit_menu, "Group", self.action_group_selected, QStyle.SP_DirIcon)
        _add(edit_menu, "Ungroup", self.action_ungroup_selected, QStyle.SP_DirOpenIcon)

        view_menu = menu.addMenu("View")
        _add(view_menu, "Fit All", self.action_fit_all, QStyle.SP_DesktopIcon)
        _add(view_menu, "Detail Focus (Selection)", self.action_focus_selection_detail, QStyle.SP_ArrowUp)
        _add(view_menu, "Reset View", self.action_reset_view, QStyle.SP_BrowserReload)
        _add(view_menu, "Toggle Side Panels (Tab)", self.action_toggle_side_panels, QStyle.SP_TitleBarShadeButton)
        _add(view_menu, "Toggle Bottom Panel (Shift+Tab)", self.action_toggle_bottom_panel, QStyle.SP_TitleBarShadeButton)
        _add(view_menu, "Toggle Left Panel", self.action_toggle_left_panel, QStyle.SP_ArrowLeft)
        _add(view_menu, "Toggle Right Panel", self.action_toggle_right_panel, QStyle.SP_ArrowRight)
        _add(view_menu, "Max Viewport (F11)", self.action_toggle_panels, QStyle.SP_TitleBarMaxButton)
        _add(view_menu, "Toggle Grid", self.action_toggle_grid, QStyle.SP_DialogResetButton)
        _add(view_menu, "Toggle Axes", self.action_toggle_axes, QStyle.SP_DialogResetButton)
        _add(view_menu, "Toggle Snap", self.action_toggle_snap, QStyle.SP_DialogResetButton)
        _add(view_menu, "Toggle Navigation LOD", self.action_toggle_navigation_lod, QStyle.SP_DialogResetButton)
        _add(view_menu, "Solid + Edges", lambda: self.action_set_visual_preset("Solid+Edges"), QStyle.SP_DialogApplyButton)
        _add(view_menu, "Wireframe", lambda: self.action_set_visual_preset("Wireframe"), QStyle.SP_DialogApplyButton)
        _add(view_menu, "X-Ray", lambda: self.action_set_visual_preset("X-Ray"), QStyle.SP_DialogApplyButton)
        _add(view_menu, "Screenshot", self.action_screenshot, QStyle.SP_DialogSaveButton)

        sel_menu = menu.addMenu("Selection")
        _add(sel_menu, "Select All", self.action_select_all, QStyle.SP_DialogYesButton)
        _add(sel_menu, "Select None", self.action_select_none, QStyle.SP_DialogNoButton)
        _add(sel_menu, "Invert", self.action_select_invert, QStyle.SP_BrowserReload)
        _add(sel_menu, "Isolate Selection", self.action_show_only_selected, QStyle.SP_FileDialogListView)
        _add(sel_menu, "Show All", self.action_show_all, QStyle.SP_DialogYesButton)
        _add(sel_menu, "Open Analysis Tab (Selected)", self.action_open_selected_analysis_tab, QStyle.SP_FileDialogContentsView)

        modeling_menu = menu.addMenu("Modeling")
        _add(modeling_menu, "Create Box", lambda: self.action_create_primitive_dialog("box"), QStyle.SP_FileDialogNewFolder)
        _add(modeling_menu, "Create Cylinder", lambda: self.action_create_primitive_dialog("cylinder"), QStyle.SP_FileDialogNewFolder)
        _add(modeling_menu, "Create Sphere", lambda: self.action_create_primitive_dialog("sphere"), QStyle.SP_FileDialogNewFolder)
        _add(modeling_menu, "Boolean Union", self.action_boolean_union, QStyle.SP_DialogApplyButton)
        _add(modeling_menu, "Boolean Subtract", self.action_boolean_subtract, QStyle.SP_DialogApplyButton)
        _add(modeling_menu, "Boolean Intersect", self.action_boolean_intersect, QStyle.SP_DialogApplyButton)

        fem_menu = menu.addMenu("FEM")
        _add(fem_menu, "New Study", self.action_fem_new_study, QStyle.SP_FileDialogNewFolder)
        _add(fem_menu, "Include Selection", self.action_fem_include_selected, QStyle.SP_DialogYesButton)
        _add(fem_menu, "Assign Material", self.action_fem_assign_material_selected, QStyle.SP_DialogApplyButton)
        _add(fem_menu, "Generate Mesh", self.action_fem_generate_mesh, QStyle.SP_BrowserReload)
        _add(fem_menu, "Validate Study", self.action_fem_validate_study, QStyle.SP_MessageBoxInformation)
        _add(fem_menu, "Run Solve", self.action_fem_run_solve, QStyle.SP_MediaPlay)
        _add(fem_menu, "Refresh Results", self.action_fem_refresh_results, QStyle.SP_BrowserReload)

        io_menu = menu.addMenu("Import / Export")
        _add(io_menu, "Export Selected STEP", lambda: self.action_export_selected("step"), QStyle.SP_DialogSaveButton)
        _add(io_menu, "Export Selected STL", lambda: self.action_export_selected("stl"), QStyle.SP_DialogSaveButton)
        _add(io_menu, "Export Selected OBJ", lambda: self.action_export_selected("obj"), QStyle.SP_DialogSaveButton)
        _add(io_menu, "Save Backend Report", self.action_save_backend_report, QStyle.SP_DialogSaveButton)

        tools_menu = menu.addMenu("Tools")
        _add(tools_menu, "Reconnect Backend", self.action_reconnect_backend, QStyle.SP_BrowserReload)
        _add(tools_menu, "Backend Diagnostics", self.action_backend_diagnostics, QStyle.SP_MessageBoxInformation)
        return menu

    def _connect(self):
        self.scene_tree.selectionRequested.connect(self._on_tree_selection)
        self.scene_tree.visibilityRequested.connect(self._on_tree_visibility_change)
        self.scene_tree.contextRequested.connect(self._show_context_menu)
        self.viewport.selectionRequested.connect(self._on_viewport_selection)
        self.viewport.pickedInfo.connect(self._on_viewport_pick_info)
        self.viewport.hoverInfo.connect(self._on_viewport_hover_info)
        self.viewport.quickActionRequested.connect(self._on_viewport_quick_action)
        self.viewport.pieceCommandRequested.connect(self._on_viewport_piece_command)
        self.viewport.contextRequested.connect(self._show_context_menu)
        self.viewport.statusMessage.connect(self._log)
        self.viewport.measurePointPicked.connect(self._on_measure_point)
        self.properties.applyTransformRequested.connect(self._on_apply_transform)
        self.properties.contextRequested.connect(self._show_context_menu)
        self.measurements.contextRequested.connect(self._show_context_menu)
        self.selection_manager.selection_changed.connect(self._on_selection_manager_changed)
        self.selection_manager.active_item_changed.connect(self._on_selection_active_item_changed)
        self.selection_manager.hover_changed.connect(self._on_selection_hover_changed)
        self.selection_manager.mode_changed.connect(self._on_selection_mode_changed)

    def _register_shortcuts(self):
        self._panel_shortcuts = []

        def _bind(seq: str, slot):
            sc = QShortcut(QKeySequence(seq), self)
            sc.setContext(Qt.WidgetWithChildrenShortcut)
            sc.activated.connect(slot)
            self._panel_shortcuts.append(sc)

        _bind("Tab", self._shortcut_toggle_side_panels)
        _bind("Shift+Tab", self._shortcut_toggle_bottom_panel)
        _bind("F11", self._shortcut_toggle_max_viewport)
        _bind("Esc", self._shortcut_clear_selection)

    def _shortcut_allowed(self) -> bool:
        fw = QApplication.focusWidget()
        if fw is None:
            return True
        if isinstance(fw, (QLineEdit, QTextEdit)):
            return False
        return True

    def _shortcut_toggle_side_panels(self):
        if self._shortcut_allowed():
            self.action_toggle_side_panels()

    def _shortcut_toggle_bottom_panel(self):
        if self._shortcut_allowed():
            self.action_toggle_bottom_panel()

    def _shortcut_toggle_max_viewport(self):
        if self._shortcut_allowed():
            self.action_toggle_panels()

    def _shortcut_clear_selection(self):
        if self._shortcut_allowed():
            self.action_select_none()

    def _set_hint(self, text: str, wizard_lines: Optional[Sequence[str]] = None):
        msg = str(text or "").strip()
        if not msg:
            msg = "Click to select. Shift adds selection. RMB opens full context menu."
        self.hint_label.setText(f"Hint: {msg}")
        if wizard_lines is not None:
            lines = [str(x) for x in wizard_lines if str(x).strip()]
            self.inline_wizard_box.setPlainText("\n".join(lines))

    def _refresh_backend_status(self):
        caps = self.engine.backend_capabilities()
        provider = str(caps.get("provider", "unknown"))
        freecad_on = bool(caps.get("freecad_available", False))
        headless_on = bool(caps.get("freecad_headless_available", False))
        fem_on = bool(caps.get("fem_available", False))
        self.backend_label.setText(
            f"Backend: {provider} | FreeCAD={'yes' if freecad_on else 'no'} | Headless={'yes' if headless_on else 'no'} | FEM={'yes' if fem_on else 'no'}"
        )

    def _default_preferences(self) -> dict:
        return {
            "profile": "CAD Classic",
            "units": "mm",
            "grid": {"enabled": True, "step_mm": 5.0, "size_mm": 2000.0, "color": "#2b2b2b"},
            "snap": {"grid": True, "step_mm": 5.0, "angle": True, "step_deg": 5.0},
            "import": {"deflection": 0.2, "target_layer": "auto"},
            "render": {"mode": "solid_edges", "background": "dark", "aa": 4, "xray": False},
            "selection": {"sub_mode": "object", "auto_analysis_tab_on_click": False},
            "shortcuts": {"focus": "F", "undo": "Ctrl+Z", "redo": "Ctrl+Y", "delete": "Del"},
            "boolean": {"tolerance": 1e-6, "keep_originals": False, "clean": True, "triangulate": True, "normals": True},
            "boundaries": {"default_type": "fixed", "default_value": 1.0, "default_direction": "normal", "overlay_enabled": True},
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
        grid_cfg = self._prefs.get("grid", {})
        self._grid_enabled = bool(grid_cfg.get("enabled", True))
        self._grid_step_mm = float(grid_cfg.get("step_mm", 5.0))
        self._grid_size_mm = float(grid_cfg.get("size_mm", 2000.0))
        self._grid_color = str(grid_cfg.get("color", "#2b2b2b") or "#2b2b2b")
        imp_cfg = self._prefs.get("import", {})
        self._mesh_deflection = float(imp_cfg.get("deflection", 0.2))
        self.engine.set_default_mesh_quality({"deflection": self._mesh_deflection})
        sel_cfg = self._prefs.get("selection", {})
        self._subselect_mode = str(sel_cfg.get("sub_mode", "object") or "object").strip().lower()
        render_cfg = self._prefs.get("render", {})
        self._xray_mode = bool(render_cfg.get("xray", False))
        bool_cfg = self._prefs.get("boolean", {})
        self._boolean_tolerance = float(bool_cfg.get("tolerance", 1e-6))
        self._boolean_keep_originals = bool(bool_cfg.get("keep_originals", False))

    def _apply_preferences(self):
        unit = str(self._prefs.get("units", "mm"))
        profile = str(self._prefs.get("profile", "CAD Classic"))
        grid_cfg = self._prefs.get("grid", {})
        render_cfg = self._prefs.get("render", {})
        snap_cfg = self._prefs.get("snap", {})
        imp_cfg = self._prefs.get("import", {})
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
        self._grid_enabled = bool(grid_cfg.get("enabled", self._grid_enabled))
        self._grid_step_mm = float(grid_cfg.get("step_mm", self._grid_step_mm))
        self._grid_size_mm = float(grid_cfg.get("size_mm", self._grid_size_mm))
        self._grid_color = str(grid_cfg.get("color", self._grid_color) or self._grid_color)
        self._mesh_deflection = float(imp_cfg.get("deflection", self._mesh_deflection))
        self.engine.set_default_mesh_quality({"deflection": self._mesh_deflection})
        sel_cfg = self._prefs.get("selection", {})
        self._subselect_mode = str(sel_cfg.get("sub_mode", self._subselect_mode) or self._subselect_mode).strip().lower()
        self._xray_mode = bool(render_cfg.get("xray", self._xray_mode))
        self._boolean_tolerance = float(bool_cfg.get("tolerance", self._boolean_tolerance))
        self._boolean_keep_originals = bool(bool_cfg.get("keep_originals", self._boolean_keep_originals))
        bc_cfg = self._prefs.get("boundaries", {})
        self._boundary_overlay_enabled = bool(bc_cfg.get("overlay_enabled", True))

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
        if hasattr(self, "boundary_type_combo"):
            btype = str(bc_cfg.get("default_type", "fixed") or "fixed").strip().lower()
            idx = self.boundary_type_combo.findText(btype)
            if idx >= 0:
                self.boundary_type_combo.setCurrentIndex(idx)
        if hasattr(self, "boundary_value_edit"):
            self.boundary_value_edit.setText(str(bc_cfg.get("default_value", 1.0)))
        if hasattr(self, "boundary_dir_combo"):
            bdir = str(bc_cfg.get("default_direction", "normal") or "normal").strip().lower()
            idx = self.boundary_dir_combo.findText(bdir)
            if idx >= 0:
                self.boundary_dir_combo.setCurrentIndex(idx)
        if hasattr(self, "boundary_overlay_chk"):
            self.boundary_overlay_chk.blockSignals(True)
            self.boundary_overlay_chk.setChecked(bool(self._boundary_overlay_enabled))
            self.boundary_overlay_chk.blockSignals(False)
        if hasattr(self, "grid_step_edit"):
            self.grid_step_edit.setText(f"{self._grid_step_mm:.6g}")
        if hasattr(self, "grid_size_edit"):
            self.grid_size_edit.setText(f"{self._grid_size_mm:.6g}")
        if hasattr(self, "grid_color_edit"):
            self.grid_color_edit.setText(str(self._grid_color))
        if hasattr(self, "grid_enable_chk"):
            self.grid_enable_chk.blockSignals(True)
            self.grid_enable_chk.setChecked(bool(self._grid_enabled))
            self.grid_enable_chk.blockSignals(False)
        if hasattr(self, "import_deflection_edit"):
            self.import_deflection_edit.setText(f"{self._mesh_deflection:.6g}")
        self._sync_subselect_combos()
        if hasattr(self, "auto_analysis_chk"):
            self.auto_analysis_chk.blockSignals(True)
            self.auto_analysis_chk.setChecked(bool(sel_cfg.get("auto_analysis_tab_on_click", False)))
            self.auto_analysis_chk.blockSignals(False)
        if hasattr(self, "view_mode_combo"):
            mode_map = {"solid_edges": "Solid+Edges", "solid": "Solid", "wireframe": "Wireframe"}
            desired = "X-Ray" if self._xray_mode else mode_map.get(str(render_cfg.get("mode", "solid_edges")), "Solid+Edges")
            idx = self.view_mode_combo.findText(desired)
            if idx >= 0:
                self.view_mode_combo.blockSignals(True)
                self.view_mode_combo.setCurrentIndex(idx)
                self.view_mode_combo.blockSignals(False)
        self.viewport.set_grid_config(
            enabled=self._grid_enabled,
            step=self._grid_step_mm,
            size=self._grid_size_mm,
            color=self._grid_color,
        )
        self.viewport.set_xray_mode(self._xray_mode)
        self.viewport.set_selection_mode_indicator(self._subselect_mode)
        self.viewport.set_snap_enabled(self._snap_enabled)

        self.action_set_background(str(render_cfg.get("background", "dark")))
        self.action_set_render_mode(str(render_cfg.get("mode", "solid_edges")))
        self._set_hint("Click to select. Shift adds selection. RMB opens full context menu.")

    def _profile_defaults(self, profile: str) -> dict:
        token = str(profile or "CAD Classic").strip().lower()
        if token == "minimal":
            return {"layout": "layout_modeling", "render_mode": "solid", "background": "dark", "snap": False}
        if token == "analysis":
            return {"layout": "layout_fem", "render_mode": "solid_edges", "background": "light", "snap": True}
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
        if hasattr(self, "tr_snap_chk") and bool(self.tr_snap_chk.isChecked()) != self._snap_enabled:
            self.tr_snap_chk.blockSignals(True)
            self.tr_snap_chk.setChecked(self._snap_enabled)
            self.tr_snap_chk.blockSignals(False)
        self._prefs.setdefault("snap", {})
        self._prefs["snap"]["grid"] = bool(self._snap_enabled)
        self.viewport.set_snap_enabled(self._snap_enabled)
        if persist:
            self._save_preferences()
        self._set_hint(f"Snap to grid {'enabled' if self._snap_enabled else 'disabled'}.")

    def _save_layout_from_combo(self):
        self.save_layout(self.layout_combo.currentText())

    def _load_layout_from_combo(self):
        self.load_layout(self.layout_combo.currentText())

    def _capture_panel_restore_sizes(self):
        try:
            sizes = [int(x) for x in self.main_split.sizes()]
            if len(sizes) >= 3:
                if self.left_tabs.isVisible() and sizes[0] > 0:
                    self._last_left_width = int(sizes[0])
                if self.right_wrap.isVisible() and sizes[2] > 0:
                    self._last_right_width = int(sizes[2])
                self._last_main_sizes = list(sizes[:3])
        except Exception:
            pass
        try:
            sizes = [int(x) for x in self.outer_split.sizes()]
            if len(sizes) >= 2:
                if self.bottom_tabs.isVisible() and sizes[1] > 0:
                    self._last_bottom_height = int(sizes[1])
                self._last_outer_sizes = list(sizes[:2])
        except Exception:
            pass

    def _apply_splitter_sizes_from_visibility(self):
        total_w = max(900, int(sum(self.main_split.sizes()) or 0))
        left_w = int(max(180, self._last_left_width)) if self.left_tabs.isVisible() else 0
        right_w = int(max(220, self._last_right_width)) if self.right_wrap.isVisible() else 0
        center_w = max(260, total_w - left_w - right_w)
        if center_w < 260:
            deficit = 260 - center_w
            if right_w >= left_w and right_w > 0:
                right_w = max(180, right_w - deficit)
            elif left_w > 0:
                left_w = max(160, left_w - deficit)
            center_w = max(260, total_w - left_w - right_w)
        self.main_split.setSizes([left_w, center_w, right_w])

        total_h = max(420, int(sum(self.outer_split.sizes()) or 0))
        bottom_h = int(max(90, self._last_bottom_height)) if self.bottom_tabs.isVisible() else 0
        top_h = max(220, total_h - bottom_h)
        if top_h < 220:
            bottom_h = max(80, total_h - 220)
            top_h = max(220, total_h - bottom_h)
        self.outer_split.setSizes([top_h, bottom_h])

    def _set_panel_visibility(
        self,
        *,
        left: Optional[bool] = None,
        right: Optional[bool] = None,
        bottom: Optional[bool] = None,
        ribbon: Optional[bool] = None,
        ops: Optional[bool] = None,
        refresh_sizes: bool = True,
    ):
        if left is not None:
            self.left_tabs.setVisible(bool(left))
        if right is not None:
            self.right_wrap.setVisible(bool(right))
        if bottom is not None:
            self.bottom_tabs.setVisible(bool(bottom))
        if ribbon is not None:
            self.ribbon_tabs.setVisible(bool(ribbon))
        if ops is not None:
            self.ops_tabs.setVisible(bool(ops))
        if refresh_sizes:
            self._apply_splitter_sizes_from_visibility()

    def _builtin_layout_preset(self, name: str) -> Optional[dict]:
        token = str(name or "layout_default").strip().lower()
        if token == "layout_analysis":
            token = "layout_fem"
        if token == "layout_default":
            return {
                "main_split_sizes": [260, 1340, 320],
                "outer_split_sizes": [900, 120],
                "left_panel_visible": True,
                "right_panel_visible": True,
                "bottom_panel_visible": True,
                "ribbon_visible": True,
                "ops_visible": True,
                "panels_collapsed": False,
            }
        if token == "layout_modeling":
            return {
                "main_split_sizes": [280, 1500, 300],
                "outer_split_sizes": [940, 100],
                "left_panel_visible": True,
                "right_panel_visible": True,
                "bottom_panel_visible": True,
                "ribbon_visible": True,
                "ops_visible": True,
                "panels_collapsed": False,
            }
        if token == "layout_fem":
            return {
                "main_split_sizes": [300, 1300, 430],
                "outer_split_sizes": [900, 140],
                "left_panel_visible": True,
                "right_panel_visible": True,
                "bottom_panel_visible": True,
                "ribbon_visible": True,
                "ops_visible": True,
                "panels_collapsed": False,
            }
        if token == "layout_results":
            return {
                "main_split_sizes": [240, 1520, 320],
                "outer_split_sizes": [860, 190],
                "left_panel_visible": True,
                "right_panel_visible": True,
                "bottom_panel_visible": True,
                "ribbon_visible": True,
                "ops_visible": True,
                "panels_collapsed": False,
            }
        return None

    def _apply_layout_payload(self, payload: dict):
        left_visible = bool(payload.get("left_panel_visible", True))
        right_visible = bool(payload.get("right_panel_visible", True))
        bottom_visible = bool(payload.get("bottom_panel_visible", True))
        ribbon_visible = bool(payload.get("ribbon_visible", True))
        ops_visible = bool(payload.get("ops_visible", True))
        self._panels_collapsed = bool(payload.get("panels_collapsed", False))
        self._set_panel_visibility(
            left=left_visible and not self._panels_collapsed,
            right=right_visible and not self._panels_collapsed,
            bottom=bottom_visible and not self._panels_collapsed,
            ribbon=ribbon_visible and not self._panels_collapsed,
            ops=ops_visible and not self._panels_collapsed,
            refresh_sizes=False,
        )

        main_sizes = payload.get("main_split_sizes", [])
        outer_sizes = payload.get("outer_split_sizes", [])
        if isinstance(main_sizes, list) and len(main_sizes) == 3:
            self.main_split.setSizes([int(x) for x in main_sizes])
        if isinstance(outer_sizes, list) and len(outer_sizes) == 2:
            self.outer_split.setSizes([int(x) for x in outer_sizes])
        self._capture_panel_restore_sizes()
        if self._panels_collapsed:
            self._max_viewport_state = {
                "left": bool(left_visible),
                "right": bool(right_visible),
                "bottom": bool(bottom_visible),
                "ribbon": bool(ribbon_visible),
                "ops": bool(ops_visible),
                "main_sizes": [int(x) for x in self._last_main_sizes],
                "outer_sizes": [int(x) for x in self._last_outer_sizes],
            }
            self._set_panel_visibility(left=False, right=False, bottom=False, ribbon=False, ops=False, refresh_sizes=True)
        self.btn_toggle_panels.setText("Restore Panels" if self._panels_collapsed else "Max Viewport")

    def save_layout(self, name: str):
        try:
            self._capture_panel_restore_sizes()
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
                "left_panel_visible": bool(self.left_tabs.isVisible()),
                "right_panel_visible": bool(self.right_wrap.isVisible()),
                "bottom_panel_visible": bool(self.bottom_tabs.isVisible()),
                "ribbon_visible": bool(self.ribbon_tabs.isVisible()),
                "ops_visible": bool(self.ops_tabs.isVisible()),
                "panels_collapsed": bool(self._panels_collapsed),
                "last_main_sizes": [int(x) for x in self._last_main_sizes],
                "last_outer_sizes": [int(x) for x in self._last_outer_sizes],
            }
            with open(self._layout_path(name), "w", encoding="utf-8", newline="\n") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            self._log(f"Layout saved: {self._layout_path(name)}")
        except Exception as e:
            QMessageBox.warning(self, "Save layout", str(e))

    def load_layout(self, name: str, silent: bool = False):
        path = self._layout_path(name)
        if not path.is_file():
            preset = self._builtin_layout_preset(name)
            if preset is not None:
                self._apply_layout_payload(preset)
                if not silent:
                    self._log(f"Built-in layout loaded: {str(name)}")
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
            self._apply_layout_payload(payload)
            last_main_sizes = payload.get("last_main_sizes", [])
            if isinstance(last_main_sizes, list) and len(last_main_sizes) == 3:
                self._last_main_sizes = [int(x) for x in last_main_sizes]
            last_outer_sizes = payload.get("last_outer_sizes", [])
            if isinstance(last_outer_sizes, list) and len(last_outer_sizes) == 2:
                self._last_outer_sizes = [int(x) for x in last_outer_sizes]
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
        self._panels_collapsed = False
        self._max_viewport_state = None
        self._set_panel_visibility(left=True, right=True, bottom=True, ribbon=True, ops=True, refresh_sizes=False)
        self.main_split.setSizes([260, 1340, 320])
        self.outer_split.setSizes([900, 120])
        self._capture_panel_restore_sizes()
        self.ops_tabs.setCurrentIndex(0)
        self.scene_tree.search_edit.setText("")
        if hasattr(self, "btn_toggle_panels"):
            self.btn_toggle_panels.setText("Max Viewport")
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
        tabs.addTab(self._build_tab_boundaries(), "Boundaries")
        tabs.addTab(self._build_tab_markers(), "Markers")
        tabs.addTab(self._build_tab_export(), "Export")
        tabs.addTab(self._build_tab_backend(), "Backend")
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
        g.addWidget(QLabel("Use search in Scene Tree to filter by name/source/group/layer."), 3, 0, 1, 8)

        g.addWidget(QLabel("Layers"), 4, 0)
        self.layer_combo = QComboBox(w)
        g.addWidget(self.layer_combo, 4, 1, 1, 2)
        btn_layer_assign = QPushButton("Assign selected", w)
        btn_layer_assign.clicked.connect(self.action_assign_selected_layer)
        g.addWidget(btn_layer_assign, 4, 3)
        btn_layer_new = QPushButton("New", w)
        btn_layer_new.clicked.connect(self.action_new_layer)
        g.addWidget(btn_layer_new, 4, 4)
        btn_layer_rename = QPushButton("Rename", w)
        btn_layer_rename.clicked.connect(self.action_rename_layer)
        g.addWidget(btn_layer_rename, 4, 5)
        btn_layer_delete = QPushButton("Delete", w)
        btn_layer_delete.clicked.connect(self.action_delete_layer)
        g.addWidget(btn_layer_delete, 4, 6)

        btn_layer_vis = QPushButton("Toggle layer visibility", w)
        btn_layer_vis.clicked.connect(self.action_toggle_layer_visibility)
        g.addWidget(btn_layer_vis, 5, 0, 1, 2)
        btn_layer_color = QPushButton("Layer color", w)
        btn_layer_color.clicked.connect(self.action_set_layer_color)
        g.addWidget(btn_layer_color, 5, 2, 1, 2)

        self.grid_enable_chk = QCheckBox("Grid enabled", w)
        self.grid_enable_chk.setChecked(bool(self._grid_enabled))
        g.addWidget(self.grid_enable_chk, 5, 4, 1, 2)
        g.addWidget(QLabel("Grid step"), 6, 0)
        self.grid_step_edit = QLineEdit(f"{self._grid_step_mm:.6g}", w)
        g.addWidget(self.grid_step_edit, 6, 1)
        g.addWidget(QLabel("Grid size"), 6, 2)
        self.grid_size_edit = QLineEdit(f"{self._grid_size_mm:.6g}", w)
        g.addWidget(self.grid_size_edit, 6, 3)
        g.addWidget(QLabel("Grid color"), 6, 4)
        self.grid_color_edit = QLineEdit(str(self._grid_color), w)
        g.addWidget(self.grid_color_edit, 6, 5)
        btn_grid_apply = QPushButton("Apply grid", w)
        btn_grid_apply.clicked.connect(self.action_apply_grid_settings)
        g.addWidget(btn_grid_apply, 6, 6)
        self._refresh_layer_controls()
        return w

    def _build_tab_create(self) -> QWidget:
        w = QWidget(self)
        g = QGridLayout(w)
        g.setContentsMargins(6, 4, 6, 4)
        g.setHorizontalSpacing(6)
        g.setVerticalSpacing(6)

        self.create_kind_combo = QComboBox(w)
        self.create_kind_combo.addItems(["box", "cylinder", "sphere", "cone", "tube", "plane"])
        self.create_name_edit = QLineEdit("primitive_1", w)
        self.create_center_combo = QComboBox(w)
        self.create_center_combo.addItems(["center on origin", "center on selection"])
        self.create_w_edit = QLineEdit("1.0", w)
        self.create_d_edit = QLineEdit("1.0", w)
        self.create_h_edit = QLineEdit("1.0", w)
        self.create_r_edit = QLineEdit("0.5", w)
        self.create_ri_edit = QLineEdit("0.25", w)
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
        g.addWidget(QLabel("Ri"), 2, 2)
        g.addWidget(self.create_ri_edit, 2, 3)
        g.addWidget(QLabel("Segments"), 2, 4)
        g.addWidget(self.create_seg_edit, 2, 5)

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

        g.addWidget(QLabel("Sub-selection"), 5, 0)
        self.tr_subselect_combo = QComboBox(w)
        self.tr_subselect_combo.addItems(["Object", "Face", "Edge", "Vertex", "Body", "Component"])
        self.tr_subselect_combo.currentTextChanged.connect(self.action_set_subselection_mode)
        g.addWidget(self.tr_subselect_combo, 5, 1)
        self.face_edge_info_label = QLabel("Face/Edge: -", w)
        g.addWidget(self.face_edge_info_label, 5, 2, 1, 4)
        g.addWidget(QLabel("Offset"), 6, 0)
        self.face_edge_offset_edit = QLineEdit("0.1", w)
        g.addWidget(self.face_edge_offset_edit, 6, 1)
        btn_face_plus = QPushButton("Face +", w)
        btn_face_minus = QPushButton("Face -", w)
        btn_edge_plus = QPushButton("Edge +", w)
        btn_edge_minus = QPushButton("Edge -", w)
        btn_face_plus.clicked.connect(lambda: self.action_adjust_selected_face(+1.0))
        btn_face_minus.clicked.connect(lambda: self.action_adjust_selected_face(-1.0))
        btn_edge_plus.clicked.connect(lambda: self.action_adjust_selected_edge(+1.0))
        btn_edge_minus.clicked.connect(lambda: self.action_adjust_selected_edge(-1.0))
        g.addWidget(btn_face_plus, 6, 2)
        g.addWidget(btn_face_minus, 6, 3)
        g.addWidget(btn_edge_plus, 6, 4)
        g.addWidget(btn_edge_minus, 6, 5)
        if hasattr(self, "tr_subselect_combo"):
            idx = self.tr_subselect_combo.findText(self._subselect_mode.capitalize())
            if idx >= 0:
                self.tr_subselect_combo.setCurrentIndex(idx)
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
        btn_validate = QPushButton("Validate", w)
        btn_heal = QPushButton("Heal", w)
        btn_validate.clicked.connect(lambda: self.action_validate_selected(self._pick_selected()))
        btn_heal.clicked.connect(lambda: self.action_heal_selected(self._pick_selected()))
        btn_heal.setEnabled(bool(self.engine.backend_has("heal_available")))
        g.addWidget(btn_validate, 2, 0)
        g.addWidget(btn_heal, 2, 1)

        g.addWidget(QLabel("Measure tools"), 3, 0, 1, 4)
        btn_dist = QPushButton("Distance (2 points)", w)
        btn_angle = QPushButton("Angle (3 points)", w)
        btn_clear = QPushButton("Clear current", w)
        btn_save = QPushButton("Save measure as marker", w)
        btn_dist.clicked.connect(lambda: self.action_enter_measure_mode("distance"))
        btn_angle.clicked.connect(lambda: self.action_enter_measure_mode("angle"))
        btn_clear.clicked.connect(self.action_clear_current_measure)
        btn_save.clicked.connect(self.action_save_measure_as_marker)
        g.addWidget(btn_dist, 4, 0)
        g.addWidget(btn_angle, 4, 1)
        g.addWidget(btn_clear, 4, 2)
        g.addWidget(btn_save, 4, 3)

        g.addWidget(QLabel("Sections / clipping"), 5, 0, 1, 4)
        btn_xy = QPushButton("Clip XY", w)
        btn_xz = QPushButton("Clip XZ", w)
        btn_yz = QPushButton("Clip YZ", w)
        btn_clr = QPushButton("Clear clips", w)
        btn_xy.clicked.connect(lambda: self.action_add_clipping_plane("xy"))
        btn_xz.clicked.connect(lambda: self.action_add_clipping_plane("xz"))
        btn_yz.clicked.connect(lambda: self.action_add_clipping_plane("yz"))
        btn_clr.clicked.connect(self.action_clear_clipping)
        g.addWidget(btn_xy, 6, 0)
        g.addWidget(btn_xz, 6, 1)
        g.addWidget(btn_yz, 6, 2)
        g.addWidget(btn_clr, 6, 3)
        return w

    def _build_tab_boundaries(self) -> QWidget:
        w = QWidget(self)
        g = QGridLayout(w)
        g.setContentsMargins(6, 4, 6, 4)
        g.setHorizontalSpacing(6)
        g.setVerticalSpacing(6)

        g.addWidget(QLabel("Boundary assignment for imported/created CAD bodies"), 0, 0, 1, 6)
        self.boundary_type_combo = QComboBox(w)
        self.boundary_type_combo.addItems(list(self._boundary_types))
        self.boundary_name_edit = QLineEdit("bc_1", w)
        self.boundary_value_edit = QLineEdit("1.0", w)
        self.boundary_dir_combo = QComboBox(w)
        self.boundary_dir_combo.addItems(["normal", "x", "y", "z"])
        self.boundary_pick_chk = QCheckBox("Use picked point/face (when available)", w)
        self.boundary_pick_chk.setChecked(True)
        self.boundary_overlay_chk = QCheckBox("Show boundary overlay on selected bodies", w)
        self.boundary_overlay_chk.setChecked(bool(self._boundary_overlay_enabled))
        self.boundary_overlay_chk.toggled.connect(self._on_boundary_overlay_toggled)

        g.addWidget(QLabel("Type"), 1, 0)
        g.addWidget(self.boundary_type_combo, 1, 1)
        g.addWidget(QLabel("Name"), 1, 2)
        g.addWidget(self.boundary_name_edit, 1, 3)
        g.addWidget(QLabel("Value"), 1, 4)
        g.addWidget(self.boundary_value_edit, 1, 5)
        g.addWidget(QLabel("Direction"), 2, 0)
        g.addWidget(self.boundary_dir_combo, 2, 1)
        g.addWidget(self.boundary_pick_chk, 2, 2, 1, 4)
        g.addWidget(self.boundary_overlay_chk, 3, 0, 1, 6)

        btn_apply = QPushButton("Apply to selected", w)
        btn_apply.clicked.connect(self.action_apply_boundary_from_tab)
        btn_summary = QPushButton("Summary selected", w)
        btn_summary.clicked.connect(lambda: self.action_show_boundary_summary(self._pick_selected()))
        btn_clear = QPushButton("Clear selected", w)
        btn_clear.clicked.connect(self.action_clear_boundaries_selected)
        g.addWidget(btn_apply, 4, 0, 1, 2)
        g.addWidget(btn_summary, 4, 2, 1, 2)
        g.addWidget(btn_clear, 4, 4, 1, 2)

        g.addWidget(QLabel("Existing boundaries"), 5, 0, 1, 6)
        self.boundary_combo = QComboBox(w)
        g.addWidget(self.boundary_combo, 6, 0, 1, 6)

        btn_remove = QPushButton("Remove selected boundary", w)
        btn_remove.clicked.connect(self.action_remove_boundary_selected)
        btn_copy = QPushButton("Copy boundary JSON", w)
        btn_copy.clicked.connect(self.action_copy_boundary_json)
        g.addWidget(btn_remove, 7, 0, 1, 3)
        g.addWidget(btn_copy, 7, 3, 1, 3)
        self._refresh_boundary_controls()
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
        self.export_fmt_combo.addItems(["STEP", "STL", "OBJ"])
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

    def _build_tab_backend(self) -> QWidget:
        w = QWidget(self)
        g = QGridLayout(w)
        g.setContentsMargins(6, 4, 6, 4)
        g.setHorizontalSpacing(6)
        g.setVerticalSpacing(6)
        g.addWidget(QLabel("Kernel diagnostics and capability report"), 0, 0, 1, 4)

        self.backend_diag_box = QTextEdit(w)
        self.backend_diag_box.setReadOnly(True)
        self.backend_diag_box.setMinimumHeight(180)
        g.addWidget(self.backend_diag_box, 1, 0, 1, 4)

        btn_refresh = QPushButton("Refresh Doctor", w)
        btn_refresh.clicked.connect(self.action_backend_diagnostics)
        btn_reconnect = QPushButton("Reconnect Backend", w)
        btn_reconnect.clicked.connect(self.action_reconnect_backend)
        btn_retess = QPushButton("Retessellate selected", w)
        btn_retess.clicked.connect(self.action_retessellate_selected)
        btn_save = QPushButton("Save Report", w)
        btn_save.clicked.connect(self.action_save_backend_report)
        g.addWidget(btn_refresh, 2, 0)
        g.addWidget(btn_reconnect, 2, 1)
        g.addWidget(btn_retess, 2, 2)
        g.addWidget(btn_save, 2, 3)
        self._refresh_backend_diagnostics_box()
        return w

    def _refresh_backend_diagnostics_box(self):
        if not hasattr(self, "backend_diag_box"):
            return
        payload = {
            "capabilities": self.engine.backend_capabilities(),
            "doctor": self.engine.backend_diagnostics(),
        }
        self.backend_diag_box.setPlainText(json.dumps(payload, ensure_ascii=False, indent=2))

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
        elif token == "tube":
            params["outer_radius"] = self._parse_float(self.create_r_edit, 0.6)
            params["inner_radius"] = self._parse_float(self.create_ri_edit, 0.3)
            params["height"] = self._parse_float(self.create_h_edit, 1.0)
            params["segments"] = max(12, self._parse_int(self.create_seg_edit, 48))
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

    def _current_layer_name(self) -> str:
        if not hasattr(self, "layer_combo"):
            return "Default"
        return str(self.layer_combo.currentData() or self.layer_combo.currentText() or "Default")

    def _refresh_layer_controls(self):
        if not hasattr(self, "layer_combo"):
            return
        current = self._current_layer_name()
        self.layer_combo.blockSignals(True)
        self.layer_combo.clear()
        for row in self.engine.layer_rows():
            name = str(row.get("name", "Default") or "Default")
            count = int(row.get("count", 0))
            vis = "on" if bool(row.get("visible", True)) else "off"
            self.layer_combo.addItem(f"{name} ({count}, {vis})", name)
        self.layer_combo.blockSignals(False)
        if current:
            for idx in range(self.layer_combo.count()):
                if str(self.layer_combo.itemData(idx)) == current:
                    self.layer_combo.setCurrentIndex(idx)
                    break

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
        fmt = str(self.export_fmt_combo.currentText() or "STEP").strip().lower()
        if fmt not in {"step", "stl", "obj"}:
            fmt = "step"
        self.action_export_selected(fmt)

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

    def _on_engine_event(self, event: str, payload: dict):
        if event == "solve_progress":
            msg = str(dict(payload or {}).get("message", "")).strip()
            if msg:
                self._solver_log(msg)
            return
        if event == "selection_changed":
            self._selection_sync_lock = True
            try:
                self.selection_manager.set_mode(SelectionMode.from_value(self._subselect_mode))
                self.selection_manager.set_from_ids(self.engine.selection, self.engine.objects, op="replace")
            finally:
                self._selection_sync_lock = False
            hover_item = self.selection_manager.hover_item()
            hover_id = str((hover_item.parent_object_id if hover_item is not None else "") or (hover_item.entity_id if hover_item is not None else "") or "")
            if hover_item is not None and hover_item.entity_type in {"face", "edge", "vertex"}:
                hover_id = str(hover_item.parent_object_id or "")
            active_item = self.selection_manager.active_item()
            active_id = str((active_item.parent_object_id if active_item is not None else "") or (active_item.entity_id if active_item is not None else "") or "")
            if active_item is not None and active_item.entity_type in {"face", "edge", "vertex"}:
                active_id = str(active_item.parent_object_id or "")
            self.scene_tree.refresh(self.engine.objects, self.engine.selection)
            self.viewport.update_selection(self.engine.selection, hover_id=hover_id, active_id=active_id)
            self._refresh_properties()
            self._refresh_boolean_selection_ui()
            self._refresh_boundary_controls()
            self._refresh_selection_info_box()
            self._update_face_edge_info_label()
            self._update_undo_redo_buttons()
            return
        if event == "solve_started":
            sid = str(dict(payload or {}).get("study_id", "") or "")
            self._solver_log(f"Solve started ({sid}).")
        if event == "solve_finished":
            sid = str(dict(payload or {}).get("study_id", "") or "")
            self._solver_log(f"Solve finished ({sid}).")
        if event in {
            "scene_changed",
            "markers_changed",
            "measurements_changed",
            "scene_updated",
            "material_assigned",
            "mesh_generated",
            "study_validation_changed",
            "solve_started",
            "solve_finished",
            "results_updated",
        }:
            self.refresh_all()
            self._update_undo_redo_buttons()
            self._refresh_backend_status()

    def _update_undo_redo_buttons(self):
        self.btn_undo.setEnabled(self.engine.command_stack.can_undo)
        self.btn_redo.setEnabled(self.engine.command_stack.can_redo)

    def refresh_all(self):
        self._selection_sync_lock = True
        try:
            self.selection_manager.set_mode(SelectionMode.from_value(self._subselect_mode))
            self.selection_manager.sync_scene(self.engine.objects)
            if self.selection_manager.selected_ids() != list(self.engine.selection):
                self.selection_manager.set_from_ids(self.engine.selection, self.engine.objects, op="replace")
        finally:
            self._selection_sync_lock = False

        boundary_map = {}
        if bool(self._boundary_overlay_enabled):
            for row in self.engine.boundary_rows():
                btype = str(row.get("type", "")).strip().lower()
                if not btype:
                    continue
                for oid in [str(x) for x in row.get("targets", []) if str(x) in self.engine.objects]:
                    boundary_map.setdefault(oid, []).append(btype)
        hover_item = self.selection_manager.hover_item()
        hover_id = str((hover_item.parent_object_id if hover_item is not None else "") or (hover_item.entity_id if hover_item is not None else "") or "")
        if hover_item is not None and hover_item.entity_type in {"face", "edge", "vertex"}:
            hover_id = str(hover_item.parent_object_id or "")
        active_item = self.selection_manager.active_item()
        active_id = str((active_item.parent_object_id if active_item is not None else "") or (active_item.entity_id if active_item is not None else "") or "")
        if active_item is not None and active_item.entity_type in {"face", "edge", "vertex"}:
            active_id = str(active_item.parent_object_id or "")
        self.scene_tree.refresh(self.engine.objects, self.engine.selection)
        self.viewport.refresh_scene(
            self.engine.objects,
            self.engine.selection,
            self.engine.markers,
            boundary_map=boundary_map,
            hover_id=hover_id,
            active_id=active_id,
        )
        self.viewport.set_selection_mode_indicator(self._subselect_mode)
        self.viewport.set_snap_enabled(self._snap_enabled)
        if self._subselect_mode == "face" and self._selected_face is not None:
            self.viewport.set_subselection(face=self._selected_face, edge=None)
        elif self._subselect_mode == "edge" and self._selected_edge is not None:
            oid, edge, face_idx = self._selected_edge
            self.viewport.set_subselection(face=(oid, face_idx), edge=(oid, edge))
        else:
            self.viewport.clear_subselection()
        self.measurements.set_rows(self._compose_measure_rows())
        self._refresh_properties()
        self._refresh_boolean_selection_ui()
        self._refresh_boundary_controls()
        self._refresh_layer_controls()
        self._refresh_layers_overview()
        self._refresh_components_library()
        self._refresh_marker_controls()
        self._refresh_backend_diagnostics_box()
        self._refresh_selection_info_box()
        self._refresh_diagnostics_box()
        self._refresh_fem_panels()
        self._update_undo_redo_buttons()
        stale = []
        for key in list(self._analysis_tabs.keys()):
            if not key.startswith("analysis::"):
                continue
            oid = key.split("::", 1)[1]
            if oid in self.engine.objects:
                self._refresh_analysis_tab(oid)
            else:
                stale.append(oid)
        for oid in stale:
            self._close_analysis_tab(oid)
        if self._selected_face is not None:
            foid, _ = self._selected_face
            if str(foid) not in self.engine.objects:
                self._selected_face = None
        if self._selected_edge is not None:
            eoid, _edge, _fi = self._selected_edge
            if str(eoid) not in self.engine.objects:
                self._selected_edge = None
        if self._selected_vertex is not None:
            void, _vid, _fi = self._selected_vertex
            if str(void) not in self.engine.objects:
                self._selected_vertex = None
        self._update_face_edge_info_label()

    def _compose_measure_rows(self):
        rows = [dict(r) for r in self.engine.measurements]
        for bc in self.engine.boundary_rows():
            rows.append(
                {
                    "type": "boundary",
                    "name": str(bc.get("name", "")),
                    "value": str(bc.get("type", "")),
                    "info": f"targets={len(bc.get('targets', []))}",
                    "boundary_id": str(bc.get("id", "")),
                    "object_id": ",".join([str(x) for x in bc.get("targets", [])]),
                }
            )
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

    def _refresh_selection_info_box(self):
        if not hasattr(self, "selection_info_box"):
            return
        items = list(self.selection_manager.current_selection())
        lines = [f"Selected items: {len(items)}", ""]
        if not items:
            lines.append("No selection.")
            self.selection_info_box.setPlainText("\n".join(lines))
            return
        active = self.selection_manager.active_item()
        if active is not None:
            lines.append(f"Active: {active.display_name or active.entity_id}")
            lines.append("")
        for idx, item in enumerate(items, start=1):
            token = str(item.display_name or item.entity_id)
            lines.append(f"{idx}. {token}")
            lines.append(f"   type={item.entity_type} layer={item.layer or '-'}")
            if item.sub_index is not None:
                lines.append(f"   sub_index={item.sub_index}")
            if item.parent_object_id:
                lines.append(f"   parent={item.parent_object_id}")
        self.selection_info_box.setPlainText("\n".join(lines))

    def _refresh_diagnostics_box(self):
        if not hasattr(self, "diagnostics_box"):
            return
        caps = self.engine.backend_capabilities()
        rows = {
            "objects": int(len(self.engine.objects)),
            "selection": int(len(self.engine.selection)),
            "markers": int(len(self.engine.markers)),
            "boundaries": int(len(self.engine.boundaries)),
            "layers": int(len(self.engine.layers)),
            "backend_provider": str(caps.get("provider", "unknown")),
            "freecad_available": bool(caps.get("freecad_available", False)),
            "fem_available": bool(caps.get("fem_available", False)),
            "active_study": str(self.engine.fem.active_study_id() or "-"),
            "studies_count": int(len(self.engine.fem.list_studies())),
        }
        try:
            rows["viewport"] = dict(self.viewport.diagnostics_snapshot())
        except Exception:
            pass
        self.diagnostics_box.setPlainText(json.dumps(rows, ensure_ascii=False, indent=2))

    def _refresh_layers_overview(self):
        if not hasattr(self, "layers_overview_tree"):
            return
        tree = self.layers_overview_tree
        tree.clear()
        for row in self.engine.layer_rows():
            name = str(row.get("name", "Default"))
            vis = "Yes" if bool(row.get("visible", True)) else "No"
            locked = "Yes" if bool(row.get("locked", False)) else "No"
            count = str(int(row.get("count", 0)))
            color = str(row.get("color", "#86b6f6"))
            item = QTreeWidgetItem([name, vis, locked, count, color])
            tree.addTopLevelItem(item)

    def _refresh_components_library(self):
        if hasattr(self, "components_imported_list"):
            self.components_imported_list.clear()
            groups: Dict[str, dict] = {}
            for oid, obj in self.engine.objects.items():
                meta = getattr(obj, "meta", {})
                if not isinstance(meta, dict):
                    meta = {}
                aid = str(meta.get("import_asset_id", "")).strip()
                if not aid:
                    continue
                row = groups.setdefault(
                    aid,
                    {
                        "name": str(meta.get("import_asset_name", "Imported asset")),
                        "path": str(meta.get("import_asset_path", "")),
                        "visible": 0,
                        "locked": 0,
                        "ids": [],
                    },
                )
                if bool(getattr(obj, "visible", True)):
                    row["visible"] = int(row.get("visible", 0)) + 1
                if bool(getattr(obj, "locked", False)):
                    row["locked"] = int(row.get("locked", 0)) + 1
                row["ids"].append(str(oid))
            for aid, row in sorted(groups.items(), key=lambda x: str(x[1].get("name", "")).lower()):
                total = int(len(row["ids"]))
                vis = int(row.get("visible", 0))
                lock = int(row.get("locked", 0))
                label = f"{row['name']} | {total} bodies | vis {vis}/{total} | lock {lock}"
                item = QListWidgetItem(label)
                item.setData(Qt.UserRole, {"asset_id": aid, "ids": list(row["ids"]), "path": row.get("path", "")})
                icon = self._style_icon(QStyle.SP_DirIcon)
                if icon is not None:
                    item.setIcon(icon)
                tip = f"Asset: {row['name']}\nPath: {row.get('path', '-')}\nBodies: {total}\nVisible: {vis}\nLocked: {lock}"
                item.setToolTip(tip)
                self.components_imported_list.addItem(item)
        if hasattr(self, "components_instances_list"):
            self.components_instances_list.clear()
            comp_groups: Dict[str, List[str]] = {}
            for oid, obj in self.engine.objects.items():
                meta = getattr(obj, "meta", {})
                if not isinstance(meta, dict):
                    continue
                comp = str(meta.get("component_name", "")).strip()
                if not comp:
                    continue
                comp_groups.setdefault(comp, []).append(str(oid))
            for name, ids in sorted(comp_groups.items(), key=lambda x: x[0].lower()):
                item = QListWidgetItem(f"{name} ({len(ids)} instance(s))")
                item.setData(Qt.UserRole, {"component_name": name, "ids": list(ids)})
                icon = self._style_icon(QStyle.SP_FileLinkIcon)
                if icon is None:
                    icon = self._style_icon(QStyle.SP_FileIcon)
                if icon is not None:
                    item.setIcon(icon)
                item.setToolTip(f"Component: {name}\nInstances: {len(ids)}")
                self.components_instances_list.addItem(item)
        if hasattr(self, "components_tabs"):
            imported_count = int(self.components_imported_list.count()) if hasattr(self, "components_imported_list") else 0
            instances_count = int(self.components_instances_list.count()) if hasattr(self, "components_instances_list") else 0
            self.components_tabs.setTabText(2, f"Imported ({imported_count})")
            self.components_tabs.setTabText(3, f"Instances ({instances_count})")
        self._filter_components_library()

    def _refresh_fem_panels(self):
        studies = list(self.engine.fem.list_studies())
        active_id = str(self.engine.fem.active_study_id() or "")

        if hasattr(self, "studies_list"):
            self.studies_list.blockSignals(True)
            self.studies_list.clear()
            for row in studies:
                sid = str(row.get("id", ""))
                name = str(row.get("name", sid))
                item = QListWidgetItem(f"{name} [{row.get('study_type', '')}]")
                item.setData(Qt.UserRole, sid)
                self.studies_list.addItem(item)
                if sid == active_id:
                    item.setSelected(True)
            self.studies_list.blockSignals(False)

        if hasattr(self, "fem_study_combo"):
            current = str(self.fem_study_combo.currentData() or "")
            self.fem_study_combo.blockSignals(True)
            self.fem_study_combo.clear()
            for row in studies:
                sid = str(row.get("id", ""))
                name = str(row.get("name", sid))
                self.fem_study_combo.addItem(f"{name} [{row.get('study_type', '')}]", sid)
            target = active_id or current
            if target:
                for idx in range(self.fem_study_combo.count()):
                    if str(self.fem_study_combo.itemData(idx)) == target:
                        self.fem_study_combo.setCurrentIndex(idx)
                        break
            self.fem_study_combo.blockSignals(False)

        study = self.engine.fem.get(active_id)
        if study is None:
            if hasattr(self, "fem_study_summary"):
                self.fem_study_summary.setPlainText("No active FEM study.")
            for key in ["fem_bcs_box", "fem_loads_box", "fem_mesh_box", "fem_validation_box", "fem_results_box"]:
                box = getattr(self, key, None)
                if box is not None:
                    box.setPlainText("")
            return

        if hasattr(self, "fem_study_type_combo"):
            idx = self.fem_study_type_combo.findText(str(study.study_type))
            if idx >= 0:
                self.fem_study_type_combo.blockSignals(True)
                self.fem_study_type_combo.setCurrentIndex(idx)
                self.fem_study_type_combo.blockSignals(False)
        if hasattr(self, "fem_units_combo"):
            idx = self.fem_units_combo.findText(str(study.units))
            if idx >= 0:
                self.fem_units_combo.blockSignals(True)
                self.fem_units_combo.setCurrentIndex(idx)
                self.fem_units_combo.blockSignals(False)

        if hasattr(self, "fem_study_summary"):
            active_bodies = [oid for oid, row in study.bodies.items() if not bool(row.get("exclude_from_solve", False))]
            lines = [
                f"Study: {study.name}",
                f"Type: {study.study_type}",
                f"Units: {study.units}",
                f"Bodies(active): {len(active_bodies)} / {len(study.bodies)}",
                f"Materials: {len(study.materials)}",
                f"Contacts: {len(study.contacts)}",
                f"BCs: {len(study.bcs)}",
                f"Loads: {len(study.loads)}",
            ]
            self.fem_study_summary.setPlainText("\n".join(lines))

        if hasattr(self, "fem_bcs_box"):
            lines = [f"{row.get('type', '')} -> {len(row.get('targets', []))} target(s)" for row in study.bcs]
            self.fem_bcs_box.setPlainText("\n".join(lines) if lines else "No BCs.")
        if hasattr(self, "fem_loads_box"):
            lines = [f"{row.get('type', '')} -> {len(row.get('targets', []))} target(s)" for row in study.loads]
            self.fem_loads_box.setPlainText("\n".join(lines) if lines else "No loads.")
        if hasattr(self, "fem_mesh_box"):
            self.fem_mesh_box.setPlainText(json.dumps(study.mesh, ensure_ascii=False, indent=2))
        if hasattr(self, "fem_validation_box"):
            checks = list(study.validation or [])
            lines = []
            if not checks:
                lines.append("Validation not executed yet.")
            else:
                for row in checks:
                    st = str(row.get("status", "")).lower()
                    tag = "[OK]"
                    if st == "warning":
                        tag = "[WARN]"
                    elif st == "error":
                        tag = "[ERROR]"
                    lines.append(f"{tag} {row.get('label', '')}")
                    lines.append(f"    {row.get('message', '')}")
            self.fem_validation_box.setPlainText("\n".join(lines))
        if hasattr(self, "fem_results_box"):
            self.fem_results_box.setPlainText(json.dumps(study.results, ensure_ascii=False, indent=2) if study.results else "No results yet.")

    def _refresh_properties(self):
        oid = self.engine.selection[0] if self.engine.selection else ""
        if not oid or oid not in self.engine.objects:
            self.properties.set_object_info("", None)
            return
        obj = self.engine.objects[oid]
        metrics = object_metrics(obj)
        metrics["boundary_count"] = int(len(self.engine.boundaries_for_object(oid)))
        metrics["layer"] = str(obj.meta.get("layer", "Default") if isinstance(obj.meta, dict) else "Default")
        if isinstance(obj.meta, dict):
            metrics["material"] = str(obj.meta.get("material", obj.meta.get("obj_material", "")) or "-")
            metrics["fem_role"] = str(obj.meta.get("fem_role", "solid") or "solid")
            metrics["exclude_from_solve"] = bool(obj.meta.get("exclude_from_solve", False))
        data = {
            "name": obj.name,
            "source": obj.source,
            "visible": obj.visible,
            "locked": obj.locked,
            "metrics": metrics,
        }
        self.properties.set_object_info(oid, data)

    def _analysis_tab_key(self, oid: str) -> str:
        return f"analysis::{str(oid)}"

    def _build_analysis_tab(self, oid: str) -> QWidget:
        w = QWidget(self)
        lay = QVBoxLayout(w)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)
        title = QLabel("", w)
        title.setObjectName("analysis_title")
        lay.addWidget(title)
        info = QTextEdit(w)
        info.setReadOnly(True)
        info.setObjectName("analysis_text")
        lay.addWidget(info, 1)
        row = QHBoxLayout()
        btn_refresh = QPushButton("Refresh", w)
        btn_focus = QPushButton("Focus", w)
        btn_measure = QPushButton("Measure BBox/Area/Volume", w)
        btn_validate = QPushButton("Validate", w)
        btn_close = QPushButton("Close tab", w)
        btn_refresh.clicked.connect(lambda: self._refresh_analysis_tab(oid))
        btn_focus.clicked.connect(lambda: self.action_focus_object(oid))
        btn_measure.clicked.connect(lambda: (self.action_measure_bbox([oid]), self.action_measure_area([oid]), self.action_measure_volume([oid])))
        btn_validate.clicked.connect(lambda: self.action_validate_selected([oid]))
        btn_close.clicked.connect(lambda: self._close_analysis_tab(oid))
        row.addWidget(btn_refresh)
        row.addWidget(btn_focus)
        row.addWidget(btn_measure)
        row.addWidget(btn_validate)
        row.addWidget(btn_close)
        lay.addLayout(row)
        self._analysis_tabs[self._analysis_tab_key(oid)] = w
        self._refresh_analysis_tab(oid)
        return w

    def _refresh_analysis_tab(self, oid: str):
        key = self._analysis_tab_key(oid)
        tab = self._analysis_tabs.get(key)
        if tab is None:
            return
        title = tab.findChild(QLabel, "analysis_title")
        text = tab.findChild(QTextEdit, "analysis_text")
        if title is None or text is None:
            return
        if oid not in self.engine.objects:
            title.setText(f"Object removed: {oid}")
            text.setPlainText("Object not found in scene.")
            return
        obj = self.engine.objects[oid]
        metrics = object_metrics(obj)
        layer = str(obj.meta.get("layer", "Default") if isinstance(obj.meta, dict) else "Default")
        lines = [
            f"ID: {oid}",
            f"Name: {obj.name}",
            f"Source: {obj.source}",
            f"Layer: {layer}",
            f"Visible: {obj.visible}",
            f"Locked: {obj.locked}",
            f"Color: {obj.color}",
            f"Opacity: {obj.opacity:.4g}",
            "",
            "Metrics:",
        ]
        for k in sorted(metrics.keys()):
            v = metrics[k]
            if isinstance(v, float):
                lines.append(f"  {k}: {v:.6g}")
            else:
                lines.append(f"  {k}: {v}")
        bcs = self.engine.boundaries_for_object(oid)
        lines.append("")
        lines.append(f"Boundaries: {len(bcs)}")
        for row in bcs:
            lines.append(f"  - {row.get('name')} [{row.get('type')}]")
        title.setText(f"Object Analysis: {obj.name}")
        text.setPlainText("\n".join(lines))

    def _close_analysis_tab(self, oid: str):
        key = self._analysis_tab_key(oid)
        tab = self._analysis_tabs.pop(key, None)
        if tab is None:
            return
        idx = self.ops_tabs.indexOf(tab)
        if idx >= 0:
            self.ops_tabs.removeTab(idx)

    def action_open_selected_analysis_tab(self, ids: Sequence[str] = ()):
        targets = [str(i) for i in ids if str(i) in self.engine.objects] or self._pick_selected()
        if not targets:
            QMessageBox.information(self, "Analysis tab", "Select one object first.")
            return
        oid = str(targets[0])
        key = self._analysis_tab_key(oid)
        tab = self._analysis_tabs.get(key)
        if tab is None:
            tab = self._build_analysis_tab(oid)
            label = str(self.engine.objects[oid].name) if oid in self.engine.objects else oid
            self.ops_tabs.addTab(tab, f"Analysis:{label[:18]}")
        self._refresh_analysis_tab(oid)
        idx = self.ops_tabs.indexOf(tab)
        if idx >= 0:
            self.ops_tabs.setCurrentIndex(idx)

    def _on_tree_selection(self, ids: list, mode: str):
        if self._selection_sync_lock:
            return
        self.selection_manager.set_mode(SelectionMode.from_value(self._subselect_mode))
        self.selection_manager.set_from_ids(ids, self.engine.objects, op=mode)

    def _on_tree_visibility_change(self, ids: list, visible: bool):
        targets = [str(i) for i in ids if str(i) in self.engine.objects]
        if targets:
            self.engine.set_visibility(targets, bool(visible))

    def _on_viewport_selection(self, ids: list, mode: str):
        if self._selection_sync_lock:
            return
        self.selection_manager.set_mode(SelectionMode.from_value(self._subselect_mode))
        self.selection_manager.set_from_ids(ids, self.engine.objects, op=mode)

    def _on_selection_manager_changed(self, items: list):
        if self._selection_sync_lock:
            return
        ids = self.selection_manager.selected_ids()
        if ids == list(self.engine.selection):
            self._refresh_properties()
            self._refresh_selection_info_box()
            return
        self.engine.select(ids, mode="replace")
        self._refresh_selection_info_box()

    def _on_selection_active_item_changed(self, item: Optional[SelectionItem]):
        if item is None:
            self._set_hint("Selection cleared.")
            return
        self._set_hint(f"Selected: {item.display_name or item.entity_id}")
        self._refresh_selection_info_box()

    def _on_selection_hover_changed(self, item: Optional[SelectionItem]):
        oid = ""
        if item is not None:
            oid = str(item.parent_object_id or item.entity_id)
            if ":" in oid and str(item.entity_type) in {"face", "edge", "vertex"}:
                oid = str(item.parent_object_id or "")
        self.viewport.set_hover_object(oid)

    def _on_selection_mode_changed(self, mode: str):
        token = str(mode or "object").strip().lower()
        text = token.capitalize()
        self._subselect_mode = token
        self.viewport.set_selection_mode_indicator(token)
        for combo in [getattr(self, "subselect_combo", None), getattr(self, "tr_subselect_combo", None)]:
            if combo is None:
                continue
            combo.blockSignals(True)
            idx = combo.findText(text)
            if idx >= 0:
                combo.setCurrentIndex(idx)
            combo.blockSignals(False)
        self._set_hint(f"Selection mode: {text}")

    def _on_viewport_hover_info(self, payload: dict):
        row = dict(payload or {})
        oid = str(row.get("object_id", "")).strip()
        if not oid or oid not in self.engine.objects:
            self.selection_manager.set_hover(None)
            return
        obj = self.engine.objects[oid]
        meta = dict(obj.meta) if isinstance(obj.meta, dict) else {}
        item = SelectionItem(
            entity_id=str(oid),
            entity_type="object",
            parent_object_id=None,
            sub_index=None,
            display_name=str(obj.name),
            layer=str(meta.get("layer", "Default") or "Default"),
            locked=bool(obj.locked),
            visible=bool(obj.visible),
            metadata={"picked_cell_id": row.get("picked_cell_id"), "picked_point": row.get("picked_point")},
        )
        self.selection_manager.set_hover(item)

    def _on_viewport_quick_action(self, token: str):
        op = str(token or "").strip().lower()
        if op == "screenshot":
            self.action_screenshot()
        elif op == "toggle_panels":
            self.action_toggle_panels()
        elif op == "selection_cycle":
            self.action_cycle_selection_mode()
        elif op == "toggle_snap":
            self.action_toggle_snap()
        elif op == "toggle_lod":
            self.action_toggle_navigation_lod()

    def _on_viewport_pick_info(self, payload: dict):
        row = dict(payload or {})
        oid = str(row.get("object_id", "")).strip()
        point = row.get("picked_point")
        cell_raw = row.get("picked_cell_id")
        cell_id = int(cell_raw) if isinstance(cell_raw, (int, float)) and int(cell_raw) >= 0 else None
        p3 = tuple(float(x) for x in point) if isinstance(point, (list, tuple)) and len(point) == 3 else None
        mode = str(self._subselect_mode or "object").strip().lower()
        op = str(row.get("selection_mode", "replace") or "replace").strip().lower()
        if oid and oid in self.engine.objects:
            obj = self.engine.objects[oid]
            meta = dict(obj.meta) if isinstance(obj.meta, dict) else {}
            row["display_name"] = str(obj.name)
            row["layer"] = str(meta.get("layer", "Default") or "Default")
            row["locked"] = bool(obj.locked)
            row["visible"] = bool(obj.visible)
        self.selection_manager.set_mode(SelectionMode.from_value(mode))
        self.selection_manager.set_from_viewport_pick(row, op=op)
        if not oid or oid not in self.engine.objects:
            if mode in {"face", "edge"}:
                self._clear_subselection_state()
            return
        self._last_cursor_ctx = ContextInfo(
            widget="viewport",
            picked_object_id=oid,
            picked_cell_id=cell_id,
            selected_ids=[oid],
            tool_mode=str(self.engine.tool_mode),
            picked_point_3d=p3,
        )
        if mode == "face":
            if cell_id is None:
                return
            self._selected_face = (oid, int(cell_id))
            self._selected_edge = None
            self._selected_vertex = None
            self.viewport.set_subselection(face=(oid, int(cell_id)), edge=None)
            self._update_face_edge_info_label()
        elif mode == "edge":
            if cell_id is None:
                return
            edge = self.engine.pick_edge_from_face(oid, int(cell_id), p3)
            if edge is None:
                return
            self._selected_face = (oid, int(cell_id))
            self._selected_edge = (oid, edge, int(cell_id))
            self._selected_vertex = None
            self.viewport.set_subselection(face=(oid, int(cell_id)), edge=(oid, edge))
            self._update_face_edge_info_label()
        elif mode == "vertex":
            if cell_id is None:
                return
            info = self.engine.face_info(oid, int(cell_id))
            vids = [int(x) for x in info.get("vertex_ids", [])]
            if not vids:
                return
            if p3 is None:
                vid = vids[0]
            else:
                verts = np.asarray(self.engine.objects[oid].mesh.vertices, dtype=float)
                p = np.asarray(p3, dtype=float).reshape(3)
                dists = [(int(v), float(np.linalg.norm(p - verts[int(v)]))) for v in vids if 0 <= int(v) < int(verts.shape[0])]
                if not dists:
                    return
                dists.sort(key=lambda x: x[1])
                vid = int(dists[0][0])
            self._selected_face = (oid, int(cell_id))
            self._selected_edge = None
            self._selected_vertex = (oid, int(vid), int(cell_id))
            self.viewport.set_subselection(face=(oid, int(cell_id)), edge=None)
            self._update_face_edge_info_label()
        else:
            self._clear_subselection_state()

        if bool(getattr(self, "auto_analysis_chk", None).isChecked() if hasattr(self, "auto_analysis_chk") else False):
            self.action_open_selected_analysis_tab([oid])

    def _on_viewport_piece_command(self, payload: dict):
        row = dict(payload or {})
        cmd = str(row.get("command", "")).strip().lower()
        oid = str(row.get("object_id", "")).strip()
        p = row.get("picked_point")
        picked_cell = row.get("picked_cell_id")
        cell_id = int(picked_cell) if isinstance(picked_cell, (int, float)) and int(picked_cell) >= 0 else None
        point = tuple(float(x) for x in p) if isinstance(p, (list, tuple)) and len(p) == 3 else None
        if cmd == "toggle_visibility":
            if oid and oid in self.engine.objects:
                self.action_toggle_visibility([oid])
                self._log(f"Mouse command: toggle visibility on {self.engine.objects[oid].name}")
            return
        if cmd == "apply_boundary":
            if oid and oid in self.engine.objects:
                ctx = ContextInfo(
                    widget="viewport",
                    picked_object_id=oid,
                    picked_cell_id=cell_id,
                    selected_ids=[oid],
                    picked_point_3d=point,
                    tool_mode=str(self.engine.tool_mode),
                )
                btype = str(row.get("boundary_type", "fixed") or "fixed")
                self.action_apply_boundary_quick(btype, ctx)
            return

    def _on_apply_transform(self, payload: dict):
        ids = self.engine.selection
        if not ids:
            return
        self.engine.transform_objects(ids=ids, **payload)

    def action_apply_right_transform(self):
        ids = self._pick_selected()
        if not ids:
            return
        payload = {
            "tx": self._parse_float(self.rt_tx, 0.0),
            "ty": self._parse_float(self.rt_ty, 0.0),
            "tz": self._parse_float(self.rt_tz, 0.0),
            "rx_deg": self._parse_float(self.rt_rx, 0.0),
            "ry_deg": self._parse_float(self.rt_ry, 0.0),
            "rz_deg": self._parse_float(self.rt_rz, 0.0),
            "scale": self._parse_float(self.rt_scale, 1.0),
        }
        self.engine.transform_objects(ids=ids, **payload)

    # ---------------- components ----------------
    def action_components_insert_primitive(self):
        item = getattr(self, "components_primitives_list", None)
        if item is None:
            return
        row = item.currentItem()
        if row is None:
            return
        data = row.data(Qt.UserRole) if row is not None else None
        token = str((data or {}).get("kind", "") if isinstance(data, dict) else "").strip().lower()
        if not token:
            token = str(row.text() or "").strip().lower()
        kind_map = {
            "box": "box",
            "cylinder": "cylinder",
            "cone": "cone",
            "sphere": "sphere",
            "plane": "plane",
            "tube/pipe": "tube",
            "polygon": "plane",
            "wedge": "box",
            "torus": "tube",
        }
        kind = kind_map.get(token, "box")
        self.action_create_primitive_dialog(kind)

    def action_components_create_from_selection(self):
        ids = self._pick_selected()
        if not ids:
            QMessageBox.information(self, "Components", "Select at least one body.")
            return
        name, ok = QInputDialog.getText(self, "Component", "Component name:", text="Component_1")
        if not ok:
            return
        token = str(name or "").strip() or "Component_1"

        def _apply(eng: SceneEngine):
            for oid in ids:
                obj = eng.objects.get(oid)
                if obj is None:
                    continue
                if not isinstance(obj.meta, dict):
                    obj.meta = {}
                obj.meta["component_name"] = str(token)
                obj.meta["component_uuid"] = str(obj.meta.get("component_uuid", obj.id))
                obj.meta["component_source"] = str(obj.source)

        self.engine.execute("Create component from selection", _apply)
        self._log(f"Component '{token}' created from {len(ids)} body(s).")

    def action_components_insert_instance(self):
        widget = getattr(self, "components_instances_list", None)
        if widget is None:
            return
        row = widget.currentItem()
        if row is None:
            QMessageBox.information(self, "Instances", "Select a component from Instances list.")
            return
        data = row.data(Qt.UserRole) or {}
        comp_name = str(data.get("component_name", "")).strip()
        if not comp_name:
            return
        src_ids = [oid for oid, obj in self.engine.objects.items() if str(getattr(obj, "meta", {}).get("component_name", "")) == comp_name]
        if not src_ids:
            QMessageBox.warning(self, "Instances", "No source object found for selected component.")
            return
        created = self.engine.duplicate_objects([src_ids[0]])
        if not created:
            return

        def _apply(eng: SceneEngine):
            for oid in created:
                obj = eng.objects.get(oid)
                if obj is None:
                    continue
                if not isinstance(obj.meta, dict):
                    obj.meta = {}
                obj.meta["component_name"] = str(comp_name)
                obj.meta["component_instance_of"] = str(src_ids[0])
                obj.meta["component_linked"] = True

        self.engine.execute("Insert component instance", _apply)
        self._log(f"Inserted instance of '{comp_name}'.")

    def action_components_insert_template(self):
        widget = getattr(self, "components_templates_list", None)
        if widget is None:
            return
        row = widget.currentItem()
        if row is None:
            return
        data = row.data(Qt.UserRole) if row is not None else None
        name = str((data or {}).get("template", "") if isinstance(data, dict) else "").strip().lower()
        if not name:
            name = str(row.text() or "").strip().lower()
        created = []
        try:
            if "panel" in name:
                created.append(self.engine.create_primitive("box", {"width": 1.2, "depth": 0.2, "height": 2.4}, name="panel_body"))
                created.append(self.engine.create_primitive("tube", {"outer_radius": 0.08, "inner_radius": 0.05, "height": 2.8}, name="panel_support"))
            elif "tower" in name:
                created.append(self.engine.create_primitive("tube", {"outer_radius": 0.4, "inner_radius": 0.32, "height": 10.0}, name="tower_main"))
                created.append(self.engine.create_primitive("box", {"width": 0.3, "depth": 0.3, "height": 3.0}, name="tower_brace"))
            elif "cable" in name:
                created.append(self.engine.create_primitive("tube", {"outer_radius": 0.03, "inner_radius": 0.02, "height": 8.0}, name="coax_1"))
                created.append(self.engine.create_primitive("tube", {"outer_radius": 0.03, "inner_radius": 0.02, "height": 8.0}, name="coax_2"))
            elif "base" in name:
                created.append(self.engine.create_primitive("box", {"width": 2.0, "depth": 2.0, "height": 0.4}, name="base"))
                created.append(self.engine.create_primitive("tube", {"outer_radius": 0.15, "inner_radius": 0.1, "height": 4.0}, name="mast"))
            else:
                created.append(self.engine.create_primitive("box", {"width": 1.0, "depth": 1.0, "height": 1.0}, name="template_body"))
        except Exception as e:
            QMessageBox.warning(self, "Template", str(e))
            return
        self.engine.select(created, mode="replace")
        self._log(f"Template inserted: {len(created)} body(s).")

    # ---------------- fem workflow ----------------
    def _current_fem_study_id(self) -> str:
        sid = ""
        combo = getattr(self, "fem_study_combo", None)
        if combo is not None:
            sid = str(combo.currentData() or "")
        if not sid:
            sid = str(self.engine.fem.active_study_id() or "")
        return sid

    def _ensure_active_fem_study(self) -> str:
        sid = self._current_fem_study_id()
        if sid:
            return sid
        return str(self.engine.fem.new_study(name="Study 1", study_type="Structural Static", units=str(self.units_combo.currentText() if hasattr(self, "units_combo") else "mm")))

    def action_fem_new_study(self):
        name, ok = QInputDialog.getText(self, "New FEM Study", "Study name:", text=f"Study {len(self.engine.fem.list_studies()) + 1}")
        if not ok:
            return
        study_type = str(self.fem_study_type_combo.currentText() if hasattr(self, "fem_study_type_combo") else "Structural Static")
        units = str(self.fem_units_combo.currentText() if hasattr(self, "fem_units_combo") else "mm")
        sid = self.engine.fem.new_study(name=str(name or "").strip(), study_type=study_type, units=units)
        self.engine.fem.set_active(sid)
        self._log(f"FEM study created: {name or sid}")
        self._set_hint(
            "FEM wizard: follow Study -> Materials -> BCs -> Loads -> Mesh -> Solve -> Results.",
            [
                "New FEM Study wizard",
                "1) Include participating bodies",
                "2) Assign materials",
                "3) Define contacts / BCs / loads",
                "4) Configure mesh and solver",
                "5) Validate and run solve",
            ],
        )
        self.refresh_all()

    def action_fem_remove_study(self):
        sid = self._current_fem_study_id()
        if not sid:
            return
        if not self._ask_confirm("FEM", "Remove selected study?"):
            return
        if self.engine.fem.remove_study(sid):
            self._log(f"FEM study removed: {sid}")
            self.refresh_all()

    def action_fem_set_active_from_list(self):
        lst = getattr(self, "studies_list", None)
        if lst is None:
            return
        item = lst.currentItem()
        if item is None:
            return
        sid = str(item.data(Qt.UserRole) or "")
        if sid and self.engine.fem.set_active(sid):
            self._log(f"Active FEM study: {sid}")
            self.refresh_all()

    def action_fem_set_active_from_combo(self, _index: int):
        sid = self._current_fem_study_id()
        if sid and self.engine.fem.set_active(sid):
            self.refresh_all()

    def action_fem_include_selected(self):
        sid = self._ensure_active_fem_study()
        ids = self._pick_selected()
        if not ids:
            QMessageBox.information(self, "FEM", "Select at least one body.")
            return
        role = str(self.fem_role_combo.currentText() if hasattr(self, "fem_role_combo") else "solid")
        n = self.engine.fem.include_bodies(ids, fem_role=role, exclude=False, study_id=sid)
        self._log(f"FEM bodies included: {n}")
        self.refresh_all()

    def action_fem_assign_material_selected(self):
        sid = self._ensure_active_fem_study()
        ids = self._pick_selected()
        if not ids:
            QMessageBox.information(self, "Material", "Select body(s) first.")
            return
        mat = str(self.material_name_combo.currentText() if hasattr(self, "material_name_combo") else "Steel").strip()
        props = {
            "density": self._parse_float(self.material_density_edit, 7850.0) if hasattr(self, "material_density_edit") else 7850.0,
            "E": self._parse_float(self.material_e_edit, 210e9) if hasattr(self, "material_e_edit") else 210e9,
            "nu": self._parse_float(self.material_nu_edit, 0.3) if hasattr(self, "material_nu_edit") else 0.3,
        }
        n = self.engine.fem.assign_material(ids, mat, props, study_id=sid)
        self._log(f"Material '{mat}' assigned to {n} body(s).")
        self.refresh_all()

    def action_fem_add_bc(self, bc_type: str):
        sid = self._ensure_active_fem_study()
        ids = self._pick_selected()
        if not ids:
            QMessageBox.information(self, "BCs", "Select body(s) first.")
            return
        value = self._parse_float(self.fem_bc_value_edit, 0.0) if hasattr(self, "fem_bc_value_edit") else 0.0
        bid = self.engine.fem.add_boundary_condition(ids, str(bc_type), {"value": float(value)}, study_id=sid)
        if bid:
            self._log(f"FEM BC added: {bc_type} ({bid[:8]})")
            self.refresh_all()

    def action_fem_sync_boundaries_as_bcs(self):
        sid = self._ensure_active_fem_study()
        count = 0
        for row in self.engine.boundary_rows():
            targets = [str(x) for x in row.get("targets", []) if str(x) in self.engine.objects]
            if not targets:
                continue
            btype = str(row.get("type", "fixed support") or "fixed support")
            params = dict(row.get("params", {})) if isinstance(row.get("params", {}), dict) else {}
            if self.engine.fem.add_boundary_condition(targets, btype, params, study_id=sid):
                count += 1
        self._log(f"Synced boundary conditions to FEM study: {count}")
        self.refresh_all()

    def action_fem_add_load(self, load_type: str):
        sid = self._ensure_active_fem_study()
        ids = self._pick_selected()
        if not ids:
            QMessageBox.information(self, "Loads", "Select body(s) first.")
            return
        value = self._parse_float(self.fem_load_value_edit, 1.0) if hasattr(self, "fem_load_value_edit") else 1.0
        direction = str(self.fem_load_dir_combo.currentText() if hasattr(self, "fem_load_dir_combo") else "normal").lower()
        lid = self.engine.fem.add_load(ids, str(load_type), {"value": float(value), "direction": direction}, study_id=sid)
        if lid:
            self._log(f"FEM load added: {load_type} ({lid[:8]})")
            self.refresh_all()

    def action_fem_apply_mesh_config(self):
        sid = self._ensure_active_fem_study()
        ok = self.engine.fem.configure_mesh(
            study_id=sid,
            global_size=self._parse_float(self.fem_mesh_size_edit, 20.0) if hasattr(self, "fem_mesh_size_edit") else 20.0,
            growth_rate=self._parse_float(self.fem_mesh_growth_edit, 1.2) if hasattr(self, "fem_mesh_growth_edit") else 1.2,
            quality_target=self._parse_float(self.fem_mesh_quality_edit, 0.7) if hasattr(self, "fem_mesh_quality_edit") else 0.7,
            curvature_refinement=bool(self.fem_mesh_curvature_chk.isChecked()) if hasattr(self, "fem_mesh_curvature_chk") else True,
        )
        if ok:
            self._log("FEM mesh configuration updated.")
            self.refresh_all()

    def action_fem_generate_mesh(self):
        sid = self._ensure_active_fem_study()
        self.action_fem_apply_mesh_config()
        q = self._parse_float(self.fem_mesh_quality_edit, 0.7) if hasattr(self, "fem_mesh_quality_edit") else 0.7
        ok = self.engine.fem.mark_mesh_generated(True, quality_avg=q, study_id=sid)
        if ok:
            self._solver_log(f"Mesh generated (quality={q:.3f}).")
            self._log("FEM mesh generated.")
            self.refresh_all()

    def action_fem_apply_solver_config(self):
        sid = self._ensure_active_fem_study()
        ok = self.engine.fem.configure_solver(
            study_id=sid,
            type=str(self.fem_solver_type_combo.currentText() if hasattr(self, "fem_solver_type_combo") else "direct"),
            tolerance=self._parse_float(self.fem_solver_tol_edit, 1e-6) if hasattr(self, "fem_solver_tol_edit") else 1e-6,
            max_iterations=self._parse_int(self.fem_solver_it_edit, 500) if hasattr(self, "fem_solver_it_edit") else 500,
            threads=self._parse_int(self.fem_solver_threads_edit, 0) if hasattr(self, "fem_solver_threads_edit") else 0,
        )
        if ok:
            self._log("FEM solver configuration updated.")
            self.refresh_all()

    def action_fem_validate_study(self):
        sid = self._ensure_active_fem_study()
        checks = self.engine.fem.validate(sid)
        errs = sum(1 for row in checks if str(row.get("status", "")).lower() == "error")
        warns = sum(1 for row in checks if str(row.get("status", "")).lower() == "warning")
        self._log(f"FEM validation: {len(checks)} checks, errors={errs}, warnings={warns}.")
        self._solver_log(f"Validation complete: checks={len(checks)} errors={errs} warnings={warns}")
        self.refresh_all()
        return checks

    def _solver_log(self, text: str):
        msg = str(text or "").strip()
        if not msg:
            return
        if hasattr(self, "solver_console_box"):
            self.solver_console_box.append(msg)

    def action_fem_run_solve(self):
        sid = self._ensure_active_fem_study()
        self.action_fem_apply_solver_config()
        try:
            def _progress(pct: int, msg: str):
                self._solver_log(msg)

            self._solver_log(f"Solve started: study={sid}")
            out = self.engine.fem.run_solve(sid, progress_cb=_progress)
            self._fem_result_cache[str(sid)] = dict(out)
            self._solver_log("Solve finished successfully.")
            self._log("FEM solve finished.")
            self.refresh_all()
            self.action_fem_refresh_results()
        except Exception as e:
            self._solver_log(f"Solve error: {e}")
            QMessageBox.warning(self, "FEM solve", str(e))

    def action_fem_refresh_results(self):
        self.refresh_all()
        if hasattr(self, "right_tabs") and hasattr(self, "fem_results_box"):
            idx = self.right_tabs.indexOf(self.fem_results_box.parentWidget())
            if idx >= 0:
                self.right_tabs.setCurrentIndex(idx)

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
            "Tab: toggle side panels\nShift+Tab: toggle bottom panel\nF11: max viewport\nEsc: clear selection\nDel: delete selected\nCtrl+Z/Ctrl+Y: undo/redo\nShift+Click: add selection\nCtrl+Click on piece: toggle selection\nCtrl+Alt+Click: toggle visibility\nAlt+Click on piece: apply fixed boundary\nShift+Alt+Click: apply force boundary\nMouse wheel: zoom at cursor focus\nCtrl/Shift + wheel: fine zoom (detalhes)\nAlt + wheel: fast zoom",
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

    def action_focus_selection_detail(self):
        ids = self._pick_selected()
        if ids:
            self.viewport.focus_on_ids(ids, detail=True)
        else:
            self.viewport.fit_all()

    def action_toggle_left_panel(self):
        if self._panels_collapsed:
            self.action_toggle_panels()
            return
        self._capture_panel_restore_sizes()
        self.left_tabs.setVisible(not self.left_tabs.isVisible())
        self._apply_splitter_sizes_from_visibility()
        self._set_hint("Left panel hidden." if not self.left_tabs.isVisible() else "Left panel restored.")

    def action_toggle_right_panel(self):
        if self._panels_collapsed:
            self.action_toggle_panels()
            return
        self._capture_panel_restore_sizes()
        self.right_wrap.setVisible(not self.right_wrap.isVisible())
        self._apply_splitter_sizes_from_visibility()
        self._set_hint("Right panel hidden." if not self.right_wrap.isVisible() else "Right panel restored.")

    def action_toggle_bottom_panel(self):
        if self._panels_collapsed:
            self.action_toggle_panels()
            return
        self._capture_panel_restore_sizes()
        self.bottom_tabs.setVisible(not self.bottom_tabs.isVisible())
        self._apply_splitter_sizes_from_visibility()
        self._set_hint("Bottom panel hidden." if not self.bottom_tabs.isVisible() else "Bottom panel restored.")

    def action_toggle_side_panels(self):
        if self._panels_collapsed:
            self.action_toggle_panels()
            return
        self._capture_panel_restore_sizes()
        visible = bool(self.left_tabs.isVisible() or self.right_wrap.isVisible())
        self.left_tabs.setVisible(not visible)
        self.right_wrap.setVisible(not visible)
        self._apply_splitter_sizes_from_visibility()
        self._set_hint("Side panels hidden (focus mode)." if visible else "Side panels restored.")

    def action_toggle_panels(self):
        if self._panels_collapsed:
            self._panels_collapsed = False
            state = dict(self._max_viewport_state or {})
            left_on = bool(state.get("left", True))
            right_on = bool(state.get("right", True))
            bottom_on = bool(state.get("bottom", True))
            ribbon_on = bool(state.get("ribbon", True))
            ops_on = bool(state.get("ops", True))
            self._set_panel_visibility(left=left_on, right=right_on, bottom=bottom_on, ribbon=ribbon_on, ops=ops_on, refresh_sizes=False)
            if hasattr(self, "btn_toggle_panels"):
                self.btn_toggle_panels.setText("Max Viewport")
            try:
                self.main_split.setSizes([int(x) for x in state.get("main_sizes", self._last_main_sizes or [260, 1340, 320])])
                self.outer_split.setSizes([int(x) for x in state.get("outer_sizes", self._last_outer_sizes or [900, 120])])
            except Exception:
                pass
            self._capture_panel_restore_sizes()
            self._max_viewport_state = None
            self._set_hint("Viewport normal mode restored.")
            return

        self._panels_collapsed = True
        self._capture_panel_restore_sizes()
        self._max_viewport_state = {
            "left": bool(self.left_tabs.isVisible()),
            "right": bool(self.right_wrap.isVisible()),
            "bottom": bool(self.bottom_tabs.isVisible()),
            "ribbon": bool(self.ribbon_tabs.isVisible()),
            "ops": bool(self.ops_tabs.isVisible()),
            "main_sizes": [int(x) for x in (self._last_main_sizes or [260, 1340, 320])],
            "outer_sizes": [int(x) for x in (self._last_outer_sizes or [900, 120])],
        }
        self._set_panel_visibility(left=False, right=False, bottom=False, ribbon=False, ops=False, refresh_sizes=True)
        if hasattr(self, "btn_toggle_panels"):
            self.btn_toggle_panels.setText("Restore Panels")
        self._set_hint("Viewport maximized. Use 'Restore Panels' or UI button in viewport toolbar.")

    def action_toggle_grid(self):
        self._grid_enabled = not bool(self._grid_enabled)
        self._prefs.setdefault("grid", {})
        self._prefs["grid"]["enabled"] = bool(self._grid_enabled)
        self._save_preferences()
        if hasattr(self, "grid_enable_chk"):
            self.grid_enable_chk.blockSignals(True)
            self.grid_enable_chk.setChecked(bool(self._grid_enabled))
            self.grid_enable_chk.blockSignals(False)
        self.viewport.set_grid_config(enabled=self._grid_enabled)

    def action_toggle_axes(self):
        self.viewport.toggle_axes()

    def action_apply_grid_settings(self):
        enabled = bool(self.grid_enable_chk.isChecked()) if hasattr(self, "grid_enable_chk") else True
        step = self._parse_float(self.grid_step_edit, self._grid_step_mm) if hasattr(self, "grid_step_edit") else self._grid_step_mm
        size = self._parse_float(self.grid_size_edit, self._grid_size_mm) if hasattr(self, "grid_size_edit") else self._grid_size_mm
        color = str(self.grid_color_edit.text() or self._grid_color).strip() if hasattr(self, "grid_color_edit") else self._grid_color
        self._grid_enabled = bool(enabled)
        self._grid_step_mm = max(1e-6, float(step))
        self._grid_size_mm = max(self._grid_step_mm * 2.0, float(size))
        self._grid_color = color or "#2b2b2b"
        self._snap_step = float(self._grid_step_mm)
        self._prefs.setdefault("grid", {})
        self._prefs["grid"]["enabled"] = bool(self._grid_enabled)
        self._prefs["grid"]["step_mm"] = float(self._grid_step_mm)
        self._prefs["grid"]["size_mm"] = float(self._grid_size_mm)
        self._prefs["grid"]["color"] = str(self._grid_color)
        self._prefs.setdefault("snap", {})
        self._prefs["snap"]["step_mm"] = float(self._snap_step)
        if hasattr(self, "tr_step_edit"):
            self.tr_step_edit.setText(f"{self._snap_step:.6g}")
        self._save_preferences()
        self.viewport.set_grid_config(enabled=self._grid_enabled, step=self._grid_step_mm, size=self._grid_size_mm, color=self._grid_color)
        self._set_hint("Grid configuration updated.")

    def action_set_render_mode(self, mode: str):
        try:
            self.viewport.set_render_mode(mode)
        except Exception as e:
            self._log(f"Render mode fallback due to error: {e}")
            self.refresh_all()

    def action_set_background(self, mode: str):
        try:
            self.viewport.set_background(mode)
        except Exception as e:
            self._log(f"Background fallback due to error: {e}")
            self.refresh_all()

    def action_set_visual_preset(self, mode: str):
        token = str(mode or "").strip().lower()
        self._prefs.setdefault("render", {})
        render_mode = "solid_edges"
        xray_mode = False
        if token == "x-ray":
            render_mode = "solid_edges"
            xray_mode = True
        elif token == "wireframe":
            render_mode = "wireframe"
            xray_mode = False
        elif token == "solid":
            render_mode = "solid"
            xray_mode = False
        else:
            render_mode = "solid_edges"
            xray_mode = False

        self._xray_mode = bool(xray_mode)
        self._prefs["render"]["xray"] = bool(xray_mode)
        self._prefs["render"]["mode"] = str(render_mode)
        self._save_preferences()
        try:
            self.viewport.set_xray_mode(bool(xray_mode))
            self.viewport.set_render_mode(str(render_mode))
            self._set_hint(f"Visualization preset: {str(mode)}.")
        except Exception as e:
            self._log(f"Visualization change fallback: {e}")
            self.refresh_all()

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

    def action_assign_selected_layer(self, ids: Sequence[str] = ()):
        targets = [str(i) for i in ids if str(i) in self.engine.objects] or self._pick_selected()
        if not targets:
            QMessageBox.information(self, "Layers", "Select at least one object.")
            return
        layer = self._current_layer_name()
        if not layer:
            layer, ok = QInputDialog.getText(self, "Layer", "Layer name:", text="Default")
            if not ok:
                return
        self.engine.set_objects_layer(targets, layer)

    def action_new_layer(self):
        name, ok = QInputDialog.getText(self, "New layer", "Layer name:", text="Layer_1")
        if not ok:
            return
        token = str(name or "").strip()
        if not token:
            return
        color = "#86b6f6"
        col = QColorDialog.getColor(QColor(color), self, "Layer color")
        if col.isValid():
            color = col.name()

        def _apply(eng: SceneEngine):
            eng._ensure_layer(token, color=color, visible=True, locked=False)

        self.engine.execute(f"Create layer {token}", _apply)

    def action_rename_layer(self):
        old_name = self._current_layer_name()
        if not old_name:
            return
        new_name, ok = QInputDialog.getText(self, "Rename layer", "New layer name:", text=str(old_name))
        if not ok:
            return
        token = str(new_name or "").strip()
        if not token:
            return
        try:
            changed = self.engine.rename_layer(old_name, token)
        except Exception as e:
            QMessageBox.warning(self, "Rename layer", str(e))
            return
        if not changed:
            QMessageBox.information(self, "Rename layer", "Layer not found.")

    def action_delete_layer(self):
        token = self._current_layer_name()
        if not token:
            return
        if str(token) == "Default":
            QMessageBox.information(self, "Layers", "Default layer cannot be deleted.")
            return
        if not self._ask_confirm("Delete layer", f"Delete layer '{token}' and move objects to Default?"):
            return
        self.engine.delete_layer(token, fallback_layer="Default")

    def _layer_from_targets(self, ids: Sequence[str]) -> str:
        targets = [str(i) for i in ids if str(i) in self.engine.objects]
        if not targets:
            return ""
        obj = self.engine.objects.get(targets[0])
        if obj is None:
            return ""
        return str(obj.meta.get("layer", "Default") if isinstance(obj.meta, dict) else "Default")

    def action_toggle_layer_visibility(self, ids: Sequence[str] = ()):
        token = self._layer_from_targets(ids) or self._current_layer_name()
        if not token:
            return
        rows = {str(r.get("name", "")): dict(r) for r in self.engine.layer_rows()}
        if token not in rows:
            return
        base = bool(rows[token].get("visible", True))
        self.engine.set_layer_visibility(token, not base)

    def action_set_layer_color(self, ids: Sequence[str] = ()):
        token = self._layer_from_targets(ids) or self._current_layer_name()
        if not token:
            return
        rows = {str(r.get("name", "")): dict(r) for r in self.engine.layer_rows()}
        start = str(rows.get(token, {}).get("color", "#86b6f6"))
        col = QColorDialog.getColor(QColor(start), self, f"Layer color: {token}")
        if not col.isValid():
            return
        apply_mode, ok = QInputDialog.getItem(
            self,
            "Layer color",
            "Apply to existing objects:",
            ["Yes", "No"],
            1,
            False,
        )
        if not ok:
            return
        apply_objects = str(apply_mode) == "Yes"
        self.engine.set_layer_color(token, col.name(), apply_to_objects=apply_objects)

    def _update_face_edge_info_label(self):
        if not hasattr(self, "face_edge_info_label"):
            return
        if self._selected_vertex is not None:
            oid, vid, face_idx = self._selected_vertex
            name = self.engine.objects.get(str(oid)).name if str(oid) in self.engine.objects else str(oid)
            self.face_edge_info_label.setText(f"Face/Edge: {name} vertex({vid}) on face {face_idx}")
            return
        if self._selected_edge is not None:
            oid, edge, face_idx = self._selected_edge
            name = self.engine.objects.get(str(oid)).name if str(oid) in self.engine.objects else str(oid)
            self.face_edge_info_label.setText(f"Face/Edge: {name} edge({edge[0]}-{edge[1]}) on face {face_idx}")
            return
        if self._selected_face is not None:
            oid, face_idx = self._selected_face
            name = self.engine.objects.get(str(oid)).name if str(oid) in self.engine.objects else str(oid)
            self.face_edge_info_label.setText(f"Face/Edge: {name} face {face_idx}")
            return
        self.face_edge_info_label.setText("Face/Edge: -")

    def _sync_subselect_combos(self):
        token = str(self._subselect_mode or "object").strip().lower()
        for key in ("subselect_combo", "tr_subselect_combo"):
            cmb = getattr(self, key, None)
            if cmb is None:
                continue
            idx = cmb.findText(token.capitalize())
            if idx >= 0 and int(cmb.currentIndex()) != idx:
                cmb.blockSignals(True)
                cmb.setCurrentIndex(idx)
                cmb.blockSignals(False)

    def _clear_subselection_state(self):
        self._selected_face = None
        self._selected_edge = None
        self._selected_vertex = None
        self._update_face_edge_info_label()
        self.viewport.clear_subselection()

    def action_set_subselection_mode(self, mode: str):
        token = str(mode or "Object").strip().lower()
        if token not in {"object", "face", "edge", "vertex", "body", "component"}:
            token = "object"
        self._subselect_mode = token
        self.viewport.set_selection_mode_indicator(token)
        self.selection_manager.set_mode(SelectionMode.from_value(token))
        self._prefs.setdefault("selection", {})
        self._prefs["selection"]["sub_mode"] = str(token)
        self._save_preferences()
        self._sync_subselect_combos()
        if token == "object":
            self._clear_subselection_state()
        self._set_hint(
            f"Left-click mode: {token}.",
            [
                "Selection mode",
                "Object: select body.",
                "Face: pick triangle face and adjust offset.",
                "Edge: pick nearest edge on clicked face and adjust offset.",
                "Vertex: pick nearest vertex from picked face.",
                "Body/Component: object-level selection with dedicated filter context.",
            ],
        )

    def _on_auto_analysis_toggle(self, value: bool):
        self._prefs.setdefault("selection", {})
        self._prefs["selection"]["auto_analysis_tab_on_click"] = bool(value)
        self._save_preferences()

    def action_adjust_selected_face(self, sign: float = 1.0):
        if self._selected_face is None:
            QMessageBox.information(self, "Face adjust", "Pick a face first (Pick mode = Face).")
            return
        oid, face_idx = self._selected_face
        step = abs(self._parse_float(self.face_edge_offset_edit, 0.1)) if hasattr(self, "face_edge_offset_edit") else 0.1
        off = float(sign) * float(step)
        ok = self.engine.adjust_face_offset(str(oid), int(face_idx), off)
        if ok:
            self._set_hint(f"Face adjusted by {off:.6g}.")
        else:
            QMessageBox.warning(self, "Face adjust", "Unable to adjust selected face.")

    def action_adjust_selected_edge(self, sign: float = 1.0):
        if self._selected_edge is None:
            QMessageBox.information(self, "Edge adjust", "Pick an edge first (Pick mode = Edge).")
            return
        oid, edge, _face_idx = self._selected_edge
        step = abs(self._parse_float(self.face_edge_offset_edit, 0.1)) if hasattr(self, "face_edge_offset_edit") else 0.1
        off = float(sign) * float(step)
        ok = self.engine.adjust_edge_offset(str(oid), edge, off)
        if ok:
            self._set_hint(f"Edge adjusted by {off:.6g}.")
        else:
            QMessageBox.warning(self, "Edge adjust", "Unable to adjust selected edge.")

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
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Import mesh/model",
            "",
            "CAD/Mesh (*.obj *.stl *.ply *.step *.stp *.iges *.igs)",
        )
        if not paths:
            return
        self._mesh_deflection = max(1e-6, self._parse_float(self.import_deflection_edit, self._mesh_deflection))
        self._prefs.setdefault("import", {})
        self._prefs["import"]["deflection"] = float(self._mesh_deflection)
        self._save_preferences()
        self.engine.set_default_mesh_quality({"deflection": self._mesh_deflection})
        target_mode = str(self._prefs.get("import", {}).get("target_layer", "auto") or "auto").strip().lower()
        imported = 0
        for p in paths:
            try:
                ext = os.path.splitext(p)[1].lower().lstrip(".")
                hint = os.path.splitext(os.path.basename(p))[0]
                layer_name = "Default" if target_mode == "default" else hint
                ids = self.engine.import_model(
                    p,
                    fmt=ext,
                    name_hint=hint,
                    layer_name=layer_name,
                    triangulation_quality={"deflection": self._mesh_deflection},
                )
                imported += int(len(ids))
                if len(ids) > 1:
                    self._log(f"Assembly imported: {os.path.basename(p)} -> {len(ids)} bodies")
            except Exception as e:
                self._log(f"Import failed {p}: {e}")
        self._log(f"Imported meshes: {imported} (deflection={self._mesh_deflection:.6g})")

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
        elif token == "tube":
            ro, ok = QInputDialog.getDouble(self, "Primitive", "Outer radius:", 0.6, 1e-6, 1e9, 6)
            if not ok:
                return
            ri, ok = QInputDialog.getDouble(self, "Primitive", "Inner radius:", 0.3, 1e-6, ro, 6)
            if not ok:
                return
            h, ok = QInputDialog.getDouble(self, "Primitive", "Height:", 1.0, 1e-6, 1e9, 6)
            if not ok:
                return
            params["outer_radius"] = ro
            params["inner_radius"] = ri
            params["height"] = h
            params["segments"] = 48
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

    def action_cycle_selection_mode(self):
        order = ["object", "face", "edge", "vertex", "body", "component"]
        token = str(self._subselect_mode or "object").strip().lower()
        try:
            idx = order.index(token)
        except ValueError:
            idx = 0
        nxt = order[(idx + 1) % len(order)]
        self.action_set_subselection_mode(nxt)
        self._log(f"Selection mode: {nxt}")

    def action_toggle_navigation_lod(self):
        enabled = self.viewport.toggle_navigation_lod()
        self._log(f"Navigation LOD {'enabled' if enabled else 'disabled'}.")

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

    def action_validate_selected(self, ids: Sequence[str]):
        targets = [str(i) for i in ids if i in self.engine.objects] or self._pick_selected()
        if not targets:
            return
        for oid in targets:
            obj = self.engine.objects[oid]
            report = self.engine.object_validation(oid)
            msg = ", ".join([str(x) for x in report.get("messages", [])]) or "ok"
            self._log(f"Validate {obj.name}: ok={report.get('ok')} :: {msg}")

    def action_heal_selected(self, ids: Sequence[str]):
        targets = [str(i) for i in ids if i in self.engine.objects] or self._pick_selected()
        if not targets:
            return
        for oid in targets:
            obj = self.engine.objects[oid]
            report = self.engine.object_heal(oid)
            msg = ", ".join([str(x) for x in report.get("messages", [])]) or "ok"
            self._log(f"Heal {obj.name}: ok={report.get('ok')} healed={report.get('healed')} :: {msg}")

    # ---------------- boundaries ----------------
    def _current_boundary_id(self) -> str:
        if not hasattr(self, "boundary_combo"):
            return ""
        return str(self.boundary_combo.currentData() or "")

    def _refresh_boundary_controls(self):
        if not hasattr(self, "boundary_combo"):
            return
        current = self._current_boundary_id()
        self.boundary_combo.blockSignals(True)
        self.boundary_combo.clear()
        for row in self.engine.boundary_rows():
            bid = str(row.get("id", ""))
            if not bid:
                continue
            label = f"{row.get('name', bid)} [{row.get('type', '-')}] -> {len(row.get('targets', []))} body(s)"
            self.boundary_combo.addItem(label, bid)
        self.boundary_combo.blockSignals(False)
        if current:
            for idx in range(self.boundary_combo.count()):
                if str(self.boundary_combo.itemData(idx)) == current:
                    self.boundary_combo.setCurrentIndex(idx)
                    break

    def _on_boundary_overlay_toggled(self, value: bool):
        self._boundary_overlay_enabled = bool(value)
        self._prefs.setdefault("boundaries", {})
        self._prefs["boundaries"]["overlay_enabled"] = bool(self._boundary_overlay_enabled)
        self._save_preferences()
        self.refresh_all()

    def action_apply_boundary_from_tab(self):
        targets = self._pick_selected()
        if not targets:
            QMessageBox.warning(self, "Boundaries", "Select at least one object.")
            return
        btype = str(self.boundary_type_combo.currentText() or "fixed").strip().lower()
        bname = str(self.boundary_name_edit.text() or "").strip()
        value = self._parse_float(self.boundary_value_edit, 1.0)
        direction = str(self.boundary_dir_combo.currentText() or "normal").strip().lower()
        self._prefs.setdefault("boundaries", {})
        self._prefs["boundaries"]["default_type"] = str(btype)
        self._prefs["boundaries"]["default_value"] = float(value)
        self._prefs["boundaries"]["default_direction"] = str(direction)
        self._save_preferences()
        point = None
        picked_cell = None
        if bool(self.boundary_pick_chk.isChecked()):
            ctx = self._last_cursor_ctx
            if ctx and ctx.picked_point_3d is not None:
                point = tuple(float(x) for x in ctx.picked_point_3d)
            if ctx and ctx.picked_cell_id is not None:
                picked_cell = int(ctx.picked_cell_id)
        params = {"value": float(value), "direction": str(direction)}
        if picked_cell is not None:
            params["cell_id"] = int(picked_cell)
        bid = self.engine.apply_boundary(
            targets=targets,
            boundary_type=btype,
            params=params,
            name=bname,
            picked_point=point,
            source="tab",
        )
        self._log(f"Boundary applied: {btype} ({bid[:8]}) on {len(targets)} body(s)")

    def action_apply_boundary_quick(self, boundary_type: str, ctx: Optional[ContextInfo] = None):
        token = str(boundary_type or "fixed").strip().lower()
        cursor_ctx = ctx or self._last_cursor_ctx
        targets: List[str] = []
        point = None
        picked_cell = None
        if cursor_ctx is not None:
            picked = str(cursor_ctx.picked_object_id or "").strip()
            if picked and picked in self.engine.objects:
                targets = [picked]
            else:
                targets = [str(i) for i in cursor_ctx.selected_ids if str(i) in self.engine.objects]
            if cursor_ctx.picked_point_3d is not None:
                point = tuple(float(x) for x in cursor_ctx.picked_point_3d)
            if cursor_ctx.picked_cell_id is not None:
                picked_cell = int(cursor_ctx.picked_cell_id)
        if not targets:
            targets = self._pick_selected()
        if not targets:
            return

        value = 1.0
        direction = "normal"
        if token in {"force", "pressure", "displacement"}:
            value, ok = QInputDialog.getDouble(self, "Boundary value", f"{token} magnitude:", 1.0, -1e12, 1e12, 6)
            if not ok:
                return
            direction, ok = QInputDialog.getItem(self, "Boundary direction", "Direction:", ["normal", "x", "y", "z"], 0, False)
            if not ok:
                return
        self._prefs.setdefault("boundaries", {})
        self._prefs["boundaries"]["default_type"] = str(token)
        self._prefs["boundaries"]["default_value"] = float(value)
        self._prefs["boundaries"]["default_direction"] = str(direction).lower()
        self._save_preferences()
        params = {"value": float(value), "direction": str(direction).lower()}
        if picked_cell is not None:
            params["cell_id"] = int(picked_cell)
        bid = self.engine.apply_boundary(
            targets=targets,
            boundary_type=token,
            params=params,
            name=f"{token}_{len(self.engine.boundaries) + 1}",
            picked_point=point,
            source="quick",
        )
        self._log(f"Boundary quick apply: {token} ({bid[:8]}) on {len(targets)} body(s)")

    def action_remove_boundary_selected(self):
        bid = self._current_boundary_id()
        if not bid:
            return
        ok = self.engine.remove_boundary(bid)
        if ok:
            self._log(f"Boundary removed: {bid}")

    def action_clear_boundaries_selected(self):
        ids = self._pick_selected()
        if ids:
            count = self.engine.clear_boundaries(ids)
        else:
            if not self._ask_confirm("Boundaries", "No object selected. Clear all boundaries?"):
                return
            count = self.engine.clear_boundaries(())
        self._log(f"Boundaries cleared: {count}")

    def action_show_boundary_summary(self, ids: Sequence[str]):
        targets = [str(i) for i in ids if str(i) in self.engine.objects] or self._pick_selected()
        if not targets:
            lines = [f"{row.get('name')} [{row.get('type')}] -> {len(row.get('targets', []))} body(s)" for row in self.engine.boundary_rows()]
            text = "\n".join(lines) if lines else "No boundaries."
            self._log(text)
            return
        lines = []
        for oid in targets:
            obj = self.engine.objects[oid]
            rows = self.engine.boundaries_for_object(oid)
            if not rows:
                lines.append(f"{obj.name}: no boundaries")
                continue
            labels = [f"{r.get('name')}[{r.get('type')}]" for r in rows]
            lines.append(f"{obj.name}: " + ", ".join(labels))
        self._log("\n".join(lines))

    def action_copy_boundary_json(self):
        bid = self._current_boundary_id()
        if not bid:
            return
        row = self.engine.boundaries.get(bid, {})
        QApplication.clipboard().setText(json.dumps(row, ensure_ascii=False, indent=2))

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
        ext_map = {
            "step": "step",
            "stp": "stp",
            "stl": "stl",
            "obj": "obj",
            "ply": "ply",
        }
        token = ext_map.get(str(fmt or "").strip().lower(), "obj")

        if len(ids) == 1:
            obj = self.engine.objects[ids[0]]
            out_path, _ = QFileDialog.getSaveFileName(self, "Export selected", f"{obj.name}.{token}", f"{token.upper()} (*.{token})")
            if not out_path:
                return
            try:
                self.engine.export_model([ids[0]], out_path, fmt=token)
                self._log(f"Exported 1 mesh to {out_path}")
            except Exception as e:
                QMessageBox.warning(self, "Export", str(e))
            return

        out_dir = QFileDialog.getExistingDirectory(self, "Export selection")
        if not out_dir:
            return
        count = 0
        for oid in ids:
            obj = self.engine.objects[oid]
            path = os.path.join(out_dir, f"{obj.name}.{token}")
            try:
                self.engine.export_model([oid], path, fmt=token)
                count += 1
            except Exception as e:
                self._log(f"Export failed for {obj.name}: {e}")
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
            "layer": str(obj.meta.get("layer", "Default") if isinstance(obj.meta, dict) else "Default"),
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
        if str(data.get("type", "")).lower() == "boundary":
            bid = str(data.get("boundary_id", "")).strip()
            if bid:
                self.engine.remove_boundary(bid)
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
            self.viewport.focus_on_ids([oid], detail=True)
            return
        self.viewport.fit_all()

    def action_open_preferences(self):
        units, ok = QInputDialog.getItem(self, "Preferences", "Units:", ["mm", "cm", "m", "in"], self.units_combo.currentIndex(), False)
        if not ok:
            return
        snap_step, ok = QInputDialog.getDouble(self, "Preferences", "Grid snap step:", self._snap_step, 1e-9, 1e9, 6)
        if not ok:
            return
        grid_size, ok = QInputDialog.getDouble(self, "Preferences", "Grid size:", self._grid_size_mm, 1e-6, 1e9, 3)
        if not ok:
            return
        angle_step, ok = QInputDialog.getDouble(self, "Preferences", "Angle snap step:", self._angle_snap_step, 0.001, 360.0, 4)
        if not ok:
            return
        mesh_defl, ok = QInputDialog.getDouble(self, "Preferences", "Import mesh deflection:", self._mesh_deflection, 1e-6, 100.0, 6)
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
        self._prefs.setdefault("grid", {})
        self._prefs["grid"]["enabled"] = bool(self._grid_enabled)
        self._prefs["grid"]["step_mm"] = float(snap_step)
        self._prefs["grid"]["size_mm"] = float(grid_size)
        self._prefs["grid"]["color"] = str(self._grid_color)
        self._prefs.setdefault("import", {})
        self._prefs["import"]["deflection"] = float(mesh_defl)
        self._prefs.setdefault("render", {})
        self._prefs["render"]["background"] = str(bg)
        self._prefs["render"]["mode"] = str(render)
        self._save_preferences()
        self._apply_preferences()
        self._set_hint("Preferences updated.")

    def action_backend_diagnostics(self):
        self._refresh_backend_status()
        self._refresh_backend_diagnostics_box()
        diag = self.engine.backend_diagnostics()
        summary = str(diag.get("summary", "no summary"))
        self._log(f"Backend diagnostics: {summary}")

    def action_reconnect_backend(self):
        caps = self.engine.reconnect_backend()
        self._refresh_backend_status()
        self._refresh_backend_diagnostics_box()
        self._log(f"Backend reconnected: {caps.get('provider', 'unknown')}")

    def action_retessellate_selected(self):
        ids = self._pick_selected()
        if not ids:
            QMessageBox.information(self, "Retessellate", "Select at least one kernel-backed object.")
            return
        self._mesh_deflection = max(1e-6, self._parse_float(self.import_deflection_edit, self._mesh_deflection))
        self._prefs.setdefault("import", {})
        self._prefs["import"]["deflection"] = float(self._mesh_deflection)
        self._save_preferences()
        self.engine.set_default_mesh_quality({"deflection": self._mesh_deflection})
        changed = self.engine.retessellate_objects(ids, quality={"deflection": self._mesh_deflection})
        if changed <= 0:
            QMessageBox.information(self, "Retessellate", "No kernel-backed object selected.")
            return
        self._set_hint(f"Retessellated {changed} object(s) with deflection={self._mesh_deflection:.6g}.")

    def action_save_backend_report(self):
        default_name = "mechanical_doctor_report.json"
        path, _ = QFileDialog.getSaveFileName(self, "Save backend report", default_name, "JSON (*.json)")
        if not path:
            return
        payload = {
            "capabilities": self.engine.backend_capabilities(),
            "doctor": self.engine.backend_diagnostics(),
        }
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        self._log(f"Backend report saved: {path}")

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

    @staticmethod
    def _unique_object_name(base: str, used: set) -> str:
        token = str(base or "Object").strip() or "Object"
        if token not in used:
            return token
        idx = 2
        while True:
            cand = f"{token}_{idx}"
            if cand not in used:
                return cand
            idx += 1

    def load_runtime_payload(self, payload: Dict[str, Any], replace: bool = False) -> int:
        row = dict(payload or {})
        meshes = row.get("meshes", [])
        if not isinstance(meshes, list) or not meshes:
            return 0
        source_tag = str(row.get("source", "Runtime") or "Runtime")
        imported_at = str(row.get("imported_at", "") or "")
        created_ids: List[str] = []

        def _apply(eng: SceneEngine):
            if bool(replace):
                eng.objects = {}
                eng.selection = []
            used_names = {obj.name for obj in eng.objects.values()}
            for idx, item in enumerate(meshes, start=1):
                if not isinstance(item, dict):
                    continue
                verts = np.asarray(item.get("vertices", []), dtype=float)
                faces = np.asarray(item.get("faces", []), dtype=int)
                if verts.ndim != 2 or verts.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
                    continue
                base_name = str(item.get("name", f"runtime_body_{idx:02d}") or f"runtime_body_{idx:02d}")
                name = self._unique_object_name(base_name, used_names)
                used_names.add(name)
                obj = SceneObject.create(name=name, mesh=MeshData(vertices=verts, faces=faces), source=f"{source_tag}:runtime")
                obj.color = str(item.get("color", "#86b6f6") or "#86b6f6")
                try:
                    obj.opacity = float(max(0.05, min(1.0, float(item.get("opacity", 0.85)))))
                except Exception:
                    obj.opacity = 0.85
                obj.meta["import_asset_name"] = str(item.get("source_path", "") or source_tag)
                obj.meta["obj_material"] = str(item.get("obj_material", "") or "")
                obj.meta["imported_at"] = imported_at
                group_name = str(item.get("group_name", "") or "").strip()
                layer_hint = group_name or str(item.get("object_name", "") or "").strip() or "Runtime"
                eng._apply_layer_defaults(obj, preferred=layer_hint)
                eng.objects[obj.id] = obj
                created_ids.append(obj.id)
            if created_ids:
                eng.selection = list(created_ids)

        self.engine.execute("Load runtime meshes", _apply, kind="import")
        if created_ids:
            self._log(f"Runtime payload loaded: {len(created_ids)} mesh(es).")
        return len(created_ids)

    def load_runtime_payload_file(self, path: str, replace: bool = False) -> int:
        src = str(path or "").strip()
        if not src or not os.path.isfile(src):
            return 0
        try:
            with open(src, "r", encoding="utf-8", errors="ignore") as f:
                payload = json.load(f)
        except Exception as e:
            self._log(f"Runtime payload load failed: {e}")
            return 0
        if not isinstance(payload, dict):
            return 0
        return self.load_runtime_payload(payload, replace=replace)


def launch_mechanics_window(runtime_payload: Optional[dict] = None, runtime_payload_path: str = ""):
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
    if isinstance(runtime_payload, dict):
        page.load_runtime_payload(runtime_payload, replace=False)
    elif str(runtime_payload_path or "").strip():
        page.load_runtime_payload_file(str(runtime_payload_path), replace=False)
    win.show()
    if owns:
        app.exec()
    return win


if __name__ == "__main__":
    launch_mechanics_window()
