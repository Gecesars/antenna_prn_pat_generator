"""Standalone power divider module.

This module runs independently from deep3.py.
"""

from __future__ import annotations

import csv
import json
import math
import queue
import threading
import traceback
import uuid
import webbrowser
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

import numpy as np

try:
    import customtkinter as ctk
except Exception as e:  # pragma: no cover
    raise ImportError("customtkinter is required to use divisor.py") from e

try:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    from matplotlib.figure import Figure

    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False
    FigureCanvasTkAgg = None
    NavigationToolbar2Tk = None
    Figure = None

try:
    import plotly.graph_objects as go

    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False
    go = None

from core.divider_aedt import (
    SUBSTRATE_MATERIALS,
    DividerGeometryError,
    analyze_impedance_adjustments,
    compute_coaxial_divider_geometry,
    generate_hfss_divider_model,
    run_hfss_divider_analysis,
)


@dataclass
class DividerResult:
    output_idx: int
    percent_norm: float
    power_w: float
    amp_rel_total: float
    amp_rel_max: float
    rel_db: float
    phase_deg: float


@dataclass
class RunRecord:
    run_id: str
    timestamp: str
    status: str
    project_path: str
    design_name: str
    setup_name: str
    sweep_name: str
    metrics: dict
    variables_snapshot: dict
    outputs_snapshot: list
    rf_data: Optional[dict] = None
    error: str = ""


class DivisorTab(ctk.CTkFrame):
    def __init__(self, master, app=None):
        super().__init__(master)
        self.app = app
        self.total_power_var = tk.StringVar(value="100")
        self.outputs_var = tk.StringVar(value="4")
        self.f_start_var = tk.StringVar(value="600")
        self.f_stop_var = tk.StringVar(value="800")
        self.d_ext_var = tk.StringVar(value="33")
        self.wall_thick_var = tk.StringVar(value="1.5")
        self.n_sections_var = tk.StringVar(value="4")
        self.diel_material_var = tk.StringVar(value="Ar")

        self.rl_func_var = tk.StringVar(value="Raw")
        self.z_func_var = tk.StringVar(value="Raw")
        self.phase_func_var = tk.StringVar(value="Raw")
        self.rl_display_var = tk.StringVar(value="S11 (dB)")
        self.plotly_metric_var = tk.StringVar(value="S11 (dB)")

        self.percent_vars: List[tk.StringVar] = []
        self.phase_vars: List[tk.StringVar] = []
        self._results: List[DividerResult] = []
        self._last_geometry: Optional[dict] = None
        self._last_project_path: Optional[str] = None
        self._last_design_name: str = "DivisorCoaxial"
        self._last_applied_params: Optional[dict] = None
        self._rf_data: Optional[dict] = None
        self._aedt_thread: Optional[threading.Thread] = None
        self._analysis_thread: Optional[threading.Thread] = None
        self._runs: List[RunRecord] = []
        self._current_run_id: Optional[str] = None
        self._compare_run_ids: set[str] = set()
        self._auto_run_var = tk.BooleanVar(value=False)
        self._lock_sum_var = tk.BooleanVar(value=True)
        self._pending_changes_var = tk.StringVar(value="Pending changes: 0")
        self._var_specs = []
        self._var_snapshot: dict = {}
        self._split_snapshot: list = []
        self._var_dirty: set[str] = set()
        self._split_dirty = False
        self._updating_var_table = False
        self._editing_widget = None
        self._status_var = tk.StringVar(value="AEDT: idle")
        self._diagnostics_visible = tk.BooleanVar(value=True)
        self._analysis_suggestions: List[dict] = []
        self._analysis_editable_keys = {"d_ext_mm", "wall_thick_mm"}

        self._rf_fig = None
        self._rf_axes = []
        self._rf_canvas = None
        self._rf_info_lbl = None
        self._ui_queue: "queue.Queue[tuple[str, object]]" = queue.Queue()
        self._ui_poller_id: Optional[str] = None

        self._build_ui()
        self._start_ui_poller()
        self._init_variable_specs()
        self._build_variable_table_rows()
        self._build_output_rows(self._parse_int(self.outputs_var.get(), 4), keep_existing=False)
        self._capture_variable_snapshot()
        self._refresh_pending_changes_label()
        self.bind("<Destroy>", self._on_destroy, add="+")

    def _build_ui(self) -> None:
        self._build_header()

        self.main_pane = tk.PanedWindow(self, orient=tk.HORIZONTAL, sashwidth=8, relief=tk.FLAT)
        self.main_pane.pack(fill=ctk.BOTH, expand=True, padx=10, pady=(0, 6))

        self.left_panel = ctk.CTkFrame(self.main_pane, width=400)
        self.right_panel = ctk.CTkFrame(self.main_pane)
        self.main_pane.add(self.left_panel, minsize=320)
        self.main_pane.add(self.right_panel, minsize=680)

        self._build_left_panel()
        self._build_right_panel()
        self._build_diagnostics_dock()

    def _build_header(self) -> None:
        box = ctk.CTkFrame(self)
        box.pack(fill=ctk.X, padx=10, pady=(10, 6))
        box.grid_columnconfigure(0, weight=1)
        box.grid_columnconfigure(1, weight=0)

        left = ctk.CTkFrame(box, fg_color="transparent")
        left.grid(row=0, column=0, sticky="ew", padx=(8, 4), pady=6)
        ctk.CTkLabel(left, text="Divisor | Project Header", anchor="w", font=("Arial", 14, "bold")).pack(
            fill=ctk.X
        )

        self.project_lbl = ctk.CTkLabel(left, text="Project/Design: -", anchor="w")
        self.project_lbl.pack(fill=ctk.X, pady=(2, 0))
        self.aedt_status_lbl = ctk.CTkLabel(left, textvariable=self._status_var, anchor="w")
        self.aedt_status_lbl.pack(fill=ctk.X, pady=(2, 0))
        self.aedt_params_lbl = ctk.CTkLabel(left, text="Design params loaded: -", anchor="w")
        self.aedt_params_lbl.pack(fill=ctk.X, pady=(2, 0))

        right = ctk.CTkFrame(box, fg_color="transparent")
        right.grid(row=0, column=1, sticky="e", padx=(4, 8), pady=6)
        ctk.CTkButton(right, text="Open AEDT", width=95, command=self._open_aedt_project_location).pack(
            side=ctk.LEFT, padx=3
        )
        ctk.CTkButton(
            right,
            text="Abrir Analise Avancada",
            width=170,
            command=self._open_divisor_advanced_analysis,
        ).pack(side=ctk.LEFT, padx=3)
        ctk.CTkButton(right, text="Save Snapshot", width=110, command=self._save_snapshot).pack(
            side=ctk.LEFT, padx=3
        )
        ctk.CTkButton(right, text="Reset", width=70, command=self._reset_state).pack(side=ctk.LEFT, padx=3)
        ctk.CTkButton(right, text="Help", width=60, command=self._show_help).pack(side=ctk.LEFT, padx=3)

    def _build_left_panel(self) -> None:
        scroll = ctk.CTkScrollableFrame(self.left_panel, fg_color="transparent")
        scroll.pack(fill=ctk.BOTH, expand=True, padx=6, pady=6)

        vars_card = self._make_collapsible_card(scroll, "Project Variables")
        self._build_variables_card(vars_card)

        split_card = self._make_collapsible_card(scroll, "Output Split Setup")
        self._build_split_card(split_card)

        runs_card = self._make_collapsible_card(scroll, "Variations / Runs")
        self._build_runs_card(runs_card)

    def _build_right_panel(self) -> None:
        kpi_frame = ctk.CTkFrame(self.right_panel)
        kpi_frame.pack(fill=ctk.X, padx=8, pady=(8, 6))
        for idx in range(6):
            kpi_frame.grid_columnconfigure(idx, weight=1)
        self._kpi_value_labels = {}
        kpi_defs = [
            ("rl_min_db", "RL min (dB)"),
            ("vswr_max", "VSWR max"),
            ("z_f0", "Z @ f0"),
            ("amp_imbalance_db", "Amp imbalance"),
            ("phase_error_deg", "Phase error"),
            ("bandwidth", "Bandwidth"),
        ]
        for col, (key, title) in enumerate(kpi_defs):
            card = ctk.CTkFrame(kpi_frame)
            card.grid(row=0, column=col, padx=4, pady=4, sticky="nsew")
            ctk.CTkLabel(card, text=title, anchor="w").pack(fill=ctk.X, padx=6, pady=(5, 0))
            lbl = ctk.CTkLabel(card, text="-", anchor="w", font=("Arial", 13, "bold"))
            lbl.pack(fill=ctk.X, padx=6, pady=(0, 5))
            self._kpi_value_labels[key] = lbl
        self._update_kpi_cards(None)

        self.results_tabs = ctk.CTkTabview(self.right_panel)
        self.results_tabs.pack(fill=ctk.BOTH, expand=True, padx=8, pady=(0, 8))

        power_tab = self.results_tabs.add("Power Results")
        geometry_tab = self.results_tabs.add("AEDT Geometry")
        rf_tab = self.results_tabs.add("RF Plots")
        analysis_tab = self.results_tabs.add("Plot Analysis")
        log_tab = self.results_tabs.add("AEDT Log")

        self.results_power_box = ctk.CTkTextbox(power_tab, height=300)
        self.results_power_box.pack(fill=ctk.BOTH, expand=True, padx=4, pady=4)
        self.results_geom_box = ctk.CTkTextbox(geometry_tab, height=300)
        self.results_geom_box.pack(fill=ctk.BOTH, expand=True, padx=4, pady=4)
        self.results_log_box = ctk.CTkTextbox(log_tab, height=300)
        self.results_log_box.pack(fill=ctk.BOTH, expand=True, padx=4, pady=4)
        self.results_power_box.configure(font=("Consolas", 12))
        self.results_geom_box.configure(font=("Consolas", 11))
        self.results_log_box.configure(font=("Consolas", 11))

        self._build_rf_plot_tab(rf_tab)
        self._build_plot_analysis_tab(analysis_tab)
        self._set_power_results_text(
            "Use Queue Run to create a variation snapshot and execute HFSS.\n"
            "Use Apply to Solver when you want only dirty variable updates."
        )
        self._set_geometry_results_text("AEDT geometry output will be shown here.")
        self._set_log_text("AEDT execution log will be shown here.")
        self.results_tabs.set("Power Results")

    def _build_diagnostics_dock(self) -> None:
        self.diagnostics_frame = ctk.CTkFrame(self)
        self.diagnostics_frame.pack(fill=ctk.X, padx=10, pady=(0, 8))
        top = ctk.CTkFrame(self.diagnostics_frame, fg_color="transparent")
        top.pack(fill=ctk.X, padx=8, pady=(6, 2))
        ctk.CTkLabel(top, text="Log / Diagnostics", anchor="w").pack(side=ctk.LEFT)
        ctk.CTkButton(top, text="Toggle", width=65, command=self._toggle_diagnostics).pack(side=ctk.RIGHT)
        self.diagnostics_box = ctk.CTkTextbox(self.diagnostics_frame, height=110)
        self.diagnostics_box.pack(fill=ctk.X, padx=8, pady=(0, 8))
        self.diagnostics_box.configure(font=("Consolas", 10))

    def _toggle_diagnostics(self) -> None:
        if self._diagnostics_visible.get():
            self.diagnostics_box.pack_forget()
            self._diagnostics_visible.set(False)
        else:
            self.diagnostics_box.pack(fill=ctk.X, padx=8, pady=(0, 8))
            self._diagnostics_visible.set(True)

    def _make_collapsible_card(self, parent, title: str):
        outer = ctk.CTkFrame(parent)
        outer.pack(fill=ctk.X, pady=(0, 6))
        head = ctk.CTkFrame(outer, fg_color="transparent")
        head.pack(fill=ctk.X, padx=8, pady=(6, 2))
        body = ctk.CTkFrame(outer, fg_color="transparent")
        body.pack(fill=ctk.X, padx=8, pady=(0, 8))
        state = {"open": True}

        def _toggle():
            if state["open"]:
                body.pack_forget()
                btn.configure(text=f"▶ {title}")
            else:
                body.pack(fill=ctk.X, padx=8, pady=(0, 8))
                btn.configure(text=f"▼ {title}")
            state["open"] = not state["open"]

        btn = ctk.CTkButton(head, text=f"▼ {title}", anchor="w", command=_toggle, width=220, height=24)
        btn.pack(side=ctk.LEFT, fill=ctk.X, expand=True)
        return body

    def _build_variables_card(self, parent) -> None:
        cols = ("var", "value", "unit", "type", "dirty", "source", "notes")
        self.var_tree = ttk.Treeview(parent, columns=cols, show="headings", height=8, selectmode="browse")
        headers = {
            "var": "Var",
            "value": "Value",
            "unit": "Unit",
            "type": "Type",
            "dirty": "Dirty",
            "source": "Source",
            "notes": "Notes",
        }
        widths = {"var": 115, "value": 85, "unit": 45, "type": 45, "dirty": 45, "source": 50, "notes": 120}
        for key in cols:
            self.var_tree.heading(key, text=headers[key])
            self.var_tree.column(key, width=widths[key], stretch=(key in {"var", "notes"}))
        self.var_tree.pack(fill=ctk.X, pady=(0, 6))
        self.var_tree.bind("<Double-1>", self._on_var_tree_double_click)

        ctk.CTkLabel(parent, textvariable=self._pending_changes_var, anchor="w").pack(fill=ctk.X, pady=(0, 4))

        row1 = ctk.CTkFrame(parent, fg_color="transparent")
        row1.pack(fill=ctk.X, pady=(0, 4))
        ctk.CTkButton(row1, text="Apply to Solver", width=115, command=self._apply_dirty_to_solver).pack(
            side=ctk.LEFT, padx=2
        )
        ctk.CTkButton(row1, text="Revert", width=70, command=self._revert_dirty_variables).pack(side=ctk.LEFT, padx=2)
        ctk.CTkButton(row1, text="Snapshot", width=80, command=self._save_snapshot).pack(side=ctk.LEFT, padx=2)
        ctk.CTkButton(row1, text="Queue Run", width=85, command=self._queue_run).pack(side=ctk.LEFT, padx=2)
        ctk.CTkSwitch(row1, text="Auto-Run", variable=self._auto_run_var).pack(side=ctk.LEFT, padx=6)

        row2 = ctk.CTkFrame(parent, fg_color="transparent")
        row2.pack(fill=ctk.X)
        ctk.CTkButton(row2, text="Calc Geometry", width=100, command=self._calculate_aedt_geometry).pack(
            side=ctk.LEFT, padx=2
        )
        ctk.CTkButton(row2, text="Create AEDT", width=90, command=self._create_aedt_model).pack(side=ctk.LEFT, padx=2)
        ctk.CTkButton(row2, text="Save As", width=80, command=self._create_aedt_model_save_as).pack(
            side=ctk.LEFT, padx=2
        )
        ctk.CTkButton(row2, text="Run Setup + Plot", width=115, fg_color="#2f7d32", command=self._run_setup_and_plot).pack(
            side=ctk.LEFT, padx=2
        )

    def _build_split_card(self, parent) -> None:
        cols = ("output", "percent", "phase", "power", "delta_equal")
        self.split_tree = ttk.Treeview(parent, columns=cols, show="headings", height=6, selectmode="browse")
        headers = {
            "output": "Output",
            "percent": "Percent (%)",
            "phase": "Phase (deg)",
            "power": "Power (W)",
            "delta_equal": "Δ from equal",
        }
        widths = {"output": 80, "percent": 90, "phase": 90, "power": 90, "delta_equal": 90}
        for key in cols:
            self.split_tree.heading(key, text=headers[key])
            self.split_tree.column(key, width=widths[key], stretch=True)
        self.split_tree.pack(fill=ctk.X, pady=(0, 6))
        self.split_tree.bind("<Double-1>", self._on_split_tree_double_click)

        self.summary_lbl = ctk.CTkLabel(parent, text="Percent sum: 0.00%", anchor="w")
        self.summary_lbl.pack(fill=ctk.X, pady=(0, 4))

        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill=ctk.X, pady=(0, 4))
        ctk.CTkButton(row, text="Build Outputs", width=95, command=self._on_build_outputs).pack(side=ctk.LEFT, padx=2)
        ctk.CTkButton(row, text="Equal", width=60, command=self._set_equal_split).pack(side=ctk.LEFT, padx=2)
        ctk.CTkButton(row, text="Normalize", width=85, command=self._normalize_percentages).pack(side=ctk.LEFT, padx=2)
        ctk.CTkButton(row, text="Calculate", width=80, command=self._calculate).pack(side=ctk.LEFT, padx=2)
        ctk.CTkSwitch(row, text="Lock sum=100%", variable=self._lock_sum_var).pack(side=ctk.LEFT, padx=6)
        ctk.CTkButton(row, text="Export CSV", width=80, command=self._export_csv).pack(side=ctk.LEFT, padx=2)

    def _build_runs_card(self, parent) -> None:
        cols = ("run_id", "timestamp", "status", "rl_min", "vswr_max", "design")
        self.runs_tree = ttk.Treeview(parent, columns=cols, show="headings", height=7, selectmode="extended")
        headers = {
            "run_id": "Run ID",
            "timestamp": "Timestamp",
            "status": "Status",
            "rl_min": "RL min",
            "vswr_max": "VSWR max",
            "design": "Design",
        }
        widths = {"run_id": 90, "timestamp": 120, "status": 70, "rl_min": 70, "vswr_max": 80, "design": 90}
        for key in cols:
            self.runs_tree.heading(key, text=headers[key])
            self.runs_tree.column(key, width=widths[key], stretch=(key in {"run_id", "timestamp", "design"}))
        self.runs_tree.pack(fill=ctk.X, pady=(0, 6))
        self.runs_tree.bind("<<TreeviewSelect>>", self._on_run_tree_select)

        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill=ctk.X, pady=(0, 2))
        ctk.CTkButton(row, text="Run Selected", width=95, command=self._run_selected_record).pack(side=ctk.LEFT, padx=2)
        ctk.CTkButton(row, text="Duplicate", width=75, command=self._duplicate_selected_run).pack(side=ctk.LEFT, padx=2)
        ctk.CTkButton(row, text="Compare", width=75, command=self._compare_selected_runs).pack(side=ctk.LEFT, padx=2)
        ctk.CTkButton(row, text="Export CSV/JSON", width=110, command=self._export_selected_run).pack(
            side=ctk.LEFT, padx=2
        )
        ctk.CTkButton(row, text="Delete", width=65, command=self._delete_selected_runs).pack(side=ctk.LEFT, padx=2)

    def _init_variable_specs(self) -> None:
        self._var_specs = [
            {
                "key": "total_power_w",
                "var": self.total_power_var,
                "unit": "W",
                "type": "float",
                "notes": "Input power",
                "source": "default",
            },
            {
                "key": "f_start_mhz",
                "var": self.f_start_var,
                "unit": "MHz",
                "type": "float",
                "notes": "Start frequency",
                "source": "default",
            },
            {
                "key": "f_stop_mhz",
                "var": self.f_stop_var,
                "unit": "MHz",
                "type": "float",
                "notes": "Stop frequency",
                "source": "default",
            },
            {"key": "d_ext_mm", "var": self.d_ext_var, "unit": "mm", "type": "float", "notes": "Outer diameter", "source": "default"},
            {"key": "wall_thick_mm", "var": self.wall_thick_var, "unit": "mm", "type": "float", "notes": "Wall thickness", "source": "default"},
            {"key": "n_sections", "var": self.n_sections_var, "unit": "-", "type": "int", "notes": "Quarter-wave sections", "source": "default"},
            {"key": "outputs", "var": self.outputs_var, "unit": "-", "type": "int", "notes": "Number of outputs", "source": "default"},
            {"key": "dielectric", "var": self.diel_material_var, "unit": "-", "type": "enum", "notes": "Dielectric material", "source": "default"},
        ]
        self._var_spec_by_key = {spec["key"]: spec for spec in self._var_specs}

    def _build_variable_table_rows(self) -> None:
        if not hasattr(self, "var_tree"):
            return
        self._updating_var_table = True
        try:
            for item in self.var_tree.get_children():
                self.var_tree.delete(item)
            for spec in self._var_specs:
                self.var_tree.insert(
                    "",
                    tk.END,
                    iid=spec["key"],
                    values=(
                        spec["key"],
                        str(spec["var"].get()),
                        spec["unit"],
                        spec["type"],
                        "No",
                        spec["source"],
                        spec["notes"],
                    ),
                )
        finally:
            self._updating_var_table = False

    def _capture_variable_snapshot(self) -> None:
        self._sync_vars_from_table()
        self._sync_split_from_tree()
        self._var_snapshot = {spec["key"]: str(spec["var"].get()) for spec in self._var_specs}
        self._split_snapshot = self._collect_outputs_snapshot()
        self._var_dirty.clear()
        for spec in self._var_specs:
            item = spec["key"]
            vals = list(self.var_tree.item(item, "values"))
            vals[4] = "No"
            vals[5] = "snapshot"
            self.var_tree.item(item, values=vals)
        self._refresh_pending_changes_label()

    def _refresh_pending_changes_label(self) -> None:
        split_dirty = 1 if getattr(self, "_split_dirty", False) else 0
        self._pending_changes_var.set(f"Pending changes: {len(self._var_dirty) + split_dirty}")

    def _on_var_tree_double_click(self, event) -> None:
        row_id = self.var_tree.identify_row(event.y)
        col_id = self.var_tree.identify_column(event.x)
        if not row_id or col_id != "#2":
            return
        spec = self._var_spec_by_key.get(row_id)
        if not spec:
            return
        current = str(spec["var"].get())
        if spec["type"] == "enum":
            value = simpledialog.askstring(
                "Edit variable",
                f"{spec['key']} values: {', '.join(SUBSTRATE_MATERIALS.keys())}",
                initialvalue=current,
                parent=self,
            )
            if value is None:
                return
            token = str(value).strip()
            if token not in SUBSTRATE_MATERIALS:
                messagebox.showerror("Invalid dielectric", "Choose a valid dielectric option.")
                return
            spec["var"].set(token)
        else:
            value = simpledialog.askstring(
                "Edit variable",
                f"{spec['key']} ({spec['unit']})",
                initialvalue=current,
                parent=self,
            )
            if value is None:
                return
            spec["var"].set(str(value).strip())

        vals = list(self.var_tree.item(row_id, "values"))
        vals[1] = str(spec["var"].get())
        vals[4] = "Yes"
        vals[5] = "UI"
        self.var_tree.item(row_id, values=vals)
        self._var_dirty.add(spec["key"])
        self._refresh_pending_changes_label()
        if spec["key"] == "outputs":
            self._on_build_outputs()

    def _on_split_tree_double_click(self, event) -> None:
        row_id = self.split_tree.identify_row(event.y)
        col_id = self.split_tree.identify_column(event.x)
        if not row_id or col_id not in {"#2", "#3"}:
            return
        vals = list(self.split_tree.item(row_id, "values"))
        col_name = "percent" if col_id == "#2" else "phase"
        initial = vals[1] if col_name == "percent" else vals[2]
        prompt = "Percent (%)" if col_name == "percent" else "Phase (deg)"
        value = simpledialog.askstring("Edit split", prompt, initialvalue=str(initial), parent=self)
        if value is None:
            return
        if col_name == "percent":
            vals[1] = str(value).strip()
        else:
            vals[2] = str(value).strip()
        self.split_tree.item(row_id, values=vals)
        self._split_dirty = True
        self._sync_split_from_tree()
        self._refresh_split_derived_columns()
        self._refresh_pending_changes_label()

    def _sync_vars_from_table(self) -> None:
        if not hasattr(self, "var_tree"):
            return
        for spec in self._var_specs:
            vals = self.var_tree.item(spec["key"], "values")
            if vals and len(vals) >= 2:
                spec["var"].set(str(vals[1]).strip())

    def _sync_split_from_tree(self) -> None:
        if not hasattr(self, "split_tree"):
            return
        pvars: List[tk.StringVar] = []
        phvars: List[tk.StringVar] = []
        for item in self.split_tree.get_children():
            vals = self.split_tree.item(item, "values")
            if not vals or len(vals) < 3:
                continue
            pvars.append(tk.StringVar(value=str(vals[1]).strip()))
            phvars.append(tk.StringVar(value=str(vals[2]).strip()))
        self.percent_vars = pvars
        self.phase_vars = phvars
        self.outputs_var.set(str(len(self.percent_vars)))
        if hasattr(self, "var_tree") and self.var_tree.exists("outputs"):
            vals = list(self.var_tree.item("outputs", "values"))
            vals[1] = str(len(self.percent_vars))
            self.var_tree.item("outputs", values=vals)

    def _rebuild_split_tree_from_vars(self) -> None:
        if not hasattr(self, "split_tree"):
            return
        for item in self.split_tree.get_children():
            self.split_tree.delete(item)
        n = max(len(self.percent_vars), len(self.phase_vars))
        for i in range(n):
            pct = self.percent_vars[i].get() if i < len(self.percent_vars) else "0"
            ph = self.phase_vars[i].get() if i < len(self.phase_vars) else "0"
            self.split_tree.insert("", tk.END, values=(f"OUT {i + 1}", pct, ph, "0.000000", "0.000000"))
        self._refresh_split_derived_columns()

    def _refresh_split_derived_columns(self) -> None:
        if not hasattr(self, "split_tree"):
            return
        self._sync_split_from_tree()
        n = max(1, len(self.percent_vars))
        equal_pct = 100.0 / float(n)
        total_power = self._parse_float(self.total_power_var.get(), 0.0)
        for item in self.split_tree.get_children():
            vals = list(self.split_tree.item(item, "values"))
            try:
                pct = self._parse_float(str(vals[1]), 0.0)
            except Exception:
                pct = 0.0
            power = total_power * (pct / 100.0)
            delta = pct - equal_pct
            vals[3] = f"{power:.6f}"
            vals[4] = f"{delta:+.6f}"
            self.split_tree.item(item, values=vals)
        self._update_percent_sum()

    def _validate_project_variables(self, show_message: bool = True) -> bool:
        self._sync_vars_from_table()
        errors = []
        try:
            f_start = self._parse_float(self.f_start_var.get(), 0.0)
            f_stop = self._parse_float(self.f_stop_var.get(), 0.0)
            d_ext = self._parse_float(self.d_ext_var.get(), 0.0)
            wall = self._parse_float(self.wall_thick_var.get(), 0.0)
            if f_start <= 0 or f_stop <= 0 or f_start >= f_stop:
                errors.append("f_start must be > 0 and < f_stop.")
            if wall <= 0:
                errors.append("wall_thick must be > 0.")
            if d_ext <= 0 or d_ext <= 2.0 * wall:
                errors.append("d_ext must be > 2 * wall_thick.")
            if self._parse_int(self.outputs_var.get(), 0) < 1:
                errors.append("outputs must be >= 1.")
            if self._parse_int(self.n_sections_var.get(), 0) < 1:
                errors.append("n_sections must be >= 1.")
            if str(self.diel_material_var.get() or "").strip() not in SUBSTRATE_MATERIALS:
                errors.append("dielectric must be one of the available materials.")
        except Exception as exc:
            errors.append(str(exc))
        if errors and show_message:
            messagebox.showerror("Invalid variables", "\n".join(errors))
        return not errors

    def _validate_split_table(self, show_message: bool = True) -> bool:
        self._sync_split_from_tree()
        if not self.percent_vars:
            if show_message:
                messagebox.showerror("Output split", "No outputs configured.")
            return False
        pcts = []
        for idx, var in enumerate(self.percent_vars, start=1):
            try:
                val = self._parse_float(var.get(), 0.0)
            except Exception:
                if show_message:
                    messagebox.showerror("Output split", f"Output {idx} has invalid percent.")
                return False
            if val < 0:
                if show_message:
                    messagebox.showerror("Output split", f"Output {idx} percent must be >= 0.")
                return False
            pcts.append(val)
        for idx, var in enumerate(self.phase_vars, start=1):
            try:
                ph = self._parse_float(var.get(), 0.0)
            except Exception:
                if show_message:
                    messagebox.showerror("Output split", f"Output {idx} has invalid phase.")
                return False
            while ph > 180.0:
                ph -= 360.0
            while ph < -180.0:
                ph += 360.0
            var.set(f"{ph:.6f}")
        total = sum(pcts)
        if bool(self._lock_sum_var.get()) and abs(total - 100.0) > 1e-6:
            if show_message:
                messagebox.showerror("Output split", "Percent sum must be 100% when Lock sum=100% is enabled.")
            return False
        self._rebuild_split_tree_from_vars()
        return True

    def _save_snapshot(self) -> None:
        if not self._validate_project_variables(show_message=True):
            return
        if not self._validate_split_table(show_message=True):
            return
        self._capture_variable_snapshot()
        self._split_dirty = False
        self._refresh_pending_changes_label()
        self._set_status("Snapshot saved.")

    def _revert_dirty_variables(self) -> None:
        if not self._var_snapshot:
            return
        for spec in self._var_specs:
            old = self._var_snapshot.get(spec["key"])
            if old is not None:
                spec["var"].set(str(old))
                vals = list(self.var_tree.item(spec["key"], "values"))
                vals[1] = str(old)
                vals[4] = "No"
                vals[5] = "snapshot"
                self.var_tree.item(spec["key"], values=vals)
        self._var_dirty.clear()
        if self._split_snapshot:
            self.percent_vars = []
            self.phase_vars = []
            for row in self._split_snapshot:
                self.percent_vars.append(tk.StringVar(value=str(row.get("percent", 0.0))))
                self.phase_vars.append(tk.StringVar(value=str(row.get("phase_deg", 0.0))))
            self._rebuild_split_tree_from_vars()
        else:
            out_count = max(1, self._parse_int(self.outputs_var.get(), 1))
            self._build_output_rows(out_count, keep_existing=False)
        self._split_dirty = False
        self._refresh_pending_changes_label()
        self._set_status("Dirty variables reverted to last snapshot.")

    def _collect_outputs_snapshot(self) -> list:
        self._sync_split_from_tree()
        out = []
        for i, (pct, ph) in enumerate(zip(self.percent_vars, self.phase_vars), start=1):
            out.append(
                {
                    "output": f"OUT {i}",
                    "percent": self._parse_float(pct.get(), 0.0),
                    "phase_deg": self._parse_float(ph.get(), 0.0),
                }
            )
        return out

    def _run_store_dir(self) -> Path:
        token = str(self._last_project_path or "").strip()
        if token:
            p = Path(token)
            return p.parent / "runs"
        return Path.cwd() / "HFSS_Projects" / "runs"

    def _create_run_record(self) -> RunRecord:
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        vars_snap = {spec["key"]: str(spec["var"].get()) for spec in self._var_specs}
        outputs_snap = self._collect_outputs_snapshot()
        return RunRecord(
            run_id=run_id,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            status="queued",
            project_path=str(self._last_project_path or ""),
            design_name=str(self._last_design_name or "DivisorCoaxial"),
            setup_name="Setup1",
            sweep_name="Sweep1",
            metrics={},
            variables_snapshot=vars_snap,
            outputs_snapshot=outputs_snap,
        )

    def _kpis_from_rf_data(self, rf: dict) -> dict:
        freq = np.asarray(rf.get("frequency", []), dtype=float).reshape(-1)
        rl = np.asarray(rf.get("return_loss_db", []), dtype=float).reshape(-1)
        z = np.asarray(rf.get("impedance_ohm", []), dtype=float).reshape(-1)
        ph = np.asarray(rf.get("phase_deg", []), dtype=float).reshape(-1)
        kpi = {
            "rl_min_db": float(np.nanmin(rl)) if rl.size else float("nan"),
            "vswr_max": float(np.nanmax((1 + 10 ** (-rl / 20.0)) / np.maximum(1e-12, 1 - 10 ** (-rl / 20.0)))) if rl.size else float("nan"),
            "z_f0": float(z[len(z) // 2]) if z.size else float("nan"),
            "amp_imbalance_db": float(max((row.rel_db for row in self._results), default=float("nan")) - min((row.rel_db for row in self._results), default=float("nan"))),
            "phase_error_deg": float(np.nanmax(ph) - np.nanmin(ph)) if ph.size else float("nan"),
            "bandwidth": f"{float(freq[0]):.3f}-{float(freq[-1]):.3f}" if freq.size >= 2 else "-",
        }
        return kpi

    def _update_kpi_cards(self, metrics: Optional[dict]) -> None:
        values = metrics or {}
        for key, lbl in getattr(self, "_kpi_value_labels", {}).items():
            raw = values.get(key)
            if raw is None:
                lbl.configure(text="-")
            elif isinstance(raw, str):
                lbl.configure(text=raw)
            elif isinstance(raw, (int, float)):
                if np.isfinite(float(raw)):
                    lbl.configure(text=f"{float(raw):.3f}")
                else:
                    lbl.configure(text="-")
            else:
                lbl.configure(text=str(raw))

    def _persist_run_record(self, run: RunRecord) -> None:
        root = self._run_store_dir() / run.run_id
        root.mkdir(parents=True, exist_ok=True)
        (root / "inputs.json").write_text(
            json.dumps(
                {
                    "variables_snapshot": run.variables_snapshot,
                    "outputs_snapshot": run.outputs_snapshot,
                    "aedt_context": {
                        "project_path": run.project_path,
                        "design_name": run.design_name,
                        "setup_name": run.setup_name,
                        "sweep_name": run.sweep_name,
                    },
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        (root / "metrics.json").write_text(
            json.dumps({"status": run.status, "metrics": run.metrics, "error": run.error}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        if run.rf_data:
            f = np.asarray(run.rf_data.get("frequency", []), dtype=float).reshape(-1)
            rl = np.asarray(run.rf_data.get("return_loss_db", []), dtype=float).reshape(-1)
            z = np.asarray(run.rf_data.get("impedance_ohm", []), dtype=float).reshape(-1)
            ph = np.asarray(run.rf_data.get("phase_deg", []), dtype=float).reshape(-1)
            n = min(len(f), len(rl), len(z), len(ph))
            with (root / "rf_curves.csv").open("w", newline="", encoding="utf-8") as fp:
                writer = csv.writer(fp)
                writer.writerow(["frequency", "return_loss_db", "impedance_ohm", "phase_deg"])
                for i in range(n):
                    writer.writerow([f[i], rl[i], z[i], ph[i]])

    def _refresh_runs_tree(self) -> None:
        if not hasattr(self, "runs_tree"):
            return
        for item in self.runs_tree.get_children():
            self.runs_tree.delete(item)
        for run in self._runs:
            rl_v = run.metrics.get("rl_min_db")
            vswr_v = run.metrics.get("vswr_max")
            rl_txt = f"{float(rl_v):.3f}" if isinstance(rl_v, (int, float)) and np.isfinite(float(rl_v)) else "-"
            vswr_txt = f"{float(vswr_v):.3f}" if isinstance(vswr_v, (int, float)) and np.isfinite(float(vswr_v)) else "-"
            self.runs_tree.insert(
                "",
                tk.END,
                iid=run.run_id,
                values=(
                    run.run_id,
                    run.timestamp,
                    run.status,
                    rl_txt,
                    vswr_txt,
                    run.design_name or "-",
                ),
            )

    def _run_by_id(self, run_id: str) -> Optional[RunRecord]:
        for run in self._runs:
            if run.run_id == run_id:
                return run
        return None

    def _queue_run(self) -> None:
        self._sync_vars_from_table()
        self._sync_split_from_tree()
        if not self._validate_project_variables(show_message=True):
            return
        if not self._validate_split_table(show_message=True):
            return
        self._calculate()
        run = self._create_run_record()
        self._runs.append(run)
        self._refresh_runs_tree()
        self._set_status(f"Run queued: {run.run_id}")
        self._execute_run(run)

    def _execute_run(self, run: RunRecord) -> None:
        if self._analysis_thread is not None and self._analysis_thread.is_alive():
            messagebox.showwarning("Queue Run", "A setup run is already in progress.")
            return
        if self._aedt_thread is not None and self._aedt_thread.is_alive():
            messagebox.showwarning("Queue Run", "Wait for geometry apply to finish.")
            return

        params_snapshot = self._aedt_input_params()
        geo = self._calculate_aedt_geometry(params=params_snapshot)
        if not geo:
            run.status = "failed"
            run.error = "Geometry validation failed."
            self._refresh_runs_tree()
            return

        target = self._resolve_execution_target(params_snapshot, explicit_project_path=None)
        if not target:
            run.status = "cancelled"
            self._refresh_runs_tree()
            return

        run.status = "running"
        run.project_path = str(target.get("project_path") or self._last_project_path or "")
        run.design_name = str(target.get("design_name") or self._last_design_name or "DivisorCoaxial")
        self._refresh_runs_tree()

        def _worker():
            try:
                out = run_hfss_divider_analysis(
                    params_snapshot,
                    status_cb=self._set_status,
                    project_path=target["project_path"],
                    design_name=target["design_name"],
                    rebuild_geometry=bool(target["rebuild_geometry"]),
                    new_desktop=False,
                    close_on_exit=False,
                )
                self._post_ui_call(lambda d=out, ps=dict(params_snapshot), rid=run.run_id: self._on_rf_data_ready(d, ps, rid))
            except Exception as exc:
                traceback.print_exc()
                self._post_ui_call(lambda e=str(exc), rid=run.run_id: self._on_run_failed(rid, e))
            finally:
                self._post_ui_call(lambda: self._on_background_worker_done("analysis"))

        self._set_status(
            f"Run {run.run_id} started (f_start={params_snapshot['f_start']:.6f}MHz, "
            f"f_stop={params_snapshot['f_stop']:.6f}MHz, design={target['design_name']})"
        )
        self._analysis_thread = threading.Thread(target=_worker, daemon=True)
        self._analysis_thread.start()

    def _on_run_failed(self, run_id: str, error: str) -> None:
        run = self._run_by_id(run_id)
        if run:
            run.status = "failed"
            run.error = str(error)
            self._persist_run_record(run)
            self._refresh_runs_tree()
        messagebox.showerror("HFSS Setup + Plot", str(error))
        self._set_status(f"Run failed: {run_id}")

    def _on_run_tree_select(self, _event=None) -> None:
        ids = list(self.runs_tree.selection())
        if len(ids) != 1:
            return
        run = self._run_by_id(ids[0])
        if not run or not run.rf_data:
            return
        self._current_run_id = run.run_id
        self._rf_data = dict(run.rf_data)
        self._last_project_path = str(run.project_path or self._last_project_path or "")
        self._last_design_name = str(run.design_name or self._last_design_name or "DivisorCoaxial")
        self._set_project_label(self._last_project_path, self._last_design_name)
        self._update_kpi_cards(run.metrics)
        self._update_rf_plot()
        self._publish_divisor_sources_to_advanced(prefer_selected=True, quiet=True)

    def _run_selected_record(self) -> None:
        ids = list(self.runs_tree.selection())
        if not ids:
            messagebox.showwarning("Runs", "Select one run to execute.")
            return
        run = self._run_by_id(ids[0])
        if not run:
            return
        for spec in self._var_specs:
            if spec["key"] in run.variables_snapshot:
                spec["var"].set(str(run.variables_snapshot[spec["key"]]))
        self._build_variable_table_rows()
        self._var_dirty.clear()
        self._refresh_pending_changes_label()
        out_count = max(1, self._parse_int(self.outputs_var.get(), 1))
        self._build_output_rows(out_count, keep_existing=False)
        if run.outputs_snapshot:
            for item in self.split_tree.get_children():
                self.split_tree.delete(item)
            self.percent_vars = []
            self.phase_vars = []
            for row in run.outputs_snapshot:
                pct = tk.StringVar(value=str(row.get("percent", 0.0)))
                ph = tk.StringVar(value=str(row.get("phase_deg", 0.0)))
                self.percent_vars.append(pct)
                self.phase_vars.append(ph)
            self._rebuild_split_tree_from_vars()
        self._split_dirty = False
        self._refresh_pending_changes_label()
        self._queue_run()

    def _duplicate_selected_run(self) -> None:
        ids = list(self.runs_tree.selection())
        if not ids:
            return
        src = self._run_by_id(ids[0])
        if not src:
            return
        new = self._create_run_record()
        new.variables_snapshot = dict(src.variables_snapshot)
        new.outputs_snapshot = [dict(x) for x in src.outputs_snapshot]
        self._runs.append(new)
        self._refresh_runs_tree()
        self._set_status(f"Run duplicated as {new.run_id}.")

    def _compare_selected_runs(self) -> None:
        ids = set(self.runs_tree.selection())
        self._compare_run_ids = set(ids)
        self._update_rf_plot()
        self._publish_divisor_sources_to_advanced(prefer_selected=False, quiet=True)
        self._set_status(f"Compare overlay updated with {len(ids)} run(s).")

    def _export_selected_run(self) -> None:
        ids = list(self.runs_tree.selection())
        if not ids:
            return
        run = self._run_by_id(ids[0])
        if not run:
            return
        self._persist_run_record(run)
        messagebox.showinfo("Run Export", f"Run exported to:\n{self._run_store_dir() / run.run_id}")

    def _delete_selected_runs(self) -> None:
        ids = set(self.runs_tree.selection())
        if not ids:
            return
        self._runs = [r for r in self._runs if r.run_id not in ids]
        self._compare_run_ids = {rid for rid in self._compare_run_ids if rid not in ids}
        if self._current_run_id in ids:
            self._current_run_id = None
        self._refresh_runs_tree()
        self._update_rf_plot()

    def _open_aedt_project_location(self) -> None:
        path = str(self._last_project_path or "").strip()
        if not path:
            messagebox.showwarning("Open AEDT", "No project path available.")
            return
        target = Path(path)
        folder = target.parent if target.suffix.lower() == ".aedt" else target
        try:
            import os

            os.startfile(str(folder))
        except Exception as exc:
            messagebox.showerror("Open AEDT", str(exc))

    def _reset_state(self) -> None:
        self._rf_data = None
        self._results = []
        self._compare_run_ids.clear()
        self._current_run_id = None
        self._update_kpi_cards(None)
        self._update_rf_plot()
        self._set_status("Divisor state reset.")

    def _show_help(self) -> None:
        messagebox.showinfo(
            "Divisor Workflow",
            "1) Edit variables in Project Variables table.\n"
            "2) Edit split in Output Split Setup table.\n"
            "3) Apply to Solver (dirty only) or Queue Run.\n"
            "4) Inspect KPIs, RF plots and runs history.",
        )

    def _apply_dirty_to_solver(self, run_after_apply: bool = False) -> None:
        self._sync_vars_from_table()
        if not self._validate_project_variables(show_message=True):
            return
        changed = sorted(self._var_dirty)
        if not changed:
            self._set_status("No dirty variables to apply.")
            return
        if self._aedt_thread is not None and self._aedt_thread.is_alive():
            messagebox.showwarning("AEDT", "An AEDT operation is already running.")
            return
        if self._analysis_thread is not None and self._analysis_thread.is_alive():
            messagebox.showwarning("AEDT", "Wait for setup execution to finish.")
            return

        params_snapshot = self._aedt_input_params()
        topology_changed = any(k in {"n_sections", "outputs"} for k in changed)
        target = self._resolve_execution_target(params_snapshot, explicit_project_path=None)
        if not target:
            return

        def _worker():
            try:
                out = generate_hfss_divider_model(
                    params_snapshot,
                    status_cb=self._set_status,
                    project_path=target["project_path"],
                    design_name=target["design_name"],
                    rebuild_geometry=bool(topology_changed or target["rebuild_geometry"]),
                    new_desktop=False,
                    close_on_exit=False,
                )
                self._post_ui_call(
                    lambda p=str(out or ""), d=str(target["design_name"]), ps=dict(params_snapshot): self._on_project_created(
                        p, d, ps
                    )
                )
                self._post_ui_call(self._capture_variable_snapshot)
                if bool(run_after_apply) or bool(self._auto_run_var.get()):
                    self._post_ui_call(self._queue_run)
            except Exception as exc:
                traceback.print_exc()
                self._post_ui_call(lambda e=str(exc): messagebox.showerror("Apply to Solver", e))
            finally:
                self._post_ui_call(lambda: self._on_background_worker_done("aedt"))

        self._set_status(f"Applying dirty variables to solver ({', '.join(changed)})")
        self._aedt_thread = threading.Thread(target=_worker, daemon=True)
        self._aedt_thread.start()

    def _build_rf_plot_tab(self, parent) -> None:
        controls = ctk.CTkFrame(parent)
        controls.pack(fill=ctk.X, padx=4, pady=(4, 2))
        ctk.CTkLabel(controls, text="Plotly:").pack(side=ctk.LEFT, padx=(6, 4))
        ctk.CTkOptionMenu(
            controls,
            variable=self.plotly_metric_var,
            values=["S11 (dB)", "Return Loss (+dB)", "|Z| (Ohm)", "Phase (deg)", "Smith Chart"],
            width=165,
        ).pack(side=ctk.LEFT, padx=(0, 8))
        ctk.CTkButton(controls, text="Open Plotly", width=95, command=self._open_plotly_view).pack(
            side=ctk.LEFT, padx=(0, 8)
        )
        ctk.CTkButton(controls, text="Analyze Plot", width=95, command=self._analyze_current_plot).pack(
            side=ctk.LEFT, padx=(0, 8)
        )
        ctk.CTkButton(
            controls,
            text="Link Avancada",
            width=105,
            command=lambda: self._publish_divisor_sources_to_advanced(prefer_selected=True, quiet=False),
        ).pack(side=ctk.LEFT, padx=(0, 8))
        ctk.CTkOptionMenu(
            controls,
            variable=self.rl_display_var,
            values=["S11 (dB)", "Return Loss (+dB)"],
            width=155,
            command=lambda _: self._update_rf_plot(),
        ).pack(side=ctk.LEFT, padx=(6, 8))
        ctk.CTkLabel(controls, text="S11 Func:").pack(side=ctk.LEFT, padx=(0, 4))
        ctk.CTkOptionMenu(
            controls,
            variable=self.rl_func_var,
            values=["Raw", "Moving Avg (5)", "Derivative"],
            width=130,
            command=lambda _: self._update_rf_plot(),
        ).pack(side=ctk.LEFT, padx=4)
        ctk.CTkLabel(controls, text="Impedance:").pack(side=ctk.LEFT, padx=(10, 4))
        ctk.CTkOptionMenu(
            controls,
            variable=self.z_func_var,
            values=["Raw", "Moving Avg (5)", "Derivative", "Normalize"],
            width=130,
            command=lambda _: self._update_rf_plot(),
        ).pack(side=ctk.LEFT, padx=4)
        ctk.CTkLabel(controls, text="Phase:").pack(side=ctk.LEFT, padx=(10, 4))
        ctk.CTkOptionMenu(
            controls,
            variable=self.phase_func_var,
            values=["Raw", "Unwrap", "Moving Avg (5)", "Derivative"],
            width=130,
            command=lambda _: self._update_rf_plot(),
        ).pack(side=ctk.LEFT, padx=4)
        self.replot_btn = ctk.CTkButton(controls, text="Replot", width=70, command=self._update_rf_plot)
        self.replot_btn.pack(side=ctk.LEFT, padx=(10, 6))
        self.replot_btn.configure(state="disabled")

        self._rf_info_lbl = ctk.CTkLabel(parent, text="No RF data loaded yet.", anchor="w")
        self._rf_info_lbl.pack(fill=ctk.X, padx=8, pady=(0, 4))

        if not MATPLOTLIB_AVAILABLE:
            ctk.CTkLabel(
                parent,
                text="Matplotlib is not available in this environment. RF plot view is disabled.",
                anchor="w",
            ).pack(fill=ctk.X, padx=8, pady=8)
            return

        fig = Figure(figsize=(10.5, 6.8), dpi=100)
        ax1 = fig.add_subplot(311)
        ax2 = fig.add_subplot(312, sharex=ax1)
        ax3 = fig.add_subplot(313, sharex=ax1)
        fig.subplots_adjust(left=0.07, right=0.99, top=0.96, bottom=0.08, hspace=0.36)
        self._rf_fig = fig
        self._rf_axes = [ax1, ax2, ax3]

        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill=ctk.BOTH, expand=True, padx=4, pady=(0, 4))
        self._rf_canvas = canvas

        toolbar_frame = ctk.CTkFrame(parent, fg_color="transparent")
        toolbar_frame.pack(fill=ctk.X, padx=4, pady=(0, 4))
        toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
        toolbar.update()

    def _build_plot_analysis_tab(self, parent) -> None:
        top = ctk.CTkFrame(parent)
        top.pack(fill=ctk.X, padx=6, pady=(6, 4))
        self.analysis_info_lbl = ctk.CTkLabel(
            top,
            text="Analyze RF data and suggest only geometry auto-tune vars (d_ext_mm, wall_thick_mm).",
            anchor="w",
        )
        self.analysis_info_lbl.pack(side=ctk.LEFT, fill=ctk.X, expand=True)
        ctk.CTkButton(top, text="Analyze Plot", width=95, command=self._analyze_current_plot).pack(
            side=ctk.RIGHT, padx=3
        )

        cols = ("variable", "current", "suggested", "delta", "priority", "confidence", "rationale")
        self.analysis_tree = ttk.Treeview(parent, columns=cols, show="headings", height=11, selectmode="extended")
        headers = {
            "variable": "Variable",
            "current": "Current",
            "suggested": "Suggested",
            "delta": "Delta",
            "priority": "Priority",
            "confidence": "Confidence",
            "rationale": "Rationale",
        }
        widths = {
            "variable": 110,
            "current": 80,
            "suggested": 90,
            "delta": 85,
            "priority": 65,
            "confidence": 80,
            "rationale": 480,
        }
        for key in cols:
            self.analysis_tree.heading(key, text=headers[key])
            self.analysis_tree.column(key, width=widths[key], stretch=(key == "rationale"))
        self.analysis_tree.pack(fill=ctk.BOTH, expand=True, padx=6, pady=(0, 6))

        bottom = ctk.CTkFrame(parent)
        bottom.pack(fill=ctk.X, padx=6, pady=(0, 6))
        ctk.CTkButton(
            bottom,
            text="Apply to Variables",
            width=125,
            command=self._apply_selected_suggestions_to_variables,
        ).pack(side=ctk.LEFT, padx=3)
        ctk.CTkButton(
            bottom,
            text="Apply to Project",
            width=115,
            command=lambda: self._apply_suggestions_to_project(run_after=False),
        ).pack(side=ctk.LEFT, padx=3)
        ctk.CTkButton(
            bottom,
            text="Apply + Run",
            width=105,
            fg_color="#2f7d32",
            command=lambda: self._apply_suggestions_to_project(run_after=True),
        ).pack(side=ctk.LEFT, padx=3)

    def _collect_plot_datasets(self) -> list[tuple[str, dict]]:
        datasets: list[tuple[str, dict]] = []
        if self._rf_data:
            datasets.append((self._current_run_id or "current", dict(self._rf_data)))
        for run_id in sorted(self._compare_run_ids):
            run = self._run_by_id(run_id)
            if run and run.rf_data:
                datasets.append((run_id, dict(run.rf_data)))
        return datasets

    def _get_advanced_view(self):
        if self.app is None:
            return None
        return getattr(self.app, "advanced_tab_view", None)

    def _build_divisor_advanced_sources(self) -> tuple[dict, str]:
        metric_to_suffix = {
            "S11 (dB)": "S11 (dB)",
            "Return Loss (+dB)": "Return Loss (+dB)",
            "|Z| (Ohm)": "|Z| (Ohm)",
            "Phase (deg)": "Phase (deg)",
            "Smith Chart": "S11 (dB)",
        }
        selected_metric = str(self.plotly_metric_var.get() or "S11 (dB)")
        wanted_suffix = metric_to_suffix.get(selected_metric, "S11 (dB)")

        out: dict = {}
        preferred_key = ""
        datasets = self._collect_plot_datasets()
        current_label = str(self._current_run_id or "current")
        for idx, (label, data) in enumerate(datasets):
            freq = np.asarray(data.get("frequency", []), dtype=float).reshape(-1)
            rl_pos = np.asarray(data.get("return_loss_db", []), dtype=float).reshape(-1)
            z = np.asarray(data.get("impedance_ohm", []), dtype=float).reshape(-1)
            ph = np.asarray(data.get("phase_deg", []), dtype=float).reshape(-1)
            n = min(len(freq), len(rl_pos), len(z), len(ph))
            if n <= 1:
                continue
            freq = freq[:n]
            rl_pos = rl_pos[:n]
            z = z[:n]
            ph = ph[:n]
            s11_db = -rl_pos

            run_label = str(label or f"run_{idx + 1}")
            project_token = str(data.get("project_path") or self._last_project_path or "").strip()
            design_token = str(data.get("design_name") or self._last_design_name or "DivisorCoaxial").strip()
            origin = f"Divisor | project={project_token or '-'} | design={design_token}"

            metrics = [
                ("S11 (dB)", s11_db, True),
                ("Return Loss (+dB)", rl_pos, False),
                ("|Z| (Ohm)", z, True),
                ("Phase (deg)", ph, True),
            ]
            for suffix, values, allow_negative in metrics:
                key = f"Divisor RF | {run_label} | {suffix}"
                out[key] = {
                    "kind": "V",
                    "angles": np.asarray(freq, dtype=float),
                    "values": np.asarray(values, dtype=float),
                    "origin": origin,
                    "normalize": False,
                    "allow_negative": bool(allow_negative),
                }
                if not preferred_key and run_label == current_label and suffix == wanted_suffix:
                    preferred_key = key
                elif not preferred_key and idx == 0 and suffix == wanted_suffix:
                    preferred_key = key

        if not preferred_key and out:
            preferred_key = next(iter(out.keys()))
        return out, preferred_key

    def _publish_divisor_sources_to_advanced(self, prefer_selected: bool = True, quiet: bool = True) -> bool:
        if not self._rf_data:
            if not quiet:
                messagebox.showwarning("Analise Avancada", "No RF data loaded yet.")
            return False

        view = self._get_advanced_view()
        if view is None:
            if not quiet:
                messagebox.showwarning("Analise Avancada", "A aba Visualizacao Avancada nao esta disponivel.")
            return False

        sources, preferred_key = self._build_divisor_advanced_sources()
        if not sources:
            if not quiet:
                messagebox.showwarning("Analise Avancada", "No valid RF dataset to link.")
            return False

        prefix = "Divisor RF | "
        user_sources = getattr(view, "user_sources", {})
        if isinstance(user_sources, dict):
            for key in list(user_sources.keys()):
                if str(key).startswith(prefix):
                    user_sources.pop(key, None)
            user_sources.update(sources)
            view.user_sources = user_sources
        else:
            view.user_sources = dict(sources)

        try:
            view.refresh_sources(preferred_key=preferred_key if prefer_selected else "")
            self._set_status(
                f"Divisor linked to Advanced tab ({len(sources)} source(s), preferred={preferred_key or '-'})"
            )
            return True
        except Exception as exc:
            if not quiet:
                messagebox.showerror("Analise Avancada", str(exc))
            return False

    def _open_divisor_advanced_analysis(self) -> None:
        if self.app is None:
            messagebox.showwarning("Analise Avancada", "Advanced tab is only available in the main application.")
            return
        try:
            if hasattr(self.app, "open_advanced_view"):
                self.app.open_advanced_view()
            elif hasattr(self.app, "tabs"):
                self.app.tabs.set("Visualizacao Avancada")
        except Exception:
            pass
        self._publish_divisor_sources_to_advanced(prefer_selected=True, quiet=False)

    def _open_plotly_view(self) -> None:
        if not self._rf_data:
            messagebox.showwarning("Plotly", "No RF data loaded.")
            return
        if not PLOTLY_AVAILABLE:
            messagebox.showwarning("Plotly", "Plotly is not installed in this environment.")
            return

        metric = str(self.plotly_metric_var.get() or "S11 (dB)")
        datasets = self._collect_plot_datasets()
        if not datasets:
            messagebox.showwarning("Plotly", "No RF dataset available for plotting.")
            return

        fig = go.Figure()
        if metric == "Smith Chart":
            for label, data in datasets:
                s11r = np.asarray(data.get("s11_real", []), dtype=float).reshape(-1)
                s11i = np.asarray(data.get("s11_imag", []), dtype=float).reshape(-1)
                n = min(len(s11r), len(s11i))
                if n <= 1:
                    continue
                fig.add_trace(go.Scattersmith(real=s11r[:n], imag=s11i[:n], mode="lines", name=label))
            fig.update_layout(title="Smith Chart (S11)")
        else:
            for label, data in datasets:
                f = np.asarray(data.get("frequency", []), dtype=float).reshape(-1)
                if metric == "S11 (dB)":
                    y = -np.asarray(data.get("return_loss_db", []), dtype=float).reshape(-1)
                    y_title = "S11 (dB)"
                elif metric == "Return Loss (+dB)":
                    y = np.asarray(data.get("return_loss_db", []), dtype=float).reshape(-1)
                    y_title = "Return Loss (+dB)"
                elif metric == "|Z| (Ohm)":
                    y = np.asarray(data.get("impedance_ohm", []), dtype=float).reshape(-1)
                    y_title = "|Z| (Ohm)"
                else:
                    y = np.asarray(data.get("phase_deg", []), dtype=float).reshape(-1)
                    y_title = "Phase (deg)"
                n = min(len(f), len(y))
                if n <= 1:
                    continue
                fig.add_trace(go.Scatter(x=f[:n], y=y[:n], mode="lines", name=label))
            unit = str(self._rf_data.get("frequency_unit", "MHz") or "MHz")
            fig.update_layout(
                title=f"{metric} | Interactive Plotly",
                xaxis_title=f"Frequency ({unit})",
                yaxis_title=y_title,
            )
        fig.update_layout(template="plotly_white", hovermode="x unified")

        plot_dir = self._run_store_dir() / "_plotly"
        plot_dir.mkdir(parents=True, exist_ok=True)
        out_file = plot_dir / f"plotly_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        fig.write_html(str(out_file), include_plotlyjs=True, full_html=True)
        webbrowser.open(out_file.resolve().as_uri())
        self._set_status(f"Plotly view generated: {out_file}")

    def _analyze_current_plot(self) -> None:
        if not self._rf_data:
            messagebox.showwarning("Analyze Plot", "No RF data loaded yet.")
            return
        params = self._aedt_input_params()
        suggestions = analyze_impedance_adjustments(self._rf_data, params)
        suggestions = [
            row
            for row in suggestions
            if str(row.get("variable", "")).strip() in self._analysis_editable_keys
        ]
        self._analysis_suggestions = list(suggestions)
        self._refresh_analysis_tree()
        self.results_tabs.set("Plot Analysis")
        if suggestions:
            self.analysis_info_lbl.configure(
                text=(
                    f"{len(suggestions)} suggestion(s) generated. "
                    "Only geometry auto-tune variables are editable."
                )
            )
        else:
            self.analysis_info_lbl.configure(text="No viable geometry-only suggestions generated.")

    def _refresh_analysis_tree(self) -> None:
        if not hasattr(self, "analysis_tree"):
            return
        for item in self.analysis_tree.get_children():
            self.analysis_tree.delete(item)
        for idx, row in enumerate(self._analysis_suggestions):
            cur = row.get("current")
            sug = row.get("suggested")
            dlt = row.get("delta")
            self.analysis_tree.insert(
                "",
                tk.END,
                iid=f"sug_{idx}",
                values=(
                    str(row.get("variable", "")),
                    f"{float(cur):.6f}" if isinstance(cur, (int, float)) else str(cur),
                    f"{float(sug):.6f}" if isinstance(sug, (int, float)) else str(sug),
                    f"{float(dlt):+.6f}" if isinstance(dlt, (int, float)) else str(dlt),
                    str(row.get("priority", "")),
                    str(row.get("confidence", "")),
                    str(row.get("rationale", "")),
                ),
            )

    def _selected_suggestions(self) -> list[dict]:
        if not self._analysis_suggestions:
            return []
        if not hasattr(self, "analysis_tree"):
            return list(self._analysis_suggestions)
        selected = list(self.analysis_tree.selection())
        if not selected:
            return list(self._analysis_suggestions)
        out = []
        for item in selected:
            try:
                idx = int(str(item).split("_", 1)[1])
            except Exception:
                continue
            if 0 <= idx < len(self._analysis_suggestions):
                out.append(self._analysis_suggestions[idx])
        return out

    def _apply_selected_suggestions_to_variables(self) -> bool:
        rows = self._selected_suggestions()
        if not rows:
            messagebox.showwarning("Plot Analysis", "No suggestion available to apply.")
            return False

        key_to_spec = {spec["key"]: spec for spec in self._var_specs}
        changed = 0
        outputs_changed = False
        for row in rows:
            var_key = str(row.get("variable", "")).strip()
            if var_key not in self._analysis_editable_keys:
                continue
            spec = key_to_spec.get(var_key)
            if not spec:
                continue
            suggested = row.get("suggested")
            if suggested is None:
                continue

            if spec["type"] == "int":
                new_val = str(int(round(float(suggested))))
            elif spec["type"] == "enum":
                new_val = str(suggested)
            else:
                new_val = f"{float(suggested):.6f}"
            if str(spec["var"].get()) == new_val:
                continue
            spec["var"].set(new_val)

            if self.var_tree.exists(spec["key"]):
                vals = list(self.var_tree.item(spec["key"], "values"))
                vals[1] = new_val
                vals[4] = "Yes"
                vals[5] = "analysis"
                self.var_tree.item(spec["key"], values=vals)
            self._var_dirty.add(spec["key"])
            if spec["key"] == "outputs":
                outputs_changed = True
            changed += 1

        if changed:
            if outputs_changed:
                self._on_build_outputs()
            self._refresh_pending_changes_label()
            self._set_status(f"Applied {changed} analysis suggestion(s) to variables.")
            return True
        self._set_status("No variable value changed by selected suggestions.")
        return False

    def _apply_suggestions_to_project(self, run_after: bool = False) -> None:
        applied = self._apply_selected_suggestions_to_variables()
        if not applied:
            return
        self._apply_dirty_to_solver(run_after_apply=bool(run_after))

    def _set_power_results_text(self, text: str) -> None:
        self.results_power_box.delete("1.0", tk.END)
        self.results_power_box.insert("1.0", text)

    def _set_geometry_results_text(self, text: str) -> None:
        self.results_geom_box.delete("1.0", tk.END)
        self.results_geom_box.insert("1.0", text)

    def _set_log_text(self, text: str) -> None:
        self.results_log_box.delete("1.0", tk.END)
        self.results_log_box.insert("1.0", text)
        if hasattr(self, "diagnostics_box"):
            self.diagnostics_box.delete("1.0", tk.END)
            self.diagnostics_box.insert("1.0", text)

    def _append_log(self, text: str) -> None:
        for box in (getattr(self, "results_log_box", None), getattr(self, "diagnostics_box", None)):
            if box is None:
                continue
            box.insert(tk.END, f"\n{text}")
            try:
                line_count = int(float(box.index("end-1c").split(".")[0]))
                max_lines = 700
                if line_count > max_lines:
                    trim_to = line_count - max_lines
                    box.delete("1.0", f"{trim_to}.0")
            except Exception:
                pass
            box.see(tk.END)

    def _start_ui_poller(self) -> None:
        if self._ui_poller_id is not None:
            return
        self._ui_poller_id = self.after(60, self._drain_ui_queue)

    def _post_ui_call(self, func) -> None:
        self._ui_queue.put(("call", func))

    def _post_ui_status(self, msg: str) -> None:
        self._ui_queue.put(("status", msg))

    def _drain_ui_queue(self) -> None:
        self._ui_poller_id = None
        handled = 0
        while handled < 120:
            try:
                kind, payload = self._ui_queue.get_nowait()
            except queue.Empty:
                break
            handled += 1
            try:
                if kind == "status":
                    self._apply_status(str(payload))
                elif kind == "call" and callable(payload):
                    payload()
            except Exception:
                traceback.print_exc()
        self._ui_poller_id = self.after(60, self._drain_ui_queue)

    def _apply_status(self, msg: str) -> None:
        if not msg:
            return
        self._status_var.set(f"AEDT: {msg}")
        if hasattr(self, "aedt_status_lbl"):
            self.aedt_status_lbl.configure(textvariable=self._status_var)
        if hasattr(self, "results_log_box"):
            self._append_log(msg)
        if ("error" in msg.lower() or "failed" in msg.lower()) and hasattr(self, "diagnostics_box"):
            if not self._diagnostics_visible.get():
                self._toggle_diagnostics()
            try:
                self.diagnostics_box.configure(height=180)
            except Exception:
                pass
        if self.app is not None and hasattr(self.app, "_set_status"):
            try:
                self.app._set_status(msg)
            except Exception:
                pass

    def _on_destroy(self, _event=None) -> None:
        try:
            exists = bool(self.winfo_exists())
        except Exception:
            exists = False
        if not exists:
            try:
                if self._ui_poller_id is not None:
                    self.after_cancel(self._ui_poller_id)
            except Exception:
                pass
            self._ui_poller_id = None

    def _on_background_worker_done(self, kind: str) -> None:
        token = str(kind or "").strip().lower()
        if token == "analysis":
            self._analysis_thread = None
        elif token == "aedt":
            self._aedt_thread = None

    def _parse_float(self, text: str, default: float = 0.0) -> float:
        raw = str(text or "").strip().replace(",", ".")
        if not raw:
            return default
        return float(raw)

    def _parse_int(self, text: str, default: int = 0) -> int:
        raw = str(text or "").strip().replace(",", ".")
        if not raw:
            return int(default)
        return int(round(float(raw)))

    def _set_status(self, text: str) -> None:
        msg = str(text or "").strip()
        if not msg:
            return
        if threading.current_thread() is threading.main_thread():
            self._apply_status(msg)
            return
        self._post_ui_status(msg)

    def _set_project_label(self, project_path: Optional[str], design_name: Optional[str] = None) -> None:
        token = str(design_name or self._last_design_name or "").strip() or "-"
        if project_path:
            self.project_lbl.configure(text=f"Project/Design: {project_path} | {token}")
        else:
            self.project_lbl.configure(text="Project/Design: -")

    def _on_build_outputs(self) -> None:
        self._sync_vars_from_table()
        try:
            count = int(float(self.outputs_var.get().strip().replace(",", ".")))
        except Exception:
            messagebox.showerror("Invalid outputs", "Outputs must be an integer.")
            return
        if count < 1 or count > 64:
            messagebox.showerror("Invalid outputs", "Outputs must be between 1 and 64.")
            return
        self._build_output_rows(count, keep_existing=True)

    def _build_output_rows(self, count: int, keep_existing: bool = True) -> None:
        old_pct = [v.get() for v in self.percent_vars]
        old_ph = [v.get() for v in self.phase_vars]
        self.percent_vars = []
        self.phase_vars = []
        if hasattr(self, "split_tree"):
            for item in self.split_tree.get_children():
                self.split_tree.delete(item)

        for i in range(count):
            default_pct = str(100.0 / count)
            if keep_existing and i < len(old_pct) and old_pct[i].strip():
                default_pct = old_pct[i]
            pct_var = tk.StringVar(value=default_pct)
            self.percent_vars.append(pct_var)

            default_phase = "0"
            if keep_existing and i < len(old_ph) and old_ph[i].strip():
                default_phase = old_ph[i]
            ph_var = tk.StringVar(value=default_phase)
            self.phase_vars.append(ph_var)

            if hasattr(self, "split_tree"):
                try:
                    pct_txt = f"{float(default_pct):.6f}"
                except Exception:
                    pct_txt = str(default_pct)
                self.split_tree.insert(
                    "",
                    tk.END,
                    values=(f"OUT {i + 1}", pct_txt, default_phase, "0.000000", "0.000000"),
                )

        self._update_percent_sum()
        self._refresh_split_derived_columns()
        if keep_existing:
            self._split_dirty = True
            self._refresh_pending_changes_label()

    def _update_percent_sum(self) -> None:
        self._sync_split_from_tree()
        total = 0.0
        for var in self.percent_vars:
            try:
                total += self._parse_float(var.get(), 0.0)
            except Exception:
                continue
        self.summary_lbl.configure(text=f"Percent sum: {total:.2f}%")

    def _set_equal_split(self) -> None:
        self._sync_split_from_tree()
        n = max(1, len(self.percent_vars))
        value = 100.0 / float(n)
        for var in self.percent_vars:
            var.set(f"{value:.6f}")
        self._rebuild_split_tree_from_vars()
        self._split_dirty = True
        self._refresh_pending_changes_label()
        self._update_percent_sum()

    def _normalize_percentages(self) -> None:
        self._sync_split_from_tree()
        pcts: List[float] = []
        for var in self.percent_vars:
            try:
                pcts.append(self._parse_float(var.get(), 0.0))
            except Exception:
                pcts.append(0.0)
        total = sum(pcts)
        if total <= 0.0:
            messagebox.showwarning("Normalize", "Percent sum must be > 0.")
            return
        scale = 100.0 / total
        for i, var in enumerate(self.percent_vars):
            var.set(f"{pcts[i] * scale:.6f}")
        self._rebuild_split_tree_from_vars()
        self._split_dirty = True
        self._refresh_pending_changes_label()
        self._update_percent_sum()

    def _aedt_input_params(self) -> dict:
        self._sync_vars_from_table()
        self._sync_split_from_tree()
        return {
            "f_start": self._parse_float(self.f_start_var.get(), 800.0),
            "f_stop": self._parse_float(self.f_stop_var.get(), 1200.0),
            "d_ext": self._parse_float(self.d_ext_var.get(), 20.0),
            "wall_thick": self._parse_float(self.wall_thick_var.get(), 1.5),
            "n_sections": self._parse_int(self.n_sections_var.get(), 4),
            "n_outputs": self._parse_int(self.outputs_var.get(), max(1, len(self.percent_vars))),
            "diel_material": str(self.diel_material_var.get() or "Ar"),
        }

    def _normalize_param_value(self, key: str, value) -> object:
        if value is None:
            return None
        if key in {"n_sections", "n_outputs"}:
            return int(round(float(value)))
        if key == "diel_material":
            return str(value or "").strip()
        return round(float(value), 9)

    def _detect_param_changes(self, params_now: dict) -> List[str]:
        if not self._last_applied_params:
            return sorted(params_now.keys())
        changed: List[str] = []
        for key, now_val in params_now.items():
            old_val = self._last_applied_params.get(key)
            if self._normalize_param_value(key, now_val) != self._normalize_param_value(key, old_val):
                changed.append(key)
        return changed

    def _topology_changed(self, changed_keys: List[str]) -> bool:
        return any(k in {"n_sections", "n_outputs"} for k in changed_keys)

    def _ask_edit_action(self, changed_keys: List[str], topology_changed: bool) -> Optional[str]:
        if not changed_keys:
            return "apply_loaded"

        chosen = {"value": None}
        dlg = ctk.CTkToplevel(self)
        dlg.title("Aplicar edicoes no divisor")
        dlg.geometry("620x230")
        dlg.resizable(False, False)
        dlg.transient(self.winfo_toplevel())
        dlg.grab_set()

        msg = (
            "Foram detectadas alteracoes nos parametros do divisor.\n"
            f"Campos alterados: {', '.join(changed_keys)}\n"
            f"Topologia alterada: {'sim' if topology_changed else 'nao'}\n\n"
            "Escolha como aplicar antes de executar:"
        )
        ctk.CTkLabel(dlg, text=msg, justify="left", anchor="w").pack(fill=ctk.X, padx=16, pady=(16, 8))

        btn_row = ctk.CTkFrame(dlg, fg_color="transparent")
        btn_row.pack(fill=ctk.X, padx=16, pady=(4, 8))

        def _pick(value: str) -> None:
            chosen["value"] = value
            try:
                dlg.grab_release()
            except Exception:
                pass
            dlg.destroy()

        ctk.CTkButton(
            btn_row,
            text="Salvar como novo projeto",
            width=180,
            command=lambda: _pick("save_new"),
        ).pack(side=ctk.LEFT, padx=6)
        ctk.CTkButton(
            btn_row,
            text="Adicionar design HFSS",
            width=170,
            command=lambda: _pick("add_design"),
        ).pack(side=ctk.LEFT, padx=6)
        ctk.CTkButton(
            btn_row,
            text="Aplicar no modelo carregado",
            width=210,
            fg_color="#2f7d32",
            command=lambda: _pick("apply_loaded"),
        ).pack(side=ctk.LEFT, padx=6)

        ctk.CTkButton(dlg, text="Cancelar", width=120, command=lambda: _pick("cancel")).pack(pady=(6, 10))
        dlg.protocol("WM_DELETE_WINDOW", lambda: _pick("cancel"))
        self.wait_window(dlg)

        value = chosen["value"]
        if value in {None, "cancel"}:
            return None
        return str(value)

    def _resolve_execution_target(self, params_snapshot: dict, explicit_project_path: Optional[str]) -> Optional[dict]:
        changed_keys = self._detect_param_changes(params_snapshot)
        topology_changed = self._topology_changed(changed_keys)
        last_project = str(self._last_project_path or "").strip() or None
        last_design = str(self._last_design_name or "").strip() or "DivisorCoaxial"

        if explicit_project_path:
            return {
                "project_path": str(explicit_project_path),
                "design_name": "DivisorCoaxial",
                "rebuild_geometry": True,
                "changed_keys": changed_keys,
                "topology_changed": topology_changed,
                "action": "save_new",
            }

        if not last_project:
            return {
                "project_path": None,
                "design_name": "DivisorCoaxial",
                "rebuild_geometry": True,
                "changed_keys": changed_keys,
                "topology_changed": topology_changed,
                "action": "new_project",
            }

        action = self._ask_edit_action(changed_keys, topology_changed)
        if action is None:
            return None

        if action == "save_new":
            default_name = f"Divisor_Coaxial_{datetime.now().strftime('%Y%m%d_%H%M%S')}.aedt"
            path = filedialog.asksaveasfilename(
                title="Salvar novo projeto AEDT",
                defaultextension=".aedt",
                filetypes=[("AEDT project", "*.aedt"), ("All files", "*.*")],
                initialfile=default_name,
            )
            if not path:
                return None
            return {
                "project_path": str(path),
                "design_name": "DivisorCoaxial",
                "rebuild_geometry": True,
                "changed_keys": changed_keys,
                "topology_changed": topology_changed,
                "action": action,
            }

        if action == "add_design":
            default_design = f"DivisorCoaxial_{datetime.now().strftime('%H%M%S')}"
            name = simpledialog.askstring(
                "Novo design HFSS",
                "Nome do novo design HFSS no projeto atual:",
                initialvalue=default_design,
                parent=self,
            )
            if name is None:
                return None
            token = str(name).strip()
            if not token:
                messagebox.showerror("Novo design HFSS", "Nome do design nao pode ser vazio.")
                return None
            return {
                "project_path": last_project,
                "design_name": token,
                "rebuild_geometry": True,
                "changed_keys": changed_keys,
                "topology_changed": topology_changed,
                "action": action,
            }

        # action == apply_loaded
        return {
            "project_path": last_project,
            "design_name": last_design,
            "rebuild_geometry": bool(topology_changed),
            "changed_keys": changed_keys,
            "topology_changed": topology_changed,
            "action": action,
        }

    def _format_aedt_geometry(self, geo: dict) -> str:
        lines = [
            "AEDT geometry (coaxial divider)",
            "",
            f"Input f_start: {float(geo['f_start']):.6f} MHz",
            f"Input f_stop : {float(geo['f_stop']):.6f} MHz",
            f"Center f0    : {float(geo['f0_mhz']):.6f} MHz",
            f"n_sections   : {int(geo['n_sections'])}",
            f"n_outputs    : {int(geo['n_outputs'])}",
            f"d_ext        : {float(geo['d_ext']):.6f} mm",
            f"wall_thick   : {float(geo['wall_thick']):.6f} mm",
            f"d_int_tube   : {float(geo['d_int_tube']):.6f} mm",
            f"sec_len_mm   : {float(geo['sec_len_mm']):.6f}",
            f"len_inner_mm : {float(geo['len_inner_mm']):.6f}",
            f"len_outer_mm : {float(geo['len_outer_mm']):.6f}",
            f"d_out_50ohm  : {float(geo['d_out_50ohm']):.6f}",
            f"dia_saida_diel: {float(geo['dia_saida_diel']):.6f}",
            f"dia_saida_cond: {float(geo['dia_saida_cond']):.6f}",
        ]
        z_vals = [f"{float(v):.6f}" for v in geo.get("z_sects", [])]
        d_vals = [f"{float(v):.6f}" for v in geo.get("main_diams", [])]
        lines.append("z_sects    : " + ", ".join(z_vals))
        lines.append("main_diams : " + ", ".join(d_vals))
        return "\n".join(lines)

    def _render_aedt_params_loaded(self, geo: dict) -> None:
        self.aedt_params_lbl.configure(
            text=(
                "Design params loaded: "
                f"f_start={float(geo['f_start']):.6f} MHz | "
                f"f_stop={float(geo['f_stop']):.6f} MHz | "
                f"f0={float(geo['f0_mhz']):.6f} MHz | "
                f"n_sections={int(geo['n_sections'])} | "
                f"n_outputs={int(geo['n_outputs'])}"
            )
        )

    def _calculate_aedt_geometry(self, params: Optional[dict] = None) -> Optional[dict]:
        input_params = dict(params or self._aedt_input_params())
        try:
            geo = compute_coaxial_divider_geometry(input_params)
        except DividerGeometryError as exc:
            messagebox.showerror("AEDT geometry", str(exc))
            return None
        except Exception as exc:
            messagebox.showerror("AEDT geometry", f"Unexpected geometry error: {exc}")
            return None
        self._last_geometry = dict(geo)
        self._set_geometry_results_text(self._format_aedt_geometry(geo))
        self._render_aedt_params_loaded(geo)
        self.results_tabs.set("AEDT Geometry")
        self._set_status("coaxial geometry computed")
        return geo

    def _create_aedt_model(self) -> None:
        self._run_create_aedt_model(project_path=None)

    def _create_aedt_model_save_as(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Save AEDT project as",
            defaultextension=".aedt",
            filetypes=[("AEDT project", "*.aedt"), ("All files", "*.*")],
            initialfile="Divisor_Coaxial.aedt",
        )
        if not path:
            return
        self._run_create_aedt_model(project_path=path)

    def _run_create_aedt_model(self, project_path: Optional[str]) -> None:
        if self._aedt_thread is not None and self._aedt_thread.is_alive():
            messagebox.showwarning("AEDT", "An AEDT generation is already running.")
            return
        if self._analysis_thread is not None and self._analysis_thread.is_alive():
            messagebox.showwarning("AEDT", "Wait for setup/plot execution to finish before creating geometry.")
            return
        params_snapshot = self._aedt_input_params()
        geo = self._calculate_aedt_geometry(params=params_snapshot)
        if not geo:
            return
        target = self._resolve_execution_target(params_snapshot, explicit_project_path=project_path)
        if not target:
            self._set_status("HFSS generation cancelled by user.")
            return

        def _worker():
            try:
                out = generate_hfss_divider_model(
                    params_snapshot,
                    status_cb=self._set_status,
                    project_path=target["project_path"],
                    design_name=target["design_name"],
                    rebuild_geometry=bool(target["rebuild_geometry"]),
                    new_desktop=False,
                    close_on_exit=False,
                )
                if out:
                    self._set_status(f"HFSS project saved: {out}")
                    self._post_ui_call(
                        lambda p=str(out), d=str(target["design_name"]), ps=dict(params_snapshot): self._on_project_created(
                            p, d, ps
                        )
                    )
                else:
                    self._set_status("HFSS model generation finished without project path.")
            except Exception as exc:
                self._set_status(f"HFSS generation error: {exc}")
                traceback.print_exc()
                self._post_ui_call(lambda e=str(exc): messagebox.showerror("AEDT", e))
            finally:
                self._post_ui_call(lambda: self._on_background_worker_done("aedt"))

        mode_txt = "rebuild" if bool(target["rebuild_geometry"]) else "variable-only"
        self._set_status(
            "starting HFSS geometry generation "
            f"(f_start={params_snapshot['f_start']:.6f} MHz, "
            f"f_stop={params_snapshot['f_stop']:.6f} MHz, "
            f"design={target['design_name']}, mode={mode_txt})"
        )
        self._aedt_thread = threading.Thread(target=_worker, daemon=True)
        self._aedt_thread.start()

    def _on_project_created(self, project_path: str, design_name: str, applied_params: dict) -> None:
        self._last_project_path = str(project_path)
        self._last_design_name = str(design_name or "DivisorCoaxial")
        self._last_applied_params = dict(applied_params or {})
        self._set_project_label(self._last_project_path, self._last_design_name)
        messagebox.showinfo("AEDT", f"HFSS model atualizado:\n{project_path}\nDesign: {self._last_design_name}")

    def _run_setup_and_plot(self) -> None:
        self._queue_run()

    def _on_rf_data_ready(self, data: dict, applied_params: dict, run_id: Optional[str] = None) -> None:
        self._rf_data = dict(data)
        self._last_project_path = str(data.get("project_path") or self._last_project_path or "")
        self._last_design_name = str(data.get("design_name") or self._last_design_name or "DivisorCoaxial")
        self._last_applied_params = dict(applied_params or {})
        self._set_project_label(self._last_project_path, self._last_design_name)
        self._capture_variable_snapshot()
        self._split_dirty = False
        self._refresh_pending_changes_label()

        if self._rf_info_lbl is not None:
            rl = np.asarray(data.get("return_loss_db", []), dtype=float).reshape(-1)
            rl_min = float(np.nanmin(rl)) if rl.size else float("nan")
            s11_max = float(np.nanmax(-rl)) if rl.size else float("nan")
            self._rf_info_lbl.configure(
                text=(
                    f"Project: {self._last_project_path or '-'} | "
                    f"Design: {self._last_design_name or '-'} | "
                    f"Setup: {data.get('setup_sweep_name', '-')} | "
                    f"Expression: {data.get('expression', '-')} | "
                    f"RL min: {rl_min:.2f} dB | "
                    f"S11 max: {s11_max:.2f} dB"
                )
            )
        metrics = self._kpis_from_rf_data(data)
        self._update_kpi_cards(metrics)
        if run_id:
            run = self._run_by_id(run_id)
            if run is not None:
                run.status = "ok"
                run.rf_data = dict(data)
                run.metrics = dict(metrics)
                run.project_path = self._last_project_path or run.project_path
                run.design_name = self._last_design_name or run.design_name
                self._persist_run_record(run)
                self._refresh_runs_tree()
                self._current_run_id = run.run_id
        self._update_rf_plot()
        self._publish_divisor_sources_to_advanced(prefer_selected=True, quiet=True)
        self.results_tabs.set("RF Plots")
        self._set_status("RF plots updated.")

    def _apply_plot_function(self, y: np.ndarray, fn_name: str, metric: str) -> np.ndarray:
        if y.size == 0:
            return y
        name = str(fn_name or "Raw").strip()
        if name == "Raw":
            return y
        if name == "Moving Avg (5)":
            kernel = np.ones(5, dtype=float) / 5.0
            return np.convolve(y, kernel, mode="same")
        if name == "Derivative":
            return np.gradient(y)
        if metric == "z" and name == "Normalize":
            ymin = float(np.min(y))
            ymax = float(np.max(y))
            span = ymax - ymin
            if span <= 1e-12:
                return np.zeros_like(y)
            return (y - ymin) / span
        if metric == "phase" and name == "Unwrap":
            return np.degrees(np.unwrap(np.radians(y)))
        return y

    def _update_rf_plot(self) -> None:
        if not MATPLOTLIB_AVAILABLE or self._rf_canvas is None or not self._rf_axes:
            return
        if not self._rf_data:
            if hasattr(self, "replot_btn"):
                try:
                    self.replot_btn.configure(state="disabled")
                except Exception:
                    pass
            ax1, ax2, ax3 = self._rf_axes
            for ax in (ax1, ax2, ax3):
                ax.clear()
                ax.grid(True, alpha=0.3)
                ax.tick_params(labelsize=9)
            ax1.set_title("Return Loss")
            ax2.set_title("Impedance")
            ax3.set_title("Phase")
            ax3.set_xlabel("Frequency")
            if self._rf_info_lbl is not None:
                self._rf_info_lbl.configure(text="RF data not available. Check AEDT path / run solver.")
            self._rf_canvas.draw_idle()
            return
        if hasattr(self, "replot_btn"):
            try:
                self.replot_btn.configure(state="normal")
            except Exception:
                pass

        freq = np.asarray(self._rf_data.get("frequency", []), dtype=float).reshape(-1)
        rl = np.asarray(self._rf_data.get("return_loss_db", []), dtype=float).reshape(-1)
        z = np.asarray(self._rf_data.get("impedance_ohm", []), dtype=float).reshape(-1)
        ph = np.asarray(self._rf_data.get("phase_deg", []), dtype=float).reshape(-1)
        n = min(len(freq), len(rl), len(z), len(ph))
        if n <= 1:
            return
        freq = freq[:n]
        rl = rl[:n]
        z = z[:n]
        ph = ph[:n]
        unit = str(self._rf_data.get("frequency_unit", "MHz") or "MHz")

        rl_plot = self._apply_plot_function(rl, self.rl_func_var.get(), "rl")
        z_plot = self._apply_plot_function(z, self.z_func_var.get(), "z")
        ph_plot = self._apply_plot_function(ph, self.phase_func_var.get(), "phase")

        rl_mode = str(self.rl_display_var.get() or "S11 (dB)")
        if rl_mode == "S11 (dB)":
            y_rl = -rl_plot
            y_rl_title = "S11 (dB)"
        else:
            y_rl = rl_plot
            y_rl_title = "Return Loss (+dB)"

        ax1, ax2, ax3 = self._rf_axes
        for ax in (ax1, ax2, ax3):
            ax.clear()
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=9)

        ax1.plot(freq, y_rl, color="#0f4c81", linewidth=2.0)
        ax1.set_ylabel("dB")
        ax1.set_title(f"{y_rl_title} | Function: {self.rl_func_var.get()}")
        ax1.margins(x=0.01, y=0.18)

        ax2.plot(freq, z_plot, color="#0a7d35", linewidth=2.0)
        ax2.set_ylabel("Ohm")
        ax2.set_title(f"Impedance | Function: {self.z_func_var.get()}")
        ax2.margins(x=0.01, y=0.18)

        ax3.plot(freq, ph_plot, color="#8c3f00", linewidth=2.0)
        ax3.set_ylabel("deg")
        ax3.set_title(f"Phase | Function: {self.phase_func_var.get()}")
        ax3.set_xlabel(f"Frequency ({unit})")
        ax3.margins(x=0.01, y=0.18)

        # Optional overlay for run comparison.
        legend_needed = False
        for run_id in sorted(self._compare_run_ids):
            if run_id == self._current_run_id:
                continue
            run = self._run_by_id(run_id)
            if not run or not run.rf_data:
                continue
            rfreq = np.asarray(run.rf_data.get("frequency", []), dtype=float).reshape(-1)
            rrl = np.asarray(run.rf_data.get("return_loss_db", []), dtype=float).reshape(-1)
            rz = np.asarray(run.rf_data.get("impedance_ohm", []), dtype=float).reshape(-1)
            rph = np.asarray(run.rf_data.get("phase_deg", []), dtype=float).reshape(-1)
            rn = min(len(rfreq), len(rrl), len(rz), len(rph))
            if rn <= 1:
                continue
            rfreq = rfreq[:rn]
            rrl = rrl[:rn]
            rz = rz[:rn]
            rph = rph[:rn]

            rrl_plot = self._apply_plot_function(rrl, self.rl_func_var.get(), "rl")
            rz_plot = self._apply_plot_function(rz, self.z_func_var.get(), "z")
            rph_plot = self._apply_plot_function(rph, self.phase_func_var.get(), "phase")
            ry_rl = -rrl_plot if rl_mode == "S11 (dB)" else rrl_plot

            ax1.plot(rfreq, ry_rl, linewidth=1.2, alpha=0.75, label=run_id)
            ax2.plot(rfreq, rz_plot, linewidth=1.2, alpha=0.75, label=run_id)
            ax3.plot(rfreq, rph_plot, linewidth=1.2, alpha=0.75, label=run_id)
            legend_needed = True

        if legend_needed:
            ax1.legend(loc="best", fontsize=8)
            ax2.legend(loc="best", fontsize=8)
            ax3.legend(loc="best", fontsize=8)

        self._rf_canvas.draw_idle()

    def _calculate(self) -> Optional[List[DividerResult]]:
        self._sync_vars_from_table()
        self._sync_split_from_tree()
        if not self._validate_split_table(show_message=True):
            return None
        try:
            total_power = self._parse_float(self.total_power_var.get(), 0.0)
        except Exception:
            messagebox.showerror("Invalid total power", "Total power must be numeric.")
            return None
        if total_power <= 0.0:
            messagebox.showerror("Invalid total power", "Total power must be > 0 W.")
            return None
        if not self.percent_vars:
            messagebox.showerror("No outputs", "Build at least one output.")
            return None

        pcts_raw: List[float] = []
        phases: List[float] = []
        for i, var in enumerate(self.percent_vars):
            try:
                val = self._parse_float(var.get(), 0.0)
            except Exception:
                messagebox.showerror("Invalid percent", f"Output {i + 1} has invalid percent.")
                return None
            if val < 0.0:
                messagebox.showerror("Invalid percent", f"Output {i + 1} percent must be >= 0.")
                return None
            pcts_raw.append(val)
        for i, var in enumerate(self.phase_vars):
            try:
                phases.append(self._parse_float(var.get(), 0.0))
            except Exception:
                messagebox.showerror("Invalid phase", f"Output {i + 1} has invalid phase.")
                return None

        pct_sum = sum(pcts_raw)
        if pct_sum <= 0.0:
            messagebox.showerror("Invalid split", "Percent sum must be > 0.")
            return None

        pcts_norm = [(p / pct_sum) * 100.0 for p in pcts_raw]
        amps = [math.sqrt(max(0.0, p / 100.0)) for p in pcts_norm]
        max_amp = max(amps) if amps else 1.0
        if max_amp <= 0.0:
            max_amp = 1.0

        results: List[DividerResult] = []
        for i, pct in enumerate(pcts_norm):
            frac = pct / 100.0
            pwr = total_power * frac
            amp_total = math.sqrt(max(0.0, frac))
            amp_max = amp_total / max_amp
            rel_db = (10.0 * math.log10(frac)) if frac > 0.0 else -120.0
            results.append(
                DividerResult(
                    output_idx=i + 1,
                    percent_norm=pct,
                    power_w=pwr,
                    amp_rel_total=amp_total,
                    amp_rel_max=amp_max,
                    rel_db=rel_db,
                    phase_deg=phases[i] if i < len(phases) else 0.0,
                )
            )

        self._results = results
        self._render_results(total_power, pct_sum, results)
        self._update_percent_sum()
        return results

    def _render_results(self, total_power: float, pct_sum_raw: float, results: List[DividerResult]) -> None:
        lines = [
            "Divider results",
            f"Total power: {total_power:.6f} W",
            f"Input percent sum (raw): {pct_sum_raw:.6f}%",
            "",
            "OUT | Percent(%) | Power(W)   | Amp(Total) | Amp(Max) | Rel dB   | Phase(deg)",
            "-" * 84,
        ]
        for row in results:
            lines.append(
                f"{row.output_idx:>3d} | "
                f"{row.percent_norm:>9.4f} | "
                f"{row.power_w:>9.4f} | "
                f"{row.amp_rel_total:>10.6f} | "
                f"{row.amp_rel_max:>8.6f} | "
                f"{row.rel_db:>8.3f} | "
                f"{row.phase_deg:>10.3f}"
            )
        self._set_power_results_text("\n".join(lines))
        self.results_tabs.set("Power Results")

    def _export_csv(self) -> None:
        if not self._results:
            computed = self._calculate()
            if not computed:
                return
        path = filedialog.asksaveasfilename(
            title="Export divider CSV",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")],
            initialfile="divisor_outputs.csv",
        )
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "output",
                    "percent_norm",
                    "power_w",
                    "amp_rel_total",
                    "amp_rel_max",
                    "rel_db",
                    "phase_deg",
                ]
            )
            for row in self._results:
                w.writerow(
                    [
                        row.output_idx,
                        f"{row.percent_norm:.6f}",
                        f"{row.power_w:.6f}",
                        f"{row.amp_rel_total:.8f}",
                        f"{row.amp_rel_max:.8f}",
                        f"{row.rel_db:.4f}",
                        f"{row.phase_deg:.4f}",
                    ]
                )
        messagebox.showinfo("Exported", f"CSV saved:\n{path}")


def register_divisor_tab(app, tabview: "ctk.CTkTabview", tab_name: str = "Divisor"):
    """Register Divisor tab in the main app tabview."""
    tab = tabview.add(tab_name)
    frame = DivisorTab(tab, app=app)
    frame.pack(fill="both", expand=True)
    return frame


class DivisorApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("EFTX - Divisor de Potencia")
        self.geometry("1200x780")
        self.minsize(980, 640)
        panel = DivisorTab(self, app=None)
        panel.pack(fill="both", expand=True, padx=8, pady=8)


def main() -> int:
    ctk.set_appearance_mode("light")
    ctk.set_default_color_theme("blue")
    app = DivisorApp()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
