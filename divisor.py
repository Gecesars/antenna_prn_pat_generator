"""Standalone power divider module.

This module runs independently from deep3.py.
"""

from __future__ import annotations

import csv
import math
import queue
import threading
import traceback
from datetime import datetime
from dataclasses import dataclass
from typing import List, Optional
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog

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

from core.divider_aedt import (
    SUBSTRATE_MATERIALS,
    DividerGeometryError,
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

        self._rf_fig = None
        self._rf_axes = []
        self._rf_canvas = None
        self._rf_info_lbl = None
        self._ui_queue: "queue.Queue[tuple[str, object]]" = queue.Queue()
        self._ui_poller_id: Optional[str] = None

        self._build_ui()
        self._start_ui_poller()
        self._build_output_rows(self._parse_int(self.outputs_var.get(), 4), keep_existing=False)
        self.bind("<Destroy>", self._on_destroy, add="+")

    def _build_ui(self) -> None:
        self._build_project_inputs_section()

        self.results_tabs = ctk.CTkTabview(self)
        self.results_tabs.pack(fill=ctk.BOTH, expand=True, padx=10, pady=(0, 8))

        power_tab = self.results_tabs.add("Power Results")
        geometry_tab = self.results_tabs.add("AEDT Geometry")
        rf_tab = self.results_tabs.add("RF Plots")
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

        self._set_power_results_text(
            "Use Build Outputs/Calculate to compute the divider split.\n"
            "Use Run Setup + Plot to execute HFSS and view Return Loss, Impedance, and Phase."
        )
        self._set_geometry_results_text("AEDT geometry output will be shown here.")
        self._set_log_text("AEDT execution log will be shown here.")
        self.results_tabs.set("Power Results")

    def _build_project_inputs_section(self) -> None:
        box = ctk.CTkFrame(self)
        box.pack(fill=ctk.X, padx=10, pady=(10, 6))
        ctk.CTkLabel(box, text="Divisor Project Inputs", anchor="w", font=("Arial", 14, "bold")).pack(
            fill=ctk.X, padx=8, pady=(6, 2)
        )

        grid = ctk.CTkFrame(box, fg_color="transparent")
        grid.pack(fill=ctk.X, padx=8, pady=(0, 4))
        for col in range(8):
            grid.grid_columnconfigure(col, weight=1)

        def _entry_cell(col: int, label: str, var: tk.StringVar, width: int = 110):
            ctk.CTkLabel(grid, text=label, anchor="w").grid(row=0, column=col, padx=4, pady=3, sticky="w")
            ctk.CTkEntry(grid, textvariable=var, width=width).grid(row=1, column=col, padx=4, pady=3, sticky="ew")

        _entry_cell(0, "Total power (W)", self.total_power_var, width=110)
        _entry_cell(1, "Outputs", self.outputs_var, width=90)
        _entry_cell(2, "f_start (MHz)", self.f_start_var, width=100)
        _entry_cell(3, "f_stop (MHz)", self.f_stop_var, width=100)
        _entry_cell(4, "d_ext (mm)", self.d_ext_var, width=100)
        _entry_cell(5, "wall_thick (mm)", self.wall_thick_var, width=100)
        _entry_cell(6, "n_sections", self.n_sections_var, width=90)

        ctk.CTkLabel(grid, text="Dielectric", anchor="w").grid(row=0, column=7, padx=4, pady=3, sticky="w")
        ctk.CTkOptionMenu(grid, variable=self.diel_material_var, values=list(SUBSTRATE_MATERIALS.keys())).grid(
            row=1, column=7, padx=4, pady=3, sticky="ew"
        )

        row_actions_1 = ctk.CTkFrame(box, fg_color="transparent")
        row_actions_1.pack(fill=ctk.X, padx=8, pady=(2, 3))
        ctk.CTkButton(row_actions_1, text="Build Outputs", width=100, command=self._on_build_outputs).pack(
            side=ctk.LEFT, padx=3
        )
        ctk.CTkButton(row_actions_1, text="Equal", width=70, command=self._set_equal_split).pack(side=ctk.LEFT, padx=3)
        ctk.CTkButton(row_actions_1, text="Normalize %", width=100, command=self._normalize_percentages).pack(
            side=ctk.LEFT, padx=3
        )
        ctk.CTkButton(row_actions_1, text="Calculate", width=90, fg_color="#2f7d32", command=self._calculate).pack(
            side=ctk.LEFT, padx=3
        )
        ctk.CTkButton(row_actions_1, text="Export CSV", width=90, command=self._export_csv).pack(side=ctk.LEFT, padx=3)

        row_actions_2 = ctk.CTkFrame(box, fg_color="transparent")
        row_actions_2.pack(fill=ctk.X, padx=8, pady=(0, 6))
        ctk.CTkButton(row_actions_2, text="Calc Geometry", width=110, command=self._calculate_aedt_geometry).pack(
            side=ctk.LEFT, padx=3
        )
        ctk.CTkButton(row_actions_2, text="Create AEDT", width=100, command=self._create_aedt_model).pack(
            side=ctk.LEFT, padx=3
        )
        ctk.CTkButton(row_actions_2, text="Save AEDT As...", width=120, command=self._create_aedt_model_save_as).pack(
            side=ctk.LEFT, padx=3
        )
        ctk.CTkButton(
            row_actions_2,
            text="Run Setup + Plot",
            width=120,
            fg_color="#2f7d32",
            command=self._run_setup_and_plot,
        ).pack(side=ctk.LEFT, padx=3)

        split_top = ctk.CTkFrame(box, fg_color="transparent")
        split_top.pack(fill=ctk.X, padx=8, pady=(2, 2))
        ctk.CTkLabel(split_top, text="Output Split Setup", anchor="w").pack(side=ctk.LEFT, padx=(0, 4))
        self.summary_lbl = ctk.CTkLabel(split_top, text="Percent sum: 0.00%", anchor="e")
        self.summary_lbl.pack(side=ctk.RIGHT, padx=(4, 0))

        self.editor = ctk.CTkScrollableFrame(box, height=110)
        self.editor.pack(fill=ctk.X, padx=8, pady=(0, 4))

        self.aedt_status_lbl = ctk.CTkLabel(box, text="AEDT: idle", anchor="w")
        self.aedt_status_lbl.pack(fill=ctk.X, padx=8, pady=(0, 2))
        self.aedt_params_lbl = ctk.CTkLabel(box, text="Design params loaded: -", anchor="w")
        self.aedt_params_lbl.pack(fill=ctk.X, padx=8, pady=(0, 2))
        self.project_lbl = ctk.CTkLabel(box, text="Project/Design: -", anchor="w")
        self.project_lbl.pack(fill=ctk.X, padx=8, pady=(0, 6))

    def _build_rf_plot_tab(self, parent) -> None:
        controls = ctk.CTkFrame(parent)
        controls.pack(fill=ctk.X, padx=4, pady=(4, 2))
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
        ctk.CTkButton(controls, text="Replot", width=70, command=self._update_rf_plot).pack(side=ctk.LEFT, padx=(10, 6))

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

    def _set_power_results_text(self, text: str) -> None:
        self.results_power_box.delete("1.0", tk.END)
        self.results_power_box.insert("1.0", text)

    def _set_geometry_results_text(self, text: str) -> None:
        self.results_geom_box.delete("1.0", tk.END)
        self.results_geom_box.insert("1.0", text)

    def _set_log_text(self, text: str) -> None:
        self.results_log_box.delete("1.0", tk.END)
        self.results_log_box.insert("1.0", text)

    def _append_log(self, text: str) -> None:
        self.results_log_box.insert(tk.END, f"\n{text}")
        try:
            line_count = int(float(self.results_log_box.index("end-1c").split(".")[0]))
            max_lines = 600
            if line_count > max_lines:
                trim_to = line_count - max_lines
                self.results_log_box.delete("1.0", f"{trim_to}.0")
        except Exception:
            pass
        self.results_log_box.see(tk.END)

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
        if hasattr(self, "aedt_status_lbl"):
            self.aedt_status_lbl.configure(text=f"AEDT: {msg}")
        if hasattr(self, "results_log_box"):
            self._append_log(msg)
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
        for child in self.editor.winfo_children():
            child.destroy()
        self.percent_vars = []
        self.phase_vars = []

        header = ctk.CTkFrame(self.editor, fg_color="transparent")
        header.pack(fill=ctk.X, padx=4, pady=(2, 4))
        ctk.CTkLabel(header, text="Output", width=80, anchor="w").pack(side=ctk.LEFT, padx=2)
        ctk.CTkLabel(header, text="Percent (%)", width=100, anchor="w").pack(side=ctk.LEFT, padx=2)
        ctk.CTkLabel(header, text="Phase (deg)", width=100, anchor="w").pack(side=ctk.LEFT, padx=2)

        for i in range(count):
            row = ctk.CTkFrame(self.editor, fg_color="transparent")
            row.pack(fill=ctk.X, padx=4, pady=2)
            ctk.CTkLabel(row, text=f"OUT {i + 1}", width=80, anchor="w").pack(side=ctk.LEFT, padx=2)

            default_pct = str(100.0 / count)
            if keep_existing and i < len(old_pct) and old_pct[i].strip():
                default_pct = old_pct[i]
            pct_var = tk.StringVar(value=default_pct)
            ctk.CTkEntry(row, textvariable=pct_var, width=100).pack(side=ctk.LEFT, padx=2)
            self.percent_vars.append(pct_var)

            default_phase = "0"
            if keep_existing and i < len(old_ph) and old_ph[i].strip():
                default_phase = old_ph[i]
            ph_var = tk.StringVar(value=default_phase)
            ctk.CTkEntry(row, textvariable=ph_var, width=100).pack(side=ctk.LEFT, padx=2)
            self.phase_vars.append(ph_var)

        self._update_percent_sum()
        self._update_output_block_height(count)

    def _update_output_block_height(self, count: int) -> None:
        # Keep output split compact and reserve space for result tabs.
        n = max(1, int(count))
        header_h = 30
        row_h = 24
        target = header_h + (row_h * min(n, 4))
        target = max(96, min(150, target))
        try:
            self.editor.configure(height=target)
        except Exception:
            pass

    def _update_percent_sum(self) -> None:
        total = 0.0
        for var in self.percent_vars:
            try:
                total += self._parse_float(var.get(), 0.0)
            except Exception:
                continue
        self.summary_lbl.configure(text=f"Percent sum: {total:.2f}%")

    def _set_equal_split(self) -> None:
        n = max(1, len(self.percent_vars))
        value = 100.0 / float(n)
        for var in self.percent_vars:
            var.set(f"{value:.6f}")
        self._update_percent_sum()

    def _normalize_percentages(self) -> None:
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
        self._update_percent_sum()

    def _aedt_input_params(self) -> dict:
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
        if not MATPLOTLIB_AVAILABLE:
            messagebox.showwarning("RF Plot", "Matplotlib is not available in this environment.")
            return
        if self._analysis_thread is not None and self._analysis_thread.is_alive():
            messagebox.showwarning("AEDT", "HFSS setup/plot execution is already running.")
            return
        if self._aedt_thread is not None and self._aedt_thread.is_alive():
            messagebox.showwarning("AEDT", "Wait for AEDT geometry creation to finish before running setup.")
            return

        params_snapshot = self._aedt_input_params()
        geo = self._calculate_aedt_geometry(params=params_snapshot)
        if not geo:
            return
        target = self._resolve_execution_target(params_snapshot, explicit_project_path=None)
        if not target:
            self._set_status("HFSS setup + plot cancelled by user.")
            return

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
                self._post_ui_call(lambda d=out, ps=dict(params_snapshot): self._on_rf_data_ready(d, ps))
            except Exception as exc:
                self._set_status(f"HFSS setup/plot error: {exc}")
                traceback.print_exc()
                self._post_ui_call(lambda e=str(exc): messagebox.showerror("HFSS Setup + Plot", e))
            finally:
                self._post_ui_call(lambda: self._on_background_worker_done("analysis"))

        mode_txt = "rebuild" if bool(target["rebuild_geometry"]) else "variable-only"
        self._set_status(
            "starting HFSS setup + plot "
            f"(f_start={params_snapshot['f_start']:.6f} MHz, "
            f"f_stop={params_snapshot['f_stop']:.6f} MHz, "
            f"design={target['design_name']}, mode={mode_txt})"
        )
        self._analysis_thread = threading.Thread(target=_worker, daemon=True)
        self._analysis_thread.start()

    def _on_rf_data_ready(self, data: dict, applied_params: dict) -> None:
        self._rf_data = dict(data)
        self._last_project_path = str(data.get("project_path") or self._last_project_path or "")
        self._last_design_name = str(data.get("design_name") or self._last_design_name or "DivisorCoaxial")
        self._last_applied_params = dict(applied_params or {})
        self._set_project_label(self._last_project_path, self._last_design_name)

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
        self._update_rf_plot()
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
            ax1, ax2, ax3 = self._rf_axes
            for ax in (ax1, ax2, ax3):
                ax.clear()
                ax.grid(True, alpha=0.3)
                ax.tick_params(labelsize=9)
            ax1.set_title("Return Loss")
            ax2.set_title("Impedance")
            ax3.set_title("Phase")
            ax3.set_xlabel("Frequency")
            self._rf_canvas.draw_idle()
            return

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

        self._rf_canvas.draw_idle()

    def _calculate(self) -> Optional[List[DividerResult]]:
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
