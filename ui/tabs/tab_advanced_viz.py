from __future__ import annotations

import csv
import datetime
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

import customtkinter as ctk
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

from core.angles import ang_dist_deg, wrap_phi_deg
from core.math_engine import (
    MarkerValue,
    MathFunctionDef,
    evaluate_functions,
    load_user_functions,
    save_user_functions,
)
from core.metrics_advanced import summarize_advanced_metrics
from core.obj_parser import parse_obj_file
from core.perf import DEFAULT_TRACER, PerfTracer
from core.reconstruct3d import cut_from_arrays, reconstruct_spherical
from ui.interactions.plot_interactor import AdvancedPlotInteractor
from ui.viewers.viewer3d_pyvista import export_obj, export_plotly_html, open_3d_view, open_meshes_view
from ui.widgets.derived_table import DerivedTable
from ui.widgets.marker_table import MarkerTable
from ui.widgets.plot_panel import PlotPanel


class AdvancedVisualizationTab(ctk.CTkFrame):
    def __init__(self, master, app, **kwargs):
        super().__init__(master, **kwargs)
        self.app = app
        self.tracer: PerfTracer = getattr(app, "perf_tracer", DEFAULT_TRACER)

        self.source_map: Dict[str, dict] = {}
        self.user_sources: Dict[str, dict] = {}
        self.current_key: Optional[str] = None
        self.current_kind: str = "H"
        self.current_angles: Optional[np.ndarray] = None
        self.current_values: Optional[np.ndarray] = None
        self.functions: List[MathFunctionDef] = load_user_functions()
        self.last_spherical = None
        self.cad_status_var = tk.StringVar(value="CAD: nenhum importado")
        self._cad_runtime_cache = None

        self.cut_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Ready")
        self.recon_status_var = tk.StringVar(value="Idle")
        self.xdb_var = tk.StringVar(value="10")

        self.recon_mode_var = tk.StringVar(value="separable")
        self.recon_sep_mode_var = tk.StringVar(value="direct")
        self.recon_alpha_var = tk.StringVar(value="1.0")
        self.recon_beta_var = tk.StringVar(value="1.0")
        self.recon_theta_pts_var = tk.StringVar(value="181")
        self.recon_phi_pts_var = tk.StringVar(value="361")
        self.recon_dbmin_var = tk.StringVar(value="-40")
        self.recon_dbmax_var = tk.StringVar(value="0")
        self.recon_gamma_var = tk.StringVar(value="1.0")
        self.recon_v_source_var = tk.StringVar(value="")
        self.recon_h_source_var = tk.StringVar(value="")
        self.wireframe_var = tk.BooleanVar(value=False)
        appearance = "Dark"
        try:
            appearance = str(ctk.get_appearance_mode() or "Dark")
        except Exception:
            appearance = "Dark"
        self.appearance_var = tk.StringVar(value=appearance)

        self._pending_markers: List[MarkerValue] = []
        self._table_refresh_job = None

        self.recon_executor = ThreadPoolExecutor(max_workers=1)
        self.recon_future = None
        self._recon_cache: Dict[tuple, object] = {}

        self._build_ui()
        self.refresh_sources()

    def _build_ui(self):
        top = ctk.CTkFrame(self)
        top.pack(fill="x", padx=8, pady=8)

        ctk.CTkLabel(top, text="Cut:", width=40).pack(side="left", padx=(4, 2))
        self.cut_menu = ctk.CTkOptionMenu(top, variable=self.cut_var, values=[""], command=lambda _: self.load_selected_cut())
        self.cut_menu.pack(side="left", padx=2)
        ctk.CTkButton(top, text="Refresh", width=80, command=self.refresh_sources).pack(side="left", padx=4)
        ctk.CTkLabel(top, text="XdB:", width=40).pack(side="left", padx=(14, 2))
        ctk.CTkEntry(top, textvariable=self.xdb_var, width=70).pack(side="left", padx=2)
        ctk.CTkButton(top, text="Recalc", width=80, command=self._refresh_metrics).pack(side="left", padx=4)
        ctk.CTkButton(top, text="Clear markers", width=110, command=self.clear_markers).pack(side="left", padx=4)

        tools = ctk.CTkFrame(self, fg_color="transparent")
        tools.pack(fill="x", padx=8, pady=(0, 6))
        ctk.CTkButton(tools, text="Import Cut", width=100, command=self.import_cut_file).pack(side="left", padx=3)
        ctk.CTkButton(tools, text="Remove Imported", width=125, command=self.remove_current_imported_source).pack(side="left", padx=3)
        ctk.CTkButton(tools, text="Export CSV", width=90, command=self.export_current_cut_csv).pack(side="left", padx=3)
        ctk.CTkButton(tools, text="Export PAT", width=90, command=self.export_current_cut_pat).pack(side="left", padx=3)
        ctk.CTkButton(tools, text="Save Session", width=105, command=self.save_advanced_session).pack(side="left", padx=3)
        ctk.CTkButton(tools, text="Load Session", width=105, command=self.load_advanced_session).pack(side="left", padx=3)
        ctk.CTkLabel(tools, text="Theme", width=45).pack(side="right", padx=(8, 2))
        ctk.CTkOptionMenu(
            tools,
            variable=self.appearance_var,
            values=["Dark", "Light", "System"],
            width=110,
            command=self.apply_appearance_theme,
        ).pack(side="right", padx=2)

        body = ctk.CTkFrame(self)
        body.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        body.grid_rowconfigure(0, weight=1)
        body.grid_columnconfigure(0, weight=2)
        body.grid_columnconfigure(1, weight=1)

        left = ctk.CTkFrame(body)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 6), pady=4)
        self.plot_panel = PlotPanel(left, polar=True, title="Advanced View")
        self.plot_panel.pack(fill="both", expand=True, padx=4, pady=4)

        right = ctk.CTkScrollableFrame(body)
        right.grid(row=0, column=1, sticky="nsew", padx=(6, 0), pady=4)

        metrics_box = ctk.CTkFrame(right)
        metrics_box.pack(fill="x", padx=4, pady=(4, 6))
        ctk.CTkLabel(metrics_box, text="Advanced Metrics", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=6, pady=(4, 2))
        self.metrics_text = ctk.CTkTextbox(metrics_box, height=120)
        self.metrics_text.pack(fill="x", padx=6, pady=(0, 6))

        marker_box = ctk.CTkFrame(right)
        marker_box.pack(fill="x", padx=4, pady=6)
        ctk.CTkLabel(marker_box, text="Markers", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=6, pady=(4, 2))
        self.marker_table = MarkerTable(marker_box)
        self.marker_table.pack(fill="x", padx=6, pady=4)
        mbtn = ctk.CTkFrame(marker_box, fg_color="transparent")
        mbtn.pack(fill="x", padx=6, pady=(0, 6))
        ctk.CTkButton(mbtn, text="Delete", width=70, command=self.delete_selected_marker).pack(side="left", padx=2)
        ctk.CTkButton(mbtn, text="Rename", width=70, command=self.rename_selected_marker).pack(side="left", padx=2)
        ctk.CTkButton(mbtn, text="Copy", width=70, command=self.copy_markers_table).pack(side="left", padx=2)
        ctk.CTkButton(mbtn, text="CSV", width=70, command=self.export_markers_csv).pack(side="left", padx=2)

        delta_box = ctk.CTkFrame(right)
        delta_box.pack(fill="x", padx=4, pady=6)
        ctk.CTkLabel(delta_box, text="Deltas", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=6, pady=(4, 2))
        self.delta_tree = ttk.Treeview(delta_box, columns=("pair", "dang", "dmag"), show="headings", height=5)
        self.delta_tree.heading("pair", text="d(mi,mj)")
        self.delta_tree.heading("dang", text="dAng")
        self.delta_tree.heading("dmag", text="dMag(dB)")
        self.delta_tree.column("pair", width=110, anchor="center")
        self.delta_tree.column("dang", width=90, anchor="center")
        self.delta_tree.column("dmag", width=90, anchor="center")
        self.delta_tree.pack(fill="x", padx=6, pady=(0, 6))

        derived_box = ctk.CTkFrame(right)
        derived_box.pack(fill="x", padx=4, pady=6)
        ctk.CTkLabel(derived_box, text="Derived", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=6, pady=(4, 2))
        self.derived_table = DerivedTable(derived_box)
        self.derived_table.pack(fill="x", padx=6, pady=4)
        dbtn = ctk.CTkFrame(derived_box, fg_color="transparent")
        dbtn.pack(fill="x", padx=6, pady=(0, 6))
        ctk.CTkButton(dbtn, text="Evaluate", width=90, command=self.evaluate_derived).pack(side="left", padx=2)
        ctk.CTkButton(dbtn, text="Add Fn", width=80, command=self.add_math_function).pack(side="left", padx=2)

        recon = ctk.CTkFrame(right)
        recon.pack(fill="x", padx=4, pady=6)
        ctk.CTkLabel(recon, text="3D Reconstruction", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=6, pady=(4, 2))
        r1 = ctk.CTkFrame(recon, fg_color="transparent")
        r1.pack(fill="x", padx=6, pady=2)
        ctk.CTkLabel(r1, text="Mode", width=70).pack(side="left")
        ctk.CTkOptionMenu(r1, variable=self.recon_mode_var, values=["omni", "separable", "harmonic"]).pack(side="left", padx=2)
        ctk.CTkLabel(r1, text="Sep", width=45).pack(side="left", padx=(8, 2))
        ctk.CTkOptionMenu(r1, variable=self.recon_sep_mode_var, values=["direct", "preserve_peak"]).pack(side="left", padx=2)

        r2 = ctk.CTkFrame(recon, fg_color="transparent")
        r2.pack(fill="x", padx=6, pady=2)
        ctk.CTkLabel(r2, text="V src", width=70).pack(side="left")
        self.recon_v_menu = ctk.CTkOptionMenu(r2, variable=self.recon_v_source_var, values=[""], width=160)
        self.recon_v_menu.pack(side="left", padx=2)

        r3 = ctk.CTkFrame(recon, fg_color="transparent")
        r3.pack(fill="x", padx=6, pady=2)
        ctk.CTkLabel(r3, text="H src", width=70).pack(side="left")
        self.recon_h_menu = ctk.CTkOptionMenu(r3, variable=self.recon_h_source_var, values=[""], width=160)
        self.recon_h_menu.pack(side="left", padx=2)

        r4 = ctk.CTkFrame(recon, fg_color="transparent")
        r4.pack(fill="x", padx=6, pady=2)
        ctk.CTkLabel(r4, text="a,b", width=70).pack(side="left")
        ctk.CTkEntry(r4, textvariable=self.recon_alpha_var, width=60).pack(side="left", padx=2)
        ctk.CTkEntry(r4, textvariable=self.recon_beta_var, width=60).pack(side="left", padx=2)
        ctk.CTkLabel(r4, text="th,ph pts", width=70).pack(side="left", padx=(8, 2))
        ctk.CTkEntry(r4, textvariable=self.recon_theta_pts_var, width=60).pack(side="left", padx=2)
        ctk.CTkEntry(r4, textvariable=self.recon_phi_pts_var, width=60).pack(side="left", padx=2)

        r5 = ctk.CTkFrame(recon, fg_color="transparent")
        r5.pack(fill="x", padx=6, pady=2)
        ctk.CTkLabel(r5, text="dB min/max", width=70).pack(side="left")
        ctk.CTkEntry(r5, textvariable=self.recon_dbmin_var, width=60).pack(side="left", padx=2)
        ctk.CTkEntry(r5, textvariable=self.recon_dbmax_var, width=60).pack(side="left", padx=2)
        ctk.CTkLabel(r5, text="gamma", width=50).pack(side="left", padx=(8, 2))
        ctk.CTkEntry(r5, textvariable=self.recon_gamma_var, width=60).pack(side="left", padx=2)

        r6 = ctk.CTkFrame(recon, fg_color="transparent")
        r6.pack(fill="x", padx=6, pady=2)
        ctk.CTkCheckBox(r6, text="Wireframe", variable=self.wireframe_var, onvalue=True, offvalue=False).pack(side="left", padx=2)
        ctk.CTkButton(r6, text="Open 3D", width=90, command=self.open_3d_viewer).pack(side="left", padx=4)
        ctk.CTkButton(r6, text="Export OBJ", width=90, command=self.export_3d_obj).pack(side="left", padx=4)
        ctk.CTkButton(r6, text="Export HTML", width=95, command=self.export_3d_html).pack(side="left", padx=4)

        r7 = ctk.CTkFrame(recon, fg_color="transparent")
        r7.pack(fill="x", padx=6, pady=(4, 6))
        self.recon_progress = ctk.CTkProgressBar(r7, mode="indeterminate", width=180)
        self.recon_progress.pack(side="left", padx=(0, 6))
        self.recon_progress.stop()
        ctk.CTkLabel(r7, textvariable=self.recon_status_var).pack(side="left")

        cad = ctk.CTkFrame(right)
        cad.pack(fill="x", padx=4, pady=6)
        ctk.CTkLabel(cad, text="CAD Importado (AEDT)", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=6, pady=(4, 2))
        ctk.CTkLabel(cad, textvariable=self.cad_status_var, anchor="w", justify="left").pack(fill="x", padx=6, pady=(0, 4))
        cbtn = ctk.CTkFrame(cad, fg_color="transparent")
        cbtn.pack(fill="x", padx=6, pady=(0, 6))
        self.btn_open_cad = ctk.CTkButton(cbtn, text="Open CAD 3D", width=110, command=self.open_cad_viewer)
        self.btn_open_cad.pack(side="left", padx=(0, 4))
        self.btn_reload_cad = ctk.CTkButton(cbtn, text="Reparse CAD", width=110, command=self.reload_cad_from_manifest)
        self.btn_reload_cad.pack(side="left", padx=4)

        ctk.CTkLabel(self, textvariable=self.status_var).pack(fill="x", padx=8, pady=(0, 6))

        self.interactor: Optional[AdvancedPlotInteractor] = None
        self._build_interactor(is_polar=True)
        self._bind_context_menus()

    def _set_status(self, text: str):
        self.status_var.set(str(text))
        try:
            self.app._set_status(str(text))
        except Exception:
            pass

    def _build_interactor(self, is_polar: bool):
        if self.interactor is not None:
            try:
                self.interactor.disconnect()
            except Exception:
                pass
        self.interactor = AdvancedPlotInteractor(
            ax=self.plot_panel.ax,
            canvas=self.plot_panel.canvas,
            get_series_callable=self._series_for_interactor,
            is_polar=is_polar,
            on_change=self._on_markers_changed,
            on_status=self._set_status,
            on_context_menu=self._show_plot_context_menu,
            tracer=self.tracer,
            drag_hz=60.0,
            table_refresh_ms=220.0,
        )

    def _bind_context_menus(self):
        self.marker_table.tree.bind("<Button-3>", self._on_marker_table_context)
        self.marker_table.tree.bind("<Button-2>", self._on_marker_table_context)
        self.delta_tree.bind("<Button-3>", self._on_delta_table_context)
        self.delta_tree.bind("<Button-2>", self._on_delta_table_context)
        self.derived_table.tree.bind("<Button-3>", self._on_derived_table_context)
        self.derived_table.tree.bind("<Button-2>", self._on_derived_table_context)

    def _series_for_interactor(self):
        return self.current_angles, self.current_values, self.current_kind

    def _series_signature(self, a: np.ndarray, v: np.ndarray) -> tuple:
        return (
            int(a.size),
            float(a[0]) if a.size else 0.0,
            float(a[-1]) if a.size else 0.0,
            float(np.sum(a)) if a.size else 0.0,
            float(np.sum(v)) if v.size else 0.0,
        )

    def apply_appearance_theme(self, value: str):
        mode = str(value or "Dark").strip().capitalize()
        if mode not in ("Dark", "Light", "System"):
            mode = "Dark"
        try:
            ctk.set_appearance_mode(mode)
            self.appearance_var.set(mode)
            self._set_status(f"Tema aplicado: {mode}")
            self._draw_current_cut()
        except Exception as e:
            messagebox.showerror("Theme", str(e))

    @staticmethod
    def _parse_cut_pairs_from_text(text: str) -> tuple[np.ndarray, np.ndarray]:
        num_re = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
        angles: List[float] = []
        values: List[float] = []
        import re

        for raw in str(text or "").splitlines():
            line = raw.strip()
            if not line:
                continue
            if line.startswith(("#", ";", "!", "//")):
                continue
            normalized = line.replace(",", ".")
            nums = re.findall(num_re, normalized)
            if len(nums) < 2:
                continue
            try:
                ang = float(nums[0])
                val = float(nums[1])
            except Exception:
                continue
            angles.append(ang)
            values.append(val)

        if len(angles) < 3:
            raise ValueError("Nao foi possivel extrair pares angulo/valor suficientes.")

        ang_arr = np.asarray(angles, dtype=float).reshape(-1)
        val_arr = np.asarray(values, dtype=float).reshape(-1)
        if ang_arr.size != val_arr.size:
            raise ValueError("Quantidade de angulos e valores divergente.")
        return ang_arr, val_arr

    @staticmethod
    def _auto_kind_from_angles(angles: np.ndarray) -> str:
        a = np.asarray(angles, dtype=float).reshape(-1)
        if a.size == 0:
            return "H"
        amin = float(np.nanmin(a))
        amax = float(np.nanmax(a))
        span = abs(amax - amin)
        if amin >= -95.0 and amax <= 95.0:
            return "V"
        if amin >= -190.0 and amax <= 190.0 and span >= 180.0:
            return "H"
        return "H" if span >= 150.0 else "V"

    @staticmethod
    def _ensure_linear(values: np.ndarray) -> np.ndarray:
        v = np.asarray(values, dtype=float).reshape(-1)
        if v.size == 0:
            return v
        if np.any(v < 0.0) or float(np.nanmax(v)) > 5.0:
            return np.power(10.0, v / 20.0)
        return np.clip(v, 0.0, None)

    def import_cut_file(self):
        path = filedialog.askopenfilename(
            title="Import cut file",
            filetypes=[
                ("Pattern/Data", "*.csv *.txt *.pat *.prn"),
                ("CSV", "*.csv"),
                ("Text", "*.txt"),
                ("PAT", "*.pat"),
                ("PRN", "*.prn"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            angles, values_raw = self._parse_cut_pairs_from_text(text)
            values = self._ensure_linear(values_raw)
            guess = self._auto_kind_from_angles(angles)
            answer = simpledialog.askstring(
                "Tipo do corte",
                "Informe tipo do corte [H ou V]:",
                initialvalue=guess,
                parent=self,
            )
            if answer is None:
                return
            kind = str(answer).strip().upper()
            if kind not in ("H", "V"):
                messagebox.showwarning("Import", "Tipo invalido. Use H ou V.")
                return

            stem = os.path.splitext(os.path.basename(path))[0].strip() or "importado"
            key_base = f"Importado {stem}"
            key = key_base
            idx = 2
            while (key in self.user_sources) or (key in self.source_map):
                key = f"{key_base} ({idx})"
                idx += 1

            self.user_sources[key] = {
                "kind": kind,
                "angles": np.asarray(angles, dtype=float),
                "values": np.asarray(values, dtype=float),
                "origin": os.path.abspath(path),
            }
            self.refresh_sources(preferred_key=key)
            self._set_status(f"Corte importado: {key}")
        except Exception as e:
            messagebox.showerror("Import cut", str(e))

    def remove_current_imported_source(self):
        key = str(self.current_key or "").strip()
        if not key:
            return
        if key not in self.user_sources:
            messagebox.showinfo("Imported sources", "A fonte atual nao e um corte importado.")
            return
        self.user_sources.pop(key, None)
        self.refresh_sources()
        self._set_status(f"Fonte importada removida: {key}")

    def export_current_cut_csv(self):
        if self.current_angles is None or self.current_values is None:
            messagebox.showwarning("Export CSV", "Nao ha corte carregado.")
            return
        path = filedialog.asksaveasfilename(
            title="Export current cut CSV",
            defaultextension=".csv",
            initialfile=f"{(self.current_key or 'cut').replace(' ', '_')}.csv",
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return
        a = np.asarray(self.current_angles, dtype=float)
        v = np.asarray(self.current_values, dtype=float)
        if self.current_kind == "H":
            aw = np.mod(a, 360.0)
            idx = np.argsort(aw)
            a = np.where(aw[idx] > 180.0, aw[idx] - 360.0, aw[idx])
            v = v[idx]
        else:
            idx = np.argsort(a)
            a = a[idx]
            v = v[idx]
        try:
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                wr = csv.writer(f)
                wr.writerow(["Angle_deg", "Level_linear", "Level_dB"])
                for ang, lin in zip(a, v):
                    db = 20.0 * math.log10(max(float(lin), 1e-12))
                    wr.writerow([f"{float(ang):.6f}", f"{float(lin):.8f}", f"{float(db):.4f}"])
            self._set_status(f"CSV exportado: {path}")
        except Exception as e:
            messagebox.showerror("Export CSV", str(e))

    def export_current_cut_pat(self):
        if self.current_angles is None or self.current_values is None:
            messagebox.showwarning("Export PAT", "Nao ha corte carregado.")
            return
        path = filedialog.asksaveasfilename(
            title="Export current cut PAT (simples)",
            defaultextension=".pat",
            initialfile=f"{(self.current_key or 'cut').replace(' ', '_')}.pat",
            filetypes=[("PAT", "*.pat"), ("All files", "*.*")],
        )
        if not path:
            return
        a = np.asarray(self.current_angles, dtype=float)
        v = np.asarray(self.current_values, dtype=float)
        if self.current_kind == "H":
            aw = np.mod(a, 360.0)
            idx = np.argsort(aw)
            a = np.where(aw[idx] > 180.0, aw[idx] - 360.0, aw[idx])
            v = v[idx]
        else:
            idx = np.argsort(a)
            a = a[idx]
            v = v[idx]
        try:
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                f.write("; EFTX Advanced Visualization - PAT simples\n")
                f.write(f"; Source: {self.current_key or '-'}\n")
                f.write(f"; Kind: {self.current_kind}\n")
                f.write("; Angle_deg Level_dB\n")
                for ang, lin in zip(a, v):
                    db = 20.0 * math.log10(max(float(lin), 1e-12))
                    f.write(f"{float(ang):.6f} {float(db):.4f}\n")
            self._set_status(f"PAT exportado: {path}")
        except Exception as e:
            messagebox.showerror("Export PAT", str(e))

    def save_advanced_session(self):
        path = filedialog.asksaveasfilename(
            title="Salvar sessao do modeler avancado",
            defaultextension=".json",
            initialfile=f"advanced_modeler_session_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        markers = []
        if self.interactor is not None:
            for mk in self.interactor.markers():
                markers.append(
                    {
                        "name": mk.name,
                        "kind": mk.kind,
                        "cut": mk.cut,
                        "theta_deg": mk.theta_deg,
                        "phi_deg": mk.phi_deg,
                        "ang_deg": mk.ang_deg,
                        "mag_lin": mk.mag_lin,
                        "mag_db": mk.mag_db,
                    }
                )

        imported = {}
        for key, item in self.user_sources.items():
            imported[key] = {
                "kind": str(item.get("kind", "H")),
                "angles": np.asarray(item.get("angles", []), dtype=float).tolist(),
                "values": np.asarray(item.get("values", []), dtype=float).tolist(),
                "origin": str(item.get("origin", "")),
            }

        payload = {
            "saved_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "appearance": str(self.appearance_var.get() or "Dark"),
            "current_key": str(self.current_key or ""),
            "current_kind": str(self.current_kind or "H"),
            "xdb": str(self.xdb_var.get() or "10"),
            "imported_sources": imported,
            "markers": markers,
        }
        try:
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            self._set_status(f"Sessao salva: {path}")
        except Exception as e:
            messagebox.showerror("Save session", str(e))

    def load_advanced_session(self):
        path = filedialog.askopenfilename(
            title="Carregar sessao do modeler avancado",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError("Formato de sessao invalido.")

            imported = data.get("imported_sources", {})
            self.user_sources = {}
            if isinstance(imported, dict):
                for key, item in imported.items():
                    if not isinstance(item, dict):
                        continue
                    ang = np.asarray(item.get("angles", []), dtype=float).reshape(-1)
                    val = np.asarray(item.get("values", []), dtype=float).reshape(-1)
                    if ang.size == 0 or val.size == 0 or ang.size != val.size:
                        continue
                    self.user_sources[str(key)] = {
                        "kind": str(item.get("kind", "H")).upper(),
                        "angles": ang,
                        "values": np.clip(val, 0.0, None),
                        "origin": str(item.get("origin", "")),
                    }

            self.xdb_var.set(str(data.get("xdb", self.xdb_var.get())))
            self.apply_appearance_theme(str(data.get("appearance", self.appearance_var.get())))
            preferred = str(data.get("current_key", ""))
            self.refresh_sources(preferred_key=preferred)

            self.clear_markers()
            markers_raw = data.get("markers", [])
            if self.interactor is not None and isinstance(markers_raw, list):
                for item in markers_raw:
                    if not isinstance(item, dict):
                        continue
                    try:
                        mk = MarkerValue(
                            name=str(item.get("name", "m")),
                            kind=str(item.get("kind", "2D")),
                            cut=item.get("cut"),
                            theta_deg=item.get("theta_deg"),
                            phi_deg=item.get("phi_deg"),
                            ang_deg=float(item.get("ang_deg", 0.0)),
                            mag_lin=float(item.get("mag_lin", 0.0)),
                            mag_db=float(item.get("mag_db", -120.0)),
                        )
                        self.interactor.add_marker_value(mk)
                    except Exception:
                        continue
            self._set_status(f"Sessao carregada: {path}")
        except Exception as e:
            messagebox.showerror("Load session", str(e))

    def refresh_sources(self, preferred_key: str = ""):
        src: Dict[str, dict] = {}

        def add(name: str, kind: str, angles, values):
            if angles is None or values is None:
                return
            a = np.asarray(angles, dtype=float).reshape(-1)
            v = np.asarray(values, dtype=float).reshape(-1)
            if a.size == 0 or v.size == 0 or a.size != v.size:
                return
            v = np.clip(v, 0.0, None)
            vmax = float(np.max(v))
            if vmax > 1e-12:
                v = v / vmax
            src[name] = {"kind": kind, "angles": a, "values": v, "sig": self._series_signature(a, v)}

        add("HRP Arquivo", "H", getattr(self.app, "h_angles", None), getattr(self.app, "h_vals", None))
        add("VRP Arquivo", "V", getattr(self.app, "v_angles", None), getattr(self.app, "v_vals", None))
        add("HRP Composicao", "H", getattr(self.app, "horz_angles", None), getattr(self.app, "horz_values", None))
        add("VRP Composicao", "V", getattr(self.app, "vert_angles", None), getattr(self.app, "vert_values", None))
        add("HRP Study H1", "H", getattr(self.app, "study_h1_angles", None), getattr(self.app, "study_h1_vals", None))
        add("VRP Study V1", "V", getattr(self.app, "study_v1_angles", None), getattr(self.app, "study_v1_vals", None))
        add("HRP Study H2", "H", getattr(self.app, "study_h2_angles", None), getattr(self.app, "study_h2_vals", None))
        add("VRP Study V2", "V", getattr(self.app, "study_v2_angles", None), getattr(self.app, "study_v2_vals", None))

        for key, item in list(self.user_sources.items()):
            try:
                add(key, str(item.get("kind", "H")).upper(), item.get("angles"), item.get("values"))
            except Exception:
                continue

        self.source_map = src
        keys = list(src.keys()) or [""]
        self.cut_menu.configure(values=keys)
        if preferred_key and preferred_key in src:
            self.cut_var.set(preferred_key)
        elif self.cut_var.get() not in src:
            self.cut_var.set(keys[0])

        v_keys = [k for k, item in src.items() if item["kind"] == "V"] or [""]
        h_keys = [k for k, item in src.items() if item["kind"] == "H"] or [""]
        self.recon_v_menu.configure(values=v_keys)
        self.recon_h_menu.configure(values=h_keys)
        if self.recon_v_source_var.get() not in v_keys:
            self.recon_v_source_var.set(v_keys[0])
        if self.recon_h_source_var.get() not in h_keys:
            self.recon_h_source_var.set(h_keys[0])

        self._refresh_cad_status()
        self.load_selected_cut()

    def _refresh_cad_status(self):
        runtime = getattr(self.app, "aedt_live_cad_3d", None)
        manifest = getattr(self.app, "aedt_live_cad_manifest", None)
        mesh_count = 0
        file_count = 0
        src = "-"
        if isinstance(runtime, dict):
            meshes = runtime.get("meshes", [])
            files = runtime.get("files", [])
            if isinstance(meshes, list):
                mesh_count = len(meshes)
            if isinstance(files, list):
                file_count = len(files)
            src = str(runtime.get("source", "-"))
        elif isinstance(manifest, dict):
            stats = manifest.get("stats", {})
            mesh_count = int(stats.get("mesh_count", 0)) if isinstance(stats, dict) else 0
            file_count = int(stats.get("file_count", 0)) if isinstance(stats, dict) else 0
            src = str(manifest.get("source", "-"))
        if mesh_count > 0:
            self.cad_status_var.set(f"CAD disponível: {mesh_count} malha(s) em {file_count} arquivo(s) | Fonte: {src}")
            if getattr(self, "btn_open_cad", None) is not None:
                self.btn_open_cad.configure(state="normal")
            if getattr(self, "btn_reload_cad", None) is not None:
                self.btn_reload_cad.configure(state="normal")
        else:
            self.cad_status_var.set("CAD: nenhum importado.")
            if getattr(self, "btn_open_cad", None) is not None:
                self.btn_open_cad.configure(state="disabled")
            if getattr(self, "btn_reload_cad", None) is not None:
                self.btn_reload_cad.configure(state="disabled")

    def _build_cad_runtime_from_manifest(self, manifest: dict) -> Optional[dict]:
        if not isinstance(manifest, dict):
            return None
        files = manifest.get("files", [])
        if not isinstance(files, list) or not files:
            return None
        style_map: Dict[tuple, dict] = {}
        for item in manifest.get("meshes", []) if isinstance(manifest.get("meshes", []), list) else []:
            if not isinstance(item, dict):
                continue
            key = (
                os.path.abspath(str(item.get("source_path", "") or "")),
                str(item.get("object_name", "") or ""),
                str(item.get("group_name", "") or ""),
                str(item.get("obj_material", "") or ""),
            )
            style_map[key] = item

        runtime_meshes: List[dict] = []
        runtime_files: List[dict] = []
        errors: List[str] = []

        for entry in files:
            if not isinstance(entry, dict):
                continue
            p = os.path.abspath(str(entry.get("path", "") or ""))
            if not p or not os.path.isfile(p):
                errors.append(f"Arquivo ausente: {p}")
                continue
            try:
                model = parse_obj_file(p)
            except Exception as e:
                errors.append(f"Falha parser OBJ ({os.path.basename(p)}): {e}")
                continue
            summary = model.summary()
            expected_sha = str(entry.get("sha256", "") or "")
            got_sha = str(summary.get("sha256", "") or "")
            if expected_sha and got_sha and expected_sha != got_sha:
                errors.append(f"Hash divergente: {os.path.basename(p)}")
            runtime_files.append(summary)
            for chunk in model.mesh_chunks(split_mode="object_group_material"):
                verts = np.asarray(chunk.get("vertices", np.zeros((0, 3), dtype=float)), dtype=float)
                faces = np.asarray(chunk.get("faces", np.zeros((0, 3), dtype=int)), dtype=int)
                if verts.size == 0 or faces.size == 0:
                    continue
                key = (
                    p,
                    str(chunk.get("object_name", "") or ""),
                    str(chunk.get("group_name", "") or ""),
                    str(chunk.get("material", "") or ""),
                )
                style = style_map.get(key, {})
                runtime_meshes.append(
                    {
                        "name": str(chunk.get("name", "") or os.path.splitext(os.path.basename(p))[0]),
                        "source_path": p,
                        "vertices": verts,
                        "faces": faces,
                        "color": str(style.get("color", "#86b6f6")),
                        "opacity": float(style.get("opacity", 0.85)),
                        "material": str(chunk.get("material", "") or "Undefined"),
                        "object_name": str(chunk.get("object_name", "") or ""),
                        "group_name": str(chunk.get("group_name", "") or ""),
                        "obj_material": str(chunk.get("material", "") or ""),
                    }
                )
        if not runtime_meshes:
            return None
        return {
            "version": int(manifest.get("version", 1) or 1),
            "source": str(manifest.get("source", "AEDT") or "AEDT"),
            "imported_at": str(manifest.get("imported_at", "") or ""),
            "files": runtime_files,
            "meshes": runtime_meshes,
            "stats": {
                "file_count": int(len(runtime_files)),
                "mesh_count": int(len(runtime_meshes)),
                "vertex_count": int(sum(int(np.asarray(x["vertices"]).shape[0]) for x in runtime_meshes)),
                "face_count": int(sum(int(np.asarray(x["faces"]).shape[0]) for x in runtime_meshes)),
            },
            "errors": errors,
        }

    def _resolve_cad_runtime(self, force_reparse: bool = False) -> Optional[dict]:
        runtime = getattr(self.app, "aedt_live_cad_3d", None)
        if isinstance(runtime, dict) and isinstance(runtime.get("meshes"), list) and runtime.get("meshes") and not force_reparse:
            self._cad_runtime_cache = runtime
            return runtime
        manifest = getattr(self.app, "aedt_live_cad_manifest", None)
        rebuilt = self._build_cad_runtime_from_manifest(manifest if isinstance(manifest, dict) else {})
        if isinstance(rebuilt, dict):
            self._cad_runtime_cache = rebuilt
            try:
                setattr(self.app, "aedt_live_cad_3d", rebuilt)
            except Exception:
                pass
            return rebuilt
        return None

    def reload_cad_from_manifest(self):
        rebuilt = self._resolve_cad_runtime(force_reparse=True)
        if rebuilt is None:
            messagebox.showwarning("CAD", "Nao foi possivel reconstruir o CAD a partir do manifesto.")
            self._refresh_cad_status()
            return
        errors = rebuilt.get("errors", [])
        if isinstance(errors, list) and errors:
            self._set_status(f"CAD reparse com avisos: {errors[0]}")
        else:
            self._set_status("CAD reparse concluido.")
        self._refresh_cad_status()

    def open_cad_viewer(self):
        payload = self._resolve_cad_runtime(force_reparse=False)
        if payload is None:
            messagebox.showwarning("CAD 3D", "Nenhum CAD importado disponivel. Use a aba Analise Mecanica.")
            return
        meshes = payload.get("meshes", [])
        if not isinstance(meshes, list) or not meshes:
            messagebox.showwarning("CAD 3D", "Payload CAD sem malhas validas.")
            return
        try:
            backend = open_meshes_view(meshes, title="EFTX CAD 3D Viewer")
            self._set_status(f"CAD 3D viewer aberto ({backend}).")
        except Exception as e:
            messagebox.showerror("CAD 3D", str(e))
    def load_selected_cut(self):
        key = self.cut_var.get().strip()
        if not key or key not in self.source_map:
            self.current_key = None
            self.current_angles = None
            self.current_values = None
            self.current_kind = "H"
            self.plot_panel.reset_axes(True, "Advanced View")
            self._build_interactor(is_polar=True)
            self._on_markers_changed([], "reset")
            self._set_status("No source selected.")
            return

        item = self.source_map[key]
        self.current_key = key
        self.current_kind = str(item["kind"]).upper()
        self.current_angles = np.asarray(item["angles"], dtype=float)
        self.current_values = np.asarray(item["values"], dtype=float)

        self._draw_current_cut()
        self.clear_markers()
        self._refresh_metrics()
        self._set_status(f"Loaded {key}.")

    def _draw_current_cut(self):
        if self.current_angles is None or self.current_values is None:
            return
        kind = self.current_kind
        a = np.asarray(self.current_angles, dtype=float)
        v = np.asarray(self.current_values, dtype=float)

        if kind == "H":
            self.plot_panel.reset_axes(True, "Advanced HRP (polar)")
            ax = self.plot_panel.ax
            aw = np.mod(a, 360.0)
            idx = np.argsort(aw)
            ax.plot(np.deg2rad(aw[idx]), v[idx], color="#d55e00", linewidth=1.4)
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
            ax.grid(True, alpha=0.3)
            self._build_interactor(is_polar=True)
        else:
            self.plot_panel.reset_axes(False, "Advanced VRP (planar)")
            ax = self.plot_panel.ax
            idx = np.argsort(a)
            ax.plot(a[idx], v[idx], color="#1f77b4", linewidth=1.4)
            ax.set_xlabel("Angle [deg]")
            ax.set_ylabel("Mag (lin)")
            ax.set_xlim([float(np.min(a)), float(np.max(a))])
            ax.set_ylim([0.0, max(1.05, float(np.max(v) * 1.05))])
            ax.grid(True, alpha=0.3)
            self._build_interactor(is_polar=False)
        self.plot_panel.canvas.draw_idle()

    def _refresh_metrics(self):
        if self.current_angles is None or self.current_values is None:
            self.metrics_text.configure(state="normal")
            self.metrics_text.delete("1.0", "end")
            self.metrics_text.insert("1.0", "No data.")
            self.metrics_text.configure(state="disabled")
            return

        try:
            xdb = float(self.xdb_var.get().replace(",", "."))
        except Exception:
            xdb = 10.0
        wrap = self.current_kind == "H"
        m = summarize_advanced_metrics(self.current_angles, self.current_values, xdb=xdb, wrap=wrap)
        lines = [
            f"Series: {self.current_key}",
            f"Max dB: {m['max_db']:.3f}",
            f"Avg dB: {m['avg_db']:.3f}",
            f"Avg lin: {m['avg_lin']:.6f}",
            f"Pk2Pk dB: {m['pk2pk_db']:.3f}",
            f"BW_{xdb:.1f}dB: {m['bw_xdb']:.3f} deg",
            f"BW edges: {m['bw_left_deg']:.3f} / {m['bw_right_deg']:.3f}",
            f"Peak: {m['peak_db']:.3f} dB @ {m['peak_deg']:.3f} deg",
        ]
        self.metrics_text.configure(state="normal")
        self.metrics_text.delete("1.0", "end")
        self.metrics_text.insert("1.0", "\n".join(lines))
        self.metrics_text.configure(state="disabled")

    def _schedule_heavy_refresh(self, delay_ms: int = 220):
        if self._table_refresh_job is not None:
            try:
                self.after_cancel(self._table_refresh_job)
            except Exception:
                pass
            self._table_refresh_job = None
        self._table_refresh_job = self.after(delay_ms, self._refresh_tables_heavy)

    def _refresh_tables_heavy(self):
        self._table_refresh_job = None
        t0 = self.tracer.start()
        markers = list(self._pending_markers)
        self.marker_table.set_markers(markers)
        self._refresh_deltas(markers)
        self.evaluate_derived()
        self.tracer.log_if_slow("TABLE_REFRESH", t0, extra="advanced-heavy")

    def _on_markers_changed(self, markers: List[MarkerValue], reason: str = ""):
        self._pending_markers = list(markers)
        if str(reason) == "motion":
            self._schedule_heavy_refresh(delay_ms=220)
            return
        if self._table_refresh_job is not None:
            try:
                self.after_cancel(self._table_refresh_job)
            except Exception:
                pass
            self._table_refresh_job = None
        self._refresh_tables_heavy()

    def _refresh_deltas(self, markers: List[MarkerValue]):
        t0 = self.tracer.start()
        self.delta_tree.delete(*self.delta_tree.get_children())
        if len(markers) < 2:
            self.tracer.log_if_slow("TABLE_REFRESH", t0, extra="deltas-empty")
            return

        pairs = []
        if len(markers) >= 2:
            pairs.append((len(markers) - 2, len(markers) - 1))
        for i in range(min(4, len(markers) - 1)):
            pairs.append((i, i + 1))

        seen = set()
        for i, j in pairs:
            if i < 0 or j < 0 or i >= len(markers) or j >= len(markers):
                continue
            if (i, j) in seen:
                continue
            seen.add((i, j))
            a = markers[i]
            b = markers[j]
            if self.current_kind == "H":
                dang = float(ang_dist_deg(a.ang_deg, b.ang_deg))
            else:
                dang = abs(float(b.ang_deg) - float(a.ang_deg))
            dmag = float(b.mag_db - a.mag_db)
            self.delta_tree.insert("", "end", values=(f"d({a.name},{b.name})", f"{dang:.3f}", f"{dmag:.3f}"))
        self.tracer.log_if_slow("TABLE_REFRESH", t0, extra="deltas")

    def clear_markers(self):
        if self.interactor:
            self.interactor.clear_markers()

    def delete_selected_marker(self):
        name = self.marker_table.selected_name()
        if name and self.interactor:
            self.interactor.delete_marker(name)

    def rename_selected_marker(self):
        name = self.marker_table.selected_name()
        if not name or not self.interactor:
            return
        new_name = simpledialog.askstring("Rename marker", "New marker name:", initialvalue=name, parent=self)
        if new_name:
            self.interactor.rename_marker(name, new_name)

    def copy_markers_table(self):
        try:
            self.marker_table.copy_to_clipboard(self)
            self._set_status("Markers table copied.")
        except Exception as e:
            messagebox.showerror("Copy error", str(e))

    def export_markers_csv(self):
        path = filedialog.asksaveasfilename(title="Export markers CSV", defaultextension=".csv", filetypes=[("CSV", "*.csv"), ("All files", "*.*")])
        if path:
            self.marker_table.export_csv(path)

    def evaluate_derived(self):
        t0 = self.tracer.start()
        markers = self.interactor.markers() if self.interactor else []
        A = markers[0] if len(markers) >= 1 else None
        B = markers[1] if len(markers) >= 2 else None
        try:
            xdb = float(self.xdb_var.get().replace(",", "."))
        except Exception:
            xdb = 10.0
        applies = "HRP" if self.current_kind == "H" else "VRP"
        rows = evaluate_functions(self.functions, A=A, B=B, params={"xdb": xdb}, applies_to=applies)
        self.derived_table.set_rows(rows)
        self.tracer.log_if_slow("TABLE_REFRESH", t0, extra="derived")

    def add_math_function(self):
        dlg = ctk.CTkToplevel(self)
        dlg.title("Add math function")
        dlg.geometry("520x220")
        dlg.transient(self)
        dlg.grab_set()

        name_var = tk.StringVar(value="New Function")
        expr_var = tk.StringVar(value="B.mag_db - A.mag_db")
        apply_var = tk.StringVar(value="ANY")

        for lbl, var in (("Name", name_var), ("Expr", expr_var)):
            row = ctk.CTkFrame(dlg)
            row.pack(fill="x", padx=10, pady=8)
            ctk.CTkLabel(row, text=lbl, width=70).pack(side="left")
            ctk.CTkEntry(row, textvariable=var, width=380).pack(side="left", padx=4)
        row = ctk.CTkFrame(dlg)
        row.pack(fill="x", padx=10, pady=8)
        ctk.CTkLabel(row, text="Applies", width=70).pack(side="left")
        ctk.CTkOptionMenu(row, variable=apply_var, values=["ANY", "HRP", "VRP", "3D"]).pack(side="left", padx=4)

        def _save():
            name = name_var.get().strip() or "Function"
            expr = expr_var.get().strip()
            if not expr:
                messagebox.showwarning("Invalid", "Expression is empty.", parent=dlg)
                return
            fn = MathFunctionDef(name=name, expr=expr, params_schema={}, applies_to=apply_var.get().strip().upper())
            self.functions.append(fn)
            save_user_functions(self.functions)
            dlg.destroy()
            self.evaluate_derived()

        row = ctk.CTkFrame(dlg, fg_color="transparent")
        row.pack(fill="x", padx=10, pady=10)
        ctk.CTkButton(row, text="Save", command=_save, width=90).pack(side="right", padx=4)
        ctk.CTkButton(row, text="Cancel", command=dlg.destroy, width=90).pack(side="right", padx=4)
    def _get_recon_cuts(self):
        v_key = self.recon_v_source_var.get().strip()
        h_key = self.recon_h_source_var.get().strip()
        cut_v = None
        cut_h = None
        if v_key and v_key in self.source_map and self.source_map[v_key]["kind"] == "V":
            x = self.source_map[v_key]
            cut_v = cut_from_arrays("V", x["angles"], x["values"], {"source": v_key})
        if h_key and h_key in self.source_map and self.source_map[h_key]["kind"] == "H":
            x = self.source_map[h_key]
            cut_h = cut_from_arrays("H", x["angles"], x["values"], {"source": h_key})
        return cut_v, cut_h

    def _parse_recon_numeric(self):
        def _num(var: tk.StringVar, default: float):
            try:
                return float(var.get().replace(",", "."))
            except Exception:
                return float(default)

        th_n = max(5, int(_num(self.recon_theta_pts_var, 181)))
        ph_n = max(5, int(_num(self.recon_phi_pts_var, 361)))
        alpha = _num(self.recon_alpha_var, 1.0)
        beta = _num(self.recon_beta_var, 1.0)
        db_min = _num(self.recon_dbmin_var, -40.0)
        db_max = _num(self.recon_dbmax_var, 0.0)
        gamma = _num(self.recon_gamma_var, 1.0)
        return th_n, ph_n, alpha, beta, db_min, db_max, gamma

    def _build_recon_request(self):
        v_key = self.recon_v_source_var.get().strip()
        h_key = self.recon_h_source_var.get().strip()
        mode = self.recon_mode_var.get().strip().lower()
        sep_mode = self.recon_sep_mode_var.get().strip().lower()
        th_n, ph_n, alpha, beta, db_min, db_max, gamma = self._parse_recon_numeric()
        v_sig = self.source_map.get(v_key, {}).get("sig") if v_key else None
        h_sig = self.source_map.get(h_key, {}).get("sig") if h_key else None
        key = (v_key, v_sig, h_key, h_sig, mode, sep_mode, float(alpha), float(beta), int(th_n), int(ph_n), float(db_min), float(db_max), float(gamma))
        return {
            "key": key,
            "mode": mode,
            "sep_mode": sep_mode,
            "th_n": th_n,
            "ph_n": ph_n,
            "alpha": alpha,
            "beta": beta,
            "db_min": db_min,
            "db_max": db_max,
            "gamma": gamma,
        }

    def _build_spherical_pattern(self):
        cut_v, cut_h = self._get_recon_cuts()
        req = self._build_recon_request()
        theta = np.linspace(0.0, 180.0, req["th_n"])
        phi = np.linspace(-180.0, 180.0, req["ph_n"])
        return reconstruct_spherical(
            cut_v=cut_v,
            cut_h=cut_h,
            mode=req["mode"],
            theta_deg=theta,
            phi_deg=phi,
            alpha=req["alpha"],
            beta=req["beta"],
            separable_mode=req["sep_mode"],
        )

    def _recon_worker(self):
        t0 = self.tracer.start()
        pat = self._build_spherical_pattern()
        self.tracer.log_if_slow("RECON_3D", t0, extra="worker-build")
        return pat

    def _poll_recon_future(self):
        if self.recon_future is None:
            return
        if not self.recon_future.done():
            self.after(120, self._poll_recon_future)
            return

        fut = self.recon_future
        self.recon_future = None
        self.recon_progress.stop()
        try:
            pat = fut.result()
            req = self._build_recon_request()
            self._recon_cache[req["key"]] = pat
            self.last_spherical = pat
            self.recon_status_var.set("Ready")
            self._open_3d_with_pattern(pat)
        except Exception as e:
            self.recon_status_var.set("Error")
            messagebox.showerror("3D error", str(e))

    def _open_3d_with_pattern(self, pat):
        _th_n, _ph_n, _a, _b, db_min, db_max, gamma = self._parse_recon_numeric()
        t0 = self.tracer.start()
        backend = open_3d_view(
            pat,
            title="EFTX 3D Viewer",
            db_min=db_min,
            db_max=db_max,
            gamma=gamma,
            wireframe=bool(self.wireframe_var.get()),
            on_pick=self._on_3d_pick,
        )
        self.tracer.log_if_slow("RECON_3D", t0, extra=f"open-viewer-{backend}")
        self._set_status(f"3D viewer opened ({backend}).")

    def _on_3d_pick(self, theta_deg: float, phi_deg: float, mag_lin: float):
        if self.interactor is None:
            return
        if self.current_kind == "H":
            ang = float(wrap_phi_deg(phi_deg))
            marker = MarkerValue(name=f"m{len(self.interactor.markers()) + 1}", kind="3D", cut="HRP", theta_deg=float(theta_deg), phi_deg=ang, ang_deg=ang, mag_lin=float(mag_lin), mag_db=float(20.0 * math.log10(max(mag_lin, 1e-12))))
        else:
            elev = 90.0 - float(theta_deg)
            marker = MarkerValue(name=f"m{len(self.interactor.markers()) + 1}", kind="3D", cut="VRP", theta_deg=float(theta_deg), phi_deg=float(wrap_phi_deg(phi_deg)), ang_deg=float(elev), mag_lin=float(mag_lin), mag_db=float(20.0 * math.log10(max(mag_lin, 1e-12))))
        self.interactor.add_marker_value(marker)
        self._set_status(f"3D pick -> marker: theta={theta_deg:.2f}, phi={phi_deg:.2f}, mag={marker.mag_db:.2f} dB")

    def open_3d_viewer(self):
        if self.recon_future is not None and not self.recon_future.done():
            messagebox.showwarning("3D", "Reconstrucao 3D em andamento.")
            return
        req = self._build_recon_request()
        if req["key"] in self._recon_cache:
            self.last_spherical = self._recon_cache[req["key"]]
            self.recon_status_var.set("Cache")
            self._open_3d_with_pattern(self.last_spherical)
            return
        self.recon_status_var.set("Reconstructing...")
        self.recon_progress.start()
        self.recon_future = self.recon_executor.submit(self._recon_worker)
        self.after(120, self._poll_recon_future)

    def export_3d_obj(self):
        try:
            pat = self.last_spherical or self._build_spherical_pattern()
        except Exception as e:
            messagebox.showerror("Build error", str(e))
            return
        path = filedialog.asksaveasfilename(title="Export OBJ", defaultextension=".obj", filetypes=[("OBJ", "*.obj"), ("All files", "*.*")])
        if not path:
            return
        _th_n, _ph_n, _a, _b, _dbmin, _dbmax, gamma = self._parse_recon_numeric()
        export_obj(pat, path, gamma=gamma)

    def export_3d_html(self):
        try:
            pat = self.last_spherical or self._build_spherical_pattern()
        except Exception as e:
            messagebox.showerror("Build error", str(e))
            return
        path = filedialog.asksaveasfilename(title="Export 3D HTML", defaultextension=".html", filetypes=[("HTML", "*.html"), ("All files", "*.*")])
        if not path:
            return
        _th_n, _ph_n, _a, _b, db_min, db_max, gamma = self._parse_recon_numeric()
        export_plotly_html(pat, path, db_min=db_min, db_max=db_max, gamma=gamma)
    def _copy_text(self, txt: str):
        self.clipboard_clear()
        self.clipboard_append(txt)

    def _copy_tree_row(self, tree: ttk.Treeview):
        sel = tree.selection()
        if not sel:
            return
        vals = tree.item(sel[0], "values")
        self._copy_text("\t".join([str(v) for v in vals]))

    def _copy_tree_all(self, tree: ttk.Treeview):
        rows = []
        for item in tree.get_children():
            vals = tree.item(item, "values")
            rows.append("\t".join([str(v) for v in vals]))
        self._copy_text("\n".join(rows))

    def _show_plot_context_menu(self, event, interactor: AdvancedPlotInteractor):
        m = None
        try:
            m = tk.Menu(self, tearoff=0)
            m.add_command(label="Copy Cursor", command=lambda: self._copy_cursor_value(event, interactor))
            m.add_command(label="Copy A/B/Delta", command=self._copy_ab_delta)
            m.add_separator()
            m.add_command(label="Clear markers", command=self.clear_markers)
            m.add_command(label="Reset view", command=self._draw_current_cut)
            m.add_separator()
            m.add_command(label="Export current cut PNG", command=self._export_current_cut_png)
            m.add_command(label="Export current cut CSV", command=self.export_current_cut_csv)
            m.add_command(label="Export current cut PAT", command=self.export_current_cut_pat)
            m.add_command(label="Export PAT (Project)", command=self.app.export_all_pat)
            m.add_command(label="Export PRN (Project)", command=self.app.export_all_prn)
            m.add_command(label="Send to Study slot", command=self._send_current_to_study)

            widget = interactor.canvas.get_tk_widget()
            ge = getattr(event, "guiEvent", None)
            gx = getattr(ge, "x", None)
            gy = getattr(ge, "y", None)
            if gx is not None and gy is not None:
                x = widget.winfo_rootx() + int(gx)
                y = widget.winfo_rooty() + int(gy)
            else:
                x = widget.winfo_pointerx()
                y = widget.winfo_pointery()
            m.tk_popup(x, y)
        finally:
            try:
                if m is not None:
                    m.grab_release()
            except Exception:
                pass

    def _copy_cursor_value(self, event, interactor: AdvancedPlotInteractor):
        kind = self.current_kind
        ang = interactor._event_ang_deg(event, kind)
        if ang is None:
            return
        pt = interactor._nearest_sample(ang, kind)
        if pt is None:
            return
        a, v = pt
        db = 20.0 * math.log10(max(float(v), 1e-12))
        txt = f"ang={float(a):.4f} deg | lin={float(v):.6f} | dB={float(db):.3f}"
        self._copy_text(txt)
        self._set_status(f"Cursor copied: {txt}")

    def _copy_ab_delta(self):
        markers = self.interactor.markers() if self.interactor else []
        if len(markers) < 2:
            return
        A, B = markers[0], markers[1]
        if self.current_kind == "H":
            dang = float(ang_dist_deg(A.ang_deg, B.ang_deg))
        else:
            dang = abs(float(B.ang_deg) - float(A.ang_deg))
        dmag = float(B.mag_db - A.mag_db)
        txt = (
            f"A: {A.name} ang={A.ang_deg:.4f} deg mag={A.mag_db:.3f} dB\n"
            f"B: {B.name} ang={B.ang_deg:.4f} deg mag={B.mag_db:.3f} dB\n"
            f"Delta: dang={dang:.4f} deg dmag={dmag:.3f} dB"
        )
        self._copy_text(txt)

    def _export_current_cut_png(self):
        if self.current_angles is None or self.current_values is None:
            return
        path = filedialog.asksaveasfilename(title="Export current cut PNG", defaultextension=".png", filetypes=[("PNG", "*.png"), ("All files", "*.*")])
        if path:
            self.plot_panel.figure.savefig(path, dpi=300)

    def _send_current_to_study(self):
        if self.current_angles is None or self.current_values is None:
            return
        if self.current_kind == "H":
            self.app._set_study_slot_data("H1", self.current_angles.copy(), self.current_values.copy(), source="Advanced View")
        else:
            self.app._set_study_slot_data("V1", self.current_angles.copy(), self.current_values.copy(), source="Advanced View")

    def _on_marker_table_context(self, event):
        row = self.marker_table.tree.identify_row(event.y)
        if row:
            self.marker_table.tree.selection_set(row)
        m = tk.Menu(self, tearoff=0)
        m.add_command(label="Copy row", command=lambda: self._copy_tree_row(self.marker_table.tree))
        m.add_command(label="Copy table", command=lambda: self._copy_tree_all(self.marker_table.tree))
        m.add_command(label="Export CSV", command=self.export_markers_csv)
        m.add_separator()
        m.add_command(label="Delete marker", command=self.delete_selected_marker)
        m.add_command(label="Rename marker", command=self.rename_selected_marker)
        m.tk_popup(event.x_root, event.y_root)
        m.grab_release()

    def _on_delta_table_context(self, event):
        row = self.delta_tree.identify_row(event.y)
        if row:
            self.delta_tree.selection_set(row)
        m = tk.Menu(self, tearoff=0)
        m.add_command(label="Copy row", command=lambda: self._copy_tree_row(self.delta_tree))
        m.add_command(label="Copy table", command=lambda: self._copy_tree_all(self.delta_tree))
        m.tk_popup(event.x_root, event.y_root)
        m.grab_release()

    def _on_derived_table_context(self, event):
        row = self.derived_table.tree.identify_row(event.y)
        if row:
            self.derived_table.tree.selection_set(row)
        m = tk.Menu(self, tearoff=0)
        m.add_command(label="Copy row", command=lambda: self._copy_tree_row(self.derived_table.tree))
        m.add_command(label="Copy table", command=lambda: self._copy_tree_all(self.derived_table.tree))
        m.tk_popup(event.x_root, event.y_root)
        m.grab_release()
