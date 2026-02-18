from __future__ import annotations

import copy
import hashlib
import logging
import os
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox

try:
    import customtkinter as ctk
except Exception as e:  # pragma: no cover
    raise ImportError("customtkinter is required for AedtLiveTab.") from e

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from .bridge import AppBridge
from .cut_tools import mag_db_from_linear, shift_cut_no_interp, transform_cut
from .export import PatternExport
from .farfield import CutRequest, FarFieldExtractor, GridRequest
from .session import AedtConnectionConfig, AedtHfssSession
from .worker import TkWorker


DEFAULT_EXPR = "dB(GainTotal)"
DEFAULT_SPHERE = "3D_Sphere"
DESIGN_PLACEHOLDER = "-- selecione um design --"
HFSS_QUANTITIES = [
    "GainTotal",
    "GainTheta",
    "GainPhi",
    "GainLHCP",
    "GainRHCP",
    "GainX",
    "GainY",
    "GainZ",
]
HFSS_FUNCTIONS = ["<none>", "dB", "normalize", "dB10normalize", "dB20normalize", "mag", "abs", "real", "imag"]


def _build_aedt_logger() -> logging.Logger:
    log_dir = Path(os.path.expanduser("~")) / ".eftx_converter" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "aedt_live.log"
    logger = logging.getLogger("eftx.aedt_live")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = RotatingFileHandler(log_file, maxBytes=2_000_000, backupCount=5, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


class AedtLiveTab(ctk.CTkFrame):
    """UI tab for live AEDT/HFSS interaction (post-processing focused)."""

    def __init__(self, master, app=None, output_dir: Optional[str] = None):
        super().__init__(master)
        self.app = app
        self.bridge = AppBridge(app) if app is not None else None
        self.logger = _build_aedt_logger()

        self.output_dir = output_dir or str(Path.cwd() / "aedt_exports")
        self.exporter = PatternExport(Path(self.output_dir))
        self.cfg = AedtConnectionConfig()
        self.session = AedtHfssSession(self.cfg)
        self.extractor = FarFieldExtractor(self.session)
        self.worker = TkWorker(self)

        self._cut_cache: Dict[tuple, dict] = {}
        self._grid_cache: Dict[tuple, dict] = {}
        self._pending_payload: Dict[str, object] = {"cuts_2d": {}, "spherical_3d": None, "meta": {}}
        self._designs_by_project: Dict[str, List[str]] = {}
        self._known_setups: List[str] = []
        self._known_solved_sweeps: List[str] = []
        self._busy_pull = False
        self._context_ready = False
        self._context_project = ""
        self._context_design = ""
        self._original_cuts: Dict[str, Dict[str, object]] = {}
        self._plot_lines: Dict[str, object] = {}
        self._plot_markers: List[Dict[str, object]] = []
        self._drag_marker: Optional[Dict[str, object]] = None
        self._last_motion_ts: float = 0.0
        self._current_cursor: Dict[str, float] = {}
        self._preview_cut: Optional[Dict[str, object]] = None
        self._plot_mode_active = "BOTH"
        self._inspector_meta: Dict[str, object] = {}
        self._resample_collapsed = False
        self._plot_series: Dict[str, Tuple[np.ndarray, np.ndarray]] = {
            "HRP": (np.asarray([], dtype=float), np.asarray([], dtype=float)),
            "VRP": (np.asarray([], dtype=float), np.asarray([], dtype=float)),
        }
        self._plot_axes: Dict[str, object] = {}

        self._build_ui()
        self._sync_expr_from_builder()
        self._update_send_state()
        self._refresh_plot()
        self._refresh_metadata_box()
        self._log("AEDT Live tab ready. Connect to start.")

    # ---------------- UI ----------------

    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(3, weight=1)

        self.var_version = tk.StringVar(value=self.cfg.version)
        self.var_conn_mode = tk.StringVar(value="Attach")
        self.var_non_graphical = tk.BooleanVar(value=False)
        self.var_project = tk.StringVar(value="")
        self.var_design = tk.StringVar(value="")
        self.var_setup = tk.StringVar(value="")
        self.var_sphere = tk.StringVar(value=DEFAULT_SPHERE)
        self.var_quantity = tk.StringVar(value="GainTotal")
        self.var_function = tk.StringVar(value="dB")
        self.var_expr = tk.StringVar(value=DEFAULT_EXPR)
        self.var_freq = tk.StringVar(value="")
        self.var_vrp_phi = tk.StringVar(value="0deg")
        self.var_lin_mode = tk.StringVar(value="field")
        self.var_theta_pts = tk.StringVar(value="181")
        self.var_phi_pts = tk.StringVar(value="361")
        self.var_db_floor = tk.StringVar(value="-40")
        self.var_gamma = tk.StringVar(value="1.0")
        self.var_view_mode = tk.StringVar(value="Both")
        self.var_resample_cut = tk.StringVar(value="VRP")
        self.var_resample_mode = tk.StringVar(value="interp")
        self.var_resample_step = tk.StringVar(value="1deg")
        self.var_resample_step_custom = tk.StringVar(value="1.0")
        self.var_resample_rot = tk.StringVar(value="0.0")
        self.var_resample_align_peak = tk.BooleanVar(value=False)
        self.var_resample_status = tk.StringVar(value="No preview yet.")

        conn = ctk.CTkFrame(self)
        conn.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 6))
        for col in range(14):
            conn.grid_columnconfigure(col, weight=0)
        conn.grid_columnconfigure(13, weight=1)

        ctk.CTkLabel(conn, text="AEDT ver").grid(row=0, column=0, padx=(10, 4), pady=10, sticky="w")
        ctk.CTkEntry(conn, textvariable=self.var_version, width=90).grid(row=0, column=1, padx=4, pady=10, sticky="w")
        ctk.CTkLabel(conn, text="Mode").grid(row=0, column=2, padx=(10, 4), pady=10, sticky="w")
        ctk.CTkOptionMenu(conn, variable=self.var_conn_mode, values=["Attach", "New Session"], width=120).grid(
            row=0, column=3, padx=4, pady=10, sticky="w"
        )
        ctk.CTkCheckBox(conn, text="Non-graphical", variable=self.var_non_graphical, onvalue=True, offvalue=False).grid(
            row=0, column=4, padx=(10, 6), pady=10, sticky="w"
        )
        ctk.CTkLabel(conn, text="Project (.aedt)").grid(row=0, column=5, padx=(10, 4), pady=10, sticky="w")
        ctk.CTkEntry(conn, textvariable=self.var_project, width=300).grid(row=0, column=6, padx=4, pady=10, sticky="w")
        ctk.CTkButton(conn, text="Browse", width=80, command=self._browse_project).grid(row=0, column=7, padx=(4, 10), pady=10)
        ctk.CTkLabel(conn, text="Design").grid(row=0, column=8, padx=(10, 4), pady=10, sticky="w")
        ctk.CTkEntry(conn, textvariable=self.var_design, width=170).grid(row=0, column=9, padx=4, pady=10, sticky="w")
        self.btn_connect = ctk.CTkButton(conn, text="Connect", width=110, command=self._connect)
        self.btn_connect.grid(row=0, column=10, padx=(10, 4), pady=10)
        self.btn_disc = ctk.CTkButton(conn, text="Disconnect", width=110, command=self._disconnect)
        self.btn_disc.grid(row=0, column=11, padx=(4, 4), pady=10)
        ctk.CTkButton(conn, text="Refresh", width=90, command=self._refresh_metadata).grid(row=0, column=12, padx=(6, 4), pady=10)
        ctk.CTkButton(conn, text="Create Sphere", width=120, command=self._create_sphere).grid(
            row=0, column=13, padx=(4, 10), pady=10, sticky="w"
        )

        meta = ctk.CTkFrame(self)
        meta.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 6))
        for col in range(8):
            meta.grid_columnconfigure(col, weight=0)
        meta.grid_columnconfigure(7, weight=1)

        ctk.CTkLabel(meta, text="Project list").grid(row=0, column=0, padx=(10, 4), pady=8, sticky="w")
        self.project_menu = ctk.CTkOptionMenu(meta, variable=self.var_project, values=[""], command=self._on_select_project, width=240)
        self.project_menu.grid(row=0, column=1, padx=4, pady=8, sticky="w")
        ctk.CTkLabel(meta, text="Design list").grid(row=0, column=2, padx=(10, 4), pady=8, sticky="w")
        self.design_menu = ctk.CTkOptionMenu(meta, variable=self.var_design, values=[""], command=self._on_select_design, width=220)
        self.design_menu.grid(row=0, column=3, padx=4, pady=8, sticky="w")
        ctk.CTkLabel(meta, text="Setup sweep").grid(row=0, column=4, padx=(10, 4), pady=8, sticky="w")
        self.setup_menu = ctk.CTkOptionMenu(meta, variable=self.var_setup, values=[""], command=self._on_select_setup, width=220)
        self.setup_menu.grid(row=0, column=5, padx=4, pady=8, sticky="w")
        ctk.CTkLabel(meta, text="Sphere").grid(row=0, column=6, padx=(10, 4), pady=8, sticky="w")
        self.sphere_menu = ctk.CTkOptionMenu(meta, variable=self.var_sphere, values=[DEFAULT_SPHERE], command=self._on_select_sphere, width=170)
        self.sphere_menu.grid(row=0, column=7, padx=4, pady=8, sticky="w")

        meta_btn = ctk.CTkFrame(meta, fg_color="transparent")
        meta_btn.grid(row=1, column=0, columnspan=8, sticky="ew", padx=8, pady=(0, 6))
        ctk.CTkButton(meta_btn, text="Open Project...", width=130, command=self._browse_and_open_project).pack(side="left", padx=4)
        ctk.CTkButton(meta_btn, text="Apply Selection", width=130, command=self._apply_selected_context).pack(side="left", padx=4)

        ctl = ctk.CTkFrame(self)
        ctl.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 6))
        for col in range(22):
            ctl.grid_columnconfigure(col, weight=0)
        ctl.grid_columnconfigure(21, weight=1)

        ctk.CTkLabel(ctl, text="Quantity").grid(row=0, column=0, padx=(10, 4), pady=8, sticky="w")
        ctk.CTkOptionMenu(
            ctl,
            variable=self.var_quantity,
            values=HFSS_QUANTITIES,
            width=140,
            command=lambda _v: self._sync_expr_from_builder(),
        ).grid(row=0, column=1, padx=4, pady=8, sticky="w")
        ctk.CTkLabel(ctl, text="Function").grid(row=0, column=2, padx=(10, 4), pady=8, sticky="w")
        ctk.CTkOptionMenu(
            ctl,
            variable=self.var_function,
            values=HFSS_FUNCTIONS,
            width=150,
            command=lambda _v: self._sync_expr_from_builder(),
        ).grid(row=0, column=3, padx=4, pady=8, sticky="w")
        ctk.CTkLabel(ctl, text="Expr").grid(row=0, column=4, padx=(10, 4), pady=8, sticky="w")
        ctk.CTkEntry(ctl, textvariable=self.var_expr, width=220).grid(row=0, column=5, padx=4, pady=8, sticky="w")
        ctk.CTkLabel(ctl, text="Freq").grid(row=0, column=6, padx=(10, 4), pady=8, sticky="w")
        ctk.CTkEntry(ctl, textvariable=self.var_freq, width=100).grid(row=0, column=7, padx=4, pady=8, sticky="w")
        ctk.CTkLabel(ctl, text="VRP Phi").grid(row=0, column=8, padx=(10, 4), pady=8, sticky="w")
        ctk.CTkOptionMenu(ctl, variable=self.var_vrp_phi, values=["0deg", "90deg"], width=90).grid(row=0, column=9, padx=4, pady=8, sticky="w")
        ctk.CTkLabel(ctl, text="Lin").grid(row=0, column=10, padx=(10, 4), pady=8, sticky="w")
        ctk.CTkOptionMenu(ctl, variable=self.var_lin_mode, values=["field", "power"], width=90).grid(row=0, column=11, padx=4, pady=8, sticky="w")
        ctk.CTkLabel(ctl, text="Theta pts").grid(row=0, column=12, padx=(10, 4), pady=8, sticky="w")
        ctk.CTkEntry(ctl, textvariable=self.var_theta_pts, width=70).grid(row=0, column=13, padx=4, pady=8, sticky="w")
        ctk.CTkLabel(ctl, text="Phi pts").grid(row=0, column=14, padx=(10, 4), pady=8, sticky="w")
        ctk.CTkEntry(ctl, textvariable=self.var_phi_pts, width=70).grid(row=0, column=15, padx=4, pady=8, sticky="w")
        ctk.CTkLabel(ctl, text="dB floor").grid(row=0, column=16, padx=(10, 4), pady=8, sticky="w")
        ctk.CTkEntry(ctl, textvariable=self.var_db_floor, width=70).grid(row=0, column=17, padx=4, pady=8, sticky="w")
        ctk.CTkLabel(ctl, text="gamma").grid(row=0, column=18, padx=(10, 4), pady=8, sticky="w")
        ctk.CTkEntry(ctl, textvariable=self.var_gamma, width=70).grid(row=0, column=19, padx=4, pady=8, sticky="w")
        ctk.CTkLabel(ctl, text="View").grid(row=0, column=20, padx=(10, 4), pady=8, sticky="w")
        ctk.CTkOptionMenu(
            ctl,
            variable=self.var_view_mode,
            values=["HRP", "VRP", "Both"],
            width=90,
            command=lambda _v: self._refresh_plot(),
        ).grid(row=0, column=21, padx=4, pady=8, sticky="w")

        btn_row = ctk.CTkFrame(ctl, fg_color="transparent")
        btn_row.grid(row=1, column=0, columnspan=22, sticky="ew", padx=8, pady=(0, 8))
        self.btn_vrp = ctk.CTkButton(btn_row, text="Pull VRP", width=110, command=self._pull_vrp)
        self.btn_hrp = ctk.CTkButton(btn_row, text="Pull HRP", width=110, command=self._pull_hrp)
        self.btn_3d = ctk.CTkButton(btn_row, text="Pull 3D", width=110, command=self._pull_3d)
        self.btn_resample = ctk.CTkButton(btn_row, text="Review/Resample", width=140, command=self._focus_resample_section)
        self.btn_send_project = ctk.CTkButton(btn_row, text="Send to Project", width=130, command=self._send_to_project)
        self.btn_send_library = ctk.CTkButton(btn_row, text="Send to Library", width=130, command=self._send_to_library)
        self.btn_clear_pending = ctk.CTkButton(btn_row, text="Clear Pending", width=120, command=self._clear_pending, fg_color="#666666")
        self.btn_vrp.pack(side="left", padx=4)
        self.btn_hrp.pack(side="left", padx=4)
        self.btn_3d.pack(side="left", padx=4)
        self.btn_resample.pack(side="left", padx=(10, 4))
        self.btn_send_project.pack(side="left", padx=(10, 4))
        self.btn_send_library.pack(side="left", padx=4)
        self.btn_clear_pending.pack(side="left", padx=4)

        cut_row = ctk.CTkFrame(ctl, fg_color="transparent")
        cut_row.grid(row=2, column=0, columnspan=22, sticky="ew", padx=8, pady=(0, 8))
        ctk.CTkLabel(cut_row, text="Insert validated cut into Project:").pack(side="left", padx=(4, 6))
        self.btn_send_vrp_project = ctk.CTkButton(
            cut_row,
            text="Insert VRP",
            width=110,
            command=lambda: self._send_cut_to_project("VRP"),
        )
        self.btn_send_hrp_project = ctk.CTkButton(
            cut_row,
            text="Insert HRP",
            width=110,
            command=lambda: self._send_cut_to_project("HRP"),
        )
        self.btn_send_vrp_project.pack(side="left", padx=4)
        self.btn_send_hrp_project.pack(side="left", padx=4)

        content = ctk.CTkFrame(self)
        content.grid(row=3, column=0, sticky="nsew", padx=10, pady=(0, 10))
        content.grid_columnconfigure(0, weight=4)
        content.grid_columnconfigure(1, weight=1)
        content.grid_rowconfigure(0, weight=1)

        plot_frame = ctk.CTkFrame(content)
        plot_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 6), pady=0)
        plot_frame.grid_rowconfigure(0, weight=1)
        plot_frame.grid_columnconfigure(0, weight=1)

        self.fig = Figure(figsize=(8.6, 6.8), dpi=100)
        self.ax_vrp = self.fig.add_subplot(211)
        self.ax_hrp = self.fig.add_subplot(212, projection="polar")
        self.fig.subplots_adjust(hspace=0.34, top=0.95, bottom=0.06, left=0.08, right=0.98)
        self.line_vrp = self.ax_vrp.plot([], [], color="#ff7f0e", linewidth=1.7, label="VRP")[0]
        self.line_vrp_preview = self.ax_vrp.plot([], [], color="#9B9B9B", linewidth=1.2, linestyle="--", label="Preview")[0]
        self.line_hrp = self.ax_hrp.plot([], [], color="#1f77b4", linewidth=1.7, label="HRP")[0]
        self.line_hrp_preview = self.ax_hrp.plot([], [], color="#9B9B9B", linewidth=1.2, linestyle="--", label="Preview")[0]
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        self._plot_lines = {"HRP": self.line_hrp, "VRP": self.line_vrp}
        self._plot_axes = {"VRP": self.ax_vrp, "HRP": self.ax_hrp}
        self.canvas.mpl_connect("button_press_event", self._on_plot_press)
        self.canvas.mpl_connect("button_release_event", self._on_plot_release)
        self.canvas.mpl_connect("motion_notify_event", self._on_plot_motion)

        right = ctk.CTkFrame(content, width=360)
        right.grid(row=0, column=1, sticky="nsew", padx=(6, 0), pady=0)
        right.grid_propagate(False)
        right.grid_rowconfigure(5, weight=1)
        right.grid_columnconfigure(0, weight=1)

        header_meta = ctk.CTkFrame(right, fg_color="transparent")
        header_meta.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 2))
        ctk.CTkLabel(header_meta, text="Inspector / Metadata", anchor="w").pack(side="left", padx=(2, 4))
        ctk.CTkButton(header_meta, text="Copy", width=68, command=self._copy_metadata).pack(side="right", padx=2)

        self.meta_box = ctk.CTkTextbox(right, height=190, wrap="word")
        self.meta_box.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 6))

        header_log = ctk.CTkFrame(right, fg_color="transparent")
        header_log.grid(row=2, column=0, sticky="ew", padx=8, pady=(0, 2))
        ctk.CTkLabel(header_log, text="Logs", anchor="w").pack(side="left", padx=(2, 4))
        ctk.CTkButton(header_log, text="Copy", width=60, command=self._copy_logs).pack(side="right", padx=2)
        ctk.CTkButton(header_log, text="Export", width=64, command=self._export_logs).pack(side="right", padx=2)
        ctk.CTkButton(header_log, text="Clear", width=60, command=self._clear_logs).pack(side="right", padx=2)

        self.log_box = ctk.CTkTextbox(right, height=190, wrap="word")
        self.log_box.grid(row=3, column=0, sticky="ew", padx=8, pady=(0, 6))

        self.resample_header = ctk.CTkButton(
            right,
            text="Review/Resample (Inline)",
            command=self._toggle_resample_section,
            anchor="w",
            fg_color="#3a3a3a",
        )
        self.resample_header.grid(row=4, column=0, sticky="ew", padx=8, pady=(0, 4))

        self.resample_body = ctk.CTkFrame(right)
        self.resample_body.grid(row=5, column=0, sticky="nsew", padx=8, pady=(0, 8))
        for c in range(2):
            self.resample_body.grid_columnconfigure(c, weight=1 if c == 1 else 0)

        ctk.CTkLabel(self.resample_body, text="Cut").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        ctk.CTkOptionMenu(
            self.resample_body,
            variable=self.var_resample_cut,
            values=["VRP", "HRP"],
            command=lambda _v: self._resample_preview_only(),
        ).grid(row=0, column=1, sticky="ew", padx=6, pady=4)
        ctk.CTkLabel(self.resample_body, text="Mode").grid(row=1, column=0, sticky="w", padx=6, pady=4)
        ctk.CTkOptionMenu(self.resample_body, variable=self.var_resample_mode, values=["interp", "snap"], width=120).grid(
            row=1, column=1, sticky="ew", padx=6, pady=4
        )
        ctk.CTkLabel(self.resample_body, text="Step").grid(row=2, column=0, sticky="w", padx=6, pady=4)
        ctk.CTkOptionMenu(self.resample_body, variable=self.var_resample_step, values=["1deg", "0.5deg", "0.1deg", "custom"]).grid(
            row=2, column=1, sticky="ew", padx=6, pady=4
        )
        ctk.CTkLabel(self.resample_body, text="Custom step").grid(row=3, column=0, sticky="w", padx=6, pady=4)
        ctk.CTkEntry(self.resample_body, textvariable=self.var_resample_step_custom).grid(row=3, column=1, sticky="ew", padx=6, pady=4)
        ctk.CTkLabel(self.resample_body, text="Rotation (deg)").grid(row=4, column=0, sticky="w", padx=6, pady=4)
        ctk.CTkEntry(self.resample_body, textvariable=self.var_resample_rot).grid(row=4, column=1, sticky="ew", padx=6, pady=4)
        ctk.CTkCheckBox(
            self.resample_body,
            text="Align peak to 0 deg",
            variable=self.var_resample_align_peak,
        ).grid(row=5, column=0, columnspan=2, sticky="w", padx=6, pady=4)
        ctk.CTkLabel(self.resample_body, textvariable=self.var_resample_status, justify="left", text_color="#B7C3D0").grid(
            row=6, column=0, columnspan=2, sticky="ew", padx=6, pady=(2, 6)
        )
        btn_res = ctk.CTkFrame(self.resample_body, fg_color="transparent")
        btn_res.grid(row=7, column=0, columnspan=2, sticky="ew", padx=4, pady=(0, 2))
        ctk.CTkButton(btn_res, text="Preview", width=92, command=self._resample_preview_only).pack(side="left", padx=2, pady=2)
        ctk.CTkButton(btn_res, text="Apply Pending", width=120, command=self._apply_resample_pending).pack(side="left", padx=2, pady=2)
        ctk.CTkButton(btn_res, text="Commit Project", width=120, command=lambda: self._apply_resample_pending(send_project=True)).pack(
            side="left", padx=2, pady=2
        )
        ctk.CTkButton(btn_res, text="Commit Library", width=118, command=lambda: self._apply_resample_pending(send_library=True)).pack(
            side="left", padx=2, pady=2
        )
        ctk.CTkButton(btn_res, text="Reset Cut", width=90, fg_color="#555555", command=self._reset_resample_cut).pack(
            side="left", padx=2, pady=2
        )

        self._set_textbox_readonly(self.meta_box, "")
        self._set_textbox_readonly(self.log_box, "")

    def _build_expr(self, quantity: str, function: str) -> str:
        q = str(quantity or "").strip() or "GainTotal"
        f = str(function or "").strip()
        if not f or f == "<none>":
            return q
        return f"{f}({q})"

    def _inner_expr(self, expr: str) -> str:
        txt = str(expr or "").strip()
        i0 = txt.find("(")
        i1 = txt.rfind(")")
        if i0 >= 0 and i1 > i0:
            return txt[i0 + 1 : i1].strip() or "GainTotal"
        return txt or "GainTotal"

    def _fallback_expr_if_flat(self, expr: str) -> Optional[str]:
        low = str(expr or "").strip().lower().replace(" ", "")
        if low.startswith("db10normalize(") or low.startswith("db20normalize(") or low.startswith("normalize("):
            inner = self._inner_expr(expr)
            return f"dB({inner})"
        return None

    def _sync_expr_from_builder(self):
        self.var_expr.set(self._build_expr(self.var_quantity.get(), self.var_function.get()))

    def _set_textbox_readonly(self, textbox: "ctk.CTkTextbox", text: str):
        try:
            textbox.configure(state="normal")
            textbox.delete("1.0", "end")
            textbox.insert("end", str(text or ""))
            textbox.configure(state="disabled")
        except Exception:
            pass

    def _append_log_text(self, line: str):
        try:
            self.log_box.configure(state="normal")
            self.log_box.insert("end", line + "\n")
            self.log_box.see("end")
            self.log_box.configure(state="disabled")
        except Exception:
            pass

    def _copy_metadata(self):
        text = ""
        try:
            text = self.meta_box.get("1.0", "end").strip()
        except Exception:
            text = ""
        if not text:
            return
        self.clipboard_clear()
        self.clipboard_append(text)
        self._log("[OK] Metadata copied to clipboard.")

    def _copy_logs(self):
        text = ""
        try:
            text = self.log_box.get("1.0", "end").strip()
        except Exception:
            text = ""
        if not text:
            return
        self.clipboard_clear()
        self.clipboard_append(text)
        self._log("[OK] Logs copied to clipboard.")

    def _export_logs(self):
        path = filedialog.asksaveasfilename(
            title="Export AEDT Live logs",
            defaultextension=".txt",
            filetypes=[("Text", "*.txt"), ("All files", "*.*")],
        )
        if not path:
            return
        text = ""
        try:
            text = self.log_box.get("1.0", "end")
        except Exception:
            text = ""
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        self._log(f"[OK] Logs exported: {path}")

    def _clear_logs(self):
        self._set_textbox_readonly(self.log_box, "")
        self.logger.info("AEDT Live UI logs cleared by user.")

    def _toggle_resample_section(self):
        self._resample_collapsed = not bool(self._resample_collapsed)
        if self._resample_collapsed:
            self.resample_body.grid_remove()
            self.resample_header.configure(text="Review/Resample (Inline) [+]")
        else:
            self.resample_body.grid()
            self.resample_header.configure(text="Review/Resample (Inline)")

    def _focus_resample_section(self):
        if self._resample_collapsed:
            self._toggle_resample_section()
        self.resample_body.focus_set()
        self._resample_preview_only()

    def _cut_from_pending(self, mode: str) -> Tuple[np.ndarray, np.ndarray]:
        cuts = self._pending_payload.get("cuts_2d", {})
        if not isinstance(cuts, dict):
            return np.asarray([], dtype=float), np.asarray([], dtype=float)
        cut = cuts.get(str(mode).upper())
        if not isinstance(cut, dict):
            return np.asarray([], dtype=float), np.asarray([], dtype=float)
        return self._cut_payload_arrays(cut)

    def _plot_mode(self) -> str:
        self._plot_mode_active = "BOTH"
        return "BOTH"

    def _mode_from_event(self, event) -> Optional[str]:
        ax = getattr(event, "inaxes", None)
        if ax is self.ax_vrp:
            return "VRP"
        if ax is self.ax_hrp:
            return "HRP"
        return None

    def _angle_deg_from_event(self, event, mode: str) -> Optional[float]:
        if getattr(event, "xdata", None) is None:
            return None
        if str(mode).upper() == "HRP":
            th = float(np.rad2deg(float(event.xdata)))
            th = ((th + 180.0) % 360.0) - 180.0
            return th
        return float(event.xdata)

    def _refresh_plot(self):
        hrp_a, hrp_lin = self._cut_from_pending("HRP")
        vrp_a, vrp_lin = self._cut_from_pending("VRP")
        self._plot_series["HRP"] = (np.asarray(hrp_a, dtype=float), np.asarray(hrp_lin, dtype=float))
        self._plot_series["VRP"] = (np.asarray(vrp_a, dtype=float), np.asarray(vrp_lin, dtype=float))

        if vrp_a.size and vrp_lin.size:
            self.line_vrp.set_data(vrp_a, vrp_lin)
        else:
            self.line_vrp.set_data([], [])

        if hrp_a.size and hrp_lin.size:
            theta = np.deg2rad((np.asarray(hrp_a, dtype=float) + 360.0) % 360.0)
            self.line_hrp.set_data(theta, hrp_lin)
        else:
            self.line_hrp.set_data([], [])

        preview = self._preview_cut if isinstance(self._preview_cut, dict) else None
        if preview:
            p_mode = str(preview.get("mode", "")).upper()
            p_ang = np.asarray(preview.get("angles", []), dtype=float)
            p_val = np.asarray(preview.get("values", []), dtype=float)
            if p_mode == "VRP" and p_ang.size and p_val.size:
                self.line_vrp_preview.set_data(p_ang, p_val)
            else:
                self.line_vrp_preview.set_data([], [])
            if p_mode == "HRP" and p_ang.size and p_val.size:
                p_th = np.deg2rad((p_ang + 360.0) % 360.0)
                self.line_hrp_preview.set_data(p_th, p_val)
            else:
                self.line_hrp_preview.set_data([], [])
        else:
            self.line_vrp_preview.set_data([], [])
            self.line_hrp_preview.set_data([], [])

        # VRP planar
        self.ax_vrp.set_title("VRP (Elevacao - Planar)")
        self.ax_vrp.set_xlabel("Elevacao (deg)")
        self.ax_vrp.set_ylabel("Magnitude (linear)")
        self.ax_vrp.grid(True, linestyle="--", alpha=0.30)
        if vrp_a.size and vrp_lin.size:
            self.ax_vrp.relim()
            self.ax_vrp.autoscale_view()
            self.ax_vrp.set_xlim(-90.0, 90.0)
            self.ax_vrp.legend(loc="best")
            if hasattr(self, "_empty_vrp_text") and self._empty_vrp_text is not None:
                try:
                    self._empty_vrp_text.remove()
                except Exception:
                    pass
                self._empty_vrp_text = None
        else:
            if (not hasattr(self, "_empty_vrp_text")) or self._empty_vrp_text is None:
                self._empty_vrp_text = self.ax_vrp.text(
                    0.5,
                    0.5,
                    "Sem dados VRP.\nUse Pull VRP.",
                    ha="center",
                    va="center",
                    transform=self.ax_vrp.transAxes,
                    color="#B8B8B8",
                )

        # HRP polar
        self.ax_hrp.set_title("HRP (Azimute - Polar)")
        self.ax_hrp.set_theta_zero_location("N")
        self.ax_hrp.set_theta_direction(-1)
        self.ax_hrp.grid(True, linestyle="--", alpha=0.30)
        if hrp_a.size and hrp_lin.size:
            rmax = float(np.max(np.asarray(hrp_lin, dtype=float)))
            if rmax <= 0:
                rmax = 1.0
            self.ax_hrp.set_ylim(0.0, rmax * 1.05)
            self.ax_hrp.legend(loc="upper right")
            if hasattr(self, "_empty_hrp_text") and self._empty_hrp_text is not None:
                try:
                    self._empty_hrp_text.remove()
                except Exception:
                    pass
                self._empty_hrp_text = None
        else:
            if (not hasattr(self, "_empty_hrp_text")) or self._empty_hrp_text is None:
                self._empty_hrp_text = self.ax_hrp.text(
                    0.5,
                    0.5,
                    "Sem dados HRP.\nUse Pull HRP.",
                    ha="center",
                    va="center",
                    transform=self.ax_hrp.transAxes,
                    color="#B8B8B8",
                )
        self.canvas.draw_idle()

    def _menu_popup_coords(self, event):
        widget = self.canvas.get_tk_widget()
        ge = getattr(event, "guiEvent", None)
        gx = getattr(ge, "x", None)
        gy = getattr(ge, "y", None)
        if gx is not None and gy is not None:
            try:
                return widget.winfo_rootx() + int(gx), widget.winfo_rooty() + int(gy)
            except Exception:
                pass
        return widget.winfo_pointerx(), widget.winfo_pointery()

    def _nearest_on_curve(self, mode: str, angle_deg: float) -> Optional[Tuple[float, float]]:
        a, v = self._plot_series.get(str(mode).upper(), (np.asarray([], dtype=float), np.asarray([], dtype=float)))
        if a.size == 0 or v.size == 0:
            return None
        if str(mode).upper() == "HRP":
            d = np.abs(((a - float(angle_deg) + 180.0) % 360.0) - 180.0)
            idx = int(np.argmin(d))
            return float(a[idx]), float(v[idx])
        idx = int(np.argmin(np.abs(a - float(angle_deg))))
        return float(a[idx]), float(v[idx])

    def _hit_existing_marker(self, event, mode: str, tol_px: float = 10.0) -> Optional[Dict[str, object]]:
        ex = float(getattr(event, "x", np.nan))
        ey = float(getattr(event, "y", np.nan))
        if not np.isfinite(ex) or not np.isfinite(ey):
            return None
        ax = self._plot_axes.get(mode)
        if ax is None:
            return None
        for mk in self._plot_markers:
            if str(mk.get("mode", "")) != mode:
                continue
            x = float(mk.get("x", np.nan))
            y = float(mk.get("y", np.nan))
            if not np.isfinite(x) or not np.isfinite(y):
                continue
            if mode == "HRP":
                px, py = ax.transData.transform((np.deg2rad((x + 360.0) % 360.0), y))
            else:
                px, py = ax.transData.transform((x, y))
            if float(np.hypot(px - ex, py - ey)) <= float(tol_px):
                return mk
        return None

    def _update_marker_artist(self, mk: Dict[str, object], x: float, y: float):
        mode = str(mk.get("mode", "VRP")).upper()
        mk["x"] = float(x)
        mk["y"] = float(y)
        artist = mk.get("artist")
        text = mk.get("text")
        label = str(mk.get("label", "M"))
        if artist is not None:
            if mode == "HRP":
                artist.set_data([np.deg2rad((x + 360.0) % 360.0)], [y])
            else:
                artist.set_data([x], [y])
        if text is not None:
            if mode == "HRP":
                text.set_position((np.deg2rad((x + 360.0) % 360.0), y))
            else:
                text.set_position((x, y))
            text.set_text(f"{label} {mode}: {x:.2f}, {y:.2f}")

    def _copy_cursor(self):
        mode = str(self._current_cursor.get("mode", "")).upper()
        ang = self._current_cursor.get("angle_deg")
        if mode not in ("HRP", "VRP") or ang is None:
            return
        near = self._nearest_on_curve(mode, float(ang))
        if near is None:
            return
        txt = f"{mode}: {near[0]:.4f}, {near[1]:.6f}"
        self.clipboard_clear()
        self.clipboard_append(txt)
        self._log(f"[OK] Cursor copied: {txt}")

    def _add_marker_at_cursor(self):
        mode = str(self._current_cursor.get("mode", "")).upper()
        ang = self._current_cursor.get("angle_deg")
        if mode not in ("HRP", "VRP") or ang is None:
            return
        near = self._nearest_on_curve(mode, float(ang))
        if near is None:
            return
        color_cycle = ["#E74C3C", "#2ECC71", "#3498DB", "#F39C12", "#9B59B6"]
        idx = len(self._plot_markers)
        color = color_cycle[idx % len(color_cycle)]
        label = f"M{idx + 1}"
        ax = self._plot_axes.get(mode)
        if ax is None:
            return
        if mode == "HRP":
            th = np.deg2rad((near[0] + 360.0) % 360.0)
            artist = ax.plot([th], [near[1]], marker="o", markersize=6, color=color, linestyle="None", zorder=8)[0]
            text = ax.text(th, near[1], f"{label} {mode}: {near[0]:.2f}, {near[1]:.2f}", color=color, fontsize=8, ha="left", va="bottom")
        else:
            artist = ax.plot([near[0]], [near[1]], marker="o", markersize=6, color=color, linestyle="None", zorder=8)[0]
            text = ax.text(near[0], near[1], f"{label} {mode}: {near[0]:.2f}, {near[1]:.2f}", color=color, fontsize=8, ha="left", va="bottom")
        mk = {"label": label, "mode": mode, "x": float(near[0]), "y": float(near[1]), "artist": artist, "text": text}
        self._plot_markers.append(mk)
        self.canvas.draw_idle()
        self._log(f"[OK] Added marker {label} ({mode}).")

    def _clear_plot_markers(self):
        for mk in list(self._plot_markers):
            for key in ("artist", "text"):
                obj = mk.get(key)
                if obj is not None:
                    try:
                        obj.remove()
                    except Exception:
                        pass
        self._plot_markers = []
        self._drag_marker = None
        self.canvas.draw_idle()
        self._log("[OK] Plot markers cleared.")

    def _reset_plot_view(self):
        self._refresh_plot()
        self._log("[OK] Plot view reset.")

    def _export_plot_png(self):
        path = filedialog.asksaveasfilename(
            title="Export plot PNG",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("All files", "*.*")],
        )
        if not path:
            return
        self.fig.savefig(path, dpi=300)
        self._log(f"[OK] Plot image exported: {path}")

    def _export_plot_csv(self):
        path = filedialog.asksaveasfilename(
            title="Export plot CSV",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return
        rows = ["curve,angle_deg,value"]
        for key in ("VRP", "HRP"):
            x, y = self._plot_series.get(key, (np.asarray([], dtype=float), np.asarray([], dtype=float)))
            n = min(x.size, y.size)
            for xi, yi in zip(x[:n], y[:n]):
                rows.append(f"{key},{float(xi):.6f},{float(yi):.6f}")
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(rows) + "\n")
        self._log(f"[OK] Plot CSV exported: {path}")

    def _show_plot_context_menu(self, event, mode: str):
        m = None
        try:
            m = tk.Menu(self, tearoff=0)
            m.add_command(label="Copy cursor (ang, mag)", command=self._copy_cursor)
            m.add_command(label="Add marker at cursor", command=self._add_marker_at_cursor)
            m.add_command(label="Clear markers", command=self._clear_plot_markers)
            m.add_separator()
            m.add_command(label="Export PNG", command=self._export_plot_png)
            m.add_command(label="Export CSV", command=self._export_plot_csv)
            m.add_separator()
            m.add_command(label="Reset view", command=self._reset_plot_view)
            x, y = self._menu_popup_coords(event)
            m.tk_popup(x, y)
        finally:
            try:
                if m is not None:
                    m.grab_release()
            except Exception:
                pass

    def _on_plot_press(self, event):
        mode = self._mode_from_event(event)
        if mode is None:
            return
        try:
            self.canvas.get_tk_widget().focus_set()
        except Exception:
            pass
        ang = self._angle_deg_from_event(event, mode)
        if ang is not None:
            self._current_cursor["mode"] = mode
            self._current_cursor["angle_deg"] = float(ang)
        if event.ydata is not None:
            self._current_cursor["y"] = float(event.ydata)
        if event.button in (2, 3):
            self._show_plot_context_menu(event, mode)
            return
        if event.button != 1:
            return
        mk = self._hit_existing_marker(event, mode)
        if mk is not None:
            self._drag_marker = mk

    def _on_plot_release(self, event):
        if event.button == 1:
            self._drag_marker = None

    def _on_plot_motion(self, event):
        if self._drag_marker is None:
            return
        mode = str(self._drag_marker.get("mode", "")).upper()
        ax = self._plot_axes.get(mode)
        if ax is None:
            return
        if event.inaxes != ax:
            return
        now = time.perf_counter()
        if (now - self._last_motion_ts) < (1.0 / 60.0):
            return
        self._last_motion_ts = now
        ang = self._angle_deg_from_event(event, mode)
        if ang is None:
            return
        near = self._nearest_on_curve(mode, float(ang))
        if near is None:
            return
        self._update_marker_artist(self._drag_marker, near[0], near[1])
        self.canvas.draw_idle()

    def _resample_step_value(self) -> float:
        preset = str(self.var_resample_step.get() or "1deg").strip().lower()
        if preset == "custom":
            step = float(self._parse_float(self.var_resample_step_custom.get(), 1.0))
        else:
            step = float(self._parse_float(preset.replace("deg", ""), 1.0))
        if step <= 0:
            step = 1.0
        return step

    def _resample_target_points(self, mode: str) -> int:
        span = 360.0 if str(mode).upper() == "HRP" else 180.0
        step = self._resample_step_value()
        pts = int(round(span / step)) + 1
        return max(3, pts)

    def _snap_resampled_values(self, src_a: np.ndarray, src_v: np.ndarray, out_a: np.ndarray, mode: str) -> np.ndarray:
        if src_a.size == 0 or src_v.size == 0 or out_a.size == 0:
            return np.asarray(out_a * 0.0, dtype=float)
        result = np.zeros_like(out_a, dtype=float)
        if str(mode).upper() == "HRP":
            for i, q in enumerate(out_a):
                d = np.abs(((src_a - q + 180.0) % 360.0) - 180.0)
                result[i] = float(src_v[int(np.argmin(d))])
            return result
        for i, q in enumerate(out_a):
            d = np.abs(src_a - q)
            result[i] = float(src_v[int(np.argmin(d))])
        return result

    def _resample_preview_only(self) -> Optional[Dict[str, object]]:
        mode = str(self.var_resample_cut.get() or "").strip().upper()
        if mode not in ("HRP", "VRP"):
            self.var_resample_status.set("Invalid cut mode.")
            return None
        ang, val = self._cut_from_pending(mode)
        if ang.size == 0 or val.size == 0:
            self.var_resample_status.set(f"No pending {mode} cut.")
            return None
        pts = self._resample_target_points(mode)
        rot = float(self._parse_float(self.var_resample_rot.get(), 0.0))
        align = bool(self.var_resample_align_peak.get())
        out_a, out_v, meta = transform_cut(ang, val, mode=mode, rotation_deg=rot, align_peak_zero=align, target_points=pts)
        if str(self.var_resample_mode.get() or "interp").lower() == "snap":
            out_v = self._snap_resampled_values(ang, val, out_a, mode)
            meta["resample_mode_snap"] = 1.0
        self._preview_cut = {"mode": mode, "angles": out_a, "values": out_v, "meta": meta}
        self.var_resample_status.set(
            f"{mode}: shift={float(meta.get('shift_deg', 0.0)):.2f} deg | "
            f"peak={float(meta.get('peak_before_deg', 0.0)):.2f}->{float(meta.get('peak_after_deg', 0.0)):.2f} deg | pts={int(pts)}"
        )
        self._refresh_plot()
        return self._preview_cut

    def _apply_resample_pending(self, send_project: bool = False, send_library: bool = False):
        preview = self._resample_preview_only()
        if not isinstance(preview, dict):
            return
        mode = str(preview.get("mode", "")).upper()
        if mode not in ("HRP", "VRP"):
            return
        cuts = self._pending_payload.get("cuts_2d", {})
        if not isinstance(cuts, dict):
            cuts = {}
        cut = cuts.get(mode)
        if not isinstance(cut, dict):
            cut = {"name": f"AEDT_{mode}"}
        tmeta = dict(preview.get("meta", {}))
        tmeta["user_rotation_deg"] = float(self._parse_float(self.var_resample_rot.get(), 0.0))
        tmeta["user_align_peak_zero"] = 1.0 if bool(self.var_resample_align_peak.get()) else 0.0
        tmeta["user_resample_step_deg"] = float(self._resample_step_value())
        tmeta["user_resample_mode"] = 1.0 if str(self.var_resample_mode.get()).lower() == "snap" else 0.0
        self._set_cut_payload_arrays(
            cut,
            np.asarray(preview.get("angles", []), dtype=float),
            np.asarray(preview.get("values", []), dtype=float),
            transform_meta=tmeta,
        )
        cuts[mode] = cut
        self._pending_payload["cuts_2d"] = cuts
        self._pending_payload["meta"] = self._base_meta()
        self._update_send_state()
        self._preview_cut = None
        self._refresh_plot()
        self._refresh_metadata_box({"timing": "resample applied"})
        self._log(f"[OK] {mode} cut updated by inline resample.")
        if send_project:
            self._send_to_project()
        if send_library:
            self._send_to_library()

    def _reset_resample_cut(self):
        mode = str(self.var_resample_cut.get() or "").strip().upper()
        if mode not in ("HRP", "VRP"):
            return
        original = self._original_cuts.get(mode)
        if not isinstance(original, dict):
            self.var_resample_status.set(f"No original baseline stored for {mode}.")
            return
        cuts = self._pending_payload.get("cuts_2d", {})
        if not isinstance(cuts, dict):
            cuts = {}
        cuts[mode] = copy.deepcopy(original)
        self._pending_payload["cuts_2d"] = cuts
        self._pending_payload["meta"] = self._base_meta()
        self._preview_cut = None
        self._update_send_state()
        self._refresh_plot()
        self._refresh_metadata_box({"timing": "resample reset"})
        self.var_resample_status.set(f"{mode} restored to extracted baseline.")
        self._log(f"[OK] {mode} reset to original extracted cut.")

    def _refresh_metadata_box(self, extra: Optional[Dict[str, object]] = None):
        meta = self._base_meta()
        cuts = self._pending_payload.get("cuts_2d", {})
        sph = self._pending_payload.get("spherical_3d")
        lines = [
            f"Project: {meta.get('project', '-') or '-'}",
            f"Design: {meta.get('design', '-') or '-'}",
            f"Setup: {meta.get('setup', '-') or '-'}",
            f"Sphere: {meta.get('sphere', '-') or '-'}",
            f"Expr: {meta.get('expr', '-') or '-'}",
            f"Quantity: {meta.get('quantity', '-') or '-'}",
            f"Function: {meta.get('function', '-') or '-'}",
            f"Freq: {meta.get('freq', '-') or '-'}",
            "",
        ]
        if isinstance(cuts, dict):
            for mode in ("HRP", "VRP"):
                cut = cuts.get(mode)
                if not isinstance(cut, dict):
                    continue
                a, v = self._cut_payload_arrays(cut)
                if a.size == 0:
                    continue
                cmeta = cut.get("meta", {}) if isinstance(cut.get("meta"), dict) else {}
                hid_src = f"{mode}|{len(a)}|{float(np.min(a)):.6f}|{float(np.max(a)):.6f}|{float(np.min(v)):.6f}|{float(np.max(v)):.6f}"
                hid = hashlib.sha1(hid_src.encode("utf-8", errors="ignore")).hexdigest()[:10]
                lines.append(
                    f"{mode}: npts={int(a.size)} | range=[{float(np.min(a)):.2f},{float(np.max(a)):.2f}] deg | "
                    f"lin=[{float(np.min(v)):.4f},{float(np.max(v)):.4f}] avg={float(np.mean(v)):.4f} | id={hid}"
                )
                primary = str(cmeta.get("primary_sweep", "-"))
                fx_theta = str(cmeta.get("Theta", "-"))
                fx_phi = str(cmeta.get("Phi", "-"))
                lines.append(f"  sweep={primary} | fixed Theta={fx_theta} Phi={fx_phi}")
                ereq = str(cmeta.get("expr_requested", "-"))
                eeff = str(cmeta.get("expr_effective", cmeta.get("expression", "-")))
                lines.append(f"  expr requested={ereq} | effective={eeff}")
        if isinstance(sph, dict):
            t = np.asarray(sph.get("theta_deg", []), dtype=float)
            p = np.asarray(sph.get("phi_deg", []), dtype=float)
            if t.size and p.size:
                lines.append(f"3D: shape={int(t.size)}x{int(p.size)}")
                lines.append(f"3D paths: npz={sph.get('npz_path', '-')}")
        if isinstance(extra, dict) and extra:
            lines.append("")
            lines.append("Timing / Last operation:")
            for k, v in extra.items():
                lines.append(f"- {k}: {v}")
        text = "\n".join(lines)
        self._inspector_meta = {"text": text}
        self._set_textbox_readonly(self.meta_box, text)

    # -------------- Basic helpers --------------

    def _on_select_project(self, value: str):
        self.var_project.set(value)
        self._update_design_menu_for_project(value)
        self._set_context_ready(False)

    def _on_select_design(self, value: str):
        self.var_design.set(value)
        self._set_context_ready(False)

    def _on_select_setup(self, value: str):
        self.var_setup.set(value)

    def _on_select_sphere(self, value: str):
        self.var_sphere.set(value)

    def _browse_project(self):
        path = filedialog.askopenfilename(
            title="Select AEDT project",
            filetypes=[("AEDT Project", "*.aedt"), ("All files", "*")],
        )
        if path:
            self.var_project.set(path)
            self._update_design_menu_for_project(path)
            self._set_context_ready(False)

    def _browse_and_open_project(self):
        path = filedialog.askopenfilename(
            title="Open AEDT project",
            filetypes=[("AEDT Project", "*.aedt"), ("All files", "*")],
        )
        if not path:
            return
        self.var_project.set(path)
        self._update_design_menu_for_project(path)
        self._set_context_ready(False)
        if not self.session.is_connected:
            self._connect()
            return
        self._apply_selected_context()

    def _project_token(self, value: str) -> str:
        aliases = self._project_aliases(value)
        return aliases[0] if aliases else ""

    def _project_aliases(self, value: str) -> List[str]:
        txt = str(value or "").strip().strip('"').strip("'")
        if txt.startswith("*"):
            txt = txt[1:].strip()
        if not txt:
            return []
        out: List[str] = [txt]
        p = Path(txt)
        if p.suffix.lower() == ".aedt":
            out.append(p.stem)
        else:
            try:
                out.append(p.stem)
            except Exception:
                pass
            if txt.lower().endswith(".aedt"):
                out.append(txt[:-5].strip())
        return self._unique([str(x).strip().lower() for x in out if str(x).strip()])

    def _project_token_for_api(self, value: str) -> str:
        txt = str(value or "").strip().strip('"').strip("'")
        if txt.startswith("*"):
            txt = txt[1:].strip()
        if not txt:
            return ""
        p = Path(txt)
        if p.suffix.lower() == ".aedt":
            return p.stem
        return txt

    def _project_for_connect(self) -> Optional[str]:
        txt = str(self.var_project.get() or "").strip().strip('"').strip("'")
        if txt.startswith("*"):
            txt = txt[1:].strip()
        if not txt:
            return None
        return txt

    def _project_lock_file(self, project: Optional[str]) -> Optional[Path]:
        txt = str(project or "").strip().strip('"').strip("'")
        if not txt:
            return None
        p = Path(txt)
        if p.suffix.lower() != ".aedt":
            return None
        lock = Path(str(p) + ".lock")
        if lock.exists():
            return lock
        return None

    def _ask_remove_project_lock(self, project: Optional[str]) -> Tuple[bool, bool]:
        """Return (allow_connect, remove_lock_override)."""
        lock = self._project_lock_file(project)
        if lock is None:
            return True, False
        ask = messagebox.askyesno(
            "AEDT Live",
            f"O projeto selecionado esta em modo lock:\n{lock}\n\n"
            "Deseja remover o lock e abrir o projeto agora?",
        )
        if not ask:
            self._log("[WARN] Conexao cancelada pelo usuario (projeto em lock).")
            return False, False
        self._log(f"[INFO] Lock detectado: {lock}. Abrindo com remocao de lock.")
        return True, True

    def _parse_int(self, text: str, default: int) -> int:
        try:
            return int(str(text).strip())
        except Exception:
            return int(default)

    def _parse_float(self, text: str, default: float) -> float:
        try:
            return float(str(text).strip().replace(",", "."))
        except Exception:
            return float(default)

    def _unique(self, values: List[str]) -> List[str]:
        out: List[str] = []
        seen = set()
        for raw in values:
            v = str(raw).strip()
            if not v or v in seen:
                continue
            seen.add(v)
            out.append(v)
        return out

    def _set_menu_values(self, menu: "ctk.CTkOptionMenu", var: tk.StringVar, values: List[str], fallback: str = ""):
        clean = self._unique(values)
        if not clean:
            clean = [fallback or ""]
        menu.configure(values=clean)
        cur = var.get().strip()
        if cur not in clean:
            var.set(clean[0])

    def _update_design_menu_for_project(self, project_value: str, fallback_designs: Optional[List[str]] = None):
        values: List[str] = []
        for token in self._project_aliases(project_value):
            values.extend(list(self._designs_by_project.get(token, [])))
        if not values and fallback_designs:
            values = list(fallback_designs)
        values = self._unique(values)
        if len(values) > 1:
            opts = [DESIGN_PLACEHOLDER] + values
            self.design_menu.configure(values=opts)
            cur = self.var_design.get().strip()
            if cur not in values:
                self.var_design.set(DESIGN_PLACEHOLDER)
        else:
            self._set_menu_values(self.design_menu, self.var_design, values, self.var_design.get().strip() or "")

    def _selected_design(self) -> Optional[str]:
        design = self.var_design.get().strip()
        if not design or design == DESIGN_PLACEHOLDER:
            return None
        return design

    def _extract_design_names(self, raw_items) -> List[str]:
        out: List[str] = []
        for raw in list(raw_items or []):
            text = str(raw).strip().strip('"')
            if not text:
                continue
            if ";" in text:
                text = text.split(";", 1)[0].strip()
            if ":" in text and text.lower().startswith("design"):
                text = text.split(":", 1)[0].strip()
            if text:
                out.append(text)
        return self._unique(out)

    def _is_db_expression(self) -> bool:
        expr = self.var_expr.get().strip().lower().replace(" ", "")
        return ("db(" in expr) or expr.startswith("db")

    def _to_mag_arrays(self, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        arr = np.asarray(values, dtype=float)
        arr = np.nan_to_num(arr, nan=-300.0, posinf=0.0, neginf=-300.0)
        if self._is_db_expression():
            arr_db = arr - float(np.max(arr))
            expo = 20.0 if self.var_lin_mode.get().strip().lower() == "field" else 10.0
            arr_lin = np.power(10.0, arr_db / expo)
            arr_lin = np.clip(arr_lin, 0.0, None)
            return arr_lin, arr_db
        arr_lin = np.clip(arr, 0.0, None)
        vmax = float(np.max(arr_lin)) if arr_lin.size else 0.0
        if vmax > 0:
            arr_lin = arr_lin / vmax
        arr_db = 20.0 * np.log10(np.maximum(arr_lin, 1e-12))
        return arr_lin, arr_db

    def _base_meta(self) -> Dict[str, object]:
        return {
            "project": self.var_project.get().strip(),
            "design": self._selected_design() or "",
            "setup": self.var_setup.get().strip(),
            "sphere": self.var_sphere.get().strip(),
            "freq": self.var_freq.get().strip(),
            "expr": self.var_expr.get().strip(),
            "quantity": self.var_quantity.get().strip(),
            "function": self.var_function.get().strip(),
            "lin_mode": self.var_lin_mode.get().strip(),
            "view_mode": self.var_view_mode.get().strip(),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

    def _payload_has_data(self) -> bool:
        cuts = self._pending_payload.get("cuts_2d", {})
        if isinstance(cuts, dict) and cuts:
            return True
        if isinstance(self._pending_payload.get("spherical_3d"), dict):
            return True
        return False

    def _payload_snapshot(self) -> Dict:
        return copy.deepcopy(self._pending_payload)

    def _clear_pending(self):
        self._pending_payload = {"cuts_2d": {}, "spherical_3d": None, "meta": {}}
        self._original_cuts = {}
        self._preview_cut = None
        self._clear_plot_markers()
        self._update_send_state()
        self._refresh_plot()
        self._refresh_metadata_box({"timing": "pending cleared"})
        self._log("Pending payload cleared.")

    def _available_cut_modes(self) -> List[str]:
        cuts = self._pending_payload.get("cuts_2d", {})
        if not isinstance(cuts, dict):
            return []
        out: List[str] = []
        for mode in ("VRP", "HRP"):
            cut = cuts.get(mode)
            if not isinstance(cut, dict):
                continue
            ang = np.asarray(cut.get("angles_deg", []), dtype=float)
            if ang.size > 0:
                out.append(mode)
        return out

    def _cut_payload_arrays(self, cut: Dict[str, object]) -> Tuple[np.ndarray, np.ndarray]:
        ang = np.asarray(cut.get("angles_deg", []), dtype=float)
        lin = np.asarray(cut.get("mag_lin", []), dtype=float)
        if lin.size == 0:
            db = np.asarray(cut.get("mag_db", []), dtype=float)
            if db.size:
                lin = np.power(10.0, db / 20.0)
        if lin.size == 0:
            lin = np.asarray(cut.get("values", []), dtype=float)
        n = min(ang.size, lin.size)
        if n <= 0:
            return np.asarray([], dtype=float), np.asarray([], dtype=float)
        ang = np.asarray(ang[:n], dtype=float)
        lin = np.clip(np.asarray(lin[:n], dtype=float), 0.0, None)
        return ang, lin

    def _set_cut_payload_arrays(self, cut: Dict[str, object], angles_deg: np.ndarray, values_lin: np.ndarray, transform_meta: Optional[Dict[str, float]] = None):
        a = np.asarray(angles_deg, dtype=float).reshape(-1)
        v = np.clip(np.asarray(values_lin, dtype=float).reshape(-1), 0.0, None)
        n = min(a.size, v.size)
        if n <= 0:
            return
        a = a[:n]
        v = v[:n]
        cut["angles_deg"] = a.tolist()
        cut["mag_lin"] = v.tolist()
        cut["mag_db"] = mag_db_from_linear(v).tolist()
        meta = dict(cut.get("meta", {}) if isinstance(cut.get("meta"), dict) else {})
        if isinstance(transform_meta, dict):
            for k, vmeta in transform_meta.items():
                meta[str(k)] = float(vmeta)
        meta["resampled"] = 1
        meta["resampled_timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        cut["meta"] = meta

    def _default_target_points(self, mode: str) -> int:
        return 361 if str(mode).upper() == "HRP" else 181

    def _update_send_state(self):
        self._update_pull_state()
        state = "normal" if self._payload_has_data() else "disabled"
        self.btn_send_project.configure(state=state)
        self.btn_send_library.configure(state=state)
        vrp_state = "normal" if self._has_cut_mode("VRP") else "disabled"
        hrp_state = "normal" if self._has_cut_mode("HRP") else "disabled"
        try:
            self.btn_send_vrp_project.configure(state=vrp_state)
        except Exception:
            pass
        try:
            self.btn_send_hrp_project.configure(state=hrp_state)
        except Exception:
            pass
        review_state = "normal" if bool(self._available_cut_modes()) else "disabled"
        try:
            self.btn_resample.configure(state=review_state)
        except Exception:
            pass

    def _open_resample_modal(self, focus_mode: Optional[str] = None):
        mode = str(focus_mode or self.var_resample_cut.get() or "VRP").strip().upper()
        if mode in ("HRP", "VRP"):
            self.var_resample_cut.set(mode)
        self._focus_resample_section()

    def _update_pull_state(self):
        enabled = bool(self.session.is_connected and self._context_ready and (not self._busy_pull))
        state = "normal" if enabled else "disabled"
        for btn in (self.btn_vrp, self.btn_hrp, self.btn_3d):
            try:
                btn.configure(state=state)
            except Exception:
                pass

    def _set_busy_pull(self, busy: bool):
        self._busy_pull = bool(busy)
        self._update_pull_state()

    def _set_context_ready(self, ready: bool):
        self._context_ready = bool(ready)
        self._update_pull_state()

    def _same_project_context(self, project_a: str, project_b: str) -> bool:
        a = set(self._project_aliases(project_a))
        b = set(self._project_aliases(project_b))
        if not a or not b:
            return False
        return bool(a.intersection(b))

    def _setup_name_from_sweep(self, setup_or_sweep: str) -> str:
        txt = str(setup_or_sweep or "").strip().strip('"')
        if not txt:
            return ""
        if ":" in txt:
            return txt.split(":", 1)[0].strip()
        return txt

    def _setup_matches(self, setup_or_sweep: str, other: str) -> bool:
        a = self._setup_name_from_sweep(setup_or_sweep).lower()
        b = self._setup_name_from_sweep(other).lower()
        return bool(a and b and a == b)

    def _resolve_effective_setup(self) -> str:
        setup = self.var_setup.get().strip()
        runtime_solved = self._runtime_solved_sweeps()
        if runtime_solved:
            self._known_solved_sweeps = self._unique(runtime_solved)
        if setup:
            for sw in self._known_solved_sweeps:
                if self._setup_matches(sw, setup):
                    return sw
            for sw in runtime_solved:
                if self._setup_matches(sw, setup):
                    return sw
            return setup
        if self._known_solved_sweeps:
            setup = self._known_solved_sweeps[0]
        elif self._known_setups:
            setup = self._known_setups[0]
        else:
            setup = ""
        if setup:
            self.var_setup.set(setup)
        return setup

    def _runtime_solved_sweeps(self) -> List[str]:
        if not self.session.is_connected:
            return []
        try:
            solved = self._list_attr_values(self.session.hfss, ("existing_analysis_sweeps",))
            return self._unique(solved)
        except Exception:
            return list(self._known_solved_sweeps)

    def _setup_is_solved_from_objects(self, setup_or_sweep: str) -> Optional[bool]:
        if not self.session.is_connected:
            return None
        setup_name = self._setup_name_from_sweep(setup_or_sweep)
        if not setup_name:
            return None
        hfss = self.session.hfss
        try:
            setups = list(getattr(hfss, "setups", []) or [])
        except Exception:
            setups = []
        if not setups:
            return None
        for s in setups:
            name = str(getattr(s, "name", "") or "").strip()
            if not self._setup_matches(name, setup_name):
                continue
            # Prefer explicit setup solved state when available.
            try:
                solved = getattr(s, "is_solved")
                if callable(solved):
                    solved = solved()
                if isinstance(solved, bool):
                    return solved
            except Exception:
                pass
            # Fallback to sweep-level solved state.
            try:
                sweeps = list(getattr(s, "sweeps", []) or [])
            except Exception:
                sweeps = []
            sweep_states: List[bool] = []
            for sw in sweeps:
                try:
                    ok = getattr(sw, "is_solved")
                    if callable(ok):
                        ok = ok()
                    if isinstance(ok, bool):
                        sweep_states.append(ok)
                except Exception:
                    continue
            if sweep_states:
                return bool(any(sweep_states))
            return None
        return None

    def _setup_is_solved(self, setup_or_sweep: str, solved_sweeps: Optional[List[str]] = None) -> bool:
        solved_obj = self._setup_is_solved_from_objects(setup_or_sweep)
        if isinstance(solved_obj, bool):
            return solved_obj
        sweeps = self._unique(list(solved_sweeps or []))
        if not sweeps:
            sweeps = self._runtime_solved_sweeps()
        if not setup_or_sweep:
            return bool(sweeps)
        return any(self._setup_matches(sw, setup_or_sweep) for sw in sweeps)

    def _analyze_selected_setup(self, setup_or_sweep: str) -> str:
        hfss = self.session.hfss
        setup_name = self._setup_name_from_sweep(setup_or_sweep)
        setup_sweep = str(setup_or_sweep or "").strip()
        attempts = (
            ("analyze_setup", (setup_name,), {}),
            ("analyze_setup", (setup_sweep,), {}),
            ("analyze", (setup_name,), {}),
            ("analyze", (setup_sweep,), {}),
            ("analyze_nominal", (), {}),
            ("analyze", (), {}),
        )
        last_error = None
        for method, args, kwargs in attempts:
            fn = getattr(hfss, method, None)
            if not callable(fn):
                continue
            try:
                out = fn(*args, **kwargs)
                if isinstance(out, bool) and (not out):
                    continue
                return method
            except Exception as e:
                last_error = e
                continue
        if last_error is not None:
            raise RuntimeError(f"Failed to run analysis for setup '{setup_name}': {last_error}") from last_error
        raise RuntimeError(f"No supported analysis method available for setup '{setup_name}'.")

    def _angle_deg_value(self, value) -> float:
        txt = str(value or "").strip().replace("°", "").replace("deg", "").replace("DEG", "")
        return self._parse_float(txt, float("nan"))

    def _find_infinite_sphere_setup(self, sphere_name: str):
        if not self.session.is_connected:
            return None
        hfss = self.session.hfss
        target = str(sphere_name or "").strip()
        if not target:
            return None
        field_setups = getattr(hfss, "field_setups", None)
        items = []
        if isinstance(field_setups, dict):
            for k, v in field_setups.items():
                if hasattr(v, "name"):
                    items.append(v)
                else:
                    items.append((k, v))
        elif isinstance(field_setups, (list, tuple, set)):
            items.extend(list(field_setups))
        for item in items:
            obj = item
            if isinstance(item, tuple) and len(item) == 2:
                obj = item[1]
            name = str(getattr(obj, "name", "") or "")
            if name and name == target:
                return obj
            if isinstance(item, tuple) and str(item[0]) == target:
                return obj
        return None

    def _sphere_profile_status(self, sphere_name: str) -> Tuple[bool, bool, str]:
        """Return (exists, compliant, detail) for required sphere profile."""
        if not self.session.is_connected:
            return False, False, "not_connected"
        hfss = self.session.hfss
        target = str(sphere_name or "").strip() or DEFAULT_SPHERE

        exists = False
        fs = self._find_infinite_sphere_setup(target)
        if fs is not None:
            exists = True
        else:
            oradfield = getattr(hfss, "oradfield", None)
            if oradfield is not None and hasattr(oradfield, "GetSetupNames"):
                try:
                    names = [str(x) for x in list(oradfield.GetSetupNames("Infinite Sphere"))]
                    exists = target in names
                except Exception:
                    exists = False

        if not exists:
            return False, False, "missing"
        if fs is None:
            return True, False, "no_props_access"

        props = getattr(fs, "props", None)
        if not isinstance(props, dict):
            return True, False, "invalid_props"

        definition = str(props.get("CSDefinition", "") or "").strip()
        if definition != "Theta-Phi":
            return True, False, f"definition={definition or '-'}"

        th0 = self._angle_deg_value(props.get("ThetaStart", "nan"))
        th1 = self._angle_deg_value(props.get("ThetaStop", "nan"))
        ths = self._angle_deg_value(props.get("ThetaStep", "nan"))
        ph0 = self._angle_deg_value(props.get("PhiStart", "nan"))
        ph1 = self._angle_deg_value(props.get("PhiStop", "nan"))
        phs = self._angle_deg_value(props.get("PhiStep", "nan"))

        want = {
            "ThetaStart": -180.0,
            "ThetaStop": 180.0,
            "ThetaStep": 1.0,
            "PhiStart": 0.0,
            "PhiStop": 90.0,
            "PhiStep": 90.0,
        }
        got = {
            "ThetaStart": th0,
            "ThetaStop": th1,
            "ThetaStep": ths,
            "PhiStart": ph0,
            "PhiStop": ph1,
            "PhiStep": phs,
        }
        tol = 1e-6
        compliant = True
        for key, ref in want.items():
            gv = got.get(key, float("nan"))
            if (not np.isfinite(gv)) or abs(float(gv) - float(ref)) > tol:
                compliant = False
                break
        if compliant:
            return True, True, "ok"
        return (
            True,
            False,
            "theta=[{:.2f},{:.2f},{:.2f}] phi=[{:.2f},{:.2f},{:.2f}]".format(th0, th1, ths, ph0, ph1, phs),
        )

    def _ensure_required_infinite_sphere(self, sphere_name: str) -> str:
        """Create or update infinite sphere to required profile.

        Required profile:
          - Definition: Theta-Phi
          - Phi: 0..90 step 90  (points 0 and 90)
          - Theta: -180..180 step 1
        """
        hfss = self.session.hfss
        target = str(sphere_name or "").strip() or DEFAULT_SPHERE

        fs = self._find_infinite_sphere_setup(target)
        if fs is not None and isinstance(getattr(fs, "props", None), dict):
            try:
                fs.props["CSDefinition"] = "Theta-Phi"
                fs.props["ThetaStart"] = "-180deg"
                fs.props["ThetaStop"] = "180deg"
                fs.props["ThetaStep"] = "1deg"
                fs.props["PhiStart"] = "0deg"
                fs.props["PhiStop"] = "90deg"
                fs.props["PhiStep"] = "90deg"
                if hasattr(fs, "update") and callable(getattr(fs, "update")):
                    fs.update()
                return "update_existing_sphere"
            except Exception:
                pass

        ins = getattr(hfss, "insert_infinite_sphere", None)
        if callable(ins):
            out = ins(
                definition="Theta-Phi",
                phi_start=0,
                phi_stop=90,
                phi_step=90,
                theta_start=-180,
                theta_stop=180,
                theta_step=1,
                units="deg",
                name=target,
            )
            if isinstance(out, bool) and (not out):
                raise RuntimeError("insert_infinite_sphere returned False.")
            return "insert_infinite_sphere"

        # Last-resort compatibility path for very old APIs.
        for method in ("create_infinite_sphere", "insert_far_field_sphere_setup"):
            fn = getattr(hfss, method, None)
            if not callable(fn):
                continue
            try:
                fn(target)
                fs2 = self._find_infinite_sphere_setup(target)
                if fs2 is not None and isinstance(getattr(fs2, "props", None), dict):
                    fs2.props["CSDefinition"] = "Theta-Phi"
                    fs2.props["ThetaStart"] = "-180deg"
                    fs2.props["ThetaStop"] = "180deg"
                    fs2.props["ThetaStep"] = "1deg"
                    fs2.props["PhiStart"] = "0deg"
                    fs2.props["PhiStop"] = "90deg"
                    fs2.props["PhiStep"] = "90deg"
                    if hasattr(fs2, "update") and callable(getattr(fs2, "update")):
                        fs2.update()
                return method
            except Exception:
                continue
        raise RuntimeError("Could not create/configure Infinite Sphere with required profile.")

    def _prepare_pull_preconditions(self, setup: str, sphere_name: str, label: str) -> Optional[Dict[str, object]]:
        run_analysis = False
        ensure_sphere = False

        solved_sweeps = self._runtime_solved_sweeps()
        if not solved_sweeps and (not self._known_setups):
            self._log("[WARN] Nenhum setup HFSS foi encontrado no design selecionado.")
            return None

        if not self._setup_is_solved(setup, solved_sweeps):
            ask = messagebox.askyesno(
                "AEDT Live",
                f"O setup '{self._setup_name_from_sweep(setup)}' nao possui resultados resolvidos.\n"
                f"Deseja executar a simulacao agora?",
            )
            if not ask:
                self._log(f"[WARN] {label} cancelado. Setup sem solucao.")
                return None
            run_analysis = True
            self._log(f"[INFO] Running simulation asynchronously for setup '{self._setup_name_from_sweep(setup)}'.")

        exists, compliant, detail = self._sphere_profile_status(sphere_name)
        if (not exists) or (not compliant):
            reason = "nao encontrada" if not exists else f"fora do padrao ({detail})"
            ask = messagebox.askyesno(
                "AEDT Live",
                f"A esfera infinita '{sphere_name}' esta {reason}.\n\n"
                "Deseja criar/configurar agora no formato obrigatorio?\n"
                "- Phi: 0 e 90 graus\n"
                "- Theta: -180 a 180 graus (passo 1 grau)",
            )
            if not ask:
                self._log(f"[WARN] {label} cancelado. Esfera infinita nao configurada.")
                return None
            ensure_sphere = True
            self._log(f"[INFO] Preparing infinite sphere '{sphere_name}' with required profile.")

        return {
            "run_analysis": run_analysis,
            "ensure_sphere": ensure_sphere,
            "solved_sweeps": solved_sweeps,
        }

    def _log_timing(self, tag: str, t0: float, ok: bool, extra: str = ""):
        dt_ms = (time.perf_counter() - t0) * 1000.0
        status = "OK" if ok else "FAIL"
        msg = f"{tag} {status} dt={dt_ms:.2f}ms"
        if extra:
            msg += f" | {extra}"
        self.logger.info(msg)

    # -------------- Metadata discovery --------------

    def _list_attr_values(self, obj, attrs: Tuple[str, ...]) -> List[str]:
        out: List[str] = []
        for attr in attrs:
            if not hasattr(obj, attr):
                continue
            try:
                val = getattr(obj, attr)
                if callable(val):
                    val = val()
            except Exception:
                continue
            if isinstance(val, (list, tuple, set)):
                out.extend([str(v) for v in val])
            elif isinstance(val, dict):
                out.extend([str(k) for k in val.keys()])
            elif val:
                out.append(str(val))
        return self._unique(out)

    def _collect_metadata(self) -> Dict[str, List[str]]:
        hfss = self.session.hfss

        projects = self._list_attr_values(hfss, ("project_list", "project_names"))
        project_name = str(getattr(hfss, "project_name", "") or "").strip()
        project_file = str(getattr(hfss, "project_file", "") or "").strip()
        if project_file:
            projects = self._unique([project_file] + projects)
        elif project_name:
            projects = self._unique([project_name] + projects)
        if not projects:
            desk = getattr(hfss, "odesktop", None)
            if desk is not None and hasattr(desk, "GetProjectList"):
                try:
                    projects = self._unique([str(x) for x in list(desk.GetProjectList())])
                except Exception:
                    projects = []

        designs = self._list_attr_values(hfss, ("design_list", "design_names"))
        design_name = str(getattr(hfss, "design_name", "") or "").strip()
        if design_name:
            designs = self._unique([design_name] + designs)

        designs_by_project: Dict[str, List[str]] = {}
        desk = getattr(hfss, "odesktop", None)
        if desk is not None and hasattr(desk, "SetActiveProject"):
            active_name = ""
            try:
                active_obj = desk.GetActiveProject()
                if active_obj is not None and hasattr(active_obj, "GetName"):
                    active_name = str(active_obj.GetName())
            except Exception:
                active_name = ""

            for proj in projects:
                token = self._project_token(proj)
                api_token = self._project_token_for_api(proj)
                candidates = self._unique([api_token, str(proj).strip(), token])
                pobj = None
                for candidate in candidates:
                    if not candidate:
                        continue
                    try:
                        pobj = desk.SetActiveProject(candidate)
                        if pobj is not None:
                            break
                    except Exception:
                        continue
                if pobj is None:
                    continue
                raw_designs = []
                if hasattr(pobj, "GetTopDesignList"):
                    raw_designs = list(pobj.GetTopDesignList())
                elif hasattr(pobj, "GetDesigns"):
                    raw = list(pobj.GetDesigns())
                    for item in raw:
                        try:
                            dtype = str(item.GetDesignType())
                        except Exception:
                            dtype = ""
                        try:
                            dname = str(item.GetName())
                        except Exception:
                            dname = str(item)
                        if (not dtype) or ("hfss" in dtype.lower()):
                            raw_designs.append(dname)
                design_list = self._extract_design_names(raw_designs)
                if design_list:
                    for alias in self._project_aliases(proj):
                        designs_by_project[alias] = design_list

            if active_name:
                try:
                    desk.SetActiveProject(active_name)
                except Exception:
                    pass

        selected_aliases = self._project_aliases(self.var_project.get().strip())
        for alias in selected_aliases:
            if alias in designs_by_project:
                designs = self._unique(designs_by_project[alias] + designs)
                break

        solved_sweeps = self._list_attr_values(hfss, ("existing_analysis_sweeps",))
        defined_setups: List[str] = []
        try:
            defined_setups = self._unique([str(s.name) for s in list(getattr(hfss, "setups", [])) if hasattr(s, "name")])
        except Exception:
            defined_setups = []
        setups = self._unique(solved_sweeps + defined_setups)

        spheres: List[str] = []
        field_setups = getattr(hfss, "field_setups", None)
        if isinstance(field_setups, dict):
            spheres.extend([str(k) for k in field_setups.keys()])
        elif isinstance(field_setups, (list, tuple, set)):
            for item in field_setups:
                name = str(getattr(item, "name", item))
                if name:
                    spheres.append(name)
        oradfield = getattr(hfss, "oradfield", None)
        if oradfield is not None and hasattr(oradfield, "GetSetupNames"):
            try:
                names = list(oradfield.GetSetupNames("Infinite Sphere"))
                spheres.extend([str(x) for x in names])
            except Exception:
                pass
        spheres = self._unique([s for s in spheres if s])
        if not spheres:
            spheres = [DEFAULT_SPHERE]

        return {
            "projects": projects,
            "designs": designs,
            "setups": setups,
            "solved_sweeps": solved_sweeps,
            "spheres": spheres,
            "designs_by_project": designs_by_project,
        }

    def _refresh_metadata(self):
        if not self.session.is_connected:
            self._log("[WARN] Not connected.")
            return

        t0 = time.perf_counter()

        def task():
            return self._collect_metadata()

        def done(res):
            if not res.ok:
                self._log_timing("list", t0, False, res.error or "")
                self._log(f"[ERR] refresh: {res.error}")
                return
            data = res.value
            self._designs_by_project = dict(data.get("designs_by_project", {}))
            self._known_setups = self._unique(list(data.get("setups", [])))
            self._known_solved_sweeps = self._unique(list(data.get("solved_sweeps", [])))
            self._set_menu_values(self.project_menu, self.var_project, data.get("projects", []), "")
            self._update_design_menu_for_project(self.var_project.get().strip(), data.get("designs", []))
            self._set_menu_values(self.setup_menu, self.var_setup, data.get("setups", []), "")
            self._set_menu_values(self.sphere_menu, self.var_sphere, data.get("spheres", []), DEFAULT_SPHERE)

            design_choices: List[str] = []
            for alias in self._project_aliases(self.var_project.get().strip()):
                if alias in self._designs_by_project:
                    design_choices = list(self._designs_by_project.get(alias, []))
                    break
            if not design_choices:
                design_choices = list(data.get("designs", []))
            design_choices = self._unique(design_choices)
            selected = self._selected_design()
            same_project_ctx = self._same_project_context(self.var_project.get().strip(), self._context_project)
            if len(design_choices) > 1:
                if selected and selected == self._context_design and same_project_ctx and self._context_ready:
                    self._set_context_ready(True)
                else:
                    self._set_context_ready(False)
                    self._log("[INFO] Projeto com multiplos designs. Selecione um design e clique em Apply Selection.")
            elif len(design_choices) == 1:
                only_design = design_choices[0]
                if self.var_design.get().strip() != only_design:
                    self.var_design.set(only_design)
                self._context_project = self.var_project.get().strip()
                self._context_design = only_design
                self._set_context_ready(True)
            else:
                self._set_context_ready(False)
                self._log("[WARN] Nenhum design HFSS encontrado no projeto selecionado.")

            self._log_timing(
                "list",
                t0,
                True,
                f"projects={len(data.get('projects', []))} designs={len(data.get('designs', []))} setups={len(data.get('setups', []))} spheres={len(data.get('spheres', []))}",
            )
            self._log(
                f"[OK] Metadata refreshed: {len(data.get('projects', []))} project(s), "
                f"{len(data.get('designs', []))} design(s), {len(data.get('setups', []))} setup(s), "
                f"{len(data.get('spheres', []))} sphere(s)."
            )
            self._refresh_metadata_box(
                {
                    "pull_time_ms": f"{(time.perf_counter() - t0) * 1000.0:.2f}",
                    "metadata_projects": len(data.get("projects", [])),
                    "metadata_designs": len(data.get("designs", [])),
                }
            )

        self.worker.run(task, done)

    def _apply_selected_context(self):
        if not self.session.is_connected:
            self._log("[WARN] Not connected.")
            return
        project = self._project_for_connect()
        design = self._selected_design()
        allow_connect, remove_lock_override = self._ask_remove_project_lock(project)
        if not allow_connect:
            return
        if not project and not design:
            self._set_context_ready(False)
            self._log("[WARN] Selecione um projeto e um design validos antes de aplicar o contexto.")
            return
        t0 = time.perf_counter()

        def task():
            # Fase 1: com projeto sem design, abre/ativa o projeto para descobrir os designs.
            # Fase 2: com projeto+design, fixa o contexto completo de extração.
            self.session.connect(
                project=project,
                design=design,
                setup=None,
                force=True,
                remove_lock_override=remove_lock_override,
            )
            hfss = self.session.hfss
            active_project = str(getattr(hfss, "project_file", "") or getattr(hfss, "project_name", "") or project or "")
            active_design = str(getattr(hfss, "design_name", "") or design or "")
            return {
                "project": active_project,
                "design": active_design,
            }

        def done(res):
            if not res.ok:
                self._log_timing("set_context", t0, False, res.error or "")
                self._log(f"[ERR] apply context: {res.error}")
                return
            out = res.value or {}
            self._log_timing("set_context", t0, True, f"project={out.get('project','')} design={out.get('design','')}")
            self._context_project = str(out.get("project", "") or "")
            self._context_design = str(out.get("design", "") or "")
            if design:
                self._set_context_ready(True)
                self._log(
                    f"[OK] Active context: project='{out.get('project','')}' design='{out.get('design','')}'."
                )
            else:
                self._set_context_ready(False)
                self._log(f"[OK] Projeto ativo: '{out.get('project','')}'. Selecione o design e clique em Apply Selection.")
            self._refresh_metadata_box({"timing": f"set_context {(time.perf_counter() - t0) * 1000.0:.2f} ms"})
            self._refresh_metadata()

        self.worker.run(task, done)

    def _ensure_selected_context(self) -> None:
        if not self._context_ready:
            raise RuntimeError("Design nao confirmado. Selecione o design e clique em Apply Selection.")
        project = self._project_for_connect()
        design = self._selected_design()
        if not design:
            raise RuntimeError("Nenhum design valido foi selecionado.")
        self.session.connect(project=project, design=design, setup=None, force=False)

    # -------------- Connection actions --------------

    def _connect(self):
        t0 = time.perf_counter()
        mode = self.var_conn_mode.get().strip()
        version = self.var_version.get().strip() or self.cfg.version
        project = self._project_for_connect()
        design = self._selected_design()
        non_graphical = bool(self.var_non_graphical.get())
        allow_connect, remove_lock_override = self._ask_remove_project_lock(project)
        if not allow_connect:
            return

        def task():
            self.cfg.version = version
            self.cfg.new_desktop = mode.lower().startswith("new")
            self.cfg.non_graphical = non_graphical
            self.session.cfg = self.cfg
            self.session.connect(
                project=project,
                design=design,
                setup=None,
                remove_lock_override=remove_lock_override,
            )
            return True

        def done(res):
            if not res.ok:
                self._log_timing("connect", t0, False, res.error or "")
                self._log(f"[ERR] {res.error}")
                messagebox.showerror("AEDT", res.error)
                return
            self._set_context_ready(False)
            self._log_timing("connect", t0, True, f"mode={mode} version={version}")
            if project and not design:
                self._log("[OK] Connected to HFSS. Projeto aberto; selecione o design e clique em Apply Selection.")
            elif (not project) and (not design):
                self._log("[OK] Connected to HFSS. Selecione/abra projeto e design para habilitar importacao.")
            else:
                self._log("[OK] Connected to HFSS.")
            self._refresh_metadata()

        self._log(f"Connecting to AEDT/HFSS ({mode})...")
        self.worker.run(task, done)

    def _disconnect(self):
        t0 = time.perf_counter()
        try:
            self.session.disconnect()
            self._set_context_ready(False)
            self._log_timing("disconnect", t0, True, "")
            self._log("[OK] Disconnected.")
        except Exception as e:
            self._log_timing("disconnect", t0, False, str(e))
            self._log(f"[WARN] disconnect: {e}")

    def _create_sphere(self):
        if not self.session.is_connected:
            self._log("[WARN] Not connected.")
            return
        sphere_name = self.var_sphere.get().strip() or DEFAULT_SPHERE
        t0 = time.perf_counter()

        def task():
            return self._ensure_required_infinite_sphere(sphere_name)

        def done(res):
            if not res.ok:
                self._log_timing("create_sphere", t0, False, res.error or "")
                self._log(f"[ERR] {res.error}")
                return
            self._cut_cache = {}
            self._grid_cache = {}
            self._log_timing("create_sphere", t0, True, f"name={sphere_name} method={res.value}")
            self._log(
                f"[OK] Sphere created/verified: {sphere_name} (Phi 0/90; Theta -180..180 step 1)."
            )
            self._refresh_metadata()

        self.worker.run(task, done)

    # -------------- Pull actions --------------

    def _pull_vrp(self):
        phi_cut = (self.var_vrp_phi.get().strip() or "0deg").replace(" ", "")
        if phi_cut not in ("0deg", "90deg"):
            phi_cut = "0deg"
            self.var_vrp_phi.set(phi_cut)
        # Keep raw Theta cut and let VRP parser choose the coherent branch in [-90, 90].
        self._pull_cut(mode="VRP", primary="Theta", fixed_phi=phi_cut, theta_to_elev=False)

    def _pull_hrp(self):
        # Padrao operacional: HRP (azimute) por corte em Phi=90 varrendo Theta.
        self._pull_cut(mode="HRP", primary="Theta", fixed_phi="90deg", theta_to_elev=False)

    def _cut_cache_key(self, mode: str, primary: str, fixed: Dict[str, str], theta_to_elev: bool, setup_sweep: str = "") -> tuple:
        return (
            mode,
            self.var_project.get().strip(),
            self.var_design.get().strip(),
            str(setup_sweep or self.var_setup.get().strip()),
            self.var_sphere.get().strip(),
            self.var_expr.get().strip(),
            self.var_freq.get().strip(),
            primary,
            tuple(sorted((str(k), str(v)) for k, v in fixed.items())),
            int(theta_to_elev),
            self.var_lin_mode.get().strip().lower(),
        )

    def _store_cut_payload(self, mode: str, cut, setup: str) -> Path:
        angles = np.asarray(cut.angles_deg, dtype=float)
        raw_values = np.asarray(cut.values, dtype=float)
        if raw_values.size:
            self._log(
                f"[INFO] {mode} raw pull stats: n={int(raw_values.size)} "
                f"min={float(np.min(raw_values)):.6f} max={float(np.max(raw_values)):.6f} "
                f"mean={float(np.mean(raw_values)):.6f} std={float(np.std(raw_values)):.6e}"
            )
        mag_lin, mag_db = self._to_mag_arrays(raw_values)
        mode_u = str(mode).upper()

        if angles.size and mag_lin.size:
            n0 = int(min(angles.size, mag_lin.size))
            ang0 = np.asarray(angles[:n0], dtype=float)
            lin0 = np.asarray(mag_lin[:n0], dtype=float)
            pk_idx = int(np.argmax(lin0))
            self._log(
                f"[INFO] {mode_u} raw axis: range=[{float(np.min(ang0)):.2f},{float(np.max(ang0)):.2f}] deg "
                f"| peak={float(ang0[pk_idx]):.2f} deg"
            )

        # Keep pulled samples intact (no interpolation): only angular shift.
        # - VRP peak target: 0 deg
        # - HRP peak target: 90 deg
        peak_target = 0.0 if mode_u == "VRP" else 90.0
        aligned_angles, aligned_lin, align_meta = shift_cut_no_interp(
            angles,
            mag_lin,
            mode=mode_u,
            target_peak_deg=peak_target,
        )
        aligned_db = mag_db_from_linear(aligned_lin)

        if aligned_lin.size:
            self._log(
                f"[INFO] {mode_u} aligned stats: n={int(aligned_lin.size)} "
                f"min={float(np.min(aligned_lin)):.6f} max={float(np.max(aligned_lin)):.6f} "
                f"mean={float(np.mean(aligned_lin)):.6f} std={float(np.std(aligned_lin)):.6e} "
                f"peak_before={float(align_meta.get('peak_before_deg', 0.0)):.2f} "
                f"peak_after={float(align_meta.get('peak_after_deg', 0.0)):.2f}"
            )
            if float(np.std(aligned_lin)) <= 1e-10:
                self._log(f"[WARN] {mode_u} aligned curve is flat (std~0). Check expression/setup/sphere.")

        safe_setup = setup.replace(" ", "_").replace(":", "").strip() or "Setup"
        out_name = "hrp" if mode_u == "HRP" else "vrp"
        cut_name = f"AEDT_{mode_u}_{safe_setup}"
        npts = int(min(aligned_angles.size, aligned_lin.size))
        min_lin = float(np.min(aligned_lin)) if npts else 0.0
        max_lin = float(np.max(aligned_lin)) if npts else 0.0
        min_db = float(np.min(aligned_db)) if npts else -300.0
        max_db = float(np.max(aligned_db)) if npts else 0.0

        cut_payload = {
            "name": cut_name,
            "angles_deg": aligned_angles.tolist(),
            "mag_lin": aligned_lin.tolist(),
            "mag_db": aligned_db.tolist(),
            "meta": {
                **self._base_meta(),
                **dict(cut.meta),
                "mode": mode_u,
                "npts": npts,
                "lin_min": min_lin,
                "lin_max": max_lin,
                "db_min": min_db,
                "db_max": max_db,
                "angle_alignment": 1,
                "angle_align_method": "shift_no_interp",
                "angle_peak_target_deg": peak_target,
                "angle_peak_before_deg": float(align_meta.get("peak_before_deg", 0.0)),
                "angle_peak_after_deg": float(align_meta.get("peak_after_deg", 0.0)),
                "angle_shift_deg": float(align_meta.get("shift_deg", 0.0)),
            },
        }
        cuts = self._pending_payload.get("cuts_2d", {})
        if not isinstance(cuts, dict):
            cuts = {}
        cuts[mode_u] = cut_payload
        self._pending_payload["cuts_2d"] = cuts
        self._pending_payload["meta"] = self._base_meta()
        self._original_cuts[mode_u] = copy.deepcopy(cut_payload)
        if isinstance(self._preview_cut, dict) and str(self._preview_cut.get("mode", "")).upper() == mode_u:
            self._preview_cut = None
        self._update_send_state()
        self._refresh_plot()
        self._refresh_metadata_box({"timing": f"store_cut {mode_u}"})

        return self.exporter.save_cut_json(out_name, aligned_angles, aligned_lin, cut_payload["meta"])

    def _pull_cut(self, mode: str, primary: str, fixed_phi: Optional[str] = None, fixed_theta: Optional[str] = None, theta_to_elev: bool = False):
        if not self.session.is_connected:
            self._log("[WARN] Not connected.")
            return
        if getattr(self.worker, "is_busy", False):
            self._log("[WARN] AEDT operation in progress. Aguarde concluir.")
            return
        if self._busy_pull:
            self._log("[WARN] Another extraction is in progress. Wait until it finishes.")
            return

        setup = self._resolve_effective_setup()
        if not setup:
            self._log("[WARN] No setup selected for extraction.")
            return
        self.var_setup.set(setup)
        self._log(f"[INFO] Using setup/sweep: {setup}")
        sphere = self.var_sphere.get().strip() or DEFAULT_SPHERE
        preflight = self._prepare_pull_preconditions(setup, sphere, f"{mode} extraction")
        if preflight is None:
            return
        run_analysis = bool(preflight.get("run_analysis", False))
        ensure_sphere = bool(preflight.get("ensure_sphere", False))

        expr = self.var_expr.get().strip() or DEFAULT_EXPR
        freq = self.var_freq.get().strip()
        fixed: Dict[str, str] = {}
        if fixed_phi:
            fixed["Phi"] = fixed_phi
        if fixed_theta:
            fixed["Theta"] = fixed_theta
        if freq:
            fixed["Freq"] = freq

        req = CutRequest(
            setup_sweep=setup,
            sphere_name=sphere,
            expression=expr,
            primary_sweep=primary,
            fixed=fixed,
            convert_theta_to_elevation=theta_to_elev,
        )

        key = self._cut_cache_key(mode, primary, fixed, theta_to_elev, setup)
        cached = self._cut_cache.get(key)
        if (not run_analysis) and (not ensure_sphere) and isinstance(cached, dict) and "cut" in cached:
            cvals = np.asarray(getattr(cached.get("cut"), "values", []), dtype=float)
            cstd = float(np.std(cvals)) if cvals.size else 0.0
            if cvals.size > 2 and cstd <= 1e-10 and self._fallback_expr_if_flat(expr):
                self._log(f"[WARN] Ignoring flat cached {mode} cut for expr '{expr}'. Re-extracting.")
            else:
                t0 = time.perf_counter()
                json_path = self._store_cut_payload(mode, cached["cut"], setup)
                self._log_timing("pull_cut_cache", t0, True, f"mode={mode}")
                self._log(f"[OK] {mode} loaded from cache ({len(cached['cut'].angles_deg)} pts).")
                self._log(f"[OK] Exported: {json_path}")
                self.var_resample_cut.set(mode)
                self._preview_cut = None
                self._refresh_plot()
                self._refresh_metadata_box({"timing": f"pull_cut_cache {(time.perf_counter() - t0) * 1000.0:.2f} ms"})
                return

        t0 = time.perf_counter()
        self._set_busy_pull(True)

        def task():
            self._ensure_selected_context()
            analysis_method = ""
            sphere_method = ""
            if run_analysis:
                analysis_method = self._analyze_selected_setup(setup)
            if ensure_sphere:
                sphere_method = self._ensure_required_infinite_sphere(sphere)
            cut = self.extractor.extract_cut(req)
            expr_requested = expr
            expr_effective = expr_requested
            fallback_note = ""
            vals = np.asarray(cut.values, dtype=float)
            std0 = float(np.std(vals)) if vals.size else 0.0
            if vals.size > 2 and std0 <= 1e-10:
                expr_fb = self._fallback_expr_if_flat(expr_requested)
                if expr_fb and (expr_fb != expr_requested):
                    req_fb = CutRequest(
                        setup_sweep=req.setup_sweep,
                        sphere_name=req.sphere_name,
                        expression=expr_fb,
                        primary_sweep=req.primary_sweep,
                        fixed=req.fixed,
                        convert_theta_to_elevation=req.convert_theta_to_elevation,
                        phi_domain=req.phi_domain,
                        unwrap=req.unwrap,
                    )
                    cut_fb = self.extractor.extract_cut(req_fb)
                    vals_fb = np.asarray(cut_fb.values, dtype=float)
                    std_fb = float(np.std(vals_fb)) if vals_fb.size else 0.0
                    if vals_fb.size == vals.size and std_fb > (std0 + 1e-10):
                        cut = cut_fb
                        expr_effective = expr_fb
                        fallback_note = f"flat_input_fallback:{expr_requested}->{expr_fb}"
            solved_after = self._runtime_solved_sweeps()
            return {
                "cut": cut,
                "analysis_method": analysis_method,
                "sphere_method": sphere_method,
                "solved_sweeps": solved_after,
                "expr_requested": expr_requested,
                "expr_effective": expr_effective,
                "fallback_note": fallback_note,
            }

        def done(res):
            self._set_busy_pull(False)
            if not res.ok:
                self._log_timing("pull_cut", t0, False, f"mode={mode} err={res.error}")
                self._log(f"[ERR] {res.error}")
                return
            out = dict(res.value or {})
            cut = out.get("cut")
            if cut is None:
                self._log_timing("pull_cut", t0, False, f"mode={mode} empty_result")
                self._log("[ERR] Empty extraction result.")
                return
            expr_requested = str(out.get("expr_requested", expr) or expr)
            expr_effective = str(out.get("expr_effective", expr_requested) or expr_requested)
            fallback_note = str(out.get("fallback_note", "") or "")
            self._known_solved_sweeps = self._unique(list(out.get("solved_sweeps", [])))
            analysis_method = str(out.get("analysis_method", "") or "")
            sphere_method = str(out.get("sphere_method", "") or "")
            if analysis_method:
                self._log(f"[OK] Simulation completed using method '{analysis_method}'.")
            if sphere_method:
                self._cut_cache = {}
                self._grid_cache = {}
                self._log(f"[OK] Infinite sphere configured using method '{sphere_method}'.")
            if analysis_method or sphere_method:
                self._refresh_metadata()
            self._cut_cache[key] = {"cut": cut}
            json_path = self._store_cut_payload(mode, cut, setup)
            self._log_timing("pull_cut", t0, True, f"mode={mode} points={len(cut.angles_deg)}")
            self._log(f"[OK] {mode} extracted: {len(cut.angles_deg)} pts.")
            self._log(f"[OK] Exported: {json_path}")
            if fallback_note:
                self._log(f"[WARN] Expression returned flat values; fallback applied: {expr_requested} -> {expr_effective}")
            cuts = self._pending_payload.get("cuts_2d", {})
            if isinstance(cuts, dict):
                c = cuts.get(mode)
                if isinstance(c, dict):
                    meta = c.get("meta", {}) if isinstance(c.get("meta"), dict) else {}
                    meta["expr_requested"] = expr_requested
                    meta["expr_effective"] = expr_effective
                    c["meta"] = meta
                    cuts[mode] = c
                    self._pending_payload["cuts_2d"] = cuts
            self.var_resample_cut.set(mode)
            self._preview_cut = None
            self._refresh_plot()
            self._refresh_metadata_box({"timing": f"pull_cut {(time.perf_counter() - t0) * 1000.0:.2f} ms", "mode": mode})
            self._log(f"[OK] {mode} ready. Validate and click 'Insert {mode}' to import into Project.")

        self._log(f"Extracting {mode}...")
        self.worker.run(task, done)

    def _grid_cache_key(self, theta_pts: int, phi_pts: int, setup_sweep: str = "") -> tuple:
        return (
            self.var_project.get().strip(),
            self.var_design.get().strip(),
            str(setup_sweep or self.var_setup.get().strip()),
            self.var_sphere.get().strip(),
            self.var_freq.get().strip(),
            self.var_expr.get().strip(),
            int(theta_pts),
            int(phi_pts),
            float(self._parse_float(self.var_db_floor.get(), -40.0)),
            float(self._parse_float(self.var_gamma.get(), 1.0)),
            self.var_lin_mode.get().strip().lower(),
        )

    def _pull_3d(self):
        if not self.session.is_connected:
            self._log("[WARN] Not connected.")
            return
        if getattr(self.worker, "is_busy", False):
            self._log("[WARN] AEDT operation in progress. Aguarde concluir.")
            return
        if self._busy_pull:
            self._log("[WARN] Another extraction is in progress. Wait until it finishes.")
            return

        setup = self._resolve_effective_setup()
        if not setup:
            self._log("[WARN] No setup selected for extraction.")
            return
        self.var_setup.set(setup)
        self._log(f"[INFO] Using setup/sweep: {setup}")
        sphere = self.var_sphere.get().strip() or DEFAULT_SPHERE
        preflight = self._prepare_pull_preconditions(setup, sphere, "3D extraction")
        if preflight is None:
            return
        run_analysis = bool(preflight.get("run_analysis", False))
        ensure_sphere = bool(preflight.get("ensure_sphere", False))

        expr = self.var_expr.get().strip() or DEFAULT_EXPR
        freq = self.var_freq.get().strip() or None
        theta_pts = max(3, self._parse_int(self.var_theta_pts.get(), 181))
        phi_pts = max(3, self._parse_int(self.var_phi_pts.get(), 361))
        db_floor = self._parse_float(self.var_db_floor.get(), -40.0)
        gamma = self._parse_float(self.var_gamma.get(), 1.0)

        cache_key = self._grid_cache_key(theta_pts, phi_pts, setup)
        cached = self._grid_cache.get(cache_key)
        if (not run_analysis) and (not ensure_sphere) and isinstance(cached, dict):
            t0 = time.perf_counter()
            self._pending_payload["spherical_3d"] = copy.deepcopy(cached.get("spherical"))
            self._pending_payload["meta"] = self._base_meta()
            self._update_send_state()
            self._log_timing("pull_3d_cache", t0, True, f"shape={cached.get('shape', '-')}")
            self._log(f"[OK] 3D loaded from cache. NPZ: {cached.get('npz_path', '-')}")
            self._refresh_metadata_box({"timing": f"pull_3d_cache {(time.perf_counter() - t0) * 1000.0:.2f} ms"})
            return

        req = GridRequest(
            setup_sweep=setup,
            sphere_name=sphere,
            expression=expr,
            theta_points=theta_pts,
            phi_points=phi_pts,
            freq=freq,
            convert_theta_to_elevation=False,
        )

        t0 = time.perf_counter()
        self._set_busy_pull(True)

        def task():
            self._ensure_selected_context()
            analysis_method = ""
            sphere_method = ""
            if run_analysis:
                analysis_method = self._analyze_selected_setup(setup)
            if ensure_sphere:
                sphere_method = self._ensure_required_infinite_sphere(sphere)
            grid = self.extractor.extract_grid(req)
            theta = np.asarray(grid.theta_deg, dtype=float)
            phi = np.asarray(grid.phi_deg, dtype=float)
            raw_values = np.asarray(grid.values, dtype=float)
            mag_lin, mag_db = self._to_mag_arrays(raw_values)
            meta = {**self._base_meta(), **dict(grid.meta), "mode": "3D"}
            npz_path = self.exporter.save_grid_npz("ff_3d", theta, phi, mag_db, meta)

            db_max = float(np.max(mag_db)) if mag_db.size else 0.0
            if db_max <= db_floor:
                db_max = db_floor + 1.0
            obj_path, mtl_path = self.exporter.export_obj_from_db_grid(
                name="ff_3d",
                theta_deg=theta,
                phi_deg=phi,
                values_db=mag_db,
                db_min=db_floor,
                db_max=db_max,
                gamma=gamma,
                scale=1.0,
            )

            spherical = {
                "name": f"AEDT_3D_{setup.replace(' ', '_').replace(':', '') or 'Setup'}",
                "theta_deg": theta.tolist(),
                "phi_deg": phi.tolist(),
                "mag_db": mag_db.tolist(),
                "mag_lin": mag_lin.tolist(),
                "npz_path": str(npz_path),
                "obj_path": str(obj_path),
                "mtl_path": str(mtl_path),
                "meta": meta,
            }
            solved_after = self._runtime_solved_sweeps()
            return spherical, npz_path, analysis_method, sphere_method, solved_after

        def done(res):
            self._set_busy_pull(False)
            if not res.ok:
                self._log_timing("pull_3d", t0, False, res.error or "")
                self._log(f"[ERR] {res.error}")
                return
            spherical, npz_path, analysis_method, sphere_method, solved_after = res.value
            self._known_solved_sweeps = self._unique(list(solved_after or []))
            if analysis_method:
                self._log(f"[OK] Simulation completed using method '{analysis_method}'.")
            if sphere_method:
                self._cut_cache = {}
                self._grid_cache = {}
                self._log(f"[OK] Infinite sphere configured using method '{sphere_method}'.")
            if analysis_method or sphere_method:
                self._refresh_metadata()
            shape = f"{len(spherical['theta_deg'])}x{len(spherical['phi_deg'])}"
            self._pending_payload["spherical_3d"] = spherical
            self._pending_payload["meta"] = self._base_meta()
            self._grid_cache[cache_key] = {
                "spherical": copy.deepcopy(spherical),
                "npz_path": str(npz_path),
                "shape": shape,
            }
            self._update_send_state()
            self._log_timing("pull_3d", t0, True, f"shape={shape}")
            self._log(f"[OK] 3D grid exported: {npz_path}")
            self._log(f"[OK] 3D shape: {shape}")
            self._refresh_metadata_box({"timing": f"pull_3d {(time.perf_counter() - t0) * 1000.0:.2f} ms", "shape": shape})
            self._log("[OK] 3D ready. Use 'Send to Project' when you want to import.")

        self._log("Extracting 3D pattern (Theta x Phi)...")
        self.worker.run(task, done)

    # -------------- Send actions --------------

    def _has_cut_mode(self, mode: str) -> bool:
        cuts = self._pending_payload.get("cuts_2d", {})
        if not isinstance(cuts, dict):
            return False
        cut = cuts.get(str(mode).upper())
        if not isinstance(cut, dict):
            return False
        ang = np.asarray(cut.get("angles_deg", []), dtype=float)
        val = np.asarray(cut.get("mag_lin", []), dtype=float)
        if val.size == 0:
            val = np.asarray(cut.get("values", []), dtype=float)
        return bool(min(ang.size, val.size) > 0)

    def _single_cut_payload(self, mode: str) -> Dict[str, object]:
        mode_u = str(mode).upper()
        cuts = self._pending_payload.get("cuts_2d", {})
        if not isinstance(cuts, dict):
            raise RuntimeError(f"No pending payload for {mode_u}.")
        cut = cuts.get(mode_u)
        if not isinstance(cut, dict):
            raise RuntimeError(f"No {mode_u} cut available.")
        if not self._has_cut_mode(mode_u):
            raise RuntimeError(f"{mode_u} cut has no valid samples.")
        return {
            "cuts_2d": {mode_u: copy.deepcopy(cut)},
            "spherical_3d": None,
            "meta": self._base_meta(),
        }

    def _send_cut_to_project(self, mode: str):
        mode_u = str(mode).upper()
        if not self._has_cut_mode(mode_u):
            self._log(f"[WARN] No pending {mode_u} cut to send.")
            return
        payload = self._single_cut_payload(mode_u)
        t0 = time.perf_counter()

        def task():
            if self.app is not None and hasattr(self.app, "import_pattern_into_project") and callable(getattr(self.app, "import_pattern_into_project")):
                self.app.import_pattern_into_project(payload)
                return f"app.import_pattern_into_project:{mode_u}"
            if self.bridge is not None:
                ok = self.bridge.push_payload_to_project(payload)
                if ok:
                    return f"bridge.push_payload_to_project:{mode_u}"
            raise RuntimeError("No project bridge available.")

        def done(res):
            if not res.ok:
                self._log_timing("send_cut_project", t0, False, f"mode={mode_u} err={res.error or ''}")
                self._log(f"[ERR] {res.error}")
                return
            self._log_timing("send_cut_project", t0, True, f"mode={mode_u}")
            self._log(f"[OK] {mode_u} sent to project workspace.")
            self._refresh_metadata_box({"timing": f"send_cut_project {mode_u} {(time.perf_counter() - t0) * 1000.0:.2f} ms"})

        self.worker.run(task, done)

    def _send_to_project(self):
        if not self._payload_has_data():
            self._log("[WARN] Nothing to send.")
            return
        payload = self._payload_snapshot()
        t0 = time.perf_counter()

        def task():
            if self.app is not None and hasattr(self.app, "import_pattern_into_project") and callable(getattr(self.app, "import_pattern_into_project")):
                self.app.import_pattern_into_project(payload)
                return "app.import_pattern_into_project"
            if self.bridge is not None:
                ok = self.bridge.push_payload_to_project(payload)
                if ok:
                    return "bridge.push_payload_to_project"
            raise RuntimeError("No project bridge available.")

        def done(res):
            if not res.ok:
                self._log_timing("send_project", t0, False, res.error or "")
                self._log(f"[ERR] {res.error}")
                return
            self._log_timing("send_project", t0, True, str(res.value))
            self._log("[OK] Payload sent to project workspace.")
            self._refresh_metadata_box({"timing": f"send_project {(time.perf_counter() - t0) * 1000.0:.2f} ms"})

        self.worker.run(task, done)

    def _send_to_library(self):
        if not self._payload_has_data():
            self._log("[WARN] Nothing to send.")
            return
        payload = self._payload_snapshot()
        t0 = time.perf_counter()

        def task():
            if self.app is not None and hasattr(self.app, "add_diagram_entry") and callable(getattr(self.app, "add_diagram_entry")):
                rid = self.app.add_diagram_entry(payload)
                return f"app.add_diagram_entry:{rid}"
            if self.bridge is not None:
                ok = self.bridge.push_payload_to_library(payload)
                if ok:
                    return "bridge.push_payload_to_library"
            raise RuntimeError("No library bridge available.")

        def done(res):
            if not res.ok:
                self._log_timing("send_library", t0, False, res.error or "")
                self._log(f"[ERR] {res.error}")
                return
            self._log_timing("send_library", t0, True, str(res.value))
            self._log("[OK] Payload sent to library.")
            self._refresh_metadata_box({"timing": f"send_library {(time.perf_counter() - t0) * 1000.0:.2f} ms"})

        self.worker.run(task, done)

    # -------------- Logging --------------

    def _log(self, msg: str):
        stamp = time.strftime("%H:%M:%S")
        line = f"[{stamp}] {msg}"
        self._append_log_text(line)
        self.logger.info(msg)
