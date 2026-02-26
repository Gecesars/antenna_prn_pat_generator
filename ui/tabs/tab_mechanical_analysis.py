from __future__ import annotations

import csv
import datetime
import importlib.util
import json
import math
import os
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import customtkinter as ctk
import numpy as np
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from tkinter import filedialog, messagebox, simpledialog, ttk

from core.obj_parser import parse_obj_file


EXPORT_FORMATS = {
    "OBJ (mesh)": ".obj",
    "STEP": ".step",
    "STL": ".stl",
    "SAT": ".sat",
    "Parasolid XT": ".x_t",
}

MATERIAL_COLUMNS: List[Tuple[str, str]] = [
    ("name", "Name"),
    ("density_kg_m3", "Density kg/m3"),
    ("conductivity_s_m", "Conductivity S/m"),
    ("epsilon_r", "Epsilon_r"),
    ("mu_r", "Mu_r"),
    ("loss_tangent", "Loss tangent"),
    ("thermal_k_w_mk", "Thermal k W/mK"),
    ("heat_capacity_j_kgk", "Cp J/kgK"),
    ("young_pa", "Young Pa"),
    ("poisson", "Poisson"),
    ("yield_pa", "Yield Pa"),
    ("color", "Color"),
]

DEFAULT_MATERIALS = [
    {
        "name": "Aluminum_6061",
        "density_kg_m3": "2700",
        "conductivity_s_m": "3.5e7",
        "epsilon_r": "1.0",
        "mu_r": "1.0",
        "loss_tangent": "0.0",
        "thermal_k_w_mk": "167",
        "heat_capacity_j_kgk": "896",
        "young_pa": "6.9e10",
        "poisson": "0.33",
        "yield_pa": "2.76e8",
        "color": "#b7bcc6",
    },
    {
        "name": "Copper",
        "density_kg_m3": "8960",
        "conductivity_s_m": "5.8e7",
        "epsilon_r": "1.0",
        "mu_r": "1.0",
        "loss_tangent": "0.0",
        "thermal_k_w_mk": "401",
        "heat_capacity_j_kgk": "385",
        "young_pa": "1.1e11",
        "poisson": "0.34",
        "yield_pa": "7.0e7",
        "color": "#c87a4f",
    },
    {
        "name": "Steel_1018",
        "density_kg_m3": "7870",
        "conductivity_s_m": "6.99e6",
        "epsilon_r": "1.0",
        "mu_r": "200",
        "loss_tangent": "0.0",
        "thermal_k_w_mk": "51.9",
        "heat_capacity_j_kgk": "486",
        "young_pa": "2.1e11",
        "poisson": "0.29",
        "yield_pa": "3.7e8",
        "color": "#7a7d82",
    },
    {
        "name": "FR4",
        "density_kg_m3": "1850",
        "conductivity_s_m": "0.0",
        "epsilon_r": "4.4",
        "mu_r": "1.0",
        "loss_tangent": "0.02",
        "thermal_k_w_mk": "0.3",
        "heat_capacity_j_kgk": "1200",
        "young_pa": "2.4e10",
        "poisson": "0.14",
        "yield_pa": "2.0e8",
        "color": "#8db16a",
    },
    {
        "name": "PTFE",
        "density_kg_m3": "2200",
        "conductivity_s_m": "0.0",
        "epsilon_r": "2.1",
        "mu_r": "1.0",
        "loss_tangent": "0.0002",
        "thermal_k_w_mk": "0.25",
        "heat_capacity_j_kgk": "1000",
        "young_pa": "5.0e8",
        "poisson": "0.46",
        "yield_pa": "2.0e7",
        "color": "#f6f6f4",
    },
    {
        "name": "ABS",
        "density_kg_m3": "1040",
        "conductivity_s_m": "0.0",
        "epsilon_r": "2.8",
        "mu_r": "1.0",
        "loss_tangent": "0.01",
        "thermal_k_w_mk": "0.18",
        "heat_capacity_j_kgk": "1300",
        "young_pa": "2.2e9",
        "poisson": "0.35",
        "yield_pa": "4.0e7",
        "color": "#d8cfba",
    },
]


@dataclass
class MechanicalObject:
    name: str
    vertices: np.ndarray
    faces: np.ndarray
    color: str = "#86b6f6"
    opacity: float = 0.85
    material: str = "Undefined"
    source: str = "Local"
    visible: bool = True
    aedt_name: str = ""
    source_path: str = ""
    obj_object: str = ""
    obj_group: str = ""
    obj_material: str = ""
    obj_smoothing: str = ""
    corner_indices: Optional[List[Dict[str, Any]]] = None
    parser_warnings: Optional[List[str]] = None


def _safe_float(value: str, default: float = 0.0) -> float:
    try:
        txt = str(value).strip().replace(",", ".")
        if not txt:
            return float(default)
        return float(txt)
    except Exception:
        return float(default)


def _safe_int(value: str, default: int = 0) -> int:
    try:
        return int(round(_safe_float(value, float(default))))
    except Exception:
        return int(default)


def _unique_name(base: str, used: Sequence[str]) -> str:
    token = str(base or "Object").strip() or "Object"
    if token not in used:
        return token
    idx = 2
    while True:
        cand = f"{token}_{idx}"
        if cand not in used:
            return cand
        idx += 1


def _hex_from_color(raw, default: str = "#86b6f6") -> str:
    if isinstance(raw, str):
        txt = raw.strip()
        if txt.startswith("#") and (len(txt) in (4, 7)):
            return txt
        return default
    if isinstance(raw, (tuple, list)) and len(raw) >= 3:
        vals = []
        for x in raw[:3]:
            try:
                v = float(x)
            except Exception:
                v = 0.0
            if v <= 1.0:
                v = max(0.0, min(1.0, v)) * 255.0
            vals.append(int(max(0, min(255, round(v)))))
        return f"#{vals[0]:02x}{vals[1]:02x}{vals[2]:02x}"
    return default


def _mesh_bounds(vertices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if vertices.size == 0:
        z = np.zeros(3, dtype=float)
        return z, z
    return np.min(vertices, axis=0), np.max(vertices, axis=0)


def _rotation_matrix_xyz(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    rx = math.radians(float(rx_deg))
    ry = math.radians(float(ry_deg))
    rz = math.radians(float(rz_deg))
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    rx_m = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=float)
    ry_m = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=float)
    rz_m = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    return rz_m @ ry_m @ rx_m


def _create_box_mesh(width: float, depth: float, height: float, cx: float, cy: float, cz: float):
    w = max(1e-9, float(width)) * 0.5
    d = max(1e-9, float(depth)) * 0.5
    h = max(1e-9, float(height)) * 0.5
    v = np.array(
        [
            [cx - w, cy - d, cz - h],
            [cx + w, cy - d, cz - h],
            [cx + w, cy + d, cz - h],
            [cx - w, cy + d, cz - h],
            [cx - w, cy - d, cz + h],
            [cx + w, cy - d, cz + h],
            [cx + w, cy + d, cz + h],
            [cx - w, cy + d, cz + h],
        ],
        dtype=float,
    )
    f = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 6, 5],
            [4, 7, 6],
            [0, 4, 5],
            [0, 5, 1],
            [1, 5, 6],
            [1, 6, 2],
            [2, 6, 7],
            [2, 7, 3],
            [3, 7, 4],
            [3, 4, 0],
        ],
        dtype=int,
    )
    return v, f


def _create_cylinder_mesh(radius: float, height: float, segments: int, cx: float, cy: float, cz: float):
    r = max(1e-9, float(radius))
    h = max(1e-9, float(height))
    n = max(6, int(segments))
    top_z = cz + h * 0.5
    bot_z = cz - h * 0.5
    ang = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    top = np.column_stack([cx + r * np.cos(ang), cy + r * np.sin(ang), np.full(n, top_z)])
    bot = np.column_stack([cx + r * np.cos(ang), cy + r * np.sin(ang), np.full(n, bot_z)])
    v = np.vstack([top, bot, [[cx, cy, top_z], [cx, cy, bot_z]]]).astype(float)
    top_center = 2 * n
    bot_center = 2 * n + 1
    faces: List[List[int]] = []
    for i in range(n):
        j = (i + 1) % n
        ti, tj = i, j
        bi, bj = n + i, n + j
        faces.append([ti, tj, bj])
        faces.append([ti, bj, bi])
        faces.append([top_center, tj, ti])
        faces.append([bot_center, bi, bj])
    return v, np.asarray(faces, dtype=int)


class MechanicalAnalysisTab(ctk.CTkFrame):
    def __init__(self, master, app, **kwargs):
        super().__init__(master, **kwargs)
        self.app = app
        self.objects: Dict[str, MechanicalObject] = {}
        self.selected_object: str = ""
        self._busy_export = False
        self._export_thread: Optional[threading.Thread] = None
        self._artists: Dict[str, Poly3DCollection] = {}
        self._last_pick_pixels: Dict[str, Tuple[float, float]] = {}
        self._last_cad_payload_runtime: Optional[dict] = None
        self._last_cad_manifest: Optional[dict] = None

        default_out = os.path.join(os.getcwd(), "aedt_exports", "mechanical")
        self.export_format_var = tk.StringVar(value=list(EXPORT_FORMATS.keys())[0])
        self.export_dir_var = tk.StringVar(value=default_out)
        self.status_var = tk.StringVar(value="Analise mecanica pronta.")
        self.push_material_aedt_var = tk.BooleanVar(value=True)
        appearance = "Dark"
        try:
            appearance = str(ctk.get_appearance_mode() or "Dark")
        except Exception:
            appearance = "Dark"
        self.appearance_var = tk.StringVar(value=appearance)

        self.t_var = {k: tk.StringVar(value="0.0") for k in ("tx", "ty", "tz", "rx", "ry", "rz")}
        self.t_var["scale"] = tk.StringVar(value="1.0")
        self.prim_var = {
            "type": tk.StringVar(value="Box"),
            "name": tk.StringVar(value="Base_1"),
            "w": tk.StringVar(value="1.0"),
            "d": tk.StringVar(value="1.0"),
            "h": tk.StringVar(value="1.0"),
            "r": tk.StringVar(value="0.5"),
            "segments": tk.StringVar(value="24"),
            "cx": tk.StringVar(value="0.0"),
            "cy": tk.StringVar(value="0.0"),
            "cz": tk.StringVar(value="0.0"),
        }
        self.mat_form_vars = {key: tk.StringVar(value="") for key, _ in MATERIAL_COLUMNS}

        self._build_ui()
        self._load_default_materials()
        self._refresh_object_tree()
        self._redraw_scene()

    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        top = ctk.CTkFrame(self)
        top.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 6))
        for col in range(15):
            top.grid_columnconfigure(col, weight=0)
        top.grid_columnconfigure(6, weight=1)

        ctk.CTkLabel(top, text="Formato CAD").grid(row=0, column=0, padx=(8, 4), pady=8, sticky="w")
        ctk.CTkOptionMenu(top, variable=self.export_format_var, values=list(EXPORT_FORMATS.keys()), width=160).grid(
            row=0, column=1, padx=4, pady=8, sticky="w"
        )
        ctk.CTkLabel(top, text="Output dir").grid(row=0, column=2, padx=(10, 4), pady=8, sticky="w")
        ctk.CTkEntry(top, textvariable=self.export_dir_var, width=360).grid(row=0, column=3, padx=4, pady=8, sticky="ew")
        ctk.CTkButton(top, text="Browse", width=80, command=self._browse_export_dir).grid(row=0, column=4, padx=4, pady=8)
        self.btn_export_aedt = ctk.CTkButton(top, text="Exportar do AEDT", width=140, command=self._export_from_aedt)
        self.btn_export_aedt.grid(row=0, column=5, padx=(6, 4), pady=8)
        ctk.CTkButton(top, text="Carregar OBJ", width=120, command=self._load_obj_files).grid(
            row=0, column=6, padx=(6, 4), pady=8, sticky="w"
        )
        ctk.CTkButton(top, text="Snapshot PNG", width=120, command=self._save_snapshot).grid(row=0, column=7, padx=4, pady=8)
        ctk.CTkButton(top, text="Limpar cena", width=110, command=self._clear_scene).grid(row=0, column=8, padx=4, pady=8)
        ctk.CTkButton(top, text="Modeler Pro (PySide6)", width=170, command=self._open_pyside6_modeler).grid(
            row=0, column=9, padx=4, pady=8
        )
        ctk.CTkButton(top, text="Salvar Cena", width=110, command=self._save_scene_json).grid(row=0, column=10, padx=4, pady=8)
        ctk.CTkButton(top, text="Carregar Cena", width=118, command=self._load_scene_json).grid(row=0, column=11, padx=4, pady=8)
        ctk.CTkLabel(top, text="Tema").grid(row=0, column=12, padx=(6, 2), pady=8, sticky="e")
        ctk.CTkOptionMenu(
            top,
            variable=self.appearance_var,
            values=["Dark", "Light", "System"],
            width=100,
            command=self._apply_theme,
        ).grid(row=0, column=13, padx=2, pady=8)
        ctk.CTkLabel(top, text="Mouse: rotate/pan/zoom via toolbar", anchor="w").grid(
            row=0, column=14, padx=(10, 8), pady=8, sticky="w"
        )

        body = ctk.CTkFrame(self)
        body.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        body.grid_rowconfigure(0, weight=1)
        body.grid_columnconfigure(0, weight=2)
        body.grid_columnconfigure(1, weight=1)

        viewer = ctk.CTkFrame(body)
        viewer.grid(row=0, column=0, sticky="nsew", padx=(0, 6), pady=4)
        viewer.grid_rowconfigure(1, weight=1)
        viewer.grid_columnconfigure(0, weight=1)

        toolbar_frame = ctk.CTkFrame(viewer, fg_color="transparent")
        toolbar_frame.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 0))

        self.figure = Figure(figsize=(8.5, 6.2), dpi=100)
        self.ax = self.figure.add_subplot(111, projection="3d")
        self.ax.set_title("Analise Mecanica 3D")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.canvas = FigureCanvasTkAgg(self.figure, master=viewer)
        self.canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew", padx=8, pady=8)
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(side="left", fill="x")
        self.canvas.mpl_connect("button_press_event", self._on_plot_press)

        side = ctk.CTkTabview(body)
        side.grid(row=0, column=1, sticky="nsew", padx=(6, 0), pady=4)
        tab_obj = side.add("Objetos")
        tab_mat = side.add("Materiais")
        tab_log = side.add("Log")

        self._build_objects_tab(tab_obj)
        self._build_materials_tab(tab_mat)
        self._build_log_tab(tab_log)

        ctk.CTkLabel(self, textvariable=self.status_var, anchor="w").grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 8))

    def _build_objects_tab(self, tab_obj):
        tab_obj.grid_columnconfigure(0, weight=1)
        tab_obj.grid_rowconfigure(0, weight=1)

        tree_wrap = ctk.CTkFrame(tab_obj)
        tree_wrap.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
        tree_wrap.grid_columnconfigure(0, weight=1)
        tree_wrap.grid_rowconfigure(0, weight=1)
        cols = ("material", "verts", "faces", "source", "visible")
        self.obj_tree = ttk.Treeview(tree_wrap, columns=cols, show="tree headings", selectmode="extended", height=8)
        self.obj_tree.heading("#0", text="Objeto")
        self.obj_tree.column("#0", width=150, anchor="w")
        self.obj_tree.heading("material", text="Material")
        self.obj_tree.heading("verts", text="V")
        self.obj_tree.heading("faces", text="F")
        self.obj_tree.heading("source", text="Fonte")
        self.obj_tree.heading("visible", text="Vis")
        self.obj_tree.column("material", width=120, anchor="w")
        self.obj_tree.column("verts", width=55, anchor="center")
        self.obj_tree.column("faces", width=55, anchor="center")
        self.obj_tree.column("source", width=90, anchor="center")
        self.obj_tree.column("visible", width=50, anchor="center")
        ysb = ttk.Scrollbar(tree_wrap, orient="vertical", command=self.obj_tree.yview)
        self.obj_tree.configure(yscrollcommand=ysb.set)
        self.obj_tree.grid(row=0, column=0, sticky="nsew")
        ysb.grid(row=0, column=1, sticky="ns")
        self.obj_tree.bind("<<TreeviewSelect>>", self._on_object_select_tree)
        self.obj_tree.bind("<Button-3>", self._on_object_context)
        self.obj_tree.bind("<Button-2>", self._on_object_context)

        btns = ctk.CTkFrame(tab_obj, fg_color="transparent")
        btns.grid(row=1, column=0, sticky="ew", padx=6, pady=(0, 4))
        ctk.CTkButton(btns, text="Renomear", width=85, command=self._rename_selected_object).pack(side="left", padx=2)
        ctk.CTkButton(btns, text="Duplicar", width=85, command=self._duplicate_selected_object).pack(side="left", padx=2)
        ctk.CTkButton(btns, text="Excluir", width=85, fg_color="#9a4747", command=self._delete_selected_objects).pack(side="left", padx=2)
        ctk.CTkButton(btns, text="Visivel", width=85, command=self._toggle_selected_visibility).pack(side="left", padx=2)
        ctk.CTkButton(btns, text="Fit", width=70, command=self._fit_view).pack(side="left", padx=2)

        tbox = ctk.CTkFrame(tab_obj)
        tbox.grid(row=2, column=0, sticky="ew", padx=6, pady=4)
        ctk.CTkLabel(tbox, text="Transform (objeto selecionado)").pack(anchor="w", padx=6, pady=(4, 2))
        row1 = ctk.CTkFrame(tbox, fg_color="transparent")
        row1.pack(fill="x", padx=6, pady=2)
        ctk.CTkLabel(row1, text="T x y z", width=60).pack(side="left")
        ctk.CTkEntry(row1, textvariable=self.t_var["tx"], width=64).pack(side="left", padx=2)
        ctk.CTkEntry(row1, textvariable=self.t_var["ty"], width=64).pack(side="left", padx=2)
        ctk.CTkEntry(row1, textvariable=self.t_var["tz"], width=64).pack(side="left", padx=2)
        row2 = ctk.CTkFrame(tbox, fg_color="transparent")
        row2.pack(fill="x", padx=6, pady=2)
        ctk.CTkLabel(row2, text="R x y z", width=60).pack(side="left")
        ctk.CTkEntry(row2, textvariable=self.t_var["rx"], width=64).pack(side="left", padx=2)
        ctk.CTkEntry(row2, textvariable=self.t_var["ry"], width=64).pack(side="left", padx=2)
        ctk.CTkEntry(row2, textvariable=self.t_var["rz"], width=64).pack(side="left", padx=2)
        ctk.CTkLabel(row2, text="Scale", width=55).pack(side="left", padx=(8, 2))
        ctk.CTkEntry(row2, textvariable=self.t_var["scale"], width=72).pack(side="left", padx=2)
        row3 = ctk.CTkFrame(tbox, fg_color="transparent")
        row3.pack(fill="x", padx=6, pady=(2, 6))
        ctk.CTkButton(row3, text="Aplicar transform", width=130, command=self._apply_transform_selected).pack(side="left", padx=2)
        ctk.CTkButton(row3, text="Reset campos", width=110, command=self._reset_transform_fields).pack(side="left", padx=2)

        pbox = ctk.CTkFrame(tab_obj)
        pbox.grid(row=3, column=0, sticky="ew", padx=6, pady=4)
        ctk.CTkLabel(pbox, text="Estruturas base").pack(anchor="w", padx=6, pady=(4, 2))
        r0 = ctk.CTkFrame(pbox, fg_color="transparent")
        r0.pack(fill="x", padx=6, pady=2)
        ctk.CTkLabel(r0, text="Tipo", width=60).pack(side="left")
        ctk.CTkOptionMenu(r0, variable=self.prim_var["type"], values=["Box", "Cylinder", "Plate"], width=110).pack(side="left", padx=2)
        ctk.CTkLabel(r0, text="Nome", width=60).pack(side="left", padx=(8, 2))
        ctk.CTkEntry(r0, textvariable=self.prim_var["name"], width=160).pack(side="left", padx=2)
        r1 = ctk.CTkFrame(pbox, fg_color="transparent")
        r1.pack(fill="x", padx=6, pady=2)
        ctk.CTkLabel(r1, text="W D H", width=60).pack(side="left")
        ctk.CTkEntry(r1, textvariable=self.prim_var["w"], width=64).pack(side="left", padx=2)
        ctk.CTkEntry(r1, textvariable=self.prim_var["d"], width=64).pack(side="left", padx=2)
        ctk.CTkEntry(r1, textvariable=self.prim_var["h"], width=64).pack(side="left", padx=2)
        ctk.CTkLabel(r1, text="R", width=20).pack(side="left", padx=(8, 2))
        ctk.CTkEntry(r1, textvariable=self.prim_var["r"], width=64).pack(side="left", padx=2)
        ctk.CTkLabel(r1, text="Seg", width=36).pack(side="left", padx=(8, 2))
        ctk.CTkEntry(r1, textvariable=self.prim_var["segments"], width=64).pack(side="left", padx=2)
        r2 = ctk.CTkFrame(pbox, fg_color="transparent")
        r2.pack(fill="x", padx=6, pady=(2, 6))
        ctk.CTkLabel(r2, text="C x y z", width=60).pack(side="left")
        ctk.CTkEntry(r2, textvariable=self.prim_var["cx"], width=64).pack(side="left", padx=2)
        ctk.CTkEntry(r2, textvariable=self.prim_var["cy"], width=64).pack(side="left", padx=2)
        ctk.CTkEntry(r2, textvariable=self.prim_var["cz"], width=64).pack(side="left", padx=2)
        ctk.CTkButton(r2, text="Adicionar", width=110, command=self._add_primitive).pack(side="left", padx=(8, 2))

        obox = ctk.CTkFrame(tab_obj)
        obox.grid(row=4, column=0, sticky="ew", padx=6, pady=4)
        ctk.CTkLabel(obox, text="Operacoes 3D (locais)").pack(anchor="w", padx=6, pady=(4, 2))
        oline = ctk.CTkFrame(obox, fg_color="transparent")
        oline.pack(fill="x", padx=6, pady=(2, 6))
        ctk.CTkButton(oline, text="Merge", width=95, command=self._merge_selected).pack(side="left", padx=2)
        ctk.CTkButton(oline, text="Subtract A-B", width=110, command=self._subtract_selected).pack(side="left", padx=2)
        ctk.CTkButton(oline, text="Intersect", width=95, command=self._intersect_selected).pack(side="left", padx=2)

    def _build_materials_tab(self, tab_mat):
        tab_mat.grid_columnconfigure(0, weight=1)
        tab_mat.grid_rowconfigure(0, weight=1)

        tree_wrap = ctk.CTkFrame(tab_mat)
        tree_wrap.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
        tree_wrap.grid_columnconfigure(0, weight=1)
        tree_wrap.grid_rowconfigure(0, weight=1)
        mat_cols = [c for c, _ in MATERIAL_COLUMNS]
        self.mat_tree = ttk.Treeview(tree_wrap, columns=mat_cols, show="headings", selectmode="browse")
        for col, label in MATERIAL_COLUMNS:
            self.mat_tree.heading(col, text=label)
            width = 126 if col == "name" else 108
            if col == "color":
                width = 90
            self.mat_tree.column(col, width=width, anchor="center")
        ysb = ttk.Scrollbar(tree_wrap, orient="vertical", command=self.mat_tree.yview)
        xsb = ttk.Scrollbar(tree_wrap, orient="horizontal", command=self.mat_tree.xview)
        self.mat_tree.configure(yscrollcommand=ysb.set, xscrollcommand=xsb.set)
        self.mat_tree.grid(row=0, column=0, sticky="nsew")
        ysb.grid(row=0, column=1, sticky="ns")
        xsb.grid(row=1, column=0, sticky="ew")
        self.mat_tree.bind("<<TreeviewSelect>>", self._on_material_tree_select)

        bline = ctk.CTkFrame(tab_mat, fg_color="transparent")
        bline.grid(row=1, column=0, sticky="ew", padx=6, pady=(0, 4))
        ctk.CTkButton(bline, text="Novo", width=70, command=self._add_material_from_form).pack(side="left", padx=2)
        ctk.CTkButton(bline, text="Atualizar", width=80, command=self._update_material_from_form).pack(side="left", padx=2)
        ctk.CTkButton(bline, text="Excluir", width=70, fg_color="#9a4747", command=self._delete_selected_material).pack(side="left", padx=2)
        ctk.CTkButton(bline, text="Import CSV", width=92, command=self._import_materials_csv).pack(side="left", padx=2)
        ctk.CTkButton(bline, text="Export CSV", width=92, command=self._export_materials_csv).pack(side="left", padx=2)
        ctk.CTkButton(bline, text="Aplicar ao objeto", width=126, command=self._apply_material_to_selected_object).pack(side="left", padx=2)
        ctk.CTkCheckBox(
            bline,
            text="Aplicar no AEDT",
            variable=self.push_material_aedt_var,
            onvalue=True,
            offvalue=False,
        ).pack(side="left", padx=(10, 2))

        form = ctk.CTkFrame(tab_mat)
        form.grid(row=2, column=0, sticky="ew", padx=6, pady=(0, 6))
        ctk.CTkLabel(form, text="Formulario de material").grid(row=0, column=0, columnspan=6, sticky="w", padx=6, pady=(4, 2))
        for i in range(6):
            form.grid_columnconfigure(i, weight=1)
        row = 1
        col = 0
        for key, label in MATERIAL_COLUMNS:
            ctk.CTkLabel(form, text=label).grid(row=row, column=col, sticky="w", padx=6, pady=(2, 0))
            ctk.CTkEntry(form, textvariable=self.mat_form_vars[key]).grid(row=row + 1, column=col, sticky="ew", padx=6, pady=(0, 4))
            col += 1
            if col >= 3:
                col = 0
                row += 2

    def _build_log_tab(self, tab_log):
        tab_log.grid_columnconfigure(0, weight=1)
        tab_log.grid_rowconfigure(0, weight=1)
        self.log_box = ctk.CTkTextbox(tab_log)
        self.log_box.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
        self.log_box.insert("1.0", "[init] Analise mecanica inicializada.\n")

    def _set_status(self, text: str):
        msg = str(text)
        self.status_var.set(msg)
        stamp = datetime.datetime.now().strftime("%H:%M:%S")
        try:
            self.log_box.insert("end", f"[{stamp}] {msg}\n")
            self.log_box.see("end")
        except Exception:
            pass
        try:
            if hasattr(self.app, "_set_status"):
                self.app._set_status(msg)
        except Exception:
            pass

    def _browse_export_dir(self):
        d = filedialog.askdirectory(title="Pasta de exportacao CAD")
        if d:
            self.export_dir_var.set(d)

    def _ensure_output_dir(self) -> str:
        out_dir = str(self.export_dir_var.get() or "").strip()
        if not out_dir:
            out_dir = os.path.join(os.getcwd(), "aedt_exports", "mechanical")
            self.export_dir_var.set(out_dir)
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

    def _get_hfss(self):
        live_tab = getattr(self.app, "aedt_live_tab", None)
        if live_tab is None:
            return None
        session = getattr(live_tab, "session", None)
        if session is None:
            return None
        try:
            if not bool(getattr(session, "is_connected", False)):
                return None
            return session.hfss
        except Exception:
            return None

    def _export_from_aedt(self):
        if self._busy_export:
            self._set_status("Exportacao CAD ja em andamento.")
            return
        hfss = self._get_hfss()
        if hfss is None:
            messagebox.showwarning(
                "AEDT nao conectado",
                "Conecte na aba AEDT Live e selecione projeto/design antes de exportar o CAD.",
            )
            return
        out_dir = self._ensure_output_dir()
        fmt_label = str(self.export_format_var.get())
        fmt = EXPORT_FORMATS.get(fmt_label, ".obj")
        self._busy_export = True
        self.btn_export_aedt.configure(state="disabled")
        self._set_status("Exportando modelo CAD do AEDT...")

        def worker():
            try:
                exported = hfss.post.export_model_obj(
                    assignment=None,
                    export_path=out_dir,
                    export_as_multiple_objects=True,
                    air_objects=False,
                )
                obj_files: List[str] = []
                meta: Dict[str, Dict[str, object]] = {}
                for item in exported or []:
                    if isinstance(item, (list, tuple)):
                        if not item:
                            continue
                        path = str(item[0])
                        color = item[1] if len(item) > 1 else None
                        opacity = _safe_float(str(item[2] if len(item) > 2 else "0.85"), 0.85)
                    else:
                        path = str(item)
                        color = None
                        opacity = 0.85
                    if not path:
                        continue
                    path_abs = os.path.abspath(path)
                    if os.path.isfile(path_abs):
                        obj_files.append(path_abs)
                        meta[path_abs] = {"color": color, "opacity": opacity}

                std_path = ""
                if fmt != ".obj":
                    base = self._suggest_export_basename(hfss)
                    ok = bool(
                        hfss.export_3d_model(
                            file_name=base,
                            file_path=out_dir,
                            file_format=fmt,
                            assignment_to_export=None,
                            assignment_to_remove=None,
                        )
                    )
                    if ok:
                        std_path = os.path.abspath(os.path.join(out_dir, f"{base}{fmt}"))

                self.after(0, lambda: self._on_export_done(obj_files, meta, std_path, fmt))
            except Exception as e:
                self.after(0, lambda: self._on_export_error(e))

        self._export_thread = threading.Thread(target=worker, daemon=True)
        self._export_thread.start()

    def _suggest_export_basename(self, hfss) -> str:
        project = ""
        design = ""
        try:
            project = str(getattr(hfss, "project_name", "") or "").strip()
            design = str(getattr(hfss, "design_name", "") or "").strip()
        except Exception:
            pass
        token = "_".join([x for x in [project, design] if x]) or "AEDT_Model"
        token = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in token)
        return token or "AEDT_Model"

    def _on_export_done(self, obj_files: List[str], meta: Dict[str, Dict[str, object]], std_path: str, fmt: str):
        try:
            if not obj_files:
                self._set_status("Exportacao concluida sem malhas OBJ.")
                messagebox.showwarning(
                    "Sem OBJ",
                    "O AEDT nao retornou malhas OBJ para visualizacao.\n"
                    "Verifique se existem solids modelados no design.",
                )
                return
            self._import_obj_paths(obj_files, source="AEDT", per_file_meta=meta)
            if std_path:
                self._set_status(f"CAD exportado e carregado. Arquivo adicional: {std_path}")
            elif fmt == ".obj":
                self._set_status(f"CAD OBJ exportado e carregado ({len(obj_files)} arquivo(s)).")
            else:
                self._set_status(f"CAD carregado via OBJ ({len(obj_files)} arquivo(s)).")
        finally:
            self._busy_export = False
            self.btn_export_aedt.configure(state="normal")

    def _on_export_error(self, err: Exception):
        self._busy_export = False
        self.btn_export_aedt.configure(state="normal")
        self._set_status(f"Erro na exportacao CAD: {err}")
        messagebox.showerror("Erro exportacao AEDT", str(err))

    def _load_obj_files(self):
        paths = filedialog.askopenfilenames(
            title="Selecione OBJ para analise mecanica",
            filetypes=[("OBJ", "*.obj"), ("All files", "*.*")],
        )
        if not paths:
            return
        self._import_obj_paths([str(p) for p in paths], source="Local")

    def _import_obj_paths(self, paths: Sequence[str], source: str, per_file_meta: Optional[Dict[str, Dict[str, object]]] = None):
        imported = 0
        used_names = set(self.objects.keys())
        parsed_files: List[dict] = []
        runtime_meshes: List[dict] = []
        manifest_meshes: List[dict] = []
        parse_errors: List[str] = []
        for path in paths:
            p = os.path.abspath(str(path))
            if not os.path.isfile(p):
                continue
            try:
                model = parse_obj_file(p)
            except Exception as e:
                parse_errors.append(f"{os.path.basename(p)}: {e}")
                continue
            chunks = model.mesh_chunks(split_mode="object_group_material")
            if not chunks:
                continue
            summary = model.summary()
            summary["chunk_count"] = int(len(chunks))
            summary["import_source"] = str(source)
            parsed_files.append(summary)
            file_meta = (per_file_meta or {}).get(p, {})
            color = _hex_from_color(file_meta.get("color"), default="#86b6f6")
            opacity = _safe_float(str(file_meta.get("opacity", "0.85")), 0.85)
            opacity = max(0.05, min(1.0, opacity))
            aedt_name = Path(p).stem if source == "AEDT" else ""
            for chunk in chunks:
                verts = np.asarray(chunk.get("vertices", np.zeros((0, 3), dtype=float)), dtype=float)
                faces = np.asarray(chunk.get("faces", np.zeros((0, 3), dtype=int)), dtype=int)
                if verts.size == 0 or faces.size == 0:
                    continue
                chunk_name = str(chunk.get("name", "") or "").strip() or str(chunk.get("object_name", "") or "").strip() or Path(p).stem
                base = chunk_name if (len(chunks) > 1) else Path(p).stem
                name = _unique_name(base, used_names)
                used_names.add(name)
                parser_warnings = list(model.warnings)
                obj_material = str(chunk.get("material", "") or "").strip()
                material_name = obj_material or "Undefined"
                corner_indices = chunk.get("corner_indices", None)
                if not isinstance(corner_indices, list):
                    corner_indices = None
                obj_smoothing = ""
                if corner_indices and isinstance(corner_indices[0], dict):
                    obj_smoothing = str(corner_indices[0].get("smoothing", "") or "")
                mesh = MechanicalObject(
                    name=name,
                    vertices=verts,
                    faces=faces,
                    color=color,
                    opacity=opacity,
                    material=material_name,
                    source=source,
                    visible=True,
                    aedt_name=aedt_name,
                    source_path=p,
                    obj_object=str(chunk.get("object_name", "") or ""),
                    obj_group=str(chunk.get("group_name", "") or ""),
                    obj_material=obj_material,
                    obj_smoothing=obj_smoothing,
                    corner_indices=corner_indices,
                    parser_warnings=parser_warnings,
                )
                self.objects[name] = mesh
                runtime_meshes.append(
                    {
                        "name": name,
                        "source_path": p,
                        "vertices": np.asarray(verts, dtype=float),
                        "faces": np.asarray(faces, dtype=int),
                        "color": str(color),
                        "opacity": float(opacity),
                        "material": material_name,
                        "object_name": str(chunk.get("object_name", "") or ""),
                        "group_name": str(chunk.get("group_name", "") or ""),
                        "obj_material": obj_material,
                        "smoothing": obj_smoothing,
                    }
                )
                manifest_meshes.append(
                    {
                        "name": name,
                        "source_path": p,
                        "vertex_count": int(verts.shape[0]),
                        "face_count": int(faces.shape[0]),
                        "color": str(color),
                        "opacity": float(opacity),
                        "material": material_name,
                        "object_name": str(chunk.get("object_name", "") or ""),
                        "group_name": str(chunk.get("group_name", "") or ""),
                        "obj_material": obj_material,
                        "smoothing": obj_smoothing,
                        "triangle_count": int(chunk.get("triangle_count", int(faces.shape[0]))),
                        "corner_index_count": int(len(corner_indices or [])),
                    }
                )
                imported += 1
        if imported <= 0:
            if parse_errors:
                self._set_status(f"Falha na importacao OBJ: {parse_errors[0]}")
            else:
                self._set_status("Nenhum OBJ valido foi importado.")
            return
        now_iso = datetime.datetime.now().isoformat(timespec="seconds")
        runtime_payload = {
            "version": 1,
            "source": str(source),
            "imported_at": now_iso,
            "files": parsed_files,
            "meshes": runtime_meshes,
            "stats": {
                "file_count": int(len(parsed_files)),
                "mesh_count": int(len(runtime_meshes)),
                "vertex_count": int(sum(int(np.asarray(x.get("vertices")).shape[0]) for x in runtime_meshes)),
                "face_count": int(sum(int(np.asarray(x.get("faces")).shape[0]) for x in runtime_meshes)),
            },
            "errors": parse_errors,
        }
        manifest_payload = {
            "version": 1,
            "source": str(source),
            "imported_at": now_iso,
            "files": parsed_files,
            "meshes": manifest_meshes,
            "stats": {
                "file_count": int(len(parsed_files)),
                "mesh_count": int(len(manifest_meshes)),
                "vertex_count": int(sum(int(x.get("vertex_count", 0)) for x in manifest_meshes)),
                "face_count": int(sum(int(x.get("face_count", 0)) for x in manifest_meshes)),
            },
            "errors": parse_errors,
        }
        self._publish_cad_payload(runtime_payload, manifest_payload)
        if not self.selected_object and self.objects:
            self.selected_object = next(iter(self.objects.keys()))
        self._refresh_object_tree()
        self._redraw_scene()
        msg = f"Importacao 3D concluida: {imported} objeto(s)."
        if parse_errors:
            msg += f" Arquivos com erro: {len(parse_errors)}."
        self._set_status(msg)

    def _publish_cad_payload(self, runtime_payload: dict, manifest_payload: dict):
        self._last_cad_payload_runtime = runtime_payload if isinstance(runtime_payload, dict) else None
        self._last_cad_manifest = manifest_payload if isinstance(manifest_payload, dict) else None
        try:
            setattr(self.app, "aedt_live_cad_3d", self._last_cad_payload_runtime)
            setattr(self.app, "aedt_live_cad_manifest", self._last_cad_manifest)
        except Exception:
            pass
        try:
            if hasattr(self.app, "_notify_advanced_data_changed"):
                self.app._notify_advanced_data_changed()
        except Exception:
            pass
        try:
            if hasattr(self.app, "_refresh_project_overview"):
                self.app._refresh_project_overview()
        except Exception:
            pass

    def _clear_cad_payload(self):
        self._last_cad_payload_runtime = None
        self._last_cad_manifest = None
        try:
            setattr(self.app, "aedt_live_cad_3d", None)
            setattr(self.app, "aedt_live_cad_manifest", None)
        except Exception:
            pass
        try:
            if hasattr(self.app, "_notify_advanced_data_changed"):
                self.app._notify_advanced_data_changed()
        except Exception:
            pass
        try:
            if hasattr(self.app, "_refresh_project_overview"):
                self.app._refresh_project_overview()
        except Exception:
            pass

    def _save_snapshot(self):
        path = filedialog.asksaveasfilename(
            title="Salvar snapshot da analise 3D",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("All files", "*.*")],
            initialfile="analise_mecanica.png",
        )
        if not path:
            return
        try:
            self.figure.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.04)
            self._set_status(f"Snapshot salvo: {path}")
        except Exception as e:
            messagebox.showerror("Erro snapshot", str(e))

    def _apply_theme(self, value: str):
        mode = str(value or "Dark").strip().capitalize()
        if mode not in ("Dark", "Light", "System"):
            mode = "Dark"
        try:
            ctk.set_appearance_mode(mode)
            self.appearance_var.set(mode)
            self._set_status(f"Tema aplicado: {mode}")
            self.canvas.draw_idle()
        except Exception as e:
            messagebox.showerror("Theme", str(e))

    def _scene_payload(self) -> dict:
        materials: List[dict] = []
        for iid in self.mat_tree.get_children():
            vals = self.mat_tree.item(iid, "values")
            if not vals:
                continue
            row = {k: str(v) for (k, _), v in zip(MATERIAL_COLUMNS, vals)}
            materials.append(row)

        objects: List[dict] = []
        for name, obj in self.objects.items():
            objects.append(
                {
                    "name": str(name),
                    "vertices": np.asarray(obj.vertices, dtype=float).tolist(),
                    "faces": np.asarray(obj.faces, dtype=int).tolist(),
                    "color": str(obj.color),
                    "opacity": float(obj.opacity),
                    "material": str(obj.material),
                    "source": str(obj.source),
                    "visible": bool(obj.visible),
                    "aedt_name": str(obj.aedt_name),
                    "source_path": str(obj.source_path),
                    "obj_object": str(obj.obj_object),
                    "obj_group": str(obj.obj_group),
                    "obj_material": str(obj.obj_material),
                    "obj_smoothing": str(obj.obj_smoothing),
                    "corner_indices": list(obj.corner_indices or []),
                    "parser_warnings": list(obj.parser_warnings or []),
                }
            )
        return {
            "format_version": 1,
            "saved_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "appearance": str(self.appearance_var.get() or "Dark"),
            "selected_object": str(self.selected_object or ""),
            "objects": objects,
            "materials": materials,
        }

    def _save_scene_json(self):
        path = filedialog.asksaveasfilename(
            title="Salvar cena 3D",
            defaultextension=".mechscene.json",
            initialfile=f"scene_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.mechscene.json",
            filetypes=[("Mechanical scene", "*.mechscene.json"), ("JSON", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            payload = self._scene_payload()
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            self._set_status(f"Cena salva: {path}")
        except Exception as e:
            messagebox.showerror("Salvar cena", str(e))

    def _load_scene_json(self):
        path = filedialog.askopenfilename(
            title="Carregar cena 3D",
            filetypes=[("Mechanical scene", "*.mechscene.json"), ("JSON", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                payload = json.load(f)
            if not isinstance(payload, dict):
                raise ValueError("Formato de cena invalido.")

            loaded: Dict[str, MechanicalObject] = {}
            for item in payload.get("objects", []):
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name", "")).strip()
                if not name:
                    continue
                v = np.asarray(item.get("vertices", []), dtype=float)
                fcs = np.asarray(item.get("faces", []), dtype=int)
                if v.ndim != 2 or v.shape[1] != 3 or fcs.ndim != 2 or fcs.shape[1] != 3:
                    continue
                loaded[name] = MechanicalObject(
                    name=name,
                    vertices=v,
                    faces=fcs,
                    color=str(item.get("color", "#86b6f6")),
                    opacity=float(item.get("opacity", 0.85)),
                    material=str(item.get("material", "Undefined")),
                    source=str(item.get("source", "Scene")),
                    visible=bool(item.get("visible", True)),
                    aedt_name=str(item.get("aedt_name", "")),
                    source_path=str(item.get("source_path", "")),
                    obj_object=str(item.get("obj_object", "")),
                    obj_group=str(item.get("obj_group", "")),
                    obj_material=str(item.get("obj_material", "")),
                    obj_smoothing=str(item.get("obj_smoothing", "")),
                    corner_indices=item.get("corner_indices"),
                    parser_warnings=item.get("parser_warnings"),
                )
            self.objects = loaded

            mats = payload.get("materials", [])
            if isinstance(mats, list) and mats:
                for iid in self.mat_tree.get_children():
                    self.mat_tree.delete(iid)
                for row in mats:
                    if isinstance(row, dict):
                        self._insert_material_row(row)

            self.selected_object = str(payload.get("selected_object", ""))
            self._clear_cad_payload()
            self._refresh_object_tree()
            self._redraw_scene()
            self._apply_theme(str(payload.get("appearance", self.appearance_var.get())))
            self._set_status(f"Cena carregada: {path}")
        except Exception as e:
            messagebox.showerror("Carregar cena", str(e))

    def _clear_scene(self):
        if not self.objects:
            return
        if not messagebox.askyesno("Limpar cena", "Remover todos os objetos da analise mecanica?"):
            return
        self.objects.clear()
        self.selected_object = ""
        self._clear_cad_payload()
        self._refresh_object_tree()
        self._redraw_scene()
        self._set_status("Cena 3D limpa.")

    def _open_pyside6_modeler(self):
        missing = []
        for mod in ("PySide6", "pyvista", "pyvistaqt", "vtk"):
            if importlib.util.find_spec(mod) is None:
                missing.append(mod)
        if missing:
            messagebox.showwarning(
                "Dependencias ausentes",
                "Modeler Pro requer os modulos:\n"
                "  PySide6, pyvista, pyvistaqt, vtk\n\n"
                f"Ausentes: {', '.join(missing)}\n\n"
                "Instale com:\n"
                "pip install pyside6 pyvista pyvistaqt vtk numpy",
            )
            return
        runtime_json = self._write_runtime_payload_for_modeler()
        cmd = [sys.executable, "-m", "mech.ui"]
        if runtime_json:
            cmd.extend(["--runtime-json", runtime_json])
        try:
            subprocess.Popen(cmd, cwd=os.getcwd())
            self._set_status("Modeler Pro (PySide6) iniciado.")
        except Exception as e:
            messagebox.showerror(
                "Falha ao abrir Modeler Pro",
                f"Nao foi possivel iniciar o modulo PySide6.\n\nComando: {' '.join(cmd)}\n\nErro: {e}",
            )

    @staticmethod
    def _json_safe(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, dict):
            return {str(k): MechanicalAnalysisTab._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [MechanicalAnalysisTab._json_safe(v) for v in value]
        return value

    def _write_runtime_payload_for_modeler(self) -> str:
        payload = self._last_cad_payload_runtime
        if not isinstance(payload, dict):
            return ""
        try:
            safe_payload = self._json_safe(payload)
            out_dir = os.path.join(os.getcwd(), "out")
            os.makedirs(out_dir, exist_ok=True)
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(out_dir, f"mechanical_runtime_{stamp}.json")
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                json.dump(safe_payload, f, ensure_ascii=False, indent=2)
            return path
        except Exception as e:
            self._set_status(f"Aviso: falha ao serializar payload para Modeler Pro: {e}")
            return ""

    def _refresh_object_tree(self):
        selected = set(self.obj_tree.selection())
        for iid in self.obj_tree.get_children():
            self.obj_tree.delete(iid)
        for name in sorted(self.objects.keys()):
            obj = self.objects[name]
            values = (
                obj.material,
                int(obj.vertices.shape[0]),
                int(obj.faces.shape[0]),
                obj.source,
                "Yes" if obj.visible else "No",
            )
            self.obj_tree.insert("", "end", iid=name, text=name, values=values)
        for iid in selected:
            if self.obj_tree.exists(iid):
                self.obj_tree.selection_add(iid)
        if self.selected_object and self.obj_tree.exists(self.selected_object):
            self.obj_tree.selection_add(self.selected_object)
            self.obj_tree.focus(self.selected_object)

    def _selected_object_names(self) -> List[str]:
        items = [str(i) for i in self.obj_tree.selection() if self.obj_tree.exists(str(i))]
        if not items and self.selected_object and self.obj_tree.exists(self.selected_object):
            items = [self.selected_object]
        focus = str(self.obj_tree.focus() or "")
        if focus and focus in items:
            ordered = [focus] + [i for i in items if i != focus]
            return ordered
        return items

    def _on_object_select_tree(self, _event=None):
        names = self._selected_object_names()
        if names:
            self.selected_object = names[0]
            self._redraw_scene()

    def _on_object_context(self, event):
        try:
            row = self.obj_tree.identify_row(event.y)
            if row:
                self.obj_tree.selection_set(row)
                self.obj_tree.focus(row)
                self.selected_object = str(row)
            menu = tk.Menu(self, tearoff=0)
            menu.add_command(label="Renomear", command=self._rename_selected_object)
            menu.add_command(label="Duplicar", command=self._duplicate_selected_object)
            menu.add_command(label="Excluir", command=self._delete_selected_objects)
            menu.add_separator()
            menu.add_command(label="Toggle visibilidade", command=self._toggle_selected_visibility)
            menu.add_command(label="Fit", command=self._fit_view)
            menu.tk_popup(event.x_root, event.y_root)
            menu.grab_release()
        except Exception:
            pass

    def _rename_selected_object(self):
        names = self._selected_object_names()
        if not names:
            return
        old = names[0]
        new_name = simpledialog.askstring("Renomear objeto", "Novo nome:", initialvalue=old, parent=self)
        if not new_name:
            return
        new_name = new_name.strip()
        if not new_name:
            return
        if new_name != old and new_name in self.objects:
            messagebox.showwarning("Nome em uso", "Ja existe um objeto com este nome.")
            return
        obj = self.objects.pop(old)
        obj.name = new_name
        self.objects[new_name] = obj
        if self.selected_object == old:
            self.selected_object = new_name
        self._refresh_object_tree()
        self._redraw_scene()
        self._set_status(f"Objeto renomeado: {old} -> {new_name}")

    def _duplicate_selected_object(self):
        names = self._selected_object_names()
        if not names:
            return
        src = self.objects.get(names[0])
        if src is None:
            return
        new_name = _unique_name(f"{src.name}_copy", self.objects.keys())
        dup = MechanicalObject(
            name=new_name,
            vertices=np.array(src.vertices, dtype=float, copy=True),
            faces=np.array(src.faces, dtype=int, copy=True),
            color=src.color,
            opacity=src.opacity,
            material=src.material,
            source=src.source,
            visible=src.visible,
            aedt_name=src.aedt_name,
            source_path=src.source_path,
            obj_object=src.obj_object,
            obj_group=src.obj_group,
            obj_material=src.obj_material,
            obj_smoothing=src.obj_smoothing,
            corner_indices=list(src.corner_indices or []),
            parser_warnings=list(src.parser_warnings or []),
        )
        dup.vertices[:, 0] += 0.05
        dup.vertices[:, 1] += 0.05
        self.objects[new_name] = dup
        self.selected_object = new_name
        self._refresh_object_tree()
        self._redraw_scene()
        self._set_status(f"Objeto duplicado: {new_name}")

    def _delete_selected_objects(self):
        names = self._selected_object_names()
        if not names:
            return
        if not messagebox.askyesno("Excluir objeto(s)", f"Excluir {len(names)} objeto(s) selecionado(s)?"):
            return
        for name in names:
            self.objects.pop(name, None)
        if self.selected_object in names:
            self.selected_object = next(iter(self.objects.keys()), "")
        self._refresh_object_tree()
        self._redraw_scene()
        self._set_status(f"{len(names)} objeto(s) removido(s).")

    def _toggle_selected_visibility(self):
        names = self._selected_object_names()
        if not names:
            return
        for name in names:
            if name in self.objects:
                self.objects[name].visible = not bool(self.objects[name].visible)
        self._refresh_object_tree()
        self._redraw_scene()
        self._set_status("Visibilidade atualizada.")

    def _reset_transform_fields(self):
        for key in ("tx", "ty", "tz", "rx", "ry", "rz"):
            self.t_var[key].set("0.0")
        self.t_var["scale"].set("1.0")

    def _apply_transform_selected(self):
        names = self._selected_object_names()
        if not names:
            messagebox.showwarning("Sem selecao", "Selecione um objeto para aplicar transform.")
            return
        name = names[0]
        obj = self.objects.get(name)
        if obj is None:
            return
        tx = _safe_float(self.t_var["tx"].get(), 0.0)
        ty = _safe_float(self.t_var["ty"].get(), 0.0)
        tz = _safe_float(self.t_var["tz"].get(), 0.0)
        rx = _safe_float(self.t_var["rx"].get(), 0.0)
        ry = _safe_float(self.t_var["ry"].get(), 0.0)
        rz = _safe_float(self.t_var["rz"].get(), 0.0)
        scale = _safe_float(self.t_var["scale"].get(), 1.0)
        scale = max(1e-6, scale)

        center = np.mean(obj.vertices, axis=0)
        v = np.asarray(obj.vertices, dtype=float)
        v = (v - center) * scale + center
        if abs(rx) > 1e-12 or abs(ry) > 1e-12 or abs(rz) > 1e-12:
            rot = _rotation_matrix_xyz(rx, ry, rz)
            v = (v - center) @ rot.T + center
        if abs(tx) > 1e-12 or abs(ty) > 1e-12 or abs(tz) > 1e-12:
            v = v + np.array([tx, ty, tz], dtype=float)
        obj.vertices = v
        self._redraw_scene()
        self._set_status(f"Transform aplicado em {name}.")

    def _add_primitive(self):
        ptype = str(self.prim_var["type"].get() or "Box").strip()
        name = str(self.prim_var["name"].get() or "").strip() or "Base"
        name = _unique_name(name, self.objects.keys())
        cx = _safe_float(self.prim_var["cx"].get(), 0.0)
        cy = _safe_float(self.prim_var["cy"].get(), 0.0)
        cz = _safe_float(self.prim_var["cz"].get(), 0.0)
        w = _safe_float(self.prim_var["w"].get(), 1.0)
        d = _safe_float(self.prim_var["d"].get(), 1.0)
        h = _safe_float(self.prim_var["h"].get(), 1.0)
        r = _safe_float(self.prim_var["r"].get(), 0.5)
        seg = _safe_int(self.prim_var["segments"].get(), 24)

        if ptype == "Box":
            verts, faces = _create_box_mesh(w, d, h, cx, cy, cz)
        elif ptype == "Plate":
            th = max(1e-4, h)
            verts, faces = _create_box_mesh(w, d, th, cx, cy, cz)
        else:
            verts, faces = _create_cylinder_mesh(r, h, seg, cx, cy, cz)
        self.objects[name] = MechanicalObject(
            name=name,
            vertices=verts,
            faces=faces,
            color="#92c17d",
            opacity=0.9,
            material="Undefined",
            source=f"Primitive:{ptype}",
            visible=True,
        )
        self.selected_object = name
        self._refresh_object_tree()
        self._redraw_scene()
        self._set_status(f"Primitiva criada: {name} ({ptype}).")

    def _merge_selected(self):
        names = self._selected_object_names()
        if len(names) < 2:
            messagebox.showwarning("Merge", "Selecione pelo menos 2 objetos para merge.")
            return
        verts_list = []
        faces_list = []
        off = 0
        src_obj = self.objects.get(names[0])
        for name in names:
            obj = self.objects.get(name)
            if obj is None:
                continue
            verts = np.asarray(obj.vertices, dtype=float)
            faces = np.asarray(obj.faces, dtype=int)
            if verts.size == 0 or faces.size == 0:
                continue
            verts_list.append(verts)
            faces_list.append(faces + off)
            off += int(verts.shape[0])
        if not verts_list:
            return
        merged_name = _unique_name(f"{names[0]}_merge", self.objects.keys())
        merged = MechanicalObject(
            name=merged_name,
            vertices=np.vstack(verts_list),
            faces=np.vstack(faces_list),
            color=getattr(src_obj, "color", "#86b6f6"),
            opacity=getattr(src_obj, "opacity", 0.85),
            material=getattr(src_obj, "material", "Undefined"),
            source="Merge",
            visible=True,
        )
        for name in names:
            self.objects.pop(name, None)
        self.objects[merged_name] = merged
        self.selected_object = merged_name
        self._refresh_object_tree()
        self._redraw_scene()
        self._set_status(f"Merge concluido: {merged_name}.")

    def _subtract_selected(self):
        names = self._selected_object_names()
        if len(names) < 2:
            messagebox.showwarning("Subtract", "Selecione dois objetos: A e B para A-B.")
            return
        a = self.objects.get(names[0])
        b = self.objects.get(names[1])
        if a is None or b is None or a.faces.size == 0:
            return
        bmin, bmax = _mesh_bounds(b.vertices)
        tri = a.vertices[a.faces]
        cent = np.mean(tri, axis=1)
        inside = np.all((cent >= bmin) & (cent <= bmax), axis=1)
        keep = ~inside
        kept = int(np.sum(keep))
        if kept <= 0:
            messagebox.showwarning("Subtract", "Operacao removeu toda a geometria de A. Ajuste a selecao.")
            return
        a.faces = a.faces[keep]
        self._refresh_object_tree()
        self._redraw_scene()
        self._set_status(f"Subtract A-B aplicado em {a.name}: {kept} faces restantes.")

    def _intersect_selected(self):
        names = self._selected_object_names()
        if len(names) < 2:
            messagebox.showwarning("Intersect", "Selecione dois objetos para intersecao aproximada.")
            return
        a = self.objects.get(names[0])
        b = self.objects.get(names[1])
        if a is None or b is None or a.faces.size == 0:
            return
        amin, amax = _mesh_bounds(a.vertices)
        bmin, bmax = _mesh_bounds(b.vertices)
        lo = np.maximum(amin, bmin)
        hi = np.minimum(amax, bmax)
        if np.any(lo >= hi):
            messagebox.showinfo("Intersect", "Sem sobreposicao de volumes (bbox).")
            return
        tri = a.vertices[a.faces]
        cent = np.mean(tri, axis=1)
        inside = np.all((cent >= lo) & (cent <= hi), axis=1)
        kept = int(np.sum(inside))
        if kept <= 0:
            messagebox.showinfo("Intersect", "Sem faces na regiao de intersecao aproximada.")
            return
        a.faces = a.faces[inside]
        self._refresh_object_tree()
        self._redraw_scene()
        self._set_status(f"Intersect aplicado em {a.name}: {kept} faces na regiao comum.")

    def _fit_view(self):
        self._redraw_scene()
        self._set_status("Camera ajustada ao conteudo.")

    def _on_plot_press(self, event):
        if event.inaxes != self.ax:
            return
        if event.button not in (1,):
            return
        if event.x is None or event.y is None:
            return
        chosen = ""
        best = 1e30
        tol = 18.0
        for name, (xp, yp) in self._last_pick_pixels.items():
            d2 = (float(event.x) - float(xp)) ** 2 + (float(event.y) - float(yp)) ** 2
            if d2 < best:
                best = d2
                chosen = name
        if chosen and best <= tol * tol:
            self.selected_object = chosen
            self._refresh_object_tree()
            self._redraw_scene()
            self._set_status(f"Objeto selecionado: {chosen}")

    def _redraw_scene(self):
        self.ax.cla()
        self._artists.clear()
        self._last_pick_pixels.clear()
        self.ax.set_title("Analise Mecanica 3D")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")

        vis = [obj for obj in self.objects.values() if obj.visible and obj.vertices.size and obj.faces.size]
        if not vis:
            self.ax.text2D(0.03, 0.96, "Sem objetos. Exporte do AEDT ou carregue OBJ.", transform=self.ax.transAxes)
            self.canvas.draw_idle()
            return

        all_pts = np.vstack([o.vertices for o in vis])
        pmin, pmax = _mesh_bounds(all_pts)
        center = 0.5 * (pmin + pmax)
        span = float(np.max(np.maximum(pmax - pmin, 1e-9)))
        half = 0.55 * span
        self.ax.set_xlim(center[0] - half, center[0] + half)
        self.ax.set_ylim(center[1] - half, center[1] + half)
        self.ax.set_zlim(center[2] - half, center[2] + half)
        try:
            self.ax.set_box_aspect((1.0, 1.0, 1.0))
        except Exception:
            pass

        for obj in vis:
            tris = obj.vertices[obj.faces]
            sel = obj.name == self.selected_object
            edge = "#cc3f3f" if sel else "#202020"
            lw = 0.9 if sel else 0.22
            face_color = "#ffd166" if sel else obj.color
            alpha = 1.0 if sel else float(max(0.05, min(1.0, obj.opacity)))
            poly = Poly3DCollection(tris, linewidths=lw, edgecolors=edge)
            poly.set_facecolor(face_color)
            poly.set_alpha(alpha)
            self.ax.add_collection3d(poly)
            self._artists[obj.name] = poly

        self.canvas.draw_idle()
        self._refresh_pick_cache()

    def _refresh_pick_cache(self):
        self._last_pick_pixels.clear()
        for name, obj in self.objects.items():
            if not obj.visible or obj.vertices.size == 0:
                continue
            c = np.mean(obj.vertices, axis=0)
            x2, y2, _ = proj3d.proj_transform(c[0], c[1], c[2], self.ax.get_proj())
            px, py = self.ax.transData.transform((x2, y2))
            self._last_pick_pixels[name] = (float(px), float(py))

    def _load_default_materials(self):
        for row in DEFAULT_MATERIALS:
            self._insert_material_row(row)
        first = self.mat_tree.get_children()
        if first:
            self.mat_tree.selection_set(first[0])
            self._on_material_tree_select()

    def _material_form_snapshot(self) -> Dict[str, str]:
        data: Dict[str, str] = {}
        for key, _ in MATERIAL_COLUMNS:
            data[key] = str(self.mat_form_vars[key].get() or "").strip()
        if not data.get("name"):
            data["name"] = "Material"
        if not data.get("color"):
            data["color"] = "#cccccc"
        return data

    def _insert_material_row(self, row: Dict[str, str], replace_iid: str = ""):
        data = {k: str(row.get(k, "") or "") for k, _ in MATERIAL_COLUMNS}
        iid = str(data.get("name") or "").strip() or "Material"
        existing = set(self.mat_tree.get_children())
        if replace_iid:
            iid = replace_iid
            if self.mat_tree.exists(iid):
                self.mat_tree.item(iid, values=[data[k] for k, _ in MATERIAL_COLUMNS])
                return
        if iid in existing:
            iid = _unique_name(iid, existing)
        self.mat_tree.insert("", "end", iid=iid, values=[data[k] for k, _ in MATERIAL_COLUMNS])

    def _on_material_tree_select(self, _event=None):
        sel = self.mat_tree.selection()
        if not sel:
            return
        iid = str(sel[0])
        vals = self.mat_tree.item(iid, "values")
        if not vals:
            return
        for (key, _), val in zip(MATERIAL_COLUMNS, vals):
            self.mat_form_vars[key].set(str(val))

    def _add_material_from_form(self):
        row = self._material_form_snapshot()
        self._insert_material_row(row)
        self._set_status(f"Material adicionado: {row.get('name', '')}")

    def _update_material_from_form(self):
        sel = self.mat_tree.selection()
        if not sel:
            self._add_material_from_form()
            return
        old_iid = str(sel[0])
        row = self._material_form_snapshot()
        new_name = str(row.get("name", "")).strip() or old_iid
        if new_name != old_iid and self.mat_tree.exists(new_name):
            messagebox.showwarning("Nome em uso", "Ja existe material com este nome.")
            return
        values = [row[k] for k, _ in MATERIAL_COLUMNS]
        if new_name == old_iid:
            self.mat_tree.item(old_iid, values=values)
            self._set_status(f"Material atualizado: {new_name}")
            return
        self.mat_tree.delete(old_iid)
        self.mat_tree.insert("", "end", iid=new_name, values=values)
        self.mat_tree.selection_set(new_name)
        self._set_status(f"Material renomeado/atualizado: {new_name}")

    def _delete_selected_material(self):
        sel = self.mat_tree.selection()
        if not sel:
            return
        iid = str(sel[0])
        if not messagebox.askyesno("Excluir material", f"Excluir material '{iid}'?"):
            return
        self.mat_tree.delete(iid)
        self._set_status(f"Material removido: {iid}")

    def _import_materials_csv(self):
        path = filedialog.askopenfilename(
            title="Importar tabela de materiais",
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return
        added = 0
        try:
            with open(path, "r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if not row:
                        continue
                    payload = {k: str(row.get(k, "") or "").strip() for k, _ in MATERIAL_COLUMNS}
                    if not payload.get("name"):
                        continue
                    self._insert_material_row(payload)
                    added += 1
        except Exception as e:
            messagebox.showerror("Erro CSV", str(e))
            return
        self._set_status(f"Tabela de materiais importada: {added} item(ns).")

    def _export_materials_csv(self):
        path = filedialog.asksaveasfilename(
            title="Exportar tabela de materiais",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")],
            initialfile="materials_table.csv",
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=[k for k, _ in MATERIAL_COLUMNS])
                writer.writeheader()
                for iid in self.mat_tree.get_children():
                    vals = self.mat_tree.item(iid, "values")
                    if not vals:
                        continue
                    row = {k: str(v) for (k, _), v in zip(MATERIAL_COLUMNS, vals)}
                    writer.writerow(row)
            self._set_status(f"Tabela de materiais exportada: {path}")
        except Exception as e:
            messagebox.showerror("Erro CSV", str(e))

    def _selected_material_row(self) -> Optional[Dict[str, str]]:
        sel = self.mat_tree.selection()
        if sel:
            iid = str(sel[0])
            vals = self.mat_tree.item(iid, "values")
            if vals:
                return {k: str(v) for (k, _), v in zip(MATERIAL_COLUMNS, vals)}
        snapshot = self._material_form_snapshot()
        if snapshot.get("name"):
            return snapshot
        return None

    def _apply_material_to_selected_object(self):
        names = self._selected_object_names()
        if not names:
            messagebox.showwarning("Sem objeto", "Selecione um objeto para aplicar material.")
            return
        material = self._selected_material_row()
        if not material:
            messagebox.showwarning("Sem material", "Selecione/preencha um material.")
            return
        mat_name = str(material.get("name", "")).strip() or "Undefined"
        color = _hex_from_color(material.get("color", "#cccccc"), default="#cccccc")

        applied = 0
        for name in names:
            obj = self.objects.get(name)
            if obj is None:
                continue
            obj.material = mat_name
            obj.color = color
            applied += 1
            if bool(self.push_material_aedt_var.get()):
                self._try_apply_material_to_aedt(obj, mat_name)

        self._refresh_object_tree()
        self._redraw_scene()
        self._set_status(f"Material '{mat_name}' aplicado em {applied} objeto(s).")

    def _try_apply_material_to_aedt(self, obj: MechanicalObject, mat_name: str):
        hfss = self._get_hfss()
        if hfss is None:
            return
        token = str(obj.aedt_name or obj.name or "").strip()
        if not token:
            return
        try:
            m = None
            try:
                m = hfss.modeler[token]
            except Exception:
                m = None
            if m is None:
                try:
                    names = list(getattr(hfss.modeler, "object_names", []))
                except Exception:
                    names = []
                for cand in names:
                    if str(cand).lower() == token.lower():
                        m = hfss.modeler[cand]
                        break
            if m is not None:
                m.material_name = mat_name
        except Exception as e:
            self._set_status(f"Aviso AEDT material ({token}): {e}")


def _register_tab(app, tabview, tab_name: str = "Analise Mecanica"):
    tab = tabview.add(tab_name)
    frame = MechanicalAnalysisTab(tab, app=app)
    frame.pack(fill="both", expand=True)
    return frame
