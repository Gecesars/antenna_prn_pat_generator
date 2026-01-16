# -*- coding: utf-8 -*-
"""
CTk PAT Converter — Arquivo | Composição Vertical | Composição Horizontal

• Aba 1 (Arquivo):
    - Lê CSV/TSV do HFSS: [Freq [GHz], Phi [deg], Theta [deg], 10^(dB10normalize(...)/20)]
    Ignora 1ª linha (header) e 2 primeiras colunas; usa Theta + última coluna (|E|/Emax linear).
    - Reamostra:
        VRP: -90..+90 passo 0.1°
        HRP: -180..+180 passo 1°
    - Plota VRP (planar) e HRP (polar) e exporta .pat.

• Aba 2 (Composição Vertical — array linear em z):
    Parâmetros: N, frequência, unidade, fase progressiva β [deg/elem], nível (amplitude), espaçamento vertical d [m].
    Modelo: AF(θ) = Σ_n w · exp{ j · n · (k d sinθ + β) }, k = 2π/λ.
    E_total(θ) = E_elem(θ) * |AF(θ)|.
    Métricas: pico, HPBW, D2D (Simpson sobre P = (E/Emáx)² no span=π).

• Aba 3 (Composição Horizontal — painéis em polígono regular):
    Parâmetros: N painéis, fase progressiva β [deg/painel], nível, espaçamento entre vizinhos s [m],
                passo angular Δφ [deg] (p.ex. 90°), frequência (+ unidade).
    Cálculo CORRETO: Considera a posição espacial de cada painel e calcula o campo resultante
    considerando a diferença de caminho e fase para cada direção.
"""

from __future__ import annotations

import os
import re
import math
import csv
import tkinter as tk
from io import StringIO
from tkinter import filedialog, messagebox
from typing import List, Tuple, Optional

import numpy as np
import customtkinter as ctk
import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ----------------------------- Constantes ----------------------------- #
C0 = 299_792_458.0  # m/s
NUM_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")

def _is_float(s: str) -> bool:
    return bool(NUM_RE.match(s.strip().replace(",", ".")))

# ----------------------------- Parsing ----------------------------- #
def parse_hfss_csv(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Leitura robusta de CSV/TSV do HFSS (4 colunas). Ignora header e 2 primeiras colunas.
    Retorna (theta_deg, E_over_Emax_linear).
    """
    text = open(path, "r", encoding="utf-8", errors="ignore").read()
    sample = text[:4096]
    dialect = csv.Sniffer().sniff(sample, delimiters=",;\t ")
    reader = csv.reader(StringIO(text), dialect)
    rows = list(reader)
    if not rows:
        raise ValueError("Arquivo vazio.")
    # Se a primeira linha tiver letras, é header
    if any(ch.isalpha() for ch in "".join(rows[0])):
        rows = rows[1:]
    thetas: List[float] = []
    vals: List[float] = []
    for r in rows:
        if len(r) < 4:
            continue
        t_raw = r[2].strip().replace(",", ".")   # Theta [deg]
        v_raw = r[-1].strip().replace(",", ".")  # última coluna (E/Emax linear)
        if _is_float(t_raw) and _is_float(v_raw):
            t = float(t_raw)
            v = float(v_raw)
            if math.isfinite(t) and math.isfinite(v):
                thetas.append(t); vals.append(v)
    if not thetas:
        raise ValueError("Falha ao ler colunas Theta e valor (E/Emax) do CSV.")
    a = np.asarray(thetas, dtype=float)
    v = np.asarray(vals, dtype=float)
    order = np.argsort(a)
    return a[order], v[order]

def parse_generic_table(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Fallback p/ TXT/TSV com 2+ números/linha; se 3+, ignora primeiro (índice)."""
    angles: List[float] = []
    vals: List[float] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            normalized = line.replace(",", ".")
            parts = re.split(r"[\t\s;,,]+", normalized)
            nums = [p for p in parts if _is_float(p)]
            if len(nums) < 2:
                continue
            if len(nums) >= 3:
                angle_deg = float(nums[1]); value = float(nums[2])
            else:
                angle_deg = float(nums[0]); value = float(nums[1])
            if math.isfinite(angle_deg) and math.isfinite(value):
                angles.append(angle_deg); vals.append(value)
    if not angles:
        raise ValueError("Nenhum dado numérico válido encontrado no arquivo.")
    a = np.asarray(angles, dtype=float)
    v = np.asarray(vals, dtype=float)
    order = np.argsort(a)
    return a[order], v[order]

def parse_auto(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Tenta HFSS CSV; se falhar, parse genérico."""
    try:
        return parse_hfss_csv(path)
    except Exception:
        return parse_generic_table(path)

# ----------------------------- Reamostragem ----------------------------- #
def _normalize_linear(values: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return values
    if mode == "max":
        m = np.max(values) if values.size else 1.0
        return values / (m if m > 0 else 1.0)
    if mode == "rms":
        rms = float(np.sqrt(np.mean(values**2))) if values.size else 1.0
        return values / (rms if rms > 0 else 1.0)
    return values

def resample_vertical(angles_deg: np.ndarray, values: np.ndarray, norm: str = "none") -> Tuple[np.ndarray, np.ndarray]:
    tgt = np.round(np.arange(-90.0, 90.0 + 1e-9, 0.1), 1)
    mask = (angles_deg >= -90.0) & (angles_deg <= 90.0)
    if not np.any(mask):
        raise ValueError("Tabela vertical não cobre o intervalo [-90, 90] graus.")
    a = angles_deg[mask]; v = values[mask]
    a_u, idx = np.unique(a, return_inverse=True)
    v_acc = np.zeros_like(a_u); cnt = np.zeros_like(a_u)
    for i, vi in zip(idx, v):
        v_acc[i] += vi; cnt[i] += 1
    v_mean = v_acc / np.maximum(cnt, 1)
    v_tgt = np.interp(tgt, a_u, v_mean)
    v_tgt = _normalize_linear(v_tgt, norm)
    return tgt, v_tgt

def _wrap_to_180(a: np.ndarray) -> np.ndarray:
    return (a + 180.0) % 360.0 - 180.0

def resample_horizontal(angles_deg: np.ndarray, values: np.ndarray, norm: str = "none") -> Tuple[np.ndarray, np.ndarray]:
    a = _wrap_to_180(angles_deg); v = values.copy()
    order = np.argsort(a); a = a[order]; v = v[order]
    a_ext = np.concatenate([a, a[:1] + 360.0])
    v_ext = np.concatenate([v, v[:1]])
    tgt = np.arange(-180.0, 181.0, 1.0)
    t_adj = tgt.copy(); t_adj[t_adj < a_ext[0]] += 360.0
    v_tgt = np.interp(t_adj, a_ext, v_ext)
    v_tgt = _normalize_linear(v_tgt, norm)
    return tgt, v_tgt

# ----------------------------- Integração e Métricas ----------------------------- #
def simpson(y: np.ndarray, dx: float) -> float:
    """Simpson composta p/ passo uniforme (ajuste de último intervalo se n par)."""
    n = len(y)
    if n < 2:
        return 0.0
    if n % 2 == 0:
        s = simpson(y[:-1], dx)
        s += 0.5 * dx * (y[-2] + y[-1])
        return s
    s = y[0] + y[-1] + 4.0 * np.sum(y[1:-1:2]) + 2.0 * np.sum(y[2:-2:2])
    return s * dx / 3.0

def hpbw_deg(angles_deg: np.ndarray, e_lin: np.ndarray) -> float:
    """HPBW pelo cruzamento em -3 dB: |E|=sqrt(0.5)."""
    if len(angles_deg) < 3:
        return float("nan")
    e = e_lin / (np.max(e_lin) if np.max(e_lin) > 0 else 1.0)
    thr = math.sqrt(0.5)
    i0 = int(np.argmax(e))
    aL = None; aR = None
    # esquerda
    for i in range(i0, 0, -1):
        if e[i] >= thr and e[i-1] < thr:
            a1, a2 = angles_deg[i-1], angles_deg[i]
            y1, y2 = e[i-1], e[i]
            aL = a1 + (thr - y1) * (a2 - a1) / (y2 - y1)
            break
    # direita
    for i in range(i0, len(e)-1):
        if e[i] >= thr and e[i+1] < thr:
            a1, a2 = angles_deg[i], angles_deg[i+1]
            y1, y2 = e[i], e[i+1]
            aR = a1 + (thr - y1) * (a2 - a1) / (y2 - y1)
            break
    if aL is None or aR is None:
        return float("nan")
    return float(aR - aL)

def directivity_2d_cut(angles_deg: np.ndarray, e_lin: np.ndarray, span_deg: float) -> float:
    """Diretividade 2D do corte: D₂D = span_rad / ∫ P dθ, com P = (E/Emax)² e θ em rad."""
    e = e_lin / (np.max(e_lin) if np.max(e_lin) > 0 else 1.0)
    p = e * e
    ang_rad = np.deg2rad(angles_deg)
    if len(ang_rad) < 2:
        return float("nan")
    dx = float(ang_rad[1] - ang_rad[0])
    integral = simpson(p, dx)
    span_rad = math.radians(span_deg)
    if integral <= 0:
        return float("nan")
    return float(span_rad / integral)

# ----------------------------- Helpers de exportação PAT ----------------------------- #
PAT_HEADER_TEMPLATE = (
    "Edited by {author}\n"
    "98\n"
    "1\n"
    "0 0 0 1 0\n"
    "voltage\n"
)

def write_pat_vertical(path: str, author: str, angles: np.ndarray, values: np.ndarray) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(PAT_HEADER_TEMPLATE.format(author=author))
        for ang, val in zip(angles, values):
            f.write(f"{ang:.1f}\t{val:.4f}\t0\n")

def write_pat_horizontal(path: str, author: str, angles: np.ndarray, values: np.ndarray) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(PAT_HEADER_TEMPLATE.format(author=author))
        for ang, val in zip(angles, values):
            f.write(f"{int(round(ang))}\t{val:.4f}\t0\n")

# ----------------------------- GUI ----------------------------- #
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class PATConverterApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("PAT Converter — Arquivo | Array Vertical | Painéis Horizontais")
        self.geometry("1250x820")

        # Estado global (aba 1)
        self.base_name_var = tk.StringVar(value="xxx")
        self.author_var    = tk.StringVar(value="gecesar")
        self.norm_mode_var = tk.StringVar(value="none")   # none, max, rms
        self.output_dir: Optional[str] = None

        # Dados carregados (aba 1)
        self.v_angles = None; self.v_vals = None
        self.h_angles = None; self.h_vals = None

        # Tabview (3 abas)
        self.tabs = ctk.CTkTabview(self)
        self.tabs.pack(fill=ctk.BOTH, expand=True, padx=10, pady=10)

        self.tab_file = self.tabs.add("Arquivo")
        self.tab_vert = self.tabs.add("Composição Vertical")
        self.tab_horz = self.tabs.add("Composição Horizontal")

        self._build_tab_file()
        self._build_tab_vertical()
        self._build_tab_horizontal()

        # Status
        self.status = ctk.CTkLabel(self, text="Pronto.")
        self.status.pack(side=ctk.BOTTOM, fill=ctk.X, padx=12, pady=8)

    # ==================== ABA 1 — ARQUIVO ==================== #
    def _build_tab_file(self):
        top = ctk.CTkFrame(self.tab_file)
        top.pack(side=ctk.TOP, fill=ctk.X, padx=8, pady=8)

        ctk.CTkLabel(top, text="Base name:").pack(side=ctk.LEFT, padx=(6, 3))
        ctk.CTkEntry(top, textvariable=self.base_name_var, width=140).pack(side=ctk.LEFT)

        ctk.CTkLabel(top, text="Author:").pack(side=ctk.LEFT, padx=(14, 3))
        ctk.CTkEntry(top, textvariable=self.author_var, width=140).pack(side=ctk.LEFT)

        ctk.CTkLabel(top, text="Normalize:").pack(side=ctk.LEFT, padx=(14, 3))
        ctk.CTkOptionMenu(top, variable=self.norm_mode_var, values=["none", "max", "rms"]).pack(side=ctk.LEFT)

        ctk.CTkButton(top, text="Output dir…", command=self.choose_output_dir).pack(side=ctk.LEFT, padx=(14, 4))
        ctk.CTkButton(top, text="Export PAT (aba 1)", command=self.export_all, fg_color="#22aa66").pack(side=ctk.LEFT, padx=6)

        loaders = ctk.CTkFrame(self.tab_file)
        loaders.pack(side=ctk.TOP, fill=ctk.X, padx=8, pady=(0, 8))
        ctk.CTkButton(loaders, text="Carregar VRP (CSV/TXT)…", command=self.load_vertical).pack(side=ctk.LEFT, padx=6)
        ctk.CTkButton(loaders, text="Carregar HRP (CSV/TXT)…", command=self.load_horizontal).pack(side=ctk.LEFT, padx=6)
        ctk.CTkButton(loaders, text="Limpar", command=self.clear_all).pack(side=ctk.LEFT, padx=6)

        # Plots
        plots = ctk.CTkFrame(self.tab_file)
        plots.pack(side=ctk.TOP, fill=ctk.BOTH, expand=True, padx=8, pady=8)

        # Vertical (planar)
        self.fig_v1 = Figure(figsize=(6.2, 4.2), dpi=100)
        self.ax_v1 = self.fig_v1.add_subplot(111)
        self.ax_v1.set_title("Vertical (VRP) — planar")
        self.ax_v1.set_xlabel("Theta [deg]")
        self.ax_v1.set_ylabel("E/Emax (linear)")
        self.ax_v1.grid(True, alpha=0.3)
        self.canvas_v1 = FigureCanvasTkAgg(self.fig_v1, master=plots)
        self.canvas_v1.get_tk_widget().pack(side=ctk.LEFT, fill=ctk.BOTH, expand=True, padx=6, pady=6)

        # Horizontal (polar)
        self.fig_h1 = Figure(figsize=(6.2, 4.2), dpi=100)
        self.ax_h1 = self.fig_h1.add_subplot(111, projection="polar")
        self.ax_h1.set_title("Horizontal (HRP) — polar")
        self.ax_h1.grid(True, alpha=0.3)
        self.canvas_h1 = FigureCanvasTkAgg(self.fig_h1, master=plots)
        self.canvas_h1.get_tk_widget().pack(side=ctk.LEFT, fill=ctk.BOTH, expand=True, padx=6, pady=6)

    def choose_output_dir(self):
        d = filedialog.askdirectory(title="Escolha a pasta de saída para .pat")
        if d:
            self.output_dir = d
            self._set_status(f"Output dir: {d}")

    def load_vertical(self):
        path = filedialog.askopenfilename(title="Selecione VRP (CSV/TXT)",
                                          filetypes=[('CSV/TXT', '*.csv *.tsv *.txt *.dat *.*')])
        if not path:
            return
        try:
            a, v = parse_auto(path)
            self.v_angles, self.v_vals = a, v
            # mostra não reamostrado aqui; export faz reamostragem
            self._plot_vertical_file(a, v)
            self._set_status(f"VRP carregado ({len(a)} amostras): {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Erro ao carregar VRP", str(e))

    def load_horizontal(self):
        path = filedialog.askopenfilename(title="Selecione HRP (CSV/TXT)",
                                          filetypes=[('CSV/TXT', '*.csv *.tsv *.txt *.dat *.*')])
        if not path:
            return
        try:
            a, v = parse_auto(path)
            self.h_angles, self.h_vals = a, v
            self._plot_horizontal_file(a, v)
            self._set_status(f"HRP carregado ({len(a)} amostras): {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Erro ao carregar HRP", str(e))

    def clear_all(self):
        self.v_angles = self.v_vals = None
        self.h_angles = self.h_vals = None
        self.ax_v1.cla(); self.ax_v1.set_title("Vertical (VRP) — planar"); self.ax_v1.set_xlabel("Theta [deg]"); self.ax_v1.set_ylabel("E/Emax (linear)"); self.ax_v1.grid(True, alpha=0.3); self.canvas_v1.draw()
        self.ax_h1.cla(); self.ax_h1 = self.fig_h1.add_subplot(111, projection="polar"); self.ax_h1.set_title("Horizontal (HRP) — polar"); self.ax_h1.grid(True, alpha=0.3); self.canvas_h1.draw()
        self._set_status("Limpo.")

    def export_all(self):
        base = self.base_name_var.get().strip() or "xxx"
        author = self.author_var.get().strip() or "gecesar"
        norm = self.norm_mode_var.get()
        out_dir = self.output_dir or os.getcwd()

        if self.v_angles is not None:
            ang_v, val_v = resample_vertical(self.v_angles, self.v_vals, norm=norm)
            path_v = os.path.join(out_dir, f"{base}_VRP.pat")
            write_pat_vertical(path_v, author, ang_v, val_v)
            self._set_status(f"VRP exportado: {path_v}")

        if self.h_angles is not None:
            ang_h, val_h = resample_horizontal(self.h_angles, self.h_vals, norm=norm)
            path_h = os.path.join(out_dir, f"{base}_HRP.pat")
            write_pat_horizontal(path_h, author, ang_h, val_h)
            self._set_status(f"HRP exportado: {path_h}")

    def _plot_vertical_file(self, angles: np.ndarray, values: np.ndarray):
        self.ax_v1.cla()
        self.ax_v1.set_title("Vertical (VRP) — planar")
        self.ax_v1.set_xlabel("Theta [deg]")
        self.ax_v1.set_ylabel("E/Emax (linear)")
        self.ax_v1.grid(True, alpha=0.3)
        self.ax_v1.plot(angles, values, linewidth=1.2)
        self.ax_v1.set_xlim([-90, 90])
        self.canvas_v1.draw()

    def _plot_horizontal_file(self, angles: np.ndarray, values: np.ndarray):
        self.ax_h1.cla(); self.ax_h1 = self.fig_h1.add_subplot(111, projection="polar")
        self.ax_h1.set_title("Horizontal (HRP) — polar")
        self.ax_h1.grid(True, alpha=0.3)
        ang_wrapped = (angles + 360.0) % 360.0
        theta = np.deg2rad(ang_wrapped)
        self.ax_h1.plot(theta, values, linewidth=1.1)
        self.canvas_h1.draw()

    # ==================== ABA 2 — COMPOSIÇÃO VERTICAL ==================== #
    def _build_tab_vertical(self):
        # Frame principal dividido em entrada e gráfico
        main_frame = ctk.CTkFrame(self.tab_vert)
        main_frame.pack(fill=ctk.BOTH, expand=True, padx=8, pady=8)
        
        # Frame de entrada à esquerda
        input_frame = ctk.CTkFrame(main_frame)
        input_frame.pack(side=ctk.LEFT, fill=ctk.Y, padx=(0, 8), pady=8)
        
        # Frame do gráfico à direita
        plot_frame = ctk.CTkFrame(main_frame)
        plot_frame.pack(side=ctk.RIGHT, fill=ctk.BOTH, expand=True, padx=8, pady=8)

        # Título
        ctk.CTkLabel(input_frame, text="Parâmetros do Array Vertical", 
                    font=ctk.CTkFont(weight="bold")).pack(pady=(8, 12))

        # Parâmetros organizados verticalmente
        self.vert_N      = tk.IntVar(value=4)
        self.vert_freq   = tk.DoubleVar(value=0.9)
        self.vert_funit  = tk.StringVar(value="GHz")
        self.vert_beta   = tk.DoubleVar(value=0.0)
        self.vert_level  = tk.DoubleVar(value=1.0)
        self.vert_space  = tk.DoubleVar(value=0.5)
        self.vert_norm   = tk.StringVar(value="max")

        def create_param_row(parent, label, variable, values=None, width=120):
            row = ctk.CTkFrame(parent)
            row.pack(fill=ctk.X, padx=8, pady=4)
            ctk.CTkLabel(row, text=label, width=100).pack(side=ctk.LEFT)
            if values:
                widget = ctk.CTkOptionMenu(row, variable=variable, values=values, width=width)
            else:
                widget = ctk.CTkEntry(row, textvariable=variable, width=width)
            widget.pack(side=ctk.RIGHT, padx=(8, 0))
            return widget

        create_param_row(input_frame, "N antenas:", self.vert_N)
        
        freq_row = ctk.CTkFrame(input_frame)
        freq_row.pack(fill=ctk.X, padx=8, pady=4)
        ctk.CTkLabel(freq_row, text="Frequência:", width=100).pack(side=ctk.LEFT)
        ctk.CTkEntry(freq_row, textvariable=self.vert_freq, width=80).pack(side=ctk.LEFT, padx=(8, 4))
        ctk.CTkOptionMenu(freq_row, variable=self.vert_funit, values=["Hz","kHz","MHz","GHz"], width=70).pack(side=ctk.LEFT)
        
        create_param_row(input_frame, "β [deg/elem]:", self.vert_beta)
        create_param_row(input_frame, "Nível (amp.):", self.vert_level)
        create_param_row(input_frame, "Esp. d [m]:", self.vert_space)
        create_param_row(input_frame, "Normalizar:", self.vert_norm, ["none","max","rms"])

        # Botões
        btn_frame = ctk.CTkFrame(input_frame)
        btn_frame.pack(fill=ctk.X, padx=8, pady=12)
        ctk.CTkButton(btn_frame, text="Calcular VRP", command=self.compute_vertical_array, 
                     fg_color="#2277cc").pack(side=ctk.TOP, fill=ctk.X, pady=2)
        ctk.CTkButton(btn_frame, text="Exportar .pat", command=self.export_vertical_array, 
                     fg_color="#22aa66").pack(side=ctk.TOP, fill=ctk.X, pady=2)

        # Métricas
        metrics_frame = ctk.CTkFrame(input_frame)
        metrics_frame.pack(fill=ctk.X, padx=8, pady=8)
        ctk.CTkLabel(metrics_frame, text="Métricas", font=ctk.CTkFont(weight="bold")).pack(pady=(4, 8))
        
        self.vert_peak = tk.StringVar(value="Pico: —")
        self.vert_hpbw = tk.StringVar(value="HPBW: —")
        self.vert_d2d  = tk.StringVar(value="D₂D: —")
        
        ctk.CTkLabel(metrics_frame, textvariable=self.vert_peak, font=ctk.CTkFont(size=12)).pack(anchor="w", pady=2)
        ctk.CTkLabel(metrics_frame, textvariable=self.vert_hpbw, font=ctk.CTkFont(size=12)).pack(anchor="w", pady=2)
        ctk.CTkLabel(metrics_frame, textvariable=self.vert_d2d, font=ctk.CTkFont(size=12)).pack(anchor="w", pady=2)

        # Plot
        self.fig_v2 = Figure(figsize=(8, 6), dpi=100)
        self.ax_v2  = self.fig_v2.add_subplot(111)
        self.ax_v2.set_title("VRP Composto - Array Vertical")
        self.ax_v2.set_xlabel("Theta [deg]")
        self.ax_v2.set_ylabel("E/Emax (linear)")
        self.ax_v2.grid(True, alpha=0.3)
        self.canvas_v2 = FigureCanvasTkAgg(self.fig_v2, master=plot_frame)
        self.canvas_v2.get_tk_widget().pack(fill=ctk.BOTH, expand=True, padx=8, pady=8)

        # buffers de resultado
        self.vert_angles = None
        self.vert_values = None

    def _freq_to_hz(self, val: float, unit: str) -> float:
        unit = unit.lower()
        if unit == "hz":  return val
        if unit == "khz": return val * 1e3
        if unit == "mhz": return val * 1e6
        if unit == "ghz": return val * 1e9
        return val

    def compute_vertical_array(self):
        if self.v_angles is None or self.v_vals is None:
            messagebox.showwarning("Dados faltando", "Carregue o VRP na aba Arquivo antes.")
            return
        try:
            # Reamostra o elemento (VRP) na grade alvo
            base_angles, base_vals = resample_vertical(self.v_angles, self.v_vals, norm=self.vert_norm.get())

            # Parâmetros
            N   = int(self.vert_N.get())
            f_hz = self._freq_to_hz(self.vert_freq.get(), self.vert_funit.get())
            lam = C0 / max(f_hz, 1.0)
            k   = 2.0 * math.pi / lam
            beta = math.radians(self.vert_beta.get())      # fase progressiva por elemento (rad)
            d    = float(self.vert_space.get())            # espaçamento [m]
            w    = float(self.vert_level.get())            # amplitude por elemento

            # Array factor: AF(θ) = Σ_{n=0..N-1} w·e^{j·n (k d sinθ + β)}
            th_rad = np.deg2rad(base_angles)
            psi = k * d * np.sin(th_rad) + beta
            n = np.arange(N, dtype=float).reshape(-1,1)    # (N,1)
            af = np.sum(w * np.exp(1j * (n * psi)), axis=0)  # (len(theta),)
            af_mag = np.abs(af)

            # Composição com padrão do elemento
            E_comp = base_vals * af_mag
            if np.max(E_comp) > 0:
                E_comp = E_comp / np.max(E_comp)

            # Métricas
            peak = float(np.max(E_comp))
            hpbw = hpbw_deg(base_angles, E_comp)
            d2d  = directivity_2d_cut(base_angles, E_comp, span_deg=180.0)
            d2d_db = 10.0 * math.log10(d2d) if d2d > 0 else float("nan")

            # Plot
            self.ax_v2.cla()
            self.ax_v2.set_title("VRP Composto - Array Vertical")
            self.ax_v2.set_xlabel("Theta [deg]")
            self.ax_v2.set_ylabel("E/Emax (linear)")
            self.ax_v2.grid(True, alpha=0.3)
            self.ax_v2.plot(base_angles, E_comp, linewidth=1.5, color='blue')
            self.ax_v2.set_xlim([-90, 90])
            self.ax_v2.set_ylim([0, 1.1])
            self.canvas_v2.draw()

            # Atualiza labels e buffers
            self.vert_peak.set(f"Pico: {peak:.3f}")
            self.vert_hpbw.set(f"HPBW: {hpbw:.2f}°" if math.isfinite(hpbw) else "HPBW: —")
            d2d_text = f"D₂D: {d2d:.3f} ({d2d_db:.2f} dB)" if math.isfinite(d2d) else "D₂D: —"
            self.vert_d2d.set(d2d_text)
            self.vert_angles = base_angles
            self.vert_values = E_comp

        except Exception as e:
            messagebox.showerror("Erro (Vertical)", str(e))

    def export_vertical_array(self):
        if self.vert_angles is None or self.vert_values is None:
            messagebox.showwarning("Nada para exportar", "Execute o cálculo da composição vertical primeiro.")
            return
        base = self.base_name_var.get().strip() or "xxx"
        author = self.author_var.get().strip() or "gecesar"
        out_dir = self.output_dir or os.getcwd()
        path_v = os.path.join(out_dir, f"{base}_VRP_composto.pat")
        write_pat_vertical(path_v, author, self.vert_angles, self.vert_values)
        self._set_status(f"VRP composto exportado: {path_v}")

    # ==================== ABA 3 — COMPOSIÇÃO HORIZONTAL ==================== #
    def _build_tab_horizontal(self):
        # Frame principal dividido em entrada e gráfico
        main_frame = ctk.CTkFrame(self.tab_horz)
        main_frame.pack(fill=ctk.BOTH, expand=True, padx=8, pady=8)
        
        # Frame de entrada à esquerda
        input_frame = ctk.CTkFrame(main_frame)
        input_frame.pack(side=ctk.LEFT, fill=ctk.Y, padx=(0, 8), pady=8)
        
        # Frame do gráfico à direita
        plot_frame = ctk.CTkFrame(main_frame)
        plot_frame.pack(side=ctk.RIGHT, fill=ctk.BOTH, expand=True, padx=8, pady=8)

        # Título
        ctk.CTkLabel(input_frame, text="Parâmetros do Array Horizontal", 
                    font=ctk.CTkFont(weight="bold")).pack(pady=(8, 12))

        # Parâmetros organizados verticalmente
        self.horz_N       = tk.IntVar(value=4)
        self.horz_beta    = tk.DoubleVar(value=0.0)
        self.horz_level   = tk.DoubleVar(value=1.0)
        self.horz_spacing = tk.DoubleVar(value=2.0)
        self.horz_stepdeg = tk.DoubleVar(value=90.0)
        self.horz_freq    = tk.DoubleVar(value=0.9)
        self.horz_funit   = tk.StringVar(value="GHz")
        self.horz_norm    = tk.StringVar(value="max")

        def create_param_row(parent, label, variable, values=None, width=120):
            row = ctk.CTkFrame(parent)
            row.pack(fill=ctk.X, padx=8, pady=4)
            ctk.CTkLabel(row, text=label, width=100).pack(side=ctk.LEFT)
            if values:
                widget = ctk.CTkOptionMenu(row, variable=variable, values=values, width=width)
            else:
                widget = ctk.CTkEntry(row, textvariable=variable, width=width)
            widget.pack(side=ctk.RIGHT, padx=(8, 0))
            return widget

        create_param_row(input_frame, "N painéis:", self.horz_N)
        
        freq_row = ctk.CTkFrame(input_frame)
        freq_row.pack(fill=ctk.X, padx=8, pady=4)
        ctk.CTkLabel(freq_row, text="Frequência:", width=100).pack(side=ctk.LEFT)
        ctk.CTkEntry(freq_row, textvariable=self.horz_freq, width=80).pack(side=ctk.LEFT, padx=(8, 4))
        ctk.CTkOptionMenu(freq_row, variable=self.horz_funit, values=["Hz","kHz","MHz","GHz"], width=70).pack(side=ctk.LEFT)
        
        create_param_row(input_frame, "β [deg/painel]:", self.horz_beta)
        create_param_row(input_frame, "Nível (amp.):", self.horz_level)
        create_param_row(input_frame, "Esp. s [m]:", self.horz_spacing)
        create_param_row(input_frame, "Δφ [deg]:", self.horz_stepdeg)
        create_param_row(input_frame, "Normalizar:", self.horz_norm, ["none","max","rms"])

        # Botões
        btn_frame = ctk.CTkFrame(input_frame)
        btn_frame.pack(fill=ctk.X, padx=8, pady=12)
        ctk.CTkButton(btn_frame, text="Calcular HRP", command=self.compute_horizontal_panels, 
                     fg_color="#2277cc").pack(side=ctk.TOP, fill=ctk.X, pady=2)
        ctk.CTkButton(btn_frame, text="Exportar .pat", command=self.export_horizontal_array, 
                     fg_color="#22aa66").pack(side=ctk.TOP, fill=ctk.X, pady=2)

        # Métricas
        metrics_frame = ctk.CTkFrame(input_frame)
        metrics_frame.pack(fill=ctk.X, padx=8, pady=8)
        ctk.CTkLabel(metrics_frame, text="Métricas", font=ctk.CTkFont(weight="bold")).pack(pady=(4, 8))
        
        self.horz_peak = tk.StringVar(value="Pico: —")
        self.horz_hpbw = tk.StringVar(value="HPBW: —")
        self.horz_d2d  = tk.StringVar(value="D₂D: —")
        
        ctk.CTkLabel(metrics_frame, textvariable=self.horz_peak, font=ctk.CTkFont(size=12)).pack(anchor="w", pady=2)
        ctk.CTkLabel(metrics_frame, textvariable=self.horz_hpbw, font=ctk.CTkFont(size=12)).pack(anchor="w", pady=2)
        ctk.CTkLabel(metrics_frame, textvariable=self.horz_d2d, font=ctk.CTkFont(size=12)).pack(anchor="w", pady=2)

        # Plot - EM POLAR
        self.fig_h2 = Figure(figsize=(8, 6), dpi=100)
        self.ax_h2  = self.fig_h2.add_subplot(111, projection='polar')
        self.ax_h2.set_title("HRP Composto - Array Horizontal", pad=20)
        self.ax_h2.grid(True, alpha=0.3)
        self.canvas_h2 = FigureCanvasTkAgg(self.fig_h2, master=plot_frame)
        self.canvas_h2.get_tk_widget().pack(fill=ctk.BOTH, expand=True, padx=8, pady=8)

        # buffers
        self.horz_angles = None
        self.horz_values = None

    def compute_horizontal_panels(self):
        if self.h_angles is None or self.h_vals is None:
            messagebox.showwarning("Dados faltando", "Carregue o HRP na aba Arquivo antes.")
            return
        try:
            # Reamostra o elemento (HRP) na grade alvo
            base_angles, base_vals = resample_horizontal(self.h_angles, self.h_vals, norm=self.horz_norm.get())

            # Parâmetros
            N   = int(self.horz_N.get())
            beta_deg = self.horz_beta.get()
            w    = float(self.horz_level.get())
            s    = float(self.horz_spacing.get())
            dphi_deg = float(self.horz_stepdeg.get())

            f_hz = self._freq_to_hz(self.horz_freq.get(), self.horz_funit.get())
            lam = C0 / max(f_hz, 1.0)
            k   = 2.0 * math.pi / lam

            # CÁLCULO CORRETO: Posições dos painéis em um polígono regular no plano horizontal
            # Cada painel está orientado radialmente (apontando para fora do polígono)
            alpha_m_deg = np.arange(N) * dphi_deg  # Ângulos de posição dos painéis
            alpha_m_rad = np.deg2rad(alpha_m_deg)
            
            # Raio do polígono para que a distância entre painéis adjacentes seja s
            if N > 1:
                R = s / (2 * np.sin(np.pi / N))
            else:
                R = 0  # Caso de um único painel

            # Inicializar o campo composto
            E_comp = np.zeros(len(base_angles), dtype=complex)

            # Para cada ângulo de observação phi (azimute)
            phi_rad = np.deg2rad(base_angles)
            
            for i in range(len(base_angles)):
                phi = phi_rad[i]
                E_total = 0.0 + 0.0j
                
                # Para cada painel no array
                for m in range(N):
                    # Posição do painel m
                    x_m = R * np.cos(alpha_m_rad[m])
                    y_m = R * np.sin(alpha_m_rad[m])
                    
                    # Vetor de direção de observação
                    u_x = np.cos(phi)
                    u_y = np.sin(phi)
                    
                    # Diferença de caminho para o painel m
                    delta_r = x_m * u_x + y_m * u_y
                    
                    # Fase devido à diferença de caminho
                    phase_geom = k * delta_r
                    
                    # Fase progressiva (excitação)
                    phase_excit = np.deg2rad(m * beta_deg)
                    
                    # Ângulo relativo entre a direção de observação e a orientação do painel
                    # Cada painel está orientado na direção radial (alpha_m)
                    rel_angle_deg = (base_angles[i] - alpha_m_deg[m]) % 360
                    if rel_angle_deg > 180:
                        rel_angle_deg -= 360
                    
                    # Diagrama do elemento na direção relativa
                    E_elem = np.interp(rel_angle_deg, base_angles, base_vals)
                    
                    # Contribuição complexa do painel m
                    E_total += w * E_elem * np.exp(1j * (phase_geom + phase_excit))
                
                E_comp[i] = E_total

            # Tomar magnitude e normalizar
            E_comp_mag = np.abs(E_comp)
            if np.max(E_comp_mag) > 0:
                E_comp_mag = E_comp_mag / np.max(E_comp_mag)

            # Métricas
            peak = float(np.max(E_comp_mag))
            hpbw = hpbw_deg(base_angles, E_comp_mag)
            d2d  = directivity_2d_cut(base_angles, E_comp_mag, span_deg=360.0)
            d2d_db = 10.0 * math.log10(d2d) if d2d > 0 else float("nan")

            # Plot POLAR
            self.ax_h2.cla()
            self.ax_h2 = self.fig_h2.add_subplot(111, projection='polar')
            self.ax_h2.set_title("HRP Composto - Array Horizontal", pad=20)
            self.ax_h2.grid(True, alpha=0.3)
            
            # Converter ângulos para radianos e ajustar para plot polar (0° no topo)
            theta_plot = np.deg2rad(base_angles)
            # Rotacionar para ter 0° no topo (convenção de antenas)
            theta_plot = (theta_plot + np.pi/2) % (2*np.pi)
            
            self.ax_h2.plot(theta_plot, E_comp_mag, linewidth=1.5, color='red')
            self.ax_h2.set_theta_zero_location('N')  # 0° no topo
            self.ax_h2.set_theta_direction(-1)       # sentido horário
            self.canvas_h2.draw()

            # Atualiza labels e buffers
            self.horz_peak.set(f"Pico: {peak:.3f}")
            self.horz_hpbw.set(f"HPBW: {hpbw:.2f}°" if math.isfinite(hpbw) else "HPBW: —")
            d2d_text = f"D₂D: {d2d:.3f} ({d2d_db:.2f} dB)" if math.isfinite(d2d) else "D₂D: —"
            self.horz_d2d.set(d2d_text)
            self.horz_angles = base_angles
            self.horz_values = E_comp_mag

        except Exception as e:
            messagebox.showerror("Erro (Horizontal)", str(e))

    def export_horizontal_array(self):
        if self.horz_angles is None or self.horz_values is None:
            messagebox.showwarning("Nada para exportar", "Execute o cálculo da composição horizontal primeiro.")
            return
        base = self.base_name_var.get().strip() or "xxx"
        author = self.author_var.get().strip() or "gecesar"
        out_dir = self.output_dir or os.getcwd()
        path_h = os.path.join(out_dir, f"{base}_HRP_composto.pat")
        write_pat_horizontal(path_h, author, self.horz_angles, self.horz_values)
        self._set_status(f"HRP composto exportado: {path_h}")

    # ----------------------------- Misc ----------------------------- #
    def _set_status(self, text: str):
        self.status.configure(text=text)

if __name__ == "__main__":
    app = PATConverterApp()
    app.mainloop()