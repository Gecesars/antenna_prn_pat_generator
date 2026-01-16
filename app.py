# -*- coding: utf-8 -*-
"""
CTk PAT Converter — VRP/HRP (v2)

• Lê CSV/TSV exportado do HFSS: [Freq [GHz], Phi [deg], Theta [deg], 10^(dB10normalize(GainL3X)/20) []]
  → Ignora as 2 primeiras colunas e a 1ª linha (cabeçalho).
• Também aceita formatos TXT genéricos (fallback).
• Reamostra:
    - Vertical (VRP):  -90.0 .. +90.0   passo 0.1° (1801 pts)
    - Horizontal (HRP): -180  .. +180    passo 1°   (361 pts)
• Exporta .pat com cabeçalho exigido e terceira coluna 0.
• Plota: VRP (planar) e HRP (polar).
• Calcula (em cada diagrama, com integração de Simpson no intervalo do corte):
    - Diretividade 2D do corte (adimensional)
        HRP: D₂D = 2π / ∫ P(φ) dφ, φ em rad, P = (E/Emax)²
        VRP: D₂D =  π / ∫ P(θ) dθ, θ em rad, P = (E/Emax)²
    - Largura de feixe a meia potência (HPBW), via cruzamento ±3 dB (|E| = √0.5)
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

# ----------------------------- Parsing ----------------------------- #
NUM_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")

def _is_float(s: str) -> bool:
    return bool(NUM_RE.match(s.strip().replace(",", ".")))

def parse_hfss_csv(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Leitura robusta de CSV/TSV do HFSS (4 colunas).
       Ignora a 1ª linha (header) e as 2 primeiras colunas.
       Retorna (theta_deg, E_over_Emax_linear).
    """
    text = open(path, "r", encoding="utf-8", errors="ignore").read()
    sample = text[:4096]
    dialect = csv.Sniffer().sniff(sample, delimiters=",;\t ")
    reader = csv.reader(StringIO(text), dialect)
    rows = list(reader)
    if not rows:
        raise ValueError("Arquivo vazio.")
    # Se houver header (contém letras), remove a 1ª linha
    if any(ch.isalpha() for ch in "".join(rows[0])):
        rows = rows[1:]
    thetas: List[float] = []
    vals: List[float] = []
    for r in rows:
        if len(r) < 4:
            continue
        t_raw = r[2].strip().replace(",", ".")  # Theta [deg]
        v_raw = r[-1].strip().replace(",", ".")  # última coluna (|E|/Emax linear)
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
    """Fallback flexível p/ TXT/TSV com 2+ números por linha.
       Se houver 3+, ignora o primeiro (índice) e usa: ângulo, valor.
    """
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
    """Tenta HFSS CSV; se falhar, usa parser genérico."""
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
    a_unique, idx = np.unique(a, return_inverse=True)
    v_accum = np.zeros_like(a_unique); counts = np.zeros_like(a_unique)
    for i, vi in zip(idx, v):
        v_accum[i] += vi; counts[i] += 1
    v_mean = v_accum / np.maximum(counts, 1)
    v_tgt = np.interp(tgt, a_unique, v_mean)
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

# ----------------------------- Integração (Simpson) e Métricas ----------------------------- #
def simpson(y: np.ndarray, dx: float) -> float:
    """Simpson composta p/ passo uniforme. Se n for par, usa Simpson em n-1 e trapézio no último."""
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
    """HPBW por cruzamento em -3 dB: |E|=sqrt(0.5)."""
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
    """Diretividade 2D do corte (adimensional) via Simpson.
       P = e^2 (normalizada por pico); D₂D = (span_rad) / ∫ P(ang) d(ang_rad)
       span_deg: 360 para HRP, 180 para VRP.
    """
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

# ----------------------------- PAT Writer ----------------------------- #
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
        self.title("PAT Converter — VRP/HRP (CTk)")
        self.geometry("1200x780")

        # Estado
        self.vertical_path: Optional[str] = None
        self.horizontal_path: Optional[str] = None
        self.base_name_var = tk.StringVar(value="xxx")
               # autor default
        self.author_var = tk.StringVar(value="gecesar")
        self.norm_mode_var = tk.StringVar(value="none")  # none, max, rms
        self.output_dir: Optional[str] = None

        # Dados
        self.v_angles = None; self.v_vals = None
        self.h_angles = None; self.h_vals = None

        # Métricas (labels)
        self.v_hpbw = tk.StringVar(value="HPBW: —")
        self.v_dir  = tk.StringVar(value="D₂D: —")
        self.h_hpbw = tk.StringVar(value="HPBW: —")
        self.h_dir  = tk.StringVar(value="D₂D: —")

        self._build_ui()

    def _build_ui(self):
        # Top controls
        top = ctk.CTkFrame(self)
        top.pack(side=ctk.TOP, fill=ctk.X, padx=12, pady=12)

        ctk.CTkLabel(top, text="Base name:").pack(side=ctk.LEFT, padx=(8, 4))
        ctk.CTkEntry(top, textvariable=self.base_name_var, width=140).pack(side=ctk.LEFT)

        ctk.CTkLabel(top, text="Author:").pack(side=ctk.LEFT, padx=(16, 4))
        ctk.CTkEntry(top, textvariable=self.author_var, width=140).pack(side=ctk.LEFT)

        ctk.CTkLabel(top, text="Normalize:").pack(side=ctk.LEFT, padx=(16, 4))
        ctk.CTkOptionMenu(top, variable=self.norm_mode_var, values=["none", "max", "rms"]).pack(side=ctk.LEFT)

        ctk.CTkButton(top, text="Output dir…", command=self.choose_output_dir).pack(side=ctk.LEFT, padx=(16, 4))
        ctk.CTkButton(top, text="Export PAT", command=self.export_all, fg_color="#22aa66").pack(side=ctk.LEFT, padx=6)

        # Loaders
        loaders = ctk.CTkFrame(self)
        loaders.pack(side=ctk.TOP, fill=ctk.X, padx=12, pady=(0, 8))
        ctk.CTkButton(loaders, text="Load Vertical CSV/TXT (VRP)…", command=self.load_vertical).pack(side=ctk.LEFT, padx=6)
        ctk.CTkButton(loaders, text="Load Horizontal CSV/TXT (HRP)…", command=self.load_horizontal).pack(side=ctk.LEFT, padx=6)
        ctk.CTkButton(loaders, text="Clear", command=self.clear_all).pack(side=ctk.LEFT, padx=6)

        # Plots area
        plots = ctk.CTkFrame(self)
        plots.pack(side=ctk.TOP, fill=ctk.BOTH, expand=True, padx=12, pady=8)

        # Vertical (planar)
        left = ctk.CTkFrame(plots)
        left.pack(side=ctk.LEFT, fill=ctk.BOTH, expand=True, padx=6, pady=6)
        self.fig_v = Figure(figsize=(5.8, 3.8), dpi=100)
        self.ax_v = self.fig_v.add_subplot(111)
        self.ax_v.set_title("Vertical (VRP) — planar")
        self.ax_v.set_xlabel("Theta [deg]")
        self.ax_v.set_ylabel("E/Emax (linear)")
        self.ax_v.grid(True, alpha=0.3)
        self.canvas_v = FigureCanvasTkAgg(self.fig_v, master=left)
        self.canvas_v.get_tk_widget().pack(side=ctk.TOP, fill=ctk.BOTH, expand=True)
        row_v = ctk.CTkFrame(left); row_v.pack(side=ctk.TOP, fill=ctk.X, padx=4, pady=4)
        ctk.CTkLabel(row_v, textvariable=self.v_hpbw).pack(side=ctk.LEFT, padx=6)
        ctk.CTkLabel(row_v, textvariable=self.v_dir ).pack(side=ctk.LEFT, padx=6)

        # Horizontal (polar)
        right = ctk.CTkFrame(plots)
        right.pack(side=ctk.LEFT, fill=ctk.BOTH, expand=True, padx=6, pady=6)
        self.fig_h = Figure(figsize=(5.8, 3.8), dpi=100)
        self.ax_h = self.fig_h.add_subplot(111, projection="polar")
        self.ax_h.set_title("Horizontal (HRP) — polar")
        self.ax_h.grid(True, alpha=0.3)
        self.canvas_h = FigureCanvasTkAgg(self.fig_h, master=right)
        self.canvas_h.get_tk_widget().pack(side=ctk.TOP, fill=ctk.BOTH, expand=True)
        row_h = ctk.CTkFrame(right); row_h.pack(side=ctk.TOP, fill=ctk.X, padx=4, pady=4)
        ctk.CTkLabel(row_h, textvariable=self.h_hpbw).pack(side=ctk.LEFT, padx=6)
        ctk.CTkLabel(row_h, textvariable=self.h_dir ).pack(side=ctk.LEFT, padx=6)

        # Status
        self.status = ctk.CTkLabel(self, text="Ready.")
        self.status.pack(side=ctk.BOTTOM, fill=ctk.X, padx=12, pady=8)

    # -------- Actions -------- #
    def choose_output_dir(self):
        d = filedialog.askdirectory(title="Choose output directory for PAT files")
        if d:
            self.output_dir = d
            self._set_status(f"Output dir: {d}")

    def load_vertical(self):
        path = filedialog.askopenfilename(title="Select vertical table (VRP)",
                                          filetypes=[('CSV/TXT', '*.csv *.tsv *.txt *.dat *.*')])
        if not path:
            return
        try:
            a, v = parse_auto(path)
            self.v_angles, self.v_vals = a, v
            ang_v, val_v = resample_vertical(a, v, norm=self.norm_mode_var.get())
            self._plot_vertical(ang_v, val_v, resampled=True)
            self._metrics_vertical(ang_v, val_v)
            self.vertical_path = path
            self._set_status(f"Vertical loaded: {os.path.basename(path)} — {len(a)} amostras")
        except Exception as e:
            messagebox.showerror("Erro ao carregar VRP", str(e))

    def load_horizontal(self):
        path = filedialog.askopenfilename(title="Select horizontal table (HRP)",
                                          filetypes=[('CSV/TXT', '*.csv *.tsv *.txt *.dat *.*')])
        if not path:
            return
        try:
            a, v = parse_auto(path)
            self.h_angles, self.h_vals = a, v
            ang_h, val_h = resample_horizontal(a, v, norm=self.norm_mode_var.get())
            self._plot_horizontal(ang_h, val_h, resampled=True)
            self._metrics_horizontal(ang_h, val_h)
            self.horizontal_path = path
            self._set_status(f"Horizontal loaded: {os.path.basename(path)} — {len(a)} amostras")
        except Exception as e:
            messagebox.showerror("Erro ao carregar HRP", str(e))

    def clear_all(self):
        self.vertical_path = None
        self.horizontal_path = None
        self.v_angles = self.v_vals = None
        self.h_angles = self.h_vals = None
        self.ax_v.cla(); self.ax_v.set_title("Vertical (VRP) — planar"); self.ax_v.set_xlabel("Theta [deg]"); self.ax_v.set_ylabel("E/Emax (linear)"); self.ax_v.grid(True, alpha=0.3)
        self.canvas_v.draw()
        self.ax_h.cla(); self.ax_h = self.fig_h.add_subplot(111, projection="polar"); self.ax_h.set_title("Horizontal (HRP) — polar"); self.ax_h.grid(True, alpha=0.3)
        self.canvas_h.draw()
        self.v_hpbw.set("HPBW: —"); self.v_dir.set("D₂D: —")
        self.h_hpbw.set("HPBW: —"); self.h_dir.set("D₂D: —")
        self._set_status("Cleared.")

    def export_all(self):
        base = self.base_name_var.get().strip() or "xxx"
        author = self.author_var.get().strip() or "gecesar"
        norm = self.norm_mode_var.get()
        out_dir = self.output_dir or os.path.dirname(self.vertical_path or self.horizontal_path or os.getcwd())

        # Vertical
        if self.v_angles is not None:
            try:
                ang_v, val_v = resample_vertical(self.v_angles, self.v_vals, norm=norm)
                v_path = os.path.join(out_dir, f"{base}_VRP.pat")
                write_pat_vertical(v_path, author, ang_v, val_v)
                self._metrics_vertical(ang_v, val_v)
                self._set_status(f"VRP salvo: {v_path}")
            except Exception as e:
                messagebox.showerror("Erro exportando VRP", str(e))

        # Horizontal
        if self.h_angles is not None:
            try:
                ang_h, val_h = resample_horizontal(self.h_angles, self.h_vals, norm=norm)
                h_path = os.path.join(out_dir, f"{base}_HRP.pat")
                write_pat_horizontal(h_path, author, ang_h, val_h)
                self._metrics_horizontal(ang_h, val_h)
                self._set_status(f"HRP salvo: {h_path}")
            except Exception as e:
                messagebox.showerror("Erro exportando HRP", str(e))

    # -------- Plots -------- #
    def _plot_vertical(self, angles: np.ndarray, values: np.ndarray, resampled: bool = False):
        self.ax_v.cla()
        self.ax_v.set_title("Vertical (VRP) — planar" + (" [resampled]" if resampled else ""))
        self.ax_v.set_xlabel("Theta [deg]")
        self.ax_v.set_ylabel("E/Emax (linear)")
        self.ax_v.grid(True, alpha=0.3)
        self.ax_v.plot(angles, values, linewidth=1.2)
        self.ax_v.set_xlim([-90, 90])
        self.canvas_v.draw()

    def _plot_horizontal(self, angles: np.ndarray, values: np.ndarray, resampled: bool = False):
        self.ax_h.cla()
        self.ax_h = self.fig_h.add_subplot(111, projection="polar")
        self.ax_h.set_title("Horizontal (HRP) — polar" + (" [resampled]" if resampled else ""))
        self.ax_h.grid(True, alpha=0.3)
        ang_wrapped = (angles + 360.0) % 360.0
        theta = np.deg2rad(ang_wrapped)
        self.ax_h.plot(theta, values, linewidth=1.1)
        self.canvas_h.draw()

    # -------- Métricas -------- #
    def _metrics_vertical(self, angles: np.ndarray, values: np.ndarray):
        hpbw = hpbw_deg(angles, values)
        d2d  = directivity_2d_cut(angles, values, span_deg=180.0)
        self.v_hpbw.set(f"HPBW: {hpbw:.2f}°" if math.isfinite(hpbw) else "HPBW: —")
        self.v_dir .set(f"D₂D: {d2d:.3f}"  if math.isfinite(d2d)  else "D₂D: —")

    def _metrics_horizontal(self, angles: np.ndarray, values: np.ndarray):
        hpbw = hpbw_deg(angles, values)
        d2d  = directivity_2d_cut(angles, values, span_deg=360.0)
        self.h_hpbw.set(f"HPBW: {hpbw:.2f}°" if math.isfinite(hpbw) else "HPBW: —")
        self.h_dir .set(f"D₂D: {d2d:.3f}"  if math.isfinite(d2d)  else "D₂D: —")

    # -------- Misc -------- #
    def _set_status(self, text: str):
        self.status.configure(text=text)

if __name__ == "__main__":
    app = PATConverterApp()
    app.mainloop()
