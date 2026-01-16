# -*- coding: utf-8 -*-
"""
CTk PAT Converter — Arquivo | Array Vertical | Painéis Horizontais
"""

from __future__ import annotations

import os
import re
import math
import csv
import json
import sqlite3
import datetime
import tkinter as tk
from io import StringIO
from tkinter import filedialog, messagebox
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image
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

# ----------------------------- Validação de Entrada ----------------------------- #
def validate_float(P):
    """Validação para campos de entrada numéricos"""
    if P == "" or P == "-" or P == "+":
        return True
    try:
        float(P)
        return True
    except ValueError:
        return False

def validate_int(P):
    """Validação para campos de entrada inteiros"""
    if P == "":
        return True
    try:
        int(P)
        return True
    except ValueError:
        return False

# ----------------------------- Helpers de exportação PRN ----------------------------- #
def write_prn_file(path: str, name: str, make: str, frequency: float, freq_unit: str,
                  h_width: float, v_width: float, front_to_back: float, gain: float,
                  h_angles: np.ndarray, h_values: np.ndarray,
                  v_angles: np.ndarray, v_values: np.ndarray) -> None:
    """Escreve arquivo PRN com valores em dB de Atenuação.
       Suporta Vertical 0..360 direto (se v_angles cobrir range) ou mapeamento de -90..90.
    """
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"NAME {name}\n")
        f.write(f"MAKE {make}\n")
        f.write(f"FREQUENCY {frequency:.2f} {freq_unit}\n")
        f.write(f"H_WIDTH {h_width:.2f}\n")
        f.write(f"V_WIDTH {v_width:.2f}\n")
        f.write(f"FRONT_TO_BACK {front_to_back:.2f}\n")
        f.write(f"GAIN {gain:.2f} dBi\n")
        f.write("TILT MECHANICAL\n")
        
        def lin_to_atten(val_lin):
            v = max(val_lin, 1e-10)
            db = 20 * math.log10(v)
            return max(-db, 0.0)

        # Dados horizontais
        f.write("HORIZONTAL 360\n")
        for i in range(360):
            angle = i
            lookup = angle if angle <= 180 else angle - 360
            val_lin = np.interp(lookup, h_angles, h_values, period=360)
            val_atten = lin_to_atten(val_lin)
            f.write(f"{i}\t{val_atten:.4f}\n")
        
        # Dados verticais
        f.write("VERTICAL 360\n")
        
        # Check if v_angles covers 360 (e.g. min < 10, max > 350, or encompasses full range)
        v_min, v_max = np.min(v_angles), np.max(v_angles)
        is_360_vertical = (v_max - v_min) > 200 # approx check
        is_elevation_data = (v_min < -10) # check if includes negative elevation (-90)
        
        for i in range(360):
            theta = i # 0..359
            
            if is_360_vertical and not is_elevation_data:
                # Direct interpolation on 0..360 data
                val_lin = np.interp(theta, v_angles, v_values, period=360)
            else:
                # Mapping Elevation (-90..90) to Theta (0..360)
                # 0=Zenith (Elev 90), 90=Horizon (Elev 0), 180=Nadir (Elev -90)
                # We map desired theta 'i' to source elevation
                if i <= 180:
                    src_elev = 90 - i
                else:
                    # 180..360 -> -90..90 (Back side)
                    # i=270 -> Elev 0. i=360 -> Elev 90
                    src_elev = i - 270
                
                # If v_angles is strictly -90..90, interp works
                val_lin = np.interp(src_elev, v_angles, v_values)
                
            val_atten = lin_to_atten(val_lin)
            f.write(f"{i}\t{val_atten:.4f}\n")

# ----------------------------- PRN Parsing ----------------------------- #
def parse_prn(path: str) -> dict:
    """Lê arquivo .PRN. Converte dB Atten -> Linear. Preserva Vertical 0-360."""
    data = {"NAME": "", "MAKE": "", "FREQUENCY": "", "H_WIDTH": "", "V_WIDTH": "",
            "GAIN": "", "TILT": "", "FRONT_TO_BACK": "",
            "h_angles": [], "h_values": [],
            "v_angles": [], "v_values": []} # Using direct destination lists
            
    current_section = "HEADER"
    
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
        
    for line in lines:
        line = line.strip()
        if not line: continue
        parts = line.split(maxsplit=1)
        key = parts[0].upper()
        
        if key == "HORIZONTAL": current_section = "HORIZONTAL"; continue
        elif key == "VERTICAL": current_section = "VERTICAL"; continue
        
        if current_section == "HEADER":
            val = parts[1] if len(parts) > 1 else ""
            if key in data: data[key] = val
        elif current_section == "HORIZONTAL":
            try:
                nums = re.split(r"[\t\s]+", line)
                if len(nums) >= 2:
                    data["h_angles"].append(float(nums[0]))
                    data["h_values"].append(float(nums[1]))
            except: continue
        elif current_section == "VERTICAL":
            try:
                nums = re.split(r"[\t\s]+", line)
                if len(nums) >= 2:
                    data["v_angles"].append(float(nums[0]))
                    data["v_values"].append(float(nums[1]))
            except: continue

    # Convert to Arrays and Normalize
    for k in ["h", "v"]:
        ang = np.array(data[f"{k}_angles"])
        val = np.array(data[f"{k}_values"])
        
        # Detect simple dB (Atten) -> max > 2
        if len(val) > 0 and np.max(val) > 1.5:
            # Linear = 10^(-Atten/20)
            val = np.power(10, -val / 20.0)
            
        # Normalize to Max=1.0
        if len(val) > 0:
            m = np.max(val)
            val = val / (m if m > 0 else 1.0)
            
        data[f"{k}_angles"] = ang
        data[f"{k}_values"] = val

    return data

class PRNViewerModal(ctk.CTkToplevel):
    def __init__(self, master, prn_data: dict, file_path: str):
        super().__init__(master)
        self.title(f"Editor/Visualizador PRN - {os.path.basename(file_path)}")
        self.geometry("1100x700")
        self.prn_data = prn_data
        self.file_path = file_path
        
        # State for plot types: "Polar" or "Planar"
        self.h_plot_type = tk.StringVar(value="Polar")
        self.v_plot_type = tk.StringVar(value="Polar") # Default V to Polar too
        
        # Main layout: Left (Metadata/Stats), Right (Plots)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        self.left_frame = ctk.CTkFrame(self, width=350)
        self.left_frame.grid(row=0, column=0, sticky="nswe", padx=10, pady=10)
        
        self.right_frame = ctk.CTkFrame(self)
        self.right_frame.grid(row=0, column=1, sticky="nswe", padx=10, pady=10)
        
        # --- Left Frame Content: Metadata Editor ---
        ctk.CTkLabel(self.left_frame, text="Metadados do Arquivo", font=ctk.CTkFont(weight="bold", size=16)).pack(pady=10)
        
        self.entries = {}
        fields = ["NAME", "MAKE", "FREQUENCY", "GAIN", "H_WIDTH", "V_WIDTH", "FRONT_TO_BACK", "TILT"]
        
        for f in fields:
            row = ctk.CTkFrame(self.left_frame)
            row.pack(fill="x", padx=5, pady=2)
            ctk.CTkLabel(row, text=f"{f}:", width=120, anchor="w").pack(side="left")
            ent = ctk.CTkEntry(row)
            ent.pack(side="right", expand=True, fill="x")
            ent.insert(0, str(prn_data.get(f, "")))
            self.entries[f] = ent

        # --- Comparison / Calc Stats ---
        ctk.CTkLabel(self.left_frame, text="Cálculos Automáticos", font=ctk.CTkFont(weight="bold", size=16)).pack(pady=(20, 10))
        
        self.stats_text = ctk.CTkTextbox(self.left_frame, height=200)
        self.stats_text.pack(fill="x", padx=5)
        
        self._calculate_and_show_stats()
        
        # Action Buttons
        btn_row = ctk.CTkFrame(self.left_frame)
        btn_row.pack(fill="x", pady=20)
        ctk.CTkButton(btn_row, text="Recalcular", command=self._calculate_and_show_stats).pack(side="left", padx=5, expand=True)
        # ctk.CTkButton(btn_row, text="Salvar PRN", fg_color="green", command=self._save_prn).pack(side="left", padx=5, expand=True) # Todo
        
        # --- Right Frame Content: Plots ---
        # 2 rows, 1 col
        self.right_frame.grid_rowconfigure(0, weight=1)
        self.right_frame.grid_rowconfigure(1, weight=1)
        self.right_frame.grid_columnconfigure(0, weight=1)
        
        # -- Row 0: Horizontal Plot --
        self.frame_h = ctk.CTkFrame(self.right_frame)
        self.frame_h.grid(row=0, column=0, sticky="nswe", padx=5, pady=5)
        
        # Control Bar H
        h_ctrl = ctk.CTkFrame(self.frame_h, height=30)
        h_ctrl.pack(side="top", fill="x", padx=5, pady=2)
        ctk.CTkLabel(h_ctrl, text="Horizontal").pack(side="left")
        ctk.CTkRadioButton(h_ctrl, text="Polar", variable=self.h_plot_type, value="Polar", command=self._update_h_plot).pack(side="right", padx=5)
        ctk.CTkRadioButton(h_ctrl, text="Planar", variable=self.h_plot_type, value="Planar", command=self._update_h_plot).pack(side="right", padx=5)
        
        self.fig_h = Figure(figsize=(4, 3), dpi=100)
        self.canvas_h = FigureCanvasTkAgg(self.fig_h, master=self.frame_h)
        self.canvas_h.get_tk_widget().pack(side="top", fill="both", expand=True)
        # Initial Plot
        self._update_h_plot()
        
        # -- Row 1: Vertical Plot --
        self.frame_v = ctk.CTkFrame(self.right_frame)
        self.frame_v.grid(row=1, column=0, sticky="nswe", padx=5, pady=5)
        
        # Control Bar V
        v_ctrl = ctk.CTkFrame(self.frame_v, height=30)
        v_ctrl.pack(side="top", fill="x", padx=5, pady=2)
        ctk.CTkLabel(v_ctrl, text="Vertical").pack(side="left")
        ctk.CTkRadioButton(v_ctrl, text="Polar", variable=self.v_plot_type, value="Polar", command=self._update_v_plot).pack(side="right", padx=5)
        ctk.CTkRadioButton(v_ctrl, text="Planar", variable=self.v_plot_type, value="Planar", command=self._update_v_plot).pack(side="right", padx=5)
        
        self.fig_v = Figure(figsize=(4, 3), dpi=100)
        self.canvas_v = FigureCanvasTkAgg(self.fig_v, master=self.frame_v)
        self.canvas_v.get_tk_widget().pack(side="top", fill="both", expand=True)
        # Initial Plot
        self._update_v_plot()

    def _calculate_and_show_stats(self):
        # Calculate from Arrays
        h_ang = self.prn_data["h_angles"]
        h_val = self.prn_data["h_values"]
        v_ang = self.prn_data["v_angles"]
        v_val = self.prn_data["v_values"]
        
        # Stats accumulation
        txt = ""
        txt += "--- DIAGRAMA HORIZONTAL ---\n"
        if len(h_val) > 0:
            h_max = np.max(h_val) if np.max(h_val) > 0 else 1.0
            h_norm = h_val / h_max
            calc_h_bw = self._calc_hpbw_generic(h_ang, h_norm)
            txt += f"HPBW Calc: {calc_h_bw:.2f}° / File: {self.entries['H_WIDTH'].get()}\n"
            
            # F/B Ratio check
            # ... (kept simple)
            txt += f"Max Val: {h_max:.4f}\n"

        txt += "\n--- DIAGRAMA VERTICAL ---\n"
        if len(v_val) > 0:
            v_max = np.max(v_val) if np.max(v_val) > 0 else 1.0
            v_norm = v_val / v_max
            calc_v_bw = self._calc_hpbw_generic(v_ang, v_norm)
            txt += f"HPBW Calc: {calc_v_bw:.2f}° / File: {self.entries['V_WIDTH'].get()}\n"
            
            idx_v_peak = np.argmax(v_norm)
            peak_ang_v = v_ang[idx_v_peak]
            txt += f"Peak Angle: {peak_ang_v:.1f}°\n"
        else:
            txt += "Sem dados.\n"

        self.stats_text.delete("0.0", "end")
        self.stats_text.insert("0.0", txt)
        
    def _calc_hpbw_generic(self, ang, val):
        if len(ang) == 0: return 0.0
        idx_max = np.argmax(val)
        ang_max = ang[idx_max]
        ang_shifted = ang - ang_max
        # Handle wrapping closer to 0
        ang_shifted = (ang_shifted + 180) % 360 - 180
        order = np.argsort(ang_shifted)
        ang_s = ang_shifted[order]
        val_s = val[order]
        return hpbw_deg(ang_s, val_s)

    def _update_h_plot(self):
        h_ang = self.prn_data["h_angles"]
        h_val = self.prn_data["h_values"]
        mode = self.h_plot_type.get()
        self.fig_h.clf()
        if mode == "Polar":
            ax = self.fig_h.add_subplot(111, projection="polar")
            ax.set_title("Horizontal (Polar)")
            theta = np.deg2rad(h_ang)
            ax.plot(theta, h_val)
            ax.set_theta_zero_location('N')
            ax.set_theta_direction(-1)
        else:
            ax = self.fig_h.add_subplot(111)
            ax.set_title("Horizontal (Planar)")
            ax.plot(h_ang, h_val)
            ax.set_xlabel("Angle [deg]")
            ax.set_ylabel("Linear")
            ax.grid(True, alpha=0.3)
        self.canvas_h.draw()
        
    def _update_v_plot(self):
        v_ang = self.prn_data["v_angles"]
        v_val = self.prn_data["v_values"]
        mode = self.v_plot_type.get()
        self.fig_v.clf()
        if mode == "Polar":
            ax = self.fig_v.add_subplot(111, projection="polar")
            ax.set_title("Vertical (Polar)")
            theta = np.deg2rad(v_ang)
            ax.plot(theta, v_val, color='orange')
            # 0 at Top (Zenith), 90 at Right (Horizon)
            ax.set_theta_zero_location('N')
            ax.set_theta_direction(-1) 
        else:
            ax = self.fig_v.add_subplot(111)
            ax.set_title("Vertical (Planar)")
            ax.plot(v_ang, v_val, color='orange')
            ax.set_xlabel("Angle [deg]")
            ax.set_ylabel("Linear")
            ax.grid(True, alpha=0.3)
        self.canvas_v.draw()

# ----------------------------- Helpers de exportação PAT ----------------------------- #
def write_pat_vertical_new_format(path: str, description: str, gain: float, num_antennas: int, 
                                angles: np.ndarray, values: np.ndarray, step: int = 1) -> None:
    """Escreve arquivo PAT no novo formato para diagrama vertical"""
    # Converter para dB (valores negativos)
    values_db = 20 * np.log10(np.maximum(values, 1e-10))
    
    # Criar ângulos de 0 a 360 com o passo especificado
    target_angles = np.arange(0, 361, step)
    
    # Para o diagrama vertical, vamos espelhar os dados
    angles_0_180 = angles + 90  # Converter de -90~90 para 0~180
    values_0_180 = values_db
    
    # Espelhar para 180~360
    angles_180_360 = 360 - angles_0_180[::-1]
    values_180_360 = values_0_180[::-1]
    
    # Combinar
    all_angles = np.concatenate([angles_0_180, angles_180_360[1:]])
    all_values = np.concatenate([values_0_180, values_180_360[1:]])
    
    # Remover duplicatas e ordenar
    unique_angles, unique_indices = np.unique(all_angles, return_index=True)
    unique_values = all_values[unique_indices]
    
    # Interpolar para os ângulos alvo
    final_values = np.interp(target_angles, unique_angles, unique_values, 
                           left=unique_values[0], right=unique_values[-1])
    
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"'{description}',{gain:.2f},{num_antennas}\n")
        for ang, val in zip(target_angles, final_values):
            if ang <= 360:  # Garantir que não ultrapasse 360
                f.write(f"{int(ang)},{val:.2f}\n")

def write_pat_horizontal_new_format(path: str, description: str, gain: float, num_antennas: int,
                                  angles: np.ndarray, values: np.ndarray, step: int = 1) -> None:
    """Escreve arquivo PAT no novo formato para diagrama horizontal"""
    # Converter para dB (valores negativos)
    values_db = 20 * np.log10(np.maximum(values, 1e-10))
    
    # Criar ângulos de 0 a 360 com o passo especificado
    target_angles = np.arange(0, 361, step)
    
    # Converter ângulos de -180~180 para 0~360
    source_angles = (angles + 360) % 360
    order = np.argsort(source_angles)
    source_angles_sorted = source_angles[order]
    source_values_sorted = values_db[order]
    
    # Interpolar para os ângulos alvo
    final_values = np.interp(target_angles, source_angles_sorted, source_values_sorted,
                           period=360)
    
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"'{description}',{gain:.2f},{num_antennas}\n")
        for ang, val in zip(target_angles, final_values):
            f.write(f"{int(ang)},{val:.2f}\n")

# ----------------------------- Helpers de exportação PRN ----------------------------- #
def write_prn_file(path: str, name: str, make: str, frequency: float, freq_unit: str,
                  h_width: float, v_width: float, front_to_back: float, gain: float,
                  h_angles: np.ndarray, h_values: np.ndarray,
                  v_angles: np.ndarray, v_values: np.ndarray) -> None:
    """Escreve arquivo PRN no formato especificado (Attenuation dB)"""
    
    # Helper: Linear -> Attenuation dB (0 = Max, Non-Negative)
    # Assumes values are Linear Normalized (0..1)
    def linear_to_atten_db(vals):
        # Prevent log(0)
        safe_v = np.maximum(vals, 1e-10)
        # Gain dB (negative for values < 1)
        # Atten dB = -20*log10(v). Should be >= 0 since v <= 1.
        atten = -20.0 * np.log10(safe_v)
        # Force non-negative to handle any minor float errors or >1 inputs
        return np.maximum(atten, 0.0)

    with open(path, "w", encoding="utf-8", newline="\n") as f:
        # Cabeçalho
        f.write(f"NAME {name}\n")
        f.write(f"MAKE {make}\n")
        f.write(f"FREQUENCY {frequency:.2f} {freq_unit}\n")
        f.write(f"H_WIDTH {h_width:.2f}\n")
        f.write(f"V_WIDTH {v_width:.2f}\n")
        f.write(f"FRONT_TO_BACK {front_to_back:.2f}\n")
        f.write(f"GAIN {gain:.2f} dBd\n") # Changed dBi to dBd as requested
        f.write("TILT MECHANICAL\n")
        
        # Dados horizontais (0..359)
        f.write("HORIZONTAL 360\n")
        
        # Prepare Horizontal Data (Atten dB)
        h_atten = linear_to_atten_db(h_values)
        
        # Interpolate 0..359
        h_ang_norm = (h_angles + 360) % 360
        h_order = np.argsort(h_ang_norm)
        h_ang_sorted = h_ang_norm[h_order]
        h_val_sorted = h_atten[h_order]
        
        for i in range(360):
            val = np.interp(i, h_ang_sorted, h_val_sorted, period=360)
            f.write(f"{i}\t{val:.2f}\n")
        
        # Dados verticais (0..359)
        f.write("VERTICAL 360\n")
        
        # Vertical Logic:
        # User requirement: "Maximo vertical deve estar em 90 e 270 de forma simetrica".
        # Input 'v_values' comes from 'resample_vertical', usually derived from 'parse_auto' or 'parse_hfss_csv'.
        # Typical input: -90..90 or 0..180 (Theta). Often Peak is at 0 or 90.
        # Deep3 VRP plots show Peak at 0 (Planar X axis).
        
        # Step 1: Find peak of input linear data
        p_idx = np.argmax(v_values)
        peak_input_angle = v_angles[p_idx]
        
        # Step 2: Create a 0..360 target array where Peak is at 90 and 270 (Symmetric)
        # We will take the input pattern, shift it so Peak aligns to 90.
        # And ensure symmetry (Mirror 90..270 to 270..90).
        
        # Shift input so Peak is at 0 first (center it)
        v_angles_centered = v_angles - peak_input_angle
        # Now Peak is at 0.
        
        # We want Peak at 90. So Target Angle = Centered Input + 90.
        # Input = Target - 90.
        
        v_atten = linear_to_atten_db(v_values)
        
        for i in range(360):
            # i is target angle (0..359)
            
            # Logic for Symmetry 90/270:
            # Map i to a "distance from peak"
            # Distance from 90: abs(i - 90)
            # Distance from 270: abs(i - 270)
            # We want the pattern to be symmetric around 90/270 axis? 
            # Or usually Front=90, Back=270.
            # If symmetric, Profile(90+delta) == Profile(90-delta).
            # And Profile(270) copy of Profile(90)?
            
            # Use distance to nearest peak (90 or 270)
            # dist_90 = min(abs(i - 90), 360 - abs(i-90)) # circular distance?
            # Actually, standard "Symmetric Dipole" means 0..180 is identical to 180..360?
            # Or 90 is max, 270 is max. 
            # Let's assume the input pattern covers the "Main Lobe" adequately.
            # We query the input pattern using offset from its peak.
            
            # Find angular distance from 90 (Front Peak) and 270 (Back Peak)
            # We assume the input pattern shape defines the shape around BOTH peaks.
            
            # Normalize i to 0..360
            angle_i = i % 360
            
            # Distance to 90
            diff_90 = abs(angle_i - 90)
            # Distance to 270
            diff_270 = abs(angle_i - 270)
            
            # Use the smaller distance to determine "off-axis angle"
            # This effectively makes 90 and 270 identical peaks.
            min_diff = min(diff_90, diff_270)
            
            # Map this 'min_diff' to the input pattern's "off-axis" angle.
            # Since input peak is at 0 (centered), we query: 0 + min_diff (if input has side lobes?)
            # Or just query the input straight? 
            # CAUTION: Input might be Asymmetric (e.g. Tilted). 
            # But user said "SIMETRICA". So we force symmetry by using 'min_diff'.
            # We query the input at 'peak_input_angle + min_diff' (or -min_diff, assuming symmetry of input?)
            # Let's average the input value at +diff and -diff if possible to smooth?
            # Simpler: Query input at (peak_input_angle + min_diff).
            # This assumes input is defined for positive deviations from peak.
            # If input is -90..90 centered at 0. Positive deviation covers 0..90 side.
            
            query_ang = peak_input_angle + min_diff
            
            # Interpolate
            val = np.interp(query_ang, v_angles, v_atten)
            
            f.write(f"{i}\t{val:.2f}\n")

# ----------------------------- GUI ----------------------------- #
ctk.set_appearance_mode("Dark")

# ----------------------------- Database Manager ----------------------------- #
class DatabaseManager:
    # Determine user data directory to avoid "Read-only file system" errors in Program Files
    APP_DIR = os.path.join(os.path.expanduser("~"), ".eftx_converter")
    if not os.path.exists(APP_DIR):
        os.makedirs(APP_DIR, exist_ok=True)
        
    DB_NAME = os.path.join(APP_DIR, "library.db")

    @classmethod
    def init_db(cls):
        # Migration: Check if legacy db exists and move it, or copy default from install dir if needed
        import shutil
        
        # If user DB doesn't exist, try to copy from install location
        if not os.path.exists(cls.DB_NAME):
            # Potential install locations
            candidates = [
                os.path.join(os.getcwd(), "library.db"),
                os.path.join(os.path.dirname(__file__), "library.db")
            ]
            for src in candidates:
                if os.path.exists(src):
                    try:
                        shutil.copy2(src, cls.DB_NAME)
                        print(f"Initialized user DB from {src}")
                        break
                    except Exception as e:
                        print(f"Failed to copy default DB: {e}")
        
        conn = sqlite3.connect(cls.DB_NAME)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS diagrams
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      name TEXT,
                      type TEXT,
                      angles BLOB,
                      values_json TEXT,  -- Changed to JSON text for simple float list storage
                      meta TEXT,
                      thumbnail_path TEXT,
                      added_at TIMESTAMP)''')
        conn.commit()
        conn.close()

    @classmethod
    def add_diagram(cls, pattern_dict: dict) -> int:
        conn = sqlite3.connect(cls.DB_NAME)
        c = conn.cursor()
        
        # Prepare data
        angles_blob = json.dumps(pattern_dict['angles'].tolist())
        values_blob = json.dumps(pattern_dict['values'].tolist())
        meta_json = json.dumps(pattern_dict.get('meta', {}))
        
        c.execute("INSERT INTO diagrams (name, type, angles, values_json, meta, thumbnail_path, added_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                  (pattern_dict['name'], pattern_dict['type'], angles_blob, values_blob, meta_json, "", datetime.datetime.now()))
        
        id_ = c.lastrowid
        conn.commit()
        conn.close()
        return id_

    @classmethod
    def get_all_diagrams(cls) -> List[dict]:
        conn = sqlite3.connect(cls.DB_NAME)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("SELECT * FROM diagrams ORDER BY added_at DESC")
        rows = c.fetchall()
        
        diagrams = []
        for r in rows:
            try:
                angles = np.array(json.loads(r['angles']))
                values = np.array(json.loads(r['values_json']))
                meta = json.loads(r['meta'])
                diagrams.append({
                    "id": r['id'],
                    "name": r['name'],
                    "type": r['type'],
                    "angles": angles,
                    "values": values,
                    "meta": meta
                })
            except Exception as e:
                print(f"Error loading diagram {r['id']}: {e}")
        conn.close()
        return diagrams
        
    @classmethod
    def delete_diagram(cls, id_: int):
        conn = sqlite3.connect(cls.DB_NAME)
        c = conn.cursor()
        c.execute("DELETE FROM diagrams WHERE id=?", (id_,))
        conn.commit()
        conn.close()


# ----------------------------- Helpers: RFS PAT ----------------------------- #
def write_pat_horizontal_rfs(path: str, name: str, gain: float, num: int, 
                            angles: np.ndarray, values: np.ndarray) -> None:
    """Writes PAT in the format seen in 'Mimo_H_POL_HRP.pat' (RFS style)"""
    # Header: 'Name',Gain,Num
    # Data: Angle,Value(dB)
    
    # 1. Convert Linear to dB
    # Handle zeros
    safe_vals = np.maximum(values, 1e-12)
    vals_db = 20 * np.log10(safe_vals)
    
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"'{name}',{gain:.2f},{num}\n")
        
        # 360 degrees
        target = np.arange(0, 360, 1)
        # Interpolate
        # Handle wrapping for interp
        v_interp = np.interp(target, angles, vals_db, period=360)
        
        for i, val in zip(target, v_interp):
            f.write(f"{int(i)},{val:.2f}\n")


def write_pat_adt_format(path: str, description: str, angles: np.ndarray, values: np.ndarray) -> None:
    """
    Writes PAT in RFS 'ADT' format (Voltage/Linear).
    Header (5 lines):
      Edited by Deep3
      98
      1
      0 0 0 1 0
      voltage
    Data:
      Angle  Voltage(Linear)  0
    """
    # Normalize to max 1.0 just in case
    m = np.max(values)
    vals_norm = values / (m if m > 0 else 1.0)
    
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"Edited by Deep3\n")
        f.write("98\n")
        f.write("1\n")
        f.write("0 0 0 1 0\n")
        f.write("voltage\n")
        
        # Sort by angle
        order = np.argsort(angles)
        ang_sorted = angles[order]
        val_sorted = vals_norm[order]
        
        for a, v in zip(ang_sorted, val_sorted):
            f.write(f"{a:.2f}\t{v:.4f}\t0\n")


def parse_rfs_pat_file(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parses RFS-style PAT file (5-line header containing 'voltage' keyword potentially).
    Format:
    Line 1..5: Headers
    Line 6+: Angle  Value(Linear)  0
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
        
    if len(lines) < 6:
        raise ValueError("File too short for RFS PAT format")
        
    # Start parsing from line 6 (index 5)
    angles = []
    values = []
    
    for line in lines[5:]:
        parts = line.split()
        if len(parts) >= 2:
            try:
                ang = float(parts[0])
                val = float(parts[1])
                angles.append(ang)
                values.append(val)
            except:
                continue
                
    if not angles:
        raise ValueError("No valid data found in RFS PAT file")
        
    a = np.asarray(angles)
    v = np.asarray(values)
    
    # Sort by angle just in case
    order = np.argsort(a)
    return a[order], v[order]


# ----------------------------- Robust Parser & Diagrams Tab ----------------------------- #

class RobustPatternParser:
    @staticmethod
    def parse(path: str) -> List[dict]:
        """
        Parses a file and returns a list of patterns found.
        """
        ext = os.path.splitext(path)[1].lower()
        filename = os.path.basename(path)
        patterns = []
        
        # Helper to compute meta
        def calc_meta(type_, ang, val):
            m = {}
            # Ripple (dB)
            # Max Linear is 1.0 (normalized). Min Linear can be small.
            # Work in dB
            safe = np.maximum(val, 1e-9)
            val_db = 20 * np.log10(safe)
            ripple = np.max(val_db) - np.min(val_db)
            m["Ripple"] = f"{ripple:.1f} dB"
            m["SourceFile"] = filename
            return m
        
        try:
            # 1. PRN (RFS/Celwave)
            if ext == ".prn":
                data = parse_prn(path)
                if len(data["h_values"]) > 0:
                    meta = {k: data[k] for k in ["NAME", "MAKE", "FREQUENCY", "GAIN"]}
                    meta.update(calc_meta("H", data["h_angles"], data["h_values"]))
                    patterns.append({
                        "name": "Horizontal", "type": "H",
                        "angles": data["h_angles"], "values": data["h_values"], # Norm Linear
                        "meta": meta
                    })
                if len(data["v_values"]) > 0:
                    meta = {k: data[k] for k in ["NAME", "V_WIDTH", "TILT"]}
                    meta.update(calc_meta("V", data["v_angles"], data["v_values"]))
                    patterns.append({
                        "name": "Vertical", "type": "V",
                        "angles": data["v_angles"], "values": data["v_values"],
                        "meta": meta
                    })
                return patterns 

            # 2. PAT/CSV/TXT/MSI/DAT
            ptype = "Unknown"
            if "HORIZONTAL" in filename.upper() or "_HRP" in filename.upper() or "AZ" in filename.upper() or "_HPOL_HRP" in filename.upper() or "_HPOL" in filename.upper():
                if "_VRP" not in filename.upper(): # Ensure not VRP if HPOL is present
                     ptype = "H"
            
            if "VERTICAL" in filename.upper() or "_VRP" in filename.upper() or "EL" in filename.upper() or "_VPOL_VRP" in filename.upper():
                ptype = "V"
                
            # SPECIAL CHECK: RFS PAT "voltage"
            is_rfs_pat = False
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    head = [next(f) for _ in range(5)]
                if any("voltage" in h.lower() for h in head):
                    is_rfs_pat = True
            except:
                pass
            
            # Parsing Attempt
            ang, val = None, None
            
            if is_rfs_pat:
                try:
                    ang, val = parse_rfs_pat_file(path)
                except:
                    pass
            
            if ang is None:
                try:
                    ang, val = parse_hfss_csv(path)
                except:
                    try:
                        ang, val = parse_generic_table(path)
                    except:
                        pass
            
            if ang is not None and len(val) > 0:
                # Normalization logic
                if np.max(val) > 3.0: # Atten dB check
                    if np.mean(val) > 0: val = np.power(10, -val/20.0)
                    else: val = np.power(10, val/20.0)
                elif np.min(val) < -3.0 and np.max(val) <= 0: # dB norm check
                    val = np.power(10, val/20.0)
                
                m = np.max(val)
                val = val / (m if m > 0 else 1.0)
                
                meta = calc_meta(ptype, ang, val)
                patterns.append({
                    "name": filename, "type": ptype,
                    "angles": ang, "values": val,
                    "meta": meta
                })

        except Exception as e:
            print(f"Skipping {path}: {e}")
            
        return patterns

    @staticmethod
    def _read_columns_robust(path) -> Tuple[np.ndarray, np.ndarray]:
        # Read all lines, look for floats
        angles = []
        values = []
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
            
        start_idx = 0
        # Skip headers if PAT (detected by straight numbers)
        # Scan for first line with 2+ numbers
        for i, line in enumerate(lines):
             parts = re.split(r"[\t\s,]+", line.strip())
             nums = [x for x in parts if validate_float(x)]
             if len(nums) >= 2:
                 # Found data start?
                 # Check if next line also has numbers to confirm
                 if i+1 < len(lines):
                     parts2 = re.split(r"[\t\s,]+", lines[i+1].strip())
                     if len([x for x in parts2 if validate_float(x)]) >= 2:
                         start_idx = i
                         break
        
        for line in lines[start_idx:]:
            parts = re.split(r"[\t\s,]+", line.strip())
            nums = [float(x) for x in parts if validate_float(x)]
            if len(nums) >= 2:
                angles.append(nums[0])
                values.append(nums[1]) # Assume 2nd col is mag
                
        if not angles: raise ValueError("No data found")
        return np.array(angles), np.array(values)


class DiagramThumbnail(ctk.CTkFrame):
    def __init__(self, master, pattern_info: dict, on_click=None, width=300, height=350):
        super().__init__(master, width=width, height=height)
        self.pack_propagate(False)
        self.pattern = pattern_info
        self.on_click = on_click
        
        # Click handler
        self.bind("<Button-1>", self._on_frame_click)
        
        # Meta
        self.lbl_name = ctk.CTkLabel(self, text=f"{pattern_info.get('name', 'Unknown')}", font=("Arial", 11, "bold"))
        self.lbl_name.pack(pady=(5, 0))
        self.lbl_name.bind("<Button-1>", self._on_frame_click)
        
        # Plot
        self.fig = Figure(figsize=(3, 2.5), dpi=80)
        self.ax = self.fig.add_subplot(111, projection='polar' if pattern_info['type'] == 'H' else None)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)
        self.canvas.get_tk_widget().bind("<Button-1>", self._on_frame_click)
        
        self.lbl_stats = ctk.CTkLabel(self, text="...", font=("Arial", 10), justify="left")
        self.lbl_stats.pack(pady=5, padx=5, fill="x")
        self.lbl_stats.bind("<Button-1>", self._on_frame_click)
        
        self._plot()
        self._calc_stats()
        
    def _on_frame_click(self, event):
        if self.on_click:
            self.on_click(self.pattern)
            
    def _plot(self):
        ang = self.pattern['angles']
        val = self.pattern['values']
        
        self.ax.clear()
        if self.pattern['type'] == 'H':
            rads = np.deg2rad(ang)
            self.ax.plot(rads, val, color='#1f77b4')
            self.ax.set_theta_zero_location('N')
            self.ax.set_theta_direction(-1)
            self.ax.set_yticklabels([])
        else:
            self.ax.plot(ang, val, color='#ff7f0e')
            self.ax.grid(True, alpha=0.3)
            
        self.fig.tight_layout()
        self.canvas.draw()
        
    def _calc_stats(self):
        ang = self.pattern['angles']
        val = self.pattern['values']
        if len(val) == 0: return
        
        try: hpbw = hpbw_deg(ang, val) 
        except: hpbw = 0.0
        
        p_idx = np.argmax(val)
        peak_ang = ang[p_idx]
        
        txt = f"HPBW: {hpbw:.1f}° | Peak: {peak_ang:.1f}°"
        meta = self.pattern.get('meta', {})
        if "GAIN" in meta: txt += f"\nGain: {meta['GAIN']}"
        if "Ripple" in meta: txt += f"\nRipple: {meta['Ripple']}"
        
        self.lbl_stats.configure(text=txt)

    def save_image(self, path):
        self.fig.savefig(path, format='png', dpi=100)
    
    def delete_db_entry(self):
        # Implementation to delete from DB if needed via ID
        pass


class DiagramsTab(ctk.CTkFrame):
    def __init__(self, master, load_callback=None):
        super().__init__(master)
        self.load_callback = load_callback
        
        # Init DB
        DatabaseManager.init_db()
        
        self.top_bar = ctk.CTkFrame(self)
        self.top_bar.pack(fill="x", padx=10, pady=10)
        
        self.btn_add = ctk.CTkButton(self.top_bar, text="Adicionar Arquivos...", command=self.add_files)
        self.btn_add.pack(side="left", padx=5)
        
        self.btn_refresh = ctk.CTkButton(self.top_bar, text="Recarregar Biblioteca", command=self.load_library)
        self.btn_refresh.pack(side="left", padx=5)
        
        self.btn_clear = ctk.CTkButton(self.top_bar, text="Limpar Visualização", command=self.clear_view, fg_color="gray")
        self.btn_clear.pack(side="right", padx=5)
        
        # --- Dual Scrollable Area ---
        self.container = ctk.CTkFrame(self)
        self.container.pack(fill="both", expand=True, padx=10, pady=10)
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)
        
        # Use simple tk.Canvas for scrolling
        bg_color = self._apply_appearance_mode(ctk.ThemeManager.theme["CTkFrame"]["fg_color"])
        self.canvas = tk.Canvas(self.container, highlightthickness=0, bg=bg_color)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        
        self.vsb = ctk.CTkScrollbar(self.container, orientation="vertical", command=self.canvas.yview)
        self.vsb.grid(row=0, column=1, sticky="ns")
        
        self.hsb = ctk.CTkScrollbar(self.container, orientation="horizontal", command=self.canvas.xview)
        self.hsb.grid(row=1, column=0, sticky="ew")
        
        self.canvas.configure(yscrollcommand=self.vsb.set, xscrollcommand=self.hsb.set)
        
        # Inner Frame
        self.scroll = ctk.CTkFrame(self.canvas, fg_color="transparent")
        self.canvas_window = self.canvas.create_window((0, 0), window=self.scroll, anchor="nw")
        
        self.thumbnails = []
        self.grid_cols = 3
        
        # Events
        self.scroll.bind("<Configure>", self._on_inner_frame_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        
        # Bind MouseWheel to canvas
        self.canvas.bind("<Enter>", self._bind_mouse_scroll)
        self.canvas.bind("<Leave>", self._unbind_mouse_scroll)

        # Auto Load
        self.load_library()
        
    def _on_inner_frame_configure(self, event):
        # Update scrollregion to fit inner frame
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        
    def _on_canvas_configure(self, event):
        # Handle responsive resize of columns
        w = event.width
        # Determine columns
        # Conservative: 320 thumb + 20 pad
        new_cols = max(1, w // 340)
        if new_cols != self.grid_cols:
            self.grid_cols = new_cols
            self._regrid()
    
    def _bind_mouse_scroll(self, event):
        self.canvas.bind_all("<MouseWheel>", self._on_mouse_scroll)
        self.canvas.bind_all("<Button-4>", self._on_mouse_scroll)
        self.canvas.bind_all("<Button-5>", self._on_mouse_scroll)
        # Shift+Wheel for horizontal
        self.canvas.bind_all("<Shift-MouseWheel>", self._on_mouse_scroll_h)

    def _unbind_mouse_scroll(self, event):
        self.canvas.unbind_all("<MouseWheel>")
        self.canvas.unbind_all("<Button-4>")
        self.canvas.unbind_all("<Button-5>")
        self.canvas.unbind_all("<Shift-MouseWheel>")

    def _on_mouse_scroll(self, event):
        # Vertical
        if event.num == 4 or event.delta > 0:
            self.canvas.yview_scroll(-1, "units")
        elif event.num == 5 or event.delta < 0:
            self.canvas.yview_scroll(1, "units")
            
    def _on_mouse_scroll_h(self, event):
        # Horizontal
        if event.delta > 0:
            self.canvas.xview_scroll(-1, "units")
        else:
            self.canvas.xview_scroll(1, "units")
            
    def _regrid(self):
        for idx, thumb in enumerate(self.thumbnails):
            thumb.grid_forget()
            row = idx // self.grid_cols
            col = idx % self.grid_cols
            thumb.grid(row=row, column=col, padx=10, pady=10)
            
    def add_files(self):
        files = filedialog.askopenfilenames(title="Selecione Arquivos", 
                                            filetypes=[("All Files", "*.*"), ("PRN", "*.prn"), ("PAT", "*.pat"), ("CSV", "*.csv")])
        if not files: return
        
        count = 0
        for f in files:
            pats = RobustPatternParser.parse(f)
            for p in pats:
                if len(p['values']) > 0:
                    # Save to DB
                    DatabaseManager.add_diagram(p)
                    count += 1
        
        if count > 0:
            self.load_library()
            messagebox.showinfo("Sucesso", f"{count} diagramas importados para a biblioteca.")
    
    def load_library(self):
        self.clear_view()
        diagrams = DatabaseManager.get_all_diagrams()
        
        for p in diagrams:
            self.add_thumbnail(p)
            
    def add_thumbnail(self, pattern):
        thumb = DiagramThumbnail(self.scroll, pattern, on_click=self.load_callback)
        self.thumbnails.append(thumb)
        
        # Also bind scroll events to the thumbnail elements
        def bind_recursive(w):
            w.bind("<MouseWheel>", self._on_mouse_scroll)
            w.bind("<Button-4>", self._on_mouse_scroll)
            w.bind("<Button-5>", self._on_mouse_scroll)
            for child in w.winfo_children():
                bind_recursive(child)
        bind_recursive(thumb)

        idx = len(self.thumbnails) - 1
        row = idx // self.grid_cols
        col = idx % self.grid_cols
        thumb.grid(row=row, column=col, padx=10, pady=10)
        
    def clear_view(self):
        for t in self.thumbnails:
            t.destroy()
        self.thumbnails = []


class PATConverterApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("PAT Converter — Arquivo | Array Vertical | Painéis Horizontais")
        self.geometry("1400x900")
        
        # Header (Logo + Help)
        self._build_header()

        # Estado global (aba 1)
        self.base_name_var = tk.StringVar(value="T_END_FEED_8_Hpol")
        self.author_var    = tk.StringVar(value="gecesar")
        self.norm_mode_var = tk.StringVar(value="none")   # none, max, rms
        self.output_dir: Optional[str] = None

        # Dados carregados (aba 1)
        self.v_angles = None; self.v_vals = None
        self.h_angles = None; self.h_vals = None

        # Tabview (4 abas)
        self.tabs = ctk.CTkTabview(self)
        self.tabs.pack(fill=ctk.BOTH, expand=True, padx=10, pady=10)

        self.tab_file = self.tabs.add("Arquivo")
        self.tab_vert = self.tabs.add("Composição Vertical")
        self.tab_horz = self.tabs.add("Composição Horizontal")
        self.tab_diag = self.tabs.add("Diagramas (Batch)")

        self._build_tab_file()
        self._build_tab_vertical()
        self._build_tab_horizontal()
        
        # Build Diagram Tab with Callback
        self.diagrams_view = DiagramsTab(self.tab_diag, load_callback=self.load_from_library)
        self.diagrams_view.pack(fill="both", expand=True)

        # Status
        self.status = ctk.CTkLabel(self, text="Pronto.")
        self.status.pack(side=ctk.BOTTOM, fill=ctk.X, padx=12, pady=8)

    def _build_header(self):
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(side="top", fill="x", padx=10, pady=(10, 0))
        
        # Logo
        try:
            if os.path.exists("eftx-logo.png"):
                pil = Image.open("eftx-logo.png")
                # Resize keeping aspect ratio, height=50
                ratio = pil.width / pil.height
                h = 50
                w = int(h * ratio)
                self.logo_image = ctk.CTkImage(light_image=pil, dark_image=pil, size=(w, h))
                ctk.CTkLabel(header, text="", image=self.logo_image).pack(side="left", padx=10)
            else:
                ctk.CTkLabel(header, text="EFTX", font=("Arial", 24, "bold")).pack(side="left", padx=10)
        except Exception as e:
            print(f"Error loading logo: {e}")
            ctk.CTkLabel(header, text="EFTX", font=("Arial", 24, "bold")).pack(side="left", padx=10)
            
        # Title/Subtitle
        ctk.CTkLabel(header, text="Conversor & Biblioteca de Diagramas", font=("Arial", 16)).pack(side="left", padx=10)
        
        # Help Button
        ctk.CTkButton(header, text="Ajuda / Workflow", command=self.show_help, width=120, fg_color="#444444").pack(side="right", padx=10)

    def show_help(self):
        w = ctk.CTkToplevel(self)
        w.title("Workflow de Uso")
        w.geometry("600x500")
        
        txt = ctk.CTkTextbox(w, wrap="word")
        txt.pack(fill="both", expand=True, padx=10, pady=10)
        
        info = """
### EDITOR DE DIAGRAMAS - FLUXO DE TRABALHO

1. MÓDULO ARQUIVO (Aba 1)
   - Utilize esta aba para visualizar e exportar diagramas individuais.
   - **Importar**: Carregue arquivos .CSV (formato HFSS) ou .TXT para VRP (Vertical) e HRP (Horizontal).
   - **Visualizar**: Os gráficos mostram a visualização Planar (VRP) e Polar (HRP).
   - **Exportar**:
     - **.PAT (Standard)**: Formato padrão NSMA.
     - **.PAT (ADT)**: Novo formato RFS Voltage.
     - **.PRN**: Exporta o conjunto VRP+HRP combinados.
     - Certifique-se de preencher os metadados no fundo da tela (Freq, Gain, etc) antes de exportar .PRN.

2. BIBLIOTECA BATCH (Aba 4 - Diagramas)
   - **Adicionar**: Selecione múltiplos arquivos (.pat, .prn, .csv) de uma vez.
   - O sistema detecta automaticamente se é Vertical ou Horizontal.
   - **Persistência**: Os arquivos ficam salvos num banco de dados local (library.db).
   - **Usar**: Clique em qualquer miniatura para carregar o diagrama para a Aba 1 (Arquivo) para edição/exportação.
   - **Redimensionável**: A grade de diagramas se ajusta à largura da janela.

3. COMPOSIÇÃO DE ARRAYS (Abas 2 e 3)
   - Ferramentas avançadas para simular arrays verticais e painéis horizontais.
   - Exporte o resultado da simulação diretamente.
        """
        txt.insert("0.0", info)
        txt.configure(state="disabled")

    def load_from_library(self, pattern: dict):
        """Callback when a thumbnail is clicked"""
        try:
            ptype = pattern['type']
            ang = pattern['angles']
            val = pattern['values']
            name = pattern['name']
            
            if ptype == 'V':
                self.v_angles, self.v_vals = ang, val
                self._plot_vertical_file(ang, val)
                self._set_status(f"Carregado da Biblioteca: {name} (Vertical)")
                # Switch tab
                self.tabs.set("Arquivo")
            elif ptype == 'H':
                self.h_angles, self.h_vals = ang, val
                self._plot_horizontal_file(ang, val)
                self._set_status(f"Carregado da Biblioteca: {name} (Horizontal)")
                self.tabs.set("Arquivo")
            else:
                # Ask user? Assume H?
                self.h_angles, self.h_vals = ang, val
                self._plot_horizontal_file(ang, val)
                self._set_status(f"Carregado da Biblioteca: {name} (Assumido Horizontal)")
                self.tabs.set("Arquivo")
                
        except Exception as e:
            messagebox.showerror("Erro ao carregar", str(e))

    # ... (rest of PATConverterApp)


    # ==================== ABA 1 — ARQUIVO ==================== #
    def _build_tab_file(self):
        # Container principal com scroll para garantir visualização em telas menores
        # Usaremos CTkScrollableFrame para a aba inteira se necessário, 
        # mas como temos plots grandes, melhor dividir em seções compactas.
        
        # 1. Top Configuration Bar (Configurações Globais)
        top = ctk.CTkFrame(self.tab_file)
        top.pack(side=ctk.TOP, fill=ctk.X, padx=8, pady=5)

        ctk.CTkLabel(top, text="Base Name:").pack(side=ctk.LEFT, padx=(5, 2))
        ctk.CTkEntry(top, textvariable=self.base_name_var, width=130).pack(side=ctk.LEFT, padx=2)

        ctk.CTkLabel(top, text="Author:").pack(side=ctk.LEFT, padx=(10, 2))
        ctk.CTkEntry(top, textvariable=self.author_var, width=100).pack(side=ctk.LEFT, padx=2)

        ctk.CTkLabel(top, text="Norm:").pack(side=ctk.LEFT, padx=(10, 2))
        ctk.CTkOptionMenu(top, variable=self.norm_mode_var, values=["none", "max", "rms"], width=80).pack(side=ctk.LEFT, padx=2)

        ctk.CTkButton(top, text="Output dir...", command=self.choose_output_dir, width=90).pack(side=ctk.LEFT, padx=(15, 5))
        
        # Botões de Exportação Global no Topo (para fácil acesso)
        ctk.CTkButton(top, text="Export .PAT (All)", command=self.export_all_pat, fg_color="#22aa66", width=110).pack(side=ctk.RIGHT, padx=5)
        ctk.CTkButton(top, text="Export .PRN (All)", command=self.export_all_prn, fg_color="#aa6622", width=110).pack(side=ctk.RIGHT, padx=5)


        # 2. Main Content Area (Plots + Loaders) - Middle
        middle = ctk.CTkFrame(self.tab_file)
        middle.pack(side=ctk.TOP, fill=ctk.BOTH, expand=True, padx=8, pady=5)
        
        # --- LEFT: Vertical ---
        frame_v = ctk.CTkFrame(middle)
        frame_v.pack(side=ctk.LEFT, fill=ctk.BOTH, expand=True, padx=(0, 4), pady=0)
        
        ctk.CTkLabel(frame_v, text="Vertical (VRP)", font=ctk.CTkFont(size=14, weight="bold")).pack(pady=2)
        
        # Plot
        self.fig_v1 = Figure(figsize=(4, 3), dpi=90)
        self.ax_v1 = self.fig_v1.add_subplot(111)
        self.ax_v1.set_title("Planar View")
        self.ax_v1.set_xlabel("Theta")
        self.ax_v1.set_ylabel("Linear")
        self.ax_v1.grid(True, alpha=0.3)
        self.canvas_v1 = FigureCanvasTkAgg(self.fig_v1, master=frame_v)
        self.canvas_v1.get_tk_widget().pack(fill=ctk.BOTH, expand=True, padx=2, pady=2)
        
        # Controls V
        cv = ctk.CTkFrame(frame_v)
        cv.pack(fill=ctk.X, pady=2)
        ctk.CTkButton(cv, text="Load VRP...", command=self.load_vertical, height=24).pack(side=ctk.LEFT, padx=2, expand=True, fill=ctk.X)
        ctk.CTkButton(cv, text="Exp VRP (.pat)", command=lambda: self.export_single_pat("V", "PAT"), fg_color="green", height=24).pack(side=ctk.LEFT, padx=2, expand=True, fill=ctk.X)
        ctk.CTkButton(cv, text="Exp VRP (ADT)", command=lambda: self.export_single_pat("V", "ADT"), fg_color="#33cc99", height=24).pack(side=ctk.LEFT, padx=2, expand=True, fill=ctk.X)

        # --- RIGHT: Horizontal ---
        frame_h = ctk.CTkFrame(middle)
        frame_h.pack(side=ctk.RIGHT, fill=ctk.BOTH, expand=True, padx=(4, 0), pady=0)
        
        ctk.CTkLabel(frame_h, text="Horizontal (HRP)", font=ctk.CTkFont(size=14, weight="bold")).pack(pady=2)
        
        # Plot
        self.fig_h1 = Figure(figsize=(4, 3), dpi=90)
        self.ax_h1 = self.fig_h1.add_subplot(111, projection="polar")
        self.ax_h1.set_title("Polar View")
        self.ax_h1.grid(True, alpha=0.3)
        self.canvas_h1 = FigureCanvasTkAgg(self.fig_h1, master=frame_h)
        self.canvas_h1.get_tk_widget().pack(fill=ctk.BOTH, expand=True, padx=2, pady=2)
        
        # Controls H
        ch = ctk.CTkFrame(frame_h)
        ch.pack(fill=ctk.X, pady=2)
        ctk.CTkButton(ch, text="Load HRP...", command=self.load_horizontal, height=24).pack(side=ctk.LEFT, padx=2, expand=True, fill=ctk.X)
        ctk.CTkButton(ch, text="Exp HRP (.pat)", command=lambda: self.export_single_pat("H", "PAT"), fg_color="green", height=24).pack(side=ctk.LEFT, padx=2, expand=True, fill=ctk.X)
        ctk.CTkButton(ch, text="Exp HRP (ADT)", command=lambda: self.export_single_pat("H", "ADT"), fg_color="#33cc99", height=24).pack(side=ctk.LEFT, padx=2, expand=True, fill=ctk.X)


        # 3. Bottom: Metadata + PRN Utils (Compact)
        bottom = ctk.CTkFrame(self.tab_file)
        bottom.pack(side=ctk.BOTTOM, fill=ctk.X, padx=8, pady=5)
        
        # Header for PRN
        ctk.CTkLabel(bottom, text="PRN Export Metadata", font=ctk.CTkFont(size=12, weight="bold")).pack(anchor="w", padx=5)
        
        # Grid for inputs
        grid_f = ctk.CTkFrame(bottom)
        grid_f.pack(fill=ctk.X, padx=2, pady=2)
        
        # Row 1
        r1 = ctk.CTkFrame(grid_f, fg_color="transparent")
        r1.pack(fill=ctk.X, pady=1)
        ctk.CTkLabel(r1, text="Name:", width=50).pack(side=ctk.LEFT)
        self.prn_name = ctk.CTkEntry(r1, width=120); self.prn_name.pack(side=ctk.LEFT, padx=2); self.prn_name.insert(0, "SLOT_END_FEED_8_Vpol")
        
        ctk.CTkLabel(r1, text="Make:", width=50).pack(side=ctk.LEFT, padx=(10,0))
        self.prn_make = ctk.CTkEntry(r1, width=120); self.prn_make.pack(side=ctk.LEFT, padx=2); self.prn_make.insert(0, "EFTX")
        
        ctk.CTkLabel(r1, text="Freq:", width=40).pack(side=ctk.LEFT, padx=(10,0))
        self.prn_freq = ctk.CTkEntry(r1, width=60); self.prn_freq.pack(side=ctk.LEFT, padx=2); self.prn_freq.insert(0, "635")
        self.prn_freq_unit = ctk.CTkOptionMenu(r1, values=["MHz", "GHz"], width=60); self.prn_freq_unit.pack(side=ctk.LEFT, padx=2)

        ctk.CTkButton(r1, text="Util: View PRN", command=self.load_prn, height=24, fg_color="purple", width=80).pack(side=ctk.RIGHT, padx=5)

        # Row 2
        r2 = ctk.CTkFrame(grid_f, fg_color="transparent")
        r2.pack(fill=ctk.X, pady=1)
        
        ctk.CTkLabel(r2, text="Gain:", width=50).pack(side=ctk.LEFT)
        self.prn_gain = ctk.CTkEntry(r2, width=60); self.prn_gain.pack(side=ctk.LEFT, padx=2); self.prn_gain.insert(0, "10.80")
        
        ctk.CTkLabel(r2, text="H_Width:", width=60).pack(side=ctk.LEFT, padx=(10,0))
        self.prn_h_width = ctk.CTkEntry(r2, width=60); self.prn_h_width.pack(side=ctk.LEFT, padx=2); self.prn_h_width.insert(0, "0.00")
        
        ctk.CTkLabel(r2, text="V_Width:", width=60).pack(side=ctk.LEFT, padx=(5,0))
        self.prn_v_width = ctk.CTkEntry(r2, width=60); self.prn_v_width.pack(side=ctk.LEFT, padx=2); self.prn_v_width.insert(0, "6.30")
        
        ctk.CTkLabel(r2, text="F/B:", width=40).pack(side=ctk.LEFT, padx=(5,0))
        self.prn_fb_ratio = ctk.CTkEntry(r2, width=60); self.prn_fb_ratio.pack(side=ctk.LEFT, padx=2); self.prn_fb_ratio.insert(0, "25.00")

        ctk.CTkButton(r2, text="Util: Clear", command=self.clear_all, height=24, fg_color="red", width=80).pack(side=ctk.RIGHT, padx=5)

    def choose_output_dir(self):
        d = filedialog.askdirectory(title="Escolha a pasta de saída para .pat")
        if d:
            self.output_dir = d
            self._set_status(f"Output dir: {d}")

    def export_single_pat(self, type_, fmt):
        """
        Exporta apenas um dos diagramas (V ou H) da aba Arquivo.
        type_: "V" ou "H"
        fmt: "PAT" (Standard) ou "ADT" (RFS Voltage)
        """
        if type_ == "V":
            ang, val = self.v_angles, self.v_vals
            name_suffix = "_VRP"
            desc_suffix = " Vertical Pattern"
        else:
            ang, val = self.h_angles, self.h_vals
            name_suffix = "_HRP"
            desc_suffix = " Horizontal Pattern"
            
        if ang is None or val is None:
            messagebox.showwarning("Aviso", f"Nenhum diagrama {type_} carregado.")
            return

        base = self.base_name_var.get().strip() or "output"
        out_dir = self.output_dir or os.getcwd()
        fname = f"{base}{name_suffix}"
        
        if fmt == "ADT":
            fname += "_ADT.pat"
        else:
            fname += ".pat"
            
        path = os.path.join(out_dir, fname)
        
        try:
            if fmt == "ADT":
                write_pat_adt_format(path, base, ang, val)
                self._set_status(f"Exportado {type_} (ADT): {path}")
            else:
                # Standard PAT
                # Use default gain=0, N=1 for single element export
                if type_ == "V":
                    write_pat_vertical_new_format(path, base+desc_suffix, 0.0, 1, ang, val)
                else:
                    write_pat_horizontal_new_format(path, base+desc_suffix, 0.0, 1, ang, val)
                self._set_status(f"Exportado {type_} (PAT): {path}")
                
        except Exception as e:
            messagebox.showerror("Erro Export", str(e))
        # Botões de exportação para aba 1
        export_frame = ctk.CTkFrame(top)
        export_frame.pack(side=ctk.LEFT, padx=(20, 0))
        ctk.CTkButton(export_frame, text="Export .PAT", command=self.export_all_pat, fg_color="#22aa66", width=100).pack(side=ctk.LEFT, padx=2)
        ctk.CTkButton(export_frame, text="Export .PRN", command=self.export_all_prn, fg_color="#aa6622", width=100).pack(side=ctk.LEFT, padx=2)

        loaders = ctk.CTkFrame(self.tab_file)
        loaders.pack(side=ctk.TOP, fill=ctk.X, padx=8, pady=(0, 8))
        ctk.CTkButton(loaders, text="Carregar VRP (CSV/TXT)…", command=self.load_vertical).pack(side=ctk.LEFT, padx=6)
        ctk.CTkButton(loaders, text="Carregar HRP (CSV/TXT)…", command=self.load_horizontal).pack(side=ctk.LEFT, padx=6)
        ctk.CTkButton(loaders, text="Carregar PRN (Visualizador)…", command=self.load_prn, fg_color="#6622aa").pack(side=ctk.LEFT, padx=6)
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

        # Seção de metadados para PRN
        prn_meta_frame = ctk.CTkFrame(self.tab_file)
        prn_meta_frame.pack(side=ctk.TOP, fill=ctk.X, padx=8, pady=8)
        
        ctk.CTkLabel(prn_meta_frame, text="Metadados para .PRN", font=ctk.CTkFont(weight="bold")).pack(pady=(4, 8))
        
        meta_grid = ctk.CTkFrame(prn_meta_frame)
        meta_grid.pack(fill=ctk.X, padx=8, pady=4)
        
        # Linha 1
        row1 = ctk.CTkFrame(meta_grid)
        row1.pack(fill=ctk.X, pady=2)
        ctk.CTkLabel(row1, text="Nome:", width=80).pack(side=ctk.LEFT, padx=4)
        self.prn_name = ctk.CTkEntry(row1, width=150)
        self.prn_name.pack(side=ctk.LEFT, padx=4)
        self.prn_name.insert(0, "SLOT_END_FEED_8_Vpol")
        
        ctk.CTkLabel(row1, text="Fabricante:", width=80).pack(side=ctk.LEFT, padx=4)
        self.prn_make = ctk.CTkEntry(row1, width=150)
        self.prn_make.pack(side=ctk.LEFT, padx=4)
        self.prn_make.insert(0, "EFTX")
        
        # Linha 2
        row2 = ctk.CTkFrame(meta_grid)
        row2.pack(fill=ctk.X, pady=2)
        ctk.CTkLabel(row2, text="Frequência:", width=80).pack(side=ctk.LEFT, padx=4)
        self.prn_freq = ctk.CTkEntry(row2, width=80)
        self.prn_freq.pack(side=ctk.LEFT, padx=4)
        self.prn_freq.insert(0, "635")
        
        self.prn_freq_unit = ctk.CTkOptionMenu(row2, values=["MHz", "GHz"], width=70)
        self.prn_freq_unit.pack(side=ctk.LEFT, padx=4)
        
        ctk.CTkLabel(row2, text="Ganho:", width=80).pack(side=ctk.LEFT, padx=4)
        self.prn_gain = ctk.CTkEntry(row2, width=80)
        self.prn_gain.pack(side=ctk.LEFT, padx=4)
        self.prn_gain.insert(0, "10.80")
        
        # Linha 3
        row3 = ctk.CTkFrame(meta_grid)
        row3.pack(fill=ctk.X, pady=2)
        ctk.CTkLabel(row3, text="H_WIDTH:", width=80).pack(side=ctk.LEFT, padx=4)
        self.prn_h_width = ctk.CTkEntry(row3, width=80)
        self.prn_h_width.pack(side=ctk.LEFT, padx=4)
        self.prn_h_width.insert(0, "0.00")
        
        ctk.CTkLabel(row3, text="V_WIDTH:", width=80).pack(side=ctk.LEFT, padx=4)
        self.prn_v_width = ctk.CTkEntry(row3, width=80)
        self.prn_v_width.pack(side=ctk.LEFT, padx=4)
        self.prn_v_width.insert(0, "6.30")
        
        ctk.CTkLabel(row3, text="F/B Ratio:", width=80).pack(side=ctk.LEFT, padx=4)
        self.prn_fb_ratio = ctk.CTkEntry(row3, width=80)
        self.prn_fb_ratio.pack(side=ctk.LEFT, padx=4)
        self.prn_fb_ratio.insert(0, "25.00")

    def choose_output_dir(self):
        d = filedialog.askdirectory(title="Escolha a pasta de saída para .pat")
        if d:
            self.output_dir = d
            self._set_status(f"Output dir: {d}")

    def export_single_pat(self, type_, fmt):
        """
        Exporta apenas um dos diagramas (V ou H) da aba Arquivo.
        type_: "V" ou "H"
        fmt: "PAT" (Standard) ou "ADT" (RFS Voltage)
        """
        if type_ == "V":
            ang, val = self.v_angles, self.v_vals
            name_suffix = "_VRP"
            desc_suffix = " Vertical Pattern"
        else:
            ang, val = self.h_angles, self.h_vals
            name_suffix = "_HRP"
            desc_suffix = " Horizontal Pattern"
            
        if ang is None or val is None:
            messagebox.showwarning("Aviso", f"Nenhum diagrama {type_} carregado.")
            return

        base = self.base_name_var.get().strip() or "output"
        out_dir = self.output_dir or os.getcwd()
        fname = f"{base}{name_suffix}"
        
        if fmt == "ADT":
            fname += "_ADT.pat"
        else:
            fname += ".pat"
            
        path = os.path.join(out_dir, fname)
        
        try:
            if fmt == "ADT":
                write_pat_adt_format(path, base, ang, val)
                self._set_status(f"Exportado {type_} (ADT): {path}")
            else:
                # Standard PAT
                # Use default gain=0, N=1 for single element export
                if type_ == "V":
                    write_pat_vertical_new_format(path, base+desc_suffix, 0.0, 1, ang, val)
                else:
                    write_pat_horizontal_new_format(path, base+desc_suffix, 0.0, 1, ang, val)
                self._set_status(f"Exportado {type_} (PAT): {path}")
                
        except Exception as e:
            messagebox.showerror("Erro Export", str(e))

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

    def load_prn(self):
        path = filedialog.askopenfilename(title="Selecione arquivo PRN",
                                          filetypes=[('PRN Files', '*.prn'), ('All Files', '*.*')])
        if not path:
            return
        try:
            data = parse_prn(path)
            # Open Modal
            PRNViewerModal(self, data, path)
            self._set_status(f"Editor PRN aberto: {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Erro ao abrir PRN", str(e))

    def clear_all(self):
        self.v_angles = self.v_vals = None
        self.h_angles = self.h_vals = None
        self.ax_v1.cla(); self.ax_v1.set_title("Vertical (VRP) — planar"); self.ax_v1.set_xlabel("Theta [deg]"); self.ax_v1.set_ylabel("E/Emax (linear)"); self.ax_v1.grid(True, alpha=0.3); self.canvas_v1.draw()
        self.ax_h1.cla(); self.ax_h1 = self.fig_h1.add_subplot(111, projection="polar"); self.ax_h1.set_title("Horizontal (HRP) — polar"); self.ax_h1.grid(True, alpha=0.3); self.canvas_h1.draw()
        self._set_status("Limpo.")

    def export_all_pat(self):
        """Exporta arquivos .PAT para VRP e HRP"""
        base = self.base_name_var.get().strip() or "xxx"
        author = self.author_var.get().strip() or "gecesar"
        norm = self.norm_mode_var.get()
        out_dir = self.output_dir or os.getcwd()

        if self.v_angles is not None:
            ang_v, val_v = resample_vertical(self.v_angles, self.v_vals, norm=norm)
            path_v = os.path.join(out_dir, f"{base}_VRP.pat")
            write_pat_vertical_new_format(path_v, f"{base}_VRP", 0.0, 1, ang_v, val_v)
            self._set_status(f"VRP .PAT exportado: {path_v}")

        if self.h_angles is not None:
            ang_h, val_h = resample_horizontal(self.h_angles, self.h_vals, norm=norm)
            path_h = os.path.join(out_dir, f"{base}_HRP.pat")
            write_pat_horizontal_new_format(path_h, f"{base}_HRP", 0.0, 1, ang_h, val_h)
            self._set_status(f"HRP .PAT exportado: {path_h}")

    def export_all_prn(self):
        """Exporta arquivo .PRN combinando VRP e HRP"""
        if self.v_angles is None or self.h_angles is None:
            messagebox.showwarning("Dados incompletos", "Carregue ambos VRP e HRP para exportar .PRN")
            return
            
        try:
            # Coletar metadados
            name = self.prn_name.get().strip() or "ANTENA_FM"
            make = self.prn_make.get().strip() or "RFS"
            
            def safe_float(v, default=0.0):
                try: 
                    return float(v)
                except: 
                    return default
            
            freq = safe_float(self.prn_freq.get(), 99.50)
            freq_unit = self.prn_freq_unit.get()
            gain = safe_float(self.prn_gain.get(), 2.77)
            h_width = safe_float(self.prn_h_width.get(), 65.0)
            v_width = safe_float(self.prn_v_width.get(), 45.0)
            fb_ratio = safe_float(self.prn_fb_ratio.get(), 25.0)
            
            # Reamostrar dados
            v_angles, v_vals = resample_vertical(self.v_angles, self.v_vals, self.norm_mode_var.get())
            h_angles, h_vals = resample_horizontal(self.h_angles, self.h_vals, self.norm_mode_var.get())
            
            # Pedir local para salvar
            file_path = filedialog.asksaveasfilename(
                defaultextension=".prn",
                filetypes=[("PRN files", "*.prn"), ("All files", "*.*")],
                title="Salvar arquivo .PRN"
            )
            
            if not file_path:
                return
                
            # Escrever arquivo
            write_prn_file(file_path, name, make, freq, freq_unit, h_width, v_width, 
                          fb_ratio, gain, h_angles, h_vals, v_angles, v_vals)
            
            self._set_status(f"Arquivo .PRN exportado: {file_path}")
            
        except Exception as e:
            messagebox.showerror("Erro ao exportar .PRN", str(e))

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

        # Validação para campos numéricos
        vcmd_float = (self.register(validate_float), '%P')
        vcmd_int = (self.register(validate_int), '%P')

        # Parâmetros organizados verticalmente
        self.vert_N      = tk.StringVar(value="4")
        self.vert_freq   = tk.StringVar(value="0.9")
        self.vert_funit  = tk.StringVar(value="GHz")
        self.vert_beta   = tk.StringVar(value="0.0")
        self.vert_level  = tk.StringVar(value="1.0")
        self.vert_space  = tk.StringVar(value="0.5")
        self.vert_norm   = tk.StringVar(value="max")

        # Parâmetros de exportação
        self.vert_desc   = tk.StringVar(value="Array Vertical")
        self.vert_gain   = tk.StringVar(value="0.0")
        self.vert_step   = tk.StringVar(value="1")
        self.vert_num_antennas = tk.StringVar(value="4")

        def create_param_row(parent, label, variable, values=None, width=120, validate=None):
            row = ctk.CTkFrame(parent)
            row.pack(fill=ctk.X, padx=8, pady=4)
            ctk.CTkLabel(row, text=label, width=100).pack(side=ctk.LEFT)
            if values:
                widget = ctk.CTkOptionMenu(row, variable=variable, values=values, width=width)
            else:
                if validate == "float":
                    widget = ctk.CTkEntry(row, textvariable=variable, width=width, validate="key", validatecommand=vcmd_float)
                elif validate == "int":
                    widget = ctk.CTkEntry(row, textvariable=variable, width=width, validate="key", validatecommand=vcmd_int)
                else:
                    widget = ctk.CTkEntry(row, textvariable=variable, width=width)
            widget.pack(side=ctk.RIGHT, padx=(8, 0))
            return widget

        create_param_row(input_frame, "N antenas:", self.vert_N, validate="int")
        
        freq_row = ctk.CTkFrame(input_frame)
        freq_row.pack(fill=ctk.X, padx=8, pady=4)
        ctk.CTkLabel(freq_row, text="Frequência:", width=100).pack(side=ctk.LEFT)
        ctk.CTkEntry(freq_row, textvariable=self.vert_freq, width=80, validate="key", 
                    validatecommand=vcmd_float).pack(side=ctk.LEFT, padx=(8, 4))
        ctk.CTkOptionMenu(freq_row, variable=self.vert_funit, values=["Hz","kHz","MHz","GHz"], width=70).pack(side=ctk.LEFT)
        
        create_param_row(input_frame, "β [deg/elem]:", self.vert_beta, validate="float")
        create_param_row(input_frame, "Nível (amp.):", self.vert_level, validate="float")
        create_param_row(input_frame, "Esp. d [m]:", self.vert_space, validate="float")
        create_param_row(input_frame, "Normalizar:", self.vert_norm, ["none","max","rms"])

        # Seção de exportação
        export_frame = ctk.CTkFrame(input_frame)
        export_frame.pack(fill=ctk.X, padx=8, pady=12)
        ctk.CTkLabel(export_frame, text="Exportação", font=ctk.CTkFont(weight="bold")).pack(pady=(4, 8))
        
        create_param_row(export_frame, "Descrição:", self.vert_desc)
        create_param_row(export_frame, "Ganho [dB]:", self.vert_gain, validate="float")
        create_param_row(export_frame, "Nº Antenas:", self.vert_num_antennas, validate="int")
        create_param_row(export_frame, "Passo [deg]:", self.vert_step, ["1","2","3","4","5"])

        # Botões de exportação
        btn_frame = ctk.CTkFrame(input_frame)
        btn_frame.pack(fill=ctk.X, padx=8, pady=12)
        ctk.CTkButton(btn_frame, text="Calcular VRP", command=self.compute_vertical_array, 
                     fg_color="#2277cc").pack(side=ctk.TOP, fill=ctk.X, pady=2)
        
        export_btn_frame = ctk.CTkFrame(btn_frame)
        export_btn_frame.pack(fill=ctk.X, pady=2)
        ctk.CTkButton(export_btn_frame, text="Export .PAT", command=self.export_vertical_array_pat, 
                     fg_color="#22aa66", width=100).pack(side=ctk.LEFT, padx=2)
        ctk.CTkButton(export_btn_frame, text="Export .PRN", command=self.export_vertical_array_prn, 
                     fg_color="#aa6622", width=100).pack(side=ctk.LEFT, padx=2)

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

    def _get_float_value(self, var: tk.StringVar, default: float = 0.0) -> float:
        """Obtém valor float de StringVar com tratamento de erro"""
        try:
            return float(var.get())
        except (ValueError, tk.TclError):
            return default

    def _get_int_value(self, var: tk.StringVar, default: int = 0) -> int:
        """Obtém valor int de StringVar com tratamento de erro"""
        try:
            return int(var.get())
        except (ValueError, tk.TclError):
            return default

    def compute_vertical_array(self):
        if self.v_angles is None or self.v_vals is None:
            messagebox.showwarning("Dados faltando", "Carregue o VRP na aba Arquivo antes.")
            return
        try:
            # Reamostra o elemento (VRP) na grade alvo
            base_angles, base_vals = resample_vertical(self.v_angles, self.v_vals, norm=self.vert_norm.get())

            # Parâmetros com tratamento de erro
            N   = self._get_int_value(self.vert_N, 4)
            freq_val = self._get_float_value(self.vert_freq, 0.9)
            f_hz = self._freq_to_hz(freq_val, self.vert_funit.get())
            lam = C0 / max(f_hz, 1.0)
            k   = 2.0 * math.pi / lam
            beta = math.radians(self._get_float_value(self.vert_beta, 0.0))
            d    = self._get_float_value(self.vert_space, 0.5)
            w    = self._get_float_value(self.vert_level, 1.0)

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

    def export_vertical_array_pat(self):
        if self.vert_angles is None or self.vert_values is None:
            messagebox.showwarning("Nada para exportar", "Execute o cálculo da composição vertical primeiro.")
            return
        base = self.base_name_var.get().strip() or "xxx"
        out_dir = self.output_dir or os.getcwd()
        
        description = self.vert_desc.get() or f"{base}_VRP_composto"
        gain = self._get_float_value(self.vert_gain, 0.0)
        num_antennas = self._get_int_value(self.vert_num_antennas, 4)
        step = self._get_int_value(self.vert_step, 1)
        
        path_v = os.path.join(out_dir, f"{base}_VRP_composto.pat")
        write_pat_vertical_new_format(path_v, description, gain, num_antennas, 
                                    self.vert_angles, self.vert_values, step)
        self._set_status(f"VRP composto .PAT exportado: {path_v}")

    def export_vertical_array_prn(self):
        if self.vert_angles is None or self.vert_values is None:
            messagebox.showwarning("Nada para exportar", "Execute o cálculo da composição vertical primeiro.")
            return
            
        if self.h_angles is None or self.h_vals is None:
            messagebox.showwarning("Dados incompletos", "Para exportar .PRN é necessário ter o HRP carregado na aba Arquivo.")
            return
            
        try:
            # Coletar metadados
            name = self.prn_name.get().strip() or "ANTENA_FM"
            make = self.prn_make.get().strip() or "RFS"
            freq = float(self.prn_freq.get() or "99.50")
            freq_unit = self.prn_freq_unit.get()
            gain = float(self.prn_gain.get() or "2.77")
            h_width = float(self.prn_h_width.get() or "65.0")
            v_width = float(self.prn_v_width.get() or "45.0")
            fb_ratio = float(self.prn_fb_ratio.get() or "25.0")
            
            # Reamostrar dados horizontais (usando dados da aba Arquivo)
            h_angles, h_vals = resample_horizontal(self.h_angles, self.h_vals, self.norm_mode_var.get())
            
            # Pedir local para salvar
            file_path = filedialog.asksaveasfilename(
                defaultextension=".prn",
                filetypes=[("PRN files", "*.prn"), ("All files", "*.*")],
                title="Salvar arquivo .PRN"
            )
            
            if not file_path:
                return
                
            # Escrever arquivo
            write_prn_file(file_path, name, make, freq, freq_unit, h_width, v_width, 
                          fb_ratio, gain, h_angles, h_vals, self.vert_angles, self.vert_values)
            
            self._set_status(f"Arquivo .PRN exportado: {file_path}")
            
        except Exception as e:
            messagebox.showerror("Erro ao exportar .PRN", str(e))

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

        # Validação para campos numéricos
        vcmd_float = (self.register(validate_float), '%P')
        vcmd_int = (self.register(validate_int), '%P')

        # Parâmetros organizados verticalmente
        self.horz_N       = tk.StringVar(value="4")
        self.horz_beta    = tk.StringVar(value="0.0")
        self.horz_level   = tk.StringVar(value="1.0")
        self.horz_spacing = tk.StringVar(value="2.0")
        self.horz_stepdeg = tk.StringVar(value="90.0")
        self.horz_freq    = tk.StringVar(value="0.9")
        self.horz_funit   = tk.StringVar(value="GHz")
        self.horz_norm    = tk.StringVar(value="max")

        # Parâmetros de exportação
        self.horz_desc   = tk.StringVar(value="Array Horizontal")
        self.horz_gain   = tk.StringVar(value="0.0")
        self.horz_step   = tk.StringVar(value="1")
        self.horz_num_antennas = tk.StringVar(value="4")

        def create_param_row(parent, label, variable, values=None, width=120, validate=None):
            row = ctk.CTkFrame(parent)
            row.pack(fill=ctk.X, padx=8, pady=4)
            ctk.CTkLabel(row, text=label, width=100).pack(side=ctk.LEFT)
            if values:
                widget = ctk.CTkOptionMenu(row, variable=variable, values=values, width=width)
            else:
                if validate == "float":
                    widget = ctk.CTkEntry(row, textvariable=variable, width=width, validate="key", validatecommand=vcmd_float)
                elif validate == "int":
                    widget = ctk.CTkEntry(row, textvariable=variable, width=width, validate="key", validatecommand=vcmd_int)
                else:
                    widget = ctk.CTkEntry(row, textvariable=variable, width=width)
            widget.pack(side=ctk.RIGHT, padx=(8, 0))
            return widget

        create_param_row(input_frame, "N painéis:", self.horz_N, validate="int")
        
        freq_row = ctk.CTkFrame(input_frame)
        freq_row.pack(fill=ctk.X, padx=8, pady=4)
        ctk.CTkLabel(freq_row, text="Frequência:", width=100).pack(side=ctk.LEFT)
        ctk.CTkEntry(freq_row, textvariable=self.horz_freq, width=80, validate="key", 
                    validatecommand=vcmd_float).pack(side=ctk.LEFT, padx=(8, 4))
        ctk.CTkOptionMenu(freq_row, variable=self.horz_funit, values=["Hz","kHz","MHz","GHz"], width=70).pack(side=ctk.LEFT)
        
        create_param_row(input_frame, "β [deg/painel]:", self.horz_beta, validate="float")
        create_param_row(input_frame, "Nível (amp.):", self.horz_level, validate="float")
        create_param_row(input_frame, "Esp. s [m]:", self.horz_spacing, validate="float")
        create_param_row(input_frame, "Δφ [deg]:", self.horz_stepdeg, validate="float")
        create_param_row(input_frame, "Normalizar:", self.horz_norm, ["none","max","rms"])

        # Seção de exportação
        export_frame = ctk.CTkFrame(input_frame)
        export_frame.pack(fill=ctk.X, padx=8, pady=12)
        ctk.CTkLabel(export_frame, text="Exportação", font=ctk.CTkFont(weight="bold")).pack(pady=(4, 8))
        
        create_param_row(export_frame, "Descrição:", self.horz_desc)
        create_param_row(export_frame, "Ganho [dB]:", self.horz_gain, validate="float")
        create_param_row(export_frame, "Nº Antenas:", self.horz_num_antennas, validate="int")
        create_param_row(export_frame, "Passo [deg]:", self.horz_step, ["1","2","3","4","5"])

        # Botões de exportação
        btn_frame = ctk.CTkFrame(input_frame)
        btn_frame.pack(fill=ctk.X, padx=8, pady=12)
        ctk.CTkButton(btn_frame, text="Calcular HRP", command=self.compute_horizontal_panels, 
                     fg_color="#2277cc").pack(side=ctk.TOP, fill=ctk.X, pady=2)
        
        export_btn_frame = ctk.CTkFrame(btn_frame)
        export_btn_frame.pack(fill=ctk.X, pady=2)

        ctk.CTkButton(export_btn_frame, text="Export .PAT", command=self.export_horizontal_array_pat, 
                     fg_color="#22aa66", width=100).pack(side=ctk.LEFT, padx=2)
        ctk.CTkButton(export_btn_frame, text="Export PAT RFS", command=self.export_horizontal_array_rfs, 
                     fg_color="#33cc99", width=100).pack(side=ctk.LEFT, padx=2)
        ctk.CTkButton(export_btn_frame, text="Export .PRN", command=self.export_horizontal_array_prn, 
                     fg_color="#aa6622", width=100).pack(side=ctk.LEFT, padx=2)

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

            # Parâmetros com tratamento de erro
            N   = self._get_int_value(self.horz_N, 4)
            beta_deg = self._get_float_value(self.horz_beta, 0.0)
            w    = self._get_float_value(self.horz_level, 1.0)
            s    = self._get_float_value(self.horz_spacing, 2.0)
            dphi_deg = self._get_float_value(self.horz_stepdeg, 90.0)

            f_hz = self._freq_to_hz(self._get_float_value(self.horz_freq, 0.9), self.horz_funit.get())
            lam = C0 / max(f_hz, 1.0)
            k   = 2.0 * math.pi / lam

            # CÁLCULO CORRETO: Posições dos painéis em um polígono regular no plano horizontal
            alpha_m_deg = np.arange(N) * dphi_deg
            alpha_m_rad = np.deg2rad(alpha_m_deg)
            
            # Raio do polígono para que a distância entre painéis adjacentes seja s
            if N > 1:
                R = s / (2 * np.sin(np.pi / N))
            else:
                R = 0

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
            theta_plot = (theta_plot + np.pi/2) % (2*np.pi)
            
            self.ax_h2.plot(theta_plot, E_comp_mag, linewidth=1.5, color='red')
            self.ax_h2.set_theta_zero_location('N')
            self.ax_h2.set_theta_direction(-1)
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

    def export_horizontal_array_pat(self):
        if self.horz_angles is None or self.horz_values is None:
            messagebox.showwarning("Nada para exportar", "Execute o cálculo da composição horizontal primeiro.")
            return
        base = self.base_name_var.get().strip() or "xxx"
        out_dir = self.output_dir or os.getcwd()
        
        description = self.horz_desc.get() or f"{base}_HRP_composto"
        gain = self._get_float_value(self.horz_gain, 0.0)
        num_antennas = self._get_int_value(self.horz_num_antennas, 4)
        step = self._get_int_value(self.horz_step, 1)
        
        path_h = os.path.join(out_dir, f"{base}_HRP_composto.pat")
        write_pat_horizontal_new_format(path_h, description, gain, num_antennas,
                                      self.horz_angles, self.horz_values, step)
        self._set_status(f"HRP composto .PAT exportado: {path_h}")

    def export_horizontal_array_rfs(self):
        if self.horz_angles is None or self.horz_values is None:
            messagebox.showwarning("Nada para exportar", "Execute o cálculo da composição horizontal primeiro.")
            return
        
        base = self.base_name_var.get().strip() or "xxx"
        out_dir = self.output_dir or os.getcwd()
        
        description = self.horz_desc.get() or f"{base}_HRP_composto"
        gain = self._get_float_value(self.horz_gain, 0.0)
        num_antennas = self._get_int_value(self.horz_num_antennas, 4)
        
        path_h = os.path.join(out_dir, f"{base}_HRP_composto_RFS.pat")
        
        try:
             write_pat_horizontal_rfs(path_h, description, gain, num_antennas, self.horz_angles, self.horz_values)
             self._set_status(f"HRP RFS .PAT exportado: {path_h}")
        except Exception as e:
             messagebox.showerror("Erro Export RFS", str(e))

    def export_horizontal_array_prn(self):
        if self.horz_angles is None or self.horz_values is None:
            messagebox.showwarning("Nada para exportar", "Execute o cálculo da composição horizontal primeiro.")
            return
            
        if self.v_angles is None or self.v_vals is None:
            messagebox.showwarning("Dados incompletos", "Para exportar .PRN é necessário ter o VRP carregado na aba Arquivo.")
            return
            
        try:
            # Coletar metadados
            name = self.prn_name.get().strip() or "ANTENA_FM"
            make = self.prn_make.get().strip() or "RFS"
            
            def safe_float(v, default=0.0):
                try: 
                    return float(v)
                except: 
                    return default
            
            freq = safe_float(self.prn_freq.get(), 99.50)
            freq_unit = self.prn_freq_unit.get()
            gain = safe_float(self.prn_gain.get(), 2.77)
            h_width = safe_float(self.prn_h_width.get(), 65.0)
            v_width = safe_float(self.prn_v_width.get(), 45.0)
            fb_ratio = safe_float(self.prn_fb_ratio.get(), 25.0)
            
            # Reamostrar dados verticais (usando dados da aba Arquivo)
            v_angles, v_vals = resample_vertical(self.v_angles, self.v_vals, self.norm_mode_var.get())
            
            # Pedir local para salvar
            file_path = filedialog.asksaveasfilename(
                defaultextension=".prn",
                filetypes=[("PRN files", "*.prn"), ("All files", "*.*")],
                title="Salvar arquivo .PRN"
            )
            
            if not file_path:
                return
                
            # Escrever arquivo
            write_prn_file(file_path, name, make, freq, freq_unit, h_width, v_width, 
                          fb_ratio, gain, self.horz_angles, self.horz_values, v_angles, v_vals)
            
            self._set_status(f"Arquivo .PRN exportado: {file_path}")
            
        except Exception as e:
            messagebox.showerror("Erro ao exportar .PRN", str(e))

    # ----------------------------- Misc ----------------------------- #
    def _set_status(self, text: str):
        self.status.configure(text=text)

if __name__ == "__main__":
    app = PATConverterApp()
    app.mainloop()