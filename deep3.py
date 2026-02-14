# -*- coding: utf-8 -*-
"""
CTk PAT Converter Ã¢â‚¬â€ Arquivo | Array Vertical | PainÃƒÂ©is Horizontais
"""

from __future__ import annotations

import os
import re
import math
import csv
import json
import sqlite3
import datetime
import io
import shutil
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
from null_fill_synthesis import synth_null_fill_by_order, weights_to_harness

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
    # Se a primeira linha tiver letras, ÃƒÂ© header
    if any(ch.isalpha() for ch in "".join(rows[0])):
        rows = rows[1:]
    thetas: List[float] = []
    vals: List[float] = []
    for r in rows:
        if len(r) < 4:
            continue
        t_raw = r[2].strip().replace(",", ".")   # Theta [deg]
        v_raw = r[-1].strip().replace(",", ".")  # ÃƒÂºltima coluna (E/Emax linear)
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
    """Fallback p/ TXT/TSV com 2+ nÃƒÂºmeros/linha; se 3+, ignora primeiro (ÃƒÂ­ndice)."""
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
        raise ValueError("Nenhum dado numÃƒÂ©rico vÃƒÂ¡lido encontrado no arquivo.")
    a = np.asarray(angles, dtype=float)
    v = np.asarray(vals, dtype=float)
    order = np.argsort(a)
    return a[order], v[order]

def parse_auto(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Tenta HFSS CSV; se falhar, parse genÃƒÂ©rico."""
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
        raise ValueError("Tabela vertical nÃƒÂ£o cobre o intervalo [-90, 90] graus.")
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

def _resample_vertical_adt(angles_deg: np.ndarray, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reamostra VRP para ADT no intervalo fixo [-90, 90] com passo de 0.1 grau.
    Tenta primeiro no sistema original; se necessÃƒÂ¡rio, aplica ajustes comuns.
    """
    a = np.asarray(angles_deg, dtype=float)
    v = np.asarray(values, dtype=float)

    # Caso padrÃƒÂ£o: arquivo jÃƒÂ¡ estÃƒÂ¡ em [-90, 90]
    try:
        return resample_vertical(a, v, norm="none")
    except Exception:
        pass

    # Caso comum de theta em [0, 180] com boresight em 90 graus
    try:
        return resample_vertical(a - 90.0, v, norm="none")
    except Exception:
        pass

    # Fallback final: wrap para [-180, 180] e extrai faixa ÃƒÂºtil vertical
    try:
        return resample_vertical(_wrap_to_180(a), v, norm="none")
    except Exception as e:
        raise ValueError("VRP nÃƒÂ£o cobre a faixa exigida para ADT (-90 a 90 graus).") from e

def apply_vertical_electrical_tilt(
    angles_deg: np.ndarray,
    values: np.ndarray,
    electrical_tilt_deg: float = 0.0,
) -> np.ndarray:
    """
    Aplica tilt eletrico por deslocamento angular do VRP.
    Tilt positivo desloca o lobo principal para baixo.
    """
    a = np.asarray(angles_deg, dtype=float)
    v = np.asarray(values, dtype=float).copy()
    if a.size == 0 or v.size == 0 or a.size != v.size:
        raise ValueError("Dados invalidos para tilt eletrico no VRP.")

    vmax = np.max(v)
    if vmax > 0:
        v = v / vmax

    tilt = float(electrical_tilt_deg)
    if abs(tilt) <= 1e-12:
        return v

    v = np.interp(a + tilt, a, v, left=v[0], right=v[-1])
    vmax = np.max(v)
    if vmax > 0:
        v = v / vmax
    return v

def build_vertical_power_asymmetry_weights(
    num_elements: int,
    null_fill_percent: float = 0.0,
    base_level: float = 1.0,
) -> np.ndarray:
    """
    Gera pesos de amplitude por nivel para o empilhamento vertical.

    O parametro `null_fill_percent` representa a variacao percentual de potencia
    por nivel adjacente do empilhamento (escala progressiva).
    Exemplo: 20 -> cada nivel inferior recebe ~20% mais potencia que o nivel
    imediatamente acima (antes da normalizacao global).

    O retorno e em amplitude (sqrt da potencia) e e normalizado para manter
    a potencia total equivalente ao caso uniforme.
    """
    n = max(int(num_elements), 1)
    base = abs(float(base_level))
    pct = float(null_fill_percent)
    if not math.isfinite(pct):
        pct = 0.0
    pct = max(-95.0, min(95.0, pct))

    if n == 1 or abs(pct) <= 1e-12:
        return np.full(n, base, dtype=float)

    # Modelo progressivo: percentual por nivel adjacente.
    # Indice menor = nivel inferior; indice maior = nivel superior.
    step_ratio = 1.0 + (pct / 100.0)
    step_ratio = max(step_ratio, 1e-3)
    center = 0.5 * (n - 1)
    exponents = center - np.arange(n, dtype=float)
    power_weights = step_ratio ** exponents

    # Mantem potencia media por elemento.
    power_weights = power_weights * (n / np.sum(power_weights))
    amp_weights = np.sqrt(power_weights)
    return base * amp_weights

def linear_to_db(values: np.ndarray) -> np.ndarray:
    """Converte valores lineares (0..1) para dB (-inf..0)."""
    v = values.copy()
    v[v <= 1e-9] = 1e-9
    return 20 * np.log10(v)

def _estimate_step_deg(angles: np.ndarray) -> float:
    a = np.asarray(angles, dtype=float).reshape(-1)
    if a.size < 2:
        return float("nan")
    da = np.diff(np.unique(a))
    if da.size == 0:
        return float("nan")
    return float(np.median(np.abs(da)))


def _first_null_db(angles: np.ndarray, values_norm: np.ndarray) -> float:
    a = np.asarray(angles, dtype=float).reshape(-1)
    v = np.asarray(values_norm, dtype=float).reshape(-1)
    if a.size < 5 or v.size != a.size:
        return float("nan")
    idx_pk = int(np.argmax(v))
    mins = np.where((v[1:-1] <= v[:-2]) & (v[1:-1] <= v[2:]))[0] + 1
    left = mins[mins < idx_pk]
    right = mins[mins > idx_pk]
    vdb = linear_to_db(v)
    if left.size and right.size:
        return float(max(vdb[int(left[-1])], vdb[int(right[0])]))
    if left.size:
        return float(vdb[int(left[-1])])
    if right.size:
        return float(vdb[int(right[0])])
    return float("nan")


def _prepare_pattern_for_export(kind: str, angles: np.ndarray, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    k = str(kind).strip().upper()
    if k == "H":
        return resample_horizontal(angles, values, norm="none")
    try:
        return resample_vertical(angles, values, norm="none")
    except Exception:
        return _resample_vertical_adt(angles, values)


def compute_diagram_metrics(kind: str, angles: np.ndarray, values: np.ndarray) -> dict:
    k = str(kind).strip().upper()
    a = np.asarray(angles, dtype=float).reshape(-1)
    v = np.asarray(values, dtype=float).reshape(-1)
    if a.size == 0 or v.size == 0 or a.size != v.size:
        return {}

    vmax = float(np.max(np.abs(v)))
    if vmax <= 1e-12:
        return {}
    vn = np.abs(v) / vmax
    vdb = linear_to_db(vn)
    idx_pk = int(np.argmax(vn))
    peak_ang = float(a[idx_pk])
    peak_db = float(vdb[idx_pk])
    hpbw = hpbw_deg(a, vn)
    span = 360.0 if k == "H" else 180.0
    d2d = directivity_2d_cut(a, vn, span_deg=span)
    d2d_db = 10.0 * math.log10(d2d) if math.isfinite(d2d) and d2d > 0 else float("nan")
    first_null = _first_null_db(a, vn)
    step_deg = _estimate_step_deg(a)
    fb_db = float("nan")
    if k == "H":
        opp = ((peak_ang + 180.0 + 180.0) % 360.0) - 180.0
        v_opp = float(np.interp(opp, a, vn, period=360.0))
        if v_opp > 1e-12:
            fb_db = float(20.0 * math.log10(max(vn[idx_pk], 1e-12) / v_opp))

    return {
        "kind": k,
        "points": int(a.size),
        "angle_min": float(np.min(a)),
        "angle_max": float(np.max(a)),
        "step_deg": step_deg,
        "peak_angle_deg": peak_ang,
        "peak_db": peak_db,
        "hpbw_deg": float(hpbw),
        "d2d": float(d2d) if math.isfinite(d2d) else float("nan"),
        "d2d_db": float(d2d_db) if math.isfinite(d2d_db) else float("nan"),
        "first_null_db": float(first_null) if math.isfinite(first_null) else float("nan"),
        "fb_db": float(fb_db) if math.isfinite(fb_db) else float("nan"),
    }


def format_diagram_metric_lines(metrics: dict) -> List[str]:
    if not metrics:
        return ["Metricas indisponiveis."]

    def _fmt(v: float, fmt: str = ".2f", suffix: str = "") -> str:
        if not isinstance(v, (int, float)) or not math.isfinite(float(v)):
            return "-"
        return f"{float(v):{fmt}}{suffix}"

    lines = [
        f"Pico: {_fmt(metrics.get('peak_db', float('nan')))} dB @ {_fmt(metrics.get('peak_angle_deg', float('nan')), '.2f')} deg",
        f"HPBW: {_fmt(metrics.get('hpbw_deg', float('nan')), '.2f', ' deg')}",
        f"D2D: {_fmt(metrics.get('d2d', float('nan')), '.3f')} ({_fmt(metrics.get('d2d_db', float('nan')))} dB)",
        f"1o nulo: {_fmt(metrics.get('first_null_db', float('nan')))} dB",
    ]
    if str(metrics.get("kind", "")).upper() == "H":
        lines.append(f"F/B: {_fmt(metrics.get('fb_db', float('nan')))} dB")
    lines.append(
        f"Pontos: {int(metrics.get('points', 0))} | Faixa: {_fmt(metrics.get('angle_min', float('nan')), '.1f')}..{_fmt(metrics.get('angle_max', float('nan')), '.1f')} deg | Passo: {_fmt(metrics.get('step_deg', float('nan')), '.3f')} deg"
    )
    return lines


def build_diagram_export_figure(
    kind: str,
    angles: np.ndarray,
    values: np.ndarray,
    title: str,
    prefer_polar: Optional[bool] = None,
    line_color: Optional[str] = None,
) -> Tuple[Figure, dict]:
    k = str(kind).strip().upper()
    a, v = _prepare_pattern_for_export(k, angles, values)
    v = np.asarray(v, dtype=float)
    vmax = float(np.max(np.abs(v)))
    v_norm = v / (vmax if vmax > 1e-12 else 1.0)

    if prefer_polar is None:
        prefer_polar = (k == "H")

    fig = Figure(figsize=(9.4, 5.4), dpi=120)
    ax = fig.add_subplot(111, projection="polar" if prefer_polar else None)
    color = line_color or ("#d55e00" if k == "H" else "#1f77b4")

    if prefer_polar:
        theta = np.deg2rad(a)
        if k == "H":
            theta = (theta + np.pi / 2.0) % (2.0 * np.pi)
        ax.plot(theta, v_norm, color=color, linewidth=1.6)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.grid(True, alpha=0.3)
    else:
        ax.plot(a, v_norm, color=color, linewidth=1.6)
        ax.set_xlabel("Angulo [deg]")
        ax.set_ylabel("Amplitude (norm.)")
        ax.grid(True, alpha=0.3)
        ax.set_xlim([-180, 180] if k == "H" else [-90, 90])
        ax.set_ylim([0, 1.05])
    ax.set_title(title)

    metrics = compute_diagram_metrics(k, a, v_norm)
    metric_text = "\n".join(format_diagram_metric_lines(metrics))
    ax.text(
        0.99,
        0.99,
        metric_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.30", facecolor="white", edgecolor="#666666", alpha=0.90),
    )
    return fig, metrics


def render_table_image(angles: np.ndarray, values: np.ndarray, unit: str, color: str, title: str) -> Image.Image:
    a = np.asarray(angles, dtype=float).reshape(-1)
    v = np.asarray(values, dtype=float).reshape(-1)
    if a.size == 0 or v.size == 0 or a.size != v.size:
        raise ValueError("Dados invalidos para tabela.")

    n = a.size
    mid = (n + 1) // 2
    step = _estimate_step_deg(a)
    ang_fmt = "{:.0f}" if math.isfinite(step) and step >= 1.0 else "{:.1f}"

    cell_text: List[List[str]] = []
    for i in range(mid):
        a1 = ang_fmt.format(float(a[i]))
        v1 = f"{float(v[i]):.2f}"
        if i + mid < n:
            a2 = ang_fmt.format(float(a[i + mid]))
            v2 = f"{float(v[i + mid]):.2f}"
        else:
            a2, v2 = "", ""
        cell_text.append([a1, v1, a2, v2])

    h = max(6.0, len(cell_text) * 0.28 + 2.6)
    fig = Figure(figsize=(10.0, h), dpi=120)
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.set_title(title, pad=14, fontsize=13, fontweight="bold")

    summary = (
        f"Pontos: {n} | Faixa: {float(np.min(a)):.1f}..{float(np.max(a)):.1f} deg | "
        f"Passo: {step:.3f} deg | Max: {float(np.max(v)):.2f} {unit} | Min: {float(np.min(v)):.2f} {unit}"
    )
    fig.text(0.5, 0.952, summary, ha="center", va="top", fontsize=9, color="#333333")

    header_color = color if color else "#4e79a7"
    col_labels = ["Ang [deg]", f"Valor [{unit}]", "Ang [deg]", f"Valor [{unit}]"]
    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        colLoc="center",
        cellLoc="center",
        bbox=[0.02, 0.02, 0.96, 0.88],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(weight="bold", color="white")
            cell.set_height(cell.get_height() * 1.10)
        else:
            cell.set_facecolor("#f7f9fc" if (row % 2 == 0) else "#ffffff")
            cell.set_edgecolor("#d7dce2")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.35)
    buf.seek(0)
    img = Image.open(buf).copy()
    buf.close()
    return img

# ----------------------------- IntegraÃƒÂ§ÃƒÂ£o e MÃƒÂ©tricas ----------------------------- #
def simpson(y: np.ndarray, dx: float) -> float:
    """Simpson composta p/ passo uniforme (ajuste de ÃƒÂºltimo intervalo se n par)."""
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
    """Diretividade 2D do corte: DÃ¢â€šâ€šD = span_rad / Ã¢Ë†Â« P dÃŽÂ¸, com P = (E/Emax)Ã‚Â² e ÃŽÂ¸ em rad."""
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

# ----------------------------- ValidaÃƒÂ§ÃƒÂ£o de Entrada ----------------------------- #
def validate_float(P):
    """ValidaÃƒÂ§ÃƒÂ£o para campos de entrada numÃƒÂ©ricos"""
    if P == "" or P == "-" or P == "+":
        return True
    try:
        float(P)
        return True
    except ValueError:
        return False

def validate_int(P):
    """ValidaÃƒÂ§ÃƒÂ£o para campos de entrada inteiros"""
    if P == "":
        return True
    try:
        int(P)
        return True
    except ValueError:
        return False

def _array_to_list(arr: Optional[np.ndarray]) -> Optional[List[float]]:
    if arr is None:
        return None
    return np.asarray(arr, dtype=float).tolist()

def _list_to_array(data: Optional[List[float]]) -> Optional[np.ndarray]:
    if data is None:
        return None
    return np.asarray(data, dtype=float)

# ----------------------------- Helpers de exportaÃƒÂ§ÃƒÂ£o PRN ----------------------------- #
def write_prn_file(path: str, name: str, make: str, frequency: float, freq_unit: str,
                  h_width: float, v_width: float, front_to_back: float, gain: float,
                  h_angles: np.ndarray, h_values: np.ndarray,
                  v_angles: np.ndarray, v_values: np.ndarray) -> None:
    """Escreve arquivo PRN com valores em dB de AtenuaÃƒÂ§ÃƒÂ£o.
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
    """LÃƒÂª arquivo .PRN. Converte dB Atten -> Linear. Preserva Vertical 0-360."""
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
        ctk.CTkLabel(self.left_frame, text="CÃƒÂ¡lculos AutomÃƒÂ¡ticos", font=ctk.CTkFont(weight="bold", size=16)).pack(pady=(20, 10))
        
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
            txt += f"HPBW Calc: {calc_h_bw:.2f}Ã‚Â° / File: {self.entries['H_WIDTH'].get()}\n"
            
            # F/B Ratio check
            # ... (kept simple)
            txt += f"Max Val: {h_max:.4f}\n"

        txt += "\n--- DIAGRAMA VERTICAL ---\n"
        if len(v_val) > 0:
            v_max = np.max(v_val) if np.max(v_val) > 0 else 1.0
            v_norm = v_val / v_max
            calc_v_bw = self._calc_hpbw_generic(v_ang, v_norm)
            txt += f"HPBW Calc: {calc_v_bw:.2f}Ã‚Â° / File: {self.entries['V_WIDTH'].get()}\n"
            
            idx_v_peak = np.argmax(v_norm)
            peak_ang_v = v_ang[idx_v_peak]
            txt += f"Peak Angle: {peak_ang_v:.1f}Ã‚Â°\n"
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

# ----------------------------- Helpers de exportaÃƒÂ§ÃƒÂ£o PAT ----------------------------- #
def write_pat_vertical_new_format(path: str, description: str, gain: float, num_antennas: int, 
                                angles: np.ndarray, values: np.ndarray, step: int = 1) -> None:
    """Escreve arquivo PAT no novo formato para diagrama vertical"""
    # Converter para dB (valores negativos)
    values_db = 20 * np.log10(np.maximum(values, 1e-10))
    
    # Criar ÃƒÂ¢ngulos de 0 a 360 com o passo especificado
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
    
    # Interpolar para os ÃƒÂ¢ngulos alvo
    final_values = np.interp(target_angles, unique_angles, unique_values, 
                           left=unique_values[0], right=unique_values[-1])
    
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"'{description}',{gain:.2f},{num_antennas}\n")
        for ang, val in zip(target_angles, final_values):
            if ang <= 360:  # Garantir que nÃƒÂ£o ultrapasse 360
                f.write(f"{int(ang)},{val:.2f}\n")

def write_pat_horizontal_new_format(path: str, description: str, gain: float, num_antennas: int,
                                  angles: np.ndarray, values: np.ndarray, step: int = 1) -> None:
    """Escreve arquivo PAT no novo formato para diagrama horizontal"""
    # Converter para dB (valores negativos)
    values_db = 20 * np.log10(np.maximum(values, 1e-10))
    
    # Criar ÃƒÂ¢ngulos de 0 a 360 com o passo especificado
    target_angles = np.arange(0, 361, step)
    
    # Converter ÃƒÂ¢ngulos de -180~180 para 0~360
    source_angles = (angles + 360) % 360
    order = np.argsort(source_angles)
    source_angles_sorted = source_angles[order]
    source_values_sorted = values_db[order]
    
    # Interpolar para os ÃƒÂ¢ngulos alvo
    final_values = np.interp(target_angles, source_angles_sorted, source_values_sorted,
                           period=360)
    
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"'{description}',{gain:.2f},{num_antennas}\n")
        for ang, val in zip(target_angles, final_values):
            f.write(f"{int(ang)},{val:.2f}\n")

# ----------------------------- Helpers de exportaÃƒÂ§ÃƒÂ£o PRN ----------------------------- #
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
        # CabeÃƒÂ§alho
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


def write_pat_adt_format(path: str, description: str, angles: np.ndarray, values: np.ndarray,
                         pattern_type: str = "") -> None:
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
    # Enforce ADT axis/range:
    # - HRP: -180 .. 180 (passo 1 deg)
    # - VRP: -90 .. 90 (passo 0.1 deg)
    ptype = (pattern_type or "").upper()
    if ptype == "V":
        ang_out, val_out = _resample_vertical_adt(angles, values)
    elif ptype == "H":
        ang_out, val_out = resample_horizontal(
            np.asarray(angles, dtype=float),
            np.asarray(values, dtype=float),
            norm="none"
        )
    else:
        # Fallback por inferÃƒÂªncia do intervalo angular de entrada
        a = np.asarray(angles, dtype=float)
        if np.nanmin(a) >= -90.0 and np.nanmax(a) <= 90.0:
            ang_out, val_out = _resample_vertical_adt(a, values)
        else:
            ang_out, val_out = resample_horizontal(a, np.asarray(values, dtype=float), norm="none")

    # Normalize to max 1.0
    m = np.max(val_out)
    vals_norm = val_out / (m if m > 0 else 1.0)
    
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"Edited by Deep3\n")
        f.write("98\n")
        f.write("1\n")
        f.write("0 0 0 1 0\n")
        f.write("voltage\n")
        
        for a, v in zip(ang_out, vals_norm):
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
        
        txt = f"HPBW: {hpbw:.1f}Ã‚Â° | Peak: {peak_ang:.1f}Ã‚Â°"
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
        
        self.btn_clear = ctk.CTkButton(self.top_bar, text="Limpar VisualizaÃƒÂ§ÃƒÂ£o", command=self.clear_view, fg_color="gray")
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
        self.title("PAT Converter - Arquivo | Array Vertical | Paineis Horizontais")
        self.geometry("1400x900")
        
        # Header (Logo + Help)
        self._build_header()

        # Estado global (aba 1)
        self.base_name_var = tk.StringVar(value="T_END_FEED_8_Hpol")
        self.author_var    = tk.StringVar(value="gecesar")
        self.norm_mode_var = tk.StringVar(value="none")   # none, max, rms
        self.output_dir: Optional[str] = None
        
        # Plot Modes
        self.v_plot_mode = tk.StringVar(value="Planar")
        self.h_plot_mode = tk.StringVar(value="Polar")
        self.v_table_unit = tk.StringVar(value="dB") # dB/Linear
        self.h_table_unit = tk.StringVar(value="dB")

        # Dados carregados (aba 1)
        self.v_angles = None; self.v_vals = None
        self.h_angles = None; self.h_vals = None
        self.export_registry: List[dict] = []
        
        # Estado da aba de estudo completo (suporte a polarizacao dupla)
        self.study_mode_var = tk.StringVar(value="simples")  # simples | duplo
        self.study_h1_angles = None; self.study_h1_vals = None
        self.study_v1_angles = None; self.study_v1_vals = None
        self.study_h2_angles = None; self.study_h2_vals = None
        self.study_v2_angles = None; self.study_v2_vals = None
        self.study_sources = {"H1": "", "V1": "", "H2": "", "V2": ""}
        self.study_widgets = {}

        # Tabview (6 abas)
        self.tabs = ctk.CTkTabview(self)
        self.tabs.pack(fill=ctk.BOTH, expand=True, padx=10, pady=10)

        self.tab_file = self.tabs.add("Arquivo")
        self.tab_vert = self.tabs.add("Composicao Vertical")
        self.tab_horz = self.tabs.add("Composicao Horizontal")
        self.tab_study = self.tabs.add("Estudo Completo")
        self.tab_proj = self.tabs.add("Dados do Projeto")
        self.tab_diag = self.tabs.add("Diagramas (Batch)")

        self._build_tab_file()
        self._build_tab_vertical()
        self._build_tab_horizontal()
        self._build_tab_study()
        self._build_tab_project()
        
        # Build Diagram Tab with Callback
        self.diagrams_view = DiagramsTab(self.tab_diag, load_callback=self.load_from_library)
        self.diagrams_view.pack(fill="both", expand=True)

        # Status
        self.status = ctk.CTkLabel(self, text="Pronto.")
        self.status.pack(side=ctk.BOTTOM, fill=ctk.X, padx=12, pady=8)
        self._refresh_project_overview()

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

1. MODULO ARQUIVO (Aba 1)
   - Utilize esta aba para visualizar e exportar diagramas individuais.
   - **Importar**: Carregue arquivos .CSV (formato HFSS) ou .TXT para VRP (Vertical) e HRP (Horizontal).
   - **Visualizar**: Os graficos mostram visualizacao Planar (VRP) e Polar (HRP).
   - **Exportar**:
     - **.PAT (Standard)**: Formato padrao NSMA.
     - **.PAT (ADT)**: Novo formato RFS Voltage.
     - **.PRN**: Exporta o conjunto VRP+HRP combinados.
     - Certifique-se de preencher os metadados no fundo da tela (Freq, Gain, etc) antes de exportar .PRN.

2. ESTUDO COMPLETO (Aba Estudo Completo)
   - Carregue todos os diagramas em estudo no mesmo lugar.
   - Suporta modo **simples** (1 azimute + 1 elevacao) e **duplo** (2 azimutes + 2 elevacoes).
   - Exporte todos os arquivos do estudo em lote (.PAT, .PAT ADT e .PRN por polarizacao).

3. COMPOSICAO DE ARRAYS (Abas 2 e 3)
   - Ferramentas avancadas para simular arrays verticais e paineis horizontais.
   - Exporte o resultado da simulacao diretamente.

4. DADOS DO PROJETO + BIBLIOTECA
   - Salve e recarregue o trabalho em andamento.
   - Verifique resumo tecnico, estado dos diagramas e historico de exportacoes.
   - Use a biblioteca para importar e reutilizar diagramas.
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

    def _study_slot_kind(self, slot: str) -> str:
        return "H" if slot.upper().startswith("H") else "V"

    def _study_attr_name(self, slot: str, field: str) -> str:
        return f"study_{slot.lower()}_{field}"

    def _get_study_slot_data(self, slot: str):
        ang = getattr(self, self._study_attr_name(slot, "angles"), None)
        val = getattr(self, self._study_attr_name(slot, "vals"), None)
        return ang, val

    def _set_study_slot_data(self, slot: str, angles: np.ndarray, values: np.ndarray, source: str = ""):
        setattr(self, self._study_attr_name(slot, "angles"), np.asarray(angles, dtype=float))
        setattr(self, self._study_attr_name(slot, "vals"), np.asarray(values, dtype=float))
        self.study_sources[slot] = source or ""
        self._plot_study_slot(slot)
        self._refresh_project_overview()

    def _clear_study_slot(self, slot: str):
        setattr(self, self._study_attr_name(slot, "angles"), None)
        setattr(self, self._study_attr_name(slot, "vals"), None)
        self.study_sources[slot] = ""
        self._plot_study_slot(slot)
        self._refresh_project_overview()

    def _read_study_pattern(self, path: str, expected_type: str) -> Tuple[np.ndarray, np.ndarray]:
        ext = os.path.splitext(path)[1].lower()

        if ext == ".prn":
            data = parse_prn(path)
            if expected_type == "H":
                return np.asarray(data["h_angles"], dtype=float), np.asarray(data["h_values"], dtype=float)
            return np.asarray(data["v_angles"], dtype=float), np.asarray(data["v_values"], dtype=float)

        angles = None
        values = None
        patterns = []
        try:
            patterns = RobustPatternParser.parse(path)
        except Exception:
            patterns = []

        if patterns:
            selected = None
            for p in patterns:
                if p.get("type") == expected_type:
                    selected = p
                    break
            if selected is None:
                selected = patterns[0]
            angles = np.asarray(selected["angles"], dtype=float)
            values = np.asarray(selected["values"], dtype=float)
        else:
            try:
                angles, values = parse_auto(path)
            except Exception:
                angles, values = parse_rfs_pat_file(path)

            angles = np.asarray(angles, dtype=float)
            values = np.asarray(values, dtype=float)

        if angles is None or values is None or len(angles) == 0 or len(values) == 0:
            raise ValueError("Nao foi possivel ler dados validos do arquivo selecionado.")

        mask = np.isfinite(angles) & np.isfinite(values)
        angles = angles[mask]
        values = values[mask]
        if len(angles) < 2:
            raise ValueError("O arquivo possui amostras insuficientes.")

        if expected_type == "H":
            if (np.max(angles) - np.min(angles)) > 200 or np.max(angles) > 190:
                angles = _wrap_to_180(angles)
        else:
            if np.min(angles) >= 0 and np.max(angles) <= 180:
                angles = angles - 90.0

        order = np.argsort(angles)
        return angles[order], values[order]

    def _load_study_slot(self, slot: str):
        expected_type = self._study_slot_kind(slot)
        name = {
            "H1": "Azimute Polarizacao 1",
            "V1": "Elevacao Polarizacao 1",
            "H2": "Azimute Polarizacao 2",
            "V2": "Elevacao Polarizacao 2",
        }.get(slot, slot)
        path = filedialog.askopenfilename(
            title=f"Selecione {name}",
            filetypes=[
                ("Arquivos de Diagrama", "*.prn *.pat *.csv *.tsv *.txt *.dat"),
                ("Todos os Arquivos", "*.*"),
            ],
        )
        if not path:
            return
        try:
            angles, values = self._read_study_pattern(path, expected_type)
            self._set_study_slot_data(slot, angles, values, source=path)
            self._set_status(f"{name} carregado: {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Erro ao carregar diagrama", str(e))

    def _use_file_tab_for_study_slot(self, slot: str):
        ptype = self._study_slot_kind(slot)
        if ptype == "H":
            if self.h_angles is None or self.h_vals is None:
                messagebox.showwarning("Sem dados", "Carregue o HRP na aba Arquivo antes.")
                return
            self._set_study_slot_data(slot, self.h_angles.copy(), self.h_vals.copy(), source="Aba Arquivo (HRP)")
        else:
            if self.v_angles is None or self.v_vals is None:
                messagebox.showwarning("Sem dados", "Carregue o VRP na aba Arquivo antes.")
                return
            self._set_study_slot_data(slot, self.v_angles.copy(), self.v_vals.copy(), source="Aba Arquivo (VRP)")

    def _use_composition_tab_for_study_slot(self, slot: str):
        ptype = self._study_slot_kind(slot)
        if ptype == "H":
            if self.horz_angles is None or self.horz_values is None:
                messagebox.showwarning("Sem dados", "Calcule o HRP composto na aba Composicao Horizontal antes.")
                return
            self._set_study_slot_data(slot, self.horz_angles.copy(), self.horz_values.copy(), source="Aba Composicao (HRP composto)")
        else:
            if self.vert_angles is None or self.vert_values is None:
                messagebox.showwarning("Sem dados", "Calcule o VRP composto na aba Composicao Vertical antes.")
                return
            self._set_study_slot_data(slot, self.vert_angles.copy(), self.vert_values.copy(), source="Aba Composicao (VRP composto)")

    def _use_file_tab_for_study_pol1(self):
        if self.h_angles is not None and self.h_vals is not None:
            self._set_study_slot_data("H1", self.h_angles.copy(), self.h_vals.copy(), source="Aba Arquivo (HRP)")
        if self.v_angles is not None and self.v_vals is not None:
            self._set_study_slot_data("V1", self.v_angles.copy(), self.v_vals.copy(), source="Aba Arquivo (VRP)")
        self._set_status("Polarizacao 1 sincronizada com a aba Arquivo.")

    def _use_composition_tab_for_study_pol1(self):
        loaded = False
        if self.horz_angles is not None and self.horz_values is not None:
            self._set_study_slot_data("H1", self.horz_angles.copy(), self.horz_values.copy(), source="Aba Composicao (HRP composto)")
            loaded = True
        if self.vert_angles is not None and self.vert_values is not None:
            self._set_study_slot_data("V1", self.vert_angles.copy(), self.vert_values.copy(), source="Aba Composicao (VRP composto)")
            loaded = True
        if loaded:
            self._set_status("Polarizacao 1 sincronizada com as composicoes.")
        else:
            messagebox.showwarning("Sem dados", "Calcule os diagramas compostos antes de sincronizar POL1.")

    def _plot_study_slot(self, slot: str):
        if slot not in self.study_widgets:
            return
        widget = self.study_widgets[slot]
        fig = widget["fig"]
        fig.clf()
        kind = self._study_slot_kind(slot)
        if kind == "H":
            ax = fig.add_subplot(111, projection="polar")
        else:
            ax = fig.add_subplot(111)
        angles, values = self._get_study_slot_data(slot)

        title_map = {
            "H1": "Azimute - Polarizacao 1",
            "V1": "Elevacao - Polarizacao 1",
            "H2": "Azimute - Polarizacao 2",
            "V2": "Elevacao - Polarizacao 2",
        }
        color_map = {"H1": "#f28e2b", "V1": "#4e79a7", "H2": "#e15759", "V2": "#76b7b2"}

        ax.set_title(title_map.get(slot, slot))
        ax.grid(True, alpha=0.3)

        if angles is None or values is None:
            ax.text(0.5, 0.5, "Sem dados", ha="center", va="center", transform=ax.transAxes)
            widget["src_label"].configure(text="Fonte: -")
        else:
            if kind == "H":
                a = np.asarray(angles, dtype=float)
                v = np.asarray(values, dtype=float)
                a_wrap = np.mod(a, 360.0)
                order = np.argsort(a_wrap)
                ax.plot(np.deg2rad(a_wrap[order]), v[order], linewidth=1.3, color=color_map.get(slot, "tab:blue"))
                ax.set_theta_zero_location("N")
                ax.set_theta_direction(-1)
            else:
                ax.plot(angles, values, linewidth=1.3, color=color_map.get(slot, "tab:blue"))
                ax.set_xlim([-90, 90])
                ax.set_xlabel("Elevacao [deg]")
            ax.set_ylabel("E/Emax")
            src = self.study_sources.get(slot) or "-"
            widget["src_label"].configure(text=f"Fonte: {os.path.basename(src) if src != '-' else '-'}")

        widget["ax"] = ax
        widget["canvas"].draw()

    def _refresh_study_mode(self, *_):
        dual = self.study_mode_var.get().lower() == "duplo"
        for slot in ("H2", "V2"):
            if slot not in self.study_widgets:
                continue
            for btn in self.study_widgets[slot]["buttons"]:
                btn.configure(state="normal" if dual else "disabled")
            self.study_widgets[slot]["mode_label"].configure(
                text="Ativo" if dual else "Desabilitado (modo simples)"
            )
        self._refresh_project_overview()

    def _create_study_slot_card(self, parent, slot: str, title: str):
        card = ctk.CTkFrame(parent)
        ctk.CTkLabel(card, text=title, font=ctk.CTkFont(size=13, weight="bold")).pack(anchor="w", padx=6, pady=(6, 2))

        fig = Figure(figsize=(4.8, 2.8), dpi=90)
        ax = fig.add_subplot(111)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        canvas = FigureCanvasTkAgg(fig, master=card)
        canvas.get_tk_widget().pack(fill=ctk.BOTH, expand=True, padx=4, pady=4)

        src_label = ctk.CTkLabel(card, text="Fonte: -", anchor="w")
        src_label.pack(fill=ctk.X, padx=6, pady=(0, 4))
        mode_label = ctk.CTkLabel(card, text="Ativo", anchor="w")
        mode_label.pack(fill=ctk.X, padx=6, pady=(0, 4))

        btn_row = ctk.CTkFrame(card, fg_color="transparent")
        btn_row.pack(fill=ctk.X, padx=4, pady=(0, 6))
        b1 = ctk.CTkButton(btn_row, text="Carregar", width=84, command=lambda s=slot: self._load_study_slot(s))
        b1.pack(side=ctk.LEFT, padx=2, expand=True, fill=ctk.X)
        b2 = ctk.CTkButton(btn_row, text="Usar Arquivo", width=84, command=lambda s=slot: self._use_file_tab_for_study_slot(s))
        b2.pack(side=ctk.LEFT, padx=2, expand=True, fill=ctk.X)
        b3 = ctk.CTkButton(btn_row, text="Usar Composicao", width=110, command=lambda s=slot: self._use_composition_tab_for_study_slot(s))
        b3.pack(side=ctk.LEFT, padx=2, expand=True, fill=ctk.X)
        b4 = ctk.CTkButton(btn_row, text="Limpar", width=70, fg_color="#aa4444", command=lambda s=slot: self._clear_study_slot(s))
        b4.pack(side=ctk.LEFT, padx=2, expand=True, fill=ctk.X)

        self.study_widgets[slot] = {
            "frame": card,
            "fig": fig,
            "ax": ax,
            "canvas": canvas,
            "src_label": src_label,
            "mode_label": mode_label,
            "buttons": [b1, b2, b3, b4],
        }
        self._plot_study_slot(slot)
        return card

    def _safe_meta_float(self, value: str, default: float) -> float:
        try:
            return float(value)
        except Exception:
            return default

    def export_study_complete(self):
        out_dir = self.output_dir or filedialog.askdirectory(title="Escolha pasta de saida para o estudo completo")
        if not out_dir:
            return

        base = self.base_name_var.get().strip() or "estudo"
        mode_duplo = self.study_mode_var.get().lower() == "duplo"
        exported = []

        slots = [("H1", "AZ_POL1", "H"), ("V1", "EL_POL1", "V")]
        if mode_duplo:
            slots.extend([("H2", "AZ_POL2", "H"), ("V2", "EL_POL2", "V")])

        table_color = {"H": "#f28e2b", "V": "#4e79a7"}

        for slot, suffix, kind in slots:
            ang, val = self._get_study_slot_data(slot)
            if ang is None or val is None:
                continue

            try:
                if kind == "H":
                    ang_out, val_out = resample_horizontal(ang, val, norm="none")
                    table_ang, table_val = self._table_points_horizontal(ang, val)
                    pat_path = os.path.join(out_dir, f"{base}_{suffix}.pat")
                    write_pat_horizontal_new_format(pat_path, f"{base} {suffix}", 0.0, 1, ang_out, val_out, step=1)
                    adt_path = os.path.join(out_dir, f"{base}_{suffix}_ADT.pat")
                    write_pat_adt_format(adt_path, f"{base} {suffix}", ang_out, val_out, pattern_type="H")
                else:
                    try:
                        ang_out, val_out = resample_vertical(ang, val, norm="none")
                    except Exception:
                        ang_out, val_out = _resample_vertical_adt(ang, val)
                    table_ang, table_val = self._table_points_vertical(ang, val)
                    pat_path = os.path.join(out_dir, f"{base}_{suffix}.pat")
                    write_pat_vertical_new_format(pat_path, f"{base} {suffix}", 0.0, 1, ang_out, val_out, step=1)
                    adt_path = os.path.join(out_dir, f"{base}_{suffix}_ADT.pat")
                    write_pat_adt_format(adt_path, f"{base} {suffix}", ang_out, val_out, pattern_type="V")

                self._register_export(pat_path, f"STUDY_{slot}_PAT")
                self._register_export(adt_path, f"STUDY_{slot}_ADT")
                exported.extend([pat_path, adt_path])

                # Imagem do diagrama do slot com metricas
                img_path = os.path.join(out_dir, f"{base}_{suffix}_DIAGRAMA.png")
                fig_diag, _ = build_diagram_export_figure(
                    kind=kind,
                    angles=ang,
                    values=val,
                    title=f"Diagrama {suffix} ({'Azimute' if kind == 'H' else 'Elevacao'})",
                    prefer_polar=(kind == "H"),
                )
                fig_diag.savefig(img_path, dpi=300)
                self._register_export(img_path, f"STUDY_{slot}_PLOT_IMG")
                exported.append(img_path)

                # Tabela (CSV + imagem) do slot
                table_csv = os.path.join(out_dir, f"{base}_{suffix}_TABELA.csv")
                with open(table_csv, "w", encoding="utf-8", newline="\n") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Angle_deg", "Level_linear", "Level_dB"])
                    vals_db = linear_to_db(table_val)
                    for a_i, v_i, d_i in zip(table_ang, table_val, vals_db):
                        writer.writerow([f"{float(a_i):.3f}", f"{float(v_i):.8f}", f"{float(d_i):.4f}"])

                table_title = f"Tabela {suffix} ({'Azimute' if kind == 'H' else 'Elevacao'})"
                table_img = render_table_image(table_ang, linear_to_db(table_val), "dB", table_color[kind], table_title)
                table_png = os.path.join(out_dir, f"{base}_{suffix}_TABELA.png")
                table_img.save(table_png)

                self._register_export(table_csv, f"STUDY_{slot}_TABLE_CSV")
                self._register_export(table_png, f"STUDY_{slot}_TABLE_IMG")
                exported.extend([table_csv, table_png])
            except Exception as e:
                messagebox.showerror("Erro de exportacao", f"Falha ao exportar {slot}: {e}")
                return

        # PRN por polarizacao (quando houver par H+V)
        for pol in (1, 2):
            if pol == 2 and not mode_duplo:
                continue
            h_slot = f"H{pol}"
            v_slot = f"V{pol}"
            h_ang, h_val = self._get_study_slot_data(h_slot)
            v_ang, v_val = self._get_study_slot_data(v_slot)
            if h_ang is None or h_val is None or v_ang is None or v_val is None:
                continue
            try:
                h_out, hv = resample_horizontal(h_ang, h_val, norm="none")
                try:
                    v_out, vv = resample_vertical(v_ang, v_val, norm="none")
                except Exception:
                    v_out, vv = _resample_vertical_adt(v_ang, v_val)

                name = (self.prn_name.get().strip() or "ANTENA_FM") + f"_POL{pol}"
                make = self.prn_make.get().strip() or "RFS"
                freq = self._safe_meta_float(self.prn_freq.get(), 99.50)
                freq_unit = self.prn_freq_unit.get()
                gain = self._safe_meta_float(self.prn_gain.get(), 2.77)
                h_width = self._safe_meta_float(self.prn_h_width.get(), 65.0)
                v_width = self._safe_meta_float(self.prn_v_width.get(), 45.0)
                fb_ratio = self._safe_meta_float(self.prn_fb_ratio.get(), 25.0)

                prn_path = os.path.join(out_dir, f"{base}_POL{pol}.prn")
                write_prn_file(prn_path, name, make, freq, freq_unit, h_width, v_width, fb_ratio, gain, h_out, hv, v_out, vv)
                self._register_export(prn_path, f"STUDY_POL{pol}_PRN")
                exported.append(prn_path)
            except Exception as e:
                messagebox.showerror("Erro de exportacao PRN", f"Falha ao exportar POL{pol}: {e}")
                return

        if not exported:
            messagebox.showwarning("Sem dados", "Nao ha diagramas suficientes para exportacao do estudo.")
            return

        self._set_status(f"Estudo completo exportado: {len(exported)} arquivos em {out_dir}")

    def _build_tab_study(self):
        top = ctk.CTkFrame(self.tab_study)
        top.pack(fill=ctk.X, padx=8, pady=8)

        ctk.CTkLabel(top, text="Modo de Estudo:", width=120).pack(side=ctk.LEFT, padx=(4, 2))
        ctk.CTkOptionMenu(top, variable=self.study_mode_var, values=["simples", "duplo"], command=self._refresh_study_mode, width=120).pack(side=ctk.LEFT, padx=2)
        ctk.CTkButton(top, text="Sincronizar POL1 da Aba Arquivo", command=self._use_file_tab_for_study_pol1, width=210).pack(side=ctk.LEFT, padx=6)
        ctk.CTkButton(top, text="Sincronizar POL1 da Composicao", command=self._use_composition_tab_for_study_pol1, width=220).pack(side=ctk.LEFT, padx=6)
        ctk.CTkButton(top, text="Exportar Estudo Completo", command=self.export_study_complete, fg_color="#22aa66", width=180).pack(side=ctk.RIGHT, padx=4)

        grid = ctk.CTkFrame(self.tab_study)
        grid.pack(fill=ctk.BOTH, expand=True, padx=8, pady=(0, 8))
        grid.grid_rowconfigure(0, weight=1)
        grid.grid_rowconfigure(1, weight=1)
        grid.grid_columnconfigure(0, weight=1)
        grid.grid_columnconfigure(1, weight=1)

        c1 = self._create_study_slot_card(grid, "H1", "Azimute - Polarizacao 1")
        c2 = self._create_study_slot_card(grid, "V1", "Elevacao - Polarizacao 1")
        c3 = self._create_study_slot_card(grid, "H2", "Azimute - Polarizacao 2")
        c4 = self._create_study_slot_card(grid, "V2", "Elevacao - Polarizacao 2")

        c1.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        c2.grid(row=0, column=1, sticky="nsew", padx=4, pady=4)
        c3.grid(row=1, column=0, sticky="nsew", padx=4, pady=4)
        c4.grid(row=1, column=1, sticky="nsew", padx=4, pady=4)

        self._refresh_study_mode()

    def _build_tab_project(self):
        top = ctk.CTkFrame(self.tab_proj)
        top.pack(fill=ctk.X, padx=8, pady=8)

        ctk.CTkButton(top, text="Atualizar Painel", command=self._refresh_project_overview, width=130).pack(side=ctk.LEFT, padx=4)
        ctk.CTkButton(top, text="Salvar Progresso", command=self.save_work_in_progress, fg_color="#2277cc", width=130).pack(side=ctk.LEFT, padx=4)
        ctk.CTkButton(top, text="Carregar Progresso", command=self.load_work_in_progress, fg_color="#4466aa", width=140).pack(side=ctk.LEFT, padx=4)
        ctk.CTkButton(
            top,
            text="Exportar Ja Exportados",
            command=self.export_recorded_files_bundle,
            fg_color="#22aa66",
            width=170
        ).pack(side=ctk.RIGHT, padx=4)

        exports = ctk.CTkFrame(self.tab_proj)
        exports.pack(fill=ctk.X, padx=8, pady=(0, 8))
        ctk.CTkLabel(exports, text="Exportacao Completa", font=ctk.CTkFont(size=12, weight="bold")).pack(anchor="w", padx=4, pady=(4, 6))

        btns = ctk.CTkFrame(exports, fg_color="transparent")
        btns.pack(fill=ctk.X, padx=2, pady=(0, 4))
        ctk.CTkButton(btns, text="Graficos PNG", command=self.export_project_graph_images, width=120, fg_color="#225588").pack(side=ctk.LEFT, padx=3, expand=True, fill=ctk.X)
        ctk.CTkButton(btns, text="Tabelas PNG", command=self.export_project_table_images, width=120, fg_color="#664488").pack(side=ctk.LEFT, padx=3, expand=True, fill=ctk.X)
        ctk.CTkButton(btns, text="PAT", command=self.export_project_pat_files, width=100, fg_color="#228855").pack(side=ctk.LEFT, padx=3, expand=True, fill=ctk.X)
        ctk.CTkButton(btns, text="PAT ADT", command=self.export_project_adt_files, width=100, fg_color="#22aa88").pack(side=ctk.LEFT, padx=3, expand=True, fill=ctk.X)
        ctk.CTkButton(btns, text="PRN", command=self.export_project_prn_files, width=100, fg_color="#aa6622").pack(side=ctk.LEFT, padx=3, expand=True, fill=ctk.X)

        self.project_info_box = ctk.CTkTextbox(self.tab_proj, wrap="word")
        self.project_info_box.pack(fill=ctk.BOTH, expand=True, padx=8, pady=(0, 8))
        self.project_info_box.insert("0.0", "Painel de dados do projeto.")
        self.project_info_box.configure(state="disabled")

    def _refresh_project_overview(self):
        if not hasattr(self, "project_info_box"):
            return

        def var_value(name: str, default: str = "-") -> str:
            var = getattr(self, name, None)
            if isinstance(var, tk.StringVar):
                try:
                    return var.get()
                except Exception:
                    return default
            return default

        def size_info(arr_name: str) -> str:
            arr = getattr(self, arr_name, None)
            if arr is None:
                return "nao carregado"
            return f"{len(arr)} amostras"

        exports_total = len(self.export_registry)
        exports_ok = sum(1 for x in self.export_registry if os.path.exists(x.get("path", "")))
        exports_missing = exports_total - exports_ok

        lines = [
            "=== DADOS DO PROJETO ===",
            "",
            f"Base Name: {var_value('base_name_var')}",
            f"Autor: {var_value('author_var')}",
            f"Normalizacao: {var_value('norm_mode_var')}",
            f"Pasta de saida: {self.output_dir or '(nao definida)'}",
            "",
            "Entradas (Aba Arquivo):",
            f"VRP: {size_info('v_angles')}",
            f"HRP: {size_info('h_angles')}",
            "",
            "Composicao Vertical:",
            f"Resultado: {size_info('vert_angles')}",
            f"N antenas: {var_value('vert_N')}",
            f"Tilt eletrico [deg]: {var_value('vert_tilt_elec_deg', '0.0')}",
            f"Null fill [%]: {var_value('vert_null_fill_pct', '0.0')}",
            f"Modo null fill: {var_value('vert_null_mode', 'both')}",
            f"Nulo alvo: {var_value('vert_null_order', '1')}",
            "",
            "Composicao Horizontal:",
            f"Resultado: {size_info('horz_angles')}",
            f"N paineis: {var_value('horz_N')}",
            "",
            "Estudo Completo:",
            f"Modo: {var_value('study_mode_var', 'simples')}",
            f"H1: {size_info('study_h1_angles')}",
            f"V1: {size_info('study_v1_angles')}",
            f"H2: {size_info('study_h2_angles')}",
            f"V2: {size_info('study_v2_angles')}",
            "",
            "Registro de Exportacoes:",
            f"Total: {exports_total}",
            f"Arquivos encontrados: {exports_ok}",
            f"Arquivos ausentes: {exports_missing}",
        ]

        if self.export_registry:
            lines.append("")
            lines.append("Ultimas exportacoes:")
            recent = self.export_registry[-20:]
            for item in recent:
                ts = item.get("timestamp", "-")
                kind = item.get("kind", "-")
                path = item.get("path", "-")
                lines.append(f"[{ts}] {kind} -> {path}")

        self.project_info_box.configure(state="normal")
        self.project_info_box.delete("0.0", "end")
        self.project_info_box.insert("0.0", "\n".join(lines))
        self.project_info_box.configure(state="disabled")

    def _register_export(self, path: str, kind: str):
        if not path:
            return
        rec = {
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "kind": kind,
            "path": os.path.abspath(path),
        }
        self.export_registry.append(rec)
        self._refresh_project_overview()

    def _collect_project_state(self) -> dict:
        string_vars = {}
        for key, value in self.__dict__.items():
            if isinstance(value, tk.StringVar):
                try:
                    string_vars[key] = value.get()
                except Exception:
                    continue

        array_fields = [
            "v_angles", "v_vals",
            "h_angles", "h_vals",
            "vert_angles", "vert_values",
            "horz_angles", "horz_values",
            "study_h1_angles", "study_h1_vals",
            "study_v1_angles", "study_v1_vals",
            "study_h2_angles", "study_h2_vals",
            "study_v2_angles", "study_v2_vals",
        ]
        arrays = {name: _array_to_list(getattr(self, name, None)) for name in array_fields}

        return {
            "format_version": 1,
            "saved_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "output_dir": self.output_dir,
            "string_vars": string_vars,
            "arrays": arrays,
            "study_sources": dict(self.study_sources),
            "export_registry": list(self.export_registry),
        }

    def _apply_project_state(self, state: dict):
        self.output_dir = state.get("output_dir")

        string_vars_state = state.get("string_vars", {})
        for key, val in string_vars_state.items():
            ref = getattr(self, key, None)
            if isinstance(ref, tk.StringVar):
                ref.set(str(val))

        # Compatibilidade com versoes antigas que salvavam null fill em dB.
        if (
            isinstance(string_vars_state, dict)
            and "vert_null_fill_pct" not in string_vars_state
            and "vert_null_fill_db" in string_vars_state
            and isinstance(getattr(self, "vert_null_fill_pct", None), tk.StringVar)
        ):
            self.vert_null_fill_pct.set(str(string_vars_state.get("vert_null_fill_db", "0.0")))

        arrays = state.get("arrays", {})
        self.v_angles = _list_to_array(arrays.get("v_angles"))
        self.v_vals = _list_to_array(arrays.get("v_vals"))
        self.h_angles = _list_to_array(arrays.get("h_angles"))
        self.h_vals = _list_to_array(arrays.get("h_vals"))
        self.vert_angles = _list_to_array(arrays.get("vert_angles"))
        self.vert_values = _list_to_array(arrays.get("vert_values"))
        self.horz_angles = _list_to_array(arrays.get("horz_angles"))
        self.horz_values = _list_to_array(arrays.get("horz_values"))
        self.study_h1_angles = _list_to_array(arrays.get("study_h1_angles"))
        self.study_h1_vals = _list_to_array(arrays.get("study_h1_vals"))
        self.study_v1_angles = _list_to_array(arrays.get("study_v1_angles"))
        self.study_v1_vals = _list_to_array(arrays.get("study_v1_vals"))
        self.study_h2_angles = _list_to_array(arrays.get("study_h2_angles"))
        self.study_h2_vals = _list_to_array(arrays.get("study_h2_vals"))
        self.study_v2_angles = _list_to_array(arrays.get("study_v2_angles"))
        self.study_v2_vals = _list_to_array(arrays.get("study_v2_vals"))
        loaded_sources = state.get("study_sources", {})
        if isinstance(loaded_sources, dict):
            for k in ("H1", "V1", "H2", "V2"):
                self.study_sources[k] = str(loaded_sources.get(k, ""))

        loaded_registry = state.get("export_registry", [])
        if isinstance(loaded_registry, list):
            self.export_registry = []
            for item in loaded_registry:
                if isinstance(item, dict):
                    self.export_registry.append({
                        "timestamp": item.get("timestamp", datetime.datetime.now().isoformat(timespec="seconds")),
                        "kind": item.get("kind", "export"),
                        "path": item.get("path", ""),
                    })

        if self.v_angles is not None and self.v_vals is not None:
            self._plot_vertical_file(self.v_angles, self.v_vals)
        if self.h_angles is not None and self.h_vals is not None:
            self._plot_horizontal_file(self.h_angles, self.h_vals)
        if self.vert_angles is not None and self.vert_values is not None:
            self._plot_vertical_composite(self.vert_angles, self.vert_values)
        if self.horz_angles is not None and self.horz_values is not None:
            self._plot_horizontal_composite(self.horz_angles, self.horz_values)
        for slot in ("H1", "V1", "H2", "V2"):
            self._plot_study_slot(slot)
        self._refresh_study_mode()

        self._refresh_project_overview()

    def save_work_in_progress(self):
        path = filedialog.asksaveasfilename(
            title="Salvar progresso do projeto",
            defaultextension=".eftxproj.json",
            filetypes=[("Projeto EFTX", "*.eftxproj.json"), ("JSON", "*.json"), ("All Files", "*.*")]
        )
        if not path:
            return
        try:
            data = self._collect_project_state()
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            self._set_status(f"Progresso salvo: {path}")
        except Exception as e:
            messagebox.showerror("Erro ao salvar progresso", str(e))

    def load_work_in_progress(self):
        path = filedialog.askopenfilename(
            title="Carregar progresso do projeto",
            filetypes=[("Projeto EFTX", "*.eftxproj.json"), ("JSON", "*.json"), ("All Files", "*.*")]
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError("Formato invalido de arquivo de projeto.")
            self._apply_project_state(data)
            self._set_status(f"Progresso carregado: {path}")
        except Exception as e:
            messagebox.showerror("Erro ao carregar progresso", str(e))

    def export_recorded_files_bundle(self):
        if not self.export_registry:
            messagebox.showwarning("Sem exportacoes", "Nao ha arquivos exportados registrados neste projeto.")
            return

        out_root = filedialog.askdirectory(title="Escolha a pasta para exportar os arquivos ja exportados")
        if not out_root:
            return

        try:
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            bundle_dir = os.path.join(out_root, f"exportados_{stamp}")
            os.makedirs(bundle_dir, exist_ok=True)

            manifest = []
            copied = 0
            missing = 0

            for idx, item in enumerate(self.export_registry, start=1):
                src = item.get("path", "")
                record = {
                    "index": idx,
                    "timestamp": item.get("timestamp", ""),
                    "kind": item.get("kind", ""),
                    "source_path": src,
                }

                if src and os.path.exists(src):
                    dst_name = f"{idx:03d}_{os.path.basename(src)}"
                    dst = os.path.join(bundle_dir, dst_name)
                    shutil.copy2(src, dst)
                    record["status"] = "copied"
                    record["bundle_file"] = dst_name
                    copied += 1
                else:
                    record["status"] = "missing"
                    record["bundle_file"] = ""
                    missing += 1
                manifest.append(record)

            manifest_json = os.path.join(bundle_dir, "manifesto_exportacoes.json")
            manifest_csv = os.path.join(bundle_dir, "manifesto_exportacoes.csv")
            with open(manifest_json, "w", encoding="utf-8", newline="\n") as f:
                json.dump(manifest, f, ensure_ascii=False, indent=2)

            with open(manifest_csv, "w", encoding="utf-8", newline="\n") as f:
                writer = csv.writer(f)
                writer.writerow(["index", "timestamp", "kind", "source_path", "status", "bundle_file"])
                for row in manifest:
                    writer.writerow([
                        row["index"],
                        row["timestamp"],
                        row["kind"],
                        row["source_path"],
                        row["status"],
                        row["bundle_file"],
                    ])

            self._set_status(
                f"Pacote exportado em {bundle_dir} (copiados: {copied}, ausentes: {missing})."
            )
        except Exception as e:
            messagebox.showerror("Erro ao exportar pacote", str(e))

    def _project_base_name(self) -> str:
        raw = self.base_name_var.get().strip() if isinstance(getattr(self, "base_name_var", None), tk.StringVar) else ""
        raw = raw or "projeto"
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)
        return safe or "projeto"

    def _project_prepare_export_dir(self, export_kind: str) -> str:
        root = self.output_dir or filedialog.askdirectory(
            title=f"Escolha a pasta de saida para exportacao de {export_kind}"
        )
        if not root:
            return ""
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(root, f"projeto_{export_kind}_{stamp}")
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

    def _collect_project_patterns(self) -> List[Tuple[str, str, np.ndarray, np.ndarray]]:
        patterns: List[Tuple[str, str, np.ndarray, np.ndarray]] = []

        if self.v_angles is not None and self.v_vals is not None:
            patterns.append(("arquivo_vrp", "V", self.v_angles, self.v_vals))
        if self.h_angles is not None and self.h_vals is not None:
            patterns.append(("arquivo_hrp", "H", self.h_angles, self.h_vals))
        if self.vert_angles is not None and self.vert_values is not None:
            patterns.append(("comp_vrp", "V", self.vert_angles, self.vert_values))
        if self.horz_angles is not None and self.horz_values is not None:
            patterns.append(("comp_hrp", "H", self.horz_angles, self.horz_values))

        for slot in ("H1", "V1", "H2", "V2"):
            ang, val = self._get_study_slot_data(slot)
            if ang is None or val is None:
                continue
            patterns.append((f"estudo_{slot.lower()}", self._study_slot_kind(slot), ang, val))
        return patterns

    def _collect_project_prn_pairs(self) -> List[Tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        pairs: List[Tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []

        if self.h_angles is not None and self.h_vals is not None and self.v_angles is not None and self.v_vals is not None:
            pairs.append(("arquivo", self.h_angles, self.h_vals, self.v_angles, self.v_vals))

        if (
            self.horz_angles is not None and self.horz_values is not None
            and self.vert_angles is not None and self.vert_values is not None
        ):
            pairs.append(("composicao", self.horz_angles, self.horz_values, self.vert_angles, self.vert_values))

        h1, hv1 = self._get_study_slot_data("H1")
        v1, vv1 = self._get_study_slot_data("V1")
        if h1 is not None and hv1 is not None and v1 is not None and vv1 is not None:
            pairs.append(("estudo_pol1", h1, hv1, v1, vv1))

        h2, hv2 = self._get_study_slot_data("H2")
        v2, vv2 = self._get_study_slot_data("V2")
        if h2 is not None and hv2 is not None and v2 is not None and vv2 is not None:
            pairs.append(("estudo_pol2", h2, hv2, v2, vv2))

        return pairs

    def _table_points_vertical(self, angles: np.ndarray, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        target = np.arange(-90.0, 90.0 + 1e-9, 1.0)
        try:
            a_src, v_src = resample_vertical(angles, values, norm="none")
        except Exception:
            a_src, v_src = _resample_vertical_adt(angles, values)
        v_interp = np.interp(target, a_src, v_src)
        return target, v_interp

    def _table_points_horizontal(self, angles: np.ndarray, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        target = np.arange(-180, 180, 5, dtype=float)  # 72 pontos
        a_src, v_src = resample_horizontal(angles, values, norm="none")
        src_ang_360 = np.mod(a_src, 360.0)
        order = np.argsort(src_ang_360)
        s_ang = src_ang_360[order]
        s_val = v_src[order]
        s_ang_ext = np.concatenate([s_ang, [s_ang[0] + 360.0]])
        s_val_ext = np.concatenate([s_val, [s_val[0]]])
        v_interp = np.interp(np.mod(target, 360.0), s_ang_ext, s_val_ext)
        return target, v_interp

    def export_project_graph_images(self):
        out_dir = self._project_prepare_export_dir("graficos")
        if not out_dir:
            return

        base = self._project_base_name()
        patterns = self._collect_project_patterns()
        if not patterns:
            messagebox.showwarning("Sem dados", "Nao ha graficos disponiveis para exportar.")
            return
        exported = 0

        for tag, kind, angles, values in patterns:
            try:
                title = f"Diagrama {tag.upper()} ({'Azimute' if kind == 'H' else 'Elevacao'})"
                fig, _ = build_diagram_export_figure(
                    kind=kind,
                    angles=angles,
                    values=values,
                    title=title,
                    prefer_polar=(kind == "H"),
                )
                path = os.path.join(out_dir, f"{base}_{tag}_plot.png")
                fig.savefig(path, dpi=300)
                self._register_export(path, f"PROJECT_{tag.upper()}_PLOT")
                exported += 1
            except Exception as e:
                messagebox.showwarning("Aviso", f"Falha ao exportar grafico '{tag}': {e}")

        self._set_status(f"Exportacao de graficos concluida: {exported} arquivo(s) em {out_dir}.")

    def export_project_table_images(self):
        out_dir = self._project_prepare_export_dir("tabelas")
        if not out_dir:
            return

        base = self._project_base_name()
        patterns = self._collect_project_patterns()
        if not patterns:
            messagebox.showwarning("Sem dados", "Nao ha diagramas disponiveis para exportar tabelas.")
            return

        exported = 0
        for tag, kind, angles, values in patterns:
            try:
                if kind == "V":
                    t_ang, t_val = self._table_points_vertical(angles, values)
                    title = f"Tabela {tag.upper()} - Vertical"
                    color = "#4e79a7"
                else:
                    t_ang, t_val = self._table_points_horizontal(angles, values)
                    title = f"Tabela {tag.upper()} - Horizontal"
                    color = "#f28e2b"

                img = render_table_image(t_ang, linear_to_db(t_val), "dB", color, title)
                path = os.path.join(out_dir, f"{base}_{tag}_tabela.png")
                img.save(path)
                self._register_export(path, f"PROJECT_{tag.upper()}_TABLE")
                exported += 1
            except Exception as e:
                messagebox.showwarning("Aviso", f"Falha ao gerar tabela '{tag}': {e}")

        if exported == 0:
            messagebox.showwarning("Sem exportacao", "Nenhuma tabela foi exportada.")
            return
        self._set_status(f"Exportacao de tabelas concluida: {exported} arquivo(s) em {out_dir}.")

    def export_project_pat_files(self):
        out_dir = self._project_prepare_export_dir("pat")
        if not out_dir:
            return

        base = self._project_base_name()
        patterns = self._collect_project_patterns()
        if not patterns:
            messagebox.showwarning("Sem dados", "Nao ha diagramas disponiveis para exportar PAT.")
            return

        exported = 0
        for tag, kind, angles, values in patterns:
            try:
                desc = f"{base} {tag}"
                path = os.path.join(out_dir, f"{base}_{tag}.pat")
                if kind == "V":
                    try:
                        a_out, v_out = resample_vertical(angles, values, norm="none")
                    except Exception:
                        a_out, v_out = _resample_vertical_adt(angles, values)
                    num = self._get_int_value(self.vert_num_antennas, 1) if tag == "comp_vrp" else 1
                    write_pat_vertical_new_format(path, desc, 0.0, max(1, num), a_out, v_out, step=1)
                else:
                    a_out, v_out = resample_horizontal(angles, values, norm="none")
                    num = self._get_int_value(self.horz_num_antennas, 1) if tag == "comp_hrp" else 1
                    write_pat_horizontal_new_format(path, desc, 0.0, max(1, num), a_out, v_out, step=1)

                self._register_export(path, f"PROJECT_{tag.upper()}_PAT")
                exported += 1
            except Exception as e:
                messagebox.showwarning("Aviso", f"Falha ao exportar PAT '{tag}': {e}")

        if exported == 0:
            messagebox.showwarning("Sem exportacao", "Nenhum PAT foi exportado.")
            return
        self._set_status(f"Exportacao PAT concluida: {exported} arquivo(s) em {out_dir}.")

    def export_project_adt_files(self):
        out_dir = self._project_prepare_export_dir("pat_adt")
        if not out_dir:
            return

        base = self._project_base_name()
        patterns = self._collect_project_patterns()
        if not patterns:
            messagebox.showwarning("Sem dados", "Nao ha diagramas disponiveis para exportar PAT ADT.")
            return

        exported = 0
        for tag, kind, angles, values in patterns:
            try:
                path = os.path.join(out_dir, f"{base}_{tag}_ADT.pat")
                write_pat_adt_format(path, f"{base} {tag}", angles, values, pattern_type=kind)
                self._register_export(path, f"PROJECT_{tag.upper()}_ADT")
                exported += 1
            except Exception as e:
                messagebox.showwarning("Aviso", f"Falha ao exportar ADT '{tag}': {e}")

        if exported == 0:
            messagebox.showwarning("Sem exportacao", "Nenhum PAT ADT foi exportado.")
            return
        self._set_status(f"Exportacao PAT ADT concluida: {exported} arquivo(s) em {out_dir}.")

    def export_project_prn_files(self):
        out_dir = self._project_prepare_export_dir("prn")
        if not out_dir:
            return

        base = self._project_base_name()
        pairs = self._collect_project_prn_pairs()
        if not pairs:
            messagebox.showwarning("Sem dados", "Nao ha pares H+V disponiveis para exportar PRN.")
            return

        name_base = self.prn_name.get().strip() or "ANTENA_FM"
        make = self.prn_make.get().strip() or "RFS"
        freq = self._safe_meta_float(self.prn_freq.get(), 99.50)
        freq_unit = self.prn_freq_unit.get()
        gain = self._safe_meta_float(self.prn_gain.get(), 2.77)
        h_width = self._safe_meta_float(self.prn_h_width.get(), 65.0)
        v_width = self._safe_meta_float(self.prn_v_width.get(), 45.0)
        fb_ratio = self._safe_meta_float(self.prn_fb_ratio.get(), 25.0)

        exported = 0
        for tag, h_ang, h_val, v_ang, v_val in pairs:
            try:
                h_out, hv = resample_horizontal(h_ang, h_val, norm="none")
                try:
                    v_out, vv = resample_vertical(v_ang, v_val, norm="none")
                except Exception:
                    v_out, vv = _resample_vertical_adt(v_ang, v_val)

                prn_name = f"{name_base}_{tag.upper()}"
                path = os.path.join(out_dir, f"{base}_{tag}.prn")
                write_prn_file(path, prn_name, make, freq, freq_unit, h_width, v_width, fb_ratio, gain, h_out, hv, v_out, vv)
                self._register_export(path, f"PROJECT_{tag.upper()}_PRN")
                exported += 1
            except Exception as e:
                messagebox.showwarning("Aviso", f"Falha ao exportar PRN '{tag}': {e}")

        if exported == 0:
            messagebox.showwarning("Sem exportacao", "Nenhum PRN foi exportado.")
            return
        self._set_status(f"Exportacao PRN concluida: {exported} arquivo(s) em {out_dir}.")

    # ... (rest of PATConverterApp)


    # ==================== ABA 1 Ã¢â‚¬â€ ARQUIVO ==================== #
    def _build_tab_file(self):
        # Container principal com scroll para garantir visualizaÃƒÂ§ÃƒÂ£o em telas menores
        # Usaremos CTkScrollableFrame para a aba inteira se necessÃƒÂ¡rio, 
        # mas como temos plots grandes, melhor dividir em seÃƒÂ§ÃƒÂµes compactas.
        
        # 1. Top Configuration Bar (ConfiguraÃƒÂ§ÃƒÂµes Globais)
        top = ctk.CTkFrame(self.tab_file)
        top.pack(side=ctk.TOP, fill=ctk.X, padx=8, pady=5)

        ctk.CTkLabel(top, text="Base Name:").pack(side=ctk.LEFT, padx=(5, 2))
        ctk.CTkEntry(top, textvariable=self.base_name_var, width=130).pack(side=ctk.LEFT, padx=2)

        ctk.CTkLabel(top, text="Author:").pack(side=ctk.LEFT, padx=(10, 2))
        ctk.CTkEntry(top, textvariable=self.author_var, width=100).pack(side=ctk.LEFT, padx=2)

        ctk.CTkLabel(top, text="Norm:").pack(side=ctk.LEFT, padx=(10, 2))
        ctk.CTkOptionMenu(top, variable=self.norm_mode_var, values=["none", "max", "rms"], width=80).pack(side=ctk.LEFT, padx=2)

        ctk.CTkButton(top, text="Output dir...", command=self.choose_output_dir, width=90).pack(side=ctk.LEFT, padx=(15, 5))
        
        # BotÃƒÂµes de ExportaÃƒÂ§ÃƒÂ£o Global no Topo (para fÃƒÂ¡cil acesso)
        ctk.CTkButton(top, text="Export .PAT (All)", command=self.export_all_pat, fg_color="#22aa66", width=110).pack(side=ctk.RIGHT, padx=5)
        ctk.CTkButton(top, text="Export .PRN (All)", command=self.export_all_prn, fg_color="#aa6622", width=110).pack(side=ctk.RIGHT, padx=5)


        # 2. Main Content Area (Plots + Loaders) - Middle
        middle = ctk.CTkFrame(self.tab_file)
        middle.pack(side=ctk.TOP, fill=ctk.BOTH, expand=True, padx=8, pady=5)
        
        # --- LEFT: Vertical ---
        frame_v = ctk.CTkFrame(middle)
        frame_v.pack(side=ctk.LEFT, fill=ctk.BOTH, expand=True, padx=(0, 4), pady=0)
        
        # Header Row V
        hv_header = ctk.CTkFrame(frame_v, fg_color="transparent")
        hv_header.pack(fill=ctk.X, pady=2, padx=5)
        ctk.CTkLabel(hv_header, text="Vertical (VRP)", font=ctk.CTkFont(size=14, weight="bold")).pack(side=ctk.LEFT)
        
        # Plot Mode Toggles V
        ctk.CTkRadioButton(hv_header, text="Planar", variable=self.v_plot_mode, value="Planar", command=self._refresh_v_plot, width=60).pack(side=ctk.RIGHT)
        ctk.CTkRadioButton(hv_header, text="Polar", variable=self.v_plot_mode, value="Polar", command=self._refresh_v_plot, width=60).pack(side=ctk.RIGHT)

        # Plot
        self.fig_v1 = Figure(figsize=(4, 3), dpi=90)
        self.ax_v1 = self.fig_v1.add_subplot(111)
        self.ax_v1.set_title("Planar View (Empty)")
        self.ax_v1.grid(True, alpha=0.3)
        self.canvas_v1 = FigureCanvasTkAgg(self.fig_v1, master=frame_v)
        self.canvas_v1.get_tk_widget().pack(fill=ctk.BOTH, expand=True, padx=2, pady=2)
        
        # Controls V
        cv = ctk.CTkFrame(frame_v)
        cv.pack(fill=ctk.X, pady=2)
        
        # Row 1: Load/Export PAT
        r1v = ctk.CTkFrame(cv, fg_color="transparent")
        r1v.pack(fill=ctk.X, pady=1)
        ctk.CTkButton(r1v, text="Load VRP...", command=self.load_vertical, height=24, width=80).pack(side=ctk.LEFT, padx=2, expand=True, fill=ctk.X)
        ctk.CTkButton(r1v, text="Exp(.pat)", command=lambda: self.export_single_pat("V", "PAT"), fg_color="green", height=24, width=60).pack(side=ctk.LEFT, padx=2, expand=True, fill=ctk.X)
        ctk.CTkButton(r1v, text="Exp(ADT)", command=lambda: self.export_single_pat("V", "ADT"), fg_color="#33cc99", height=24, width=60).pack(side=ctk.LEFT, padx=2, expand=True, fill=ctk.X)

        # Row 2: Images
        r2v = ctk.CTkFrame(cv, fg_color="transparent")
        r2v.pack(fill=ctk.X, pady=(2, 5))
        ctk.CTkButton(r2v, text="Img Grafico", command=lambda: self.export_plot_img("V"), fg_color="#4477aa", height=24, width=80).pack(side=ctk.LEFT, padx=2, expand=True, fill=ctk.X)
        ctk.CTkButton(r2v, text="Img Tabela", command=lambda: self.export_table_img("V"), fg_color="#aa4477", height=24, width=80).pack(side=ctk.LEFT, padx=2, expand=True, fill=ctk.X)


        # --- RIGHT: Horizontal ---
        frame_h = ctk.CTkFrame(middle)
        frame_h.pack(side=ctk.RIGHT, fill=ctk.BOTH, expand=True, padx=(4, 0), pady=0)
        
        # Header Row H
        hh_header = ctk.CTkFrame(frame_h, fg_color="transparent")
        hh_header.pack(fill=ctk.X, pady=2, padx=5)
        ctk.CTkLabel(hh_header, text="Horizontal (HRP)", font=ctk.CTkFont(size=14, weight="bold")).pack(side=ctk.LEFT)
        
        # Plot Mode Toggles H
        ctk.CTkRadioButton(hh_header, text="Planar", variable=self.h_plot_mode, value="Planar", command=self._refresh_h_plot, width=60).pack(side=ctk.RIGHT)
        ctk.CTkRadioButton(hh_header, text="Polar", variable=self.h_plot_mode, value="Polar", command=self._refresh_h_plot, width=60).pack(side=ctk.RIGHT)
        
        # Plot
        self.fig_h1 = Figure(figsize=(4, 3), dpi=90)
        self.ax_h1 = self.fig_h1.add_subplot(111, projection="polar")
        self.ax_h1.set_title("Polar View (Empty)")
        self.ax_h1.grid(True, alpha=0.3)
        self.canvas_h1 = FigureCanvasTkAgg(self.fig_h1, master=frame_h)
        self.canvas_h1.get_tk_widget().pack(fill=ctk.BOTH, expand=True, padx=2, pady=2)
        
        # Controls H
        ch = ctk.CTkFrame(frame_h)
        ch.pack(fill=ctk.X, pady=2)
        
        # Row 1
        r1h = ctk.CTkFrame(ch, fg_color="transparent")
        r1h.pack(fill=ctk.X, pady=1)
        ctk.CTkButton(r1h, text="Load HRP...", command=self.load_horizontal, height=24, width=80).pack(side=ctk.LEFT, padx=2, expand=True, fill=ctk.X)
        ctk.CTkButton(r1h, text="Exp(.pat)", command=lambda: self.export_single_pat("H", "PAT"), fg_color="green", height=24, width=60).pack(side=ctk.LEFT, padx=2, expand=True, fill=ctk.X)
        ctk.CTkButton(r1h, text="Exp(ADT)", command=lambda: self.export_single_pat("H", "ADT"), fg_color="#33cc99", height=24, width=60).pack(side=ctk.LEFT, padx=2, expand=True, fill=ctk.X)
        
        # Row 2
        r2h = ctk.CTkFrame(ch, fg_color="transparent")
        r2h.pack(fill=ctk.X, pady=(2, 5))
        ctk.CTkButton(r2h, text="Img Grafico", command=lambda: self.export_plot_img("H"), fg_color="#4477aa", height=24, width=80).pack(side=ctk.LEFT, padx=2, expand=True, fill=ctk.X)
        ctk.CTkButton(r2h, text="Img Tabela", command=lambda: self.export_table_img("H"), fg_color="#aa4477", height=24, width=80).pack(side=ctk.LEFT, padx=2, expand=True, fill=ctk.X)


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
        d = filedialog.askdirectory(title="Escolha a pasta de saida para .pat")
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
                write_pat_adt_format(path, base, ang, val, pattern_type=type_)
                self._register_export(path, f"{type_}_ADT_PAT")
                self._set_status(f"Exportado {type_} (ADT): {path}")
            else:
                # Standard PAT
                # Use default gain=0, N=1 for single element export
                if type_ == "V":
                    write_pat_vertical_new_format(path, base+desc_suffix, 0.0, 1, ang, val)
                else:
                    write_pat_horizontal_new_format(path, base+desc_suffix, 0.0, 1, ang, val)
                self._register_export(path, f"{type_}_PAT")
                self._set_status(f"Exportado {type_} (PAT): {path}")
                
        except Exception as e:
            messagebox.showerror("Erro Export", str(e))

    def _refresh_v_plot(self):
        if self.v_angles is None or self.v_vals is None:
            return
        self._plot_vertical_file(self.v_angles, self.v_vals)

    def _refresh_h_plot(self):
        if self.h_angles is None or self.h_vals is None:
            return
        self._plot_horizontal_file(self.h_angles, self.h_vals)

    def export_plot_img(self, type_):
        fname = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG Image", "*.png")])
        if not fname: return
        
        try:
            if type_ == "V":
                if self.v_angles is None or self.v_vals is None:
                    messagebox.showwarning("Aviso", "Nenhum diagrama vertical carregado.")
                    return
                fig, _ = build_diagram_export_figure(
                    kind="V",
                    angles=self.v_angles,
                    values=self.v_vals,
                    title="Diagrama Vertical (Arquivo)",
                    prefer_polar=(self.v_plot_mode.get() == "Polar"),
                )
            else:
                if self.h_angles is None or self.h_vals is None:
                    messagebox.showwarning("Aviso", "Nenhum diagrama horizontal carregado.")
                    return
                fig, _ = build_diagram_export_figure(
                    kind="H",
                    angles=self.h_angles,
                    values=self.h_vals,
                    title="Diagrama Horizontal (Arquivo)",
                    prefer_polar=(self.h_plot_mode.get() == "Polar"),
                )
            fig.savefig(fname, dpi=300)
            self._register_export(fname, f"{type_}_PLOT_IMG")
            self._set_status(f"Imagem salva: {fname}")
        except Exception as e:
            messagebox.showerror("Erro", str(e))

    def export_table_img(self, type_):
        fname = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG Image", "*.png")])
        if not fname: return
        
        try:
            if type_ == "V":
                if self.v_angles is None or self.v_vals is None:
                    messagebox.showwarning("Aviso", "Nenhum diagrama vertical carregado.")
                    return
                target_ang, target_val = self._table_points_vertical(self.v_angles, self.v_vals)
                vals_db = linear_to_db(target_val)
                img = render_table_image(target_ang, vals_db, "dB", "#4e79a7", "Tabela Vertical Padrao")
            else:
                if self.h_angles is None or self.h_vals is None:
                    messagebox.showwarning("Aviso", "Nenhum diagrama horizontal carregado.")
                    return
                target_ang, target_val = self._table_points_horizontal(self.h_angles, self.h_vals)
                vals_db = linear_to_db(target_val)
                img = render_table_image(target_ang, vals_db, "dB", "#f28e2b", "Tabela Horizontal Padrao")
            
            img.save(fname)
            self._register_export(fname, f"{type_}_TABLE_IMG")
            self._set_status(f"Tabela salva: {fname}")
        except Exception as e:
            messagebox.showerror("Erro", str(e))

    def load_vertical(self):
        path = filedialog.askopenfilename(title="Selecione VRP (CSV/TXT)",
                                          filetypes=[('CSV/TXT', '*.csv *.tsv *.txt *.dat *.*')])
        if not path:
            return
        try:
            a, v = parse_auto(path)
            self.v_angles, self.v_vals = a, v
            self._plot_vertical_file(a, v)
            self._set_status(f"VRP carregado ({len(a)} amostras): {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Erro ao carregar VRP", str(e))

    def _plot_vertical_file(self, ang, val):
        a = np.asarray(ang, dtype=float)
        v = np.asarray(val, dtype=float)
        if a.size == 0 or v.size == 0 or a.size != v.size:
            return

        self.fig_v1.clf()
        mode = self.v_plot_mode.get()
        if mode == "Polar":
            ax = self.fig_v1.add_subplot(111, projection="polar")
            ax.set_title("Vertical (VRP) - polar")
            ax.plot(np.deg2rad(a), v, color="tab:blue", linewidth=1.2)
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
            ax.grid(True, alpha=0.3)
        else:
            ax = self.fig_v1.add_subplot(111)
            ax.set_title("Vertical (VRP) - planar")
            ax.plot(a, v, color="tab:blue", linewidth=1.2)
            ax.set_xlabel("Theta [deg]")
            ax.set_ylabel("Amplitude")
            ax.grid(True, alpha=0.3)
            ax.set_xlim([float(np.min(a)), float(np.max(a))])

        self.ax_v1 = ax
        self.canvas_v1.draw_idle()

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

    def _plot_horizontal_file(self, ang, val):
        a = np.asarray(ang, dtype=float)
        v = np.asarray(val, dtype=float)
        if a.size == 0 or v.size == 0 or a.size != v.size:
            return

        self.fig_h1.clf()
        mode = self.h_plot_mode.get()
        if mode == "Planar":
            ax = self.fig_h1.add_subplot(111)
            ax.set_title("Horizontal (HRP) - planar")
            ax.plot(a, v, color="tab:orange", linewidth=1.2)
            ax.set_xlabel("Theta [deg]")
            ax.set_ylabel("Amplitude")
            ax.grid(True, alpha=0.3)
            ax.set_xlim([float(np.min(a)), float(np.max(a))])
        else:
            ax = self.fig_h1.add_subplot(111, projection="polar")
            ax.set_title("Horizontal (HRP) - polar")
            a_wrap = np.mod(a, 360.0)
            order = np.argsort(a_wrap)
            theta = np.deg2rad(a_wrap[order])
            ax.plot(theta, v[order], color="tab:orange", linewidth=1.2)
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
            ax.grid(True, alpha=0.3)

        self.ax_h1 = ax
        self.canvas_h1.draw_idle()

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

        self.fig_v1.clf()
        if self.v_plot_mode.get() == "Polar":
            self.ax_v1 = self.fig_v1.add_subplot(111, projection="polar")
            self.ax_v1.set_title("Vertical (VRP) - polar")
            self.ax_v1.set_theta_zero_location("N")
            self.ax_v1.set_theta_direction(-1)
        else:
            self.ax_v1 = self.fig_v1.add_subplot(111)
            self.ax_v1.set_title("Vertical (VRP) - planar")
            self.ax_v1.set_xlabel("Theta [deg]")
            self.ax_v1.set_ylabel("Amplitude")
        self.ax_v1.grid(True, alpha=0.3)
        self.ax_v1.text(0.5, 0.5, "Sem dados", ha="center", va="center", transform=self.ax_v1.transAxes)
        self.canvas_v1.draw_idle()

        self.fig_h1.clf()
        if self.h_plot_mode.get() == "Planar":
            self.ax_h1 = self.fig_h1.add_subplot(111)
            self.ax_h1.set_title("Horizontal (HRP) - planar")
            self.ax_h1.set_xlabel("Theta [deg]")
            self.ax_h1.set_ylabel("Amplitude")
        else:
            self.ax_h1 = self.fig_h1.add_subplot(111, projection="polar")
            self.ax_h1.set_title("Horizontal (HRP) - polar")
            self.ax_h1.set_theta_zero_location("N")
            self.ax_h1.set_theta_direction(-1)
        self.ax_h1.grid(True, alpha=0.3)
        self.ax_h1.text(0.5, 0.5, "Sem dados", ha="center", va="center", transform=self.ax_h1.transAxes)
        self.canvas_h1.draw_idle()
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
            self._register_export(path_v, "ALL_VRP_PAT")
            self._set_status(f"VRP .PAT exportado: {path_v}")

        if self.h_angles is not None:
            ang_h, val_h = resample_horizontal(self.h_angles, self.h_vals, norm=norm)
            path_h = os.path.join(out_dir, f"{base}_HRP.pat")
            write_pat_horizontal_new_format(path_h, f"{base}_HRP", 0.0, 1, ang_h, val_h)
            self._register_export(path_h, "ALL_HRP_PAT")
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
            self._register_export(file_path, "ALL_PRN")
            self._set_status(f"Arquivo .PRN exportado: {file_path}")
            
        except Exception as e:
            messagebox.showerror("Erro ao exportar .PRN", str(e))

    # ==================== ABA 2 Ã¢â‚¬â€ COMPOSIÃƒâ€¡ÃƒÆ’O VERTICAL ==================== #
    def _build_tab_vertical(self):
        main_frame = ctk.CTkFrame(self.tab_vert)
        main_frame.pack(fill=ctk.BOTH, expand=True, padx=8, pady=8)

        input_frame = ctk.CTkFrame(main_frame, width=350)
        input_frame.pack(side=ctk.LEFT, fill=ctk.Y, padx=(0, 8), pady=8)
        input_frame.pack_propagate(False)

        plot_frame = ctk.CTkFrame(main_frame)
        plot_frame.pack(side=ctk.RIGHT, fill=ctk.BOTH, expand=True, padx=8, pady=8)

        ctk.CTkLabel(
            input_frame,
            text="Composicao Vertical - Elevacao",
            font=ctk.CTkFont(weight="bold"),
        ).pack(pady=(8, 10))

        vcmd_float = (self.register(validate_float), "%P")
        vcmd_int = (self.register(validate_int), "%P")

        # Variaveis
        self.vert_N = tk.StringVar(value="4")
        self.vert_freq = tk.StringVar(value="0.9")
        self.vert_funit = tk.StringVar(value="GHz")
        self.vert_beta = tk.StringVar(value="0.0")
        self.vert_level = tk.StringVar(value="1.0")
        self.vert_space = tk.StringVar(value="0.5")
        self.vert_norm = tk.StringVar(value="max")
        self.vert_tilt_elec_deg = tk.StringVar(value="0.0")
        self.vert_null_fill_pct = tk.StringVar(value="20.0")
        self.vert_null_mode = tk.StringVar(value="both")
        self.vert_null_order = tk.StringVar(value="1")
        self.vert_fill_weight = tk.StringVar(value="32.0")
        self.vert_main_weight = tk.StringVar(value="30.0")
        self.vert_reg_lambda = tk.StringVar(value="1e-5")
        self.vert_max_iters = tk.StringVar(value="24")
        self.vert_vf = tk.StringVar(value="0.78")

        self.vert_desc = tk.StringVar(value="Array Vertical")
        self.vert_gain = tk.StringVar(value="0.0")
        self.vert_step = tk.StringVar(value="1")
        self.vert_num_antennas = tk.StringVar(value="4")

        def create_param_row(parent, label, variable, values=None, width=120, validate=None):
            row = ctk.CTkFrame(parent)
            row.pack(fill=ctk.X, padx=8, pady=3)
            ctk.CTkLabel(row, text=label, width=118, anchor="w").pack(side=ctk.LEFT)
            if values:
                widget = ctk.CTkOptionMenu(row, variable=variable, values=values, width=width)
            else:
                kwargs = {"textvariable": variable, "width": width}
                if validate == "float":
                    kwargs.update({"validate": "key", "validatecommand": vcmd_float})
                elif validate == "int":
                    kwargs.update({"validate": "key", "validatecommand": vcmd_int})
                widget = ctk.CTkEntry(row, **kwargs)
            widget.pack(side=ctk.RIGHT, padx=(8, 0))
            return row

        essentials = ctk.CTkFrame(input_frame)
        essentials.pack(fill=ctk.X, padx=8, pady=(0, 8))
        ctk.CTkLabel(essentials, text="Parametros Essenciais", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=8, pady=(6, 4))

        create_param_row(essentials, "N antenas:", self.vert_N, validate="int")

        freq_row = ctk.CTkFrame(essentials)
        freq_row.pack(fill=ctk.X, padx=8, pady=3)
        ctk.CTkLabel(freq_row, text="Frequencia:", width=118, anchor="w").pack(side=ctk.LEFT)
        ctk.CTkEntry(
            freq_row,
            textvariable=self.vert_freq,
            width=78,
            validate="key",
            validatecommand=vcmd_float,
        ).pack(side=ctk.LEFT, padx=(8, 4))
        ctk.CTkOptionMenu(freq_row, variable=self.vert_funit, values=["Hz", "kHz", "MHz", "GHz"], width=80).pack(side=ctk.LEFT)

        create_param_row(essentials, "Espacamento d [m]:", self.vert_space, validate="float")
        create_param_row(essentials, "Tilt eletrico [deg]:", self.vert_tilt_elec_deg, validate="float")
        create_param_row(essentials, "Null fill [%]:", self.vert_null_fill_pct, validate="float")
        create_param_row(essentials, "Nulo alvo:", self.vert_null_order, ["1", "2", "3", "4", "5"], width=120)
        create_param_row(essentials, "Modo null fill:", self.vert_null_mode, ["amplitude", "phase", "both"], width=140)

        ctk.CTkButton(
            essentials,
            text="Calcular Diagrama de Elevacao",
            command=self.compute_vertical_array,
            fg_color="#2277cc",
        ).pack(fill=ctk.X, padx=8, pady=(8, 6))

        export_frame = ctk.CTkFrame(input_frame)
        export_frame.pack(fill=ctk.X, padx=8, pady=(0, 8))
        ctk.CTkLabel(export_frame, text="Exportacao", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=8, pady=(6, 4))
        btns = ctk.CTkFrame(export_frame, fg_color="transparent")
        btns.pack(fill=ctk.X, padx=6, pady=(0, 6))
        ctk.CTkButton(btns, text="PAT", command=self.export_vertical_array_pat, fg_color="#22aa66", width=78).pack(side=ctk.LEFT, padx=2, expand=True, fill=ctk.X)
        ctk.CTkButton(btns, text="PRN", command=self.export_vertical_array_prn, fg_color="#aa6622", width=78).pack(side=ctk.LEFT, padx=2, expand=True, fill=ctk.X)
        ctk.CTkButton(btns, text="Harness", command=self.export_vertical_harness, fg_color="#4455aa", width=90).pack(side=ctk.LEFT, padx=2, expand=True, fill=ctk.X)

        self._vert_adv_visible = tk.BooleanVar(value=False)

        def _toggle_adv():
            visible = self._vert_adv_visible.get()
            if visible:
                adv_frame.pack_forget()
                self._vert_adv_visible.set(False)
                adv_btn.configure(text="Mostrar Opcoes Avancadas")
            else:
                adv_frame.pack(fill=ctk.X, padx=8, pady=(0, 8))
                self._vert_adv_visible.set(True)
                adv_btn.configure(text="Ocultar Opcoes Avancadas")

        adv_btn = ctk.CTkButton(
            input_frame,
            text="Mostrar Opcoes Avancadas",
            command=_toggle_adv,
            fg_color="#505e75",
        )
        adv_btn.pack(fill=ctk.X, padx=8, pady=(0, 8))

        adv_frame = ctk.CTkFrame(input_frame)
        ctk.CTkLabel(adv_frame, text="Avancado", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=8, pady=(6, 4))
        create_param_row(adv_frame, "Beta [deg/elem]:", self.vert_beta, validate="float")
        create_param_row(adv_frame, "Nivel (amp):", self.vert_level, validate="float")
        create_param_row(adv_frame, "Normalizar:", self.vert_norm, ["none", "max", "rms"], width=140)
        create_param_row(adv_frame, "Forca no nulo:", self.vert_fill_weight, validate="float")
        create_param_row(adv_frame, "Preservar lobo principal:", self.vert_main_weight, validate="float")
        create_param_row(adv_frame, "Reg lambda:", self.vert_reg_lambda, validate="float")
        create_param_row(adv_frame, "Iteracoes:", self.vert_max_iters, validate="int")
        create_param_row(adv_frame, "VF cabo:", self.vert_vf, validate="float")
        create_param_row(adv_frame, "Descricao PAT:", self.vert_desc, width=180)
        create_param_row(adv_frame, "Ganho [dB]:", self.vert_gain, validate="float")
        create_param_row(adv_frame, "N ant export:", self.vert_num_antennas, validate="int")
        create_param_row(adv_frame, "Passo [deg]:", self.vert_step, ["1", "2", "3", "4", "5"], width=120)

        # Painel direito: grafico + campo tecnico
        right_grid = ctk.CTkFrame(plot_frame, fg_color="transparent")
        right_grid.pack(fill=ctk.BOTH, expand=True, padx=4, pady=4)
        right_grid.grid_rowconfigure(0, weight=1)
        right_grid.grid_columnconfigure(0, weight=3)
        right_grid.grid_columnconfigure(1, weight=2)

        graph_card = ctk.CTkFrame(right_grid)
        graph_card.grid(row=0, column=0, sticky="nsew", padx=(0, 6), pady=0)

        self.fig_v2 = Figure(figsize=(8, 6), dpi=100)
        self.ax_v2 = self.fig_v2.add_subplot(111)
        self.ax_v2.set_title("VRP Composto - Array Vertical")
        self.ax_v2.set_xlabel("Theta [deg]")
        self.ax_v2.set_ylabel("E/Emax (linear)")
        self.ax_v2.grid(True, alpha=0.3)
        self.canvas_v2 = FigureCanvasTkAgg(self.fig_v2, master=graph_card)
        self.canvas_v2.get_tk_widget().pack(fill=ctk.BOTH, expand=True, padx=8, pady=8)

        info_card = ctk.CTkFrame(right_grid)
        info_card.grid(row=0, column=1, sticky="nsew", padx=(6, 0), pady=0)

        ctk.CTkLabel(info_card, text="Metricas do Diagrama", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=8, pady=(8, 4))
        self.vert_peak = tk.StringVar(value="Pico: -")
        self.vert_hpbw = tk.StringVar(value="HPBW: -")
        self.vert_d2d = tk.StringVar(value="D2D: -")
        ctk.CTkLabel(info_card, textvariable=self.vert_peak).pack(anchor="w", padx=8, pady=2)
        ctk.CTkLabel(info_card, textvariable=self.vert_hpbw).pack(anchor="w", padx=8, pady=2)
        ctk.CTkLabel(info_card, textvariable=self.vert_d2d).pack(anchor="w", padx=8, pady=2)

        ctk.CTkLabel(info_card, text="Dados da Composicao", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=8, pady=(10, 4))
        self.vert_comp_info_text = ctk.CTkTextbox(info_card, wrap="none")
        self.vert_comp_info_text.pack(fill=ctk.BOTH, expand=True, padx=8, pady=(0, 8))
        self.vert_comp_info_text.insert(
            "0.0",
            "Calcule o diagrama para visualizar os dados tecnicos da composicao.",
        )
        self.vert_comp_info_text.configure(state="disabled")
        # Compatibilidade com metodos existentes
        self.vert_weights_text = self.vert_comp_info_text

        self.vert_angles = None
        self.vert_values = None
        self.vert_synth_result = None
        self.vert_harness = None

    def _freq_to_hz(self, val: float, unit: str) -> float:
        unit = unit.lower()
        if unit == "hz":  return val
        if unit == "khz": return val * 1e3
        if unit == "mhz": return val * 1e6
        if unit == "ghz": return val * 1e9
        return val

    def _get_float_value(self, var: tk.StringVar, default: float = 0.0) -> float:
        """ObtÃƒÂ©m valor float de StringVar com tratamento de erro"""
        try:
            return float(var.get())
        except (ValueError, tk.TclError):
            return default

    def _get_int_value(self, var: tk.StringVar, default: int = 0) -> int:
        """ObtÃƒÂ©m valor int de StringVar com tratamento de erro"""
        try:
            return int(var.get())
        except (ValueError, tk.TclError):
            return default

    def _update_vertical_weights_view(
        self,
        w_complex: np.ndarray,
        harness: dict,
        mode: str,
        synth_info: Optional[dict] = None,
    ):
        if not hasattr(self, "vert_weights_text"):
            return

        amp = np.abs(np.asarray(w_complex, dtype=complex).reshape(-1))
        n = len(amp)
        if n == 0:
            return

        p_frac = np.asarray(harness.get("p_frac", np.zeros(n)), dtype=float).reshape(-1)
        phase_deg = np.asarray(harness.get("phase_deg", np.zeros(n)), dtype=float).reshape(-1)
        delta_len_m = np.asarray(harness.get("delta_len_m", np.zeros(n)), dtype=float).reshape(-1)
        att_db_ref = np.asarray(harness.get("att_db_ref", np.zeros(n)), dtype=float).reshape(-1)

        cond_txt = "-"
        peak_eps = "-"
        null_summary = []
        null_order = "-"
        target_percent = "-"
        achieved_percent = "-"
        phase_limit = "-"
        if isinstance(synth_info, dict):
            cond_val = synth_info.get("condition_number", None)
            if cond_val is not None and math.isfinite(float(cond_val)):
                cond_txt = f"{float(cond_val):.3e}"
            peak_val = synth_info.get("peak_eps_deg", None)
            if peak_val is not None and math.isfinite(float(peak_val)):
                peak_eps = f"{float(peak_val):.2f} deg"
            null_order = str(synth_info.get("null_order", "-"))
            tp = synth_info.get("target_percent", None)
            if tp is not None and math.isfinite(float(tp)):
                target_percent = f"{float(tp):.2f}%"
            ap = synth_info.get("achieved_percent", None)
            if ap is not None and math.isfinite(float(ap)):
                achieved_percent = f"{float(ap):.2f}%"
            pl = synth_info.get("phase_limit_deg", None)
            if pl is not None and math.isfinite(float(pl)):
                phase_limit = f"{float(pl):.1f} deg"
            for nl in synth_info.get("null_levels", []) or []:
                try:
                    side = str(nl.get("side", "?")).strip()
                    tgt_db = nl.get("target_db", None)
                    tgt_txt = (
                        f"{float(tgt_db):.2f} dB"
                        if tgt_db is not None and math.isfinite(float(tgt_db))
                        else "-"
                    )
                    ach = nl.get("achieved_percent", None)
                    ach_txt = (
                        f"{float(ach):.2f}%"
                        if ach is not None and math.isfinite(float(ach))
                        else "-"
                    )
                    null_summary.append(
                        f"Nulo {side} @{float(nl.get('eps_deg', 0.0)):.2f} deg: {float(nl.get('initial_db', 0.0)):.2f} -> {float(nl.get('final_db', 0.0)):.2f} dB (alvo {tgt_txt}, atingido {ach_txt})"
                    )
                except Exception:
                    continue

        freq_hz = self._freq_to_hz(self._get_float_value(self.vert_freq, 0.0), self.vert_funit.get())
        lam0 = C0 / max(freq_hz, 1.0)
        lines = [
            f"Modo de controle: {mode}",
            f"Niveis (N): {self._get_int_value(self.vert_N, n)}",
            f"Frequencia: {self.vert_freq.get()} {self.vert_funit.get()} (lambda0={lam0:.4f} m)",
            f"Espacamento: {self._get_float_value(self.vert_space, 0.0):.4f} m",
            f"Tilt eletrico: {self._get_float_value(self.vert_tilt_elec_deg, 0.0):.2f} deg",
            f"Null fill alvo: {target_percent}",
            f"Null fill atingido (medio): {achieved_percent}",
            f"Nulo selecionado: {null_order}o",
            f"Pico em elevacao: {peak_eps}",
            f"Limite automatico de fase: {phase_limit}",
            f"Condicionamento LSQ: {cond_txt}",
            "",
            "Evolucao do(s) nulo(s):",
        ]
        lines.extend(null_summary if null_summary else ["-"])
        lines.extend([
            "",
            "Nivel |   |w|    | Pot[%] | Fase[deg] | AttRef[dB] | DeltaL[m]",
            "---------------------------------------------------------------",
        ])

        for idx in range(1, n + 1):
            a = float(amp[idx - 1])
            p = float(100.0 * p_frac[idx - 1]) if idx - 1 < p_frac.size else 0.0
            ph = float(phase_deg[idx - 1]) if idx - 1 < phase_deg.size else 0.0
            at = float(att_db_ref[idx - 1]) if idx - 1 < att_db_ref.size else 0.0
            dl = float(delta_len_m[idx - 1]) if idx - 1 < delta_len_m.size else 0.0
            pos = "inferior" if idx == 1 else ("superior" if idx == n else "intermediario")
            lines.append(
                f"{idx:>3d} ({pos}) | {a:>8.5f} | {p:>6.2f} | {ph:>9.3f} | {at:>10.3f} | {dl:>8.5f}"
            )

        box = self.vert_weights_text
        box.configure(state="normal")
        box.delete("0.0", "end")
        box.insert("0.0", "\n".join(lines))
        box.configure(state="disabled")

    def compute_vertical_array(self):
        if self.v_angles is None or self.v_vals is None:
            messagebox.showwarning("Dados faltando", "Carregue o VRP na aba Arquivo antes.")
            return
        try:
            # Reamostra o elemento (VRP) na grade alvo
            base_angles, base_vals = resample_vertical(self.v_angles, self.v_vals, norm=self.vert_norm.get())

            # Parametros de sintese
            N = max(1, self._get_int_value(self.vert_N, 4))
            freq_val = self._get_float_value(self.vert_freq, 0.9)
            f_hz = self._freq_to_hz(freq_val, self.vert_funit.get())
            z_step_m = self._get_float_value(self.vert_space, 0.5)
            z_m = np.arange(N, dtype=float) * z_step_m
            beta_deg = self._get_float_value(self.vert_beta, 0.0)
            tilt_deg = self._get_float_value(self.vert_tilt_elec_deg, 0.0)
            mode = self.vert_null_mode.get().strip().lower()
            if mode not in ("amplitude", "phase", "both"):
                messagebox.showwarning("Modo invalido", "Escolha um modo: amplitude, phase ou both.")
                return

            prompt_map = {
                "amplitude": "Amplitude: altera potencia por baia.",
                "phase": "Fase: altera comprimento eletrico (fase).",
                "both": "Ambos: melhor desempenho, com maior complexidade de harness.",
            }
            if not messagebox.askyesno(
                "Modo de controle do Null Fill",
                f"Modo selecionado: {mode}\n{prompt_map.get(mode, '')}\n\nDeseja continuar o calculo?",
            ):
                return

            null_order = max(1, self._get_int_value(self.vert_null_order, 1))
            fill_weight = self._get_float_value(self.vert_fill_weight, 32.0)
            preserve_main_weight = self._get_float_value(self.vert_main_weight, 12.0)
            reg_lambda = self._get_float_value(self.vert_reg_lambda, 1e-5)
            max_iters = max(1, self._get_int_value(self.vert_max_iters, 8))
            vf = self._get_float_value(self.vert_vf, 0.78)
            amp_fixed = np.full(N, abs(self._get_float_value(self.vert_level, 1.0)), dtype=float)

            null_fill_pct = self._get_float_value(self.vert_null_fill_pct, 0.0)
            pct = max(0.0, min(100.0, null_fill_pct))

            # Autoajuste para garantir foco no nulo com baixa deformacao do lobo principal.
            if pct > 0.0:
                max_iters = max(max_iters, 30)
                reg_lambda = min(reg_lambda, 5e-5)
                fill_weight = max(fill_weight, 32.0)
                preserve_main_weight = max(preserve_main_weight, 30.0)

            elem_pattern = lambda eps_deg: np.interp(
                np.asarray(eps_deg, dtype=float),
                base_angles,
                base_vals,
                left=float(base_vals[0]),
                right=float(base_vals[-1]),
            )

            synth = synth_null_fill_by_order(
                f_hz=f_hz,
                z_m=z_m,
                eps_grid_deg=base_angles,
                null_order=null_order,
                null_fill_percent=pct,
                mode=mode,
                mainlobe_tilt_deg=tilt_deg,
                elem_pattern=elem_pattern,
                reg_lambda=reg_lambda,
                max_iters=max_iters,
                preserve_mainlobe_weight=preserve_main_weight,
                fill_weight=fill_weight,
                progressive_phase_deg_per_elem=beta_deg,
                amp_fixed=amp_fixed if mode == "phase" else None,
            )

            E_comp = np.abs(np.asarray(synth["E_final"], dtype=complex))
            E_ini = np.abs(np.asarray(synth["E_initial"], dtype=complex))
            E_comp = E_comp / (np.max(E_comp) if np.max(E_comp) > 0 else 1.0)
            E_ini = E_ini / (np.max(E_ini) if np.max(E_ini) > 0 else 1.0)

            harness = weights_to_harness(np.asarray(synth["w"], dtype=complex), f_hz=f_hz, vf=vf, ref_index=0)
            self.vert_synth_result = synth
            self.vert_harness = harness
            self._update_vertical_weights_view(np.asarray(synth["w"], dtype=complex), harness, mode, synth)

            # MÃƒÂ©tricas
            peak = float(np.max(E_comp))
            hpbw = hpbw_deg(base_angles, E_comp)
            d2d  = directivity_2d_cut(base_angles, E_comp, span_deg=180.0)
            d2d_db = 10.0 * math.log10(d2d) if d2d > 0 else float("nan")

            # Plot
            self._plot_vertical_composite(
                base_angles,
                E_comp,
                values_initial=E_ini,
                null_regions=synth.get("null_regions", []),
            )

            # Atualiza labels e buffers
            self.vert_peak.set(f"Pico: {peak:.3f}")
            self.vert_hpbw.set(f"HPBW: {hpbw:.2f} deg" if math.isfinite(hpbw) else "HPBW: -")
            d2d_text = f"D2D: {d2d:.3f} ({d2d_db:.2f} dB)" if math.isfinite(d2d) else "D2D: -"
            self.vert_d2d.set(d2d_text)
            self.vert_angles = base_angles
            self.vert_values = E_comp
            self._set_status("VRP composto calculado com preenchimento por ordem de nulo.")

        except Exception as e:
            messagebox.showerror("Erro (Vertical)", str(e))

    def _plot_vertical_composite(
        self,
        angles: np.ndarray,
        values: np.ndarray,
        values_initial: Optional[np.ndarray] = None,
        null_regions: Optional[list] = None,
    ):
        self.ax_v2.cla()
        self.ax_v2.set_title("VRP Composto - Array Vertical")
        self.ax_v2.set_xlabel("Theta [deg]")
        self.ax_v2.set_ylabel("E/Emax (linear)")
        self.ax_v2.grid(True, alpha=0.3)
        if values_initial is not None:
            self.ax_v2.plot(angles, values_initial, linewidth=1.1, color="#888888", linestyle="--", label="Inicial")
        self.ax_v2.plot(angles, values, linewidth=1.5, color='blue', label="Final")
        if null_regions:
            for r in null_regions:
                c = float(r.get("eps_deg", 0.0))
                hw = float(r.get("half_width_deg", 0.5))
                self.ax_v2.axvspan(c - hw, c + hw, color="#66cc99", alpha=0.14)
                self.ax_v2.axvline(c, color="#228b22", linestyle=":", linewidth=1.1)
        self.ax_v2.set_xlim([-90, 90])
        self.ax_v2.set_ylim([0, 1.1])
        if values_initial is not None or null_regions:
            self.ax_v2.legend(loc="upper right", fontsize=8)
        self.canvas_v2.draw()

    def export_vertical_harness(self):
        if self.vert_synth_result is None or self.vert_harness is None:
            messagebox.showwarning("Nada para exportar", "Execute o calculo de null fill antes.")
            return

        out_dir = self.output_dir or os.getcwd()
        base = self.base_name_var.get().strip() or "xxx"
        csv_path = os.path.join(out_dir, f"{base}_nullfill_harness.csv")
        json_path = os.path.join(out_dir, f"{base}_nullfill_harness.json")
        plot_path = os.path.join(out_dir, f"{base}_nullfill_af.png")

        try:
            w = np.asarray(self.vert_synth_result.get("w"), dtype=complex).reshape(-1)
            eps = np.asarray(self.vert_synth_result.get("eps_deg"), dtype=float).reshape(-1)
            e_ini = np.abs(np.asarray(self.vert_synth_result.get("E_initial"), dtype=complex))
            e_fin = np.abs(np.asarray(self.vert_synth_result.get("E_final"), dtype=complex))
            e_ini = e_ini / (np.max(e_ini) if np.max(e_ini) > 0 else 1.0)
            e_fin = e_fin / (np.max(e_fin) if np.max(e_fin) > 0 else 1.0)
            ini_db = 20.0 * np.log10(np.maximum(e_ini, 1e-12))
            fin_db = 20.0 * np.log10(np.maximum(e_fin, 1e-12))

            p_frac = np.asarray(self.vert_harness.get("p_frac"), dtype=float).reshape(-1)
            phase_deg = np.asarray(self.vert_harness.get("phase_deg"), dtype=float).reshape(-1)
            delta_len = np.asarray(self.vert_harness.get("delta_len_m"), dtype=float).reshape(-1)
            att_db_ref = np.asarray(self.vert_harness.get("att_db_ref"), dtype=float).reshape(-1)

            with open(csv_path, "w", encoding="utf-8", newline="\n") as f:
                writer = csv.writer(f)
                writer.writerow(["baia", "w_real", "w_imag", "amp", "p_frac", "phase_deg", "att_db_ref", "delta_len_m"])
                for i in range(len(w)):
                    writer.writerow([
                        i,
                        float(np.real(w[i])),
                        float(np.imag(w[i])),
                        float(np.abs(w[i])),
                        float(p_frac[i]) if i < p_frac.size else 0.0,
                        float(phase_deg[i]) if i < phase_deg.size else 0.0,
                        float(att_db_ref[i]) if i < att_db_ref.size else 0.0,
                        float(delta_len[i]) if i < delta_len.size else 0.0,
                    ])

            payload = {
                "mode": self.vert_synth_result.get("mode", "both"),
                "null_order": self.vert_synth_result.get("null_order", 1),
                "target_percent": self.vert_synth_result.get("target_percent", 0.0),
                "achieved_percent": self.vert_synth_result.get("achieved_percent", 0.0),
                "phase_limit_deg": self.vert_synth_result.get("phase_limit_deg", None),
                "null_regions": self.vert_synth_result.get("null_regions", []),
                "null_levels": self.vert_synth_result.get("null_levels", []),
                "condition_number": float(self.vert_synth_result.get("condition_number", float("nan"))),
                "peak_eps_deg": float(self.vert_synth_result.get("peak_eps_deg", float("nan"))),
                "weights": [{"real": float(np.real(x)), "imag": float(np.imag(x))} for x in w],
                "harness": {
                    "amp": [float(x) for x in np.abs(w)],
                    "p_frac": [float(x) for x in p_frac],
                    "phase_deg": [float(x) for x in phase_deg],
                    "att_db_ref": [float(x) for x in att_db_ref],
                    "delta_len_m": [float(x) for x in delta_len],
                },
            }
            with open(json_path, "w", encoding="utf-8", newline="\n") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

            fig = Figure(figsize=(8, 4.5), dpi=120)
            ax = fig.add_subplot(111)
            ax.set_title("AF Inicial vs Final (dB de campo)")
            ax.set_xlabel("Elevacao [deg]")
            ax.set_ylabel("Amplitude [dB]")
            ax.grid(True, alpha=0.3)
            ax.plot(eps, ini_db, "--", color="#888888", linewidth=1.1, label="Inicial")
            ax.plot(eps, fin_db, "-", color="#1f77b4", linewidth=1.5, label="Final")
            for r in (self.vert_synth_result.get("null_regions", []) or []):
                c = float(r.get("eps_deg", 0.0))
                hw = float(r.get("half_width_deg", 0.5))
                ax.axvspan(c - hw, c + hw, color="#66cc99", alpha=0.15)
                ax.axvline(c, color="#228b22", linestyle=":", linewidth=1.1)
            try:
                m = compute_diagram_metrics("V", eps, e_fin)
                txt = "\n".join(format_diagram_metric_lines(m))
                ax.text(
                    0.99,
                    0.99,
                    txt,
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=8.2,
                    bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor="#666666", alpha=0.90),
                )
            except Exception:
                pass
            ax.set_xlim([-90, 90])
            ax.legend(loc="best", fontsize=8)
            fig.savefig(plot_path, dpi=300)

            self._register_export(csv_path, "VERT_NULLFILL_HARNESS_CSV")
            self._register_export(json_path, "VERT_NULLFILL_HARNESS_JSON")
            self._register_export(plot_path, "VERT_NULLFILL_AF_IMG")
            self._set_status(f"Harness exportado: {csv_path}, {json_path}, {plot_path}")
        except Exception as e:
            messagebox.showerror("Erro export harness", str(e))

    def export_vertical_array_pat(self):
        if self.vert_angles is None or self.vert_values is None:
            messagebox.showwarning("Nada para exportar", "Execute o cÃƒÂ¡lculo da composiÃƒÂ§ÃƒÂ£o vertical primeiro.")
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
        self._register_export(path_v, "VERT_COMP_PAT")
        self._set_status(f"VRP composto .PAT exportado: {path_v}")

    def export_vertical_array_prn(self):
        if self.vert_angles is None or self.vert_values is None:
            messagebox.showwarning("Nada para exportar", "Execute o cÃƒÂ¡lculo da composiÃƒÂ§ÃƒÂ£o vertical primeiro.")
            return
            
        if self.h_angles is None or self.h_vals is None:
            messagebox.showwarning("Dados incompletos", "Para exportar .PRN ÃƒÂ© necessÃƒÂ¡rio ter o HRP carregado na aba Arquivo.")
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
            self._register_export(file_path, "VERT_COMP_PRN")
            self._set_status(f"Arquivo .PRN exportado: {file_path}")
            
        except Exception as e:
            messagebox.showerror("Erro ao exportar .PRN", str(e))

    # ==================== ABA 3 Ã¢â‚¬â€ COMPOSIÃƒâ€¡ÃƒÆ’O HORIZONTAL ==================== #
    def _build_tab_horizontal(self):
        # Frame principal dividido em entrada e grÃƒÂ¡fico
        main_frame = ctk.CTkFrame(self.tab_horz)
        main_frame.pack(fill=ctk.BOTH, expand=True, padx=8, pady=8)
        
        # Frame de entrada ÃƒÂ  esquerda
        input_frame = ctk.CTkFrame(main_frame)
        input_frame.pack(side=ctk.LEFT, fill=ctk.Y, padx=(0, 8), pady=8)
        
        # Frame do grÃƒÂ¡fico ÃƒÂ  direita
        plot_frame = ctk.CTkFrame(main_frame)
        plot_frame.pack(side=ctk.RIGHT, fill=ctk.BOTH, expand=True, padx=8, pady=8)

        # TÃƒÂ­tulo
        ctk.CTkLabel(input_frame, text="ParÃƒÂ¢metros do Array Horizontal", 
                    font=ctk.CTkFont(weight="bold")).pack(pady=(8, 12))

        # ValidaÃƒÂ§ÃƒÂ£o para campos numÃƒÂ©ricos
        vcmd_float = (self.register(validate_float), '%P')
        vcmd_int = (self.register(validate_int), '%P')

        # ParÃƒÂ¢metros organizados verticalmente
        self.horz_N       = tk.StringVar(value="4")
        self.horz_beta    = tk.StringVar(value="0.0")
        self.horz_level   = tk.StringVar(value="1.0")
        self.horz_spacing = tk.StringVar(value="2.0")
        self.horz_stepdeg = tk.StringVar(value="90.0")
        self.horz_freq    = tk.StringVar(value="0.9")
        self.horz_funit   = tk.StringVar(value="GHz")
        self.horz_norm    = tk.StringVar(value="max")

        # ParÃƒÂ¢metros de exportaÃƒÂ§ÃƒÂ£o
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

        create_param_row(input_frame, "N paineis:", self.horz_N, validate="int")
        
        freq_row = ctk.CTkFrame(input_frame)
        freq_row.pack(fill=ctk.X, padx=8, pady=4)
        ctk.CTkLabel(freq_row, text="Frequencia:", width=100).pack(side=ctk.LEFT)
        ctk.CTkEntry(freq_row, textvariable=self.horz_freq, width=80, validate="key", 
                    validatecommand=vcmd_float).pack(side=ctk.LEFT, padx=(8, 4))
        ctk.CTkOptionMenu(freq_row, variable=self.horz_funit, values=["Hz","kHz","MHz","GHz"], width=70).pack(side=ctk.LEFT)
        
        create_param_row(input_frame, "Beta [deg/painel]:", self.horz_beta, validate="float")
        create_param_row(input_frame, "Nivel (amp.):", self.horz_level, validate="float")
        create_param_row(input_frame, "Esp. s [m]:", self.horz_spacing, validate="float")
        create_param_row(input_frame, "DeltaPhi [deg]:", self.horz_stepdeg, validate="float")
        create_param_row(input_frame, "Normalizar:", self.horz_norm, ["none","max","rms"])

        # SeÃƒÂ§ÃƒÂ£o de exportaÃƒÂ§ÃƒÂ£o
        export_frame = ctk.CTkFrame(input_frame)
        export_frame.pack(fill=ctk.X, padx=8, pady=12)
        ctk.CTkLabel(export_frame, text="Exportacao", font=ctk.CTkFont(weight="bold")).pack(pady=(4, 8))
        
        create_param_row(export_frame, "Descricao:", self.horz_desc)
        create_param_row(export_frame, "Ganho [dB]:", self.horz_gain, validate="float")
        create_param_row(export_frame, "N Antenas:", self.horz_num_antennas, validate="int")
        create_param_row(export_frame, "Passo [deg]:", self.horz_step, ["1","2","3","4","5"])

        # BotÃƒÂµes de exportaÃƒÂ§ÃƒÂ£o
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

        # Metricas
        metrics_frame = ctk.CTkFrame(input_frame)
        metrics_frame.pack(fill=ctk.X, padx=8, pady=8)
        ctk.CTkLabel(metrics_frame, text="Metricas", font=ctk.CTkFont(weight="bold")).pack(pady=(4, 8))
        
        self.horz_peak = tk.StringVar(value="Pico: -")
        self.horz_hpbw = tk.StringVar(value="HPBW: -")
        self.horz_d2d  = tk.StringVar(value="D2D: -")
        
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

            # ParÃƒÂ¢metros com tratamento de erro
            N   = self._get_int_value(self.horz_N, 4)
            beta_deg = self._get_float_value(self.horz_beta, 0.0)
            w    = self._get_float_value(self.horz_level, 1.0)
            s    = self._get_float_value(self.horz_spacing, 2.0)
            dphi_deg = self._get_float_value(self.horz_stepdeg, 90.0)

            f_hz = self._freq_to_hz(self._get_float_value(self.horz_freq, 0.9), self.horz_funit.get())
            lam = C0 / max(f_hz, 1.0)
            k   = 2.0 * math.pi / lam

            # CÃƒÂLCULO CORRETO: PosiÃƒÂ§ÃƒÂµes dos painÃƒÂ©is em um polÃƒÂ­gono regular no plano horizontal
            alpha_m_deg = np.arange(N) * dphi_deg
            alpha_m_rad = np.deg2rad(alpha_m_deg)
            
            # Raio do polÃƒÂ­gono para que a distÃƒÂ¢ncia entre painÃƒÂ©is adjacentes seja s
            if N > 1:
                R = s / (2 * np.sin(np.pi / N))
            else:
                R = 0

            # Inicializar o campo composto
            E_comp = np.zeros(len(base_angles), dtype=complex)

            # Para cada ÃƒÂ¢ngulo de observaÃƒÂ§ÃƒÂ£o phi (azimute)
            phi_rad = np.deg2rad(base_angles)
            
            for i in range(len(base_angles)):
                phi = phi_rad[i]
                E_total = 0.0 + 0.0j
                
                # Para cada painel no array
                for m in range(N):
                    # PosiÃƒÂ§ÃƒÂ£o do painel m
                    x_m = R * np.cos(alpha_m_rad[m])
                    y_m = R * np.sin(alpha_m_rad[m])
                    
                    # Vetor de direÃƒÂ§ÃƒÂ£o de observaÃƒÂ§ÃƒÂ£o
                    u_x = np.cos(phi)
                    u_y = np.sin(phi)
                    
                    # DiferenÃƒÂ§a de caminho para o painel m
                    delta_r = x_m * u_x + y_m * u_y
                    
                    # Fase devido ÃƒÂ  diferenÃƒÂ§a de caminho
                    phase_geom = k * delta_r
                    
                    # Fase progressiva (excitaÃƒÂ§ÃƒÂ£o)
                    phase_excit = np.deg2rad(m * beta_deg)
                    
                    # Ãƒâ€šngulo relativo entre a direÃƒÂ§ÃƒÂ£o de observaÃƒÂ§ÃƒÂ£o e a orientaÃƒÂ§ÃƒÂ£o do painel
                    rel_angle_deg = (base_angles[i] - alpha_m_deg[m]) % 360
                    if rel_angle_deg > 180:
                        rel_angle_deg -= 360
                    
                    # Diagrama do elemento na direÃƒÂ§ÃƒÂ£o relativa
                    E_elem = np.interp(rel_angle_deg, base_angles, base_vals)
                    
                    # ContribuiÃƒÂ§ÃƒÂ£o complexa do painel m
                    E_total += w * E_elem * np.exp(1j * (phase_geom + phase_excit))
                
                E_comp[i] = E_total

            # Tomar magnitude e normalizar
            E_comp_mag = np.abs(E_comp)
            if np.max(E_comp_mag) > 0:
                E_comp_mag = E_comp_mag / np.max(E_comp_mag)

            # MÃƒÂ©tricas
            peak = float(np.max(E_comp_mag))
            hpbw = hpbw_deg(base_angles, E_comp_mag)
            d2d  = directivity_2d_cut(base_angles, E_comp_mag, span_deg=360.0)
            d2d_db = 10.0 * math.log10(d2d) if d2d > 0 else float("nan")

            # Plot POLAR
            self._plot_horizontal_composite(base_angles, E_comp_mag)

            # Atualiza labels e buffers
            self.horz_peak.set(f"Pico: {peak:.3f}")
            self.horz_hpbw.set(f"HPBW: {hpbw:.2f} deg" if math.isfinite(hpbw) else "HPBW: -")
            d2d_text = f"D2D: {d2d:.3f} ({d2d_db:.2f} dB)" if math.isfinite(d2d) else "D2D: -"
            self.horz_d2d.set(d2d_text)
            self.horz_angles = base_angles
            self.horz_values = E_comp_mag
            self._set_status("HRP composto calculado.")

        except Exception as e:
            messagebox.showerror("Erro (Horizontal)", str(e))

    def _plot_horizontal_composite(self, angles: np.ndarray, values: np.ndarray):
        self.ax_h2.cla()
        self.ax_h2 = self.fig_h2.add_subplot(111, projection='polar')
        self.ax_h2.set_title("HRP Composto - Array Horizontal", pad=20)
        self.ax_h2.grid(True, alpha=0.3)

        theta_plot = np.deg2rad(angles)
        theta_plot = (theta_plot + np.pi / 2) % (2 * np.pi)

        self.ax_h2.plot(theta_plot, values, linewidth=1.5, color='red')
        self.ax_h2.set_theta_zero_location('N')
        self.ax_h2.set_theta_direction(-1)
        self.canvas_h2.draw()

    def export_horizontal_array_pat(self):
        if self.horz_angles is None or self.horz_values is None:
            messagebox.showwarning("Nada para exportar", "Execute o cÃƒÂ¡lculo da composiÃƒÂ§ÃƒÂ£o horizontal primeiro.")
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
        self._register_export(path_h, "HORZ_COMP_PAT")
        self._set_status(f"HRP composto .PAT exportado: {path_h}")

    def export_horizontal_array_rfs(self):
        if self.horz_angles is None or self.horz_values is None:
            messagebox.showwarning("Nada para exportar", "Execute o cÃƒÂ¡lculo da composiÃƒÂ§ÃƒÂ£o horizontal primeiro.")
            return
        
        base = self.base_name_var.get().strip() or "xxx"
        out_dir = self.output_dir or os.getcwd()
        
        description = self.horz_desc.get() or f"{base}_HRP_composto"
        gain = self._get_float_value(self.horz_gain, 0.0)
        num_antennas = self._get_int_value(self.horz_num_antennas, 4)
        
        path_h = os.path.join(out_dir, f"{base}_HRP_composto_RFS.pat")
        
        try:
             write_pat_horizontal_rfs(path_h, description, gain, num_antennas, self.horz_angles, self.horz_values)
             self._register_export(path_h, "HORZ_COMP_RFS_PAT")
             self._set_status(f"HRP RFS .PAT exportado: {path_h}")
        except Exception as e:
             messagebox.showerror("Erro Export RFS", str(e))

    def export_horizontal_array_prn(self):
        if self.horz_angles is None or self.horz_values is None:
            messagebox.showwarning("Nada para exportar", "Execute o cÃƒÂ¡lculo da composiÃƒÂ§ÃƒÂ£o horizontal primeiro.")
            return
            
        if self.v_angles is None or self.v_vals is None:
            messagebox.showwarning("Dados incompletos", "Para exportar .PRN ÃƒÂ© necessÃƒÂ¡rio ter o VRP carregado na aba Arquivo.")
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
            self._register_export(file_path, "HORZ_COMP_PRN")
            self._set_status(f"Arquivo .PRN exportado: {file_path}")
            
        except Exception as e:
            messagebox.showerror("Erro ao exportar .PRN", str(e))

    # ----------------------------- Misc ----------------------------- #
    def _set_status(self, text: str):
        self.status.configure(text=text)
        if hasattr(self, "project_info_box"):
            self._refresh_project_overview()

if __name__ == "__main__":
    app = PATConverterApp()
    app.mainloop()

