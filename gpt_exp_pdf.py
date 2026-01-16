# -*- coding: utf-8 -*-
"""
CTk PAT Converter — VRP/HRP + Export PDF (template overlay)

Novidades:
- Botão "Exportar PDF" no topo.
- Modal com formulário (cliente, projeto, modelo, frequência, data, autor, observações, etc).
- Geração de overlay (ReportLab) + merge com template PDF (PyPDF2).
- Insere imagens dos diagramas (VRP/HRP das Abas e compostos se existirem)
  e métricas (pico, HPBW, D2D).

Ajustes finos de layout:
- Edite o dicionário PDF_PLACEMENT para alinhar exatamente com seu
  template: /mnt/data/Antena_Log_Banda_Larga_2024-1.pdf

Dependências:
    customtkinter numpy matplotlib reportlab pypdf2 pillow
"""

from __future__ import annotations

import io
import os
import re
import sys
import math
import tempfile
import datetime as dt
import tkinter as tk
from tkinter import filedialog, messagebox
from typing import List, Tuple, Optional

import numpy as np
import customtkinter as ctk
import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from reportlab.pdfgen import canvas as rl_canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from PyPDF2 import PdfReader, PdfWriter, PdfMerger

from PIL import Image

# ----------------------------- Parsing & Resampling ----------------------------- #
NUM_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")

def _is_float(s: str) -> bool:
    return bool(NUM_RE.match(s.strip()))

def parse_pattern_table(path: str) -> Tuple[np.ndarray, np.ndarray]:
    angles: List[float] = []
    vals: List[float] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            normalized = line.replace(",", ".")
            parts = re.split(r"[\t\s;]+", normalized)
            nums = [p for p in parts if _is_float(p)]
            if len(nums) < 2:
                continue
            fnums = [float(x) for x in nums]
            if len(fnums) >= 3:
                angle_deg = fnums[1]; value = fnums[2]
            else:
                angle_deg = fnums[0]; value = fnums[1]
            if math.isfinite(angle_deg) and math.isfinite(value):
                angles.append(angle_deg); vals.append(value)
    if not angles:
        raise ValueError("Nenhum dado numérico válido encontrado no arquivo.")
    a = np.asarray(angles, dtype=float); v = np.asarray(vals, dtype=float)
    order = np.argsort(a)
    return a[order], v[order]

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
    v_acc = np.zeros_like(a_unique); cnt = np.zeros_like(a_unique)
    for i, vi in zip(idx, v):
        v_acc[i] += vi; cnt[i] += 1
    v_mean = v_acc / np.maximum(cnt, 1)
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

# ----------------------------- Métricas ----------------------------- #
def simpson(y: np.ndarray, dx: float) -> float:
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
    if len(angles_deg) < 3:
        return float("nan")
    e = e_lin / (np.max(e_lin) if np.max(e_lin) > 0 else 1.0)
    thr = math.sqrt(0.5)
    i0 = int(np.argmax(e))
    aL = None; aR = None
    for i in range(i0, 0, -1):
        if e[i] >= thr and e[i-1] < thr:
            a1, a2 = angles_deg[i-1], angles_deg[i]
            y1, y2 = e[i-1], e[i]
            aL = a1 + (thr - y1) * (a2 - a1) / (y2 - y1)
            break
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

# ----------------------------- Constantes/Layout PDF ----------------------------- #
# Coordenadas em pontos (1pt = 1/72 in). A4 = 595 x 842 pts approx.
# Ajuste estas caixas para casar com seu template.
PDF_PLACEMENT = {
    "page": 0,                       # usa a primeira página do template
    "page_size": A4,                 # fallback se não conseguirmos ler tamanho do template
    "fields": {
        "cliente":   (60, 770),
        "projeto":   (60, 750),
        "modelo":    (60, 730),
        "freq":      (340, 770),
        "data":      (340, 750),
        "autor":     (340, 730),
        "obs":       (60, 700),      # bloco de observações (texto multi-linha)
    },
    # Caixas de imagens (x, y, width, height)
    "images": {
        "vrp_file":      (50, 420, 230, 180),
        "hrp_file":      (315, 420, 230, 180),
        "vrp_comp":      (50, 220, 230, 180),
        "hrp_comp":      (315, 220, 230, 180),
    },
    # Onde imprimir as métricas (Emax, HPBW, D2D) de cada gráfico
    "metrics": {
        "vrp_file":  (50, 605),
        "hrp_file":  (315, 605),
        "vrp_comp":  (50, 205),
        "hrp_comp":  (315, 205),
        # formato de 3 linhas
    }
}

# Caminho default do template (ajuste se necessário)
DEFAULT_TEMPLATE_PATH = "/mnt/data/Antena_Log_Banda_Larga_2024-1.pdf"

# ----------------------------- GUI ----------------------------- #
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class PATConverterApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("PAT Converter — VRP/HRP (CTk) + PDF")
        self.geometry("1200x780")

        # Estado base
        self.vertical_path: Optional[str] = None
        self.horizontal_path: Optional[str] = None
        self.base_name_var = tk.StringVar(value="xxx")
        self.author_var = tk.StringVar(value="gecesar")
        self.norm_mode_var = tk.StringVar(value="none")  # none, max, rms
        self.output_dir: Optional[str] = None

        # Dados carregados
        self.v_angles = None; self.v_vals = None
        self.h_angles = None; self.h_vals = None

        # Resultado composição (se existir no seu fluxo)
        self.vert_angles = None; self.vert_values = None  # VRP composto
        self.horz_angles = None; self.horz_values = None  # HRP composto

        # Métricas — se você preencher em outras abas, podemos mostrar no PDF
        self.vert_peak = tk.StringVar(value="Pico: —")
        self.vert_hpbw = tk.StringVar(value="HPBW: —")
        self.vert_d2d  = tk.StringVar(value="D₂D: —")

        self.horz_peak = tk.StringVar(value="Pico: —")
        self.horz_hpbw = tk.StringVar(value="HPBW: —")
        self.horz_d2d  = tk.StringVar(value="D₂D: —")

        self._build_ui()

    # ---------------- UI ---------------- #
    def _build_ui(self):
        top = ctk.CTkFrame(self)
        top.pack(side=ctk.TOP, fill=ctk.X, padx=12, pady=10)

        ctk.CTkLabel(top, text="Base name:").pack(side=ctk.LEFT, padx=(8, 4))
        ctk.CTkEntry(top, textvariable=self.base_name_var, width=140).pack(side=ctk.LEFT)

        ctk.CTkLabel(top, text="Author:").pack(side=ctk.LEFT, padx=(16, 4))
        ctk.CTkEntry(top, textvariable=self.author_var, width=140).pack(side=ctk.LEFT)

        ctk.CTkLabel(top, text="Normalize:").pack(side=ctk.LEFT, padx=(16, 4))
        ctk.CTkOptionMenu(top, variable=self.norm_mode_var, values=["none", "max", "rms"]).pack(side=ctk.LEFT)

        ctk.CTkButton(top, text="Output dir…", command=self.choose_output_dir).pack(side=ctk.LEFT, padx=(16, 6))
        ctk.CTkButton(top, text="Export PAT", command=self.export_all, fg_color="#22aa66").pack(side=ctk.LEFT, padx=6)
        ctk.CTkButton(top, text="Exportar PDF", command=self.open_pdf_modal, fg_color="#aa6622").pack(side=ctk.LEFT, padx=6)

        loaders = ctk.CTkFrame(self)
        loaders.pack(side=ctk.TOP, fill=ctk.X, padx=12, pady=(0, 8))
        ctk.CTkButton(loaders, text="Load Vertical Table (VRP)…", command=self.load_vertical).pack(side=ctk.LEFT, padx=6)
        ctk.CTkButton(loaders, text="Load Horizontal Table (HRP)…", command=self.load_horizontal).pack(side=ctk.LEFT, padx=6)
        ctk.CTkButton(loaders, text="Clear", command=self.clear_all).pack(side=ctk.LEFT, padx=6)

        plots = ctk.CTkFrame(self)
        plots.pack(side=ctk.TOP, fill=ctk.BOTH, expand=True, padx=12, pady=8)

        # VRP planar
        self.fig_v = Figure(figsize=(5.8, 3.9), dpi=100)
        self.ax_v = self.fig_v.add_subplot(111)
        self.ax_v.set_title("Vertical (VRP) — planar")
        self.ax_v.set_xlabel("Theta [deg]")
        self.ax_v.set_ylabel("E/Emax (linear)")
        self.ax_v.grid(True, alpha=0.3)
        self.canvas_v = FigureCanvasTkAgg(self.fig_v, master=plots)
        self.canvas_v.get_tk_widget().pack(side=ctk.LEFT, fill=ctk.BOTH, expand=True, padx=6, pady=6)

        # HRP polar
        self.fig_h = Figure(figsize=(5.8, 3.9), dpi=100)
        self.ax_h = self.fig_h.add_subplot(111, projection="polar")
        self.ax_h.set_title("Horizontal (HRP) — polar")
        self.ax_h.grid(True, alpha=0.3)
        self.canvas_h = FigureCanvasTkAgg(self.fig_h, master=plots)
        self.canvas_h.get_tk_widget().pack(side=ctk.LEFT, fill=ctk.BOTH, expand=True, padx=6, pady=6)

        self.status = ctk.CTkLabel(self, text="Pronto.")
        self.status.pack(side=ctk.BOTTOM, fill=ctk.X, padx=12, pady=8)

    # ------------- Ações base ------------- #
    def choose_output_dir(self):
        d = filedialog.askdirectory(title="Escolha a pasta de saída")
        if d:
            self.output_dir = d
            self._set_status(f"Output dir: {d}")

    def load_vertical(self):
        path = filedialog.askopenfilename(title="Selecione VRP (CSV/TXT)",
                                          filetypes=[('Text/CSV', '*.txt *.csv *.tsv *.dat *.*')])
        if not path:
            return
        try:
            a, v = parse_pattern_table(path)
            self.v_angles, self.v_vals = a, v
            self._plot_vertical(a, v)
            self.vertical_path = path
            self._set_status(f"VRP carregado: {os.path.basename(path)} — {len(a)} amostras")
        except Exception as e:
            messagebox.showerror("Erro ao carregar VRP", str(e))

    def load_horizontal(self):
        path = filedialog.askopenfilename(title="Selecione HRP (CSV/TXT)",
                                          filetypes=[('Text/CSV', '*.txt *.csv *.tsv *.dat *.*')])
        if not path:
            return
        try:
            a, v = parse_pattern_table(path)
            self.h_angles, self.h_vals = a, v
            self._plot_horizontal(a, v)
            self.horizontal_path = path
            self._set_status(f"HRP carregado: {os.path.basename(path)} — {len(a)} amostras")
        except Exception as e:
            messagebox.showerror("Erro ao carregar HRP", str(e))

    def clear_all(self):
        self.vertical_path = None
        self.horizontal_path = None
        self.v_angles = self.v_vals = None
        self.h_angles = self.h_vals = None
        self.vert_angles = self.vert_values = None
        self.horz_angles = self.horz_values = None
        self.ax_v.cla(); self.ax_v.set_title("Vertical (VRP) — planar"); self.ax_v.set_xlabel("Theta [deg]"); self.ax_v.set_ylabel("E/Emax (linear)"); self.ax_v.grid(True, alpha=0.3); self.canvas_v.draw()
        self.ax_h.cla(); self.ax_h = self.fig_h.add_subplot(111, projection="polar"); self.ax_h.set_title("Horizontal (HRP) — polar"); self.ax_h.grid(True, alpha=0.3); self.canvas_h.draw()
        self._set_status("Limpo.")

    def export_all(self):
        base = self.base_name_var.get().strip() or "xxx"
        author = self.author_var.get().strip() or "gecesar"
        norm = self.norm_mode_var.get()
        out_dir = self.output_dir or os.getcwd()

        if self.v_angles is not None:
            try:
                ang_v, val_v = resample_vertical(self.v_angles, self.v_vals, norm=norm)
                v_path = os.path.join(out_dir, f"{base}_VRP.pat")
                write_pat_vertical(v_path, author, ang_v, val_v)
                self._set_status(f"VRP salvo: {v_path}")
            except Exception as e:
                messagebox.showerror("Erro exportando VRP", str(e))
        if self.h_angles is not None:
            try:
                ang_h, val_h = resample_horizontal(self.h_angles, self.h_vals, norm=norm)
                h_path = os.path.join(out_dir, f"{base}_HRP.pat")
                write_pat_horizontal(h_path, author, ang_h, val_h)
                self._set_status(f"HRP salvo: {h_path}")
            except Exception as e:
                messagebox.showerror("Erro exportando HRP", str(e))

        # Atualiza plots com dados reamostrados (informativo)
        try:
            if self.v_angles is not None:
                self._plot_vertical(ang_v, val_v, resampled=True)
            if self.h_angles is not None:
                self._plot_horizontal(ang_h, val_h, resampled=True)
        except Exception:
            pass

    # ------------- Plot helpers ------------- #
    def _plot_vertical(self, angles: np.ndarray, values: np.ndarray, resampled: bool = False):
        self.ax_v.cla()
        self.ax_v.set_title("Vertical (VRP) — planar" + (" [reamostrado]" if resampled else ""))
        self.ax_v.set_xlabel("Theta [deg]")
        self.ax_v.set_ylabel("E/Emax (linear)")
        self.ax_v.grid(True, alpha=0.3)
        self.ax_v.plot(angles, values, linewidth=1.3)
        self.ax_v.set_xlim([-90, 90]); self.ax_v.set_ylim(bottom=0)
        self.canvas_v.draw()

    def _plot_horizontal(self, angles: np.ndarray, values: np.ndarray, resampled: bool = False):
        self.ax_h.cla(); self.ax_h = self.fig_h.add_subplot(111, projection="polar")
        self.ax_h.set_title("Horizontal (HRP) — polar" + (" [reamostrado]" if resampled else ""))
        self.ax_h.grid(True, alpha=0.3)
        ang_wrapped = (angles + 360.0) % 360.0
        theta = np.deg2rad(ang_wrapped)
        self.ax_h.plot(theta, values, linewidth=1.2)
        self.canvas_h.draw()

    # --------------------- PDF Export --------------------- #
    def open_pdf_modal(self):
        dlg = ctk.CTkToplevel(self)
        dlg.title("Exportar PDF — Preencha os campos")
        dlg.geometry("520x520")
        dlg.grab_set()

        # Variáveis do formulário
        cliente = tk.StringVar(value="")
        projeto = tk.StringVar(value=self.base_name_var.get())
        modelo  = tk.StringVar(value="—")
        freq    = tk.StringVar(value="")
        data    = tk.StringVar(value=dt.date.today().isoformat())
        autor   = tk.StringVar(value=self.author_var.get())
        obs     = tk.StringVar(value="")

        # Template + saída
        template_path = tk.StringVar(value=DEFAULT_TEMPLATE_PATH)
        out_dir = self.output_dir or os.getcwd()
        pdf_out = tk.StringVar(value=os.path.join(out_dir, f"{self.base_name_var.get().strip() or 'xxx'}_relatorio.pdf"))

        def row(parent, label, var, width=320):
            fr = ctk.CTkFrame(parent)
            fr.pack(fill=ctk.X, padx=10, pady=4)
            ctk.CTkLabel(fr, text=label, width=90).pack(side=ctk.LEFT)
            ctk.CTkEntry(fr, textvariable=var, width=width).pack(side=ctk.LEFT, padx=(8,0))
            return fr

        row(dlg, "Cliente:", cliente)
        row(dlg, "Projeto:", projeto)
        row(dlg, "Modelo:",  modelo)
        row(dlg, "Frequência:", freq)
        row(dlg, "Data:", data)
        row(dlg, "Autor:", autor)

        fr_obs = ctk.CTkFrame(dlg); fr_obs.pack(fill=ctk.X, padx=10, pady=6)
        ctk.CTkLabel(fr_obs, text="Observações:").pack(anchor="w")
        obs_box = ctk.CTkTextbox(fr_obs, height=90, width=480)
        obs_box.pack(fill=ctk.BOTH, expand=True, pady=4)
        obs_box.insert("1.0", "")

        fr_paths = ctk.CTkFrame(dlg); fr_paths.pack(fill=ctk.X, padx=10, pady=6)
        ctk.CTkLabel(fr_paths, text="Template PDF:").grid(row=0, column=0, sticky="w", padx=(0,6))
        e1 = ctk.CTkEntry(fr_paths, textvariable=template_path, width=360); e1.grid(row=0, column=1, padx=4)
        ctk.CTkButton(fr_paths, text="…", width=40,
                      command=lambda: self._choose_file_into_var(template_path, ("PDF", "*.pdf"))).grid(row=0, column=2)

        ctk.CTkLabel(fr_paths, text="Salvar em:").grid(row=1, column=0, sticky="w", padx=(0,6), pady=(6,0))
        e2 = ctk.CTkEntry(fr_paths, textvariable=pdf_out, width=360); e2.grid(row=1, column=1, padx=4, pady=(6,0))
        ctk.CTkButton(fr_paths, text="…", width=40,
                      command=lambda: self._choose_save_into_var(pdf_out, "relatorio.pdf")).grid(row=1, column=2, pady=(6,0))

        fr_btn = ctk.CTkFrame(dlg); fr_btn.pack(fill=ctk.X, padx=10, pady=12)
        def on_ok():
            obs.set(obs_box.get("1.0", "end").strip())
            try:
                self._export_pdf_impl(template_path.get(), pdf_out.get(),
                                      {
                                        "cliente": cliente.get(),
                                        "projeto": projeto.get(),
                                        "modelo":  modelo.get(),
                                        "freq":    freq.get(),
                                        "data":    data.get(),
                                        "autor":   autor.get(),
                                        "obs":     obs.get(),
                                      })
                messagebox.showinfo("PDF", f"Relatório salvo em:\n{pdf_out.get()}")
                dlg.destroy()
            except Exception as e:
                messagebox.showerror("Erro ao gerar PDF", str(e))
        ctk.CTkButton(fr_btn, text="Gerar PDF", fg_color="#aa6622", command=on_ok).pack(side=ctk.RIGHT, padx=6)
        ctk.CTkButton(fr_btn, text="Cancelar", command=dlg.destroy).pack(side=ctk.RIGHT)

    def _choose_file_into_var(self, tkvar: tk.StringVar, ftypes):
        p = filedialog.askopenfilename(title="Selecione o template PDF", filetypes=[ftypes, ('Todos', '*.*')])
        if p:
            tkvar.set(p)

    def _choose_save_into_var(self, tkvar: tk.StringVar, default_name: str):
        p = filedialog.asksaveasfilename(title="Salvar PDF", defaultextension=".pdf",
                                         initialfile=default_name, filetypes=[('PDF', '*.pdf')])
        if p:
            tkvar.set(p)

    # -------- Render helpers -------- #
    def _save_fig_to_png_bytes(self, fig: Figure, dpi: int = 140) -> Optional[bytes]:
        try:
            bio = io.BytesIO()
            fig.savefig(bio, format="png", dpi=dpi, bbox_inches="tight")
            return bio.getvalue()
        except Exception:
            return None

    def _image_fit_into_box(self, pil_img: Image.Image, box_w: int, box_h: int) -> Image.Image:
        # preserva aspecto e letterbox com fundo branco
        img = pil_img.convert("RGB")
        iw, ih = img.size
        scale = min(box_w/iw, box_h/ih)
        new_w = max(1, int(iw*scale)); new_h = max(1, int(ih*scale))
        img2 = img.resize((new_w, new_h), Image.LANCZOS)
        canvas = Image.new("RGB", (box_w, box_h), (255, 255, 255))
        off_x = (box_w - new_w)//2; off_y = (box_h - new_h)//2
        canvas.paste(img2, (off_x, off_y))
        return canvas

    def _collect_metrics_text(self, label_peak: tk.StringVar, label_hpbw: tk.StringVar, label_d2d: tk.StringVar) -> str:
        # Espera strings já no formato "Pico: x", "HPBW: y", "D₂D: z"
        lines = []
        if label_peak.get() != "Pico: —":  lines.append(label_peak.get())
        if label_hpbw.get() != "HPBW: —":  lines.append(label_hpbw.get())
        if label_d2d.get() != "D₂D: —":    lines.append(label_d2d.get())
        return "\n".join(lines) if lines else ""

    # -------- PDF impl -------- #
    def _export_pdf_impl(self, template_path: str, out_pdf_path: str, form: dict):
        # Ler template
        if not os.path.isfile(template_path):
            raise FileNotFoundError(f"Template PDF não encontrado:\n{template_path}")
        reader = PdfReader(template_path)
        page_index = PDF_PLACEMENT.get("page", 0)
        if page_index >= len(reader.pages):
            page_index = 0
        page = reader.pages[page_index]
        # Tamanho real da página
        try:
            media_box = page.mediabox
            page_w = float(media_box.width)
            page_h = float(media_box.height)
        except Exception:
            page_w, page_h = PDF_PLACEMENT.get("page_size", A4)

        # Renderizar figuras existentes em PNGs
        images_bytes = {}
        # (1) VRP da Aba Arquivo (self.fig_v)
        if self.v_angles is not None and self.v_vals is not None:
            img = self._save_fig_to_png_bytes(self.fig_v)
            if img: images_bytes["vrp_file"] = img
        # (2) HRP da Aba Arquivo (self.fig_h)
        if self.h_angles is not None and self.h_vals is not None:
            img = self._save_fig_to_png_bytes(self.fig_h)
            if img: images_bytes["hrp_file"] = img
        # (3) VRP composto (se você alimentar self.vert_angles/values + plot dedicado)
        # Se não houver figura dedicada, usamos a mesma self.fig_v (já mostra VRP atual)
        if self.vert_angles is not None and self.vert_values is not None:
            # Cria uma figura temporária só para composto (planar)
            fig_tmp = Figure(figsize=(5.0, 3.4), dpi=110)
            ax = fig_tmp.add_subplot(111)
            ax.set_title("VRP Composto")
            ax.set_xlabel("Theta [deg]"); ax.set_ylabel("E/Emax")
            ax.grid(True, alpha=0.3)
            ax.plot(self.vert_angles, self.vert_values, linewidth=1.3)
            ax.set_xlim([-90, 90]); ax.set_ylim(bottom=0)
            img = self._save_fig_to_png_bytes(fig_tmp)
            if img: images_bytes["vrp_comp"] = img
        # (4) HRP composto (polar)
        if self.horz_angles is not None and self.horz_values is not None:
            fig_tmp = Figure(figsize=(5.0, 3.4), dpi=110)
            ax = fig_tmp.add_subplot(111, projection='polar')
            ax.set_title("HRP Composto"); ax.grid(True, alpha=0.3)
            ang_wrapped = (self.horz_angles + 360.0) % 360.0
            ax.plot(np.deg2rad(ang_wrapped), self.horz_values, linewidth=1.3)
            img = self._save_fig_to_png_bytes(fig_tmp)
            if img: images_bytes["hrp_comp"] = img

        # Criar overlay (mesmo tamanho da página do template)
        tmpdir = tempfile.mkdtemp(prefix="patpdf_")
        overlay_path = os.path.join(tmpdir, "overlay.pdf")
        c = rl_canvas.Canvas(overlay_path, pagesize=(page_w, page_h))

        # Escrever campos de texto
        c.setFont("Helvetica", 11)
        fields = PDF_PLACEMENT["fields"]
        def draw_text(label, value):
            if value:
                x, y = fields[label]
                c.drawString(x, y, str(value))

        draw_text("cliente", form.get("cliente", ""))
        draw_text("projeto", form.get("projeto", ""))
        draw_text("modelo",  form.get("modelo",  ""))
        draw_text("freq",    form.get("freq",    ""))
        draw_text("data",    form.get("data",    ""))
        draw_text("autor",   form.get("autor",   ""))

        # Observações (multilinha, quebra simples)
        obs_txt = form.get("obs", "")
        if obs_txt:
            x, y = fields["obs"]
            c.setFont("Helvetica", 10)
            max_chars = 92
            lines = []
            for para in obs_txt.splitlines():
                s = para.strip()
                while len(s) > max_chars:
                    lines.append(s[:max_chars])
                    s = s[max_chars:]
                if s:
                    lines.append(s)
            # desenha até ~5 linhas
            for i, ln in enumerate(lines[:5]):
                c.drawString(x, y - 14*i, ln)

        # Inserir imagens nas caixas
        img_boxes = PDF_PLACEMENT["images"]
        for key, bbox in img_boxes.items():
            if key in images_bytes:
                x, y, w, h = bbox
                pil = Image.open(io.BytesIO(images_bytes[key]))
                pil_fit = self._image_fit_into_box(pil, int(w), int(h))
                c.drawImage(ImageReader(pil_fit), x, y, width=w, height=h, preserveAspectRatio=False, mask=None)

        # Inserir métricas se disponíveis (usa labels existentes; ajuste conforme suas abas)
        metric_pos = PDF_PLACEMENT["metrics"]
        c.setFont("Helvetica", 10)
        # VRP arquivo
        txt = self._collect_metrics_text(self.vert_peak, self.vert_hpbw, self.vert_d2d)
        if txt and "vrp_file" in metric_pos:
            x, y = metric_pos["vrp_file"]
            for i, ln in enumerate(txt.splitlines()):
                c.drawString(x, y - 12*i, ln)

        # HRP arquivo
        txt_h = self._collect_metrics_text(self.horz_peak, self.horz_hpbw, self.horz_d2d)
        if txt_h and "hrp_file" in metric_pos:
            x, y = metric_pos["hrp_file"]
            for i, ln in enumerate(txt_h.splitlines()):
                c.drawString(x, y - 12*i, ln)

        # VRP composto — se quiser usar outras métricas específicas do composto, adapte aqui
        if txt and "vrp_comp" in metric_pos:
            x, y = metric_pos["vrp_comp"]
            for i, ln in enumerate(txt.splitlines()):
                c.drawString(x, y - 12*i, ln)

        # HRP composto
        if txt_h and "hrp_comp" in metric_pos:
            x, y = metric_pos["hrp_comp"]
            for i, ln in enumerate(txt_h.splitlines()):
                c.drawString(x, y - 12*i, ln)

        c.showPage()
        c.save()

        # Mesclar overlay com a página do template
        writer = PdfWriter()
        for i in range(len(reader.pages)):
            pg = reader.pages[i]
            if i == page_index:
                # carrega overlay como página
                over_reader = PdfReader(overlay_path)
                over_page = over_reader.pages[0]
                pg.merge_page(over_page)
            writer.add_page(pg)

        # Salvar final
        with open(out_pdf_path, "wb") as f:
            writer.write(f)

        self._set_status(f"PDF exportado: {out_pdf_path}")

    # ----------------------------- Misc ----------------------------- #
    def _set_status(self, text: str):
        self.status.configure(text=text)


if __name__ == "__main__":
    app = PATConverterApp()
    app.mainloop()
