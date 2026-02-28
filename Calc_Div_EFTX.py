import os
import sys
import warnings
from pathlib import Path

if getattr(sys, "frozen", False):
    _dll_dirs = []
    _seen_dll_dirs = set()

    if hasattr(sys, "_MEIPASS"):
        _mei_dir = Path(getattr(sys, "_MEIPASS"))
        _dll_dirs.extend([
            _mei_dir,
            _mei_dir / "vtk.libs",
            _mei_dir / "vtkmodules",
        ])

    _exe_dir = Path(sys.executable).resolve().parent
    _dll_dirs.extend([
        _exe_dir / "vtk.libs",
        _exe_dir / "vtkmodules",
        _exe_dir / "lib" / "vtk.libs",
        _exe_dir / "lib" / "vtkmodules",
    ])

    for _dll_dir in _dll_dirs:
        _dir_key = str(_dll_dir).lower()
        if _dir_key in _seen_dll_dirs or not _dll_dir.is_dir():
            continue
        _seen_dll_dirs.add(_dir_key)
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(str(_dll_dir))
        os.environ["PATH"] = str(_dll_dir) + os.pathsep + os.environ.get("PATH", "")

import customtkinter as ctk
import pyvista as pv
import numpy as np
from skrf.plotting import plot_s_smith

import matplotlib.pyplot as plt


import threading, queue, math, traceback, json, tempfile
from abc import ABC, abstractmethod
from enum import Enum
from datetime import datetime

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import tkinter.filedialog as fd
from tkinter import messagebox
import skrf as rf
from skrf.network import Network
from skrf.frequency import Frequency
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle,
    Paragraph, Image as RLImage, Spacer
)
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from skrf.media import DistributedCircuit

# --------------------------------------------------------------------
# 1. Imports HFSS/AEDT
# --------------------------------------------------------------------
try:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*Graphics dependencies are required.*",
            category=UserWarning,
        )
        from ansys.aedt.core import Hfss

    HFSS_AVAILABLE = True
except ImportError:
    HFSS_AVAILABLE = False
    print("ansys-aedt-core não encontrado. Funcionalidades HFSS desativadas.")

# --------------------------------------------------------------------
# 2. Configuração CTk
# --------------------------------------------------------------------
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


# --------------------------------------------------------------------
# 3. Topologia
# --------------------------------------------------------------------
class Topology(Enum):
    COAXIAL = "Coaxial"


# --------------------------------------------------------------------
# 4. Materiais
# --------------------------------------------------------------------
SUBSTRATE_MATERIALS = {
    "Ar": (1.0006, 0.0),
    "Teflon (PTFE)": (2.1, 0.0002),
    "FR-4": (4.4, 0.02),
    "Rogers RO4003C": (3.55, 0.0027),
}


def _get_user_appdata_dir(app_name="DIVISOR"):
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        base_dir = Path(local_app_data)
    else:
        # fallback para cenarios sem LOCALAPPDATA definido
        base_dir = Path.home() / "AppData" / "Local"
    app_dir = base_dir / app_name
    app_dir.mkdir(parents=True, exist_ok=True)
    return app_dir


def _get_output_diel_diameter_mm(d_int_tube_mm, n_outputs, len_inner_mm=None, min_hf_scale=0.70,
                                 ref_len_to_diam_ratio=10.0):
    if n_outputs > 6:
        base_fraction = 0.25
    elif n_outputs > 4:
        base_fraction = 0.32
    else:
        base_fraction = 0.40

    if len_inner_mm is None:
        hf_scale = 1.0
    else:
        length_ratio = len_inner_mm / max(d_int_tube_mm, 1e-9)
        normalized_ratio = max(min(length_ratio / ref_len_to_diam_ratio, 1.0), 0.0)
        hf_scale = min_hf_scale + (1.0 - min_hf_scale) * math.sqrt(normalized_ratio)

    return d_int_tube_mm * base_fraction * hf_scale


def _get_output_length_mm(len_inner_mm, d_int_tube_mm, base_ratio=0.1, min_tube_diam_factor=1.0):
    # Comprimento depende somente do transformador e nao pode ser menor que o diametro do tubo.
    return max(len_inner_mm * base_ratio, d_int_tube_mm * min_tube_diam_factor)


# --------------------------------------------------------------------
# 5. Classe Base de Cálculo
# --------------------------------------------------------------------
class RFCalculator(ABC):
    def __init__(self, params):
        self.params = params
        self.results = {}
        self.s_params_th = None
        self.s_params_sim = None
        self.project_path = ""
        self.frequencies = np.linspace(params['f_start'], params['f_stop'], 201)

    @abstractmethod
    def calculate(self):
        pass

    @abstractmethod
    def calculate_s_parameters(self):
        pass
    def theoretical_return_loss(self):
        """
        Retorna S11 teórico via cascata de seções λ/4 Chebyshev.
        """
        p = self.params
        freqs_hz = self.frequencies * 1e6
        fr = Frequency.from_f(freqs_hz, unit='Hz')
        sec_len_m = self.results['sec_len_mm'] / 1000.0

        z0    = 50.0
        z_eff = z0 / p['n_outputs']
        Zs = [
            z0 * (z_eff / z0) ** ((2 * i - 1) / (2 * p['n_sections']))
            for i in range(1, p['n_sections'] + 1)
        ]

        ntw = None
        for Z in Zs:
            media = DistributedCircuit(frequency=fr, z0=Z)
            seg   = media.line(sec_len_m, unit='m')
            ntw   = seg if ntw is None else ntw ** seg

        # S11 contra terminador 50Ω
        return ntw.s[:, 0, 0]
    def get_geometry_for_viewer(self):
        return {**self.params, **self.results}

    def _get_wavelength_mm(self, f_mhz, er):
        return 299792.458 / (f_mhz * math.sqrt(er))

    def get_theoretical_network(self):
        if self.s_params_th is None:
            return None

        freq = Frequency.from_f(self.frequencies * 1e6, unit='Hz')
        num_ports = self.params['n_outputs'] + 1
        s_matrix = np.zeros((len(self.frequencies), num_ports, num_ports), dtype=complex)

        s_matrix[:, 0, 0] = self.s_params_th.get('S11', 0)
        s_matrix[:, 1, 1] = self.s_params_th.get('S22', 0)
        s_matrix[:, 2, 1] = self.s_params_th.get('S32', 0)

        for i in range(self.params['n_outputs']):
            s_matrix[:, i + 1, 0] = self.s_params_th.get(f'S{i + 2}1', 0)
            s_matrix[:, 0, i + 1] = self.s_params_th.get(f'S{i + 2}1', 0)

        return Network(frequency=freq, s=s_matrix)

    def plot_performance(self):
        """
        Gera 3 gráficos de desempenho comparando teoria vs simulação:
          1) Return Loss (S11) em dB
          2) Insertion Loss (S(n,1)) em dB para cada saída
          3) Carta de Smith para S11
        """
        # frequência em MHz e Hz
        f_mhz = self.frequencies
        f_hz = f_mhz * 1e6
        N = self.params['n_outputs']
        eps = 1e-6

        # Calcula gamma teórico
        gamma_th = self.theoretical_return_loss()

        # Cálculo de Insertion Loss teórico
        il_lin_th = np.sqrt(np.clip((1 - np.abs(gamma_th) ** 2) / N, eps, None))

        # Prepara a figura
        fig = plt.figure(figsize=(18, 5))
        gs = fig.add_gridspec(1, 3)

        # 1) Return Loss (S11)
        ax1 = fig.add_subplot(gs[0, 0])
        s11_db_th = 20 * np.log10(np.clip(np.abs(gamma_th), eps, None))
        ax1.plot(f_mhz, s11_db_th, '-', label="S11 Teórico")
        if self.s_params_sim:
            s11_sim = 20 * np.log10(np.clip(np.abs(self.s_params_sim.s[:, 0, 0]), eps, None))
            ax1.plot(self.s_params_sim.frequency.f / 1e6, s11_sim, '--', label="S11 Simulado")
        ax1.set(title="Return Loss (S11)", xlabel="MHz", ylabel="dB")
        ax1.set_ylim(-60, 0)
        ax1.grid(True)
        ax1.legend()

        # 2) Insertion Loss (S(n,1))
        ax2 = fig.add_subplot(gs[0, 1])
        il_db_th = 20 * np.log10(il_lin_th)
        colors = plt.cm.viridis(np.linspace(0, 1, N))
        for i in range(N):
            ax2.plot(f_mhz, il_db_th, color=colors[i], linestyle='-', label=f"S{i + 2}1 Teórico")
            if self.s_params_sim and self.s_params_sim.s.shape[1] > i + 1:
                s_sim = 20 * np.log10(np.clip(np.abs(self.s_params_sim.s[:, i + 1, 0]), eps, None))
                ax2.plot(self.s_params_sim.frequency.f / 1e6, s_sim,
                         color=colors[i], linestyle='--', label=f"S{i + 2}1 Simulado")
        ax2.set(title="Insertion Loss (S(n,1))", xlabel="MHz", ylabel="dB")
        ax2.set_ylim(-30, 0)
        ax2.grid(True)
        ax2.legend()

        # 3) Smith Chart (S11)
        ax3 = fig.add_subplot(gs[0, 2])
        # rede teórica
        ntw_th = rf.Network()
        ntw_th.frequency = Frequency.from_f(f_hz, 'Hz')
        s_tmp = np.zeros((len(f_hz), 2, 2), dtype=complex)
        s_tmp[:, 0, 0] = gamma_th
        ntw_th.s = s_tmp
        plot_s_smith(ntw_th, ax=ax3, show_legend=False)
        # rede simulada
        if self.s_params_sim:
            plot_s_smith(self.s_params_sim, ax=ax3, show_legend=False, linestyle='--')
        ax3.set_title("Smith Chart (S11)")

        plt.tight_layout()
        return fig


# --------------------------------------------------------------------
# 6. CoaxialCalculator
# --------------------------------------------------------------------
class CoaxialCalculator(RFCalculator):
    def calculate(self):
        p = self.params
        f0 = (p['f_start'] + p['f_stop']) / 2
        er = SUBSTRATE_MATERIALS[p['diel_material']][0]
        sec_len = self._get_wavelength_mm(f0, er) / 4
        len_inner = p['n_sections'] * sec_len
        # Mantem a referencia das saidas em z=len_inner e adiciona 10% apenas no comprimento externo.
        len_outer = (len_inner * 1.05) * 1.10
        d_int_tube = p['d_ext'] - 2 * p['wall_thick']

        d_out_50 = d_int_tube / math.exp(50 * math.sqrt(er) / 59.952)

        z0 = 50.0
        z_eff = 50.0 / p['n_outputs']
        z_sects = [z0 * (z_eff / z0) ** ((2 * i - 1) / (2 * p['n_sections'])) for i in range(1, p['n_sections'] + 1)]
        main_diams = [d_int_tube / math.exp(z_i * math.sqrt(er) / 59.952) for z_i in z_sects]
        output_len_mm = _get_output_length_mm(len_inner, d_int_tube)

        self.results = {
            'sec_len_mm': sec_len,
            'len_inner_mm': len_inner,
            'len_outer_mm': len_outer,
            'd_int_tube': d_int_tube,
            'main_diams': main_diams,
            'd_out_50ohm': d_out_50,
            'z_sects': z_sects,
            'output_len_mm': output_len_mm
        }
        self.calculate_s_parameters()
        return True

    def calculate_s_parameters(self):
        p = self.params
        er = SUBSTRATE_MATERIALS[p['diel_material']][0]

        N_freq = len(self.frequencies)
        freqs_hz = self.frequencies * 1e6
        fr = Frequency.from_f(freqs_hz, unit='Hz')

        media = DistributedCircuit(frequency=fr, z0=50)
        ntw_cascade = media.line(0, 'm')
        for z_val in self.results['z_sects']:
            sec_media = DistributedCircuit(frequency=fr, z0=z_val)
            sec_ntw = sec_media.line(self.results['sec_len_mm'] / 1000.0, 'm')
            ntw_cascade = ntw_cascade ** sec_ntw

        load_ntw = media.resistor(p['n_outputs'] * 50)
        final_ntw = ntw_cascade ** load_ntw
        s11 = final_ntw.s[:, 0, 0]

        Gamma = s11
        mismatch_loss_db = -10 * np.log10(1 - np.abs(Gamma) ** 2)
        ideal_split_loss_db = 10 * np.log10(p['n_outputs'])
        s21_db = -(mismatch_loss_db + ideal_split_loss_db)

        phase_velocity = 299792458 / np.sqrt(er)
        physical_length_m = self.results['len_inner_mm'] / 1000.0
        electrical_length = 2 * np.pi * freqs_hz * physical_length_m / phase_velocity
        s21 = 10 ** (s21_db / 20) * np.exp(-1j * electrical_length)

        s22 = s11
        isolation_db = -6
        s32 = 10 ** (isolation_db / 20) * np.ones(N_freq)

        self.s_params_th = {f'S{i + 2}1': s21 for i in range(p['n_outputs'])}
        self.s_params_th['S11'] = s11
        self.s_params_th['S22'] = s22
        self.s_params_th['S32'] = s32

        return True


# --------------------------------------------------------------------
# 7. Funções de Visualização e Automação (PyVista, HFSS, PDF)
# --------------------------------------------------------------------
def add_cota(plotter, p1, p2, txt, off_dir=np.array([0, 1, 0]), text_offset_factor=1.2, is_2d=False):
    """
    Função aprimorada para adicionar dimensões (cotas) com setas.
    """
    p1, p2 = np.array(p1), np.array(p2)
    d = np.linalg.norm(p2 - p1)
    if d < 1e-6: return

    mid = (p1 + p2) / 2
    offset_val = 0.08 * d + 4  # Ajuste dinâmico do offset

    # Linhas de extensão
    e1 = p1 + off_dir * offset_val * 0.9
    e2 = p2 + off_dir * offset_val * 0.9
    plotter.add_lines(np.vstack((p1, e1)), color='dimgray', width=1)
    plotter.add_lines(np.vstack((p2, e2)), color='dimgray', width=1)

    # Linha de dimensão principal com setas
    vec = (p2 - p1) / d
    arrow_size = d * 0.05
    arrow1 = pv.Arrow(start=e1 + vec * arrow_size, direction=-vec, scale=arrow_size)
    arrow2 = pv.Arrow(start=e2 - vec * arrow_size, direction=vec, scale=arrow_size)

    plotter.add_mesh(arrow1, color='black')
    plotter.add_mesh(arrow2, color='black')
    plotter.add_lines(np.vstack((e1, e2)), color='black', width=2)

    label_pos = mid + off_dir * offset_val * text_offset_factor
    plotter.add_point_labels(
        [label_pos], [txt],
        font_size=16, text_color='black' if is_2d else 'cyan',
        always_visible=True, shape=None, show_points=False
    )


def create_coaxial_geometry(p, for_report=False):
    """
    Gera a geometria 3D com iluminação e materiais aprimorados.
    """
    plotter_opts = {"window_size": (1200, 900), "title": "Divisor Coaxial 3D"}
    if for_report:
        plotter_opts['off_screen'] = True

    plotter = pv.Plotter(**plotter_opts)
    if for_report:
        plotter.set_background('white')
    else:
        plotter.set_background('darkgrey', top='grey')

    # Iluminação aprimorada
    plotter.add_light(pv.Light(position=(1, 1, 1), intensity=1.0))
    plotter.add_light(pv.Light(position=(-1, -1, 1), intensity=0.5))

    # Geometria...
    diel_principal = pv.Cylinder(center=(0, 0, p['len_outer_mm'] / 2), direction=(0, 0, 1),
                                 radius=p['d_int_tube'] / 2, height=p['len_outer_mm'])
    plotter.add_mesh(diel_principal, color='#d3d3d3', opacity=0.15)

    z_pos = 0
    for i, di in enumerate(p['main_diams']):
        sec = pv.Cylinder(center=(0, 0, z_pos + p['sec_len_mm'] / 2), direction=(0, 0, 1),
                          radius=di / 2, height=p['sec_len_mm'])
        plotter.add_mesh(sec, color='#FFD700', pbr=True, metallic=0.9, roughness=0.2)
        z_pos += p['sec_len_mm']

    # Lógica de diâmetro de saída...
    num_saidas = int(p['n_outputs'])
    dia_saida_diel = _get_output_diel_diameter_mm(p['d_int_tube'], num_saidas, p['len_inner_mm'])
    ratio_inner_outer = p.get('d_out_50ohm', p['d_int_tube'] / 2.3) / max(p['d_int_tube'], 1e-9)
    dia_saida_cond = dia_saida_diel * ratio_inner_outer

    output_len = p.get('output_len_mm', _get_output_length_mm(p['len_inner_mm'], p['d_int_tube']))
    z0 = p['len_inner_mm']
    for k in range(num_saidas):
        angle_rad = math.radians(360 * k / num_saidas)
        x, y = math.cos(angle_rad), math.sin(angle_rad)
        center_pos = [x * output_len / 2, y * output_len / 2, z0]
        od = pv.Cylinder(center=center_pos, direction=(x, y, 0), radius=dia_saida_diel / 2, height=output_len)
        oi = pv.Cylinder(center=center_pos, direction=(x, y, 0), radius=dia_saida_cond / 2, height=output_len)
        plotter.add_mesh(od, color='#4682b4', opacity=0.4)
        plotter.add_mesh(oi, color='#FFD700', pbr=True, metallic=0.9, roughness=0.2)

    # Cotas
    add_cota(plotter, [0, 0, 0], [0, 0, p['len_outer_mm']], f"L Total: {p['len_outer_mm']:.2f} mm",
             off_dir=np.array([0, 1.2, 0]))

    plotter.camera_position = 'iso'
    plotter.camera.zoom(1.3)
    plotter.enable_parallel_projection()
    return plotter


def create_coaxial_geometry_2d_side_view(p):
    """
    Gera uma vista lateral 2D (corte) do transformador com dimensionamento detalhado.
    """
    plotter = pv.Plotter(off_screen=True, window_size=(1600, 800))
    plotter.set_background('white')

    # Contorno do dielétrico
    diel_height = p['len_outer_mm']
    diel_radius = p['d_int_tube'] / 2
    diel_outline_points = np.array([
        [0, -diel_radius, 0], [diel_height, -diel_radius, 0],
        [diel_height, diel_radius, 0], [0, diel_radius, 0], [0, -diel_radius, 0]
    ])
    plotter.add_mesh(pv.lines_from_points(diel_outline_points), color='grey', line_width=2)

    # Seções do condutor
    z_pos = 0
    for di in p['main_diams']:
        sec_len = p['sec_len_mm']
        sec_radius = di / 2
        vertices = np.array(
            [[z_pos, -sec_radius, 0], [z_pos + sec_len, -sec_radius, 0], [z_pos + sec_len, sec_radius, 0],
             [z_pos, sec_radius, 0]])
        faces = np.hstack([[4, 0, 1, 2, 3]])
        plotter.add_mesh(pv.PolyData(vertices, faces), color='gold')
        z_pos += sec_len

    # Cotas
    add_cota(plotter, [0, -diel_radius, 0], [diel_height, -diel_radius, 0], f"Comprimento Total: {diel_height:.2f} mm",
             off_dir=np.array([0, -1, 0]), is_2d=True)
    z_pos = 0
    for i, di in enumerate(p['main_diams']):
        sec_len = p['sec_len_mm']
        add_cota(plotter, [z_pos, di / 2, 0], [z_pos + sec_len, di / 2, 0], f"L{i + 1}: {sec_len:.2f}",
                 off_dir=np.array([0, 1, 0]), is_2d=True)
        add_cota(plotter, [z_pos + sec_len / 2, -di / 2, 0], [z_pos + sec_len / 2, di / 2, 0], f"d{i + 1}: {di:.2f}",
                 off_dir=np.array([-1, 0, 0]), is_2d=True)
        z_pos += sec_len

    plotter.view_zy()
    plotter.enable_parallel_projection()
    return plotter

def create_coaxial_geometry_2d(p):
    """
    Gera uma vista de topo (2D) da seção transversal do divisor com dimensionamento.
    **CORRIGIDO**: A criação do arco de cota agora usa plotter.add_lines() para
    desenhar uma polilinha a partir de múltiplos pontos, resolvendo o TypeError.
    """
    plotter = pv.Plotter(off_screen=True, window_size=(1000, 1000))
    plotter.set_background('white')

    # --- Geometria 2D ---
    plotter.add_mesh(pv.Circle(radius=p['d_int_tube'] / 2, resolution=100), color='gray', style='wireframe',
                     line_width=2)
    plotter.add_mesh(pv.Circle(radius=p['main_diams'][-1] / 2, resolution=100), color='gold')

    num_saidas = int(p['n_outputs'])
    if num_saidas > 6:
        dia_saida_diel = p['d_int_tube'] / 3.0
    elif num_saidas > 4:
        dia_saida_diel = p['d_int_tube'] / 2.2
    else:
        dia_saida_diel = p['d_int_tube']
    dia_saida_cond = dia_saida_diel / 2.3

    dist_saida = (p['d_int_tube'] + dia_saida_diel) / 2
    for k in range(num_saidas):
        angle_rad = math.radians(360 * k / num_saidas)
        center_pos = [dist_saida * math.cos(angle_rad), dist_saida * math.sin(angle_rad), 0]

        output_diel_circle = pv.Circle(radius=dia_saida_diel / 2, resolution=100)
        output_diel_circle.translate(center_pos, inplace=True)
        plotter.add_mesh(output_diel_circle, color='lightsteelblue', style='wireframe', line_width=2)

        output_inner_circle = pv.Circle(radius=dia_saida_cond / 2, resolution=100)
        output_inner_circle.translate(center_pos, inplace=True)
        plotter.add_mesh(output_inner_circle, color='gold')

    # --- Cotas 2D ---
    r_main = p['d_int_tube'] / 2
    plotter.add_lines(np.array([[0, 0, 0], [r_main, 0, 0]]), color='red')
    plotter.add_point_labels([r_main / 2, 0, 0], [f"R: {r_main:.2f}"], text_color='black', font_size=16, shape=None,
                             show_points=False)

    angle_deg = 360 / num_saidas if num_saidas > 0 else 0
    if num_saidas > 1:
        # ** CORREÇÃO APLICADA AQUI **
        # Gera os pontos do arco
        arc_angles = np.linspace(0, math.radians(angle_deg), num=20)
        arc_points = np.zeros((len(arc_angles), 3))
        arc_points[:, 0] = dist_saida * np.cos(arc_angles)
        arc_points[:, 1] = dist_saida * np.sin(arc_angles)

        # Usa plotter.add_lines() para desenhar o arco a partir dos pontos
        plotter.add_lines(arc_points, color='red', width=2)

        # Posiciona o texto do ângulo
        mid_arc_point = [dist_saida * 1.1 * math.cos(math.radians(angle_deg / 2)),
                         dist_saida * 1.1 * math.sin(math.radians(angle_deg / 2)), 0]
        plotter.add_point_labels(mid_arc_point, [f"{angle_deg:.1f}°"], text_color='black', font_size=16, shape=None,
                                 show_points=False)

    plotter.view_xy()
    plotter.enable_parallel_projection()
    return plotter


def export_full_pdf(calc: RFCalculator, queue: queue.Queue):
    pdf_path = fd.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF", "*.pdf")])
    if not pdf_path:
        queue.put("Exportação de PDF cancelada.")
        return

    queue.put("Gerando imagens para o relatório PDF...")
    params = calc.get_geometry_for_viewer()

    with tempfile.TemporaryDirectory() as tempdir:
        img3d_path = os.path.join(tempdir, "view3d_iso.png")
        plotter_3d = create_coaxial_geometry(params, for_report=True)
        plotter_3d.screenshot(img3d_path)
        plotter_3d.close()

        img2d_path = os.path.join(tempdir, "view2d_side.png")
        plotter_2d = create_coaxial_geometry_2d_side_view(params)
        plotter_2d.screenshot(img2d_path)
        plotter_2d.close()

        img_perf_path = os.path.join(tempdir, "performance.png")
        fig = calc.plot_performance()
        fig.savefig(img_perf_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        queue.put("Montando o PDF...")
        doc = SimpleDocTemplate(pdf_path, pagesize=letter, topMargin=30, bottomMargin=30)
        styles = getSampleStyleSheet()
        story = [Paragraph("Relatório de Análise de Divisor Coaxial", styles['Title']), Spacer(1, 12)]

        story.append(Paragraph("Vista Isométrica (3D)", styles['h2']))
        if os.path.exists(img3d_path): story.append(RLImage(img3d_path, width=500, height=375))
        story.append(Spacer(1, 12))

        story.append(Paragraph("Vista Lateral 2D (Corte e Dimensionamento)", styles['h2']))
        if os.path.exists(img2d_path): story.append(RLImage(img2d_path, width=500, height=250))
        story.append(Spacer(1, 12))

        story.append(Paragraph("Gráficos de Desempenho Teórico", styles['h2']))
        if os.path.exists(img_perf_path): story.append(RLImage(img_perf_path, width=540, height=180))
        story.append(Spacer(1, 24))

        p, r = calc.params, calc.results
        data = [["Parâmetro", "Valor"], ["Frequência (MHz)", f"{p['f_start']} - {p['f_stop']}"],
                ["Nº de Seções", f"{p['n_sections']}"], ["Nº de Saídas", f"{p['n_outputs']}"]]
        table = Table(data, colWidths=[250, 250])
        table.setStyle(TableStyle(
            [('BACKGROUND', (0, 0), (-1, 0), colors.darkblue), ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
             ('ALIGN', (0, 0), (-1, -1), 'CENTER'), ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
             ('BOTTOMPADDING', (0, 0), (-1, 0), 12), ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
             ('GRID', (0, 0), (-1, -1), 1, colors.black)]))
        story.append(table)

        doc.build(story)
        queue.put(f"PDF salvo com sucesso em: {pdf_path}")


def launch_pyvista(params, q):
    try:
        plotter = create_coaxial_geometry(params)
        plotter.show()
        q.put("Visualização 3D fechada.")
    except Exception as e:
        q.put(f"Erro na visualização 3D: {e}")
        traceback.print_exc()


def generate_hfss_model(params, status_queue):
    """
    ** VERSÃO FINAL E APRIMORADA **
    Esta função implementa a lógica de criação de geometria, portas e análise,
    e agora ajusta dinamicamente o diâmetro das portas de saída para evitar
    interseções quando há mais de 4 saídas.
    """
    if not HFSS_AVAILABLE:
        status_queue.put("ERRO: módulo HFSS não disponível.")
        return None

    project_path = None
    try:
        status_queue.put("Iniciando geração HFSS (lógica aprimorada)...")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        proj_name = f"Divisor_Coaxial_{ts}"

        user_app_dir = _get_user_appdata_dir("DIVISOR")
        project_dir = user_app_dir / "HFSS_Projects"
        project_dir.mkdir(parents=True, exist_ok=True)
        project_path = str(project_dir / f"{proj_name}.aedt")

        with Hfss(
            project=project_path,
            design="DivisorCoaxial",
            solution_type="Modal",
            new_desktop=True,
            close_on_exit=False,
        ) as hfss:
            mdl = hfss.modeler
            p = params
            f0_ghz = (p['f_start'] + p['f_stop']) / 2000.0

            # 1. Definir Variáveis no HFSS
            vm = hfss.variable_manager
            vm["comp_total"] = f"{p['len_outer_mm']}mm"
            vm["comp_secoes"] = f"{p['len_inner_mm']}mm"
            vm["dia_int_tubo"] = f"{p['d_int_tube']}mm"

            # Obter o número de saídas para facilitar a leitura
            num_saidas = int(p['n_outputs'])
            dia_saida_diel_mm = _get_output_diel_diameter_mm(p['d_int_tube'], num_saidas, p['len_inner_mm'])
            ratio_inner_outer = p['d_out_50ohm'] / max(p['d_int_tube'], 1e-9)
            dia_saida_cond_mm = dia_saida_diel_mm * ratio_inner_outer

            # 1. Definir o diâmetro da saída (dielétrico e condutor) com base no número de saídas
            if num_saidas > 6:
                status_queue.put(f"Detectado {num_saidas} saídas (>6). Reduzindo diâmetro para evitar colisão.")

            elif num_saidas > 4:
                status_queue.put(f"Detectado {num_saidas} saídas (>4). Reduzindo diâmetro para evitar colisão.")
            vm["dia_saida_diel"] = f"{dia_saida_diel_mm}mm"
            vm["dia_saida_cond"] = f"{dia_saida_cond_mm}mm"

            # 2. Definir o comprimento da saída
            output_len_mm = p.get(
                'output_len_mm',
                _get_output_length_mm(p['len_inner_mm'], p['d_int_tube'])
            )
            vm["comp_saida"] = f"{output_len_mm}mm"

            for i, d in enumerate(p['main_diams']):
                vm[f"dia_sc{i + 1}"] = f"{d}mm"

            # 2. Criar Geometrias Individuais
            diel_principal = mdl.create_cylinder("Z", [0, 0, 0], "dia_int_tubo/2", "comp_total", name="Diel_Principal",
                                                 num_sides=0)

            inner_parts = [
                mdl.create_cylinder("Z", [0, 0, f"{p['sec_len_mm'] * i}"], f"dia_sc{i + 1}/2", f"{p['sec_len_mm']}mm",
                                    name=f"Cond_Sec{i + 1}", num_sides=0)
                for i in range(p['n_sections'])
            ]

            # ** APRIMORAMENTO: Usa as novas variáveis de diâmetro **
            output_outer_1 = mdl.create_cylinder("X", [0, 0, "comp_secoes"], "dia_saida_diel/2", "comp_saida",
                                                 num_sides=0,
                                                 name="Output_Outer_1")
            output_inner_1 = mdl.create_cylinder("X", [0, 0, "comp_secoes"], "dia_saida_cond/2", "comp_saida",
                                                 num_sides=0,
                                                 name="Output_Inner_1")

            # 3. Duplicar apenas a GEOMETRIA das saídas
            if p['n_outputs'] > 1:
                mdl.duplicate_around_axis(
                    [output_outer_1.name, output_inner_1.name],
                    axis="Z",
                    angle=f"360deg/{p['n_outputs']}",
                    clones=p['n_outputs']
                )

            # 4. Unir as partes em dois corpos finais
            all_outers_names = [diel_principal.name] + [name for name in mdl.object_names if
                                                        name.startswith("Output_Outer_")]
            united_diel_name = mdl.unite(all_outers_names)
            final_diel_obj = mdl[united_diel_name]
            final_diel_obj.name = "Volume_Diel"

            inner_parts_names = [part.name for part in inner_parts]
            output_inners_names = [name for name in mdl.object_names if
                                   name.startswith("Cond_Saida_") or name.startswith("Output_Inner_")]
            all_inners_to_unite = inner_parts_names + output_inners_names
            united_cond_name = mdl.unite(all_inners_to_unite)
            final_cond_obj = mdl[united_cond_name]
            final_cond_obj.name = "Condutor_Unico"

            # 5. Atribuir Materiais aos corpos FINAIS
            er_val, tand = SUBSTRATE_MATERIALS[p['diel_material']]
            mat_name = p['diel_material'].replace(' ', '_').replace('(', '').replace(')', '')
            if mat_name not in hfss.materials:
                new_mat = hfss.materials.add_material(mat_name)
                new_mat.permittivity = er_val
                new_mat.dielectric_loss_tangent = tand

            final_diel_obj.material_name = mat_name
            final_cond_obj.material_name = "pec"

            # 6. Criar Portas nos corpos FINAIS
            for face in final_diel_obj.faces:
                if abs(face.center[2]) < 1e-6:
                    hfss.wave_port(assignment=face.id, name="P1", renormalize=True, impedance="50")
                    break

            output_port_len_float = output_len_mm
            for k in range(p['n_outputs']):
                angle_rad = math.radians(360 * k / p['n_outputs'])
                pos_x = output_port_len_float * math.cos(angle_rad)
                pos_y = output_port_len_float * math.sin(angle_rad)
                pos_z = p['len_inner_mm']

                for face in final_diel_obj.faces:
                    center = face.center
                    if np.linalg.norm(np.array(center) - np.array([pos_x, pos_y, pos_z])) < 1e-3:
                        hfss.wave_port(assignment=face.id, name=f"P{k + 2}", renormalize=True, impedance="50")
                        break

            # 7. Setup e Sweep
            setup = hfss.create_setup("Setup1")
            setup.props["Frequency"] = f"{f0_ghz}GHz"
            setup.props["MaximumPasses"] = 8
            setup.props["MaxDeltaS"] = 0.02

            setup.create_frequency_sweep(
                unit="MHz",

                start_frequency=p['f_start'],
                stop_frequency=p['f_stop'],
                num_of_freq_points=201,
                name="Sweep1",
                sweep_type="Fast",
                save_fields=False
            )

            hfss.save_project()
            status_queue.put(f"HFSS salvo em: {project_path}")
            return project_path

    except Exception as e:
        status_queue.put(f"ERRO HFSS: {e}")
        traceback.print_exc()
        return None

# def export_full_pdf(calc: RFCalculator, queue: queue.Queue):
#     pdf_path = fd.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF", "*.pdf")])
#     if not pdf_path:
#         queue.put("Exportação de PDF cancelada.")
#         return
#
#     queue.put("Gerando PDF...")
#     with tempfile.TemporaryDirectory() as tempdir:
#         # ** CORREÇÃO APLICADA AQUI **
#         img3d_path = os.path.join(tempdir, "view3d.png")
#         pv_plot = create_coaxial_geometry(calc.get_geometry_for_viewer())
#         pv_plot.off_screen = True
#         pv_plot.screenshot(img3d_path, window_size=(1200, 800))
#         pv_plot.close()
#
#         img_perf_path = os.path.join(tempdir, "performance.png")
#         fig = calc.plot_performance()
#         fig.savefig(img_perf_path, dpi=150)
#         plt.close(fig)
#
#         p, r = calc.params, calc.results
#         data = [["Parâmetro", "Valor"]]
#         data.extend([
#             ["Frequência Inicial (MHz)", f"{p['f_start']:.2f}"],
#             ["Frequência Final (MHz)", f"{p['f_stop']:.2f}"],
#             ["Diâmetro Externo (mm)", f"{p['d_ext']:.3f}"],
#             ["Espessura da Parede (mm)", f"{p['wall_thick']:.3f}"],
#             ["Número de Seções", f"{p['n_sections']}"],
#             ["Número de Saídas", f"{p['n_outputs']}"],
#             ["Material Dielétrico", p['diel_material']],
#             ["Comprimento de Seção (mm)", f"{r['sec_len_mm']:.3f}"],
#             ["Comprimento Interno (mm)", f"{r['len_inner_mm']:.3f}"],
#         ])
#         for i, (diam, z) in enumerate(zip(r['main_diams'], r['z_sects']), start=1):
#             data.append([f"Diâmetro Seção {i} (mm)", f"{diam:.3f}"])
#             data.append([f"Impedância Seção {i} (Ohm)", f"{z:.3f}"])
#
#         doc = SimpleDocTemplate(pdf_path, pagesize=letter)
#         styles = getSampleStyleSheet()
#         elements = [Paragraph("Relatório de Análise - Divisor Coaxial Chebyshev", styles['Title']), Spacer(1, 12)]
#         elements.append(RLImage(img3d_path, width=400, height=267))
#         elements.append(Spacer(1, 12))
#         elements.append(Paragraph("Gráficos de Desempenho", styles['h2']))
#         elements.append(RLImage(img_perf_path, width=500, height=400))
#         elements.append(Spacer(1, 12))
#         elements.append(Paragraph("Parâmetros Geométricos e Elétricos", styles['h2']))
#
#         table = Table(data, colWidths=[200, 200])
#         table.setStyle(TableStyle([
#             ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
#             ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
#             ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
#             ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
#             ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
#             ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
#             ('GRID', (0, 0), (-1, -1), 1, colors.black)
#         ]))
#         elements.append(table)
#
#         doc.build(elements)
#         queue.put(f"PDF salvo em: {pdf_path}")


# --------------------------------------------------------------------
# 10. GUI CTk
# --------------------------------------------------------------------
class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Ferramenta de Design de Divisor de Potência RF")
        self.geometry("1400x900")
        self.queue = queue.Queue()
        self.calc = None
        self.project_path = None

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        self.param_frame = ParameterFrame(self, self)
        self.param_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ns")

        self.result_frame = ResultFrame(self, self)
        self.result_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        self.status_bar = ctk.CTkLabel(self, text="Pronto. Insira os parâmetros e clique em 'Calcular'.", anchor="w")
        self.status_bar.grid(row=1, column=0, columnspan=2, padx=10, pady=5, sticky="ew")

        self.after(100, self.process_queue)
        self.update_button_states()

    def process_queue(self):
        try:
            msg = self.queue.get_nowait()
            self.status_bar.configure(text=str(msg))
        except queue.Empty:
            pass
        finally:
            self.after(100, self.process_queue)

    def calculate_and_display(self, params):
        self.calc = CoaxialCalculator(params)
        self.queue.put("Calculando parâmetros geométricos e teóricos...")
        if self.calc.calculate():
            self.queue.put("Cálculo concluído. Gerando visualizações...")
            self.result_frame.display_results(self.calc)
            self.update_button_states()
        else:
            self.queue.put("Erro no cálculo. Verifique os parâmetros.")
            messagebox.showerror("Erro de Cálculo", "Não foi possível calcular o divisor com os parâmetros fornecidos.")
            self.calc = None
            self.update_button_states()

    def export_hfss(self):
        if self.calc:
            thread = threading.Thread(target=self._export_hfss_thread, daemon=True)
            thread.start()

    def _export_hfss_thread(self):
        # ** CORREÇÃO APLICADA AQUI **
        path = generate_hfss_model(self.calc.get_geometry_for_viewer(), self.queue)
        if path:
            self.project_path = path
            self.after(0, self.update_button_states)

    def run_simulation(self):
        if self.project_path:
            thread = threading.Thread(target=run_hfss_simulation, args=(self.project_path, self.queue), daemon=True)
            thread.start()

    def import_hfss_results(self):
        if not self.calc:
            messagebox.showwarning("Aviso", "Calcule um projeto primeiro antes de importar resultados.")
            return

        proj_dir = os.path.dirname(self.project_path) if self.project_path else None
        touchstone_file = ""

        if proj_dir and os.path.exists(proj_dir):
            proj_name = os.path.splitext(os.path.basename(self.project_path))[0]
            num_ports = self.calc.params['n_outputs'] + 1
            expected_file = os.path.join(proj_dir, f"{proj_name}.s{num_ports}p")
            if os.path.exists(expected_file):
                touchstone_file = expected_file

        if not touchstone_file:
            touchstone_file = fd.askopenfilename(
                title="Selecione o arquivo Touchstone (.sNp)",
                filetypes=[("Touchstone files", "*.s*p"), ("All files", "*.*")]
            )

        if touchstone_file and os.path.exists(touchstone_file):
            try:
                self.queue.put(f"Importando resultados de {touchstone_file}...")
                sim_net = Network(touchstone_file)
                self.calc.s_params_sim = sim_net
                self.queue.put("Resultados importados. Atualizando gráficos...")
                self.result_frame.display_results(self.calc)
            except Exception as e:
                self.queue.put(f"Erro ao importar arquivo: {e}")
                messagebox.showerror("Erro de Importação", f"Não foi possível ler o arquivo Touchstone.\n{e}")

    def export_pdf(self):
        if self.calc:
            thread = threading.Thread(target=export_full_pdf, args=(self.calc, self.queue), daemon=True)
            thread.start()

    def save_project(self):
        if self.calc:
            filepath = fd.asksaveasfilename(defaultextension=".json", filetypes=[("JSON Project Files", "*.json")])
            if filepath:
                try:
                    with open(filepath, 'w') as f:
                        json.dump(self.calc.params, f, indent=4)
                    self.queue.put(f"Projeto salvo em {filepath}")
                except Exception as e:
                    messagebox.showerror("Erro ao Salvar", f"Não foi possível salvar o projeto.\n{e}")

    def load_project(self):
        filepath = fd.askopenfilename(filetypes=[("JSON Project Files", "*.json")])
        if filepath:
            try:
                with open(filepath, 'r') as f:
                    params = json.load(f)
                self.param_frame.load_params(params)
                self.queue.put(f"Projeto carregado de {filepath}")
            except Exception as e:
                messagebox.showerror("Erro ao Carregar", f"Não foi possível carregar o projeto.\n{e}")

    def update_button_states(self):
        calc_done = self.calc is not None
        hfss_exported = self.project_path is not None

        self.param_frame.export_hfss_btn.configure(state="normal" if calc_done else "disabled")
        self.param_frame.run_sim_btn.configure(state="normal" if hfss_exported else "disabled")
        self.param_frame.import_res_btn.configure(state="normal" if calc_done else "disabled")
        self.param_frame.export_pdf_btn.configure(state="normal" if calc_done else "disabled")


class ParameterFrame(ctk.CTkFrame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app

        self.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(self, text="Parâmetros de Entrada", font=("Arial", 16, "bold")).grid(row=0, column=0, columnspan=2,
                                                                                          pady=10)

        self.entries = {}
        params_layout = {
            "Freq. Inicial (MHz):": ("f_start", "800"), "Freq. Final (MHz):": ("f_stop", "1200"),
            "D_ext Coaxial (mm):": ("d_ext", "20.0"), "Espessura Parede (mm):": ("wall_thick", "1.5"),
            "Nº de Seções (N):": ("n_sections", "4"), "Nº de Saídas (N_out):": ("n_outputs", "4")
        }

        row = 1
        for label_text, (key, default_val) in params_layout.items():
            ctk.CTkLabel(self, text=label_text).grid(row=row, column=0, padx=10, pady=5, sticky="w")
            entry = ctk.CTkEntry(self)
            entry.insert(0, default_val)
            entry.grid(row=row, column=1, padx=10, pady=5, sticky="ew")
            self.entries[key] = entry
            row += 1

        ctk.CTkLabel(self, text="Material Dielétrico:").grid(row=row, column=0, padx=10, pady=5, sticky="w")
        self.mat_var = ctk.StringVar(value=list(SUBSTRATE_MATERIALS.keys())[0])
        self.mat_menu = ctk.CTkOptionMenu(self, values=list(SUBSTRATE_MATERIALS.keys()), variable=self.mat_var)
        self.mat_menu.grid(row=row, column=1, padx=10, pady=5, sticky="ew")
        row += 1

        self.calc_btn = ctk.CTkButton(self, text="Calcular", command=self._validate_and_calc, fg_color="#2aa198",
                                      hover_color="#268c84")
        self.calc_btn.grid(row=row, column=0, columnspan=2, pady=10, sticky="ew");
        row += 1

        self.export_hfss_btn = ctk.CTkButton(self, text="Exportar Modelo HFSS", command=self.app.export_hfss)
        self.export_hfss_btn.grid(row=row, column=0, columnspan=2, pady=5, sticky="ew");
        row += 1

        self.run_sim_btn = ctk.CTkButton(self, text="Executar Simulação HFSS", command=self.app.run_simulation)
        self.run_sim_btn.grid(row=row, column=0, columnspan=2, pady=5, sticky="ew");
        row += 1

        self.import_res_btn = ctk.CTkButton(self, text="Importar Resultados HFSS (.sNp)",
                                            command=self.app.import_hfss_results)
        self.import_res_btn.grid(row=row, column=0, columnspan=2, pady=5, sticky="ew");
        row += 1

        self.export_pdf_btn = ctk.CTkButton(self, text="Exportar Relatório PDF", command=self.app.export_pdf)
        self.export_pdf_btn.grid(row=row, column=0, columnspan=2, pady=5, sticky="ew");
        row += 1

        proj_frame = ctk.CTkFrame(self, fg_color="transparent")
        proj_frame.grid(row=row, column=0, columnspan=2, pady=10, sticky="ew")
        proj_frame.grid_columnconfigure((0, 1), weight=1)
        ctk.CTkButton(proj_frame, text="Salvar Projeto", command=self.app.save_project).grid(row=0, column=0, padx=5,
                                                                                             sticky="ew")
        ctk.CTkButton(proj_frame, text="Carregar Projeto", command=self.app.load_project).grid(row=0, column=1, padx=5,
                                                                                               sticky="ew")

    def _validate_and_calc(self):
        try:
            params = {key: float(entry.get()) for key, entry in self.entries.items()}
            params['n_sections'] = int(params['n_sections'])
            params['n_outputs'] = int(params['n_outputs'])
            params['diel_material'] = self.mat_var.get()

            if params['f_stop'] <= params['f_start']:
                raise ValueError("Frequência final deve ser maior que a inicial.")
            if params['d_ext'] <= 2 * params['wall_thick']:
                raise ValueError("Diâmetro externo deve ser maior que o dobro da espessura da parede.")
            if params['n_sections'] <= 0 or params['n_outputs'] <= 0:
                raise ValueError("Número de seções e saídas devem ser inteiros positivos.")

            self.app.calculate_and_display(params)
        except ValueError as e:
            self.app.queue.put(f"Erro de Entrada: {e}")
            messagebox.showerror("Entrada Inválida", str(e))
        except Exception as e:
            self.app.queue.put(f"Erro inesperado: {e}")
            traceback.print_exc()

    def load_params(self, params):
        for key, value in params.items():
            if key in self.entries:
                self.entries[key].delete(0, 'end')
                self.entries[key].insert(0, str(value))
            elif key == 'diel_material':
                self.mat_var.set(value)


class ResultFrame(ctk.CTkFrame):
    def __init__(self, parent, app):
        super().__init__(parent, fg_color="transparent")
        self.app = app
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.tab_view = ctk.CTkTabview(self)
        self.tab_view.grid(row=0, column=0, sticky="nsew")

        self.tab_view.add("Gráficos de Desempenho")
        self.tab_view.add("Parâmetros Geométricos")
        self.tab_view.add("Visualização 3D")

        self.canvas_frame = self.tab_view.tab("Gráficos de Desempenho")
        self.geo_frame = self.tab_view.tab("Parâmetros Geométricos")
        self.vis_frame = self.tab_view.tab("Visualização 3D")

        ctk.CTkLabel(self.vis_frame,
                     text="A visualização 3D é iniciada em uma janela separada.\nUse este botão para reabri-la se necessário.").pack(
            pady=20)
        ctk.CTkButton(self.vis_frame, text="Atualizar Visualização 3D", command=self.launch_3d_viewer).pack(pady=10)

    def display_results(self, calc: RFCalculator):
        for widget in self.canvas_frame.winfo_children():
            widget.destroy()
        fig = calc.plot_performance()
        canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=ctk.TOP, fill=ctk.BOTH, expand=1)

        for widget in self.geo_frame.winfo_children():
            widget.destroy()
        text_area = ctk.CTkTextbox(self.geo_frame, font=("Courier New", 12))
        text_area.pack(fill="both", expand=True, padx=10, pady=10)

        p, r = calc.params, calc.results
        report = "--- Parâmetros Geométricos e Elétricos ---\n\n"
        report += f"{'Parâmetro':<30} | {'Valor'}\n"
        report += "-" * 50 + "\n"
        report += f"{'Comprimento de Seção (mm)':<30} | {r['sec_len_mm']:.3f}\n"
        report += f"{'Comprimento Interno Total (mm)':<30} | {r['len_inner_mm']:.3f}\n"
        report += f"{'Diâmetro Interno do Tubo (mm)':<30} | {r['d_int_tube']:.3f}\n"
        report += f"{'Diâmetro Saída 50 Ohm (mm)':<30} | {r['d_out_50ohm']:.3f}\n\n"

        for i, (diam, z) in enumerate(zip(r['main_diams'], r['z_sects']), start=1):
            report += f"{f'Diâmetro Seção {i} (mm)':<30} | {diam:.3f}\n"
            report += f"{f'Impedância Seção {i} (Ohm)':<30} | {z:.3f}\n"

        text_area.insert("1.0", report)
        text_area.configure(state="disabled")

    def launch_3d_viewer(self):
        if self.app.calc:
            params = self.app.calc.get_geometry_for_viewer()
            thread = threading.Thread(target=launch_pyvista, args=(params, self.app.queue), daemon=True)
            thread.start()
        else:
            messagebox.showwarning("Aviso", "Nenhum cálculo foi realizado ainda.")


if __name__ == "__main__":
    app = App()
    app.mainloop()
