
import os
import re
import math
import csv
import numpy as np
from io import StringIO, BytesIO
import matplotlib.figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import base64
from typing import List, Tuple, Optional

# Constants
NUM_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")

def _is_float(s: str) -> bool:
    return bool(NUM_RE.match(s.strip().replace(",", ".")))

# ----------------------------- Parsing ----------------------------- #

def parse_hfss_csv(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Leitura robusta de CSV/TSV do HFSS (4 colunas). Ignora header e 2 primeiras colunas.
    Retorna (theta_deg, E_over_Emax_linear).
    """
    try:
        text = open(path, "r", encoding="utf-8", errors="ignore").read()
    except Exception as e:
        raise ValueError(f"Erro ao abrir arquivo: {e}")
        
    sample = text[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t ")
        reader = csv.reader(StringIO(text), dialect)
        rows = list(reader)
    except:
        # Fallback se sniffer falhar
        rows = [line.split() for line in text.splitlines()]

    if not rows:
        raise ValueError("Arquivo vazio.")
        
    # Se a primeira linha tiver letras, é header
    if any(ch.isalpha() for ch in "".join(rows[0])):
        rows = rows[1:]
        
    thetas: List[float] = []
    vals: List[float] = []
    
    for r in rows:
        if len(r) < 2: continue
        
        # Try to find numeric columns
        # HFSS often: [Theta, Phi, Re, Im, Mag, ...] or just [Theta, Value]
        # We try to find the LAST numeric column as Value, and one before as Angle if relevant
        # Implementation from desktop:
        try:
           # Assuming HFSS structure: 3rd col=Theta, last=Value? 
           # Actually simplified logic:
           # If len >= 4, use index 2 (Theta) and -1 (Value)
           if len(r) >= 4:
               t_raw = r[2].strip().replace(",", ".")
               v_raw = r[-1].strip().replace(",", ".")
           elif len(r) >= 2:
               t_raw = r[0].strip().replace(",", ".")
               v_raw = r[1].strip().replace(",", ".")
           else:
               continue

           if _is_float(t_raw) and _is_float(v_raw):
               t = float(t_raw)
               v = float(v_raw)
               if math.isfinite(t) and math.isfinite(v):
                   thetas.append(t)
                   vals.append(v)
        except:
            continue

    if not thetas:
        raise ValueError("Falha ao ler colunas Theta e valor (E/Emax) do CSV.")
        
    a = np.asarray(thetas, dtype=float)
    v = np.asarray(vals, dtype=float)
    
    # Normalize if needed? Assuming raw linear or dB?
    # Usually we detect. But let's keep it raw here.
    
    order = np.argsort(a)
    return a[order], v[order]

def parse_generic_table(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Fallback p/ TXT/TSV com 2+ números/linha"""
    angles: List[float] = []
    vals: List[float] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line: continue
            normalized = line.replace(",", ".")
            parts = re.split(r"[\t\s;,,]+", normalized)
            nums = [p for p in parts if _is_float(p)]
            
            if len(nums) < 2: continue
            
            # Simple heuristic: 1st=Angle, 2nd=Value
            try:
                angle_deg = float(nums[0])
                value = float(nums[1])
                if math.isfinite(angle_deg) and math.isfinite(value):
                    angles.append(angle_deg)
                    vals.append(value)
            except: continue
            
    if not angles:
        raise ValueError("Nenhum dado numérico válido encontrado.")
        
    a = np.asarray(angles, dtype=float)
    v = np.asarray(vals, dtype=float)
    order = np.argsort(a)
    return a[order], v[order]

def parse_auto(path: str) -> Tuple[np.ndarray, np.ndarray]:
    try:
        return parse_hfss_csv(path)
    except:
        return parse_generic_table(path)

# ----------------------------- Math ----------------------------- #

def normalize_linear(values: np.ndarray) -> np.ndarray:
    m = np.max(values) if values.size > 0 else 1.0
    return values / (m if m > 0 else 1.0)

def linear_to_db(values: np.ndarray) -> np.ndarray:
    """Converte valores lineares (0..1) para dB (-inf..0)."""
    # Evitar log(0)
    v = values.copy()
    v[v <= 1e-9] = 1e-9
    return 20 * np.log10(v)

# ----------------------------- Export ----------------------------- #

def render_table_image(angles: np.ndarray, values: np.ndarray, unit: str, color: str) -> str:
    """
    Gera uma imagem (base64) de uma tabela com 4 colunas (dobrando os dados).
    Colunas: [Angle, Value, |  Angle, Value]
    """
    # Formatar dados
    rows = []
    n = len(angles)
    mid = (n + 1) // 2
    
    # Parametros de estilo
    col_labels = ["Ângulo (º)", f"Valor ({unit})", "Ângulo (º)", f"Valor ({unit})"]
    cell_text = []
    
    for i in range(mid):
        # Esquerda
        a1 = angles[i]
        v1 = values[i]
        
        # Direita (se existir)
        if i + mid < n:
            a2 = angles[i+mid]
            v2 = values[i+mid]
            row = [f"{a1:.1f}", f"{v1:.2f}", f"{a2:.1f}", f"{v2:.2f}"]
        else:
            row = [f"{a1:.1f}", f"{v1:.2f}", "", ""]
            
        cell_text.append(row)
        
    # Plotar tabela
    # Altura baseada no numero de linhas
    h = max(4, len(cell_text) * 0.3 + 1)
    fig = matplotlib.figure.Figure(figsize=(8, h), dpi=100)
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    table = ax.table(cellText=cell_text, colLabels=col_labels, loc='center', cellLoc='center')
    
    # Estilizacao basica
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    # Header color
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor(color if color else "#dddddd")
            cell.set_text_props(weight='bold', color='white' if color else 'black')
            
    fig.tight_layout()
    
    buf = BytesIO()
    canvas = FigureCanvasAgg(fig)
    canvas.print_png(buf)
    data = base64.b64encode(buf.getvalue()).decode("utf-8")
    return data
