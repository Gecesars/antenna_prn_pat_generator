# numba.md — Técnica e Plano de Implementação para Acelerar o Código com Numba (EFTX Diagram Suite)
Versão: 1.0 • Alvo: acelerar métricas/loops de diagramas e rotinas numéricas (HRP/VRP/3D) • UI: PySide6 (não bloquear UI)

> **Objetivo:** introduzir Numba de forma controlada e profissional para tornar o sistema rápido, com:
- kernels numéricos `@njit` em módulos isolados
- wrappers Python que fazem validação + conversões de dtype/contiguidade
- fallback “numpy/puro” quando numba indisponível
- warm-up (pré-compilação) fora do thread da UI
- benchmarks + suíte de testes para garantir correção numérica

---

## 0) Regras obrigatórias (não negociar)
1) **Isolar kernels**: nenhum código de UI chama Numba diretamente. UI → core → wrappers → kernels.
2) **Hot-path sem Python objects**: dentro de `@njit` usar apenas números e `np.ndarray` contíguos.
3) **Dtypes fixos**: padronizar `float64` (ou `float32` em perfis) e não misturar sem intenção.
4) **Sem alocação em loops**: prealocar buffers e arrays de saída.
5) **Determinismo**: por padrão, `fastmath=False`. Só habilitar via config “fast” com tolerância nos testes.
6) **Não travar UI**: a primeira compilação Numba deve ocorrer em background (warm-up) ou na inicialização com feedback.
7) **Testes e benchmarks obrigatórios**: cada kernel deve ter teste unitário + comparação contra referência NumPy.

---

## 1) Onde aplicar Numba no projeto (prioridade)
### P1 — Métricas 1D de cortes (HRP/VRP)
- pico (max + ângulo)
- HPBW (3 dB e configurável)
- first null (mínimo local após lóbulo principal)
- front-to-back (F/B)
- integração de potência ao longo do ângulo (trapézio)
- **D2D do corte** (HRP/VRP) conforme fórmula correta:
  - HRP: `D2D = 2π / ∫ P(φ) dφ`
  - VRP: `D2D =  π / ∫ P(θ) dθ`
  - com `P = (E/Emax)^2` e ângulo em rad
> Essas rotinas geralmente têm loops e busca em arrays → ganho alto com Numba.

### P2 — Reamostragem e tabela
- “smart decimation”: reduzir N pontos para ~60–90 linhas para PDF/relatório, preservando pico, mínimos e bordas.
- conversões rápidas deg↔rad e normalizações.

### P3 — Geração de grids (quando houver)
- montagem de `theta×phi` e cortes representativos (se existir reconstrução 3D local).

---

## 2) Estrutura de pastas (obrigatória)
Criar:

```
core/
  numba_kernels/
    __init__.py
    integrate.py        # trapz, deg2rad, etc.
    metrics_1d.py       # pico, HPBW, nulls, F/B, D2D
    resample.py         # decimation inteligente
    utils.py            # helpers numba-safe
  analysis/
    pattern_metrics.py  # wrappers: valida inputs, chama kernels
tests/
  numba/
    test_integrate.py
    test_metrics_1d.py
    test_resample.py
  perf/
    bench_numba.py
```

**Regra:** `core/analysis/pattern_metrics.py` é o ponto de entrada para o resto do sistema.

---

## 3) Dependências e configuração
### 3.1 Instalar
```bash
pip install numba numpy pytest pytest-benchmark
```

### 3.2 Config runtime
Criar config global do app (ex.: `core/config/perf.py`):
- `USE_NUMBA = True`
- `NUMBA_FASTMATH = False`
- `NUMBA_PARALLEL = False`
- `NUMBA_DTYPE = "float64"` (ou "float32" em modo rápido)

---

## 4) Padrão de implementação (wrapper + kernel)
### 4.1 Wrapper (Python)
O wrapper:
- valida shape
- converte dtype e contiguidade
- chama kernel
- retorna dict limpo para UI

Exemplo padrão:
```python
import numpy as np
from core.numba_kernels.metrics_1d import metrics_cut_1d_numba

def metrics_cut_1d(angles_deg, e_lin, span_mode, dtype=np.float64):
    ang = np.ascontiguousarray(angles_deg, dtype=dtype)
    e = np.ascontiguousarray(e_lin, dtype=dtype)
    if ang.ndim != 1 or e.ndim != 1 or len(ang) != len(e):
        raise ValueError("angles_deg e e_lin devem ser 1D e ter o mesmo tamanho")
    return metrics_cut_1d_numba(ang, e, span_mode)
```

### 4.2 Kernel (Numba)
O kernel:
- só números e arrays
- sem strings/dicts
- retorna valores escalares e arrays simples (ou tuple)

---

## 5) Kernels recomendados (especificação detalhada)
A seguir estão os kernels que o agente deve implementar.

### 5.1 `deg2rad` + `trapz` rápido (`integrate.py`)
- `deg2rad_inplace(deg, out_rad)`
- `trapz(x_rad, y)`

Regras:
- `trapz`: loop `for i in range(n-1)`
- `dx = x[i+1]-x[i]`
- `sum += 0.5*(y[i]+y[i+1])*dx`

### 5.2 Normalização e potência (`metrics_1d.py`)
Entrada:
- `angles_deg: float[]`
- `e_lin: float[]` (amplitude linear)

Processo:
1) `emax = max(e_lin)`
2) `p[i] = (e_lin[i]/emax)^2`
3) `p_db[i] = 10*log10(p[i])` (quando necessário)

### 5.3 Pico (max) e ângulo do pico
- retornar: `peak_idx`, `peak_angle_deg`, `peak_db` (se calculado)

### 5.4 HPBW (3 dB por padrão)
- dado o pico (0 dB), achar os dois cruzamentos `-3 dB` ao redor do lóbulo principal.
- HPBW = |ang_right - ang_left|
- Regras:
  - varredura a partir do pico para esquerda/direita
  - interpolação linear entre amostras para cruzamento preciso

### 5.5 First Null (mínimo local após lóbulo principal)
- procurar mínimo local após o pico até encontrar o primeiro vale significativo.
- heurística mínima:
  - buscar mudança de derivada (descendo → subindo)
- Se não encontrar, retornar NaN ou flag.

### 5.6 Front-to-Back (F/B)
- frente = pico
- costas = valor no ângulo oposto:
  - HRP (360): `ang_back = ang_peak ± 180°`
  - VRP (180): usar ângulo mais distante do pico (ou equivalente)
- interpolar valor em `ang_back`
- `FB = peak_db - back_db`

### 5.7 D2D do corte (HRP/VRP)
Implementar conforme fórmula correta (obrigatório):
- `P = (E/Emax)^2`
- integrar em rad
- HRP: `D2D = 2π / ∫ P(φ) dφ`
- VRP: `D2D =  π / ∫ P(θ) dθ`
- retornar `d2d_lin` e `d2d_db = 10*log10(d2d_lin)`

O kernel recebe `span_mode`:
- `0` → HRP (2π)
- `1` → VRP (π)

### 5.8 Decimation inteligente (`resample.py`)
Objetivo: reduzir N pontos para no máximo `target_rows` preservando pontos relevantes.
- manter: extremos, pico, mínimo global
- completar com stride uniforme

Saída:
- índices selecionados (int32) em ordem crescente

---

## 6) Paralelismo e fastmath (política)
- `fastmath=False` por padrão.
- `parallel=True` só para loops muito grandes ou batch de muitos cortes.
- Para N ~ 1801, geralmente **não** usar parallel.

---

## 7) Warm-up (pré-compilação) sem travar UI (obrigatório)
Criar `core/numba_kernels/warmup.py`:
- `warmup_all_kernels()` chama kernels com arrays pequenos (ex.: 32 pontos)
- UI chama warmup via worker thread e loga o progresso.

---

## 8) Suite de testes (obrigatória)
### 8.1 Tipos
1) Unit tests: trapz, deg2rad, pico, HPBW, D2D, decimation
2) Comparação com NumPy: implementar referência e comparar
3) Regressão: datasets reais pequenos do projeto
4) Bench: medir speedup vs baseline

### 8.2 Tolerâncias
- dB: `abs_diff <= 1e-2` (float64)
- ângulo: `abs_diff <= 1e-2 deg`
- D2D: `rel_err <= 1e-6`

### 8.3 Execução
```bash
pytest -q
pytest -q tests/perf --benchmark-only
```

---

## 9) Critérios de aceite
- kernels em `core/numba_kernels/*`
- wrappers em `core/analysis/pattern_metrics.py`
- warm-up rodando em background
- testes passando + benchmarks implementados
- speedup real em métricas 1D

---

## 10) Observações finais
- Evitar `np.interp` dentro de numba; implementar interpolação linear em loop.
- Garantir que o vetor de ângulos seja monotônico; se não for, ordenar no wrapper.
- Padronizar ranges (0..360 ou -180..180) e manter consistente.

