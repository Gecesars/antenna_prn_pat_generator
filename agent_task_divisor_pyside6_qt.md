# agent_task_divisor_pyside6_qt.md
## EFTX — Comando para o agente (ENTREGA COMPLETA): refatorar “Divisor” em **PySide6/Qt** + gráficos interativos + integração HFSS (PyAEDT) + “Abrir Análise Avançada”

> **Objetivo**: Reimplementar a página **Divisor** (que hoje está em CTk/CustomTkinter) como uma UI **PySide6 (Qt)** moderna, fluida e escalável, com:
> - Variáveis do projeto **editáveis** (com validação, unidades e *dirty state*)
> - “Apply to Solver” aplicando direto no **HFSS via PyAEDT**
> - “Queue Run” criando **variações/runs** e armazenando resultados por variação
> - **Gráficos interativos** (zoom/scroll fluido, pan, crosshair, markers)
> - Botão **“Abrir Análise Avançada”** abrindo uma janela Qt dedicada com ferramentas avançadas
>
> **Regra**: não travar UI durante solve — usar threads/worker e progress/log em tempo real.

---

## 0) Escopo obrigatório (o que o agente deve construir)

### A) UI principal “Divisor” (PySide6)
- Layout com **QSplitter** e painéis colapsáveis
- Tabela editável de **Project Variables** (QTableView + Model)
- Tabela editável de **Output Split Setup** (QTableView + Model)
- Painel **Runs/Variations** (lista/tabela com status e métricas)
- Área de **Resultados** (tabs + KPIs + gráficos)
- Dock inferior: **Log/Diagnostics** (compacto, auto-expand em erro)

### B) Integração solver HFSS (PyAEDT)
- Aplicar variáveis **somente dirty**
- Rodar setup/sweep
- Extrair S-parameters (S11, etc.), impedância, fase
- Calcular KPIs
- Persistir resultados por run

### C) Janela “Análise Avançada” (PySide6)
- Botão no topo: **Abrir Análise Avançada**
- Janela com:
  - overlay de múltiplos runs (comparação)
  - ferramentas de medição (Δx/Δy, marker A/B)
  - exportação de curvas (CSV/JSON)
  - filtros (banda, smoothing, unwrap)
  - inspeção de KPIs e tabela de resultados

---

## 1) Stack obrigatório (dependências)

### UI
- `PySide6`
- `pyqtgraph` (preferido para plots interativos fluidos)  
  - Alternativa: QtCharts (menos flexível); evitar matplotlib se o objetivo é máxima fluidez.

### Solver
- `pyaedt` (HFSS)
- `numpy`, `pandas`
- (opcional) `scipy` para smoothing/unwrap/metrics

### Persistência
- JSON + CSV (pandas)
- Estrutura em disco por run: `runs/<run_id>/...`

---

## 2) Estrutura de pastas (obrigatório)

Criar (ou ajustar) para:

```
app/
  ui/
    divisor/
      divisor_view.py              # QWidget principal da aba Divisor
      divisor_advanced_view.py     # Janela/QMainWindow “Análise Avançada”
      models/
        variables_table_model.py   # QAbstractTableModel p/ variáveis
        outputs_table_model.py     # QAbstractTableModel p/ outputs
        runs_table_model.py        # QAbstractTableModel p/ runs
      widgets/
        kpi_bar.py                 # cards de KPIs
        plot_panel.py              # wrapper pyqtgraph + controles
        collapsible_card.py        # card colapsável
  core/
    divisor/
      divisor_controller.py        # orquestra UI ↔ solver ↔ store
      state.py                     # ProjectState/RunRecord dataclasses
      validation.py                # validação de inputs e splits
      metrics.py                   # RLmin, VSWR, imbalance, phase error...
      results_store.py             # salvar/carregar runs
      hfss_adapter.py              # interface PyAEDT/HFSS (isolamento)
      worker.py                    # QRunnable/QThreadPool p/ solve async
  assets/
  main.py
```

**Regra**: separar UI (view) de lógica (core) usando padrão MVC/MVVM.

---

## 3) Layout e organização da UI (Divisor)

### 3.1 Header compacto (topo)
No topo da aba Divisor, mostrar:
- projeto/design/setup atual
- status do backend (HFSS conectado? caminho .aedt? versão)
- botões:
  - `Open AEDT`
  - `Snapshot`
  - `Apply to Solver`
  - `Queue Run`
  - `Run Selected`
  - `Compare`
  - **`Abrir Análise Avançada`** (novo)

### 3.2 Corpo com QSplitter horizontal (viewport-first para resultados)
- **Esquerda** (Inputs & Variations) ~320–420px, colapsável
- **Direita** (Results) ocupa o resto

### 3.3 Painel esquerdo (accordion de 3 cards)
**Card A — Project Variables**
- `QTableView` com colunas:
  - Var | Value (edit) | Unit | Type | Dirty | Source | Notes
- Mostrar “Pending changes: N”
- Validação inline (vermelho + tooltip)
- Botões: Apply, Revert, Snapshot, Auto-Run toggle

**Card B — Output Split Setup**
- `QTableView` com:
  - Output | Percent (%) | Phase (deg) | Power (W auto) | Δ equal (auto)
- Botões:
  - Build Outputs
  - Equal
  - Normalize (sum=100%)
  - Lock sum=100%

**Card C — Runs/Variations**
- Lista/tabela:
  - run_id, timestamp, status, RLmin, VSWRmax, imbalance, phase_error
- Ações:
  - Run Selected
  - Duplicate as New
  - Compare (overlay)
  - Export CSV/JSON
  - Delete

### 3.4 Painel direito (Results)
- **KPI Bar** (cards):
  - RL min (dB), VSWR max, Z@f0, bandwidth, imbalance, phase error
- Tabs:
  1) RF Plots
  2) Power Results
  3) AEDT Geometry (info + botão abrir)
  4) AEDT Log

---

## 4) Gráficos interativos (pyqtgraph) — requisitos de UX

Implementar em `plot_panel.py`:

- Zoom/pan com mouse **fluido**
- Scroll zoom **cursor-centric** (quando possível)
- Crosshair + leitura X/Y em tempo real
- Marcadores A/B (máx 2 verticais + 2 horizontais por plot)
- “Fit All” e “Fit Selection”
- Overlay de múltiplos runs (cores automáticas)
- Toggle: raw / smoothed / unwrap phase
- Export plot (PNG/SVG) e export data (CSV)

**Não bloquear UI**: atualização deve ser rápida mesmo com muitos pontos.

---

## 5) Pipeline funcional “Editar → Apply → Run → Store → Compare”

### 5.1 Estados
- `DIRTY_UI` (mudou local)
- `APPLIED_SOLVER` (enviado ao AEDT)
- `RUNNING`
- `SOLVED`
- `FAILED`

### 5.2 Apply to Solver (somente dirty)
- Validar campos (ex.: f_start < f_stop)
- Enviar apenas variáveis modificadas para HFSS (via `hfss_adapter`)
- Atualizar Source=“AEDT” e Dirty=False ao confirmar

### 5.3 Queue Run
- Validar inputs
- Criar `RunRecord` com snapshot completo
- Persistir `inputs.json` antes de rodar (reprodutibilidade)
- Rodar solve em worker thread:
  - atualizar status/progresso
  - capturar log em tempo real

### 5.4 Após solve
- Extrair curvas (S11 etc.)
- Calcular KPIs
- Persistir:
  - `metrics.json`
  - `rf_curves.csv` (pandas)
  - `aedt_log.txt`
  - (opcional) imagens geradas
- Atualizar UI:
  - KPIs
  - plots
  - run status

### 5.5 Compare
- Selecionar 2..N runs e sobrepor curvas
- KPIs comparativos e tabela de diferenças

---

## 6) Robustez (erro NoneType path / sem dados)
**Obrigatório**:
- Desabilitar “Replot” e controles de plot se não houver dataset carregado
- Antes de extrair resultados, checar:
  - project_path/design/setup/sweep não nulos
- Se falhar:
  - marcar run FAILED
  - salvar stack trace no run
  - mostrar banner e ações: Retry, Open AEDT, Copy error, Diagnostics

---

## 7) HFSS Adapter (isolamento do PyAEDT)
Criar `core/divisor/hfss_adapter.py` com API clara:

- `connect(project_path, design_name)`
- `apply_variables(dict)`
- `build_geometry_if_needed()`
- `run_solve(setup_name, sweep_name)`
- `export_sparams()` → retorna arrays freq + S11/S21...
- `get_impedance()` → freq + Z
- `get_phase()` → freq + phase
- `get_log()` → texto
- `close()`

**Regra**: UI nunca chama PyAEDT diretamente — sempre via adapter.

---

## 8) Threading / worker
Usar `QThreadPool + QRunnable` (ou `QThread`) para:
- Apply (se pesado)
- Solve (sempre async)
- Export/parse de resultados (se grande)

Emitir sinais:
- progress(int)
- log_line(str)
- finished(result)
- failed(error)

---

## 9) Critérios de aceitação (QA)
- [ ] UI Divisor totalmente em PySide6/Qt (sem CTk)
- [ ] Variáveis editáveis em tabela com dirty state + validação
- [ ] Apply to Solver aplica só o que mudou
- [ ] Queue Run cria run, roda, salva resultados por variação
- [ ] Lista de runs carrega e compara resultados
- [ ] Plots interativos fluidos (pyqtgraph) com markers/crosshair
- [ ] Botão **Abrir Análise Avançada** abre janela funcional e integrada
- [ ] Erros (ex.: None path) não quebram UI e guiam correção
- [ ] Reabrir app permite ver runs anteriores (persistência)

---

## 10) Plano de implementação (sprints)
### Sprint 1 — Base PySide6 + layout + tabelas
- divisor_view.py com splitters e cards
- models de variáveis/outputs
- validação + dirty state

### Sprint 2 — HFSS adapter + Apply
- implementar hfss_adapter
- Apply to Solver robusto (diferenças)

### Sprint 3 — Runs + Solve async + store
- worker thread
- Queue Run
- results_store
- KPIs e RF Plots

### Sprint 4 — Análise Avançada + compare + refinamentos
- janela advanced
- overlay runs + export
- estabilidade/perf e UX final

---

## 11) Nota de qualidade (obrigatório)
- A UI deve ser “viewport-first”: resultados dominam a tela; painéis colapsáveis.
- Nada de “área morta” no centro.
- Não travar UI durante solve.
- Logs claros e rastreáveis por run.
