# divisor_page_refactor.md
## EFTX — Instrução para o agente: organizar a página “Divisor” (Inputs → Solver → Resultados + Variações)

### Objetivo
Reorganizar a página **Divisor** para:
1) deixar **variáveis do projeto totalmente editáveis** (com validação, unidades e “apply”),
2) aplicar mudanças **direto no solver (AEDT/HFSS)**,
3) **armazenar resultados por variação** (histórico de runs), permitindo comparar curvas/metrics.

---

## 1) Problemas observados na UI atual (com base nas telas)
- Área central grande “vazia” (layout não prioriza o que o usuário faz mais: parâmetros ↔ solver ↔ resultados).
- Inputs e resultados estão “soltos” e competem pelo espaço.
- Falta um mecanismo claro de **“dirty state”**: usuário altera valor → o sistema não indica o que mudou e o que já foi aplicado no AEDT.
- Resultados não têm **organização por run/variação** (não dá para comparar facilmente).
- Quando falha o plot (erro de path/NoneType), a UI não guia o usuário (ex.: botão Replot habilitado sem dados).

---

## 2) Layout proposto (Viewport-first para resultados, mas com Inputs editáveis)
Usar um layout **3-zonas** com splitters e painéis colapsáveis.

### 2.1 Estrutura
**Topo (fixo, compacto):** “Project Header”
- Nome do projeto/design + status do backend (AEDT/FreeCAD/FEM) + caminho do .aedt (clicável)
- Botões pequenos: `Open AEDT`, `Save Snapshot`, `Reset`, `Help`

**Corpo (principal):** `QSplitter(Qt.Horizontal)`  
- **Esquerda (Inputs + Variáveis + Splits)** — largura ~320–420 px (colapsável)
- **Direita (Resultados)** — ocupa o resto (prioridade)

**Inferior (dock):** `Log / Diagnostics` (altura mínima + auto-expand em erro)

---

## 3) Painel esquerdo — “Inputs & Variations”
Dividir em 3 cards colapsáveis (accordion):

### Card A — Project Variables (EDITÁVEL)
Substituir vários QLineEdit soltos por **tabela de variáveis** (QTableView):

**Colunas sugeridas**
- `Var` (nome interno: `total_power_w`, `f_start_mhz`, `d_ext_mm`, etc.)
- `Value` (editável)
- `Unit` (mm, MHz, W, deg…)
- `Type` (float/int/enum)
- `Dirty` (badge/ícone)
- `Source` (UI/AEDT/default)
- `Notes` (opcional)

**Controles**
- `Apply to Solver` (aplica somente dirty)
- `Revert` (volta ao último snapshot aplicado)
- `Snapshot` (salva “baseline” com timestamp)
- `Queue Run` (cria variação e roda)
- `Auto-Run` (toggle: aplicar → rodar setup automaticamente)

**Validação**
- ranges (ex.: f_start < f_stop)
- n_sections e outputs inteiros ≥ 1
- wall_thick > 0
- dielectric enum válido
- impedir apply se inválido (mostrar erro inline)

**UX crítico**
- Quando usuário edita, marcar `Dirty=True` e mudar cor de fundo da célula.
- Mostrar “Pending changes: N”.

---

### Card B — Output Split Setup (EDITÁVEL e consistente)
Transformar a lista OUT1..OUTN em uma **QTableView** também:

**Colunas**
- `Output` (OUT 1..N)
- `Percent (%)` (editável)
- `Phase (deg)` (editável)
- `Power (W)` (auto: total_power * percent/100)
- `Δ from equal` (auto: para debug)

**Ações**
- `Build Outputs` (gera N linhas com defaults)
- `Equal` (percent = 100/N)
- `Normalize` (reescala soma para 100%)
- `Lock sum=100%` (toggle)
- `Phase presets` (0°, 90°, 180° etc., opcional)

**Validação**
- soma = 100% (com tolerância 1e-6)
- percent ≥ 0
- phase normalizada [-180, 180] ou [0, 360] (padronizar)

---

### Card C — Variations / Runs (Histórico)
Adicionar um painel “Runs” (lista):

**Cada run salva**
- run_id, timestamp
- parâmetros completos (variáveis + split)
- AEDT project path, design, setup, sweep
- status (queued/running/ok/error)
- métricas principais (RL_min, VSWR_max, imbalance_dB, phase_error_deg)
- referência aos resultados (arquivos/cache)

**UI**
- lista (QListView/QTableView) + botão:
  - `Run Selected`
  - `Duplicate as New`
  - `Compare`
  - `Export CSV/JSON`
  - `Delete`

**Comportamento**
- clicar num run carrega resultados no painel direito
- compare permite sobrepor curvas (até N runs)

---

## 4) Painel direito — “Results”
Resultados devem ser claros e rápidos de interpretar.

### 4.1 Topo: Summary cards (KPIs)
Uma faixa com 4–6 cards (em grid):
- `RL min (dB)` / `VSWR max`
- `Z @ f0` (ou banda)
- `Insertion Loss` (se houver)
- `Amplitude imbalance (dB)`
- `Phase error (deg)`
- `Bandwidth (spec met)`

**Regra:** se não há dados, mostrar “—” e um hint “Run the solver”.

---

### 4.2 Tabs de resultados (melhorar a hierarquia)
Manter tabs, mas organizar melhor:

1) **RF Plots**
   - dropdown: métrica (S11, S21.., Z, phase…)
   - dropdown: função (raw/smoothed/unwrap etc.)
   - toggles: show markers, show spec lines
   - botão: `Replot`
   - overlay: seleção de runs (current + compare)

2) **Power Results**
   - tabela de potência por saída
   - eficiência / perdas

3) **AEDT Geometry**
   - mini-preview (se existir) + dados de geometria
   - botão `Open in AEDT`

4) **AEDT Log**
   - log filtrável + botão copiar/exportar

---

## 5) Pipeline “Editar → Aplicar → Rodar → Armazenar”
Este é o núcleo funcional.

### 5.1 Estados
- `DIRTY_UI`: usuário alterou valores localmente
- `APPLIED_SOLVER`: valores enviados ao AEDT
- `SOLVED`: resultados disponíveis
- `FAILED`: erro + run marcado

### 5.2 Sequência padrão (botão “Queue Run”)
1. Validar inputs
2. Criar run (snapshot completo)
3. Aplicar variáveis no AEDT (somente diferenças)
4. Regenerar geometria se necessário
5. Rodar setup/sweep
6. Extrair resultados
7. Calcular KPIs
8. Salvar em cache (JSON + CSV + imagens opcional)
9. Atualizar UI (KPIs + plots + logs)

---

## 6) Robustez (corrigir erro típico do print)
Erro visto: **NoneType** em path ao extrair S11.

### 6.1 Regras
- Nunca chamar extração/plot se `project_path` ou `result_path` for None.
- Desabilitar `Replot` quando não houver dataset carregado.
- Se solver falhar, registrar:
  - stack trace
  - contexto (design/setup/sweep)
  - run_id

### 6.2 UX em erro
- Mostrar banner: “RF data not available. Check AEDT path / run solver.”
- Botões úteis: `Open AEDT`, `Retry`, `Copy error`, `Show diagnostics`.

---

## 7) Estrutura de dados (sugestão)
Criar objetos claros:

- `ProjectState`
  - `variables: dict[str, VarValue]`
  - `outputs: list[OutputRow]`
  - `aedt: AEDTContext`
  - `runs: list[RunRecord]`

- `RunRecord`
  - `run_id`, `timestamp`, `status`
  - `variables_snapshot`, `outputs_snapshot`
  - `aedt_context_snapshot`
  - `metrics: dict`
  - `artifacts: paths`

- `ResultsStore`
  - `save_run(run_record)`
  - `load_run(run_id)`
  - `export_csv(run_id)`
  - `export_json(run_id)`

Persistência recomendada:
- pasta do projeto: `runs/<run_id>/`
  - `inputs.json`
  - `metrics.json`
  - `rf_curves.csv` (ou parquet)
  - `plots/*.png` (opcional)

---

## 8) Checklist de implementação (ordem recomendada)
### Sprint 1 — UI e editabilidade
- [ ] Converter inputs em `QTableView` de variáveis + validação + dirty state
- [ ] Converter Output Split Setup em `QTableView` com soma/normalize
- [ ] Inserir splitters e colapsáveis (viewport/results ganham espaço)
- [ ] Desabilitar ações quando não aplicável (ex.: replot sem dados)

### Sprint 2 — Apply e integração solver
- [ ] Implementar `Apply to Solver` (diferenças)
- [ ] Implementar `Queue Run` (snapshot + status)
- [ ] Implementar “run pipeline” com logs

### Sprint 3 — Results + store
- [ ] Armazenar resultados por run
- [ ] KPIs e summary cards
- [ ] Comparação de runs (overlay de curvas)

### Sprint 4 — Qualidade/robustez
- [ ] Tratamento completo de erros (None paths, AEDT offline)
- [ ] Export CSV/JSON
- [ ] Reprodutibilidade: reabrir projeto e recarregar runs

---

## 9) Prompts descritivos (referência das imagens recebidas)
### Imagem 1 (Divisor com resultados)
“Dark-themed engineering desktop app interface titled ‘Conversor & Biblioteca de Diagramas’, Divisor tab selected. Top row contains project input fields like total power, outputs, start/stop frequency, diameter, wall thickness, number of sections and dielectric selector, with action buttons. Middle section shows output split setup table (OUT1–OUT4 with percent and phase). Bottom section shows RF plots with dropdown controls and three stacked line charts.”

### Imagem 2 (Divisor com anotação e erro)
“Same dark-themed engineering app Divisor tab with an error message about failing to extract S11 solution data from HFSS (NoneType path). Red handwritten annotations circle the output split table and point to an empty central area, indicating where to reorganize the layout. Bottom plot area shows blank graphs and message ‘No RF data loaded yet’.”

---

## 10) Critérios de aceitação (QA)
- [ ] Variáveis do projeto editáveis em tabela com validação e dirty state.
- [ ] Botão `Apply to Solver` aplica somente o que mudou e atualiza status.
- [ ] `Queue Run` cria run, roda solver, salva resultados e atualiza plots.
- [ ] Lista de runs permite recarregar e comparar variações.
- [ ] UI não mostra áreas “mortas”: resultados têm prioridade; painéis colapsáveis.
- [ ] Erros (como path None) não quebram a UI e guiam o usuário para corrigir.
