# hfss_ui_inline.md — AEDT Live: remover modal, aplicar funções nativas HFSS e plotar inline com inspector à direita

Este documento instrui o agente a **refatorar apenas a aba AEDT Live** para:
1) **eliminar qualquer janela modal** (Toplevel/CTkToplevel) usada para “Report/Review”.
2) **aplicar funções nativas do HFSS** (ex.: `dB()`, `normalize()`, `dB10normalize()`, `dB20normalize()`, `mag()`, etc.) **no próprio HFSS/AEDT** antes de trazer os dados para o app.
3) **plotar os diagramas na área grande central** (abaixo dos comandos), exatamente onde o usuário mostrou.
4) mover **metadados + logs** para um **painel à direita** (Inspector) com rolagem e export/copy.
5) garantir **botão direito do mouse** funcionando no plot desta aba (context menu + markers).

> Regras inegociáveis:
- Não alterar comportamento das outras abas.
- Não usar `matplotlib.pyplot` (`plt.*`) em UI embutida.
- Não travar a UI: chamadas AEDT e parsing pesado em worker thread.
- O agente deve testar com AEDT/HFSS real (Attach e/ou New Session) e anexar evidências.

---

## 0) DoD (Definition of Done)

### Aceitação (PASS)
- [ ] Não existe mais **nenhuma modal** para report/review dentro da aba AEDT Live.
- [ ] A área grande central (abaixo dos comandos) exibe o(s) gráfico(s) HRP/VRP **inline**.
- [ ] O painel da direita mostra:
  - [ ] metadados do dataset atual (setup, sphere, expr, freq, pontos, min/max, timing)
  - [ ] logs (scroll) com botões Copy/Export/Clear
- [ ] O campo de expressão usa **funções nativas HFSS**:
  - Ex.: `dB10normalize(GainTotal)` ou `dB(GainTotal)`
- [ ] Pull HRP / Pull VRP atualizam o plot e o metadata imediatamente.
- [ ] Botão direito do mouse no plot abre menu contextual (copy cursor/add marker/clear/export/reset).
- [ ] UI continua responsiva durante pull/3D (worker thread).

---

## 1) Layout alvo (igual ao que o usuário quer)

### 1.1. Estrutura visual
Na aba **AEDT Live**:

**A) Toolbar (topo):** mantém os controles existentes  
- AEDT ver, Mode (Attach/New), Non-graphical, Project/Design/Setup/Sphere, Connect/Disconnect/Refresh/Create Sphere, etc.
- Controles de post-process: Quantity/Function/Expr, Freq, VRP Phi, Theta/Phi pts, dB floor, gamma, Pull HRP/VRP/3D, Send to Project/Library.

**B) Centro (área grande escura abaixo dos comandos):** vira **Plot Area**  
- Matplotlib embutido (OO), com opção de view:
  - HRP | VRP | Both
- Exibe o gráfico de irradiação onde hoje o app “plota metadados”.

**C) Painel à direita (Inspector):**  
- seção Metadata (readonly)
- seção Logs (readonly)
- seção Review/Resample (colapsável) — substitui modal
- botões Copy/Export/Clear (logs) e Copy meta

---

## 2) Implementação: remover modal e plotar inline

### 2.1. Localizar e remover modal
O agente deve procurar por qualquer ocorrência de:
- `CTkToplevel(...)` / `tk.Toplevel(...)`
- `.grab_set()`, `.wait_window()`, `.transient()`
- “Review/Resample window”, “Report window”, “Modal”, etc.

**Ação:** eliminar o código de criação da janela modal.  
**Substituir por:** componentes permanentes dentro do painel da direita.

### 2.2. Criar split layout (Plot + Inspector)
No módulo da aba AEDT Live (ex.: `plugins/aedt_live/aedt_live_plugin/ui_tab.py` ou equivalente), depois de montar a toolbar:

1) criar `content_frame`
2) criar `plot_frame` (col 0) e `right_frame` (col 1)
3) configurar pesos de grid:

```python
content_frame.grid_columnconfigure(0, weight=4)
content_frame.grid_columnconfigure(1, weight=1)
content_frame.grid_rowconfigure(0, weight=1)
```

### 2.3. PlotPanel OO (sem plt)
Implementar/reusar um `PlotPanel` com:
- `Figure()`, `ax = fig.add_subplot(111)`
- `FigureCanvasTkAgg(fig, master=plot_frame)`
- (opcional) toolbar mínima própria (reset/export)

**Importante:** updates via `set_data` e `canvas.draw_idle()`.

---

## 3) Expression Builder: funções nativas do HFSS antes do plot

### 3.1. Remover dependência de pós-process no Python (quando “HFSS-like”)
A expressão deve ser montada como string HFSS e enviada ao `far_field()`.

### 3.2. UI do Expression Builder
Substituir o campo único `Expr` por:

- Dropdown `Quantity`: `GainTotal, GainTheta, GainPhi, GainLHCP, GainRHCP, GainX, GainY, GainZ, ...`
- Dropdown `Function`: `<none>, dB, normalize, dB10normalize, dB20normalize, mag, abs, real, imag`
- Campo `Expr` (readonly ou editável) refletindo a expressão final

Regra de montagem:
- se func == `<none>`: `expr = quantity`
- else: `expr = f"{func}({quantity})"`

### 3.3. Aplicar expr na extração (HRP/VRP/3D)
Ao clicar Pull:

- HRP: `Theta=90deg`, `primary_sweep="Phi"`
- VRP: `Phi=0deg` (ou selecionável), `primary_sweep="Theta"`
- 3D: grid θ×φ usando expr escolhida (preferencialmente via export/FFD ou multi-sweep)

**Obrigatório:** salvar `expr` em `meta["expr"]` e exibir no Inspector.

---

## 4) Onde plotar (área central) e o que plotar

### 4.1. Modos de visualização
Adicionar um toggle `View` no topo do plot:
- `HRP`, `VRP`, `Both`

### 4.2. Plot HRP
- eixo x: Phi (graus) [-180..180] ou [0..360] conforme dataset
- eixo y: mag (dB ou lin conforme modo)
- label: `expr`, `setup`, `sphere`, `freq`

### 4.3. Plot VRP
- eixo x: Elevação (graus), com conversão:
  - `elev = 90 - theta`
- eixo y: mag (dB/lin)
- label idem

### 4.4. Plot Both
- overlay com cores diferentes e legenda
- NÃO recriar artists: manter `line_hrp`, `line_vrp` persistentes.

---

## 5) Mover metadados e logs para o lado direito

### 5.1. MetadataBox
Criar um `CTkTextbox` readonly:
- `Project`, `Design`, `Setup`, `Sphere`
- `Expr`, `Quantity`, `Function`
- `Freq`, `Npts`, `Primary sweep`, `Fixed vars`
- `Min/Max/Avg`, `Beamwidth (XdB)` se disponível
- `Timing` (pull time, parse time, export time)
- `Dataset id/hash` (para cache)

### 5.2. LogBox
Criar `CTkTextbox` readonly com:
- timestamp + nível + mensagem
- botões:
  - `Copy`
  - `Export` (txt)
  - `Clear`

**Regra:** qualquer log que hoje vai para a área central deve ir para LogBox.

---

## 6) Review/Resample: substituir modal por seção colapsável no Inspector

Criar uma seção “Review/Resample” no painel direito:
- `Resample mode`: `snap|interp`
- `Step`: `1deg/0.5deg/0.1deg/custom`
- `Apply (pending)` cria dataset “pending”
- `Commit to Project` / `Commit to Library`

**Importante:** sem abrir janela modal.

---

## 7) Botão direito e markers na aba AEDT Live

### 7.1. Context menu do plot
No canvas Matplotlib:
- conectar `button_press_event`
- se `event.button in (2,3)`: abrir menu contextual

Menu mínimo:
- Copy cursor (ang, mag)
- Add marker at cursor
- Clear markers
- Export PNG/CSV
- Reset view

### 7.2. Markers destravados
Markers devem ser:
- artists persistentes
- drag com `motion_notify_event`
- update rápido com `draw_idle()`

---

## 8) Performance (para não ficar lento)

- nunca chamar `canvas.draw()` em mousemove → usar `draw_idle()`
- throttle em motion events (30–60 Hz)
- update de tabelas/metadados no `button_release_event`
- cache por chave:
  `(project, design, setup, sphere, expr, freq, sweep, fixed_vars, pts)`

---

## 9) Passos de implementação (checklist por commit)

### Commit 1 — Layout split + PlotPanel
- [ ] criar `content_frame`, `plot_frame`, `right_frame`
- [ ] embutir Matplotlib OO no plot_frame
- [ ] criar MetadataBox + LogBox no right_frame

### Commit 2 — Remover modal e mover Review/Resample para Inspector
- [ ] remover `CTkToplevel`/modal
- [ ] implementar seção colapsável Review/Resample no painel direito

### Commit 3 — Expression Builder (HFSS native functions)
- [ ] dropdown Quantity + Function + Expr final
- [ ] passar `expr` diretamente ao `far_field()`
- [ ] registrar `expr` em metadados

### Commit 4 — Plot inline (HRP/VRP/Both)
- [ ] Pull HRP plota na área central
- [ ] Pull VRP plota na área central
- [ ] Both sobrepõe
- [ ] draw_idle + artists persistentes

### Commit 5 — Context menu + markers (botão direito)
- [ ] menu contextual no plot
- [ ] add marker + drag + clear

### Commit 6 — Testes reais (evidência)
- [ ] Attach com AEDT aberto
- [ ] Pull HRP/VRP com `dB10normalize(GainTotal)` (ou similar)
- [ ] logs + metadata atualizando
- [ ] screenshot do resultado

---

## 10) Evidência obrigatória no PR
O agente deve anexar:
- screenshot da aba AEDT Live com HRP/VRP plotados inline
- screenshot do Inspector com metadata + logs
- log com timings (pull/parse/export)
- confirmação de que não existe modal

---

## 11) Observações finais
- A UI deve ficar “HFSS-like”: o usuário escolhe Quantity/Function, o HFSS aplica e retorna.
- Não duplicar funcionalidades já existentes em outras abas; AEDT Live apenas injeta dados no pipeline e fornece visualização inline.
