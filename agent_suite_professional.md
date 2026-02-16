# Instrução completa para o agente: transformar o PAT Converter em uma suíte profissional (EFTX)

**Contexto técnico (estado atual)**  
A aplicação atual é um desktop app em **CustomTkinter + Matplotlib**, com abas `Arquivo`, `Composição Vertical`, `Composição Horizontal` e `Diagramas (Batch)` (biblioteca). Ela já possui: parsing robusto (PRN/PAT/CSV/TXT), reamostragem padronizada (HRP -180..180 @1°; VRP -90..90 @0,1°), export PAT/ADT/PRN, biblioteca SQLite e *null fill* por ordem de nulo (modo amplitude/phase/both). fileciteturn6file0L10-L31 fileciteturn6file1L27-L42 fileciteturn6file6L24-L38

A missão deste trabalho é **elevar a aplicação ao padrão de suíte profissional**:
- UX de engenharia (controles de mouse, medições, cursores, zoom/pan, contexto);
- barra de menus completa com atalhos e itens típicos (File/Edit/View/Tools/Window/Help);
- robustez “à prova de usuário”: validações, logs, autosave, atomic writes, tratamento de erros, testes de round-trip;
- arquitetura modular e testável (core vs UI vs IO).

---

## 0) Objetivo e definição de pronto (Definition of Done)

### 0.1. DoD (o que deve estar entregue)
1) **Menu bar nativa** (Windows) com: File, Edit, View, Tools, Window, Help; com atalhos padrão.
2) **Interação por mouse (botão esquerdo)**:
   - click: captura ponto (ângulo, nível); exibe leitura em status bar;
   - dois pontos (A/B): delta θ, delta nível, distância angular (wrap) e “ripple local”;
   - arrastar marcador: move e atualiza leitura;
   - scroll: zoom; click+drag: pan (ou toolbar); double-click: reset view;
   - em polar: comportamento coerente (theta/r).
3) **Painel de medições** (dock/side) com A/B, Δ, e botão “copiar” (clipboard).
4) **Robustez**:
   - logging em arquivo (rotacionado) + janela “Log”;
   - exportação com escrita atômica (tempfile + replace);
   - validação de PRN/PAT/CSV (erros claros, sem crash);
   - correção estrutural: remover duplicação de funções e centralizar conversões de ângulo.
5) **Qualidade**:
   - testes unitários (core) + testes de round-trip (PRN↔interno↔PRN; PAT↔interno↔PAT);
   - lint (ruff/black) + type hints (mypy opcional).
6) **Empacotamento**:
   - build MSI (cx_Freeze) com ícone, versão semântica, pasta AppData, e atualização de dependências.

---

## 1) Refatoração mínima de arquitetura (sem “reescrever tudo”)

### 1.1. Estrutura de pastas sugerida
Criar uma organização clara “core vs ui”:

```
eftx_converter/
  app/
    main.py                      # entrypoint
    ui/
      app_shell.py               # janela principal, menus, status bar
      tabs/
        tab_file.py
        tab_vertical.py
        tab_horizontal.py
        tab_batch.py
      widgets/
        plot_panel.py            # wrapper Matplotlib + interações
        measure_panel.py         # A/B + Δ + copiar
        dialogs.py               # about, prefs, errors
      interactions/
        plot_interactor.py       # mouse/keys (matplotlib events)
        markers.py               # markers/crosshair
    core/
      patterns.py                # dataclasses PatternCut/Pattern2D
      angles.py                  # convenções + wrap + conversões
      resample.py                # reamostragem HRP/VRP (único lugar)
      metrics.py                 # HPBW, D2D, F/B, 1º nulo
      io/
        parse_auto.py            # roteamento por extensão
        parse_pat.py
        parse_prn.py
        write_pat.py
        write_prn.py
      synthesis/
        null_fill_synthesis.py   # já existe — manter e isolar dependências UI
    persistence/
      db.py                      # SQLite (biblioteca)
      project.py                 # eftxproj.json + autosave + export registry
    util/
      logging_setup.py
      atomic_write.py
      validation.py
      threading.py               # run_in_worker + progress
tests/
  test_angles.py
  test_resample.py
  test_parse_prn_roundtrip.py
  test_parse_pat_roundtrip.py
  test_null_fill_synthesis.py
```

**Regra de ouro**: `core/` nunca importa `ctk`/UI. UI apenas chama core.

### 1.2. Dataclasses para padronizar o estado
Criar um tipo canônico para cortes (VRP/HRP):

```python
from dataclasses import dataclass
import numpy as np
from typing import Literal, Dict, Any

CutType = Literal["H", "V"]

@dataclass(frozen=True)
class PatternCut:
    type: CutType                 # "H" ou "V"
    angles_deg: np.ndarray        # HRP em [-180,180], VRP em [-90,90]
    values_lin: np.ndarray        # linear normalizado (E/Emax)
    meta: Dict[str, Any]          # NAME/MAKE/FREQ/Gain/etc + origem
```

Tudo (parsers, resample, plots, exports) deve operar com `PatternCut` para evitar “valores soltos”.

---

## 2) Barra de menus profissional (Tk Menu + CTk)

A aplicação atual é `PATConverterApp(ctk.CTk)` e já tem topbar custom. fileciteturn6file4L48-L80  
Adicionar **menu bar nativa** via `tk.Menu` (funciona em CTk porque CTk herda Tk).

### 2.1. Implementação base (app_shell.py)
```python
import tkinter as tk

def build_menubar(app):
    menubar = tk.Menu(app)

    # FILE
    m_file = tk.Menu(menubar, tearoff=0)
    m_file.add_command(label="New Project", accelerator="Ctrl+N", command=app.project_new)
    m_file.add_command(label="Open Project…", accelerator="Ctrl+O", command=app.project_open)
    m_file.add_command(label="Save Project", accelerator="Ctrl+S", command=app.project_save)
    m_file.add_command(label="Save Project As…", accelerator="Ctrl+Shift+S", command=app.project_save_as)
    m_file.add_separator()
    m_file.add_command(label="Load VRP…", accelerator="Ctrl+1", command=app.load_vrp)
    m_file.add_command(label="Load HRP…", accelerator="Ctrl+2", command=app.load_hrp)
    m_file.add_command(label="Import Batch…", accelerator="Ctrl+I", command=app.batch_import)
    m_file.add_separator()
    m_file.add_command(label="Export PAT (All)", accelerator="Ctrl+E", command=app.export_all_pat)
    m_file.add_command(label="Export PRN (All)", accelerator="Ctrl+P", command=app.export_all_prn)
    m_file.add_separator()
    m_file.add_command(label="Exit", accelerator="Alt+F4", command=app.quit)
    menubar.add_cascade(label="File", menu=m_file)

    # EDIT
    m_edit = tk.Menu(menubar, tearoff=0)
    m_edit.add_command(label="Copy Measurement", accelerator="Ctrl+C", command=app.copy_measurement)
    m_edit.add_separator()
    m_edit.add_command(label="Preferences…", accelerator="Ctrl+,", command=app.open_preferences)
    menubar.add_cascade(label="Edit", menu=m_edit)

    # VIEW
    m_view = tk.Menu(menubar, tearoff=0)
    m_view.add_command(label="Reset View", accelerator="R", command=app.view_reset)
    m_view.add_command(label="Toggle Grid", accelerator="G", command=app.view_toggle_grid)
    m_view.add_separator()
    m_view.add_command(label="Show Log", accelerator="Ctrl+L", command=app.open_log_window)
    menubar.add_cascade(label="View", menu=m_view)

    # TOOLS
    m_tools = tk.Menu(menubar, tearoff=0)
    m_tools.add_command(label="Null Fill Wizard…", accelerator="Ctrl+W", command=app.open_null_fill_wizard)
    m_tools.add_command(label="Validate PRN/PAT…", command=app.open_validator)
    menubar.add_cascade(label="Tools", menu=m_tools)

    # WINDOW
    m_win = tk.Menu(menubar, tearoff=0)
    m_win.add_command(label="Go to Arquivo", accelerator="Alt+1", command=lambda: app.tabs.set("Arquivo"))
    m_win.add_command(label="Go to Composição Vertical", accelerator="Alt+2", command=lambda: app.tabs.set("Composição Vertical"))
    m_win.add_command(label="Go to Composição Horizontal", accelerator="Alt+3", command=lambda: app.tabs.set("Composição Horizontal"))
    m_win.add_command(label="Go to Diagramas (Batch)", accelerator="Alt+4", command=lambda: app.tabs.set("Diagramas (Batch)"))
    menubar.add_cascade(label="Window", menu=m_win)

    # HELP
    m_help = tk.Menu(menubar, tearoff=0)
    m_help.add_command(label="Workflow / Help", command=app.show_help)
    m_help.add_command(label="About", command=app.show_about)
    menubar.add_cascade(label="Help", menu=m_help)

    app.config(menu=menubar)
    return menubar
```

### 2.2. Atalhos globais (bind_all)
Depois de criar o menu, registrar atalhos:
```python
def bind_shortcuts(app):
    app.bind_all("<Control-n>", lambda e: app.project_new())
    app.bind_all("<Control-o>", lambda e: app.project_open())
    app.bind_all("<Control-s>", lambda e: app.project_save())
    app.bind_all("<Control-Shift-S>", lambda e: app.project_save_as())
    app.bind_all("<Control-1>", lambda e: app.load_vrp())
    app.bind_all("<Control-2>", lambda e: app.load_hrp())
    app.bind_all("<Control-i>", lambda e: app.batch_import())
    app.bind_all("<Control-e>", lambda e: app.export_all_pat())
    app.bind_all("<Control-p>", lambda e: app.export_all_prn())
    app.bind_all("<Control-l>", lambda e: app.open_log_window())
    app.bind_all("g", lambda e: app.view_toggle_grid())
    app.bind_all("r", lambda e: app.view_reset())
```

**Aceitação:** menu funciona, atalhos funcionam em qualquer aba.

---

## 3) Interações de mouse (botão esquerdo) — padrão de suíte RF

A aplicação já plota VRP/HRP via Matplotlib (`ax.plot(...)`) e alterna Polar/Planar. fileciteturn6file5L59-L66  
Vamos padronizar um **PlotPanel** que encapsula: figura + canvas + interator.

### 3.1. Requisitos de interação
Implementar no mínimo:

- **Left click**: cria/atualiza marcador A (primeiro clique) e B (segundo clique).
- **Shift + left click**: força selecionar o marcador B.
- **Drag com left button** em cima do marcador: move marcador.
- **Mouse move**: crosshair + tooltip (ângulo e nível).
- **Scroll**: zoom em torno do cursor.
- **Double click**: reset view.
- **Right click** (extra profissional): menu contexto (copiar, limpar, exportar, enviar para estudo).

### 3.2. Modelo de dados da medição
A medição deve considerar wrap:
- HRP: wrap em 360 e exibir em [-180,180]
- VRP: limitado [-90,90]

E deve exibir:
- Ponto A: θ_A, E_A (dB e linear)
- Ponto B: θ_B, E_B
- Δθ (menor distância angular no wrap para HRP)
- ΔdB = dB(B) - dB(A)
- Ripple local (opcional): max-min numa janela ±X° em torno do ponto (configurável)

### 3.3. Implementação (plot_interactor.py)
Implementar um interator Matplotlib que usa `mpl_connect`.

**Observação:** para polar, `event.xdata` é theta em rad, `event.ydata` é r (linear). Para planar, `xdata` em deg e `ydata` linear.

```python
import numpy as np

class PlotInteractor:
    def __init__(self, ax, canvas, get_series_callable, kind: str):
        self.ax = ax
        self.canvas = canvas
        self.get_series = get_series_callable  # retorna (angles_deg, values_lin)
        self.kind = kind  # "H_planar" | "H_polar" | "V_planar"

        self.markerA = None
        self.markerB = None
        self.active = None  # "A" ou "B"
        self.dragging = False

        self._connect()

    def _connect(self):
        c = self.canvas.mpl_connect
        c("button_press_event", self.on_press)
        c("button_release_event", self.on_release)
        c("motion_notify_event", self.on_motion)
        c("scroll_event", self.on_scroll)

    def _nearest_point(self, theta_deg):
        ang, val = self.get_series()
        if ang is None or val is None or len(ang) == 0:
            return None
        # HRP: ang em [-180,180] (interno). Para distância, usar wrap.
        if self.kind.startswith("H"):
            # distância circular
            d = np.abs(((ang - theta_deg + 180) % 360) - 180)
            idx = int(np.argmin(d))
        else:
            idx = int(np.argmin(np.abs(ang - theta_deg)))
        return float(ang[idx]), float(val[idx])

    def _event_theta_deg(self, event):
        if self.kind.endswith("_polar"):
            if event.xdata is None:
                return None
            return float(np.rad2deg(event.xdata))
        else:
            if event.xdata is None:
                return None
            return float(event.xdata)

    def on_press(self, event):
        if event.inaxes != self.ax or event.button != 1:
            return
        theta = self._event_theta_deg(event)
        if theta is None:
            return

        # SHIFT: força B
        forceB = bool(getattr(event, "key", None) == "shift")
        self.active = "B" if (forceB or self.markerA is not None) else "A"

        # se click perto de um marcador, entra em drag
        if self._hit_marker(theta):
            self.dragging = True
            return

        self._set_marker(self.active, theta)
        self.canvas.draw_idle()

    def on_release(self, event):
        if event.button == 1:
            self.dragging = False

    def on_motion(self, event):
        if event.inaxes != self.ax:
            return
        theta = self._event_theta_deg(event)
        if theta is None:
            return

        if self.dragging and self.active:
            self._set_marker(self.active, theta)
            self.canvas.draw_idle()
        else:
            # opcional: crosshair + tooltip
            pass

    def on_scroll(self, event):
        # zoom centrado no cursor (profissional)
        if event.inaxes != self.ax:
            return
        base_scale = 1.15
        scale = 1/base_scale if event.button == "up" else base_scale

        if self.kind.endswith("_polar"):
            rmin, rmax = self.ax.get_ylim()
            r = event.ydata if event.ydata is not None else (rmin+rmax)/2
            new_min = r - (r - rmin)*scale
            new_max = r + (rmax - r)*scale
            self.ax.set_ylim(max(0, new_min), new_max)
        else:
            xmin, xmax = self.ax.get_xlim()
            x = event.xdata if event.xdata is not None else (xmin+xmax)/2
            new_min = x - (x - xmin)*scale
            new_max = x + (xmax - x)*scale
            self.ax.set_xlim(new_min, new_max)

        self.canvas.draw_idle()

    def _hit_marker(self, theta_deg, tol_deg=2.0):
        # se existe marcador e theta está próximo → hit
        for name, mk in [("A", self.markerA), ("B", self.markerB)]:
            if mk is None:
                continue
            t = mk["theta_deg"]
            d = abs(((theta_deg - t + 180) % 360) - 180) if self.kind.startswith("H") else abs(theta_deg - t)
            if d <= tol_deg:
                self.active = name
                return True
        return False

    def _set_marker(self, which, theta_deg):
        pt = self._nearest_point(theta_deg)
        if pt is None:
            return
        t_deg, v_lin = pt

        if which == "A":
            self.markerA = self._draw_marker(which, t_deg, v_lin)
        else:
            self.markerB = self._draw_marker(which, t_deg, v_lin)

        # emitir callback para UI atualizar painel de medidas
        if hasattr(self, "on_measure_update") and callable(self.on_measure_update):
            self.on_measure_update(self.markerA, self.markerB)

    def _draw_marker(self, which, t_deg, v_lin):
        # apagar existente
        prev = self.markerA if which=="A" else self.markerB
        if prev and prev.get("artist") is not None:
            try: prev["artist"].remove()
            except: pass

        if self.kind.endswith("_polar"):
            # polar: linha radial com theta fixo
            th = np.deg2rad(t_deg)
            artist = self.ax.plot([th, th], [0, v_lin], linewidth=1.2)[0]
        else:
            artist = self.ax.axvline(t_deg, linewidth=1.2)

        return {"which": which, "theta_deg": float(t_deg), "v_lin": float(v_lin), "artist": artist}
```

### 3.4. Painel de medições (measure_panel.py)
Criar um widget CTk com campos A/B/Δ e botões:

- `Clear markers`
- `Copy` (texto formatado para clipboard)
- `Export CSV (markers)` (opcional)

---

## 4) Context menu (profissional)
Adicionar um menu de contexto no plot (botão direito), com ações:

- Copy cursor (θ, dB)
- Copy A/B/Δ
- Clear markers
- Reset view
- Send cut to “Estudo Completo” slot (H1/V1 etc.)
- Export current cut (PAT/ADT/PNG)

Implementar isso via `tk.Menu` popup no evento de botão 3 (Matplotlib `button_press_event` com `event.button==3`), convertendo coordenadas para `canvas.get_tk_widget().winfo_rootx()` etc.

---

## 5) Robustez: logs, atomic write, validação, falhas previsíveis

### 5.1. Logging de produção
Criar `util/logging_setup.py`:
- `RotatingFileHandler` em `%APPDATA%\EFTX\PATConverter\logs\app.log`
- logs de exceções com stack trace
- UI “Show Log” abre `CTkToplevel` com tail do arquivo

### 5.2. Escrita atômica para exportações
Criar `util/atomic_write.py`:

- escrever em `path.tmp` (mesma pasta)
- `flush + fsync`
- `os.replace(tmp, path)` (atômico no Windows)

Usar isso em:
- write_prn
- write_pat
- export de JSON de projeto
- export bundle/manifest

### 5.3. Validação forte de PRN/PAT
O parser atual tem heurísticas e vários formatos. fileciteturn6file9L73-L92  
Criar `util/validation.py`:
- validações mínimas: monotonicidade de ângulo, faixa esperada, tamanho mínimo, valores não-NaN, normalização.
- warnings (não bloqueantes) e errors (bloqueantes).
- UI “Validate PRN/PAT” exibe relatório estruturado.

### 5.4. Corrigir risco estrutural conhecido
**Remover duplicações** e centralizar funções “canônicas”:
- existir **uma única** `write_prn_file()` (um módulo).
- existir **uma única** função de mapeamento vertical PRN 0..360 ↔ VRP -90..90 (um módulo: `core/angles.py`).

---

## 6) UX de suíte: status bar, progress, não congelar UI

### 6.1. Não congelar durante imports/exports
Para batch import e exports em lote, usar worker thread:

- `concurrent.futures.ThreadPoolExecutor(max_workers=1)`
- o worker executa parse/export
- a UI atualiza via `app.after(0, ...)`

### 6.2. Barra de progresso e cancelamento
Adicionar:
- progressbar na status bar
- botão Cancel (sinaliza flag thread-safe)
- “X/Y arquivos processados”

---

## 7) Funcionalidades “de suíte” (alta percepção de valor)

### 7.1. Recent Files + Recent Projects
No `project.py`, manter:
- `recent_files.json` e `recent_projects.json`
- menu “File > Recent”
- limite (ex.: 12 itens)
- limpar itens inexistentes automaticamente

### 7.2. Autosave do projeto
A cada mudança relevante (load VRP/HRP, alteração de parâmetros, síntese), salvar `autosave.eftxproj.json` em AppData.
- em crash, oferecer “Recover autosave”.

### 7.3. Export Wizard
Adicionar `Tools > Export Wizard…`:
- seleciona quais saídas: PAT / ADT / PRN / PNG / CSV / harness
- define pasta e prefixo
- executa com progress bar
- gera manifesto JSON (já existe a ideia de registry/bundle na doc) fileciteturn6file0L163-L178

### 7.4. Clipboard e relatórios
- Copy measurement (A/B/Δ) como texto
- Copy current metrics (HPBW, D2D, F/B, 1º nulo) (já há métricas em `deep3.py` segundo a doc) fileciteturn6file6L73-L89

---

## 8) Plano de execução por etapas (sem risco)

### Etapa 1 — Infra e menus (rápida, alto impacto)
- Implementar `build_menubar()` + `bind_shortcuts()`
- Implementar logger + “Show Log”
- Implementar atomic write em PRN/PAT/projeto

**Aceitação:** app abre com menu; exports não corrompem arquivo; logs funcionam.

### Etapa 2 — PlotPanel + interações por mouse
- Criar `PlotPanel` (fig+ax+canvas) e `PlotInteractor`
- Implementar markers A/B + painel de medição
- Implementar zoom scroll + reset

**Aceitação:** clique esquerdo cria marcadores e mede; arrastar move; zoom funciona.

### Etapa 3 — Context menu e ferramentas
- Context menu (right click) com ações úteis
- Validator PRN/PAT
- Wizard de export

**Aceitação:** usuário opera sem “clicar em 15 botões”; ações típicas em menu.

### Etapa 4 — Robustez e testes
- Centralizar `core/angles.py` e `core/resample.py`
- Remover duplicações de função
- Testes round-trip PRN/PAT e resample

**Aceitação:** testes passam; regressões reduzidas.

### Etapa 5 — Empacotamento e “polish”
- Ícone, versão, “About”, “Check updates” (opcional)
- MSI com cx_Freeze
- Documentação interna (Help) atualizada

---

## 9) Critérios de engenharia (não negociáveis)

1) **Sem crash por arquivo ruim**: todo parse deve virar error/warn no UI, nunca exception não tratada.
2) **Sem inconsistência de eixos**: HRP sempre [-180,180], VRP sempre [-90,90] internamente. fileciteturn6file1L29-L41
3) **Sem escrita parcial**: export sempre atômico.
4) **Sem UI congelar**: batch sempre em worker com progress.
5) **Sem “estado implícito” solto**: use `PatternCut` e `ProjectState` (dataclass) para serializar.

---

## 10) Entregáveis (arquivos que o agente deve criar/modificar)

### Novos arquivos (mínimo)
- `app/ui/app_shell.py` (menus + shortcuts)
- `app/ui/widgets/plot_panel.py`
- `app/ui/interactions/plot_interactor.py`
- `app/ui/widgets/measure_panel.py`
- `app/util/logging_setup.py`
- `app/util/atomic_write.py`
- `app/core/angles.py`
- `app/util/validation.py`
- `tests/test_parse_prn_roundtrip.py`
- `tests/test_parse_pat_roundtrip.py`

### Modificar
- `deep3.py` (ou mover para `app/` e adaptar):
  - substituir plots diretos por `PlotPanel`
  - conectar callbacks do interator → status bar e measure panel
  - usar atomic write nas exportações
  - remover duplicações de função (ex.: write_prn)

---

## 11) Checklist final de UX (o que “parece” profissional)
- Menus e atalhos “padrão Windows”
- Tooltip/cursor readout no gráfico
- Marcadores com A/B e delta
- Context menu no gráfico
- Status bar com progresso e mensagens curtas
- Logs acessíveis via View > Log
- Preferências persistentes (tema, norm mode default, VF default)
- Autosave e recuperação
- Export wizard com manifesto

---

## 12) Nota sobre o null fill (integração com UI)
A síntese já suporta `amplitude`, `phase`, `both` na aba vertical. fileciteturn6file0L49-L55  
A suíte profissional deve:
- expor isso também no menu Tools > Null Fill Wizard,
- permitir o usuário escolher “amplitude/fase/ambos” explicitamente,
- e sempre exportar harness (CSV/JSON + Δcomprimento) quando modo inclui fase.

---

### Resultado esperado
Ao final, o usuário opera como em softwares RF “de verdade”:
- abre arquivos com Ctrl+1/Ctrl+2,
- clica no gráfico para medir e entender rapidamente,
- exporta com wizard e manifesto,
- valida PRN/PAT sem medo,
- e nunca perde dados (autosave + atomic write + logs).
