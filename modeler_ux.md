# modeler_ux.md — 3D Modeler (Análises Mecânicas) Intuitivo, Customizável e Avançado (PySide6 + PyVistaQt)
Versão: 1.0 • UI: PySide6 • Viewport: PyVista + pyvistaqt (QtInteractor) • Engine: mech/engine (agnóstico de UI)

> **Objetivo:** especificar, em nível de UX + arquitetura + execução, como construir um **3D Modeler** avançado e funcional dentro da aba **Análises Mecânicas**, com operações executáveis por **Tabs** (menu de abas) e também por **botão direito** (menu completo em qualquer contexto).

> **Gate obrigatório:** a aba Análises Mecânicas só deve ser implementada após migração completa para PySide6 com paridade total das features existentes.

---

## 1) Conceito: “Modeler como sub-suite”
O 3D Modeler deve operar como um mini-CAD técnico, com foco em:
- criação e edição de geometria (primitivas + transform + boolean)
- análise (medidas, seções, markers, relatórios)
- uso intuitivo (hints + micro-wizards sem modal)
- customização total (perfil, layout, grid, snaps, atalhos)
- estabilidade/robustez (undo/redo e validação de malha)

**Regra de arquitetura:** engine e UI desacoplados (engine não importa Qt).

---

## 2) Layout CAD-like (dockable) e customizável
### 2.1 Regiões fixas
- **Top Bar** (2 linhas):
  - Linha 1: File/Project, Undo/Redo, Profile, Units, Snap toggles, Preferences
  - Linha 2: **Menu de Abas** (Tabs) para executar operações
- **Left Dock**: Scene (Scene Tree + Search + Layers/Groups)
- **Center**: Viewport 3D (QtInteractor)
- **Right Dock**: Inspector (Properties + History + Measurements + Markers)
- **Bottom**: Status/Hint bar + Console local

### 2.2 Docks e layouts (customização)
- Todos os docks devem ser rearranjáveis (Qt docking).
- O sistema deve salvar/carregar layouts:
  - `layout_default.json`
  - `layout_modeling.json`
  - `layout_analysis.json`
- Layout inclui: posição dos docks, visibilidade, tamanhos, última tab ativa.

**API recomendada (UI):**
- `save_layout(name)` → JSON
- `load_layout(name)`
- `reset_layout_default()`

---

## 3) Menu de Abas (Tabs) para execução das operações
As operações principais devem existir como **tabs** (sub-aba dentro de Análises Mecânicas).  
Cada tab possui widgets próprios e “micro-wizards” inline quando necessário.

### Tab 1 — Scene
**Objetivo:** organizar, selecionar, visibilidade e estrutura.
- Scene Tree (sempre visível no dock esquerdo)
- Botões rápidos:
  - Create Group / Ungroup
  - Rename
  - Duplicate
  - Delete
  - Hide / Show / Isolate / Show All
  - Focus (camera no selecionado)
- Search box (filtra por nome/tag)
- Layers (fase 2, opcional)

### Tab 2 — Create
**Objetivo:** criar geometria com preview.
- Primitive palette:
  - Box, Cylinder, Sphere, Cone, Plane (MVP)
- Parâmetros:
  - dimensões + posição inicial + orientação
  - opção “center on origin” / “center on selection”
- Preview ao vivo (wireframe ghost)
- Botões:
  - Create
  - Create & Duplicate
  - Cancel preview

### Tab 3 — Transform
**Objetivo:** mover/rotacionar/escalar com precisão.
- Tool mode:
  - Select, Move, Rotate, Scale
- Numeric transform:
  - Translate (x,y,z)
  - Rotate (rx,ry,rz) (graus)
  - Scale (sx,sy,sz)
  - Apply / Reset
- Snap settings:
  - Snap to grid (toggle)
  - Grid step (mm)
  - Angle step (°)
- “Bake transform” (avançado, opcional)

### Tab 4 — Boolean
**Objetivo:** união/subtração/interseção com validação.
- Seletor de alvos:
  - A = Primary (base) (primeiro selecionado)
  - B = Tool (segundo selecionado)
  - botão “Swap A/B”
- Operações:
  - Union
  - Subtract (A − B)
  - Intersect
- Opções:
  - Clean before boolean (on/off)
  - Triangulate (on/off)
  - Normals consistent (on/off)
  - Tolerance (float)
  - Keep originals (on/off)
- Diagnóstico:
  - “Diagnose” mostra: manifold, buracos, tri degeneradas, normals

### Tab 5 — Analyze
**Objetivo:** medidas e seções.
- Compute:
  - Bounding Box
  - Volume
  - Area
  - Centroid
- Measure Tools:
  - Distance (2 pontos)
  - Angle (3 pontos)
  - Clear current measure
  - Save measure (marker)
- Sections:
  - Add clipping plane XY/XZ/YZ
  - Remove clipping planes
  - Slice preview (fase 2)

### Tab 6 — Markers
**Objetivo:** anotações e funções matemáticas.
- Add marker:
  - at pick point
  - at centroid
  - free marker
- Math marker:
  - editor seguro de expressões
  - variáveis disponíveis: area, volume, bbox_dx/dy/dz, centroid_x/y/z, etc.
- Style:
  - color, size, label
- Export markers (JSON/CSV)

### Tab 7 — Export
**Objetivo:** exportar geometria e snapshots.
- Export selection / scene:
  - STL, OBJ, PLY (MVP: STL)
- Screenshot PNG:
  - DPI, background, edges on/off
- Export report (fase 2):
  - PDF com imagens + tabela de medidas + metadata

> **Regra:** tudo que existe nas Tabs também deve existir via **botão direito** (menu contextual). Tabs são “guia”, RMB é “poder completo”.

---

## 4) Botão direito (RMB): “tudo por ali” em qualquer tela
### 4.1 Context Menu Dispatcher (obrigatório)
Criar um dispatcher único (`mech/ui/context_menu.py`) que:
1) detecta o contexto (viewport/tree/properties/measurements)
2) identifica alvo (picked object, selection, empty)
3) monta QMenu com ações habilitadas/desabilitadas
4) executa ações via CommandStack (undo/redo)
5) atualiza UI por eventos

### 4.2 Contextos mínimos
- Viewport vazio
- Viewport sobre objeto
- Multi-selection
- Scene tree item
- Scene tree vazio
- Properties panel
- Measurements panel
- Tool mode ativo (Measure/Move/Rotate/Scale)

### 4.3 Seções padrão do menu (consistência)
- Target (item desabilitado no topo, ex.: “Target: ObjName”)
- View / Tools / Create / Transform / Edit / Boolean / Analyze / Markers / Export / Undo-Redo / Danger Zone

---

## 5) Sistema intuitivo: hints + micro-wizards inline (sem modal)
### 5.1 Hint bar (obrigatório)
A barra inferior sempre descreve “o que fazer agora”:
- Select mode: “Clique para selecionar. Shift adiciona. RMB abre ações.”
- Boolean tab: “Selecione 2 objetos: A (base) e B (tool).”
- Measure mode: “Clique 2 pontos para distância. Enter finaliza. Esc cancela.”

### 5.2 Micro-wizard inline (obrigatório para operações multi-step)
Operações como:
- Boolean (se quiser escolher A/B com UI)
- Array/Mirror (quando implementado)
- Measure Distance/Angle

Devem usar um painel inline no Inspector:
- passo 1/2/3
- Next/Back/Cancel
- sem bloquear o viewport

---

## 6) Customização avançada (perfis, atalhos, grid, render)
### 6.1 Profiles (obrigatório)
Criar perfis salvos:
- CAD Classic
- Minimal
- Analysis

Cada perfil define:
- layout default dos docks
- grid/snap defaults
- render defaults (wireframe/edges/background)
- atalhos

### 6.2 Preferences (schema JSON)
Arquivo sugerido: `project/mech/preferences.json` (ou global em AppData)

Exemplo:
```json
{
  "units": "mm",
  "grid": {"enabled": true, "step_mm": 5.0, "size_mm": 2000.0, "color": "#2b2b2b"},
  "snap": {"grid": true, "step_mm": 5.0, "angle": true, "step_deg": 5.0},
  "render": {"mode": "solid_edges", "background": "dark", "aa": 4},
  "shortcuts": {"focus": "F", "undo": "Ctrl+Z", "redo": "Ctrl+Y", "delete": "Del"}
}
```

### 6.3 Layout persistence (schema JSON)
Arquivo: `project/mech/layouts/layout_default.json` etc.
- salvar estado dos docks (Qt)
- aba ativa
- preferências de viewport (camera preset, clipping enabled)

---

## 7) Engine: estabilidade e funcionalidade real (não “demo”)
### 7.1 Command system (undo/redo obrigatório)
Toda operação que altera a cena deve ser Command:
- CreatePrimitiveCmd
- TransformCmd
- DeleteCmd
- DuplicateCmd
- BooleanCmd
- AddMarkerCmd
- SaveMeasurementCmd

Undo/Redo deve funcionar para:
- criação
- transform
- boolean
- delete
- markers
- medidas salvas

### 7.2 Seleção consistente
Implementar camadas de seleção:
1) objeto (MVP)
2) face (fase 2)
3) aresta/vértice (fase 3)

A UI deve manter seleção mesmo quando o picking falha.

### 7.3 Robustez de boolean
Antes do boolean:
- clean
- triangulate
- compute_normals (consistent)

Se falhar:
- manter originais
- log detalhado
- exibir hint no status
- permitir tentar com tolerância diferente

---

## 8) Sequência de implementação (para garantir “avançado e funcional”)
### Sprint 1 — Base e interação
1) Aba + docks + viewport PyVistaQt
2) SceneEngine + SceneTree + Inspector
3) Seleção por objeto + highlight
4) RMB menu “Viewport vazio” e “Viewport objeto” (mínimo)

**Aceite:** criar cena, selecionar, menu RMB sempre abre.

### Sprint 2 — Criação e transform
1) Primitivas (Create tab)
2) Transform (numérico) + snap básico
3) Undo/Redo para create/transform
4) RMB contendo Create/Transform completo

**Aceite:** usuário cria e posiciona objetos facilmente e desfaz tudo.

### Sprint 3 — Medidas e markers
1) Measures (bbox/area/volume/centroid)
2) Distance/Angle tool mode (Analyze tab)
3) Markers + persistência
4) RMB para medidas/markers em todas telas

**Aceite:** medidas confiáveis e salvas no projeto.

### Sprint 4 — Boolean + diagnóstico
1) Union/Subtract/Intersect (Boolean tab)
2) Diagnóstico de falhas
3) Undo/Redo boolean
4) RMB boolean com validação e tooltips

**Aceite:** booleans funcionam com feedback claro.

### Sprint 5 — Customização e polimento
1) Preferences + profiles
2) Layout save/load
3) performance (throttle, caching)
4) Export STL + screenshot

**Aceite:** sistema “suite” com personalização real e UX madura.

---

## 9) Critérios de aceite finais
- O usuário consegue, sem manual:
  - criar primitiva
  - selecionar
  - mover/rotacionar
  - medir
  - executar boolean
  - exportar STL
- Botão direito funciona em TODAS as áreas e contém TODAS as operações.
- Undo/Redo confiável.
- Preferências e layout persistem.
- Sem travar UI em operações pesadas (workers + progress/log).
- Logs claros e diagnósticos de boolean.

---

## 10) Anexos: lista curta de ações obrigatórias no RMB (checklist)
- Viewport vazio: Create / View / Tools / Undo-Redo
- Viewport objeto: Select / Transform / Edit / Boolean / Analyze / Markers / Export / Delete / Undo-Redo
- Scene Tree item: Select/Focus / Rename / Show-Hide / Duplicate / Export / Delete
- Properties: Copy/Paste/Reset transform / Rename / Lock / Delete
- Measurements: Copy / Save as marker / Export / Delete measurement
