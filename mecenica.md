# mecenica.md — Aba “Análises Mecânicas” (3D Modeler) com Menu Completo no Botão Direito (PySide6)
Versão: 1.0 • Alvo: EFTX Diagram Suite • UI: PySide6 • Viewport 3D: PyVista + pyvistaqt (QtInteractor)

> **Gate obrigatório:** implementar esta aba **somente após** a migração completa para **PySide6** com paridade total das funcionalidades existentes.  
> **Regra obrigatória (este documento):** **todas** as operações do modeler e de análise devem estar disponíveis via **botão direito do mouse** (context menu) em **todas as telas** desta aba (Viewport 3D, Scene Tree, Properties, Tabela/Lista de medidas, etc.).

---

## 1) Objetivo e Escopo
Implementar uma aba de **Análises Mecânicas** que funcione como um **3D Modeler técnico leve**, com:
- Visualização 3D avançada (orbit/pan/zoom, grid, eixos, wireframe/solid, clipping).
- Seleção e manipulação de objetos (transformações).
- Criação de primitivas (box/cylinder/sphere/cone/plane no MVP).
- Operações booleanas (union/subtract/intersect) com robustez.
- Medições (bbox/área/volume/centroide/distâncias/ângulos).
- Sistema de comandos com **Undo/Redo obrigatório**.
- **Menus de contexto (botão direito)**: **todas** as operações acessíveis por ali, de forma **context-aware**.

**Fora de escopo (por enquanto):**
- CAD B-Rep completo (STEP/IGES nativo).
- Meshing avançado e constraints paramétricas de alto nível.
- Solver estrutural completo (isso pode vir depois).

---

## 2) Stack e Dependências (obrigatório)
- **PySide6**
- **pyvista**
- **pyvistaqt** (QtInteractor)
- **vtk** (via pyvista)
- (Opcional) **numpy** para transformações e medidas auxiliares

Instalação (Windows):
```bash
pip install pyside6 pyvista pyvistaqt vtk numpy
```

---

## 3) Arquitetura (não misturar UI e Engine)
### 3.1 Módulos
- `mech/engine/` — núcleo agnóstico de UI:
  - `scene_engine.py` (cena, seleção, add/remove, eventos)
  - `scene_object.py` (objeto, mesh, transform, meta)
  - `commands.py` (Command pattern + undo/redo)
  - `geometry_ops.py` (primitivas, boolean, clean/triangulate/normals)
  - `measures.py` (bbox, area, volume, centroid, dist/angle)
  - `validators.py` (checagens/manifold/degenerado e diagnóstico)
- `mech/ui/` — widgets PySide6:
  - `page_mechanics.py` (aba principal)
  - `viewport_pyvista.py` (viewport 3D + picking + overlays)
  - `scene_tree.py` (árvore de objetos)
  - `properties_panel.py` (transform + info)
  - `measurements_panel.py` (tabela/lista de medições)
  - `context_menu.py` (**dispatcher** de menus do botão direito)

### 3.2 Regras
- Engine **não** importa PySide6.
- UI **não** implementa boolean/medidas; apenas chama o engine via comandos.
- Toda alteração de cena = **Command** (para undo/redo), inclusive as ações disparadas pelo botão direito.

---

## 4) Layout da Aba (UI CAD-like)
- **Centro:** Viewport 3D (QtInteractor)
- **Esquerda:** Scene Tree (lista de objetos/grupos)
- **Direita:** Properties Panel (transform, meta, medidas rápidas)
- **Inferior (opcional):** Measurements (tabela/lista) + Log local da aba

Além disso:
- Toolbar (topo) pode existir, mas **não substitui** o botão direito. O botão direito tem **tudo**.

---

## 5) Modelo de Interação (Mouse e Atalhos)
### 5.1 Viewport 3D (padrão)
- Botão esquerdo: **selecionar** (objeto) / adicionar à seleção com Shift
- Botão do meio (wheel press): **pan**
- Scroll: **zoom**
- Botão direito: **menu de contexto** (sempre)
- Teclas:
  - `Del`: deletar selecionado (com confirmação opcional)
  - `Ctrl+Z / Ctrl+Y`: undo / redo
  - `F`: focar no selecionado
  - `Esc`: sair de tool mode (voltar para Select)

### 5.2 Scene Tree
- Clique esquerdo: selecionar item (sincroniza com viewport)
- Botão direito: menu com operações (rename, visibility, lock, etc.)

### 5.3 Properties / Measurements
- Botão direito: menu com ações do objeto/medição (copy values, export, create marker, etc.)

---

## 6) Requisito-chave: Botão Direito em TODAS as telas
### 6.1 Estratégia obrigatória
Implementar um **Context Menu Dispatcher** (roteador) que:
1) Identifica o **contexto** (qual widget, estado da seleção, onde o mouse clicou, se há “picked object/face”).
2) Monta o menu correto (QMenu + QAction) com ações coerentes.
3) Encaminha a ação para o **Command System** (undo/redo) chamando o engine.
4) Atualiza UI (viewport/tree/properties) por eventos do engine.

> **Regra:** nenhum widget deve “ignorar” clique direito. Se não houver ação aplicável, o menu deve ao menos conter:
- “Help (context) / Atalhos”
- “Reset View” (viewport) ou “No actions available” com itens desabilitados

### 6.2 Implementação PySide6 (padrões)
Para widgets Qt nativos:
- `setContextMenuPolicy(Qt.CustomContextMenu)` + `customContextMenuRequested.connect(...)`  
OU sobrescrever `contextMenuEvent(self, event)`.

Para Viewport PyVistaQt (QtInteractor):
- Interceptar eventos do interactor e disparar menu Qt:
  - capturar botão direito (`RightButtonPressEvent`) no VTK interactor
  - mapear coordenadas para `QPoint` no widget
  - chamar dispatcher para construir `QMenu` e `exec(globalPos)`

Para Matplotlib/Canvas (se existir na aba mecânica):
- Conectar ao evento `button_press_event` e no botão 3 abrir menu Qt com base no elemento clicado (marker/line/empty).

---

## 7) Definição de “Contextos” do Botão Direito (obrigatório)
O menu deve variar conforme:
1) **Viewport vazio** (nenhum objeto sob o cursor)
2) **Viewport sobre objeto** (picked object)
3) **Viewport com múltiplos selecionados**
4) **Scene Tree sobre item**
5) **Properties (objeto selecionado)**
6) **Measurements (sobre uma medição / marker)**
7) **Modo de ferramenta ativo** (Measure/Move/Rotate/Scale) — menu deve ter “Exit tool mode”

Abaixo estão os menus “completos” por contexto.

---

## 8) Menu Completo — Viewport (vazio)
**Quando:** clique direito no viewport sem objeto sob o cursor.

### 8.1 Seção “View”
- Reset View
- Fit All
- Toggle Grid
- Toggle Axes
- Render Mode:
  - Solid
  - Wireframe
  - Solid + Edges
- Background:
  - Dark / Light (ou abrir dialog de cores)
- Screenshot… (salvar PNG)

### 8.2 Seção “Create”
- Add Primitive:
  - Box…
  - Cylinder…
  - Sphere…
  - Cone…
  - Plane…
- Create Marker… (marker livre no espaço, ex.: ponto 3D/label)

### 8.3 Seção “Clipping / Sections”
- Add Clipping Plane (XY / XZ / YZ)
- Remove All Clipping Planes
- Toggle Clipping (enable/disable)

### 8.4 Seção “Selection”
- Select All
- Select None
- Invert Selection

### 8.5 Seção “Tools”
- Enter Tool Mode:
  - Select
  - Move
  - Rotate
  - Scale
  - Measure
- Preferences… (unidades, snap, incrementos)

### 8.6 Seção “Undo/Redo”
- Undo
- Redo

---

## 9) Menu Completo — Viewport (sobre objeto)
**Quando:** clique direito com objeto sob o cursor (picked) — mesmo que não esteja selecionado, o menu deve usar esse objeto como “target” e, se necessário, selecionar antes de operar.

### 9.1 Seção “Selection”
- Select This
- Add to Selection
- Remove from Selection
- Isolate (hide others)
- Show Only Selected
- Show All

### 9.2 Seção “Transform”
- Move…
- Rotate…
- Scale…
- Align:
  - Align to World Axes
  - Align to Plane… (se existir)
- Snap:
  - Snap to Grid (toggle)
  - Set Snap Step…

### 9.3 Seção “Edit”
- Duplicate
- Mirror… (X/Y/Z)
- Array… (linear/circular)
- Rename…
- Set Color…
- Set Opacity…
- Lock/Unlock

### 9.4 Seção “Boolean”
> Habilitar somente se houver seleção válida (ex.: 2 objetos) — se não, deixar desabilitado e mostrar hint no status.
- Union (A ∪ B)
- Subtract (A − B)
- Intersect (A ∩ B)
- Boolean Settings… (tolerância/clean/triangulate)

### 9.5 Seção “Measure / Analyze”
- Show Bounding Box
- Compute Volume
- Compute Area
- Compute Centroid
- Measure Distance… (entrar modo distância)
- Measure Angle… (entrar modo ângulo)

### 9.6 Seção “Markers”
- Add Marker at Pick Point
- Add Label (name/value)
- Add Custom Math Marker… (ver seção 13)

### 9.7 Seção “Export”
- Export Selected as STL…
- Export Selected as OBJ/PLY… (se suportado)

### 9.8 Seção “Danger Zone”
- Delete Selected
- Reset Transform (identity)

### 9.9 Undo/Redo
- Undo
- Redo

---

## 10) Menu Completo — Multi-selection (viewport ou tree)
**Quando:** existem 2+ selecionados.

- Group… / Ungroup… (se houver grupos)
- Align:
  - Align centers X/Y/Z
  - Distribute equally (X/Y/Z)
- Boolean:
  - Union (all)
  - Subtract (A minus others) — exigir “Primary” (primeiro selecionado)
  - Intersect (all)
- Transform:
  - Move… / Rotate… / Scale… (aplicar em conjunto)
- Export:
  - Export Selection…
- Visibility:
  - Hide Selected / Show Selected
- Delete Selected
- Undo/Redo

> **Regra:** se boolean multi for arriscado, o agente deve implementar pelo menos “Union all” e “Subtract primary minus others” com validação.

---

## 11) Menu Completo — Scene Tree (sobre item)
**Quando:** clique direito em um item na árvore.

- Select / Focus (camera)
- Rename…
- Visibility:
  - Show / Hide
  - Isolate
- Lock/Unlock
- Duplicate
- Delete
- Color/Opacity…
- Export…
- Measurements:
  - BBox / Area / Volume / Centroid
- Add Marker:
  - Marker no centroide
  - Label com nome
- Undo/Redo

> **Regra:** a árvore deve sempre abrir menu; clique em área vazia da árvore abre menu “Tree empty” com “Create primitive”, “Import mesh”, “Show all”.

---

## 12) Menu Completo — Properties Panel
**Quando:** clique direito nos campos/propriedades do objeto selecionado.

- Copy Value (do campo)
- Copy All Properties (JSON)
- Paste Transform (se compatível)
- Reset Transform
- Apply Transform to Mesh (bake transform) — opcional / avançado
- Rename…
- Lock/Unlock
- Delete
- Undo/Redo

---

## 13) Markers com Funções Matemáticas Personalizáveis (botão direito obrigatório)
### 13.1 Conceito
“Marker” é um objeto de anotação (ponto/linha/label) associado a:
- uma posição 3D (pick point, centroide, etc.)
- um conjunto de **expressões matemáticas** que calculam valores a partir de dados do objeto/medição

Exemplos de expressões:
- `vol = volume_mm3 / 1e9`
- `mass = density * volume_m3`
- `stress_proxy = force / area_m2`
- `dB = 10*log10(x)`

### 13.2 Política de segurança/robustez (obrigatória)
- Não executar `eval` livre.
- Implementar um mini-parser seguro:
  - permitir apenas operadores básicos `+ - * / **`
  - funções whitelisted (`sqrt`, `log10`, `sin`, `cos`, `tan`, `abs`, `min`, `max`)
  - variáveis disponíveis: `area`, `volume`, `bbox_dx`, `bbox_dy`, `bbox_dz`, `centroid_x/y/z`, etc.
- Se expressão falhar: mostrar erro no log e manter marker com status “invalid”.

### 13.3 UI do marker (mínimo)
- Clique direito em marker no viewport:
  - Edit Expression…
  - Toggle Visibility
  - Change Style (color/size)
  - Delete Marker
  - Copy computed values

### 13.4 Persistência
- Marker deve ser salvo no projeto (ex.: `project/mech/markers.json`):
  - id, name, position, expressions, last_values, style, target_object_id (se houver)

---

## 14) Implementação do Dispatcher de Context Menu (obrigatório)
### 14.1 API sugerida
`mech/ui/context_menu.py`:

- `build_menu(context: ContextInfo) -> QMenu`
- `ContextInfo` inclui:
  - `widget` (viewport/tree/properties/measurements)
  - `mouse_pos` (QPoint local)
  - `global_pos` (QPoint global)
  - `picked_object_id` (ou None)
  - `selected_ids` (list)
  - `tool_mode` (Select/Move/Rotate/Scale/Measure)
  - `picked_point_3d` (se houver)

### 14.2 Encaminhamento obrigatório para comandos
Cada QAction deve chamar um handler que:
1) Resolve alvo(s) (picked vs selected)
2) Cria e aplica um `Command` no `CommandStack`
3) Engine emite evento → UI atualiza viewport/tree/properties

> **Regra:** ações destrutivas (Delete, Boolean) devem pedir confirmação (dialog) OU ter “Undo” garantido e fácil.

---

## 15) Como capturar botão direito no PyVistaQt (orientação prática)
### 15.1 VTK observer
No `Viewport3D`:
- adicionar observer para `RightButtonPressEvent`.

Pseudo:
```python
# dentro do Viewport3D.__init__
iren = self.plotter.interactor  # QVTKRenderWindowInteractor
iren.AddObserver("RightButtonPressEvent", self._on_right_click_vtk)

def _on_right_click_vtk(self, obj, ev):
    # 1) obter posição do clique no render window
    x, y = iren.GetEventPosition()  # coords VTK
    # 2) converter para coords Qt (y invertido conforme widget)
    # 3) fazer picking (objeto sob cursor)
    # 4) montar ContextInfo e abrir menu:
    menu = dispatcher.build_menu(ctx)
    menu.exec(QCursor.pos())
```
**Obrigatório:** garantir que o clique direito não execute orbit/pan default do VTK. Bloquear o comportamento padrão se necessário.

### 15.2 Picking
- Usar `vtkCellPicker` ou API do pyvista para identificar o objeto/ator.
- Mapeamento `actor -> obj_id` deve existir.

---

## 16) Booleans e Robustez (obrigatório)
### 16.1 Pré-processamento
Antes de boolean:
- clean
- triangulate
- normals consistentes
- checar degenerados

### 16.2 Política de falha
Se boolean falhar:
- manter objetos originais
- emitir log detalhado
- mostrar toast/alert (não travar UI)

### 16.3 Undo/Redo
Boolean deve ser Command:
- `do`: remove A/B (opcional), cria C
- `undo`: restaura A/B e remove C

---

## 17) Medidas e Modo “Measure” (com botão direito)
### 17.1 Entrar em modo de medida via menu
- Clique direito (objeto) → Measure Distance…
- Clique direito (objeto) → Measure Angle…

### 17.2 Menu durante modo Measure
Clique direito deve oferecer:
- Exit Measure Mode
- Clear Current Measure
- Save Measure as Marker
- Copy last measure

---

## 18) Critérios de Aceite (obrigatório)
### 18.1 Reatividade e performance
- Viewport não pode congelar ao selecionar, mover, abrir menus ou recalcular medidas simples.
- Booleans e operações pesadas devem rodar em worker thread (quando necessário) com progress/log.

### 18.2 Botão direito “em todo lugar”
- Viewport: sempre abre menu
- Scene Tree: sempre abre menu
- Properties: sempre abre menu
- Measurements: sempre abre menu
- Em áreas vazias: abre menu “global” (Create/View/Preferences)

### 18.3 Paridade de ações
Tudo que estiver na toolbar/botões também deve estar no menu de contexto.

### 18.4 Undo/Redo
- Todas ações listadas (exceto “apenas visualizar”) devem ser undoable.

---

## 19) Ordem recomendada de implementação (para o agente)
1) Criar a aba e layout (viewport/tree/properties) — sem engine completo ainda
2) Implementar `SceneEngine` + `CommandStack`
3) Implementar seleção e sincronização (tree ↔ viewport)
4) Implementar Context Menu Dispatcher e menus “vazio” e “objeto”
5) Implementar primitivas + transform numérico
6) Implementar medições básicas
7) Implementar booleans com validação
8) Implementar markers + math expressions seguras
9) Endurecer (tests manuais, logs, erros)

---

## 20) Notas finais (padrão profissional)
- Menus devem ser **coerentes**, com seções e separadores consistentes.
- Itens desabilitados devem ter tooltips explicando “por que” estão desabilitados.
- O menu deve mostrar o **alvo** (ex.: “Target: ObjName (id…)”) no topo (desabilitado).
- Sempre manter o usuário no controle: ações destrutivas com Undo fácil e/ou confirmação.
