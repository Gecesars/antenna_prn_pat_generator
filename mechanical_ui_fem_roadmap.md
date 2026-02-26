# mechanical_ui_fem_roadmap.md
## EFTX — Roadmap de Evolução da Interface de Análises Mecânicas (3D Modeler + FEM)

> Documento de instrução para agente de implementação.  
> Objetivo: evoluir a interface atual para uma plataforma de modelagem 3D + preparação e execução de análises FEM com UX clara, seleção visual robusta e arquitetura escalável.

---

## 1. Objetivo Geral

Evoluir a interface atual para suportar de forma clara e produtiva:

- **Modelagem 3D** (criação, edição, operações booleanas, organização)
- **Seleção visual inequívoca** (troca de cor/realce + contorno + sincronização com Scene Tree)
- **Biblioteca de componentes** (primitivas, importados, assemblies, instâncias)
- **Preparação FEM completa** (materiais, contatos, cargas, restrições, malha, solver)
- **Fluxo guiado por modos de trabalho**
- **Pós-processamento e diagnóstico**

### Requisito central (prioridade máxima)
Ao selecionar um objeto no viewport ou na árvore:
- o objeto deve **mudar de aparência visual** (highlight claro),
- a seleção deve sincronizar com a árvore,
- o painel de propriedades deve atualizar,
- e a barra de status/log deve indicar o item selecionado.

---

## 2. Diagnóstico da Interface Atual (com base na tela enviada)

### Pontos positivos
- Estrutura geral já existe: **Scene Tree**, **Viewport 3D**, **Properties**, **Log**
- Abas temáticas já iniciadas (Scene, Create, Transform, Boolean, Analyze...)
- Integração com backend/runtime já visível
- Grid/layers/layout já existem

### Problemas observados
1. **Toolbar superior excessivamente densa**
   - muitos botões pequenos
   - abreviações pouco claras
   - baixa descobribilidade

2. **Hierarquia visual fraca**
   - ações de cena, modelagem e FEM misturadas
   - difícil entender “o que fazer agora”

3. **Seleção visual insuficiente**
   - não fica inequívoco qual objeto está selecionado
   - falta highlight padronizado + outline

4. **Painel direito mistura contextos**
   - propriedades + transform + wizard + log sem separação clara

5. **Ausência de modos explícitos de seleção**
   - objeto / face / aresta / vértice / corpo / componente

6. **Aba de componentes pouco estruturada**
   - falta biblioteca reutilizável
   - falta noção clara de instância/assembly/componente

---

## 3. Meta de UX (resultado esperado)

O usuário deve conseguir, sem treinamento extenso:

1. **Criar geometria**
2. **Selecionar com clareza**
3. **Transformar/editar**
4. **Aplicar materiais**
5. **Definir BCs/cargas**
6. **Gerar malha**
7. **Rodar FEM**
8. **Visualizar resultados**
9. **Exportar**

---

## 4. Layout Proposto (macroestrutura)

Manter a estrutura base, porém reorganizar em 5 regiões principais:

### 4.1 Barra superior (Ribbon contextual)
Substituir a barra densa por grupos nomeados (ícone + texto + tooltip).

### 4.2 Painel esquerdo com abas
- `Scene`
- `Components`
- `Layers`
- `Studies` (opcional)

### 4.3 Viewport 3D central
- gizmo de transformação
- highlight de seleção
- view cube
- barra rápida de câmera
- overlays (grid/eixos/section)

### 4.4 Painel direito contextual
Subabas conforme contexto:
- `Properties`
- `Transform`
- `Material`
- `FEM`
- `BCs`
- `Loads`
- `Mesh`
- `Solve`
- `Results`

### 4.5 Dock inferior
- `Log`
- `Measurements`
- `Selection Info`
- `Diagnostics`
- `Solver Console`

---

## 5. Reestruturação da Toolbar / Ribbon (com nomenclatura clara)

### 5.1 Grupos na barra superior

#### A) Projeto
- New
- Open
- Save
- Save Layout
- Load Layout
- Preferences
- Units

#### B) Edit
- Undo
- Redo
- Duplicate
- Rename
- Delete
- Group / Ungroup

#### C) View
- Orbit / Pan / Zoom
- Fit All / Fit Selection
- Top / Front / Right / Iso
- Wireframe / Shaded / X-Ray / Hidden Line
- Grid toggle
- Axes toggle
- Section plane
- Exploded view (assemblies)

#### D) Selection
- Mode: Object / Face / Edge / Vertex / Body / Component
- Box select
- Lasso select
- Brush select
- Filter select (type/layer/material/tag)
- Isolate
- Hide / Show all
- Invert selection

#### E) Modeling
- Primitives
- Sketch
- Extrude / Revolve / Sweep / Loft
- Boolean (Union / Subtract / Intersect)
- Fillet / Chamfer / Shell
- Mirror / Array / Pattern
- Align / Snap

#### F) FEM
- New Study
- Materials
- Contacts
- BCs
- Loads
- Mesh
- Solver
- Run
- Results

#### G) Import / Export
- Import STEP/IGES/STL/OBJ/DXF
- Export STEP/STL/OBJ/Mesh
- Export report
- Export image/screenshot

### 5.2 Regra de UX
- Remover abreviações pouco claras (ex.: “Retessell” → “Re-tessellate Mesh”)
- Sempre usar tooltip técnico + atalho
- Ações principais com ícone + texto (não apenas ícone)

---

## 6. Seleção Visual Clara (REQUISITO OBRIGATÓRIO)

## 6.1 Estados visuais de seleção

### Hover
- Outline fino (ex.: ciano)
- Sem alterar material/cor permanente

### Selecionado (single)
- Overlay de cor de seleção (ex.: amarelo/ciano forte)
- Outline mais espesso
- Gizmo no pivot do objeto
- Item destacado na árvore

### Multi-seleção
- Todos com outline
- Item ativo com highlight primário
- Demais com highlight secundário

### Bloqueado
- estilo escurecido / ícone cadeado
- não selecionável no clique comum (somente override)

### Oculto
- não renderiza no viewport
- item “apagado” na árvore

---

## 6.2 Requisitos técnicos (render + picking)

### A) Não sobrescrever cor/material original
A seleção deve ser uma **camada de visualização (overlay)**.

### B) Picking robusto por ID
Implementar picking com **ID buffer** (color picking / object id pass):
- objeto
- face
- aresta
- vértice

### C) Outline pass
Implementar pós-processamento de contorno baseado em:
- diferença de IDs
- depth/normal edges (quando aplicável)

### D) Sincronização total
Evento de seleção precisa refletir em:
- viewport
- scene tree
- painel properties
- barra de status
- painel de medições (se aplicável)

---

## 7. SelectionManager — Especificação (módulo central)

## 7.1 Responsabilidades
- controlar seleção atual
- hover atual
- multi-seleção
- modo de seleção (obj/face/edge/vertex/body/component)
- filtros de seleção
- emitir eventos de mudança

## 7.2 Estrutura sugerida (pseudocódigo)

```python
class SelectionMode(Enum):
    OBJECT = "object"
    FACE = "face"
    EDGE = "edge"
    VERTEX = "vertex"
    BODY = "body"
    COMPONENT = "component"

@dataclass
class SelectionItem:
    entity_id: str
    entity_type: str          # object, face, edge, vertex...
    parent_object_id: str | None
    sub_index: int | None     # face idx, edge idx...
    display_name: str
    layer: str | None = None
    locked: bool = False

class SelectionManager(QObject):
    selection_changed = Signal(list)   # list[SelectionItem]
    hover_changed = Signal(object)     # SelectionItem | None
    active_item_changed = Signal(object)

    def __init__(self):
        self.mode = SelectionMode.OBJECT
        self._selected: list[SelectionItem] = []
        self._hover: SelectionItem | None = None
        self._active: SelectionItem | None = None
        self._filters = {
            "allow_hidden": False,
            "allow_locked": False,
            "layers": None,
            "types": None,
        }

    def set_mode(self, mode: SelectionMode): ...
    def set_hover(self, item: SelectionItem | None): ...
    def clear(self): ...
    def set_single(self, item: SelectionItem): ...
    def add(self, item: SelectionItem): ...
    def toggle(self, item: SelectionItem): ...
    def remove(self, item: SelectionItem): ...
    def set_from_tree(self, items: list[SelectionItem]): ...
    def set_from_viewport_pick(self, pick_result): ...
    def box_select(self, frustum, op="replace"): ...
    def lasso_select(self, poly2d, op="replace"): ...
```

---

## 7.3 Regras de comportamento
- **Clique simples**: replace selection
- **Shift+Clique**: add
- **Ctrl+Clique**: toggle
- **Alt+Clique** (opcional): remove/subtract
- **Double click**: select connected / focus (configurável)
- **Esc**: clear selection

---

## 8. Viewport 3D — Evolução de interação

## 8.1 Controles visíveis no viewport (quick toolbar)
Adicionar barra flutuante com:
- Iso / Top / Front / Right
- Fit All
- Fit Selection
- Grid on/off
- Axes on/off
- Wireframe/Shaded/X-Ray
- Section plane
- Screenshot

## 8.2 Gizmo de transformação
Suportar:
- Move (X/Y/Z + planar)
- Rotate (X/Y/Z)
- Scale (uniforme e não uniforme)
- Pivot mode (object center / world / custom)
- Entrada numérica rápida (dx,dy,dz / rx,ry,rz / sx,sy,sz)

## 8.3 Snap
Toggles claros:
- Grid
- Vertex
- Edge midpoint
- Face center
- Axis
- Angle increment (ex.: 5°, 15°)

---

## 9. Scene Tree — Reestruturação

## 9.1 Funções essenciais
- hierarquia (groups/components/assemblies)
- filtro por nome/tipo/layer/material/tag
- seleção sincronizada com viewport
- menus contextuais ricos

## 9.2 Colunas sugeridas
- Name
- Type
- Visible
- Locked
- Layer
- Material
- FEM Role
- Solve (include/exclude)

## 9.3 Menu de contexto (RMB)
- Rename
- Duplicate
- Delete
- Hide / Show
- Isolate
- Lock / Unlock
- Group / Ungroup
- Move to Layer
- Assign Material
- Convert to Component
- Include/Exclude in FEM Study
- Zoom to Selection
- Properties

---

## 10. Aba “Components” — Organização completa

## 10.1 Estrutura da aba
### A) Primitives
- Box
- Cylinder
- Cone
- Sphere
- Torus
- Plane
- Tube/Pipe
- Polygon
- Wedge (opcional)

### B) Sketch / Profiles
- Line
- Rectangle
- Circle
- Arc
- Polyline
- Spline

### C) Parametric Components (biblioteca)
- Bolt / Nut / Washer
- Flange
- Plate
- Tube profile
- Bracket
- Panel
- Mast / Tower segment
- Clamp
- Cable / Coax (geométrico + metadados)
- Splitter/Divider (placeholder mecânico + metadados)

### D) Imported
- STEP / IGES / STL / OBJ / DXF
- unidade detectada
- escala
- tesselação
- origem do arquivo
- histórico de importação

### E) Instances & Assemblies
- Create component from selection
- Insert instance
- Link/unlink instance
- Create assembly
- Explode assembly

### F) Project Templates (muito útil para o domínio do projeto)
- Painel + suporte
- Torre + travamentos
- Conjunto de cabos
- Sistema irradiantes simplificado (mecânico)
- Base + mast + painéis

---

## 10.2 Metadados por componente (padronização)
Cada item de componente deve carregar:
- `uuid`
- `name`
- `type`
- `source`
- `layer`
- `visible`
- `locked`
- `material`
- `color`
- `transform`
- `tags`
- `fem_role` (solid/shell/beam/ref)
- `exclude_from_solve`
- `notes` (opcional)

---

## 11. Funções de Modelagem 3D — Escopo funcional

## 11.1 Prioridade (MVP forte)

### Criação
- Box, Cylinder, Sphere, Cone, Plane, Tube
- Sketch 2D (line, rectangle, circle, arc)
- Extrude
- Revolve
- Sweep
- Loft

### Transformações
- Move
- Rotate
- Scale
- Mirror
- Align
- Snap

### Booleanas (obrigatórias)
- Union
- Subtract
- Intersect
- Split / Slice (fortemente recomendado)

### Edição direta
- Fillet
- Chamfer
- Shell
- Offset face
- Move face (push/pull) — fase 1.5 / 2

### Organização
- Group/Ungroup
- Layers
- Duplicate
- Array linear
- Array circular
- Hide/Show/Isolate

---

## 11.2 Avançado (fase 2+)
- Curvas 3D/splines
- Pattern por caminho
- Draft angle
- Thicken surface
- Surface patches
- Geometry healing / repair
- Feature history (árvore paramétrica)
- Dimensões paramétricas e expressões

---

## 12. Measurements / Markers (painel inferior)

Adicionar ferramentas completas de medição:
- Distância ponto-ponto
- Distância aresta-aresta
- Ângulo entre arestas/faces
- Área
- Volume
- Bounding box
- Centro de massa (se material definido)
- Momento de inércia (fase 2)
- Marcadores persistentes com nome
- Exportação de medições (CSV/JSON)

Requisitos:
- medições devem sobreviver à navegação
- opcionalmente atualizar após transformação geométrica
- poder “fixar” medição no modelo

---

## 13. FEM — Estrutura completa da interface

## 13.1 Workflow explícito (não espalhado)
A UI deve guiar o usuário por etapas:

1. Criar estudo
2. Selecionar corpos participantes
3. Definir materiais
4. Definir contatos
5. Definir BCs
6. Definir cargas
7. Configurar malha
8. Configurar solver
9. Rodar
10. Visualizar resultados

---

## 13.2 Wizard “New FEM Study” (recomendado)

### Etapa 1 — Study Type
- Structural Static
- Modal
- Harmonic (fase 2)
- Buckling (fase 2)
- Thermal (fase 2)
- Thermo-structural (fase 3)

### Etapa 2 — Bodies Participation
- incluir/excluir corpos
- papel FEM: solid/shell/beam/reference
- suppress in solve

### Etapa 3 — Materials
- biblioteca
- material customizado
- propriedades mecânicas:
  - densidade
  - E
  - ν
  - limite de escoamento (opcional)
- propriedades térmicas (se aplicável)

### Etapa 4 — Contacts
- bonded
- frictionless
- frictional
- no separation
- auto-detect + revisão manual

### Etapa 5 — Boundary Conditions
- fixed support
- displacement
- symmetry
- remote support
- pin/hinge (fase 2)

### Etapa 6 — Loads
- force
- pressure
- gravity
- moment/torque
- acceleration
- distributed load
- **wind load** (simplificado, importante para domínio da aplicação)

### Etapa 7 — Mesh
- global size
- local refinement (body/face/edge)
- growth rate
- curvature refinement
- quality controls
- preview / re-tessellate

### Etapa 8 — Solver
- direct/iterative (se suportado)
- tolerance
- max iterations
- threads/memory (se disponível)

### Etapa 9 — Solve
- progress bar
- warnings/errors
- cancel / safe stop

### Etapa 10 — Results
- displacement
- stress (von Mises)
- principal stresses
- strain
- reactions
- safety factor (fase 2)

---

## 13.3 Painel direito FEM (subabas)
Criar subabas e listas editáveis:
- `Study`
- `Materials`
- `Contacts`
- `BCs`
- `Loads`
- `Mesh`
- `Solve`
- `Results`

Cada aba deve permitir:
- adicionar/editar/excluir
- ativar/desativar item
- localizar item no modelo (zoom/focus)
- validação local (ex.: load sem face associada)

---

## 14. FEM Validation Engine (pré-solve)

Implementar checklist de validação antes de permitir o solve:

- [ ] Existe pelo menos 1 corpo ativo?
- [ ] Todos os corpos ativos possuem material?
- [ ] O sistema está adequadamente restringido?
- [ ] Existe pelo menos uma carga/condição aplicada?
- [ ] Contatos inválidos/ambíguos?
- [ ] Malha foi gerada com sucesso?
- [ ] Unidades coerentes?

### Semáforo de status
- ✅ OK
- ⚠ Warning
- ❌ Error (bloqueia solve)

Exibir no painel `Solve` e também no log.

---

## 15. Arquitetura interna recomendada (UI + núcleo)

## 15.1 Módulos centrais

### `SceneGraph`
Responsável por:
- nós/hierarquia
- transforms
- visibility/lock
- metadados
- layers
- assemblies

### `SelectionManager`
Responsável por:
- seleção e hover
- modos de seleção
- multi-seleção
- eventos de sincronização

### `ViewportRenderer`
Responsável por:
- render normal
- picking ID pass
- outline/highlight pass
- gizmos
- overlays (grid, axes, section)

### `CommandSystem` (Undo/Redo)
Usar padrão Command:
- `CreatePrimitiveCommand`
- `DeleteObjectCommand`
- `TransformCommand`
- `BooleanCommand`
- `AssignMaterialCommand`
- `AddLoadCommand`
- `SetBoundaryConditionCommand`
- etc.

### `FEMStudyManager`
Responsável por:
- estudos
- materiais
- contatos
- BCs
- cargas
- malha
- solver config
- resultados
- validação

### `BackendAdapter`
Camada de integração com backend (FreeCAD/AEDT/runtime/etc.)
- API comum para operações geométricas/FEM
- isolamento de dependências específicas
- tratamento de erro centralizado

---

## 15.2 Contrato de eventos (padronizar)
Definir eventos de aplicação:
- `selection_changed`
- `hover_changed`
- `scene_updated`
- `object_modified`
- `object_deleted`
- `material_assigned`
- `mesh_generated`
- `study_validation_changed`
- `solve_started`
- `solve_progress`
- `solve_finished`
- `results_updated`

---

## 16. Pseudocódigo — Fluxo de seleção (viewport ↔ árvore ↔ properties)

```python
def on_viewport_click(mouse_pos, modifiers):
    pick = viewport_renderer.pick(mouse_pos, mode=selection_manager.mode)
    item = selection_manager.set_from_viewport_pick(pick, modifiers=modifiers)
    if item:
        scene_tree.select_item(item.entity_id, sub=item.sub_index)
        properties_panel.bind_to_selection(selection_manager.current_selection())
        status_bar.show_selection(item)

def on_scene_tree_selection_changed(tree_items):
    items = map_tree_items_to_selection_items(tree_items)
    selection_manager.set_from_tree(items)
    viewport_renderer.set_selection_overlay(items)
    properties_panel.bind_to_selection(items)
    if len(items) == 1:
        status_bar.show_selection(items[0])

def on_selection_changed(items):
    viewport_renderer.set_selection_overlay(items)
    scene_tree.sync_selection(items)
    selection_info_panel.update(items)
```

---

## 17. Pseudocódigo — Highlight / Outline no renderer

```python
def render_frame():
    render_scene_normal_pass()

    if selection_manager.has_hover_or_selection():
        render_id_buffer_if_needed()

        # Highlight fill overlay
        for item in selection_manager.selected_items:
            render_highlight_overlay(item, mode="selected")

        if selection_manager.hover_item and selection_manager.hover_item not in selection_manager.selected_items:
            render_highlight_overlay(selection_manager.hover_item, mode="hover")

        # Outline pass (post-process based on id/depth)
        render_outline_postprocess(
            selected_ids=selection_manager.selected_ids(),
            hover_id=selection_manager.hover_id()
        )

    render_gizmos()
    render_overlays()
```

---

## 18. Padrões visuais / UX (legibilidade)

### 18.1 Clareza de nomenclatura
- evitar siglas não óbvias na UI
- usar termos consistentes (PT-BR ou EN, mas padronizados)

### 18.2 Hierarquia visual
- blocos com títulos
- espaçamento consistente
- estados visuais claros (normal/hover/active/disabled)

### 18.3 Feedback de ação
Toda ação relevante deve gerar:
- feedback curto (toast/status)
- log detalhado no painel `Log`

Exemplos:
- “3 bodies selected”
- “Boolean subtract completed”
- “Mesh regenerated (avg quality: 0.73)”
- “Study validation failed: missing fixed support”

---

## 19. Plano de Implementação por Fases (sprints)

## Fase 1 — UX + Seleção (PRIORIDADE MÁXIMA)
### Objetivo
Tornar a seleção inequívoca e a UI navegável.

### Entregas
- Reorganização da toolbar em grupos claros
- `SelectionManager` funcional
- highlight/outline no viewport
- sincronização viewport ↔ tree ↔ properties
- modos de seleção (obj/face/edge/vertex)

### Critério de sucesso
Usuário sempre sabe qual item está selecionado.

---

## Fase 2 — Aba Components + Modelagem essencial
### Entregas
- Aba `Components` estruturada
- Primitivas + importados + instâncias
- Transform + gizmo + snap
- Booleans (union/subtract/intersect)
- Medições básicas

### Critério de sucesso
Fluxo de modelagem 3D utilizável sem fricção.

---

## Fase 3 — FEM workflow guiado
### Entregas
- Wizard de estudo FEM
- Painéis `Materials / BCs / Loads / Mesh / Solve`
- Validação pré-solve
- Integração backend mesh/solve
- Logs e diagnostics

### Critério de sucesso
Usuário consegue configurar e executar análise sem “caça ao botão”.

---

## Fase 4 — Pós-processamento + refinamentos
### Entregas
- visualização de resultados (contours/probes)
- markers persistentes
- export de relatório
- templates de componentes/estudos
- otimizações de performance/estabilidade

---

## 20. Critérios de Aceitação (QA funcional)

### 20.1 Seleção
- [ ] Clique no viewport destaca visualmente o objeto selecionado
- [ ] Highlight não altera permanentemente o material
- [ ] Hover é visível e distinto de seleção
- [ ] Multi-seleção funciona (Shift/Ctrl)
- [ ] Seleção sincroniza com a Scene Tree
- [ ] Painel de propriedades atualiza corretamente

### 20.2 Usabilidade
- [ ] Toolbar organizada por grupos compreensíveis
- [ ] Tooltips descritivos em ações principais
- [ ] Modos de seleção visíveis e claros
- [ ] Barra de status informa seleção atual

### 20.3 Modelagem 3D
- [ ] Criação de primitivas funciona
- [ ] Transformações com gizmo e entrada numérica funcionam
- [ ] Boolean union/subtract/intersect funcionam
- [ ] Scene tree organiza groups/components/assemblies

### 20.4 FEM
- [ ] Wizard cria estudo com etapas claras
- [ ] Materiais/BCs/cargas/malha configuráveis
- [ ] Validação pré-solve aponta erros corretamente
- [ ] Solve exibe progresso/log
- [ ] Resultados básicos visualizáveis

---

## 21. Primeira tarefa imediata (ordem recomendada para o agente)

> Implementar nesta ordem para entregar valor rápido sem quebrar a base existente:

1. **SelectionManager + highlight/outline**
2. **Sincronização viewport/tree/properties**
3. **Refatorar toolbar em grupos nomeados**
4. **Criar aba Components (primitives/imported/instances)**
5. **Completar booleans + transform + snap**
6. **Estruturar FEM wizard + painéis + validação**

---

## 22. Observações de engenharia (compatibilidade / estabilidade)

- Preservar a lógica atual de runtime/backend e adaptar via `BackendAdapter`
- Evitar acoplamento forte UI ↔ backend geométrico/FEM
- Toda operação editável deve entrar no `CommandSystem` (undo/redo)
- Logs técnicos devem continuar completos (útil para depuração de integração)
- Se houver backends múltiplos, manter interface comum e feature flags

---

## 23. Entregáveis esperados do agente (por sprint)

### Sprint 1 (Seleção + UX base)
- Módulo `SelectionManager`
- Highlight/outline funcional
- Toolbar reorganizada
- Sync tree↔viewport
- Demonstração em cena de teste

### Sprint 2 (Modeling core)
- Aba Components
- Primitives + import
- Transform gizmo + snap
- Booleans
- Medições

### Sprint 3 (FEM setup)
- Wizard
- Painéis FEM
- Validação
- Mesh/solve hooks
- Logs/diagnostics

### Sprint 4 (Results)
- Results panel
- probes/markers
- export report
- refinamento visual/performance

---

## 24. Nota final (prioridade funcional)

O maior ganho imediato para o usuário será:

- **clareza de controle**
- **seleção visual inequívoca**
- **fluxo organizado**
- **FEM guiado**

Mesmo antes de implementar todas as features avançadas, a interface já parecerá profissional e muito mais produtiva se a **seleção + ribbon + components + workflow** forem resolvidos primeiro.
