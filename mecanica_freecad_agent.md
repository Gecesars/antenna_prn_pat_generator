# AFTX Suite Framework — Mecânica (FreeCAD/OCCT) — Instrução do Agente
**Objetivo:** implementar o módulo/aba de Mecânica (3D Modeler + base para análise mecânica) **mantendo a lógica atual do repositório**, integrando o kernel geométrico via **FreeCAD (OCCT)** com arquitetura por camadas (provider/adapter) e fallback seguro.

> Regra de ouro: **não reescrever o app atual**. O FreeCAD entra como **motor geométrico** (kernel), não como substituto do framework/UI.

---

## 0) Princípios obrigatórios de compatibilidade
1. **Preservar fluxo, estado, logging, testes e padrões atuais**: o módulo mecânico deve “plugar” no sistema existente.
2. **Feature-gating**: tudo novo deve ser controlado por flags/capabilities.
3. **Fallback**: a aplicação deve iniciar e operar **sem FreeCAD** (modo reduzido).
4. **Não travar UI**: operações pesadas devem rodar em worker/thread/processo.
5. **Undo/Redo & Audit trail**: adotar Command Pattern para operações do modeler.
6. **Separar precisão geométrica de visualização**:
   - B-Rep/OCCT: verdade geométrica
   - Mesh/triangulação: render no viewport

---

## 1) Escopo por fases (o agente deve executar nesta ordem)

### Fase 1 — MVP do Modeler (entrega rápida sem regressão)
- Nova aba **Mecânica** (UI mínima, sem quebrar navegação atual).
- Kernel mecânico com interface **MechanicalKernel**.
- Provider **NullMechanicalProvider** (fallback).
- Provider **FreeCADKernelProvider** (in-process, mínimo):
  - primitivas: Box/Cylinder/Sphere/Cone + Tube (prioridade alta)
  - transformações: translate/rotate
  - booleanas: union/cut/common
  - import/export: STEP + STL (mínimo)
- Diagnóstico do backend (capabilities):
  - `freecad_available`, `freecad_headless_available`, `fem_available`, etc.
- Menu de contexto (botão direito) com todas operações essenciais.

### Fase 2 — Robustez CAD
- Edição paramétrica (update sem trocar ID).
- Validação/healing (quando disponível).
- Import/export ampliado: IGES/OBJ.
- Medições: bbox, área, volume, centro geométrico, massa estimada (com densidade).
- Worker headless via **FreeCADCmd** (fallback robusto para operações pesadas).

### Fase 3 — Análise mecânica (FEM-ready)
- Atribuição de materiais.
- Geração de malha (quando toolchain disponível).
- Cargas/vínculos e export de caso.
- Integração FEM (FreeCAD FEM / CalculiX / Gmsh) **somente se habilitado por capability**.

---

## 2) Arquitetura recomendada (Provider/Adapter)

### 2.1 Núcleo (agnóstico ao FreeCAD)
Criar pacote/módulo:
```
src/.../mechanical/
  interfaces.py
  models.py
  commands.py
  diagnostics.py
  validators.py
  io.py
  providers/
    null_provider.py
    freecad_provider.py
  worker/
    freecad_headless_worker.py   (fase 2)
```

### 2.2 UI (mantém padrões atuais)
```
src/.../ui/mechanical/
  mechanical_tab.py
  object_tree.py
  properties_panel.py
  mechanical_toolbar.py
  viewport_adapter.py
  context_menu.py
```

### 2.3 Contratos mínimos (interfaces)
**MechanicalKernel** deve expor métodos simples e estáveis:
- `create_primitive(kind, params) -> obj_id`
- `transform(obj_id, matrix/pos/rot)`
- `boolean(op, a_id, b_id) -> obj_id`
- `import_model(path, fmt) -> [obj_id]`
- `export_model(obj_ids, path, fmt)`
- `get_properties(obj_id) -> dict`
- `set_properties(obj_id, dict)`
- `triangulate(obj_id, quality) -> mesh_payload`
- `validate(obj_id) -> report`
- `heal(obj_id) -> report` (quando disponível)
- `delete(obj_id)`

**NullMechanicalProvider** deve implementar tudo retornando erros controlados e/ou no-op.

---

## 3) Backend FreeCAD: dois modos (in-process + headless)

### 3.1 Modo A — In-process (preferencial)
- Detectar se `import FreeCAD` e `import Part` funcionam.
- Criar documento interno (`FreeCAD.newDocument` se aplicável).
- Produzir shapes (Part.Shape) para primitivas/booleanas.
- Para render, triangulação → payload (vertices/faces) ou arquivo temporário STL/OBJ.

**Regras de segurança:**
- Nunca deixar exceções escaparem para UI.
- Sempre logar stacktrace no logger do app.

### 3.2 Modo B — Headless via FreeCADCmd (fallback robusto)
- Detectar `FreeCADCmd` no PATH ou em caminhos comuns.
- Implementar worker executável com protocolo JSON:
  - request: `{command, params, input_files, output_files}`
  - response: `{ok, logs, metrics, outputs}`
- Usar headless para:
  - import/export pesado
  - healing/validate
  - booleanas complexas
  - batch recompute

---

## 4) UI/UX — Aba Mecânica (intuitiva e customizável)

### 4.1 Layout (dock-like)
- **Viewport 3D** (centro)
- **Tree de Objetos** (esquerda)
- **Propriedades** (direita)
- **Console/Log Mecânico** (inferior, dobrável)
- **Diagnostics Backend** (painel “Sobre/Diagnóstico”)

### 4.2 Toolbar mínima (Fase 1)
- New/Clear Scene
- Import / Export
- Select / Multi-select
- Move / Rotate
- Primitives: Box, Cylinder, Sphere, Cone, Tube
- Boolean: Union, Cut, Intersect
- Measure (bbox/volume)
- Validate / Heal (feature-gated)
- Recompute

### 4.3 Menu de contexto (botão direito) — **todas operações por aqui**
Em qualquer objeto (viewport e árvore), ao clicar com botão direito:
- Selecionar / Multi-selecionar
- Focar câmera / Zoom to selection
- Ocultar / Mostrar
- Renomear
- Duplicar
- Agrupar / Desagrupar
- Transformar:
  - Move
  - Rotate
  - Reset transform
- Booleanas:
  - Union (com seleção atual)
  - Cut (A - B)
  - Intersect
- Propriedades:
  - Editar parâmetros (primitiva)
  - Material/densidade (fase 2/3)
- Medições:
  - BBox / Área / Volume / Centro
- Validar / Curar shape (heal) (fase 2, se disponível)
- Exportar objeto selecionado (STEP/STL)
- Excluir

**Regras:**
- Se uma operação não estiver disponível (capability/seleção), item fica desabilitado e exibe motivo no tooltip.
- Operações longas disparam job assíncrono e exibem progresso + cancelar.

---

## 5) Modelo de dados mecânico (persistência no projeto)

### 5.1 Estruturas (mínimo)
- `mechanical_scene`:
  - `objects[]`
  - `groups[]`
  - `selection_state`
  - `view_state` (camera)
  - `backend` (provider atual)
  - `dirty` flag
- `mechanical_object`:
  - `id`, `name`, `type`
  - `params` (ex.: radius, height)
  - `transform` (matriz 4x4 ou pos/rot/scale)
  - `shape_ref` (referência interna do provider)
  - `mesh_cache` (render)
  - `metadata` (source, timestamp, tags)

### 5.2 Serialização
- JSON versionado (ex.: `mechanical_scene_v1.json`)
- Regras:
  - IDs estáveis
  - compatibilidade retroativa (migrations leves)
  - assets grandes (mesh) fora do JSON, referenciados por caminho/hash

---

## 6) Command Pattern (Undo/Redo + auditoria)

### 6.1 Comandos obrigatórios (Fase 1)
- `CreatePrimitiveCommand`
- `TransformCommand`
- `BooleanCommand`
- `ImportCommand`
- `DeleteCommand`
- `RenameCommand`
- `VisibilityCommand`

### 6.2 Auditoria
Cada comando:
- gera log estruturado
- anexa payload mínimo (sem dados gigantes)
- registra tempo e status
- permite replay em modo “diagnóstico”

---

## 7) Instalação/Dependências (sem quebrar o core)

### 7.1 Estratégia recomendada
- Preferir **FreeCAD instalado externamente** + detecção no app.
- Se necessário, usar **Conda/Mamba** em ambiente separado para toolchain CAD.

### 7.2 “Doctor” / Diagnostics (obrigatório)
Implementar comando/aba de diagnóstico que:
- detecta `FreeCADCmd`
- tenta `import FreeCAD`, `import Part`
- lista versões (se possível)
- detecta FEM toolchain (gmsh/calculix) se habilitado
- valida OpenGL mínimo (para viewport)
- salva relatório em arquivo de log

---

## 8) Testes (robustos, sem regressão)

### 8.1 Regras
- Testes do core atual devem continuar passando.
- Novos testes devem:
  - rodar sem FreeCAD (fallback)
  - rodar com FreeCAD (quando ambiente disponível) via marker/flag

### 8.2 Matriz mínima de testes
**Unitários (sem FreeCAD)**
- Null provider: chamadas retornam “not available” com mensagens claras
- Command stack: undo/redo e auditoria

**Integração (com FreeCAD, gated)**
- create primitive -> triangulate -> render payload válido
- boolean union/cut/common em sólidos simples
- export STEP/STL e reimport (smoke)
- transform move/rotate e valida propriedades

**UI smoke (pytest-qt)**
- abrir aba mecânica
- criar primitive via toolbar/context menu
- seleção no tree sincroniza com viewport

### 8.3 Gate de qualidade
Adicionar script/target:
- `pytest -q` (core)
- `pytest -q -m mechanical` (se FreeCAD disponível)
- “doctor” check em CI (sem exigir FreeCAD)

---

## 9) Licença e conformidade (obrigatório)
- Adicionar `THIRD_PARTY_LICENSES.md` com FreeCAD/OCCT.
- Adicionar seção “Licenças” no About.
- Evitar qualquer acoplamento que implique static linking do núcleo.

---

## 10) Checklist de entrega (Definition of Done)

### Fase 1 — DoD
- [ ] App inicia e funciona sem FreeCAD.
- [ ] Aba Mecânica abre, mostra UI base e Diagnostics.
- [ ] Com FreeCAD disponível:
  - [ ] cria primitivas (Box/Cylinder/Sphere/Cone/Tube)
  - [ ] move/rotate
  - [ ] boolean union/cut/intersect
  - [ ] import/export STEP e STL
  - [ ] menu de contexto contém operações essenciais
- [ ] Jobs pesados não travam UI
- [ ] Logs e erros claros
- [ ] Testes mínimos adicionados e passando

---

## 11) Plano de execução (ordem exata, passo a passo)

1. **Inventariar padrões do repositório atual**: logging, threading, state, UI tabs, comandos.
2. Implementar `MechanicalKernel` + modelos + diagnostics.
3. Implementar `NullMechanicalProvider`.
4. Implementar detecção:
   - `FreeCADCmd` no PATH
   - tentativa `import FreeCAD`
   - capabilities
5. Implementar `FreeCADKernelProvider` (in-process):
   - primitivas
   - transform
   - boolean
   - triangulação para viewport
   - export/import (STEP/STL)
6. Criar aba mecânica (UI mínima):
   - viewport adapter
   - tree
   - properties
   - toolbar
   - context menu completo
7. Implementar Command Pattern + undo/redo básico.
8. Adicionar testes unit + integração gated.
9. Implementar worker `FreeCADCmd` (fase 2) como fallback.
10. Expandir: medidas/heal/validate, import/export extra (fase 2).
11. Preparar base FEM (fase 3) apenas se capability habilitar.

---

## 12) Notas de integração com AFTX Suite Framework
- O módulo mecânico deve ser registrado como “recurso” do framework principal:
  - plug-in/feature module
  - inicialização sob demanda
  - dependências opcionais
- Nenhuma alteração estrutural agressiva no core antes do MVP funcionar.

---

## Apêndice A — Operações mínimas que DEVEM existir via botão direito
- Create primitive (sub-menu)
- Transform (Move/Rotate/Reset)
- Boolean (Union/Cut/Intersect)
- Import/Export (objeto selecionado)
- Measure (bbox/volume)
- Validate/Heal (se disponível)
- Rename/Hide/Show/Delete/Duplicate
