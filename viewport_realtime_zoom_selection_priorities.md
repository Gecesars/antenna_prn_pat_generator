# viewport_realtime_zoom_selection_priorities.md
## EFTX — Diretriz de Evolução: Organização da UI + Renderização em Tempo Real + Zoom por Scroll + Prioridade do Viewport

> Documento de instrução para agente de implementação.
> Foco desta etapa: **viewport-first**, renderização em tempo real fluida, zoom por scroll de alta qualidade, refinamento de seleção e organização visual com prioridade à área de desenho.

---

## 1. Objetivo desta etapa

A interface já evoluiu bem (ribbon, viewport com quick toolbar, scene tree, propriedades e dock inferior).  
Nesta etapa, o objetivo é consolidar a experiência de uso com foco em produtividade de modelagem e inspeção:

1. **Priorizar a área de desenho (viewport)**
2. **Melhorar a renderização em tempo real**
3. **Garantir zoom com scroll suave, previsível e preciso**
4. **Refinar mecanismos de seleção (já funcionais)**
5. **Reduzir ruído visual sem perder capacidade operacional**

---

## 2. Diagnóstico do estado atual (com base na tela enviada)

### 2.1 Pontos fortes já visíveis
- Ribbon principal organizada por grupos:
  - `Project`, `Edit`, `View`, `Selection`, `Modeling`, `FEM`, `Import / Export`
- Viewport central com quick toolbar de navegação/render
- Scene Tree com colunas úteis (`Type`, `Visible`, `Locked`, `Layer`)
- Painel de propriedades contextual à direita
- Dock inferior com `Log / Measurements / Selection Info / Diagnostics / Solver Console`
- Seleção visual no viewport já perceptível (highlight e contorno)

### 2.2 Gargalos atuais (próxima prioridade)
1. **Muitas barras ocupando altura vertical**
   - topo com múltiplos blocos
   - sub-abas e quick toolbars competindo por espaço

2. **Viewport ainda não é claramente dominante**
   - a UI ainda “briga” com a área de desenho

3. **Painéis auxiliares grandes por padrão**
   - útil para debug/configuração
   - ruim para modelagem contínua

4. **Quick toolbar do viewport ainda pode ser compactada/contextualizada**
   - alguns toggles podem ser reorganizados em grupos

---

## 3. Princípio orientador desta fase: Viewport First

## 3.1 Regra central
A interface deve comunicar visualmente e funcionalmente que o **viewport é o centro do trabalho**.

### Implicações práticas
- Painéis laterais e inferior devem ser:
  - **colapsáveis**
  - **redimensionáveis**
  - **persistentes por layout**
- Controles raramente usados não devem ocupar espaço fixo contínuo
- O viewport deve ser maximizado em:
  - modelagem
  - seleção
  - inspeção geométrica
  - pós-processamento

---

## 4. Priorização da área de desenho (viewport) — mudanças concretas

## 4.1 Layout responsivo por modo de trabalho
A UI deve se adaptar ao modo ativo:

### A) Modo Modelagem / Seleção (Viewport prioritário)
- Painel esquerdo (Scene Tree): **compacto** (ex.: 260–320 px)
- Painel direito (Properties): **compacto** (ex.: 300–360 px)
- Dock inferior (Log etc.): **altura mínima** (ex.: 80–120 px)
- Viewport: ocupa o restante da janela com máxima prioridade

### B) Modo FEM Setup
- Painel direito pode expandir (materiais, BCs, loads, mesh, solve)
- Viewport permanece grande, mas aceita UI contextual mais rica

### C) Modo Resultados
- Viewport continua dominante
- Painel direito com resultados/probes/escala de contorno
- Dock inferior pode crescer para solver console/diagnostics se necessário

---

## 4.2 Painéis colapsáveis (obrigatório)
Implementar controles claros para:
- colapsar **painel esquerdo**
- colapsar **painel direito**
- colapsar **dock inferior**
- restaurar **layout padrão**

### Atalhos sugeridos
- `Tab` → alternar visibilidade dos painéis laterais (focus mode)
- `Shift+Tab` → alternar painel inferior
- `F11` (ou botão existente) → **Max Viewport** (quase tela cheia mantendo overlays)

### Requisito de UX
O botão **Max Viewport** deve ser:
- rápido
- previsível
- reversível
- sem quebrar a sincronização de layout

---

## 4.3 Ribbon compacta e progressiva
A parte superior está melhor, porém ainda consome altura.

### Melhorias esperadas
- no máximo **2 níveis visuais** permanentes
- grupos avançados em **expand/collapse**
- ações pouco usadas devem migrar para:
  - menu “More”
  - preferências
  - menus contextuais

### Regra prática
Se a ação não é usada continuamente em modelagem/seleção, ela não deve ocupar espaço fixo no topo.

---

## 4.4 Quick toolbar do viewport (compacta e contextual)
A quick toolbar do viewport é útil e deve permanecer, porém mais organizada.

### Organização sugerida em 3 grupos

#### Grupo 1 — Câmera
- Iso
- Top / Front / Right
- Fit All
- Fit Selection
- Zoom + / Zoom -

#### Grupo 2 — Visualização
- Shaded
- Wire
- X-Ray
- Grid
- Axes
- Section

#### Grupo 3 — Interação
- Selection mode
- Snap toggle
- Detail / LOD
- Screenshot

### Regras
- usar ícones consistentes + texto curto
- tooltips técnicos
- estados ativos bem destacados (toggle state)
- reduzir largura total para não competir com o desenho

---

## 5. Renderização em tempo real — metas e requisitos

## 5.1 Meta de experiência
A navegação deve ser **fluida e precisa**, mesmo com cenas intermediárias/complexas.

### Referência de UX (não rígida, mas alvo)
- Navegação (orbit/pan/zoom): ideal > **50–60 FPS**
- Hover/seleção: feedback visual em **< 50 ms**
- Highlight/outline: **sem flicker**
- Troca de modo visual (wire/xray/shaded): quase instantânea

---

## 5.2 Pipeline de render recomendado
Separar a renderização em passes para estabilidade e desempenho:

1. **Pass principal da cena**
   - shaded / wire / xray
   - grid / eixos
   - cores/materiais base

2. **Pass de picking (ID buffer)**
   - IDs únicos para objeto/subentidade
   - usado para hover/seleção

3. **Pass de highlight / overlay**
   - realce de hover/seleção
   - sem alterar material original

4. **Pass de outline (pós-processamento)**
   - contornos consistentes para selected e hover

5. **Pass de gizmos e overlays UI**
   - gizmo de transformação
   - view cube
   - hints de snap
   - HUD/status visual

---

## 5.3 Otimizações essenciais de tempo real

### A) LOD adaptativo durante navegação
Enquanto o usuário orbita/pan/zoom:
- reduzir detalhe visual temporariamente (se necessário)
- simplificar edges/wire
- restaurar qualidade total quando movimento cessa

### B) Throttling de hover/picking
Evitar picking pesado em toda microvariação de mouse.

Recomendação:
- hover pick limitado a ~30–60 Hz
- suspender atualização de hover se backend estiver ocupado
- retomar ao estabilizar movimento

### C) Cache de tesselação / buffers
- não retesselar por hover/seleção
- retesselar apenas quando geometria mudar
- reaproveitar buffers GPU sempre que possível

### D) Atualização parcial / redraw inteligente
- frustum culling (quando aplicável)
- minimizar redraws redundantes
- separar input/camera update/render para evitar travamentos

---

## 5.4 Estabilidade visual (qualidade perceptiva)
Evitar:
- flicker em outline
- z-fighting entre wire/overlay/face
- clipping agressivo em zoom próximo
- grid mudando de escala de forma brusca

---

## 6. Zoom com scroll — comportamento ideal (ponto crítico)

## 6.1 Requisitos funcionais
O zoom por scroll deve ser:

- **centrado no cursor** (cursor-centric zoom) sempre que possível
- suave e progressivo
- previsível (sem saltos)
- robusto em wheel e touchpad
- estável após `Fit All` / `Fit Selection`
- sem atravessar a geometria involuntariamente

---

## 6.2 Estratégia de foco do zoom (ordem de prioridade)
Ao usar scroll, definir o ponto de foco nesta ordem:

1. **Ponto sob o cursor** via raycast/picking
2. **Objeto selecionado** (pivot/centro)
3. **Foco atual da câmera**
4. **Centro do bounding box da cena**

Isso produz sensação de zoom inteligente e “grudado” no trabalho real do usuário.

---

## 6.3 Curva de zoom (evitar linear bruto)
Usar fator exponencial/logarítmico, não zoom linear simples.

### Motivo
- longe da cena → precisa passos maiores
- perto da peça → precisa passos finos e estáveis

### Resultado esperado
- zoom rápido em macroescala
- zoom delicado em detalhes
- sem overshoot próximo da geometria

---

## 6.4 Proteções e limites (obrigatório)

### A) Limite mínimo de aproximação (near safety)
Evitar que a câmera atravesse a geometria com scroll comum.

### B) Limite máximo (far cap)
Evitar afastamento excessivo e perda do contexto da cena.

### C) Clipping adaptativo (near/far)
Atualizar `near` e `far` com base em:
- bounding box da cena
- distância câmera-foco
- escala atual do modelo

**Objetivo:** evitar corte de faces ao aproximar.

### D) Anti-“túnel”
Se o scroll aproximar demais:
- travar em distância mínima
- ou converter comportamento em órbita ao redor do foco (sem penetrar a malha)

---

## 6.5 Scroll com modificadores (opcional, recomendado)
- `Scroll` → zoom normal
- `Ctrl + Scroll` → zoom fino
- `Shift + Scroll` → pan (opcional, dependendo da convenção escolhida)
- `Alt + Scroll` → ajuste de section/clipping (somente se fizer sentido e for consistente)

---

## 6.6 Critérios de qualidade do zoom por scroll
- [ ] zoom responde sempre
- [ ] não “engasga”
- [ ] sem inversões inesperadas
- [ ] mantém foco no ponto relevante
- [ ] funciona bem após seleção e `Fit Selection`
- [ ] comportamento consistente entre diferentes escalas de cena

---

## 7. Seleção — refinamento (mecanismo base já OK)

## 7.1 Regras de interação (padronizar e manter)
- Clique simples → selecionar único
- `Shift + Clique` → adicionar à seleção
- `Ctrl + Clique` → toggle
- `Esc` → limpar seleção
- Duplo clique → foco ou selecionar conjunto conectado (configurável)

---

## 7.2 Picking por modo (prioridade e tolerância)
Quando houver múltiplas entidades sob o cursor, respeitar o modo de seleção:

### Modo Object
- priorizar corpo/objeto inteiro

### Modo Face
- priorizar face visível sob cursor

### Modo Edge
- usar tolerância por pixel (hit slop) maior
- edge picking precisa ser utilizável mesmo sem zoom extremo

### Modo Vertex
- usar raio de hit em pixels (maior que ideal geométrico)
- manter precisão sem sacrificar usabilidade

> Edge/vertex selection deve ser calibrado por percepção do usuário, não apenas por precisão geométrica teórica.

---

## 7.3 Estados visuais de hover e seleção (refino)
Como o highlight já existe, consolidar padrão visual:

### Hover
- outline fino
- cor discreta (ex.: ciano)
- sem fill intrusivo

### Seleção ativa (single)
- outline mais espesso
- fill overlay leve (ex.: amarelo)
- gizmo no pivot (quando aplicável)

### Multi-seleção
- todos com outline consistente
- item ativo com destaque adicional

### Locked
- não responder a seleção normal
- feedback visual de bloqueio (ícone/estado)

---

## 7.4 Sincronização obrigatória
Seleção no viewport deve atualizar imediatamente:
- `Scene Tree`
- `Properties`
- `Selection Info`
- `Status Hint / Status Bar`

Seleção na árvore deve:
- realçar no viewport
- permitir `Zoom to Selection` (duplo clique ou menu contexto)

---

## 7.5 Seleção por área (box/lasso) — refinamento recomendado
Se já existir ou estiver em implementação, adicionar:
- modo `Window` vs `Crossing`
- somente visíveis vs incluir ocultos por profundidade
- filtros por tipo (obj/face/edge/vertex)
- overlay da área de seleção com transparência e borda clara

---

## 8. Organização visual — reduzir ruído sem perder capacidade

## 8.1 Separar ações contínuas de ações eventuais

### Ações contínuas (devem permanecer visíveis)
- navegação de viewport
- seleção
- modos de visualização
- transformações principais
- snapping
- scene tree (compacto)
- properties (contextual)

### Ações eventuais (podem ficar recolhidas)
- import setup avançado
- perfis de layout
- re-tessellate avançado
- backend diagnostics detalhado
- preferências menos frequentes

---

## 8.2 Painel de propriedades por contexto (progressivo)
Evitar exibir tudo simultaneamente.

### Exemplo de prioridade de abas
#### Se objeto selecionado (modo modelagem)
1. Properties
2. Transform
3. Material
4. FEM (apenas se estudo ativo)

#### Se modo FEM ativo
1. Study / BCs / Loads / Mesh / Solve
2. Properties (contexto geométrico)
3. Material

---

## 8.3 Dock inferior (Log etc.) com comportamento inteligente
O dock inferior é útil, mas deve ser menos intrusivo por padrão.

### Melhorias recomendadas
- altura mínima padrão (compacta)
- auto-expand em erro crítico
- destaque visual para warnings/errors
- filtros por categoria:
  - UI
  - Selection
  - Geometry
  - Mesh
  - FEM
  - Backend

---

## 9. Requisitos visuais do viewport (dominância da área de desenho)

## 9.1 O viewport deve dominar a composição
- borda sutil
- contraste bom
- grid discreto
- eixo útil sem poluição
- overlays compactos

## 9.2 Overlays não devem competir com o desenho
- quick toolbar menor
- HUD de status enxuto
- dados densos devem ficar no painel lateral/dock, não sobre a malha

## 9.3 Modo “Focus Modeling” (fortemente recomendado)
Criar um modo dedicado para modelagem:
- esconde painéis laterais e inferior
- mantém:
  - viewport
  - quick toolbar
  - ribbon compacta (ou auto-hide)
  - mini painel flutuante opcional (transform/snap)

---

## 10. Implementação técnica recomendada

## 10.1 Gestão de layout com splitters persistentes
Usar splitters (ou equivalente) com persistência por perfil:

- largura painel esquerdo
- largura painel direito
- altura dock inferior
- estado colapsado/expandido dos painéis

### Perfis de layout sugeridos
- `layout_modeling`
- `layout_fem`
- `layout_results`

---

## 10.2 Loop de interação do viewport (responsivo)
Separar responsabilidades:
- input events
- atualização de câmera
- hover/picking (com throttling)
- render frame
- atualização de painéis (quando necessário)

### Regra importante
Evitar acoplamento entre:
- mouse move
- retesselação
- chamadas pesadas de backend

---

## 10.3 Fluxo de zoom por scroll (pseudocomportamento)

1. Capturar delta do wheel/touchpad
2. Normalizar delta
3. Tentar raycast/pick no cursor
4. Definir foco do zoom (pick → seleção → foco atual → centro da cena)
5. Aplicar dolly com fator exponencial
6. Ajustar clipping near/far dinamicamente
7. Solicitar redraw sem invalidar buffers desnecessários

---

## 10.4 Seleção + render (latência baixa)
- highlight de seleção deve atualizar no mesmo frame (ou no próximo frame)
- painel de propriedades pode atualizar de forma desacoplada se consulta for pesada
- hover não deve travar o viewport

---

## 11. Critérios de aceitação (foco desta etapa)

## 11.1 Viewport prioritário
- [ ] viewport é claramente a maior área útil
- [ ] painéis laterais e inferior são colapsáveis
- [ ] `Max Viewport` funciona bem e retorna ao layout anterior
- [ ] layout permanece utilizável em telas menores

## 11.2 Zoom com scroll
- [ ] zoom centrado no cursor (quando possível)
- [ ] movimento suave e previsível
- [ ] sem saltos bruscos
- [ ] clipping adaptativo evita cortes indevidos
- [ ] não atravessa geometria facilmente
- [ ] funciona bem com mouse wheel e touchpad

## 11.3 Renderização em tempo real
- [ ] orbit/pan/zoom fluídos
- [ ] highlight/outline sem flicker
- [ ] hover responsivo
- [ ] troca de visual rápida (shaded/wire/xray)
- [ ] sem retesselação desnecessária em hover/seleção

## 11.4 Seleção refinada
- [ ] hover distinto de seleção
- [ ] multi-seleção consistente
- [ ] scene tree ↔ viewport sincronizados
- [ ] properties e selection info atualizam corretamente
- [ ] edge/vertex picking utilizável com tolerância adequada

---

## 12. Priorização de implementação (ordem recomendada)

## Sprint desta etapa — ordem ideal
1. **Viewport-first layout**
   - painéis colapsáveis
   - splitters persistentes
   - `Max Viewport` estável
   - redução de altura ocupada no topo

2. **Zoom com scroll premium**
   - cursor-centric zoom
   - foco por raycast/pick
   - curva suave (exponencial)
   - clipping adaptativo

3. **Refino de render em tempo real**
   - throttling de hover
   - caching de picking/render
   - redraw inteligente

4. **Refino de seleção**
   - estados visuais finais
   - tolerâncias por modo
   - sincronização total tree ↔ viewport ↔ properties

5. **Compactação da quick toolbar**
   - grupos mais claros
   - toggles com estados visíveis
   - menor ruído visual

---

## 13. Texto direto para task do agente (resumo operacional)

**Implementar evolução viewport-first com foco em renderização em tempo real e zoom por scroll de alta qualidade.**  
Priorizar a área de desenho com painéis colapsáveis, splitters persistentes e layouts por modo (Modeling/FEM/Results), incluindo `Max Viewport` estável. Refinar o zoom por scroll para comportamento cursor-centric (dolly toward focus/pick point), suave, previsível, com clipping adaptativo e bom suporte a mouse wheel e touchpad. Consolidar seleção já funcional com refinamento visual (hover vs selected vs active multi-select), sincronização completa Scene Tree ↔ Viewport ↔ Properties e tolerâncias adequadas para edge/vertex. Reduzir ruído visual das barras e overlays, mantendo a quick toolbar do viewport compacta, clara e contextual. Garantir renderização fluida na navegação, highlight sem flicker e sem retesselação desnecessária durante hover/seleção.

---

## 14. Observações finais de engenharia
- Preservar compatibilidade com a base atual e com o runtime/backend já integrado
- Evitar acoplamento excessivo entre UI e backend de geometria
- Priorizar latência percebida do viewport acima de atualizações secundárias de painel
- Toda mudança de layout deve ser reversível e persistente
- Instrumentar logs/perf counters (FPS, hover latency, redraw cause) para depuração durante desenvolvimento
