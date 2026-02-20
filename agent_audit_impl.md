# agent_audit_impl.md — Agente de Auditoria, Teste e Implementação Completa (EFTX Diagram Suite)
Versão: 1.0 • Idioma: PT-BR • Objetivo: cobertura funcional total do app (auditar → testar → corrigir → implementar faltantes)

> **Missão do agente**  
Analisar **cada função** e **cada fluxo** do aplicativo, identificar o que está **funcional**, o que está **quebrado** e o que está **faltando**, criar uma **suite de testes robusta**, corrigir problemas e implementar funcionalidades ausentes, mantendo a aplicação **estável** e com **regressão controlada**.

> **Quando precisar de retorno do usuário**  
O agente deve perguntar somente quando houver ambiguidade real (ex.: comportamento esperado, formato de saída, regra de negócio). O protocolo de perguntas está na seção **11**.

---

## 0) Regras rígidas (obrigatórias)
1) **Não quebrar features existentes**: qualquer mudança deve manter compatibilidade com projetos/arquivos já existentes.
2) **Mudança mínima por PR**: alterações pequenas e rastreáveis; evitar “mega refactor” sem necessidade.
3) **Test-first quando possível**: bug → reproduzir → escrever teste → corrigir → garantir que o teste passa.
4) **Nada de suposições silenciosas**: se o comportamento esperado não for dedutível do código e docs, **perguntar ao usuário**.
5) **Logs e métricas**: toda falha deve ficar registrada (stack trace, inputs mínimos, versão).
6) **Sem travar UI**: tarefas pesadas devem rodar em worker (Qt) e ter progress/log/cancel.
7) **Documentação viva**: toda implementação nova deve atualizar docs e checklist.

---

## 1) Entregáveis finais do agente
### 1.1 Relatórios (arquivos .md)
- `docs/feature_inventory.md` — inventário completo de features/fluxos e onde ficam no código
- `docs/function_map.md` — mapa por módulo → funções → chamadas → efeitos
- `docs/test_plan.md` — plano de testes (unit/integration/ui)
- `docs/gap_analysis.md` — lacunas: faltantes/quebrados + severidade + prioridade
- `docs/change_log.md` — changelog técnico por PR
- `docs/qa_report.md` — evidências: comandos rodados, outputs, screenshots, hashes

### 1.2 Testes
- `tests/unit/` — testes unitários (core)
- `tests/integration/` — testes de pipelines e I/O
- `tests/ui/` — smoke tests de UI (quando aplicável)
- `tests/golden/` — regressão visual (PNG/PDF), com tolerâncias e diffs

### 1.3 Correções e implementações
- Bugs corrigidos com testes cobrindo o caso
- Features faltantes implementadas com critérios de aceite explícitos

---

## 2) Método de trabalho (pipeline do agente)
O agente deve seguir o fluxo abaixo SEM pular etapas:

### Fase A — Descoberta e inventário
1) Clonar/abrir repo, listar estrutura
2) Identificar entrypoints (ex.: `main.py`, `deep3.py`)
3) Mapear páginas/abas/telas e seus handlers
4) Gerar inventário de features (checklist) e “como testar” cada uma

### Fase B — Mapeamento de funções (function map)
1) Rodar busca estática (ripgrep) por:
   - handlers UI (`clicked.connect`, slots, callbacks)
   - funções de I/O (import/export)
   - cálculos numéricos (métricas)
2) Construir `function_map.md` com:
   - caminho do arquivo
   - assinatura da função
   - quem chama
   - efeitos (I/O, mutação de estado, UI)
   - dependências externas

### Fase C — Baseline (rodar e registrar)
1) Rodar a aplicação e registrar:
   - tempo de startup
   - consumo de memória
   - erros no log
2) Executar smoke manual guiado (script) e registrar falhas

### Fase D — Suite de testes (base)
1) Configurar `pytest` + `coverage`
2) Criar testes unitários para core (sem UI)
3) Criar testes de integração para fluxos principais (I/O, export PDF, import diagrama)
4) Criar golden tests (outputs fixos) para imagens/PDFs críticos

### Fase E — Gap analysis e priorização
Classificar problemas em:
- P0: crash, perda de dados, export incorreto, travamento UI
- P1: feature crítica faltando, erro numérico grande, UI travando frequentemente
- P2: inconsistência visual, UX ruim, warnings, performance moderada
- P3: melhorias e refinos

### Fase F — Implementar e corrigir em ciclos curtos
Para cada item:
1) Reproduzir
2) Escrever teste que falha
3) Corrigir/implementar
4) Garantir testes passam
5) Atualizar docs + checklist
6) Abrir PR com descrição objetiva

---

## 3) Instrumentação obrigatória (para “ver” tudo)
### 3.1 Logger unificado
Criar `core/logging/logger.py` (ou equivalente) com:
- log em arquivo (rotativo)
- log em console
- níveis: DEBUG/INFO/WARN/ERROR
- contexto: versão, módulo, thread, projeto atual, arquivo atual

### 3.2 “Audit hooks” (rastreamento de chamadas críticas)
Adicionar hooks (sem poluir) para rastrear:
- import/export (arquivo, tempo, tamanho)
- geração de relatório (PDF/PNG), tempo por página
- cálculo de métricas (tempo e parâmetros)
- integração externa (HFSS/AEDT), tempos e falhas

> Regra: hooks devem ser “no-op” em modo normal, ativados por `DEBUG_AUDIT=1`.

---

## 4) Como o agente valida “funcional”
Uma função/feature só é considerada **funcional** se:
1) Executa sem exceção
2) Produz saída correta (ou comparável com golden/reference)
3) Não degrada performance de forma inaceitável
4) Não quebra outro fluxo (regressão zero)

Para cada feature, o agente deve definir um **critério de aceite** mensurável.

---

## 5) Suite de testes robusta — especificação
### 5.1 Unit tests (core)
- Funções puras: métricas, parsing, normalização, conversões
- Validar tolerâncias numéricas
- Validar invariantes (monotonicidade, limites, NaN handling)

### 5.2 Integration tests (pipelines)
- Import → parse → métricas → export (PNG/CSV/PDF)
- Projeto: criar → salvar → reabrir → consistência
- Plugins: carregar → executar → salvar outputs

### 5.3 Golden tests (regressão visual)
- Comparar PNGs (diagramas/tabelas) por:
  - pixel-perfect quando determinístico
  - PSNR/SSIM quando houver variação pequena
- Para PDF:
  - extrair imagens/páginas (render) e comparar com baseline
  - ou comparar metadados + hashes por página

### 5.4 Performance tests
- `pytest-benchmark` para:
  - cálculo de métricas em 1801 pontos
  - export de relatório multi-página
  - import de arquivos grandes
- Critério: não regredir performance sem justificativa

### 5.5 UI smoke (PySide6)
- testes de abertura/fechamento
- clicar em botões principais
- executar export em background sem travar
- verificar que logs/progress aparecem

---

## 6) Matriz de cobertura (coverage matrix)
Gerar e manter `docs/coverage_matrix.md` com colunas:
- Feature ID
- Local (arquivo/aba)
- Teste (unit/integration/ui/golden)
- Status (PASS/FAIL)
- Observações

Nada entra em “Done” sem teste associado (exceto hotfix explicitamente marcado).

---

## 7) Implementação de funcionalidades faltantes: processo
Quando o agente identificar uma função faltante:
1) Criar um “spec mínimo” dentro de `docs/gap_analysis.md`:
   - o que falta
   - comportamento esperado
   - inputs/outputs
   - critérios de aceite
2) Se houver ambiguidade → perguntar ao usuário (seção 11)
3) Implementar no local correto (preferir `core/` + controller + UI)
4) Criar testes
5) Atualizar docs

---

## 8) Estratégia de refatoração segura (quando necessário)
Se o código estiver acoplado (UI misturada com core):
- refatorar em pequenos passos:
  1) extrair função pura
  2) manter assinatura antiga com wrapper (compatibilidade)
  3) escrever testes
  4) remover duplicações somente após cobertura

---

## 9) Organização de PRs (padrão obrigatório)
Cada PR deve conter:
- Resumo curto
- Lista de mudanças (arquivos)
- Evidências (testes rodados, outputs gerados)
- “Antes/depois” (quando for visual)
- Riscos e mitigação
- Plano de rollback (se aplicável)

Nomenclatura:
- `fix/...` para correções
- `feat/...` para features
- `refactor/...` para refatoração

---

## 10) Checklist operacional diário do agente
1) `git pull` / atualizar branch
2) Rodar `pytest -q` (rápido)
3) Rodar smoke mínimo (1–3 fluxos)
4) Implementar 1–3 itens (ciclo curto)
5) Rodar testes completos relevantes
6) Atualizar docs (gap + change log)
7) Abrir PR

---

## 11) Protocolo de perguntas ao usuário (quando necessário)
O agente só deve perguntar quando:
- houver múltiplas interpretações razoáveis do comportamento
- ou quando a saída esperada precisa de confirmação (formato/valores/UX)

### 11.1 Formato das perguntas (curto e objetivo)
O agente deve enviar um bloco com:
- **Contexto** (1–2 linhas)
- **Pergunta** (uma por vez, se possível)
- **Opções** (A/B/C) quando aplicável
- **Impacto** (o que muda conforme escolha)

Exemplo:
- “No export PDF, quando a tabela não couber, prefere (A) amostrar e salvar CSV completo, (B) quebrar em múltiplas páginas, ou (C) duas colunas na mesma página?”

### 11.2 Quando NÃO perguntar
- Quando o comportamento pode ser inferido por consistência com o restante do app
- Quando existe doc anterior especificando
- Quando a solução é claramente padrão de mercado (ex.: undo/redo, logs)

---

## 12) Comandos úteis (o agente deve usar e registrar)
### 12.1 Busca e inspeção
```bash
rg -n "def |class |connect\(|clicked|slot|export|import|pdf|hfss|aedt" .
python -m compileall .
```

### 12.2 Testes
```bash
pytest -q
pytest -q --maxfail=1
pytest -q tests/integration
pytest -q tests/golden
pytest -q tests/perf --benchmark-only
```

### 12.3 Coverage
```bash
pytest --cov=core --cov-report=term-missing
```

---

## 13) Definition of Done (DoD)
O trabalho do agente termina quando:
- 100% das features do inventário estão marcadas como implementadas e testadas
- zero falhas em unit/integration/golden (no baseline definido)
- performance não regrediu (benchmarks)
- docs atualizados
- usuário aprovou qualquer decisão que exigiu escolha

