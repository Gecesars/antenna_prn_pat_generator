# Roadmap de Execucao (EFTX Diagram Suite)

Data de criacao: 2026-02-20
Horizonte: 5 sprints semanais

## Objetivo
Fechar lacunas de qualidade, validar fluxos AEDT reais, aumentar cobertura de testes e consolidar release profissional com rastreabilidade.

## Prioridades
- `P0`: estabilidade, fluxo AEDT real, exportacao auditada fim-a-fim.
- `P1`: cobertura, regressao visual PDF, benchmark e performance.
- `P2`: automacao documental e checklist de release.

## Sprint 1 (2026-02-23 a 2026-03-01) - Fechamento P0 inicial
### Escopo
- Commit/push limpo da base de auditoria/testes/docs.
- Expandir auditoria para todas as rotas de export do `deep3.py`:
  - PAT, PRN, CSV, PNG, PDF, PAT ADT.
- Revisao de logs de erro com padrao unico.
- Consolidar modulo mecanico provider-based (`mech/mechanical/*`) com doctor e fallback sem FreeCAD.

### Criterios de aceite
- Todas as rotas de export emitem evento de auditoria com `DEBUG_AUDIT=1`.
- `venv\Scripts\python.exe -m pytest -q --maxfail=1` sem falhas.
- `python -m compileall core eftx_aedt_live reports deep3.py` sem erro.
- `venv\Scripts\python.exe tools\mechanical_doctor.py --json out\mechanical_doctor_report.json` executa sem falha.

## Sprint 2 (2026-03-02 a 2026-03-08) - Fluxo AEDT/HFSS real
### Escopo
- Criar `tests/external/` para validacao real com AEDT.
- Cobrir fluxo completo:
  - conectar;
  - selecionar projeto/design/setup;
  - verificar solucao;
  - criar/verificar esfera infinita;
  - extrair HRP/VRP;
  - enviar para projeto.
- Skip automatico quando AEDT nao estiver disponivel.

### Criterios de aceite
- Suite externa executa em ambiente com AEDT e gera log de evidencias.
- Em ambiente sem AEDT, testes externos ficam `SKIPPED`, sem quebrar CI local.
- Fluxo manual da aba `AEDT Live` validado sem travamento da UI.

## Sprint 3 (2026-03-09 a 2026-03-15) - Cobertura de codigo
### Escopo
- Aumentar cobertura de `core` com foco em:
  - `core/analysis/*`
  - `core/reconstruct3d.py`
  - `core/math_engine.py`
  - `core/numba_kernels/*`
- Adicionar testes de borda numerica e consistencia angular.

### Criterios de aceite
- Cobertura `core` >= 60% no baseline definido.
- Nenhuma regressao em testes existentes.
- Casos de NaN/inf/entrada degenerada cobertos por teste.

## Sprint 4 (2026-03-16 a 2026-03-22) - Regressao visual e performance
### Escopo
- Implementar regressao visual de PDF por pagina renderizada.
- Consolidar baseline de benchmark:
  - drag marker;
  - refresh de tabela;
  - reconstrucao 3D;
  - export PDF multipagina.
- Publicar metas de desempenho em documento.

### Criterios de aceite
- Alteracao de layout PDF gera diff detectavel em teste.
- Benchmarks registrados em arquivo versionado.
- Sem regressao acima de 15% nos cenarios definidos (salvo justificativa documentada).

## Sprint 5 (2026-03-23 a 2026-03-29) - Release e automacao final
### Escopo
- Automatizar smoke pos-instalacao MSI:
  - abertura do app;
  - carregamento de dependencias HFSS/AEDT;
  - teste rapido de conectividade.
- Automatizar geracao de docs:
  - `feature_inventory`, `function_map`, `coverage_matrix`.
- Padronizar checklist de release.

### Criterios de aceite
- Instalador validado em ambiente limpo sem erro de dependencia critica.
- Checklist de release executavel e versionado.
- Roadmap marcado como concluido com evidencias em `docs/qa_report.md`.

## Dependencias e riscos
- Disponibilidade de maquina com AEDT/HFSS para testes externos.
- Variacoes de ambiente Windows (DLL/runtime) podem afetar conectividade.
- Regressao visual de PDF depende de renderizacao deterministica do ambiente.

## Definicao de pronto (DoD)
- Todos os itens `P0` concluidos e validados.
- Testes verdes no baseline.
- Evidencias de QA atualizadas.
- Documentacao tecnica sincronizada com o codigo.
