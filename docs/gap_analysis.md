# Gap Analysis (estado atual)

Data: 2026-02-20

## Classificacao
- P0: crash/perda de dados/export incorreto/travamento.
- P1: feature critica faltando ou sem validacao robusta.
- P2: melhoria funcional/visual/performance moderada.
- P3: refinamentos de manutencao/documentacao.

## Itens abertos

| ID | Severidade | Gap | Impacto | Plano |
|---|---|---|---|---|
| GAP-001 | P1 | Execucao real AEDT/HFSS ainda nao validada em ambiente configurado | Regressao pode passar local e falhar em ambiente sem AEDT configurado | **Fechado**: validacao externa executada em 2026-02-20 com `scripts/run_external_auto.ps1` (`3 passed, 2 warnings`) e gate completo `run_quality_gate.ps1 -WithExternalAedt` |
| GAP-002 | P1 | Cobertura `core` ainda abaixo da meta no recorte minimo (37% em unit/integration/ui/golden) | Risco de regressao no recorte rapido de auditoria | **Meta global atingida** na suite completa (`65%`); manter plano de elevar o recorte minimo |
| GAP-003 | P2 | Falta baseline golden para PDF renderizado por pagina | Mudanca de layout pode passar despercebida | **Fechado** com `tests/golden/test_pdf_report_page_golden.py` + baseline `tests/golden/expected/pdf_report_page0_expected.png` |
| GAP-004 | P2 | Instrumentacao de auditoria ainda concentrada em pontos criticos, nao em 100% dos exports do app principal | Logs de trilha podem ficar incompletos em alguns fluxos antigos | **Fechado**: todas as rotas `export_*` + `generate_all_project_artifacts` em `deep3.py` cobertas por `_audit_export_start`, com testes estruturais em `tests/unit/test_export_audit_coverage.py` |
| GAP-005 | P3 | Nao ha script unico para gerar inventario/mapa automaticamente | Atualizacao documental manual pode ficar defasada | **Fechado** com `tools/generate_audit_docs.py` + saida versionada em `docs/*_auto.md` |

## Itens fechados neste ciclo

| ID | Severidade | Correcao aplicada |
|---|---|---|
| FIX-001 | P1 | Logger unificado implementado em `core/logging/logger.py` e integrado em `deep3.py`/`eftx_aedt_live/ui_tab.py` |
| FIX-002 | P1 | Audit hooks no-op por padrao implementados em `core/audit.py` com ativacao por `DEBUG_AUDIT=1` |
| FIX-003 | P1 | Instrumentacao adicionada para metricas, exportes AEDT e relatorio PDF |
| FIX-004 | P2 | Estrutura de testes expandida para `tests/unit`, `tests/integration`, `tests/ui`, `tests/golden` |
| FIX-005 | P2 | Documentacao de auditoria criada (`docs/*.md`) |
| FIX-006 | P2 | Auditoria adicionada nos fluxos assincronos `EXPORT_REPORT_PDF` e `EXPORT_WIZARD` em `deep3.py` |
| FIX-007 | P3 | Gerador automatico de docs adicionado em `tools/generate_audit_docs.py` com inventario/mapa auto |
| FIX-008 | P2 | Teste estrutural de cobertura de auditoria em exportacoes (`tests/unit/test_export_audit_coverage.py`) |
| FIX-009 | P1 | Fluxo externo AEDT validado em ambiente real com `tests/external` e integrado ao gate (`-WithExternalAedt`) |
| FIX-010 | P1 | Integracao mecanica provider-based com FreeCAD (capabilities/doctor/fallback) em `mech/mechanical/*` e `mech/engine/scene_engine.py` |
