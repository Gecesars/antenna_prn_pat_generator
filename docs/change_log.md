# Change Log Tecnico

Data: 2026-02-20  
Escopo: implementacao de `agent_audit_impl.md`

## 1) Infra de logging/auditoria
- Adicionado `core/logging/logger.py`:
  - `LoggerConfig`
  - `build_logger()` com RotatingFileHandler e opcao de console.
- Adicionado `core/audit.py`:
  - `audit_enabled()`
  - `emit_audit()` (no-op por padrao)
  - `audit_span()` (medicao de duracao + status).
- Integracao no app:
  - `deep3.py` agora usa `build_logger`.
  - `deep3.py` emite evento de startup de auditoria.

## 2) Integracao AEDT/HFSS com audit hooks
- `eftx_aedt_live/ui_tab.py`:
  - logger migrado para `build_logger`.
  - spans/eventos de auditoria em connect, pull cut e pull 3D.
- `eftx_aedt_live/session.py`:
  - auditoria em connect/disconnect (`HFSS_SESSION_*`).
- `eftx_aedt_live/export.py`:
  - auditoria nos exports CSV/JSON/NPZ/OBJ.

## 3) Relatorio PDF e metricas
- `reports/pdf_report.py`:
  - eventos de auditoria para inicio/fim do export e por pagina.
- `core/analysis/pattern_metrics.py`:
  - spans/eventos em metricas, decimacao e integracao de potencia.

## 4) Novas suites de teste
- `tests/unit/test_audit_hooks.py`
- `tests/unit/test_pattern_metrics_core.py`
- `tests/unit/test_logging_and_perf.py`
- `tests/integration/test_pattern_export_pipeline.py`
- `tests/ui/test_aedt_live_logger_smoke.py`
- `tests/golden/test_tensor_pipeline_golden.py`
- `tests/external/conftest.py`
- `tests/external/test_aedt_live_external.py`
- `scripts/test_external_aedt.ps1` para execucao padronizada da suite externa.
- `scripts/run_external_auto.ps1` para execucao automatica com autodeteccao de projeto.
- `scripts/run_quality_gate.ps1` para gate completo (compile + tests + coverage + externo opcional).
- `pytest.ini` atualizado com `testpaths = tests` para evitar coleta de subprojetos externos.

## 5) Documentacao gerada
- `docs/feature_inventory.md`
- `docs/function_map.md`
- `docs/test_plan.md`
- `docs/gap_analysis.md`
- `docs/change_log.md`
- `docs/qa_report.md`
- `docs/coverage_matrix.md`
- `docs/perf_baseline.md`

## 6) Incrementos desta iteracao automatica
- `deep3.py`:
  - auditoria de inicio/fim adicionada para exportacoes assincronas:
    - `EXPORT_REPORT_PDF`
    - `EXPORT_WIZARD`
  - finalizacao de export agora registra status `ok/error/cancelled/skipped` com contadores.
- `tests/unit/test_export_audit_coverage.py`:
  - teste estrutural AST que valida presenca de `_audit_export_start/_audit_export_end`
    nas principais rotas de exportacao.
- `tools/generate_audit_docs.py`:
  - script AST para gerar automaticamente:
    - `docs/feature_inventory_auto.md`
    - `docs/function_map_auto.md`

## 7) Golden visual de PDF por pagina
- Dependencia adicionada: `pypdfium2==5.5.0` em `requirements.txt`.
- Baseline de regressao visual PDF:
  - `tests/golden/inputs/pdf_template_base.pdf`
  - `tests/golden/inputs/pdf_plot_base.png`
  - `tests/golden/expected/pdf_report_page0_expected.png`
- Novo teste:
  - `tests/golden/test_pdf_report_page_golden.py`
  - Exporta PDF de referencia, renderiza a pagina 0 e compara com baseline por PSNR.
- `pytest.ini` atualizado com marker `golden`.

## 8) Validacao externa AEDT concluida
- Execucao real de `tests/external` via `scripts/run_external_auto.ps1` em ambiente com AEDT:
  - `3 passed, 2 warnings`.
- Gate completo atualizado e validado com `-WithExternalAedt`:
  - compile + pytest + coverage + docs check + external AEDT.

## 9) Integracao mecanica FreeCAD/OCCT (provider-based)
- Nova camada `mech/mechanical/*`:
  - `interfaces.py` com contrato `MechanicalKernel`.
  - `providers/null_provider.py` para fallback seguro sem FreeCAD.
  - `providers/freecad_provider.py` para kernel in-process.
  - `diagnostics.py` para capabilities/doctor.
  - `worker/freecad_headless_worker.py` para protocolo JSON headless.
  - `io.py` + `validators.py` para `mechanical_scene_v1`.
- `mech/engine/commands.py`:
  - classes de comando dedicadas:
    - `CreatePrimitiveCommand`
    - `TransformCommand`
    - `BooleanCommand`
    - `ImportCommand`
    - `DeleteCommand`
    - `RenameCommand`
    - `VisibilityCommand`
- `mech/engine/scene_engine.py`:
  - bootstrap automatico do provider (`freecad` ou `null`);
  - diagnostics/capabilities (`backend_diagnostics`, `backend_capabilities`);
  - primitive nova `tube`;
  - import/export com STEP/STL/IGES via provider quando disponivel;
  - booleans/transform com tentativa de kernel + fallback mesh.
- `mech/ui/page_mechanics.py` e `mech/ui/context_menu.py`:
  - aba/painel Backend Diagnostics;
  - acao `Doctor`;
  - create `Tube...` no RMB;
  - export STEP no RMB e na tab Export;
  - acoes `Validate Shape` e `Heal Shape` com capability-gating.
- Scripts:
  - `scripts/install_mechanical_freecad.ps1`
  - `scripts/mechanical_doctor.ps1`
  - `tools/mechanical_doctor.py`
  - `scripts/run_quality_gate.ps1` atualizado com doctor e `-WithMechanical`.
- Testes novos em `tests/mech/`:
  - diagnostics/fallback
  - command classes
  - scene IO
  - provider FreeCAD gated por disponibilidade.

## 10) Ajuste de compatibilidade FreeCAD x Python (2026-02-21)
- `mech/mechanical/diagnostics.py`:
  - doctor passou a validar import real (`import FreeCAD` / `import Part`) ao inves de apenas `find_spec`.
  - adicao de campos:
    - `freecad.host_python_minor`
    - `freecad.freecad_cmd_python_minor`
    - `freecad.python_abi_match`
    - `freecad.import_errors`
  - resumo (`summary`) agora inclui `freecad:abi-mismatch` quando aplicavel.
- `scripts/install_mechanical_freecad.ps1`:
  - valida compatibilidade de Python minor entre app e `FreeCADCmd`.
  - emite warning claro em caso de mismatch ABI.
  - executa doctor ao final e gera `out/mechanical_doctor_report.json`.
- `tests/mech/test_freecad_provider_gated.py`:
  - gate alterado para import real de modulos (evita falso positivo com DLL/ABI incompativel).

## 11) Modelagem CAD importado (AEDT) + Boundaries (2026-02-24)
- `mech/engine/scene_engine.py`:
  - novo estado `boundaries` com metodos:
    - `apply_boundary`
    - `remove_boundary`
    - `clear_boundaries`
    - `boundaries_for_object`
  - import de modelos com metadata de assembly/body (`import_asset_*`, `import_body_*`).
  - fallback pyvista com suporte a multiblock (multi-body).
  - limpeza automatica de boundaries quando corpos sao removidos.
- `mech/mechanical/providers/freecad_provider.py`:
  - import STEP/IGES/STL com split de multi-corpos (`Solids`/`Shells`) em IDs separados.
- `mech/ui/scene_tree.py`:
  - arvore hierarquica por assembly importado;
  - checkbox de visibilidade por corpo e por grupo.
- `mech/ui/page_mechanics.py` + `mech/ui/context_menu.py` + `mech/ui/viewport_pyvista.py`:
  - nova aba `Boundaries`;
  - comandos de mouse direto na peca (`Ctrl+Click`, `Alt+Click`, `Shift+Alt+Click`);
  - submenu de boundaries no RMB da peca/arvore;
  - overlay visual de boundaries no viewport para selecao, com toggle na aba Boundaries.
- testes:
  - `tests/mech/test_import_metadata_and_boundaries.py`
  - `tests/mech/test_scene_engine_commands.py` (BoundaryCommand)
  - `tests/mech/test_modeler_ux_smoke.py` atualizado.
