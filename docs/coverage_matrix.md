# Coverage Matrix

Data: 2026-02-20

| Feature ID | Local | Teste(s) associado(s) | Tipo | Status |
|---|---|---|---|---|
| F-ARQ-001 | `deep3.py` carga/parse | `tests/test_parse_pat_roundtrip.py`, `tests/test_parse_prn_roundtrip.py` | Unit | PASS |
| F-ARQ-003 | export PAT | `tests/test_parse_pat_roundtrip.py` | Integration | PASS |
| F-ARQ-005 | export PRN | `tests/test_parse_prn_roundtrip.py` | Integration | PASS |
| F-VERT-002 | null fill | `tests/test_null_fill_synthesis.py` | Unit | PASS |
| F-VERT-003 | tilt/metrica elevacao | `tests/test_metrics_full_table.py`, `tests/test_beamwidth_xdb.py` | Unit | PASS |
| F-HORZ-001 | composicao horizontal | `tests/test_angles.py`, `tests/test_cut_tools.py` | Unit | PASS |
| F-STUDY-002 | export estudo completo | `tests/test_standard_table_sampling.py` | Integration | PASS |
| F-PROJ-002 | artefatos do projeto | `tests/integration/test_pattern_export_pipeline.py` | Integration | PASS |
| F-ADV-001 | markers + delta | `ui/tabs/tab_advanced_viz.py` smoke indireto + interactor tests existentes | UI/Unit | PASS parcial |
| F-ADV-003 | reconstrucao 3D | `tests/test_reconstruct3d_omni.py`, `tests/test_reconstruct3d_separable.py` | Unit | PASS |
| F-AEDT-001 | connect/disconnect | `tests/ui/test_aedt_live_logger_smoke.py`, `tests/external/test_aedt_live_external.py::test_external_connect_context` | UI/External | PASS |
| F-AEDT-003 | pull HRP/VRP | `tests/test_farfield_coerce.py`, `tests/external/test_aedt_live_external.py::test_external_extract_hrp_vrp_cuts` | Unit/External | PASS |
| F-AEDT-004 | export NPZ/OBJ | `tests/integration/test_pattern_export_pipeline.py`, `tests/external/test_aedt_live_external.py::test_external_extract_3d_grid` | Integration/External | PASS |
| F-BATCH-001 | biblioteca batch | `deep3.py` + smoke manual | Manual | PASS parcial |
| F-REP-001 | PDF multipagina | `tests/test_pdf_report_export.py`, `tests/golden/test_pdf_report_page_golden.py` | Integration/Golden | PASS |
| F-REP-002 | compactacao tabela | `tests/test_pdf_report_export.py::test_pdf_report_large_table_compaction_and_full_csv` | Integration | PASS |
| F-TENSOR-001 | pipeline tensor | `tests/tensor/*.py`, `tests/golden/test_tensor_pipeline_golden.py` | Unit/Golden | PASS |
| F-NUMBA-001 | kernels numba | `tests/numba/*.py`, `tests/perf/bench_numba.py`, `tests/unit/test_pattern_metrics_core.py` | Unit/Perf | PASS |
| F-MECH-001 | analise mecanica UI | `tests/mech/test_modeler_ux_smoke.py` | UI | PASS |
| F-MECH-002 | kernel provider (FreeCAD + fallback + doctor) | `tests/mech/test_mechanical_diagnostics.py`, `tests/mech/test_null_provider.py`, `tests/mech/test_scene_engine_commands.py`, `tests/mech/test_freecad_provider_gated.py` | Unit/Integration | PASS parcial |
| F-DIV-001 | modulo divisor | smoke manual de registro na aba | Manual | PASS parcial |

Legenda:
- `PASS`: coberto com teste automatizado passando.
- `PASS parcial`: validado parcialmente (smoke/manual e/ou sem fixture externa completa).
