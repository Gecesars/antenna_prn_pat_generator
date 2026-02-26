# Plano de Testes (audit suite)

Data: 2026-02-20

## Objetivo
Cobrir fluxos criticos com regressao controlada em quatro camadas: unit, integration, ui e golden.

## Estrutura
- `tests/unit/`: funcoes puras e hooks de auditoria.
- `tests/integration/`: pipelines de export/import e encadeamento de componentes.
- `tests/ui/`: smoke tests de construcao de componentes visuais.
- `tests/golden/`: regressao visual e estabilidade de pipeline de imagem.
- `tests/external/`: validacao real com AEDT/HFSS (opt-in por variavel de ambiente).
- `tests/mech/`: regressao do modeler mecanico e provider FreeCAD (gated).

## Casos obrigatorios por camada

### Unit
- Validar `emit_audit` no-op com `DEBUG_AUDIT` desligado.
- Validar `emit_audit` e `audit_span` com `DEBUG_AUDIT=1`.
- Validar cobertura estrutural de auditoria nas rotas principais de export (`tests/unit/test_export_audit_coverage.py`).
- Validar invariantes de parser/conversao/metricas (suite existente em `tests/test_*.py`).

### Integration
- Pipeline AEDT export local:
  - corte -> CSV/JSON;
  - grid -> NPZ;
  - grid dB -> OBJ/MTL.
- Pipeline relatorio:
  - payload multipagina;
  - compactacao de tabela;
  - integridade de pagina.

### UI smoke
- Import e construcao dos componentes:
  - `AedtLiveTab` logger builder;
  - `MechanicalAnalysisTab` smoke (suite existente).
  - `MechanicsPage` com diagnostics backend e acoes de validate/heal.

### Mechanical kernel
- Sem FreeCAD:
  - fallback `NullMechanicalProvider` sem crash;
  - command stack com classes dedicadas (`CreatePrimitiveCommand`, `TransformCommand`, etc.);
  - doctor report com capacidades.
- Com FreeCAD (gated por disponibilidade):
  - create primitive -> triangulate;
  - transform + boolean;
  - export/import STEP/STL.

### Golden
- Pipeline de imagem com baseline fixo (`sample_input.png` -> `sample_expected.png`).
- Comparacao por PSNR (limiar >= 45 dB).
- PDF por pagina renderizada com baseline fixo (`test_pdf_report_page_golden.py`).

### External (AEDT real)
- Conexao real de sessao (`AedtHfssSession.connect/disconnect`).
- Extracao real de cortes HRP/VRP.
- Extracao real de grid 3D.
- Skip automatico quando `EFTX_RUN_EXTERNAL_AEDT != 1` ou ambiente nao elegivel.

## Comandos padrao
```powershell
venv\Scripts\python.exe -m pytest -q tests --maxfail=1
venv\Scripts\python.exe -m pytest -q tests/unit tests/integration tests/ui tests/golden
venv\Scripts\python.exe -m pytest -q tests/external
venv\Scripts\python.exe -m pytest -q -m mechanical --maxfail=1
venv\Scripts\python.exe -m pytest -q tests/mech/test_freecad_provider_gated.py --maxfail=1
venv\Scripts\python.exe tools/mechanical_doctor.py --json out/mechanical_doctor_report.json
.\scripts\test_external_aedt.ps1 -Project "D:\Simulations\...\projeto.aedt" -Design "HFSSDesign1" -Setup "Setup1 : LastAdaptive"
.\scripts\run_external_auto.ps1
.\scripts\run_quality_gate.ps1
.\\scripts\\run_quality_gate.ps1 -WithMechanical
.\scripts\run_quality_gate.ps1 -WithExternalAedt
venv\Scripts\python.exe tools\generate_audit_docs.py --check
venv\Scripts\python.exe -m pytest --cov=core --cov-report=term-missing -q tests/unit tests/integration tests/ui tests/golden
python -m compileall core eftx_aedt_live reports deep3.py
```

## Variaveis para testes externos
```powershell
$env:EFTX_RUN_EXTERNAL_AEDT="1"
$env:EFTX_AEDT_VERSION="2025.2"
$env:EFTX_AEDT_PROJECT="D:\Simulations\...\projeto.aedt"   # opcional
$env:EFTX_AEDT_DESIGN="HFSSDesign1"                        # opcional
$env:EFTX_AEDT_SETUP="Setup1 : LastAdaptive"               # opcional
$env:EFTX_AEDT_SPHERE="3D_Sphere"                          # opcional
$env:EFTX_AEDT_EXPR="dB(GainTotal)"                        # opcional
$env:EFTX_AEDT_FREQ="635MHz"                               # opcional
$env:EFTX_AEDT_CONNECT_MODE="attach"                       # attach | new
$env:EFTX_AEDT_NON_GRAPHICAL="0"
$env:EFTX_AEDT_REMOVE_LOCK="1"
```

## Criterios de aceite
- Zero falhas em `tests/` no ambiente baseline.
- Nenhum crash em compile/smoke.
- Hooks de auditoria nao alteram comportamento quando `DEBUG_AUDIT` nao esta ativo.
- Exportacoes de corte/grid/relatorio continuam gerando artefatos validos.
- Modulo mecanico opera sem FreeCAD e habilita recursos CAD quando FreeCAD estiver disponivel.
