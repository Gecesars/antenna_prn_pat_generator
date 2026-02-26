# Inventario de Features (EFTX Diagram Suite)

Data de auditoria: 2026-02-20

## Escopo
Inventario funcional do app principal (`deep3.py`) e modulos acoplados (`eftx_aedt_live`, `reports`, `core`, `mech`).

## Features por area

| Feature ID | Area | Entrada principal | Local de codigo | Status |
|---|---|---|---|---|
| F-ARQ-001 | Carga VRP | Botao `Load VRP...` | `deep3.py:6344` | Funcional |
| F-ARQ-002 | Carga HRP | Botao `Load HRP...` | `deep3.py:6383` | Funcional |
| F-ARQ-003 | Export PAT padrao | `Exp(.pat)` / `Export .PAT (All)` | `deep3.py:6307`, `deep3.py:6384` | Funcional |
| F-ARQ-004 | Export PAT ADT | `Exp(ADT)` | `deep3.py:6346`, `deep3.py:6385` | Funcional |
| F-ARQ-005 | Export PRN | `Export .PRN (All)` | `deep3.py:6308` | Funcional |
| F-ARQ-006 | Export imagem grafico | `Img Grafico` | `deep3.py:6351`, `deep3.py:6390` | Funcional |
| F-ARQ-007 | Export imagem tabela | `Img Tabela` | `deep3.py:6352`, `deep3.py:6391` | Funcional |
| F-VERT-001 | Composicao vertical | Aba `Composicao Vertical` | `deep3.py:6977` | Funcional |
| F-VERT-002 | Null fill por percentual | Calculo vertical + harness | `null_fill_synthesis.py`, `deep3.py:7060` | Funcional |
| F-VERT-003 | Tilt eletrico | Parametros verticais | `deep3.py:7193` | Funcional |
| F-HORZ-001 | Composicao horizontal | Aba `Composicao Horizontal` | `deep3.py:7760` | Funcional |
| F-HORZ-002 | Export HRP/PAT/PRN | botoes da aba horizontal | `deep3.py:7847-7851` | Funcional |
| F-STUDY-001 | Estudo completo simples/duplo | Aba `Estudo Completo` | `deep3.py:4824` | Funcional |
| F-STUDY-002 | Export completo por slot/polarizacao | `Exportar Estudo Completo` | `deep3.py:4832` | Funcional |
| F-PROJ-001 | Dados do projeto | Aba `Dados do Projeto` | `deep3.py:4853` | Funcional |
| F-PROJ-002 | Geracao de artefatos do projeto | exportacoes em lote | `deep3.py:5831`, `deep3.py:6132` | Funcional |
| F-ADV-001 | Marcadores A/B/delta | Aba `Visualizacao Avancada` | `ui/tabs/tab_advanced_viz.py:561` | Funcional |
| F-ADV-002 | Context menu completo (plot/tabela) | clique direito | `ui/tabs/tab_advanced_viz.py:861`, `ui/tabs/tab_advanced_viz.py:940` | Funcional |
| F-ADV-003 | Reconstrucao 3D assíncrona | Worker + cache | `ui/tabs/tab_advanced_viz.py:751`, `ui/tabs/tab_advanced_viz.py:805` | Funcional |
| F-AEDT-001 | Conectar/Desconectar AEDT | Aba `AEDT Live` | `eftx_aedt_live/ui_tab.py:1970` | Funcional |
| F-AEDT-002 | Selecao projeto/design/setup/esfera | metadata e apply | `eftx_aedt_live/ui_tab.py` | Funcional |
| F-AEDT-003 | Pull VRP/HRP sem interpolar | `Pull VRP` / `Pull HRP` | `eftx_aedt_live/ui_tab.py:2051`, `eftx_aedt_live/ui_tab.py:2059` | Funcional |
| F-AEDT-004 | Pull 3D + export NPZ/OBJ | `Pull 3D` | `eftx_aedt_live/ui_tab.py:2346`, `eftx_aedt_live/export.py:55` | Funcional |
| F-AEDT-005 | Inserir no projeto sob confirmacao | botoes `Insert`/`Send` | `eftx_aedt_live/ui_tab.py:2540` | Funcional |
| F-BATCH-001 | Biblioteca de diagramas | Aba `Diagramas (Batch)` | `deep3.py:2378` | Funcional |
| F-BATCH-002 | Renomear/Excluir thumbnail | contexto/acoes no thumb | `deep3.py:2708`, `deep3.py:2726` | Funcional |
| F-REP-001 | Relatorio PDF multipagina | exportacao profissional | `reports/pdf_report.py:336` | Funcional |
| F-REP-002 | Compactacao de tabela + CSV completo | logica de pagina/tabela | `reports/pdf_report.py:490` | Funcional |
| F-TENSOR-001 | Pipeline de imagem (GPU/CPU) | modulo tensor | `core/image_tensor/*` | Funcional |
| F-NUMBA-001 | Kernels numericos acelerados | modulo numba | `core/analysis/pattern_metrics.py:97` | Funcional |
| F-MECH-001 | Analise mecanica (scene/materials) | Aba `Analise Mecanica` | `ui/tabs/tab_mechanical_analysis.py:288`, `mech/*` | Funcional |
| F-MECH-002 | Kernel mecanico provider (FreeCAD + fallback + doctor) | `MechanicsPage` + `mech/mechanical/*` | `mech/engine/scene_engine.py`, `mech/mechanical/providers/*`, `tools/mechanical_doctor.py` | Funcional |
| F-DIV-001 | Modulo divisor separado | aba plugin | `divisor.py` + registro em `deep3.py:110` | Funcional |

## Observacoes
- Todas as features acima estao mapeadas na matriz `docs/coverage_matrix.md`.
- Pontos ainda sem automacao completa de teste real (HFSS/AEDT instalado) estao em `docs/gap_analysis.md`.
