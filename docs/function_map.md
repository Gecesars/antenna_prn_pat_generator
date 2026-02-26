# Function Map (modulos criticos)

Data de auditoria: 2026-02-20

## 1) Entrypoints e orquestracao

| Arquivo | Funcao/Classe | Chamadores | Efeitos |
|---|---|---|---|
| `deep3.py:2766` | `class PATConverterApp` | bootstrap `deep3.py` | Cria UI principal, abas, estado global de projeto |
| `deep3.py:129` | `setup_logging()` | init modulo | Inicializa logger principal `eftx` |
| `deep3.py:277` | `parse_auto(path)` | carga de arquivo VRP/HRP | Parse CSV HFSS ou tabela generica |

## 2) UI principal por abas

| Arquivo | Funcao | Papel | Dependencias |
|---|---|---|---|
| `deep3.py:6286` | `_build_tab_file()` | Aba Arquivo (import/export individual) | matplotlib, parsers, exporters |
| `deep3.py:6977` | `_build_tab_vertical()` | Composicao vertical, null fill, tilt | `null_fill_synthesis.py`, metricas |
| `deep3.py:7760` | `_build_tab_horizontal()` | Composicao horizontal | metricas, export PAT/PRN |
| `deep3.py:4824` | `_build_tab_study()` | Estudo completo, slot/polarizacao | sincronizacao com arquivo/composicao |
| `deep3.py:4853` | `_build_tab_project()` | Projeto completo, artefatos e preview | exportadores, tabela/plot, contexto |
| `deep3.py:6132` | `export_project_table_images()` | Export CSV+PNG de tabelas | I/O de projeto |

## 3) AEDT/HFSS live stack

| Arquivo | Funcao/Classe | Chamadores | Efeitos |
|---|---|---|---|
| `eftx_aedt_live/session.py:462` | `_import_hfss()` | `AedtHfssSession.connect` | Import robusto da API HFSS |
| `eftx_aedt_live/session.py:649` | `class AedtHfssSession` | `AedtLiveTab` | Mantem sessao HFSS e contexto |
| `eftx_aedt_live/session.py:947` | `connect(...)` | `AedtLiveTab._connect` | Attach/new session, abre projeto/design |
| `eftx_aedt_live/session.py:1170` | `disconnect()` | `AedtLiveTab._disconnect` | Libera desktop/session |
| `eftx_aedt_live/farfield.py:318` | `class FarFieldExtractor` | `AedtLiveTab` | Extrai cortes e grid 3D |
| `eftx_aedt_live/farfield.py:336` | `extract_cut(req)` | `_pull_cut` | retorna angulos+valores do corte |
| `eftx_aedt_live/farfield.py:366` | `extract_grid(req)` | `_pull_3d` | retorna grid esferico |
| `eftx_aedt_live/export.py:14` | `class PatternExport` | `AedtLiveTab` | persiste JSON/CSV/NPZ/OBJ |
| `eftx_aedt_live/export.py:40` | `save_cut_json(...)` | `_store_cut_payload` | exporta corte 2D para arquivo |
| `eftx_aedt_live/export.py:55` | `save_grid_npz(...)` | `_pull_3d` | exporta grid 3D comprimido |
| `eftx_aedt_live/ui_tab.py:1970` | `_connect()` | botao Connect | inicia sessao e metadata |
| `eftx_aedt_live/ui_tab.py:2051` | `_pull_vrp()` | botao Pull VRP | requisita corte vertical |
| `eftx_aedt_live/ui_tab.py:2059` | `_pull_hrp()` | botao Pull HRP | requisita corte horizontal |
| `eftx_aedt_live/ui_tab.py:2346` | `_pull_3d()` | botao Pull 3D | extrai grid e gera OBJ/NPZ |
| `eftx_aedt_live/ui_tab.py:2540` | `_send_to_project()` | botao Send to Project | envia payload validado ao app |

## 4) Numerico e performance

| Arquivo | Funcao/Classe | Papel | Efeitos |
|---|---|---|---|
| `core/perf.py:8` | `class PerfTracer` | tracing de hot-path UI | logs de latencia > limiar |
| `core/analysis/pattern_metrics.py:97` | `metrics_cut_1d(...)` | metricas 1D (HPBW, pico, F/B etc.) | usa numba/python fallback |
| `core/analysis/pattern_metrics.py:139` | `smart_decimate_indices(...)` | decimacao para tabelas | preserva pontos relevantes |
| `core/analysis/pattern_metrics.py:165` | `integrate_power_numpy(...)` | integracao de potencia | apoio a metricas |
| `core/analysis/pattern_metrics.py:178` | `start_numba_warmup_thread(...)` | aquecimento JIT | reduz latencia inicial |

## 5) Relatorio PDF

| Arquivo | Funcao | Papel | Efeitos |
|---|---|---|---|
| `reports/pdf_report.py:336` | `export_report_pdf(...)` | gera PDF multipagina | header+plot+metricas+tabela por pagina |
| `reports/pdf_report.py:123` | `_format_table_rows(...)` | arredondamento/formatacao | padroniza angulos e dB |
| `reports/pdf_report.py:166` | `_metrics_table(...)` | tabela de metricas | layout consistente no PDF |
| `reports/pdf_report.py:234` | `_draw_plot_fitted(...)` | escala adaptativa do grafico | evita whitespace e distorcao |

## 6) Auditoria e logging adicionados

| Arquivo | Funcao | Comportamento |
|---|---|---|
| `core/logging/logger.py` | `build_logger(config)` | logger unificado com RotatingFileHandler |
| `core/audit.py` | `emit_audit(...)` | no-op por padrao; ativo com `DEBUG_AUDIT=1` |
| `core/audit.py` | `audit_span(...)` | mede blocos criticos e registra status/elapsed |

## Dependencias externas relevantes
- `customtkinter`, `matplotlib`, `numpy`, `Pillow`
- `reportlab`, `pypdf` (relatorio)
- `ansys.aedt.core`/PyAEDT (AEDT Live)
- `numba` (aceleracao opcional)

