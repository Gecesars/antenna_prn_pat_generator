# EFTX Converter - Documentacao Completa da Aplicacao

Data de referencia: 2026-02-14
Repositorio: `D:\dev\Conversor_hfss_pat_RFS`
Aplicacao principal: `deep3.py`

## 1. Visao geral

O sistema e uma aplicacao desktop em Python para:

- importar diagramas de antena (VRP/HRP) em diferentes formatos;
- visualizar diagramas em modo planar e polar;
- compor arrays verticais e horizontais;
- aplicar sintese de preenchimento de nulos (null fill) por ordem de nulo;
- trabalhar com estudo completo em modo simples ou duplo (2 polarizacoes);
- exportar arquivos tecnicos (`.pat`, `.pat ADT`, `.prn`);
- exportar imagens de graficos e tabelas;
- salvar/carregar trabalho em andamento;
- manter biblioteca local de diagramas (batch) com SQLite.

Interface: CustomTkinter + Matplotlib.
Processamento numerico: NumPy.

## 2. Arquitetura tecnica

### 2.1 Camadas

- GUI e orquestracao:
  - classe `PATConverterApp` em `deep3.py`.
  - classe modal `PRNViewerModal` para abrir/inspecionar PRN.
  - tab de biblioteca `DiagramsTab`.
- Parsing e IO:
  - leitura CSV/TXT/PAT/PRN.
  - escrita PAT padrao, PAT ADT, PRN.
- Processamento de sinal/antena:
  - reamostragem horizontal/vertical.
  - metricas de diagrama (HPBW, D2D, F/B, primeiro nulo).
  - composicao vertical/horizontal.
  - null fill em `null_fill_synthesis.py`.
- Persistencia:
  - SQLite para biblioteca (`~/.eftx_converter/library.db`).
  - JSON para salvar progresso (`*.eftxproj.json`).
  - registro interno de exportacoes (`export_registry`).

### 2.2 Arquivos principais

- `deep3.py`: aplicacao principal, UI, parsers, exportacoes, composicoes.
- `null_fill_synthesis.py`: motor numerico de sintese de null fill e harness.
- `tests/test_null_fill_synthesis.py`: testes de unidade do motor de null fill.
- `setup.py`: empacotamento Windows com `cx_Freeze` e geracao MSI.
- `requirements.txt`: dependencias Python.
- `library.db`: base SQLite inicial (copiada para area do usuario).

## 3. Abas da aplicacao (fluxo funcional)

## 3.1 Aba `Arquivo`

Objetivo: operacao base por diagrama individual.

Funcionalidades:

- carga de VRP e HRP (`Load VRP`, `Load HRP`);
- visualizacao VRP (planar/polar) e HRP (planar/polar);
- exportacao individual:
  - PAT padrao;
  - PAT ADT;
  - imagem do grafico;
  - imagem de tabela;
- exportacao combinada:
  - `Export .PAT (All)`;
  - `Export .PRN (All)`;
- edicao de metadados PRN:
  - `NAME`, `MAKE`, `FREQUENCY`, `GAIN`, `H_WIDTH`, `V_WIDTH`, `FRONT_TO_BACK`;
- utilitario `View PRN` abre modal de inspeccao e calculo de estatisticas.

Variaveis de contexto globais usadas em varias abas:

- `base_name_var`
- `author_var`
- `norm_mode_var`
- `output_dir`

## 3.2 Aba `Composicao Vertical`

Objetivo: compor elevacao de array vertical e aplicar null fill.

Entradas principais:

- numero de antenas (`N`);
- frequencia + unidade;
- espacamento vertical;
- tilt eletrico em graus;
- `null_fill_percent` em porcentagem;
- `null_order` (1o, 2o, 3o nulo etc);
- modo de controle:
  - `amplitude`
  - `phase`
  - `both`

Saidas:

- diagrama composto de elevacao;
- metricas (pico, HPBW, D2D);
- painel tecnico com dados da composicao;
- tabela de pesos por nivel (amplitude/potencia/fase/atenuacao/comprimento eletrico);
- exportacao:
  - PAT composto;
  - PRN composto (usando HRP carregado na aba Arquivo);
  - harness (`csv`, `json`, grafico comparativo inicial vs final).

## 3.3 Aba `Composicao Horizontal`

Objetivo: compor HRP de array de paineis no plano horizontal.

Entradas:

- numero de paineis;
- frequencia + unidade;
- beta progressivo;
- nivel (amplitude);
- espacamento fisico;
- passo angular entre paineis (`DeltaPhi`);
- normalizacao.

Saidas:

- diagrama horizontal composto (plot polar);
- metricas (pico, HPBW, D2D);
- exportacao:
  - PAT composto;
  - PAT RFS;
  - PRN composto (usando VRP carregado da aba Arquivo).

## 3.4 Aba `Estudo Completo`

Objetivo: centralizar todos os diagramas em estudo.

Modos:

- `simples`: H1 + V1;
- `duplo`: H1 + V1 + H2 + V2 (duas polarizacoes).

Cada slot (H1, V1, H2, V2) permite:

- carregar arquivo direto;
- usar dados da aba Arquivo;
- usar dados da Composicao;
- limpar slot.

Visualizacao:

- azimute em polar para slots `H*`;
- elevacao em planar para slots `V*`.

Exportacao em lote (`Exportar Estudo Completo`):

- PAT por slot;
- PAT ADT por slot;
- imagem do diagrama por slot (com metricas);
- tabela CSV e tabela PNG por slot;
- PRN por polarizacao (POL1 e opcionalmente POL2 quando modo duplo).

## 3.5 Aba `Dados do Projeto`

Objetivo: governanca do projeto e exportacoes globais.

Funcoes:

- atualizar painel consolidado do projeto;
- salvar progresso (`*.eftxproj.json`);
- carregar progresso;
- exportar conjunto de arquivos ja exportados (bundle com manifesto JSON/CSV);
- exportacoes globais:
  - todos os graficos PNG;
  - todas as tabelas PNG;
  - todos PAT;
  - todos PAT ADT;
  - todos PRN.

Painel mostra:

- estado das entradas/composicoes/estudo;
- parametros chave;
- contagem e status das exportacoes;
- ultimas exportacoes registradas.

## 3.6 Aba `Diagramas (Batch)`

Objetivo: biblioteca local de diagramas para reuso.

Funcoes:

- importar arquivos em lote;
- parse automatico robusto (PRN/PAT/CSV/TXT etc);
- armazenamento SQLite;
- visualizacao em miniaturas com metricas basicas;
- clique em miniatura para carregar de volta na aba Arquivo.

## 4. Formatos suportados

## 4.1 Entrada

- `.csv`, `.tsv`, `.txt`, `.dat`: leitura via `parse_hfss_csv` / `parse_generic_table` / `parse_auto`.
- `.pat`: leitura por parser robusto e parser RFS ADT (`voltage`).
- `.prn`: leitura por `parse_prn` com conversao para linear normalizado.

## 4.2 Saida

- PAT padrao (horizontal/vertical).
- PAT ADT (RFS voltage).
- PRN (HORIZONTAL 360 + VERTICAL 360 em atenuacao dB).
- Imagens PNG de graficos.
- Imagens PNG de tabelas.
- CSV de tabelas (estudo completo).
- CSV/JSON/PNG de harness (null fill vertical).

## 5. Regras de reamostragem e eixos

## 5.1 Horizontal (HRP)

- referencia interna: `-180 .. 180` graus.
- exportacao HRP:
  - reamostragem passo `1 deg` para PAT/ADT;
  - interpolacao periodica (360 deg).

## 5.2 Vertical (VRP)

- referencia interna: `-90 .. 90` graus.
- exportacao VRP ADT:
  - faixa fixa `-90 .. 90`;
  - passo `0.1 deg`.

## 5.3 PAT ADT

Cabecalho ADT gerado:

1. `Edited by Deep3`
2. `98`
3. `1`
4. `0 0 0 1 0`
5. `voltage`

Corpo:

- colunas: `angulo`, `valor_linear_normalizado`, `0`.
- HRP: forca faixa `-180 .. 180`.
- VRP: forca faixa `-90 .. 90`.

## 6. Logica de composicao vertical e null fill

Modulo: `null_fill_synthesis.py`
Funcao principal usada na UI: `synth_null_fill_by_order(...)`.

Conceito implementado:

- preenchimento por ordem de nulo (1o nulo, 2o nulo, etc), nao por "banda em graus" como conceito principal de uso da aba;
- `null_fill_percent` em porcentagem;
- alvo por lado (esquerda/direita) calculado relativo ao pico:
  - `target_abs = amp_null_inicial + pct * (peak - amp_null_inicial)`;
- resultado reportado:
  - `achieved_percent = (amp_final - amp_inicial) / (peak - amp_inicial) * 100`.

Protecoes de qualidade:

- preservacao do lobo principal com pesos de restricao dedicados;
- trava do pico principal para evitar deslocamento excessivo;
- limitacao automatica de fase em modo `both` quando necessario;
- regularizacao Tikhonov para estabilidade numerica;
- projecao por modo (`amplitude`, `phase`, `both`).

Saidas tecnicas da sintese:

- pesos complexos `w`;
- campo inicial/final;
- regioes de nulo consideradas;
- niveis inicial/alvo/final por nulo;
- percentual alvo e percentual atingido;
- numero de condicao;
- pico em elevacao.

Harness (`weights_to_harness`):

- `amp`, `p_frac`, `phase_deg`, `att_db_ref`, `delta_len_m`;
- estimativa de comprimento de onda no cabo via `vf`.

## 7. Logica de composicao horizontal

Implementacao principal: `compute_horizontal_panels()`.

Modelo:

- paineis posicionados em poligono regular;
- raio definido para manter distancia `s` entre adjacentes;
- soma complexa das contribuicoes com:
  - fase geometrica por diferenca de caminho;
  - fase de excitacao progressiva (`beta`);
  - diagrama elementar interpolado por angulo relativo.

Pos-processamento:

- magnitude normalizada;
- metricas (pico, HPBW, D2D);
- plot polar.

## 8. Metricas de diagrama

Funcoes relevantes em `deep3.py`:

- `hpbw_deg(...)`
- `directivity_2d_cut(...)`
- `compute_diagram_metrics(...)`
- `_first_null_db(...)`

Metricas exibidas/exportadas:

- pico (dB e angulo);
- HPBW;
- D2D linear e dB;
- primeiro nulo (dB);
- F/B (para HRP);
- pontos, faixa angular e passo estimado.

As imagens exportadas usam layout com painel lateral de metricas para evitar sobreposicao com o grafico.

## 9. Persistencia do projeto

Arquivo de progresso: `*.eftxproj.json`

Campos principais serializados:

- `output_dir`
- `string_vars` (parametros de UI)
- arrays de dados:
  - arquivo, composicoes e estudo completo (inclui H2/V2)
- `study_sources`
- `export_registry`

Compatibilidade:

- estado antigo com `vert_null_fill_db` e convertido para `vert_null_fill_pct`.

## 10. Registro e pacote de exportacoes

`export_registry` guarda:

- timestamp;
- tipo (`kind`);
- caminho absoluto do arquivo.

`export_recorded_files_bundle()`:

- copia arquivos registrados para pasta `exportados_YYYYMMDD_HHMMSS`;
- gera manifestos:
  - `manifesto_exportacoes.json`
  - `manifesto_exportacoes.csv`
- marca itens ausentes quando arquivo nao existe mais.

## 11. Banco local de biblioteca (SQLite)

Local:

- `~/.eftx_converter/library.db`

Tabela `diagrams`:

- `id`
- `name`
- `type`
- `angles` (JSON texto)
- `values_json` (JSON texto)
- `meta` (JSON texto)
- `thumbnail_path`
- `added_at`

Comportamento:

- se nao existir DB do usuario, app tenta copiar `library.db` do diretorio de execucao;
- leitura em grade responsiva com miniaturas.

## 12. Fluxo de exportacao por aba

## 12.1 Arquivo

- exportacao individual PAT/PAT ADT por VRP ou HRP;
- exportacao de imagem de grafico e tabela;
- exportacao global:
  - `ALL_VRP_PAT`
  - `ALL_HRP_PAT`
  - `ALL_PRN`

## 12.2 Composicao Vertical

- `VERT_COMP_PAT`
- `VERT_COMP_PRN`
- `VERT_NULLFILL_HARNESS_CSV`
- `VERT_NULLFILL_HARNESS_JSON`
- `VERT_NULLFILL_AF_IMG`

## 12.3 Composicao Horizontal

- `HORZ_COMP_PAT`
- `HORZ_COMP_RFS_PAT`
- `HORZ_COMP_PRN`

## 12.4 Estudo Completo

Por slot (H1/V1/H2/V2 ativo):

- PAT
- PAT ADT
- DIAGRAMA PNG
- TABELA CSV
- TABELA PNG

Por polarizacao valida:

- `STUDY_POL1_PRN`
- `STUDY_POL2_PRN` (modo duplo).

## 12.5 Dados do Projeto

Exportacao consolidada por categoria:

- `graficos`
- `tabelas`
- `pat`
- `pat_adt`
- `prn`

## 13. Dependencias

Definidas em `requirements.txt`:

- `numpy`
- `matplotlib`
- `pillow`
- `customtkinter`
- utilitarios de parsing e PDF (`pypdf`, `PyPDF2`, `reportlab`, etc).

## 14. Build e instalacao MSI

Script: `setup.py`
Ferramenta: `cx_Freeze`

Configuracao principal:

- executavel:
  - origem: `deep3.py`
  - nome alvo: `EFTX_Converter.exe`
  - icone: `eftx-ico.ico`
- includes:
  - logo, icone, banco inicial, licencas, pasta do `customtkinter`.
- MSI:
  - atalhos Desktop/Menu;
  - destino padrao:
    - `[ProgramFiles64Folder]\\EFTX Broadcast\\Antenna Converter`
  - `upgrade_code` definido.

Comandos comuns:

```powershell
pip install -r requirements.txt
pip install cx_Freeze
python setup.py build
python setup.py bdist_msi
```

Saidas esperadas:

- executavel em `build\...`
- instalador `.msi` em `dist\`.

## 15. Testes

Arquivo de testes:

- `tests/test_null_fill_synthesis.py`

Cobertura principal:

- modos `amplitude`/`phase`/`both`;
- normalizacao dos pesos;
- robustez numerica com regularizacao;
- consistencia de `weights_to_harness`;
- efetividade de null fill por ordem de nulo e estabilidade do pico principal.

Execucao:

```powershell
pytest -q
```

## 16. Observacoes tecnicas e pontos de atencao

- O codigo possui duas definicoes de `write_prn_file` em `deep3.py`; a segunda definicao sobrescreve a primeira em runtime.
- Existem strings antigas com problemas de encoding em partes da UI, mas a logica funcional principal esta ativa.
- O ADT esta configurado com eixo/range fixos:
  - HRP `-180..180`
  - VRP `-90..90`
- A aba de Estudo Completo usa azimute em polar por padrao nos slots `H*`.

## 17. Resumo executivo

A aplicacao esta estruturada para uso profissional de engenharia de diagramas:

- ingestao robusta multi-formato;
- composicao vertical/horizontal com metricas;
- null fill por ordem de nulo com controle em porcentagem e preservacao de lobo principal;
- estudo completo com suporte a dupla polarizacao;
- exportacao tecnica completa (PAT, PAT ADT, PRN, tabelas, imagens);
- persistencia de projeto e biblioteca local para fluxo de trabalho continuo.

