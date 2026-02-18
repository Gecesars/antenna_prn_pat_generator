# EFTX Pattern Tool - C++ Port

Este diretorio contem a conversao em C++ do nucleo de processamento do aplicativo.

## O que ja foi convertido

- Parsing de diagramas:
  - `parse_hfss_csv`
  - `parse_generic_table`
  - `parse_auto`
- Reamostragem:
  - Vertical `-90..90` com passo `0.1`
  - Horizontal `-180..180` com passo `1`
  - Fallback ADT para VRP
- Metricas:
  - `HPBW`
  - `D2D`
- Exportacao:
  - PAT horizontal
  - PAT vertical
  - PAT combinado (estilo EDX)
  - PRN
  - PAT ADT por corte
  - CSV de tabela por corte
  - Dashboard HTML com identidade visual profissional + diagramas SVG
    - AZ em polar
    - EL em planar
- Projeto:
  - Leitura de `.eftxproj.json`
  - Exportacao em lote (`project-export`) com artefatos POL1/POL2
- CLI executavel:
  - `eftx_pattern_tool_cpp`

## Estrutura

- `include/eftx/*.hpp`: API publica
- `src/*.cpp`: implementacao
- `CMakeLists.txt`: build e packaging base

## Build

```bash
cmake -S . -B build_cpp
cmake --build build_cpp --config Release
```

Binario:

- `build_cpp/Release/eftx_pattern_tool_cpp.exe`

Build automatizado (PowerShell):

```powershell
.\build_cpp.ps1 -Config Release
```

## Comandos CLI

```bash
eftx_pattern_tool_cpp metrics --in HRP_HRP.pat --kind H
eftx_pattern_tool_cpp resample --in HRP_HRP.pat --kind H --out hrp_resampled.txt --norm max
eftx_pattern_tool_cpp export-pat-h --in HRP_HRP.pat --out hrp_cpp.pat --desc HRP_CPP
eftx_pattern_tool_cpp export-pat-v --in painel_VRP.pat --out vrp_cpp.pat --desc VRP_CPP
eftx_pattern_tool_cpp export-pat-combined --h HRP_HRP.pat --v painel_VRP.pat --out combined_cpp.pat
eftx_pattern_tool_cpp export-prn --h HRP_HRP.pat --v painel_VRP.pat --out antenna_cpp.prn
eftx_pattern_tool_cpp project-export --project DATA/novbo.eftxproj.json --out out_cpp
```

Saidas do `project-export`:

- PAT convencional (AZ/EL por polarizacao)
- PAT ADT (AZ/EL por polarizacao)
- CSV de tabelas (AZ/EL por polarizacao)
- PRN por polarizacao
- Diagramas SVG por polarizacao
- Dashboard HTML consolidado

## Proxima etapa para conversao total

1. Portar a UI (atual Python/Tk) para Qt.
2. Portar o modulo AEDT Live para C++ (bridge COM/gRPC).
3. Migrar relatorios PDF para engine C++.
4. Integrar MSI nativo do app C++.
