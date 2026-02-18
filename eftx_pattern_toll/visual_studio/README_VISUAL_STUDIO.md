# EFTX Pattern Studio (Visual Studio)

Este workspace foi migrado para uso direto no Visual Studio com interface visual (WinForms Designer).

## Solucao

- `eftx_pattern_toll/visual_studio/EFTXPatternStudio.sln`
- Projeto UI: `eftx_pattern_toll/visual_studio/EFTX.PatternStudio`

## O que ja esta implementado

- Interface visual com componentes arrastaveis no designer.
- Selecao de:
- EXE do core C++
- Arquivo de azimute (AZ/HRP)
- Arquivo de elevacao (EL/VRP)
- Pasta de saida
- Nome base e normalizacao (`none|max|rms`)
- Preview dos diagramas:
- AZ em polar
- EL em planar
- Painel de metricas (pico, HPBW, min dB, quantidade de pontos)
- Integracao com o core C++:
- `Build Core` (CMake + Visual Studio 2022)
- `Exportar Tudo` via `project-export`
- Gera PAT/PRN/CSV/SVG/dashboard HTML no output.

## Como abrir no Visual Studio

1. Abra `EFTXPatternStudio.sln`.
2. Defina `EFTX.PatternStudio` como startup project.
3. Rode com `F5`.

## Execucao via CLI

```powershell
cd eftx_pattern_toll\visual_studio\EFTX.PatternStudio
dotnet run
```

