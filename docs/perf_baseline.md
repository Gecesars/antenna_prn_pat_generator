# Baseline de Performance

Data: 2026-02-20  
Fonte: `pytest-benchmark` durante `venv\Scripts\python.exe -m pytest --cov=core -q tests --maxfail=1`

## Cenarios medidos

| Benchmark | Mean | Unidade | Observacao |
|---|---:|---|---|
| `test_bench_decimation_numba` | 35.2830 | us | Decimacao de pontos 1D |
| `test_bench_metrics_cut_numba_vs_numpy` | 39.1960 | us | Metricas de corte 1D |
| `test_bench_resize_2k` | 63472.8529 | us | Resize imagem 2k |
| `test_bench_pipeline_export` | 408909.8600 | us | Pipeline de exportacao de imagem |

## Politica de regressao
- Regressao aceitavel: ate **15%** sobre o mean baseline por cenario.
- Acima de 15%: abrir item em `docs/gap_analysis.md` com causa e plano.

## Comando de reproduzibilidade
```powershell
venv\Scripts\python.exe -m pytest -q tests --maxfail=1
```

## Notas
- Valores podem variar com carga do host, I/O e estado de warmup do JIT.
- Para comparacoes consistentes, executar com menos processos em paralelo no host.

