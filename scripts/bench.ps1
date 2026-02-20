param()

$ErrorActionPreference = "Stop"
Write-Host "[bench] Running tensor+numba benchmarks..."
$py = Join-Path $PSScriptRoot "..\venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }
& $py -m pytest -q tests/perf --benchmark-only
