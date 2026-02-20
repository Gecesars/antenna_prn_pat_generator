param()

$ErrorActionPreference = "Stop"
Write-Host "[test_cpu] Running tensor+numba tests (CPU)..."
$py = Join-Path $PSScriptRoot "..\venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }
& $py -m pytest -q tests/tensor tests/numba tests/mech -m "not cuda"
