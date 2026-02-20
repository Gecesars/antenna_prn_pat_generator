param()

$ErrorActionPreference = "Stop"
Write-Host "[test_cuda] Running tensor tests (CUDA marker)..."
$py = Join-Path $PSScriptRoot "..\venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }
& $py -m pytest -q tests/tensor -m cuda
