param(
    [switch]$WithExternalAedt,
    [switch]$WithMechanical
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$venvPy = Join-Path $root "venv\Scripts\python.exe"
if (!(Test-Path $venvPy)) {
    throw "Python do venv nao encontrado em: $venvPy"
}

Push-Location $root
try {
    Write-Host "[EFTX] Quality Gate: compileall" -ForegroundColor Cyan
    & $venvPy -m compileall core eftx_aedt_live reports deep3.py

    Write-Host "[EFTX] Quality Gate: pytest (full)" -ForegroundColor Cyan
    & $venvPy -m pytest -q --maxfail=1

    Write-Host "[EFTX] Quality Gate: coverage core (tests/)" -ForegroundColor Cyan
    & $venvPy -m pytest --cov=core --cov-report=term-missing -q tests --maxfail=1

    Write-Host "[EFTX] Quality Gate: docs auto-check" -ForegroundColor Cyan
    & $venvPy (Join-Path $root "tools\generate_audit_docs.py") --check

    Write-Host "[EFTX] Quality Gate: mechanical doctor" -ForegroundColor Cyan
    & $venvPy -m mech.mechanical_doctor --json (Join-Path $root "out\mechanical_doctor_report.json")
    if ($LASTEXITCODE -ne 0) {
        throw "Mechanical doctor failed (exit code $LASTEXITCODE)."
    }

    if ($WithMechanical.IsPresent) {
        Write-Host "[EFTX] Quality Gate: pytest mechanical marker" -ForegroundColor Cyan
        & $venvPy -m pytest -q -m mechanical --maxfail=1
        if ($LASTEXITCODE -ne 0) {
            throw "pytest -m mechanical failed (exit code $LASTEXITCODE)."
        }
    }

    if ($WithExternalAedt.IsPresent) {
        Write-Host "[EFTX] Quality Gate: external AEDT (auto)" -ForegroundColor Cyan
        powershell -ExecutionPolicy Bypass -File (Join-Path $root "scripts\run_external_auto.ps1")
    }
}
finally {
    Pop-Location
}
