$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$venvPy = Join-Path $root "venv\Scripts\python.exe"
if (!(Test-Path $venvPy)) {
    throw "Python do venv nao encontrado em: $venvPy"
}

Push-Location $root
try {
    & $venvPy -m mech.mechanical_doctor --json (Join-Path $root "out\mechanical_doctor_report.json")
    if ($LASTEXITCODE -ne 0) {
        throw "Mechanical doctor failed (exit code $LASTEXITCODE)."
    }
}
finally {
    Pop-Location
}
