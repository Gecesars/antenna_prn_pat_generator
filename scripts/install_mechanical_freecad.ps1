param(
    [switch]$InstallFreeCAD,
    [switch]$InstallPytestQt
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$venvPy = Join-Path $root "venv\Scripts\python.exe"
if (!(Test-Path $venvPy)) {
    throw "Python do venv nao encontrado em: $venvPy"
}

Push-Location $root
try {
    Write-Host "[EFTX] Installing mechanical Python dependencies" -ForegroundColor Cyan
    & $venvPy -m pip install --upgrade pip
    & $venvPy -m pip install --upgrade-strategy only-if-needed --upgrade "PySide6>=6.6" "pyvista==0.46.5" "pyvistaqt>=0.11" "numpy==2.2.6"

    if ($InstallPytestQt.IsPresent) {
        Write-Host "[EFTX] Installing pytest-qt" -ForegroundColor Cyan
        & $venvPy -m pip install --upgrade pytest-qt
    }

    if ($InstallFreeCAD.IsPresent) {
        $winget = Get-Command winget -ErrorAction SilentlyContinue
        if ($null -eq $winget) {
            Write-Warning "winget nao encontrado. Instale FreeCAD manualmente: https://www.freecad.org/downloads.php"
        }
        else {
            Write-Host "[EFTX] Installing FreeCAD via winget" -ForegroundColor Cyan
            winget install --id FreeCAD.FreeCAD --exact --accept-source-agreements --accept-package-agreements
        }
    }

    $freecadCmdPath = ""
    $fcCmd = Get-Command FreeCADCmd -ErrorAction SilentlyContinue
    if ($null -ne $fcCmd) {
        $freecadCmdPath = [string]$fcCmd.Source
    }
    if (-not $freecadCmdPath) {
        $candidates = Get-ChildItem -Path "$env:ProgramFiles\\FreeCAD*\\bin\\FreeCADCmd.exe" -ErrorAction SilentlyContinue
        if ($candidates) {
            $freecadCmdPath = [string](@($candidates)[0].FullName)
        }
    }

    if ($freecadCmdPath) {
        try {
            $hostPy = (& $venvPy -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')").Trim()
            $fcPy = (& $freecadCmdPath -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')").Trim()
            if ($hostPy -and $fcPy -and ($hostPy -ne $fcPy)) {
                Write-Warning "Python ABI mismatch detectado: app=$hostPy, FreeCADCmd=$fcPy. In-process FreeCAD pode ficar indisponivel; headless fallback permanece disponivel."
            }
        }
        catch {
            Write-Warning "Falha ao verificar compatibilidade Python do FreeCADCmd: $($_.Exception.Message)"
        }
    }

    Write-Host "[EFTX] Running mechanical doctor check" -ForegroundColor Cyan
    & $venvPy tools\mechanical_doctor.py --json out\mechanical_doctor_report.json

    Write-Host "[EFTX] Mechanical install completed." -ForegroundColor Green
}
finally {
    Pop-Location
}
