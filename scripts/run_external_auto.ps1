param(
    [string]$Project = "",
    [string]$Design = "",
    [string]$Setup = "",
    [string]$Sphere = "3D_Sphere",
    [string]$Expr = "dB(GainTotal)",
    [string]$Freq = "",
    [string]$Version = "2025.2"
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$venvPy = Join-Path $root "venv\Scripts\python.exe"
if (!(Test-Path $venvPy)) {
    throw "Python do venv nao encontrado em: $venvPy"
}

function Resolve-ProjectPath([string]$explicit) {
    if ($explicit -and (Test-Path $explicit)) { return $explicit }
    if ($env:EFTX_AEDT_PROJECT -and (Test-Path $env:EFTX_AEDT_PROJECT)) { return $env:EFTX_AEDT_PROJECT }

    $candidates = @(
        "D:\Simulations\Simulations\#IFTX\teste_app.aedt",
        "D:\Simulations\Simulations\#IFTX\FM_ANEL_REFLETOR_TORRE.aedt"
    )
    foreach ($c in $candidates) {
        if (Test-Path $c) { return $c }
    }
    return ""
}

$resolvedProject = Resolve-ProjectPath $Project
if (-not $resolvedProject) {
    Write-Host "[EFTX] Nenhum projeto AEDT encontrado automaticamente." -ForegroundColor Yellow
    Write-Host "[EFTX] Informe -Project ou defina EFTX_AEDT_PROJECT para executar testes externos."
    exit 0
}

$env:EFTX_RUN_EXTERNAL_AEDT = "1"
$env:EFTX_AEDT_VERSION = $Version
$env:EFTX_AEDT_PROJECT = $resolvedProject
$env:EFTX_AEDT_DESIGN = $Design
$env:EFTX_AEDT_SETUP = $Setup
$env:EFTX_AEDT_SPHERE = $Sphere
$env:EFTX_AEDT_EXPR = $Expr
$env:EFTX_AEDT_FREQ = $Freq
$env:EFTX_AEDT_CONNECT_MODE = "attach"
$env:EFTX_AEDT_NON_GRAPHICAL = "0"
$env:EFTX_AEDT_REMOVE_LOCK = "1"

Write-Host "[EFTX] Running external AEDT suite (auto mode)..." -ForegroundColor Cyan
Write-Host "  Project: $resolvedProject"
if ($Design) { Write-Host "  Design : $Design" }
if ($Setup) { Write-Host "  Setup  : $Setup" }

Push-Location $root
try {
    & $venvPy -m pytest -q tests/external
}
finally {
    Pop-Location
}

