param(
    [string]$Project = "",
    [string]$Design = "",
    [string]$Setup = "",
    [string]$Sphere = "3D_Sphere",
    [string]$Expr = "dB(GainTotal)",
    [string]$Freq = "",
    [string]$Version = "2025.2",
    [ValidateSet("attach", "new")]
    [string]$ConnectMode = "attach",
    [ValidateSet("0", "1")]
    [string]$NonGraphical = "0",
    [ValidateSet("0", "1")]
    [string]$RemoveLock = "1"
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$venvPy = Join-Path $root "venv\Scripts\python.exe"
if (!(Test-Path $venvPy)) {
    throw "Python do venv nao encontrado em: $venvPy"
}

$env:EFTX_RUN_EXTERNAL_AEDT = "1"
$env:EFTX_AEDT_VERSION = $Version
$env:EFTX_AEDT_PROJECT = $Project
$env:EFTX_AEDT_DESIGN = $Design
$env:EFTX_AEDT_SETUP = $Setup
$env:EFTX_AEDT_SPHERE = $Sphere
$env:EFTX_AEDT_EXPR = $Expr
$env:EFTX_AEDT_FREQ = $Freq
$env:EFTX_AEDT_CONNECT_MODE = $ConnectMode
$env:EFTX_AEDT_NON_GRAPHICAL = $NonGraphical
$env:EFTX_AEDT_REMOVE_LOCK = $RemoveLock

Write-Host "[EFTX] Running external AEDT tests..." -ForegroundColor Cyan
Write-Host "  Project      : $($env:EFTX_AEDT_PROJECT)"
Write-Host "  Design       : $($env:EFTX_AEDT_DESIGN)"
Write-Host "  Setup        : $($env:EFTX_AEDT_SETUP)"
Write-Host "  Sphere       : $($env:EFTX_AEDT_SPHERE)"
Write-Host "  Expression   : $($env:EFTX_AEDT_EXPR)"
Write-Host "  Frequency    : $($env:EFTX_AEDT_FREQ)"
Write-Host "  Version      : $($env:EFTX_AEDT_VERSION)"
Write-Host "  Connect mode : $($env:EFTX_AEDT_CONNECT_MODE)"
Write-Host "  Non-graphical: $($env:EFTX_AEDT_NON_GRAPHICAL)"

Push-Location $root
try {
    & $venvPy -m pytest -q tests/external
}
finally {
    Pop-Location
}

