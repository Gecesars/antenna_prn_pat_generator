param(
    [switch]$Clean
)

$ErrorActionPreference = "Stop"

$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $repo

$py = Join-Path $repo "venv\Scripts\python.exe"
if (!(Test-Path $py)) {
    throw "Python do venv nao encontrado em: $py"
}

if ($Clean) {
    Remove-Item -Recurse -Force ".\build" -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force ".\dist\EFTX_DiagramSuite" -ErrorAction SilentlyContinue
}

& $py -m PyInstaller `
  --noconfirm `
  --clean `
  --onedir `
  --name "EFTX_DiagramSuite" `
  --icon ".\installer\wix\icon.ico" `
  --add-data "assets;assets" `
  --add-data "plugins;plugins" `
  --add-data "ui;ui" `
  --add-data "core;core" `
  --add-data "eftx_aedt_live;eftx_aedt_live" `
  --add-data "eftx-logo.png;." `
  --add-data "eftx-ico.ico;." `
  --add-data "README.md;." `
  --add-data "README_INSTALLER.txt;." `
  --add-data "LICENSE.txt;." `
  --add-data "LICENSE.rtf;." `
  --add-data "installer\\EULA_README_GPL.rtf;." `
  ".\deep3.py"

Write-Host ""
Write-Host "Build PyInstaller concluido em dist\EFTX_DiagramSuite"

