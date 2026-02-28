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
    Remove-Item -Recurse -Force ".\build\Calc_Div_EFTX" -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force ".\dist\Calc_Div_EFTX" -ErrorAction SilentlyContinue
}

& $py -m PyInstaller `
  --noconfirm `
  --clean `
  --onedir `
  --exclude-module "django" `
  --exclude-module "flet" `
  --exclude-module "flet_desktop" `
  --hidden-import "ansys.aedt.core" `
  --hidden-import "ansys.aedt.core.hfss" `
  --hidden-import "ansys.aedt.core.desktop" `
  --hidden-import "ansys.api.edb.v1" `
  --hidden-import "psutil._psutil_windows" `
  --collect-all "ansys.aedt.core" `
  --collect-all "ansys.api.edb" `
  --collect-all "ansys.edb.core" `
  --collect-all "pyedb" `
  --collect-all "pyvista" `
  --collect-all "rfc3987_syntax" `
  --collect-data "jsonschema_specifications" `
  --collect-data "referencing" `
  --collect-submodules "grpc" `
  --collect-submodules "psutil" `
  --collect-binaries "psutil" `
  --name "Calc_Div_EFTX" `
  --icon ".\installer\wix\icon.ico" `
  --add-data "eftx-logo.png;." `
  --add-data "eftx-ico.ico;." `
  --add-data "README.md;." `
  --add-data "README_INSTALLER.txt;." `
  --add-data "LICENSE.txt;." `
  --add-data "LICENSE.rtf;." `
  --add-data "installer\\EULA_README_GPL.rtf;." `
  ".\Calc_Div_EFTX.py"

if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller falhou (exit code $LASTEXITCODE)."
}

Write-Host ""
Write-Host "Build PyInstaller (Calc_Div_EFTX) concluido em dist\Calc_Div_EFTX"
