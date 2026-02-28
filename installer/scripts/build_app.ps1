param(
    [switch]$Clean,
    [int]$Threads = 16
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

if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller falhou (exit code $LASTEXITCODE)."
}

$threadsSafe = [Math]::Max(1, [Math]::Min(64, [int]$Threads))
$nbSrcRoot = Join-Path $repo "analise_cobertura\Notebook_Cover"
$nbDstRoot = Join-Path $repo "dist\EFTX_DiagramSuite\analise_cobertura\Notebook_Cover"
if (Test-Path $nbSrcRoot) {
    New-Item -ItemType Directory -Force $nbDstRoot | Out-Null
    $runtimeDirs = @("APP", "ANATEL", "MANUAL", "tools")
    foreach ($dir in $runtimeDirs) {
        $srcDir = Join-Path $nbSrcRoot $dir
        if (Test-Path $srcDir) {
            $dstDir = Join-Path $nbDstRoot $dir
            New-Item -ItemType Directory -Force $dstDir | Out-Null
            Write-Host "Copiando Notebook_Cover\$dir para dist..."
            & robocopy $srcDir $dstDir /E /R:1 /W:1 /MT:$threadsSafe /XD "__pycache__" ".pytest_cache" /NFL /NDL /NJH /NJS /NP
            $rc = $LASTEXITCODE
            if ($rc -gt 7) {
                throw "Falha ao copiar Notebook_Cover\$dir para dist (robocopy exit code $rc)."
            }
        }
    }
    foreach ($placeholder in @("DEM", "IBGE", "EXPORTS", "logs")) {
        New-Item -ItemType Directory -Force (Join-Path $nbDstRoot $placeholder) | Out-Null
    }
    Write-Host "Notebook_Cover runtime copiado (APP/ANATEL/MANUAL/tools). DEM/IBGE serao carregados sob demanda."
} else {
    throw "Diretorio Notebook_Cover nao encontrado: $nbSrcRoot"
}

Write-Host ""
Write-Host "Build PyInstaller concluido em dist\EFTX_DiagramSuite"
