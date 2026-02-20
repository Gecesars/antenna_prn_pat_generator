param(
    [switch]$SkipBuildApp,
    [switch]$SkipHarvest,
    [switch]$Clean
)

$ErrorActionPreference = "Stop"

$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $repo

$wixExe = ".\installer\tools\wix\wix.exe"
if (!(Test-Path $wixExe)) {
    throw "WiX local nao encontrado em $wixExe. Instale com: dotnet tool install --tool-path .\\installer\\tools\\wix wix"
}

$wixVersion = & $wixExe --version
Write-Host "WiX: $wixVersion"
Write-Host "dotnet: $((& dotnet --version))"
Write-Host "python: $((& python --version))"
Write-Host "pyinstaller: $((& .\venv\Scripts\python.exe -m PyInstaller --version))"

$extensionsText = (& $wixExe extension list | Out-String)
if ($extensionsText -notmatch 'WixToolset\.UI\.wixext\s+6\.0\.2') {
    Write-Host "Instalando extensao WixToolset.UI.wixext..."
    & $wixExe extension add WixToolset.UI.wixext/6.0.2
    if ($LASTEXITCODE -ne 0) {
        throw "Falha ao instalar extensao WixToolset.UI.wixext."
    }
}

if ($Clean) {
    Remove-Item -Recurse -Force ".\out" -ErrorAction SilentlyContinue
}
New-Item -ItemType Directory -Force ".\out" | Out-Null

# Force WiX to use a local writable temp directory.
$wixTmp = (Resolve-Path ".\out").Path
$wixTmpDir = Join-Path $wixTmp "wix_tmp"
New-Item -ItemType Directory -Force $wixTmpDir | Out-Null
$env:TEMP = $wixTmpDir
$env:TMP = $wixTmpDir

if (-not $SkipBuildApp) {
    & ".\installer\scripts\build_app.ps1" -Clean:$Clean
}

if (-not $SkipHarvest) {
    & ".\installer\scripts\harvest_files.ps1"
}

$distDir = Resolve-Path ".\dist\EFTX_DiagramSuite"
$msiOut = ".\out\EFTX_DiagramSuite.msi"

& $wixExe build ".\installer\wix\Product.wxs" ".\installer\wix\Components.wxs" `
  -arch x64 `
  -o $msiOut `
  -d SourceDir="$distDir" `
  -b ".\installer\wix" `
  -ext WixToolset.UI.wixext
$wixExit = $LASTEXITCODE
if ($wixExit -ne 0) {
    throw "Falha no build WiX (exit code $wixExit)."
}

Write-Host ""
Write-Host "MSI gerado em $msiOut"
