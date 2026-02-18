$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$sln = Join-Path $root "EFTXPatternStudio.sln"
$vs = "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\devenv.exe"

if (-not (Test-Path $sln)) {
    throw "Solution not found: $sln"
}

if (Test-Path $vs) {
    Start-Process -FilePath $vs -ArgumentList "`"$sln`""
    Write-Host "Visual Studio opened: $sln"
} else {
    Write-Warning "Visual Studio Community not found at expected path."
    Write-Host "Opening solution with shell association..."
    Start-Process -FilePath $sln
}
