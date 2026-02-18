param(
    [ValidateSet("Debug", "Release")]
    [string]$Config = "Release"
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$BuildDir = Join-Path $Root "build_$Config"

Get-Process eftx_pattern_tool_cpp -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue

cmake -S $Root -B $BuildDir
if ($LASTEXITCODE -ne 0) { throw "cmake configure failed" }

cmake --build $BuildDir --config $Config
if ($LASTEXITCODE -ne 0) { throw "cmake build failed" }

Push-Location $BuildDir
try {
    cpack -C $Config
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "CPack ZIP failed."
    }
    # Optional MSI via WiX, if installed.
    cpack -G WIX -C $Config
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "CPack WIX (MSI) not available on this machine."
    }
} finally {
    Pop-Location
}

Write-Host "Build finished: $BuildDir"
