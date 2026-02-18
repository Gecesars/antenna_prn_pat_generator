param(
    [switch]$Clean
)

$ErrorActionPreference = "Stop"

& ".\installer\scripts\build_msi.ps1" -Clean:$Clean
