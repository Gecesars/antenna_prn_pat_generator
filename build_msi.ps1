param(
    [switch]$Clean,
    [int]$Threads = 16
)

$ErrorActionPreference = "Stop"

& ".\installer\scripts\build_msi.ps1" -Clean:$Clean -Threads:$Threads
