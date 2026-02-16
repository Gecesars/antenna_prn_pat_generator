param(
    [switch]$Clean
)

$ErrorActionPreference = "Stop"

if ($Clean) {
    if (Test-Path ".\build") { Remove-Item ".\build" -Recurse -Force }
    if (Test-Path ".\dist") { Remove-Item ".\dist" -Recurse -Force }
}

python -m pip install --upgrade pip
python -m pip install --upgrade cx_Freeze

python .\setup.py bdist_msi

Write-Host ""
Write-Host "MSI build finished. Check the .\dist folder."

