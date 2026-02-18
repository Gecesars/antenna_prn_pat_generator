param(
    [string]$MsiPath = ".\out\EFTX_DiagramSuite.msi"
)

$ErrorActionPreference = "Stop"

if (!(Test-Path $MsiPath)) {
    throw "MSI nao encontrado: $MsiPath"
}

signtool sign /fd SHA256 /tr http://timestamp.digicert.com /td SHA256 /a $MsiPath

Write-Host "Assinatura aplicada em: $MsiPath"

