param()

$ErrorActionPreference = "Stop"

$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $repo

$dist = ".\dist\EFTX_DiagramSuite"
if (!(Test-Path $dist)) {
    throw "Diretorio de build nao encontrado: $dist"
}

$out = ".\installer\wix\Components.wxs"
$xml = @'
<?xml version="1.0" encoding="utf-8"?>
<Wix xmlns="http://wixtoolset.org/schemas/v4/wxs">
  <Fragment>
    <ComponentGroup Id="AppComponents" Directory="INSTALLFOLDER">
      <Files Include="$(var.SourceDir)\**" />
    </ComponentGroup>
  </Fragment>
</Wix>
'@

Set-Content -Path $out -Value $xml -Encoding UTF8
Write-Host "Components.wxs atualizado (modo WiX v6): $out"
