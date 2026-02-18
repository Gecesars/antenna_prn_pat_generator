# Instalador profissional (.MSI) — Instruções para o agente

> **Objetivo:** gerar um instalador **Windows Installer (.MSI)** completo e profissional para a aplicação (desktop), com **modo de manutenção** (Modificar/Reparar/Remover), seleção de **pasta de instalação**, criação opcional de **atalhos** (Menu Iniciar e Área de Trabalho), tela de **licença** (“Licença GPT”), e **upgrade** automático quando houver versão mais nova.

---

## 1) Estratégia recomendada (2 etapas)

### Etapa A — Empacotar a aplicação em binários distribuíveis
A aplicação deve ser empacotada como executável/bundle (sem depender do Python instalado no PC do usuário).

**Recomendado (Python desktop):** `PyInstaller --onedir` (mais previsível para MSI)  
- Saída: `dist/<AppName>/...` com `.exe` + DLLs + assets.

> Evite `--onefile` em MSI (AV false-positive e manutenção/upgrade piores).

### Etapa B — Criar o MSI com **WiX Toolset**
Use **WiX Toolset v4** (preferencial) para construir o MSI (UI, features, upgrade, atalhos, etc.).

---

## 2) Requisitos no ambiente do agente (Windows)

1) Instalar **WiX Toolset v4**  
2) Instalar **.NET SDK** (para build do WiX v4)  
3) Instalar **Python** (somente para gerar o build PyInstaller)  
4) Instalar **PyInstaller** no venv da aplicação  
5) (Opcional, mas profissional) Instalar **signtool** (Windows SDK) para assinar `.exe` e `.msi`

### Verificações (executar e registrar)
```powershell
wix --version
dotnet --version
python --version
pyinstaller --version
```

---

## 3) Convenções obrigatórias (Windows Installer)

### 3.1 Detectar instalação existente (Modify/Repair/Remove)
Para o MSI suportar “já existe → Modificar/Reparar/Remover” automaticamente:

- Definir um **UpgradeCode** FIXO (GUID) para o produto e **não mudar nunca**.
- A cada release, aumentar **Version** (recomendado: `MAJOR.MINOR.BUILD`).
- Configurar **MajorUpgrade**.

✅ Resultado: se já estiver instalado, o MSI abre em **Maintenance Mode** (Modify/Repair/Remove), sem lógica manual extra.

---

## 4) Metadados do produto (preencher com valores reais)

Centralizar no `installer/wix/Product.wxs`:

- **ProductName**: nome real do produto
- **Manufacturer**: EFTX (ou razão social)
- **Version**: `1.0.0` (atualizar a cada release)
- **UpgradeCode**: GUID fixo (gerar uma vez)
- **ExeName**: `.exe` principal
- **AppId**: identificador estável (ex.: `EFTX.DiagramSuite`)

### Gerar GUID (PowerShell)
```powershell
[guid]::NewGuid().ToString()
```

---

## 5) Estrutura de pastas proposta

```
repo/
  app/
  assets/
  dist/                      # saída PyInstaller
  installer/
    wix/
      Product.wxs
      Components.wxs
      license.rtf             # "Licença GPT"
      icon.ico
    scripts/
      build_app.ps1
      harvest_files.ps1
      build_msi.ps1
      sign.ps1                # opcional
  out/                        # MSI final
```

---

## 6) Build do executável (PyInstaller)

Criar `installer/scripts/build_app.ps1`:

```powershell
$ErrorActionPreference = "Stop"

# Ativar venv do projeto (ajuste se o venv tiver outro nome)
.\.venv\Scripts\Activate.ps1

Remove-Item -Recurse -Force .\build, .\dist -ErrorAction SilentlyContinue

pyinstaller `
  --noconfirm `
  --onedir `
  --name "EFTX_DiagramSuite" `
  --icon ".\installer\wix\icon.ico" `
  --add-data "assets;assets" `
  .\app\main.py

Write-Host "Build gerado em dist\EFTX_DiagramSuite"
```

> Se a aplicação for PySide6/Qt, incluir plugins (platforms/styles) explicitamente no spec.  
> Se for Tk/CTk, incluir temas/imagens.

---

## 7) Gerar Components.wxs (harvest do diretório dist)

Criar `installer/scripts/harvest_files.ps1`:

```powershell
$ErrorActionPreference = "Stop"

$dist = Resolve-Path ".\dist\EFTX_DiagramSuite"
$out  = ".\installer\wix\Components.wxs"

wix harvest dir $dist `
  -o $out `
  -dr INSTALLFOLDER `
  -cg AppComponents `
  -var var.SourceDir `
  -gg `
  -srd `
  -sreg `
  -sfrag

Write-Host "Components.wxs atualizado: $out"
```

**Notas:**
- `-var var.SourceDir` evita caminhos hardcoded.
- `-gg` gera GUIDs de componentes (ok para harvest).
- `-dr INSTALLFOLDER` deve bater com o diretório de instalação.

---

## 8) MSI com UI completa (Licença + Pasta + Atalhos)

### Requisitos de UI
1) Welcome  
2) License (aceitar/recusar)  
3) InstallDir (local de instalação)  
4) Seleção de recursos (features) para:
   - Atalho Menu Iniciar (default ON)
   - Atalho Desktop (default OFF)
5) Progresso  
6) Finish  

✅ Implementação recomendada: **WixUI_FeatureTree** + `WixUILicenseRtf` + `WIXUI_INSTALLDIR`.

---

## 9) Product.wxs (modelo base)

Criar/editar `installer/wix/Product.wxs` e ajustar nomes reais.

```xml
<?xml version="1.0" encoding="utf-8"?>
<Wix xmlns="http://wixtoolset.org/schemas/v4/wxs">

  <Package
    Name="EFTX Diagram Suite"
    Manufacturer="EFTX"
    Version="1.0.0"
    UpgradeCode="PUT-GUID-HERE"
    InstallerVersion="500"
    Compressed="yes"
    InstallScope="perMachine" />

  <!-- upgrade / manutenção -->
  <MajorUpgrade DowngradeErrorMessage="Uma versão mais recente já está instalada." />

  <!-- ícone no ARP -->
  <Icon Id="AppIcon" SourceFile="icon.ico" />
  <Property Id="ARPPRODUCTICON" Value="AppIcon" />

  <!-- licença RTF -->
  <WixVariable Id="WixUILicenseRtf" Value="license.rtf" />

  <!-- diretórios -->
  <StandardDirectory Id="ProgramFilesFolder">
    <Directory Id="INSTALLROOT" Name="EFTX">
      <Directory Id="INSTALLFOLDER" Name="DiagramSuite" />
    </Directory>
  </StandardDirectory>

  <StandardDirectory Id="ProgramMenuFolder">
    <Directory Id="AppProgramMenuFolder" Name="EFTX Diagram Suite" />
  </StandardDirectory>

  <StandardDirectory Id="DesktopFolder" />

  <!-- Importa os arquivos do app (harvest) -->
  <?include Components.wxs ?>

  <!-- Atalhos como componentes separados (para features opcionais) -->
  <Fragment>
    <DirectoryRef Id="INSTALLFOLDER">

      <!-- Menu Iniciar -->
      <Component Id="CmpShortcutStartMenu" Guid="*">
        <Shortcut
          Id="StartMenuShortcut"
          Directory="AppProgramMenuFolder"
          Name="EFTX Diagram Suite"
          Description="EFTX Diagram Suite"
          Target="[INSTALLFOLDER]EFTX_DiagramSuite.exe"
          WorkingDirectory="INSTALLFOLDER" />
        <RemoveFolder Id="RemoveAppProgramMenuFolder" Directory="AppProgramMenuFolder" On="uninstall" />
        <RegistryValue Root="HKLM" Key="Software\EFTX\DiagramSuite" Name="StartMenuShortcut" Type="integer" Value="1" KeyPath="yes" />
      </Component>

      <!-- Desktop -->
      <Component Id="CmpShortcutDesktop" Guid="*">
        <Shortcut
          Id="DesktopShortcut"
          Directory="DesktopFolder"
          Name="EFTX Diagram Suite"
          Description="EFTX Diagram Suite"
          Target="[INSTALLFOLDER]EFTX_DiagramSuite.exe"
          WorkingDirectory="INSTALLFOLDER" />
        <RegistryValue Root="HKLM" Key="Software\EFTX\DiagramSuite" Name="DesktopShortcut" Type="integer" Value="1" KeyPath="yes" />
      </Component>

    </DirectoryRef>
  </Fragment>

  <!-- Features -->
  <Feature Id="FeatMain" Title="Aplicação" Level="1" Absent="disallow">
    <ComponentGroupRef Id="AppComponents" />
  </Feature>

  <Feature Id="FeatStartMenu" Title="Atalho no Menu Iniciar" Level="1">
    <ComponentRef Id="CmpShortcutStartMenu" />
  </Feature>

  <Feature Id="FeatDesktop" Title="Atalho na Área de Trabalho" Level="1">
    <ComponentRef Id="CmpShortcutDesktop" />
  </Feature>

  <!-- UI -->
  <Property Id="WIXUI_INSTALLDIR" Value="INSTALLFOLDER" />
  <UI>
    <UIRef Id="WixUI_FeatureTree" />
    <UIRef Id="WixUI_ErrorProgressText" />
  </UI>

</Wix>
```

**Como isso atende o pedido:**
- “Se já existe → modificar ou excluir”: o MSI entra em Maintenance Mode automaticamente.
- “Perguntar local”: InstallDirDlg (via WIXUI_INSTALLDIR).
- “Menu iniciar e desktop”: features selecionáveis.
- “Licença GPT”: license.rtf obrigatório (aceitar/recusar).

---

## 10) Licença GPT (arquivo `license.rtf`)

Criar `installer/wix/license.rtf` (texto real).  
A UI deve bloquear instalação se o usuário não aceitar.

---

## 11) Build do MSI (script)

Criar `installer/scripts/build_msi.ps1`:

```powershell
$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Force .\out | Out-Null

Push-Location .\installer\wix

wix build Product.wxs `
  -arch x64 `
  -o ..\..\out\EFTX_DiagramSuite.msi `
  -d SourceDir="..\..\dist\EFTX_DiagramSuite" `
  -ext WixToolset.UI.wixext

Pop-Location

Write-Host "MSI gerado em out\EFTX_DiagramSuite.msi"
```

---

## 12) Assinatura (opcional, mas profissional)

Se houver certificado:
- assinar o `.msi` com `signtool`.

Exemplo `installer/scripts/sign.ps1`:
```powershell
$ErrorActionPreference = "Stop"
$msi = ".\out\EFTX_DiagramSuite.msi"
signtool sign /fd SHA256 /tr http://timestamp.digicert.com /td SHA256 /a $msi
```

---

## 13) Testes obrigatórios (o agente deve executar de verdade)

### 13.1 Instalação limpa
```powershell
msiexec /i .\out\EFTX_DiagramSuite.msi /l*v install.log
```
Validar:
- pasta escolhida
- app abre
- atalhos conforme escolha

### 13.2 Manutenção
Executar o MSI novamente e confirmar:
- Modify
- Repair
- Remove

### 13.3 Upgrade real
1) Instalar v1.0.0  
2) Gerar v1.1.0 (mesmo UpgradeCode)  
3) Instalar v1.1.0 por cima  
Validar:
- atualiza sem duplicar atalhos
- remove arquivos antigos necessários

### 13.4 Desinstalação
Validar remoção limpa (atalhos + pastas).

---

## 14) “Definition of Done”

- MSI x64
- UI completa: Welcome → License → InstallDir → FeatureTree → Install → Finish
- Maintenance Mode funcionando (Modify/Repair/Remove)
- Upgrade funcionando (MajorUpgrade + UpgradeCode fixo)
- Atalhos opcionais (Start Menu e Desktop)
- Scripts reproduzíveis (build_app + harvest + build_msi)
- Logs e testes executados

