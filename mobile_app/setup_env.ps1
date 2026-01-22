# Setup Mobile Environment & Emulator
# Goal: Install Flutter, Android SDK, Create Emulator, Build APK, Run on Emulator
Write-Host "--- EFTX Mobile Automation Setup ---" -ForegroundColor Cyan

# Check for Chocolatey (Package Manager)
if (-not (Get-Command choco -ErrorAction SilentlyContinue)) {
    Write-Host "Installing Chocolatey..."
    Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
}

# 1. Install Flutter SDK
if (-not (Get-Command flutter -ErrorAction SilentlyContinue)) {
    Write-Host "Installing Flutter SDK via Chocolatey..."
    choco install flutter -y
    # Refresh env vars
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
}

# 2. Check Android SDK
# We need 'cmdline-tools' at minimum. 
if (-not $env:ANDROID_HOME) {
    Write-Host "ANDROID_HOME not set. Attempting to install Android Command Line Tools..."
    # This is tricky without Android Studio, but we can try choco
    choco install android-sdk -y
    choco install android-studio -y # Full studio is safer for emulator management
}

# 3. Setup Requirements
Write-Host "Installing pip dependencies..."
pip install -r requirements.txt

# 4. Build APK (if flutter ready)
Write-Host "Checking Flutter Doctor..."
flutter doctor

Write-Host "Building APK..."
flet build apk

# 5. Emulator & Install
# This assumes an emulator exists or tries to launch one.
# List emulators: flutter emulators
$emus = flutter emulators --machine
if ($emus) {
    Write-Host "Launching Emulator..."
    flutter emulators --launch "pixel" # Try generic name or parse list
    
    Write-Host "Installing on Emulator..."
    flet run --android
}
else {
    Write-Host "No emulator found. Please open Android Studio and create a Virtual Device (AVD)."
    Write-Host "Then run this script again."
    # Try to open AVD manager
    # & "C:\Program Files\Android\Android Studio\bin\studio64.exe"
}
