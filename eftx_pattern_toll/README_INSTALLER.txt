EFTX Diagram Suite - Installer Notes
====================================

This project ships a professional Windows Installer (.MSI) using:
- PyInstaller (application bundle)
- WiX Toolset v6 (MSI UI + maintenance + upgrade)

Build pipeline
--------------
1. Build app bundle (onedir):
   - `installer/scripts/build_app.ps1`
2. Harvest bundle files to WiX components:
   - `installer/scripts/harvest_files.ps1`
3. Build MSI:
   - `installer/scripts/build_msi.ps1`

One-shot command
----------------
- `build_msi.ps1` (delegates to `installer/scripts/build_msi.ps1`)

Installer UX
------------
- Welcome
- License (Licenca GPT / GPL)
- Install directory selection
- Feature tree (Start Menu shortcut ON, Desktop shortcut optional)
- Progress and Finish
- Single-file MSI (embedded CABs, no external `cab1.cab/cab2.cab`)

Maintenance and upgrades
------------------------
- Maintenance mode is supported automatically (Modify/Repair/Remove).
- Major upgrade is enabled with fixed UpgradeCode.

Support
-------
- Project: https://github.com/Gecesars/antenna_prn_pat_generator
- Author: Gecesars
