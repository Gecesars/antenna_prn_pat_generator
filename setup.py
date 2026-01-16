import sys
import os
from cx_Freeze import setup, Executable

# Increase recursion depth just in case
sys.setrecursionlimit(5000)

# Files to include
files = [
    ("eftx-logo.png", "eftx-logo.png"),
    ("eftx-ico.ico", "eftx-ico.ico"),
    ("library.db", "library.db"),
    ("LICENSE.txt", "LICENSE.txt"),
    ("LICENSE.rtf", "LICENSE.rtf")
]

# Add customtkinter data manually if needed
import customtkinter
ctk_path = os.path.dirname(customtkinter.__file__)
files.append((ctk_path, "lib/customtkinter"))

# Shortcut table for MSI
shortcut_table = [
    ("DesktopShortcut",        # Shortcut
     "DesktopFolder",          # Directory_
     "EFTX Converter",         # Name
     "TARGETDIR",              # Component_
     "[TARGETDIR]EFTX_Converter.exe",# Target
     None,                     # Arguments
     "Conversor e Biblioteca de Antenas EFTX", # Description
     None,                     # Hotkey
     None,                     # Icon
     0,                        # IconIndex (0 = Use first icon in target)
     None,                     # ShowCmd
     "TARGETDIR"               # WkDir
     ),
    ("ProgramMenuShortcut",    # Shortcut
     "ProgramMenuFolder",      # Directory_
     "EFTX Converter",         # Name
     "TARGETDIR",              # Component_
     "[TARGETDIR]EFTX_Converter.exe",# Target
     None,                     # Arguments
     "Conversor e Biblioteca de Antenas EFTX", # Description
     None,                     # Hotkey
     None,                     # Icon
     0,                        # IconIndex (0 = Use first icon in target)
     None,                     # ShowCmd
     "TARGETDIR"               # WkDir
     )
]

msi_data = {"Shortcut": shortcut_table}

bdist_msi_options = {
    "data": msi_data,
    "initial_target_dir": r"[ProgramFiles64Folder]\EFTX Broadcast\Antenna Converter",
    "upgrade_code": "{952E3852-5953-4393-9467-336714151614}", # CHANGED GUID for v1.4
}

build_exe_options = {
    "packages": ["os", "sys", "numpy", "PIL", "customtkinter", "matplotlib", "sqlite3", "tkinter"],
    "include_files": files,
    "excludes": [],
    "include_msvcr": True
}

base = None
if sys.platform == "win32":
    base = "Win32GUI"

setup(
    name="EFTX_Antenna_Converter",
    version="1.2",
    description="Conversor e Biblioteca de Diagramas (EFTX)",
    author="Gecesars",
    author_email="gecesars@gmail.com",
    url="https://github.com/Gecesars/antenna_prn_pat_generator",
    options={
        "build_exe": build_exe_options,
        "bdist_msi": bdist_msi_options
    },
    executables=[Executable("deep3.py", 
                            base=base, 
                            target_name="EFTX_Converter.exe", 
                            icon="eftx-ico.ico", 
                            shortcut_name="EFTX Converter",
                            shortcut_dir="DesktopFolder")] 
)
