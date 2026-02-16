import os
import sys

from cx_Freeze import Executable, setup


APP_NAME = "EFTX Antenna Converter"
APP_VERSION = "1.3.0"
APP_DESCRIPTION = "Conversor profissional de diagramas de antena (PAT/PRN/ADT)"
APP_COMPANY = "EFTX Broadcast"
APP_AUTHOR = "Gecesars"
APP_URL = "https://github.com/Gecesars/antenna_prn_pat_generator"

EXE_NAME = "EFTX_Converter.exe"
MSI_NAME = "EFTX_Antenna_Converter_1.3.0_win64.msi"

# Keep this constant across releases so MSI upgrades work correctly.
UPGRADE_CODE = "{952E3852-5953-4393-9467-336714151614}"

LICENSE_RTF = os.path.join("installer", "EULA_README_GPL.rtf")


def build_include_files():
    files = [
        ("eftx-logo.png", "eftx-logo.png"),
        ("eftx-ico.ico", "eftx-ico.ico"),
        ("library.db", "library.db"),
        ("assets/templates/eftx_report_template.pdf", "assets/templates/eftx_report_template.pdf"),
        ("README_INSTALLER.txt", "README_INSTALLER.txt"),
        ("README.md", "README.md"),
        ("LICENSE.txt", "LICENSE.txt"),
        ("LICENSE.rtf", "LICENSE.rtf"),
        (LICENSE_RTF, "EULA_README_GPL.rtf"),
    ]

    # Ensure CustomTkinter assets are packaged.
    import customtkinter

    ctk_path = os.path.dirname(customtkinter.__file__)
    files.append((ctk_path, "lib/customtkinter"))
    return files


shortcut_table = [
    (
        "DesktopShortcut",
        "DesktopFolder",
        "EFTX Antenna Converter",
        "TARGETDIR",
        f"[TARGETDIR]{EXE_NAME}",
        None,
        "Abrir EFTX Antenna Converter",
        None,
        None,
        0,
        None,
        "TARGETDIR",
    ),
    (
        "ProgramMenuAppShortcut",
        "ProgramMenuFolder",
        "EFTX Antenna Converter",
        "TARGETDIR",
        f"[TARGETDIR]{EXE_NAME}",
        None,
        "Abrir EFTX Antenna Converter",
        None,
        None,
        0,
        None,
        "TARGETDIR",
    ),
    (
        "ProgramMenuReadmeShortcut",
        "ProgramMenuFolder",
        "EFTX Readme",
        "TARGETDIR",
        "[TARGETDIR]README_INSTALLER.txt",
        None,
        "Abrir README instalado",
        None,
        None,
        0,
        None,
        "TARGETDIR",
    ),
]

msi_data = {"Shortcut": shortcut_table}

build_exe_options = {
    "packages": [
        "os",
        "sys",
        "json",
        "csv",
        "math",
        "sqlite3",
        "logging",
        "threading",
        "concurrent",
        "tkinter",
        "numpy",
        "PIL",
        "customtkinter",
        "matplotlib",
        "core",
        "reports",
        "ui",
        "eftx_aedt_live",
        "plugins",
        "plugins.aedt_live",
        "plugins.aedt_live.aedt_live_plugin",
    ],
    "include_files": build_include_files(),
    "include_msvcr": True,
    "optimize": 1,
    "excludes": [],
}

bdist_msi_options = {
    "data": msi_data,
    "upgrade_code": UPGRADE_CODE,
    "all_users": True,
    "target_name": MSI_NAME,
    "install_icon": "eftx-ico.ico",
    "license_file": LICENSE_RTF,
    "initial_target_dir": r"[ProgramFiles64Folder]\EFTX Broadcast\Antenna Converter",
    "summary_data": {
        "author": f"{APP_AUTHOR} / {APP_COMPANY}",
        "comments": "Professional MSI installer for EFTX Antenna Converter.",
        "keywords": "EFTX, antenna, converter, PAT, PRN, ADT",
    },
}

base = "Win32GUI" if sys.platform == "win32" else None

executables = [
    Executable(
        "deep3.py",
        base=base,
        target_name=EXE_NAME,
        icon="eftx-ico.ico",
    )
]

setup(
    name=APP_NAME,
    version=APP_VERSION,
    description=APP_DESCRIPTION,
    author=APP_AUTHOR,
    url=APP_URL,
    options={"build_exe": build_exe_options, "bdist_msi": bdist_msi_options},
    executables=executables,
)
