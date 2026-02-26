from __future__ import annotations

import importlib.util
import os
import platform
from dataclasses import dataclass
from pathlib import Path

import pytest


def _as_bool(value: str, default: bool = False) -> bool:
    raw = str(value or "").strip().lower()
    if not raw:
        return bool(default)
    return raw in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class ExternalAedtEnv:
    version: str
    project: str
    design: str
    setup_sweep: str
    sphere: str
    expression: str
    freq: str
    connect_mode: str
    non_graphical: bool
    remove_lock: bool


@pytest.fixture(scope="session")
def external_aedt_env() -> ExternalAedtEnv:
    """Load and validate external AEDT test environment.

    Tests in this folder run only when EFTX_RUN_EXTERNAL_AEDT=1.
    """
    if not _as_bool(os.environ.get("EFTX_RUN_EXTERNAL_AEDT", "")):
        pytest.skip("External AEDT tests disabled. Set EFTX_RUN_EXTERNAL_AEDT=1 to enable.")
    if os.name != "nt":
        pytest.skip("External AEDT tests are supported only on Windows hosts.")
    if platform.system().lower() != "windows":
        pytest.skip("External AEDT tests require Windows.")

    has_modern = importlib.util.find_spec("ansys.aedt.core") is not None
    has_legacy = importlib.util.find_spec("pyaedt") is not None
    if (not has_modern) and (not has_legacy):
        pytest.skip("PyAEDT API is not available in this environment.")

    project = str(os.environ.get("EFTX_AEDT_PROJECT", "") or "").strip()
    if project:
        p = Path(project)
        if not p.exists():
            pytest.skip(f"EFTX_AEDT_PROJECT not found: {project}")
        project = str(p)

    return ExternalAedtEnv(
        version=str(os.environ.get("EFTX_AEDT_VERSION", "2025.2") or "2025.2").strip(),
        project=project,
        design=str(os.environ.get("EFTX_AEDT_DESIGN", "") or "").strip(),
        setup_sweep=str(os.environ.get("EFTX_AEDT_SETUP", "") or "").strip(),
        sphere=str(os.environ.get("EFTX_AEDT_SPHERE", "3D_Sphere") or "3D_Sphere").strip(),
        expression=str(os.environ.get("EFTX_AEDT_EXPR", "dB(GainTotal)") or "dB(GainTotal)").strip(),
        freq=str(os.environ.get("EFTX_AEDT_FREQ", "") or "").strip(),
        connect_mode=str(os.environ.get("EFTX_AEDT_CONNECT_MODE", "attach") or "attach").strip().lower(),
        non_graphical=_as_bool(os.environ.get("EFTX_AEDT_NON_GRAPHICAL", "0")),
        remove_lock=_as_bool(os.environ.get("EFTX_AEDT_REMOVE_LOCK", "1"), default=True),
    )

