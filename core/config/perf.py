from __future__ import annotations

import os


def _env_bool(name: str, default: bool) -> bool:
    raw = str(os.getenv(name, "")).strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return bool(default)


USE_NUMBA = _env_bool("EFTX_USE_NUMBA", True)
NUMBA_FASTMATH = _env_bool("EFTX_NUMBA_FASTMATH", False)
NUMBA_PARALLEL = _env_bool("EFTX_NUMBA_PARALLEL", False)
NUMBA_DTYPE = str(os.getenv("EFTX_NUMBA_DTYPE", "float64") or "float64").strip().lower()
if NUMBA_DTYPE not in {"float64", "float32"}:
    NUMBA_DTYPE = "float64"

