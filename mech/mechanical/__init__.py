from __future__ import annotations

from .diagnostics import build_default_kernel, collect_mechanical_diagnostics
from .interfaces import MechanicalKernel
from .models import CapabilityReport, MechanicalDiagnostics, MechanicalError, MeshPayload
from .providers.freecad_provider import FreeCADKernelProvider
from .providers.null_provider import NullMechanicalProvider

__all__ = [
    "MechanicalKernel",
    "MechanicalError",
    "CapabilityReport",
    "MeshPayload",
    "MechanicalDiagnostics",
    "NullMechanicalProvider",
    "FreeCADKernelProvider",
    "collect_mechanical_diagnostics",
    "build_default_kernel",
]
