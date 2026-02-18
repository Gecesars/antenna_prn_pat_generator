"""EFTX AEDT Live Integration (HFSS-focused).

This package is designed as an *add-on* to the existing PAT Converter / Diagram Suite.
It avoids touching existing modules and exposes a small surface:

- AedtHfssSession: connect/disconnect to AEDT/HFSS via PyAEDT
- FarFieldExtractor: robust far-field extraction (2D cuts + 3D grid)
- PatternExport: save outputs to disk (NPZ/CSV/OBJ) and hand off to your app

UI integration is handled by `eftx_aedt_live.ui_tab.AedtLiveTab`.
"""

from .session import AedtHfssSession
from .farfield import FarFieldExtractor, CutRequest, GridRequest, CutResult, GridResult
from .export import PatternExport
