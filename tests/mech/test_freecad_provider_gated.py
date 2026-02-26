from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from mech.mechanical.providers.freecad_provider import FreeCADKernelProvider


pytestmark = [pytest.mark.mechanical, pytest.mark.freecad]


def _module_importable(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except Exception:
        return False


HAS_FREECAD = _module_importable("FreeCAD") and _module_importable("Part")


@pytest.mark.skipif(not HAS_FREECAD, reason="FreeCAD/Part modules are not available")
def test_freecad_provider_basic_flow(tmp_path: Path):
    provider = FreeCADKernelProvider()
    try:
        a = provider.create_primitive("box", {"width": 10.0, "depth": 10.0, "height": 10.0, "center": (0.0, 0.0, 0.0)})
        b = provider.create_primitive("box", {"width": 8.0, "depth": 8.0, "height": 8.0, "center": (2.0, 0.0, 0.0)})
        provider.transform(b, {"tx": 1.0, "rz_deg": 5.0})

        c = provider.boolean("union", a, [b])
        tri = provider.triangulate(c, {"deflection": 1.0})

        assert len(tri["vertices"]) > 0
        assert len(tri["faces"]) > 0

        step_path = tmp_path / "shape.step"
        stl_path = tmp_path / "shape.stl"
        provider.export_model([c], str(step_path), fmt="step")
        provider.export_model([c], str(stl_path), fmt="stl")

        assert step_path.exists()
        assert stl_path.exists()

        imported = provider.import_model(str(step_path), fmt="step")
        assert imported
    finally:
        provider.close()
