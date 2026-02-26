from __future__ import annotations

import pytest

from mech.mechanical import build_default_kernel, collect_mechanical_diagnostics


pytestmark = pytest.mark.mechanical


def test_collect_mechanical_diagnostics_has_expected_shape():
    report = collect_mechanical_diagnostics().to_dict()

    assert isinstance(report, dict)
    assert "summary" in report
    assert "freecad" in report
    assert "fem" in report

    freecad = report["freecad"]
    assert "inprocess_available" in freecad
    assert "headless_available" in freecad
    assert "python_abi_match" in freecad
    assert "import_errors" in freecad
    assert isinstance(freecad["import_errors"], dict)


def test_build_default_kernel_always_returns_provider():
    provider, report = build_default_kernel()

    caps = provider.capabilities.to_dict()
    assert isinstance(report, dict)
    assert isinstance(caps, dict)
    assert caps.get("provider") in {"null", "freecad"}
