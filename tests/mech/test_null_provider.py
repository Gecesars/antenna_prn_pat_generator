from __future__ import annotations

import pytest

from mech.mechanical.models import MechanicalError
from mech.mechanical.providers.null_provider import NullMechanicalProvider


pytestmark = pytest.mark.mechanical


def test_null_provider_reports_capabilities():
    provider = NullMechanicalProvider(reason="missing freecad")
    caps = provider.capabilities.to_dict()

    assert caps["provider"] == "null"
    assert caps["freecad_available"] is False
    assert caps["freecad_headless_available"] is False


def test_null_provider_raises_controlled_error_on_create():
    provider = NullMechanicalProvider(reason="missing freecad")
    with pytest.raises(MechanicalError) as exc:
        provider.create_primitive("box", {})

    assert exc.value.code == "backend_unavailable"
    assert "create_primitive" in str(exc.value)


def test_null_provider_validate_and_heal_are_non_crashing():
    provider = NullMechanicalProvider()

    validate = provider.validate("any")
    heal = provider.heal("any")

    assert validate["ok"] is False
    assert heal["ok"] is False
