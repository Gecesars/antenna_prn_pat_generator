from __future__ import annotations

import pytest


def test_aedt_live_logger_builder_is_idempotent():
    pytest.importorskip("customtkinter")
    from eftx_aedt_live.ui_tab import _build_aedt_logger

    a = _build_aedt_logger()
    b = _build_aedt_logger()
    assert a is b
    assert a.handlers

