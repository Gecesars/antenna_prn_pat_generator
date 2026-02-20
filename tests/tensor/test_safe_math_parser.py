from __future__ import annotations

import pytest

from core.image_tensor.safe_math.parser import SafeMathParser


def test_safe_math_accepts_whitelist():
    p = SafeMathParser()
    out = p.evaluate("sqrt(a*a + b*b)", {"a": 3.0, "b": 4.0})
    assert abs(out - 5.0) < 1e-9


def test_safe_math_blocks_unsafe_tokens():
    p = SafeMathParser()
    with pytest.raises(RuntimeError):
        p.evaluate("__import__('os').system('dir')", {})
