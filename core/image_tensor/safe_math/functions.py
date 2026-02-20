from __future__ import annotations

import math
from typing import Callable, Dict


SAFE_FUNCTIONS: Dict[str, Callable] = {
    "sqrt": math.sqrt,
    "log10": math.log10,
    "log": math.log,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "abs": abs,
    "min": min,
    "max": max,
    "pow": pow,
    "round": round,
}
