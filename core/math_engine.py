from __future__ import annotations

import ast
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from core.angles import ang_dist_deg, wrap_phi_deg


_ALLOWED_FUNCS = {
    "abs": abs,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
    "log10": math.log10,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "atan2": math.atan2,
    "deg2rad": math.radians,
    "rad2deg": math.degrees,
    "wrap_phi": lambda x: float(wrap_phi_deg(float(x))),
    "ang_dist": lambda a, b: float(ang_dist_deg(float(a), float(b))),
}

_ALLOWED_CONSTS = {
    "pi": math.pi,
    "e": math.e,
}

_ALLOWED_NODE_TYPES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Call,
    ast.Name,
    ast.Constant,
    ast.Attribute,
    ast.Load,
    ast.Subscript,
    ast.Index,
    ast.Slice,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.Mod,
    ast.USub,
    ast.UAdd,
)


@dataclass
class MarkerValue:
    name: str
    kind: str = "2D"
    cut: Optional[str] = None
    theta_deg: Optional[float] = None
    phi_deg: Optional[float] = None
    ang_deg: float = 0.0
    mag_lin: float = 0.0
    mag_db: float = -120.0


@dataclass
class MathFunctionDef:
    name: str
    expr: str
    params_schema: Optional[dict] = None
    applies_to: str = "ANY"


def default_math_functions_path() -> str:
    appdata = os.getenv("APPDATA")
    if appdata:
        base = os.path.join(appdata, "EFTX", "PATConverter")
    else:
        base = os.path.join(os.path.expanduser("~"), ".eftx_converter")
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, "marker_math.json")


def _validate_ast(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODE_TYPES):
            raise ValueError(f"Forbidden syntax: {type(node).__name__}")

        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Only direct function calls are allowed.")
            if node.func.id not in _ALLOWED_FUNCS:
                raise ValueError(f"Function not allowed: {node.func.id}")

        if isinstance(node, ast.Attribute):
            if not isinstance(node.value, ast.Name):
                raise ValueError("Nested attributes are not allowed.")
            if node.value.id not in {"A", "B", "pattern"}:
                raise ValueError("Attribute base not allowed.")

        if isinstance(node, ast.Subscript):
            # Only params["key"] is allowed
            if not isinstance(node.value, ast.Name) or node.value.id != "params":
                raise ValueError("Only params[...] access is allowed.")
            key_node = node.slice
            if isinstance(key_node, ast.Index):  # py<3.9 compatibility
                key_node = key_node.value
            if not isinstance(key_node, ast.Constant) or not isinstance(key_node.value, (str, int, float)):
                raise ValueError("params index must be a literal.")



def evaluate_expression(expr: str, context: Optional[dict] = None) -> Any:
    expr = str(expr or "").strip()
    if not expr:
        raise ValueError("Expression is empty.")
    tree = ast.parse(expr, mode="eval")
    _validate_ast(tree)

    env: Dict[str, Any] = {}
    env.update(_ALLOWED_FUNCS)
    env.update(_ALLOWED_CONSTS)

    ctx = dict(context or {})
    env["params"] = dict(ctx.get("params", {}))
    env["A"] = ctx.get("A")
    env["B"] = ctx.get("B")
    if "pattern" in ctx:
        env["pattern"] = ctx.get("pattern")

    code = compile(tree, filename="<marker_math>", mode="eval")
    return eval(code, {"__builtins__": {}}, env)


def default_presets() -> List[MathFunctionDef]:
    return [
        MathFunctionDef(
            name="Delta ang (wrap)",
            expr="ang_dist(A.ang_deg, B.ang_deg)",
            params_schema={},
            applies_to="HRP",
        ),
        MathFunctionDef(
            name="Delta Mag dB",
            expr="B.mag_db - A.mag_db",
            params_schema={},
            applies_to="ANY",
        ),
        MathFunctionDef(
            name="Midpoint phi",
            expr="wrap_phi((A.ang_deg + B.ang_deg)/2)",
            params_schema={},
            applies_to="HRP",
        ),
    ]


def load_user_functions(path: Optional[str] = None) -> List[MathFunctionDef]:
    p = path or default_math_functions_path()
    if not os.path.exists(p):
        presets = default_presets()
        save_user_functions(presets, p)
        return presets

    try:
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            raw = json.load(f)
        out: List[MathFunctionDef] = []
        if isinstance(raw, list):
            for item in raw:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name", "Function"))
                expr = str(item.get("expr", "")).strip()
                if not expr:
                    continue
                out.append(
                    MathFunctionDef(
                        name=name,
                        expr=expr,
                        params_schema=item.get("params_schema", {}),
                        applies_to=str(item.get("applies_to", "ANY")).upper(),
                    )
                )
        return out or default_presets()
    except Exception:
        return default_presets()


def save_user_functions(functions: List[MathFunctionDef], path: Optional[str] = None) -> None:
    p = path or default_math_functions_path()
    os.makedirs(os.path.dirname(p), exist_ok=True)
    payload = [
        {
            "name": f.name,
            "expr": f.expr,
            "params_schema": f.params_schema or {},
            "applies_to": f.applies_to,
        }
        for f in functions
    ]
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def evaluate_functions(
    functions: List[MathFunctionDef],
    A: Optional[MarkerValue],
    B: Optional[MarkerValue],
    params: Optional[dict] = None,
    applies_to: str = "ANY",
) -> List[dict]:
    out: List[dict] = []
    applies_to = str(applies_to or "ANY").upper()
    for fn in functions:
        if fn.applies_to not in ("ANY", applies_to):
            continue
        try:
            value = evaluate_expression(
                fn.expr,
                context={
                    "A": A,
                    "B": B,
                    "params": dict(params or {}),
                },
            )
            out.append({"name": fn.name, "expr": fn.expr, "value": value, "error": ""})
        except Exception as e:
            out.append({"name": fn.name, "expr": fn.expr, "value": None, "error": str(e)})
    return out
