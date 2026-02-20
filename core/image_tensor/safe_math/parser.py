from __future__ import annotations

import ast
from typing import Dict

from .functions import SAFE_FUNCTIONS


_ALLOWED = {
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Constant,
    ast.Name,
    ast.Load,
    ast.Call,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.Mod,
    ast.FloorDiv,
    ast.USub,
    ast.UAdd,
    ast.Compare,
    ast.Gt,
    ast.GtE,
    ast.Lt,
    ast.LtE,
    ast.Eq,
    ast.NotEq,
    ast.BoolOp,
    ast.And,
    ast.Or,
    ast.IfExp,
}


class SafeMathParser:
    def __init__(self):
        self.functions = dict(SAFE_FUNCTIONS)

    def parse(self, expr: str) -> ast.AST:
        text = str(expr or "").strip()
        if not text:
            raise RuntimeError("Empty expression.")
        for token in ("__", "import", "lambda", "open", "exec", "eval", "class", "globals", "locals"):
            if token in text:
                raise RuntimeError(f"Forbidden token: {token}")
        tree = ast.parse(text, mode="eval")
        for node in ast.walk(tree):
            if type(node) not in _ALLOWED:
                raise RuntimeError(f"Unsupported token: {type(node).__name__}")
            if isinstance(node, ast.Call):
                if not isinstance(node.func, ast.Name):
                    raise RuntimeError("Only direct function calls are allowed.")
                if node.func.id not in self.functions:
                    raise RuntimeError(f"Function not allowed: {node.func.id}")
        return tree

    def evaluate(self, expr: str, variables: Dict[str, float]) -> float:
        tree = self.parse(expr)
        env = {k: float(v) for k, v in dict(variables or {}).items()}

        def _eval(node):
            if isinstance(node, ast.Expression):
                return _eval(node.body)
            if isinstance(node, ast.Constant):
                return float(node.value)
            if isinstance(node, ast.Name):
                if node.id in env:
                    return float(env[node.id])
                if node.id in self.functions:
                    return self.functions[node.id]
                raise RuntimeError(f"Unknown variable: {node.id}")
            if isinstance(node, ast.UnaryOp):
                v = float(_eval(node.operand))
                if isinstance(node.op, ast.UAdd):
                    return +v
                if isinstance(node.op, ast.USub):
                    return -v
                raise RuntimeError("Unsupported unary op")
            if isinstance(node, ast.BinOp):
                a = float(_eval(node.left))
                b = float(_eval(node.right))
                if isinstance(node.op, ast.Add):
                    return a + b
                if isinstance(node.op, ast.Sub):
                    return a - b
                if isinstance(node.op, ast.Mult):
                    return a * b
                if isinstance(node.op, ast.Div):
                    return a / b
                if isinstance(node.op, ast.Pow):
                    return a ** b
                if isinstance(node.op, ast.Mod):
                    return a % b
                if isinstance(node.op, ast.FloorDiv):
                    return a // b
                raise RuntimeError("Unsupported binary op")
            if isinstance(node, ast.Call):
                fn = _eval(node.func)
                args = [_eval(x) for x in node.args]
                return fn(*args)
            if isinstance(node, ast.Compare):
                left = _eval(node.left)
                out = True
                for op, comp in zip(node.ops, node.comparators):
                    right = _eval(comp)
                    if isinstance(op, ast.Gt):
                        out = out and (left > right)
                    elif isinstance(op, ast.GtE):
                        out = out and (left >= right)
                    elif isinstance(op, ast.Lt):
                        out = out and (left < right)
                    elif isinstance(op, ast.LtE):
                        out = out and (left <= right)
                    elif isinstance(op, ast.Eq):
                        out = out and (left == right)
                    elif isinstance(op, ast.NotEq):
                        out = out and (left != right)
                    else:
                        raise RuntimeError("Unsupported comparison")
                    left = right
                return 1.0 if out else 0.0
            if isinstance(node, ast.BoolOp):
                vals = [_eval(v) for v in node.values]
                if isinstance(node.op, ast.And):
                    return 1.0 if all(bool(v) for v in vals) else 0.0
                if isinstance(node.op, ast.Or):
                    return 1.0 if any(bool(v) for v in vals) else 0.0
                raise RuntimeError("Unsupported bool op")
            if isinstance(node, ast.IfExp):
                return _eval(node.body) if bool(_eval(node.test)) else _eval(node.orelse)
            raise RuntimeError(f"Unsupported AST node: {type(node).__name__}")

        val = _eval(tree)
        try:
            return float(val)
        except Exception as e:
            raise RuntimeError("Expression did not produce numeric value.") from e
