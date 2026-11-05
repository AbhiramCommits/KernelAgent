"""Safe evaluation of the FLOP / byte models declared in ``meta.yaml``."""

from __future__ import annotations

import ast
import operator
from typing import Any

from .types import KernelTask, Shape

_BIN_OPS: dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS: dict[type, Any] = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def safe_eval(expression: str, variables: dict[str, float]) -> float:
    """Evaluate an arithmetic expression over named variables only."""
    tree = ast.parse(expression, mode="eval")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Load)):
            continue
        if isinstance(node, ast.Name):
            if node.id not in variables:
                raise ValueError(f"unknown variable {node.id!r} in {expression!r}")
            continue
        if isinstance(node, tuple(_BIN_OPS)) or isinstance(node, tuple(_UNARY_OPS)):
            continue
        raise ValueError(f"unsupported syntax {type(node).__name__} in {expression!r}")

    def walk(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)):
                raise ValueError("only numeric constants are allowed")
            return float(node.value)
        if isinstance(node, ast.Name):
            return float(variables[node.id])
        if isinstance(node, ast.BinOp):
            return _BIN_OPS[type(node.op)](walk(node.left), walk(node.right))
        if isinstance(node, ast.UnaryOp):
            return _UNARY_OPS[type(node.op)](walk(node.operand))
        raise ValueError(f"unsupported node {type(node).__name__}")

    return float(walk(tree))


def flops_for(task: KernelTask, shape: Shape) -> float:
    """Modelled floating-point operations for one call."""
    return safe_eval(task.flops_expr, {k: float(v) for k, v in shape.dims.items()})


def bytes_for(task: KernelTask, shape: Shape) -> float:
    """Modelled bytes moved for one call."""
    return safe_eval(task.bytes_expr, {k: float(v) for k, v in shape.dims.items()})
