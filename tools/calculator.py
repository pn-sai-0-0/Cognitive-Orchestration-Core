"""
CognitiveOC v3 — Calculator Tool
Safe arithmetic via AST whitelist (no eval/exec).
"""
from __future__ import annotations
import ast
import math
import operator as op
import re

_OPS = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
    ast.Div: op.truediv, ast.Pow: op.pow, ast.Mod: op.mod,
    ast.FloorDiv: op.floordiv, ast.USub: op.neg, ast.UAdd: op.pos,
}
_FUNCS = {
    'sqrt': math.sqrt, 'log': math.log, 'sin': math.sin,
    'cos': math.cos,   'tan': math.tan, 'abs': abs, 'round': round,
    'floor': math.floor, 'ceil': math.ceil, 'exp': math.exp,
}


def _eval_node(node):
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.operand))
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id in _FUNCS):
        return _FUNCS[node.func.id](*[_eval_node(a) for a in node.args])
    raise ValueError(f'unsupported expression node: {type(node).__name__}')


def calculate(expression: str) -> dict:
    """Evaluate a safe arithmetic expression. Returns {ok, result} or {ok, error}."""
    expr = expression.lower().replace('^', '**')
    m    = re.search(r'[0-9(][0-9a-z+\-*/(). ,_]*', expr)
    if not m:
        return {'ok': False, 'error': 'no arithmetic expression found'}
    try:
        result = _eval_node(ast.parse(m.group(0).strip(), mode='eval'))
        return {'ok': True, 'result': result, 'expression': m.group(0).strip()}
    except ZeroDivisionError:
        return {'ok': False, 'error': 'division by zero'}
    except Exception as e:
        return {'ok': False, 'error': str(e)}
