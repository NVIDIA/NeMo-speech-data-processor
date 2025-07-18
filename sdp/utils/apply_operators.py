# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import operator
import ast
import re
from typing import Any, Dict

"""
This module provides a safe evaluator for simple Python expressions using the abstract syntax tree (AST).
It restricts execution to a subset of safe operations (arithmetic, logical, comparisons, indexing, etc.)
and selected built-in functions (e.g., max, min, len), while preventing arbitrary code execution.

Useful in cases where dynamic expressions need to be evaluated using a provided variable context,
such as configuration systems, data transformation pipelines, or manifest filtering.

Functions:
    - evaluate_expression: Safely evaluates a Python expression string using restricted AST operations.
"""

OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.BitOr: operator.or_,
    ast.BitAnd: operator.and_,
    ast.BitXor: operator.xor,
    ast.LShift: operator.lshift,
    ast.RShift: operator.rshift,
    ast.Invert: operator.invert,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Is: operator.is_,
    ast.IsNot: operator.is_not,
    ast.And: operator.and_,
    ast.Or: operator.or_,
    ast.Not: operator.not_,
}

SAFE_FUNCTIONS = {
    'max': max,
    'min': min,
    'len': len,
    'sum': sum,
    'abs': abs,
    'sorted': sorted,
}


def evaluate_expression(expression: str, variables: Dict[str, Any] = None, var_prefix: str = None) -> Any:
    """
    Safely evaluates a Python expression string using a restricted set of AST nodes and operators.

    Args:
        expression (str): The expression to evaluate.
        variables (Dict[str, Any], optional): A dictionary of variable names and values to use in evaluation.
        var_prefix (str, optional): If specified, this prefix will be removed from variable names
            in the expression before evaluation.

    Returns:
        any: The result of evaluating the expression.

    Raises:
        ValueError: If the expression contains unsupported operations or names.
    """
    if variables is None:
        variables = {}

    def _eval(node):
        match node:
            case ast.Expression():
                return _eval(node.body)

            case ast.BinOp():
                left = _eval(node.left)
                right = _eval(node.right)
                return OPERATORS[type(node.op)](left, right)

            case ast.UnaryOp():
                operand = _eval(node.operand)
                return OPERATORS[type(node.op)](operand)

            case ast.Subscript():
                value = _eval(node.value)
                match node.slice:
                    case ast.Slice():
                        start = _eval(node.slice.lower) if node.slice.lower else None
                        stop = _eval(node.slice.upper) if node.slice.upper else None
                        step = _eval(node.slice.step) if node.slice.step else None
                        return value[start:stop:step]
                    case _:
                        key = _eval(node.slice)
                        return value[key]

            case ast.Compare():
                left = _eval(node.left)
                right = _eval(node.comparators[0])
                return OPERATORS[type(node.ops[0])](left, right)

            case ast.BoolOp():
                values = [_eval(v) for v in node.values]
                match node.op:
                    case ast.And():
                        return all(values)
                    case ast.Or():
                        return any(values)

            case ast.IfExp():
                test = _eval(node.test)
                return _eval(node.body) if test else _eval(node.orelse)

            case ast.Constant():
                return node.value

            case ast.Name():
                var_name = node.id
                if var_name in variables:
                    return variables[var_name]
                elif var_name in {"True", "False"}:
                    return eval(var_name)
                raise ValueError(f"Unsupported name: {var_name}")

            case ast.Call():
                func_name = node.func.id if isinstance(node.func, ast.Name) else None
                if func_name in SAFE_FUNCTIONS:
                    func = SAFE_FUNCTIONS[func_name]
                    args = [_eval(arg) for arg in node.args]
                    return func(*args)
                else:
                    raise ValueError(f"Function {func_name} is not allowed")

            case ast.List():
                return [_eval(elt) for elt in node.elts]

            case ast.Dict():
                return {_eval(k): _eval(v) for k, v in zip(node.keys, node.values)}

            case ast.Attribute():
                value = _eval(node.value)
                return getattr(value, node.attr)

            case _:
                raise ValueError(f"Unsupported node type: {type(node)}")

    if var_prefix:
        var_prefix += '.'
        expression = re.sub(rf'{re.escape(var_prefix)}(\w+)', r'\1', expression)

    tree = ast.parse(expression, mode='eval')
    return _eval(tree.body)