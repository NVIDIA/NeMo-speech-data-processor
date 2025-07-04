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
    ast.Add: operator.add,         # Addition (a + b)
    ast.Sub: operator.sub,         # Subtraction (a - b)
    ast.Mult: operator.mul,        # Multiplication (a * b)
    ast.Div: operator.truediv,     # True Division (a / b)
    ast.FloorDiv: operator.floordiv, # Floor Division (a // b)
    ast.Mod: operator.mod,         # Modulus (a % b)
    ast.Pow: operator.pow,         # Exponentiation (a ** b)
    ast.BitOr: operator.or_,       # Bitwise OR (a | b)
    ast.BitAnd: operator.and_,     # Bitwise AND (a & b)
    ast.BitXor: operator.xor,      # Bitwise XOR (a ^ b)
    ast.LShift: operator.lshift,   # Left Shift (a << b)
    ast.RShift: operator.rshift,   # Right Shift (a >> b)
    ast.Invert: operator.invert,   # Bitwise NOT (~a)
    ast.USub: operator.neg,        # Negation (-a)
    ast.UAdd: operator.pos,        # Unary Plus (+a)
    ast.Eq: operator.eq,           # Equality Check (a == b)
    ast.NotEq: operator.ne,        # Inequality Check (a != b)
    ast.Lt: operator.lt,           # Less Than (a < b)
    ast.LtE: operator.le,          # Less Than or Equal To (a <= b)
    ast.Gt: operator.gt,           # Greater Than (a > b)
    ast.GtE: operator.ge,          # Greater Than or Equal To (a >= b)
    ast.Is: operator.is_,          # Identity Check (a is b)
    ast.IsNot: operator.is_not,    # Negated Identity Check (a is not b)
    ast.And: operator.and_,        # Logical AND (a and b)
    ast.Or: operator.or_,          # Logical OR (a or b)
    ast.Not: operator.not_,        # Logical NOT (not a)
}

SAFE_FUNCTIONS = {
    'max': max,
    'min': min,
    'len': len,
    'sum': sum,
    'abs': abs,
    'sorted': sorted,
}

def evaluate_expression(expression: str, variables: Dict[str, Any] = None, var_prefix: str = None) -> any:
    """
    Safely evaluates a Python expression string using a restricted set of AST nodes and operators.

    This function supports arithmetic operations, comparisons, logical expressions, list/dict indexing,
    conditional expressions, and a whitelist of safe built-in functions. It allows evaluating expressions
    in the context of a user-provided dictionary of variables.

    Args:
        expression (str): The expression to evaluate. Example: "x + 2 if y > 3 else 0".
        variables (Dict[str, Any], optional): A dictionary of variable names and values to use in evaluation.
        var_prefix (str, optional): If specified, this prefix will be removed from variable names
            in the expression before evaluation (e.g., "data.x" → "x").

    Returns:
        any: The result of evaluating the expression.

    Raises:
        ValueError: If the expression contains unsupported operations or names.
    """
    
    if variables is None:
        variables = {}

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        elif isinstance(node, ast.BinOp):  # Binary operations
            left = _eval(node.left)
            right = _eval(node.right)
            return OPERATORS[type(node.op)](left, right)
        elif isinstance(node, ast.UnaryOp):  # Unary operations
            operand = _eval(node.operand)
            return OPERATORS[type(node.op)](operand)
        elif isinstance(node, ast.Subscript): # Accessing elements with []
            value = _eval(node.value)  # The collection (e.g., list, dict)
            if isinstance(node.slice, ast.Slice):  # Slice processing
                start = _eval(node.slice.lower) if node.slice.lower else None
                stop = _eval(node.slice.upper) if node.slice.upper else None
                step = _eval(node.slice.step) if node.slice.step else None
                return value[start:stop:step]
            else:
                key = _eval(node.slice) # The index/key
                return value[key]
        elif isinstance(node, ast.Compare):  # Comparisons
            left = _eval(node.left)
            right = _eval(node.comparators[0])
            return OPERATORS[type(node.ops[0])](left, right)
        elif isinstance(node, ast.BoolOp):  # Logical operations
            values = [_eval(value) for value in node.values]
            if isinstance(node.op, ast.And):
                return all(values)
            elif isinstance(node.op, ast.Or):
                return any(values)
        elif isinstance(node, ast.IfExp):  # Ternary if (condition ? true_value : false_value)
            test = _eval(node.test)
            return _eval(node.body) if test else _eval(node.orelse)
        elif isinstance(node, ast.Constant):  # Numbers, strings, etc.
            return node.value
        elif isinstance(node, ast.NameConstant):  # True, False, None
            return node.value
        elif isinstance(node, ast.Name):  # For identifiers
            var_name = node.id
            if var_name in variables:  # Look for variables in the provided dictionary
                return variables[var_name]
            elif var_name in {"True", "False"}:
                return eval(var_name)
            raise ValueError(f"Unsupported name: {node.id}")
        elif isinstance(node, ast.Call):  # Function call handling
            func_name = node.func.id if isinstance(node.func, ast.Name) else None
            if func_name in SAFE_FUNCTIONS:
                func = SAFE_FUNCTIONS[func_name]
                args = [_eval(arg) for arg in node.args]
                return func(*args)
            else:
                raise ValueError(f"Function {func_name} is not allowed")
        elif isinstance(node, ast.List):  # List literal
            return [_eval(elt) for elt in node.elts]
        elif isinstance(node, ast.Dict):  # Dict literal
            return {_eval(k): _eval(v) for k, v in zip(node.keys, node.values)}
        elif isinstance(node, ast.Attribute):  # Object attribute access
            value = _eval(node.value)
            return getattr(value, node.attr)
        else:
            raise ValueError(f"Unsupported node type: {type(node)}")

    var_prefix += '.'
    expression = re.sub(rf'{re.escape(var_prefix)}(\w+)', r'\1', expression)

    # Parse the expression into an AST tree
    tree = ast.parse(expression, mode='eval')
    return _eval(tree.body)