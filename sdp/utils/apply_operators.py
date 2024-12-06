import operator
import ast

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


def evaluate_expression(expression: str):
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
            if node.id in {"True", "False"}:
                return eval(node.id)
            raise ValueError(f"Unsupported name: {node.id}")
        else:
            raise ValueError(f"Unsupported node type: {type(node)}")

    # Parse the expression into an AST tree
    tree = ast.parse(expression, mode='eval')
    return _eval(tree.body)
