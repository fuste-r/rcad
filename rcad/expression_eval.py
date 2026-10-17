def _evaluate_expression(expression: str, variables: dict[str, float] | None = None) -> float:
    """Evaluate a small arithmetic expression using Python syntax.
    Names resolve to `variables`; numbers/operators/functions use Python semantics."""
    import ast
    import operator as _op
    _bin = {
        ast.Add: _op.add, ast.Sub: _op.sub, ast.Mult: _op.mul,
        ast.Div: _op.truediv, ast.FloorDiv: _op.floordiv,
        ast.Mod: _op.mod, ast.Pow: _op.pow,
    }
    _unary = {ast.UAdd: _op.pos, ast.USub: _op.neg}

    def _node(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _node(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return float(node.value)
            raise ValueError(f"Unsupported constant {node.value!r}")
        if isinstance(node, ast.Name):
            if node.id in variables:
                return float(variables[node.id])
            raise ValueError(f"Unknown variable: {node.id}")
        if isinstance(node, ast.BinOp):
            if type(node.op) in _bin:
                return _bin[type(node.op)](_node(node.left), _node(node.right))
            raise ValueError(f"Unsupported binary operator {type(node.op).__name__}")
        if isinstance(node, ast.UnaryOp):
            if type(node.op) in _unary:
                return _unary[type(node.op)](_node(node.operand))
            raise ValueError(f"Unsupported unary operator {type(node.op).__name__}")
        if isinstance(node, ast.Call):
            raise ValueError("Function calls are not supported yet")
        raise ValueError(f"Unsupported expression node {type(node).__name__}")

    return _node(ast.parse(expression, mode="eval"))

