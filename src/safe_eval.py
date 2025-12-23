import ast
import math

class SafeEvalError(Exception):
    pass

class SafeEvaluator:
    def __init__(self):
        self.allowed_names = {
            "pi": math.pi,
            "e": math.e
        }
        self.allowed_funcs = {
            "abs": abs,
            "round": round,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "asin": math.asin,
            "acos": math.acos,
            "atan": math.atan,
            "sinh": math.sinh,
            "cosh": math.cosh,
            "tanh": math.tanh,
            "exp": math.exp,
            "log": math.log,
            "log10": math.log10,
            "sqrt": math.sqrt
        }

    def eval_expr(self, expr: str, **variables):
        try:
            node = ast.parse(expr, mode="eval")
        except SyntaxError:
            raise SafeEvalError("Expresión inválida (syntax error).")

        return self._eval(node.body, variables)

    def _eval(self, node, variables):
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return float(node.value)
            raise SafeEvalError("Constante no permitida.")

        if isinstance(node, ast.Name):
            if node.id in variables:
                return float(variables[node.id])
            if node.id in self.allowed_names:
                return float(self.allowed_names[node.id])
            raise SafeEvalError(f"Variable no permitida: {node.id}")

        if isinstance(node, ast.BinOp):
            left = self._eval(node.left, variables)
            right = self._eval(node.right, variables)
            if isinstance(node.op, ast.Add): return left + right
            if isinstance(node.op, ast.Sub): return left - right
            if isinstance(node.op, ast.Mult): return left * right
            if isinstance(node.op, ast.Div): return left / right
            if isinstance(node.op, ast.Pow): return left ** right
            if isinstance(node.op, ast.Mod): return left % right
            raise SafeEvalError("Operación binaria no permitida.")

        if isinstance(node, ast.UnaryOp):
            val = self._eval(node.operand, variables)
            if isinstance(node.op, ast.UAdd): return +val
            if isinstance(node.op, ast.USub): return -val
            raise SafeEvalError("Operación unaria no permitida.")

        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise SafeEvalError("Llamada no permitida.")
            fname = node.func.id
            if fname not in self.allowed_funcs:
                raise SafeEvalError(f"Función no permitida: {fname}")
            args = [self._eval(a, variables) for a in node.args]
            return float(self.allowed_funcs[fname](*args))

        raise SafeEvalError("Expresión contiene nodos no permitidos.")
