from typing import Sequence

import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr

from src.models import Array, Matrix


_ALLOWED_FUNCTIONS = {
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "exp": sp.exp,
    "log": sp.log,
    "sqrt": sp.sqrt,
    "abs": sp.Abs,
}


def validate_allowed_functions(expression: sp.Expr, source: str) -> None:
    unknown_functions = {
        function.func.__name__
        for function in expression.atoms(sp.Function)
        if function.func not in _ALLOWED_FUNCTIONS.values()
    }

    if unknown_functions:
        names = ", ".join(sorted(unknown_functions))
        raise ValueError(
            f"В выражении '{source}' найдены неизвестные функции: {names}. "
            "Доступны только: sin, cos, tan, exp, log, sqrt, abs."
        )


def create_symbols(dim: int) -> tuple[sp.Symbol, list[sp.Symbol]]:
    """
    Создаёт символьные переменные t, x1, ..., xn.
    """
    if dim <= 0:
        raise ValueError("Размерность системы должна быть положительной.")

    t_symbol = sp.Symbol("t")
    x_symbols = [
        sp.Symbol(f"x{index}")
        for index in range(1, dim + 1)
    ]

    return t_symbol, x_symbols


def parse_rhs_expressions(
    expressions: Sequence[str],
    dim: int,
) -> tuple[list[sp.Expr], sp.Symbol, list[sp.Symbol]]:
    """
    Разбирает строки правых частей системы ОДУ.
    """
    if len(expressions) != dim:
        raise ValueError(
            "Количество правых частей должно совпадать "
            "с размерностью системы."
        )

    t_symbol, x_symbols = create_symbols(dim)

    local_dict = {
        "t": t_symbol,
        **{
            f"x{index}": x_symbols[index - 1]
            for index in range(1, dim + 1)
        },
        **_ALLOWED_FUNCTIONS,
    }

    parsed_expressions = []

    for expression in expressions:
        try:
            parsed = parse_expr(
                expression,
                local_dict=local_dict,
                evaluate=True,
            )
        except Exception as error:
            raise ValueError(
                f"Не удалось разобрать правую часть: {expression}"
            ) from error

        validate_allowed_functions(parsed, expression)
        parsed_expressions.append(parsed)

    return parsed_expressions, t_symbol, x_symbols


def build_rhs_functions(
    expressions: Sequence[str],
    dim: int,
):
    """
    По строкам правых частей строит численные функции rhs и rhs_jacobian.
    """
    parsed_expressions, t_symbol, x_symbols = parse_rhs_expressions(
        expressions=expressions,
        dim=dim,
    )

    rhs_matrix = sp.Matrix(parsed_expressions)
    jacobian_matrix = rhs_matrix.jacobian(x_symbols)

    variables = [t_symbol, *x_symbols]

    rhs_raw = sp.lambdify(
        variables,
        rhs_matrix,
        modules="numpy",
    )

    jacobian_raw = sp.lambdify(
        variables,
        jacobian_matrix,
        modules="numpy",
    )

    def rhs(t: float, x: Array) -> Array:
        values = [t, *np.asarray(x, dtype=float)]
        result = rhs_raw(*values)
        return np.asarray(result, dtype=float).reshape(dim)

    def rhs_jacobian(t: float, x: Array) -> Matrix:
        values = [t, *np.asarray(x, dtype=float)]
        result = jacobian_raw(*values)
        return np.asarray(result, dtype=float).reshape(dim, dim)

    return rhs, rhs_jacobian


def create_boundary_symbols(
    dim: int,
) -> tuple[list[sp.Symbol], list[sp.Symbol]]:
    """
    Создаёт символьные переменные для левого и правого концов.

    x1_left, ..., xn_left  соответствуют x(t0)
    x1_right, ..., xn_right соответствуют x(t1)
    """
    if dim <= 0:
        raise ValueError("Размерность системы должна быть положительной.")

    left_symbols = [
        sp.Symbol(f"x{index}_left")
        for index in range(1, dim + 1)
    ]

    right_symbols = [
        sp.Symbol(f"x{index}_right")
        for index in range(1, dim + 1)
    ]

    return left_symbols, right_symbols


def normalize_boundary_expression(expression: str) -> str:
    """
    Преобразует пользовательскую запись граничного условия
    в выражение R_i, которое должно быть равно нулю.
    """
    normalized = expression.strip()

    normalized = normalized.replace("(t0)", "_left")
    normalized = normalized.replace("(t1)", "_right")

    if "=" not in normalized:
        return normalized

    parts = normalized.split("=")

    if len(parts) != 2:
        raise ValueError(
            f"Некорректное граничное условие: {expression}"
        )

    left_part = parts[0].strip()
    right_part = parts[1].strip()

    if not left_part or not right_part:
        raise ValueError(
            f"Некорректное граничное условие: {expression}"
        )

    return f"({left_part}) - ({right_part})"


def parse_boundary_expressions(
    expressions: Sequence[str],
    dim: int,
) -> tuple[list[sp.Expr], list[sp.Symbol], list[sp.Symbol]]:
    """
    Разбирает строки граничных условий.

    Условия задаются в виде невязок.

    Например:
        ["x1_left", "x1_right - 1"]

    означает:
        x1(t0) = 0,
        x1(t1) = 1.
    """
    if len(expressions) != dim:
        raise ValueError(
            "Количество граничных условий должно совпадать "
            "с размерностью системы."
        )

    left_symbols, right_symbols = create_boundary_symbols(dim)

    local_dict = {
        **{
            f"x{index}_left": left_symbols[index - 1]
            for index in range(1, dim + 1)
        },
        **{
            f"x{index}_right": right_symbols[index - 1]
            for index in range(1, dim + 1)
        },
        **_ALLOWED_FUNCTIONS,
    }

    parsed_expressions = []

    for expression in expressions:
        normalized_expression = normalize_boundary_expression(expression)

        try:
            parsed = parse_expr(
                normalized_expression,
                local_dict=local_dict,
                evaluate=True,
            )
        except Exception as error:
            raise ValueError(
                f"Не удалось разобрать граничное условие: {expression}"
            ) from error

        validate_allowed_functions(parsed, expression)
        parsed_expressions.append(parsed)

    return parsed_expressions, left_symbols, right_symbols


def build_boundary_functions(
    expressions: Sequence[str],
    dim: int,
):
    """
    По строкам граничных условий строит функции:
    boundary_residual и boundary_jacobian.

    expressions задаются как невязки, которые должны быть равны нулю.
    """
    parsed_expressions, left_symbols, right_symbols = (
        parse_boundary_expressions(
            expressions=expressions,
            dim=dim,
        )
    )

    residual_matrix = sp.Matrix(parsed_expressions)

    jacobian_left_matrix = residual_matrix.jacobian(left_symbols)
    jacobian_right_matrix = residual_matrix.jacobian(right_symbols)

    variables = [*left_symbols, *right_symbols]

    residual_raw = sp.lambdify(
        variables,
        residual_matrix,
        modules="numpy",
    )

    jacobian_left_raw = sp.lambdify(
        variables,
        jacobian_left_matrix,
        modules="numpy",
    )

    jacobian_right_raw = sp.lambdify(
        variables,
        jacobian_right_matrix,
        modules="numpy",
    )

    def boundary_residual(x_left: Array, x_right: Array) -> Array:
        values = [
            *np.asarray(x_left, dtype=float),
            *np.asarray(x_right, dtype=float),
        ]

        result = residual_raw(*values)
        return np.asarray(result, dtype=float).reshape(dim)

    def boundary_jacobian(
        x_left: Array,
        x_right: Array,
    ) -> tuple[Matrix, Matrix]:
        values = [
            *np.asarray(x_left, dtype=float),
            *np.asarray(x_right, dtype=float),
        ]

        jacobian_left = jacobian_left_raw(*values)
        jacobian_right = jacobian_right_raw(*values)

        return (
            np.asarray(jacobian_left, dtype=float).reshape(dim, dim),
            np.asarray(jacobian_right, dtype=float).reshape(dim, dim),
        )

    return boundary_residual, boundary_jacobian
