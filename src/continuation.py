import numpy as np
from scipy.integrate import solve_ivp

from src.jacobian import jacobian_phi
from src.models import Array, BVPProblem
from src.residuals import phi


def continuation_step(problem: BVPProblem, p0: Array) -> Array:
    """
    Выполняет один шаг метода продолжения по параметру.

    На вход подаётся текущее приближение p0.
    Строится внешняя задача Коши по параметру mu:

        dp/dmu = -[Phi'(p)]^(-1) Phi(p0),
        p(0) = p0,
        mu in [0, 1].

    Результат p(1) используется как новое приближение.
    """
    p0 = np.asarray(p0, dtype=float)
    initial_residual = phi(problem, p0)

    def outer_rhs(mu: float, p: Array) -> Array:
        jacobian = jacobian_phi(problem, p)

        try:
            direction = np.linalg.solve(jacobian, initial_residual)
        except np.linalg.LinAlgError as error:
            raise RuntimeError(
                "Матрица Phi'(p) вырождена или плохо обусловлена."
            ) from error

        return -direction

    solution = solve_ivp(
        fun=outer_rhs,
        t_span=(0.0, 1.0),
        y0=p0,
        rtol=problem.rtol_inner,
        atol=problem.atol_inner,
        method="RK45",
    )

    if not solution.success:
        raise RuntimeError(
            f"Не удалось решить внешнюю задачу: {solution.message}"
        )

    return solution.y[:, -1]


def newton_step(problem: BVPProblem, p: Array) -> Array:
    """
    Выполняет один классический шаг Ньютона:

        p_new = p - [Phi'(p)]^(-1) Phi(p).

    Эта функция нужна для сравнения с методом продолжения.
    """
    p = np.asarray(p, dtype=float)

    residual = phi(problem, p)
    jacobian = jacobian_phi(problem, p)

    try:
        correction = np.linalg.solve(jacobian, residual)
    except np.linalg.LinAlgError as error:
        raise RuntimeError(
            "Матрица Phi'(p) вырождена или плохо обусловлена."
        ) from error

    return p - correction
