import numpy as np
from scipy.integrate import solve_ivp

from src.models import BVPProblem, Array


def solve_inner_ivp(problem: BVPProblem, p: Array) -> tuple[Array, Array]:
    t_grid = np.linspace(problem.t0, problem.t1, problem.num_points)

    solution = solve_ivp(
        fun=problem.rhs,
        t_span=(problem.t0, problem.t1),
        y0=np.asarray(p, dtype=float),
        t_eval=t_grid,
        rtol=problem.rtol_inner,
        atol=problem.atol_inner,
        method="RK45",
    )

    if not solution.success:
        raise RuntimeError(
            f"Не удалось решить внутреннюю задачу: {solution.message}"
        )

    return solution.t, solution.y.T
