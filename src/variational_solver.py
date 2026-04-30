import numpy as np
from scipy.integrate import solve_ivp

from src.models import Array, BVPProblem, Matrix


def solve_inner_with_variations(
    problem: BVPProblem,
    p: Array,
) -> tuple[Array, Array, Array]:
    dim = problem.dim
    t_grid = np.linspace(problem.t0, problem.t1, problem.num_points)

    initial_matrix = np.eye(dim, dtype=float)
    initial_state = np.concatenate([
        np.asarray(p, dtype=float),
        initial_matrix.reshape(-1),
    ])

    def extended_rhs(t: float, y: Array) -> Array:
        x = y[:dim]
        matrix = y[dim:].reshape(dim, dim)

        dx_dt = problem.rhs(t, x)
        jacobian = problem.rhs_jacobian(t, x)
        dmatrix_dt = jacobian @ matrix

        return np.concatenate([
            dx_dt,
            dmatrix_dt.reshape(-1),
        ])

    solution = solve_ivp(
        fun=extended_rhs,
        t_span=(problem.t0, problem.t1),
        y0=initial_state,
        t_eval=t_grid,
        rtol=problem.rtol_inner,
        atol=problem.atol_inner,
        method="RK45",
    )

    if not solution.success:
        raise RuntimeError(
            f"Не удалось решить внутреннюю задачу с вариациями: "
            f"{solution.message}"
        )

    full_states = solution.y.T
    states = full_states[:, :dim]
    matrices = full_states[:, dim:].reshape(-1, dim, dim)

    return solution.t, states, matrices
