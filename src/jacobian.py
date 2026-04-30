import numpy as np

from src.models import Array, BVPProblem, Matrix
from src.residuals import phi
from src.variational_solver import solve_inner_with_variations


def jacobian_phi(problem: BVPProblem, p: Array) -> Matrix:
    _, states, matrices = solve_inner_with_variations(problem, p)

    x_left = states[0]
    x_right = states[-1]

    matrix_left = matrices[0]
    matrix_right = matrices[-1]

    jac_left, jac_right = problem.boundary_jacobian(x_left, x_right)

    return jac_left @ matrix_left + jac_right @ matrix_right


def finite_difference_jacobian(
    problem: BVPProblem,
    p: Array,
    step: float = 1e-6,
) -> Matrix:
    p = np.asarray(p, dtype=float)
    base_value = phi(problem, p)
    result = np.zeros((base_value.size, p.size), dtype=float)

    for index in range(p.size):
        shifted = p.copy()
        shifted[index] += step

        result[:, index] = (phi(problem, shifted) - base_value) / step

    return result
