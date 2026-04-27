import numpy as np

from src.inner_solver import solve_inner_ivp
from src.models import BVPProblem, Array


def phi(problem: BVPProblem, p: Array) -> Array:
    _, states = solve_inner_ivp(problem, p)

    x_left = states[0]
    x_right = states[-1]

    return problem.boundary_residual(x_left, x_right)


def residual_norm(problem: BVPProblem, p: Array) -> float:
    value = phi(problem, p)
    return float(np.linalg.norm(value, ord=2))
