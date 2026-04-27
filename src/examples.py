import numpy as np

from src.models import BVPProblem, Array


def oscillator_rhs(t: float, x: Array) -> Array:
    return np.array([x[1], -x[0]], dtype=float)


def oscillator_boundary_residual(x_left: Array, x_right: Array) -> Array:
    return np.array([
        x_left[0] - 0.0,
        x_right[0] - 1.0,
    ], dtype=float)


def make_test_problem() -> BVPProblem:
    return BVPProblem(
        t0=0.0,
        t1=float(np.pi / 2),
        dim=2,
        rhs=oscillator_rhs,
        boundary_residual=oscillator_boundary_residual,
        p0=np.array([0.0, 1.0], dtype=float),
        num_points=300,
    )
