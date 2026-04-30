import numpy as np

from src.models import Array, BVPProblem, Matrix


def oscillator_rhs(t: float, x: Array) -> Array:
    return np.array([x[1], -x[0]], dtype=float)


def oscillator_rhs_jacobian(t: float, x: Array) -> Matrix:
    return np.array([
        [0.0, 1.0],
        [-1.0, 0.0],
    ], dtype=float)


def oscillator_boundary_residual(x_left: Array, x_right: Array) -> Array:
    return np.array([
        x_left[0] - 0.0,
        x_right[0] - 1.0,
    ], dtype=float)


def oscillator_boundary_jacobian(
    x_left: Array,
    x_right: Array,
) -> tuple[Matrix, Matrix]:
    jac_left = np.array([
        [1.0, 0.0],
        [0.0, 0.0],
    ], dtype=float)

    jac_right = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
    ], dtype=float)

    return jac_left, jac_right


def make_test_problem() -> BVPProblem:
    return BVPProblem(
        t0=0.0,
        t1=float(np.pi / 2),
        dim=2,
        rhs=oscillator_rhs,
        rhs_jacobian=oscillator_rhs_jacobian,
        boundary_residual=oscillator_boundary_residual,
        boundary_jacobian=oscillator_boundary_jacobian,
        p0=np.array([0.2, 0.8], dtype=float),
        num_points=300,
    )
