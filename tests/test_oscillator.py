import numpy as np

from src.bvp_solver import solve_bvp_continuation
from src.examples import make_test_problem
from src.residuals import phi, residual_norm


def test_oscillator_solution_is_close_to_exact_parameter():
    problem = make_test_problem()

    solution = solve_bvp_continuation(
        problem,
        tolerance=1e-8,
        max_iterations=5,
    )

    assert solution.converged
    assert solution.residual_norm < 1e-8
    assert np.allclose(solution.p, np.array([0.0, 1.0]), atol=1e-6)


def test_phi_is_near_zero_for_exact_oscillator_parameter():
    problem = make_test_problem()
    exact_p = np.array([0.0, 1.0], dtype=float)

    value = phi(problem, exact_p)

    assert np.linalg.norm(value) < 1e-8
    assert residual_norm(problem, exact_p) < 1e-8
