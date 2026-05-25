import numpy as np

from src.examples import make_test_problem
from src.jacobian import finite_difference_jacobian, jacobian_phi


def test_variational_jacobian_is_close_to_finite_difference_jacobian():
    problem = make_test_problem()
    p = np.array([0.2, 0.8], dtype=float)

    variational = jacobian_phi(problem, p)
    finite_difference = finite_difference_jacobian(problem, p)

    assert np.allclose(variational, finite_difference, atol=1e-5)
