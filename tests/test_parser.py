import numpy as np
import pytest

from src.parser import build_boundary_functions, build_rhs_functions


def test_parser_builds_rhs_and_jacobian_from_text():
    rhs, rhs_jacobian = build_rhs_functions(
        expressions=["x2", "-sin(x1)"],
        dim=2,
    )

    x = np.array([0.5, 2.0], dtype=float)

    assert np.allclose(rhs(0.0, x), np.array([2.0, -np.sin(0.5)]))
    assert rhs_jacobian(0.0, x).shape == (2, 2)


def test_parser_builds_boundary_functions_from_human_readable_text():
    boundary_residual, boundary_jacobian = build_boundary_functions(
        expressions=["x1(t0) = 0", "x1(t1) = 1"],
        dim=2,
    )

    x_left = np.array([0.0, 1.0], dtype=float)
    x_right = np.array([1.0, 0.0], dtype=float)

    residual = boundary_residual(x_left, x_right)
    jac_left, jac_right = boundary_jacobian(x_left, x_right)

    assert np.allclose(residual, np.zeros(2))
    assert jac_left.shape == (2, 2)
    assert jac_right.shape == (2, 2)


def test_parser_rejects_unknown_function():
    with pytest.raises(ValueError):
        build_rhs_functions(
            expressions=["x2", "-son(x1)"],
            dim=2,
        )
