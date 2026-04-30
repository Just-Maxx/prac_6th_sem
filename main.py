import matplotlib.pyplot as plt
import numpy as np

from src.examples import make_test_problem
from src.inner_solver import solve_inner_ivp
from src.jacobian import finite_difference_jacobian, jacobian_phi
from src.residuals import phi, residual_norm


def main() -> None:
    problem = make_test_problem()

    p = problem.p0.copy()

    t_grid, states = solve_inner_ivp(problem, p)
    residual = phi(problem, p)
    analytical_jacobian = jacobian_phi(problem, p)
    numerical_jacobian = finite_difference_jacobian(problem, p)

    print("Тестовый вектор p:")
    print(p)

    print("\nНевязка Phi(p):")
    print(residual)

    print(f"\nНорма невязки: {residual_norm(problem, p):.6e}")

    print("\nЯкобиан Phi'(p), найденный через уравнение в вариациях:")
    print(analytical_jacobian)

    print("\nЯкобиан Phi'(p), найденный конечными разностями:")
    print(numerical_jacobian)

    print("\nРазность якобианов:")
    print(analytical_jacobian - numerical_jacobian)

    print("\nНорма разности якобианов:")
    print(np.linalg.norm(analytical_jacobian - numerical_jacobian))

    plt.figure(figsize=(8, 5))
    plt.plot(t_grid, states[:, 0], label="x1(t)")
    plt.plot(t_grid, states[:, 1], label="x2(t)")
    plt.xlabel("t")
    plt.ylabel("state")
    plt.title("Решение внутренней задачи")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
