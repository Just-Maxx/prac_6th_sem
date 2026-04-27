import matplotlib.pyplot as plt

from src.examples import make_test_problem
from src.inner_solver import solve_inner_ivp
from src.residuals import phi, residual_norm


def main() -> None:
    problem = make_test_problem()

    p = problem.p0.copy()

    t_grid, states = solve_inner_ivp(problem, p)
    residual = phi(problem, p)

    print("Тестовый вектор p:")
    print(p)
    print("\nНевязка Phi(p):")
    print(residual)
    print(f"\nНорма невязки: {residual_norm(problem, p):.6e}")

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
