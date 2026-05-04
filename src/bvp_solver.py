from src.continuation import continuation_step
from src.inner_solver import solve_inner_ivp
from src.models import Array, BVPProblem, BVPSolution
from src.residuals import phi, residual_norm


def solve_bvp_continuation(
    problem: BVPProblem,
    p0: Array | None = None,
    tolerance: float = 1e-8,
    max_iterations: int = 10,
) -> BVPSolution:
    """
    Решает краевую задачу методом продолжения по параметру.

    На каждой итерации:
    1. считается невязка Phi(p);
    2. если невязка мала, процесс останавливается;
    3. иначе выполняется один шаг метода продолжения.
    """
    if p0 is None:
        p = problem.p0.copy()
    else:
        p = p0.copy()

    residual_history = []

    for iteration in range(max_iterations + 1):
        current_norm = residual_norm(problem, p)
        residual_history.append(current_norm)

        if current_norm < tolerance:
            t_grid, states = solve_inner_ivp(problem, p)
            current_residual = phi(problem, p)

            return BVPSolution(
                p=p,
                t=t_grid,
                states=states,
                residual=current_residual,
                residual_norm=current_norm,
                iterations=iteration,
                converged=True,
                residual_history=residual_history,
            )

        if iteration == max_iterations:
            break

        p = continuation_step(problem, p)

    t_grid, states = solve_inner_ivp(problem, p)
    current_residual = phi(problem, p)
    current_norm = residual_norm(problem, p)

    return BVPSolution(
        p=p,
        t=t_grid,
        states=states,
        residual=current_residual,
        residual_norm=current_norm,
        iterations=max_iterations,
        converged=False,
        residual_history=residual_history,
    )
