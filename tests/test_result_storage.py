import json

from src import result_storage
from src.bvp_solver import solve_bvp_continuation
from src.examples import make_test_problem


def test_save_full_result_creates_expected_files(tmp_path, monkeypatch):
    monkeypatch.setattr(result_storage, "RESULTS_DIR", tmp_path)

    problem = make_test_problem()
    solution = solve_bvp_continuation(
        problem,
        tolerance=1e-8,
        max_iterations=5,
    )

    task_data = {
        "name": "oscillator",
        "t0": 0.0,
        "t1": 1.5708,
        "dim": 2,
    }

    result_dir = result_storage.save_full_result(
        task_name="oscillator",
        task_data=task_data,
        solution=solution,
        figure=None,
        result_text="test result",
    )

    assert result_dir.exists()
    assert (result_dir / "task.json").exists()
    assert (result_dir / "solution.json").exists()
    assert (result_dir / "solution_table.csv").exists()
    assert (result_dir / "result.txt").exists()

    with (result_dir / "solution.json").open("r", encoding="utf-8") as file:
        data = json.load(file)

    assert data["converged"] is True
    assert data["phi_norm"] < 1e-8
