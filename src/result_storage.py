import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from src.task_storage import RESULTS_DIR
from src.task_storage import ensure_project_dirs
from src.task_storage import normalize_task_name


def _to_serializable_array(values) -> list:
    return np.asarray(values, dtype=float).tolist()


def create_result_dir(task_name: str) -> Path:
    ensure_project_dirs()

    timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    normalized_name = normalize_task_name(task_name)
    result_dir = RESULTS_DIR / f"{normalized_name}_{timestamp}"

    result_dir.mkdir(parents=True, exist_ok=False)

    return result_dir


def save_json(file_path: Path, data: dict[str, Any]) -> None:
    with file_path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=4,
        )


def build_solution_data(solution) -> dict[str, Any]:
    return {
        "converged": bool(solution.converged),
        "iterations": int(solution.iterations),
        "p": _to_serializable_array(solution.p),
        "phi": _to_serializable_array(solution.residual),
        "phi_norm": float(solution.residual_norm),
        "phi_norm_history": _to_serializable_array(
            solution.residual_history,
        ),
        "t": _to_serializable_array(solution.t),
        "states": _to_serializable_array(solution.states),
    }


def save_solution_table_csv(file_path: Path, solution) -> None:
    t_values = np.asarray(solution.t, dtype=float)
    states = np.asarray(solution.states, dtype=float)

    headers = ["t"] + [
        f"x{index + 1}"
        for index in range(states.shape[1])
    ]

    with file_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(headers)

        for row_index, t_value in enumerate(t_values):
            row = [t_value] + states[row_index].tolist()
            writer.writerow(row)


def save_full_result(
    task_name: str,
    task_data: dict[str, Any],
    solution,
    figure,
    result_text: str = "",
) -> Path:
    result_dir = create_result_dir(task_name)

    save_json(result_dir / "task.json", task_data)
    save_json(result_dir / "solution.json", build_solution_data(solution))
    save_solution_table_csv(result_dir / "solution_table.csv", solution)

    if result_text:
        with (result_dir / "result.txt").open("w", encoding="utf-8") as file:
            file.write(result_text)

    if figure is not None:
        figure.savefig(
            result_dir / "plot.png",
            dpi=200,
            bbox_inches="tight",
        )

    return result_dir


def save_current_plot(file_path: Path, figure) -> None:
    if figure is None:
        raise ValueError("Нет построенного графика для сохранения.")

    figure.savefig(
        file_path,
        dpi=200,
        bbox_inches="tight",
    )
