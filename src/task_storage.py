import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PRESET_TASKS_DIR = PROJECT_ROOT / "preset_tasks"
USER_TASKS_DIR = PROJECT_ROOT / "user_tasks"
RESULTS_DIR = PROJECT_ROOT / "results"


def ensure_project_dirs() -> None:
    PRESET_TASKS_DIR.mkdir(exist_ok=True)
    USER_TASKS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)


def normalize_task_name(name: str) -> str:
    normalized = name.strip().lower()
    normalized = normalized.replace(" ", "_")

    normalized = "".join(
        char
        for char in normalized
        if char.isalnum() or char in "_-"
    )

    if not normalized:
        raise ValueError("Название задачи не должно быть пустым.")

    return normalized


def save_user_task(task_name: str, task_data: dict[str, Any]) -> Path:
    ensure_project_dirs()

    filename = normalize_task_name(task_name) + ".json"
    file_path = USER_TASKS_DIR / filename

    with file_path.open("w", encoding="utf-8") as file:
        json.dump(
            task_data,
            file,
            ensure_ascii=False,
            indent=4,
        )

    return file_path


def load_task(file_path: Path) -> dict[str, Any]:
    with file_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def list_user_tasks() -> list[Path]:
    ensure_project_dirs()
    return sorted(USER_TASKS_DIR.glob("*.json"))


def list_preset_tasks() -> list[Path]:
    ensure_project_dirs()
    return sorted(PRESET_TASKS_DIR.glob("*.json"))


def delete_user_task(file_path: Path) -> None:
    ensure_project_dirs()

    if file_path.parent.resolve() != USER_TASKS_DIR.resolve():
        raise ValueError("Можно удалять только пользовательские задачи.")

    if file_path.exists():
        file_path.unlink()


def delete_all_user_tasks() -> None:
    ensure_project_dirs()

    for file_path in USER_TASKS_DIR.glob("*.json"):
        file_path.unlink()
