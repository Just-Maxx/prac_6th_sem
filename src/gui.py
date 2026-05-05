import tkinter as tk
from pathlib import Path
from tkinter import messagebox
from tkinter import simpledialog
from tkinter import ttk

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from src.bvp_solver import solve_bvp_continuation
from src.models import BVPProblem
from src.parser import build_boundary_functions, build_rhs_functions
from src.app_texts import (
    ABOUT_AUTHOR_TEXT,
    ABOUT_PROGRAM_TEXT,
    INPUT_HELP_TEXT,
)
from src.task_storage import (
    delete_all_user_tasks,
    delete_user_task,
    list_preset_tasks,
    list_user_tasks,
    load_task,
    save_user_task,
)
from pathlib import Path
from tkinter import simpledialog


class BVPApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Метод продолжения по параметру")
        self.root.geometry("1250x820")

        self.canvas = None

        self._create_menu()
        self._create_widgets()

    def _create_widgets(self) -> None:
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        task_frame = ttk.Frame(left_frame)
        task_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        view_frame = ttk.Frame(left_frame)
        view_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        output_frame = ttk.Frame(main_frame)
        output_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        ttk.Label(
            task_frame,
            text="Параметры задачи",
            font=("Arial", 14, "bold"),
        ).pack(anchor="w", pady=(0, 10))

        self.t0_entry = self._add_entry(task_frame, "t0:", "0.0", width=24)
        self.t1_entry = self._add_entry(task_frame, "t1:", "1.5708", width=24)
        self.dim_entry = self._add_entry(
            task_frame,
            "Размерность:",
            "2",
            width=24,
        )

        ttk.Label(task_frame, text="Правые части ОДУ:").pack(anchor="w")
        ttk.Label(
            task_frame,
            text="Каждая строка — отдельное уравнение",
            foreground="gray",
        ).pack(anchor="w")

        self.rhs_text = tk.Text(task_frame, height=5, width=30)
        self.rhs_text.pack(fill=tk.X, pady=(0, 10))
        self.rhs_text.insert("1.0", "x2\n-x1")

        ttk.Label(task_frame, text="Граничные условия:").pack(anchor="w")
        ttk.Label(
            task_frame,
            text="Каждая строка — одно условие вида xk(t0)=a или xk(t1)=b",
            foreground="gray",
        ).pack(anchor="w")

        self.boundary_text = tk.Text(task_frame, height=5, width=30)
        self.boundary_text.pack(fill=tk.X, pady=(0, 10))
        self.boundary_text.insert("1.0", "x1(t0) = 0\nx1(t1) = 1")

        self.p0_entry = self._add_entry(
            task_frame,
            "Начальное приближение p0:",
            "0.2, 0.8",
            width=24,
        )
        self.tolerance_entry = self._add_entry(
            task_frame,
            "Точность:",
            "1e-8",
            width=24,
        )
        self.max_iter_entry = self._add_entry(
            task_frame,
            "Макс. итераций:",
            "10",
            width=24,
        )

        ttk.Label(
            view_frame,
            text="Параметры графика",
            font=("Arial", 14, "bold"),
        ).pack(anchor="w", pady=(0, 10))

        self.num_points_entry = self._add_entry(
            view_frame,
            "Точек графика:",
            "300",
            width=22,
        )

        ttk.Label(view_frame, text="Тип графика:").pack(anchor="w")
        self.plot_type_var = tk.StringVar(value="Компоненты от времени")
        plot_type_box = ttk.Combobox(
            view_frame,
            textvariable=self.plot_type_var,
            values=[
                "Компоненты от времени",
                "Фазовая плоскость",
                "Сходимость"
            ],
            state="readonly",
            width=22,
        )
        plot_type_box.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(view_frame, text="Ось X:").pack(anchor="w")
        self.phase_x_var = tk.StringVar(value="x1")
        self.phase_x_box = ttk.Combobox(
            view_frame,
            textvariable=self.phase_x_var,
            values=["x1", "x2"],
            state="readonly",
            width=22,
        )
        self.phase_x_box.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(view_frame, text="Ось Y:").pack(anchor="w")
        self.phase_y_var = tk.StringVar(value="x2")
        self.phase_y_box = ttk.Combobox(
            view_frame,
            textvariable=self.phase_y_var,
            values=["x1", "x2"],
            state="readonly",
            width=22,
        )
        self.phase_y_box.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(
            view_frame,
            text="Для фазовой плоскости\nвыберите две разные компоненты.",
            foreground="gray",
        ).pack(anchor="w", pady=(0, 10))

        buttons_frame = ttk.Frame(view_frame)
        buttons_frame.pack(fill=tk.X, pady=(10, 0))

        close_button = ttk.Button(
            buttons_frame,
            text="Закрыть",
            command=self.root.destroy,
        )
        close_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

        solve_button = ttk.Button(
            buttons_frame,
            text="Решить",
            command=self._solve_problem,
        )
        solve_button.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(5, 0))

        top_output_frame = ttk.Frame(output_frame)
        top_output_frame.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(
            top_output_frame,
            text="Результат",
            font=("Arial", 14, "bold"),
        ).pack(side=tk.LEFT)

        self.result_text = tk.Text(output_frame, height=14)
        self.result_text.pack(fill=tk.X, pady=(0, 10))

        self.plot_frame = ttk.Frame(output_frame)
        self.plot_frame.pack(fill=tk.BOTH, expand=True)

    def _create_menu(self) -> None:
        menu_bar = tk.Menu(self.root)

        file_menu = tk.Menu(menu_bar, tearoff=0)
        file_menu.add_command(
            label="Сохранить задачу...",
            command=self._save_current_task,
        )
        file_menu.add_command(
            label="Загрузить сохранённую задачу...",
            command=self._load_user_task,
        )
        file_menu.add_command(
            label="Удалить сохранённую задачу...",
            command=self._delete_user_task,
        )
        file_menu.add_command(
            label="Удалить все сохранённые задачи...",
            command=self._delete_all_user_tasks,
        )
        file_menu.add_separator()
        file_menu.add_command(
            label="Выход",
            command=self.root.destroy,
        )
        menu_bar.add_cascade(label="Файл", menu=file_menu)

        examples_menu = tk.Menu(menu_bar, tearoff=0)
        examples_menu.add_command(
            label="Загрузить пример...",
            command=self._load_preset_task,
        )
        menu_bar.add_cascade(label="Примеры", menu=examples_menu)

        help_menu = tk.Menu(menu_bar, tearoff=0)
        help_menu.add_command(
            label="Формат ввода",
            command=self._show_input_help,
        )
        help_menu.add_command(
            label="О программе",
            command=self._show_about_program,
        )
        help_menu.add_command(
            label="Об авторе",
            command=self._show_about_author,
        )
        menu_bar.add_cascade(label="Справка", menu=help_menu)

        self.root.config(menu=menu_bar)

    def _show_input_help(self) -> None:
        messagebox.showinfo("Формат ввода", INPUT_HELP_TEXT)

    def _show_about_program(self) -> None:
        messagebox.showinfo("О программе", ABOUT_PROGRAM_TEXT)

    def _show_about_author(self) -> None:
        author_window = tk.Toplevel(self.root)
        author_window.title("Об авторе")
        author_window.resizable(False, False)

        content_frame = ttk.Frame(author_window, padding=20)
        content_frame.pack(fill=tk.BOTH, expand=True)

        project_root = Path(__file__).resolve().parents[1]
        photo_path = project_root / "assets" / "author.png"

        try:
            photo = tk.PhotoImage(file=str(photo_path))
            photo_label = ttk.Label(content_frame, image=photo)
            photo_label.image = photo
            photo_label.pack(pady=(0, 12))
        except tk.TclError:
            ttk.Label(
                content_frame,
                text="Фото автора не найдено.",
                foreground="gray",
            ).pack(pady=(0, 12))

        ttk.Label(
            content_frame,
            text=ABOUT_AUTHOR_TEXT,
            justify=tk.CENTER,
            wraplength=450,
        ).pack(pady=(0, 15))

        close_button = ttk.Button(
            content_frame,
            text="OK",
            command=author_window.destroy,
        )
        close_button.pack(anchor="e")

    def _collect_task_data(self) -> dict:
        dim = int(self.dim_entry.get())

        return {
            "t0": float(self.t0_entry.get()),
            "t1": float(self.t1_entry.get()),
            "dim": dim,
            "rhs": self._read_lines(self.rhs_text),
            "boundary_conditions": self._read_lines(self.boundary_text),
            "p0": self._read_p0(dim).tolist(),
            "tolerance": float(self.tolerance_entry.get()),
            "max_iterations": int(self.max_iter_entry.get()),
            "num_points": int(self.num_points_entry.get()),
            "plot_type": self.plot_type_var.get(),
            "phase_x": self.phase_x_var.get(),
            "phase_y": self.phase_y_var.get(),
        }

    def _set_entry_value(self, entry: ttk.Entry, value: str) -> None:
        entry.delete(0, tk.END)
        entry.insert(0, value)

    def _set_text_value(self, text_widget: tk.Text, value: str) -> None:
        text_widget.delete("1.0", tk.END)
        text_widget.insert("1.0", value)

    def _apply_task_data(self, task_data: dict) -> None:
        self._set_entry_value(self.t0_entry, str(task_data["t0"]))
        self._set_entry_value(self.t1_entry, str(task_data["t1"]))
        self._set_entry_value(self.dim_entry, str(task_data["dim"]))

        self._set_text_value(
            self.rhs_text,
            "\n".join(task_data["rhs"]),
        )
        self._set_text_value(
            self.boundary_text,
            "\n".join(task_data["boundary_conditions"]),
        )

        self._set_entry_value(
            self.p0_entry,
            ", ".join(str(value) for value in task_data["p0"]),
        )
        self._set_entry_value(
            self.tolerance_entry,
            str(task_data.get("tolerance", "1e-8")),
        )
        self._set_entry_value(
            self.max_iter_entry,
            str(task_data.get("max_iterations", 10)),
        )
        self._set_entry_value(
            self.num_points_entry,
            str(task_data.get("num_points", 300)),
        )

        self.plot_type_var.set(
            task_data.get("plot_type", "Компоненты от времени")
        )
        self.phase_x_var.set(task_data.get("phase_x", "x1"))
        self.phase_y_var.set(task_data.get("phase_y", "x2"))

    def _choose_task_file(
        self,
        title: str,
        task_files: list[Path],
    ) -> Path | None:
        if not task_files:
            messagebox.showinfo(title, "Нет доступных задач.")
            return None

        task_names = "\n".join(
            f"{index + 1}. {file_path.stem}"
            for index, file_path in enumerate(task_files)
        )

        choice = simpledialog.askinteger(
            title,
            f"Выберите номер задачи:\n\n{task_names}",
            minvalue=1,
            maxvalue=len(task_files),
        )

        if choice is None:
            return None

        return task_files[choice - 1]

    def _save_current_task(self) -> None:
        try:
            task_name = simpledialog.askstring(
                "Сохранить задачу",
                "Введите название задачи:",
            )

            if not task_name:
                return

            task_data = self._collect_task_data()
            task_data["name"] = task_name

            file_path = save_user_task(task_name, task_data)

            messagebox.showinfo(
                "Сохранение",
                f"Задача сохранена:\n{file_path.name}",
            )

        except Exception as error:
            messagebox.showerror("Ошибка", str(error))

    def _load_user_task(self) -> None:
        try:
            task_file = self._choose_task_file(
                "Загрузить сохранённую задачу",
                list_user_tasks(),
            )

            if task_file is None:
                return

            task_data = load_task(task_file)
            self._apply_task_data(task_data)

            messagebox.showinfo(
                "Загрузка",
                f"Задача загружена:\n{task_file.name}",
            )

        except Exception as error:
            messagebox.showerror("Ошибка", str(error))

    def _load_preset_task(self) -> None:
        try:
            task_file = self._choose_task_file(
                "Загрузить пример",
                list_preset_tasks(),
            )

            if task_file is None:
                return

            task_data = load_task(task_file)
            self._apply_task_data(task_data)

            messagebox.showinfo(
                "Загрузка примера",
                f"Пример загружен:\n{task_file.name}",
            )

        except Exception as error:
            messagebox.showerror("Ошибка", str(error))

    def _delete_user_task(self) -> None:
        try:
            task_file = self._choose_task_file(
                "Удалить сохранённую задачу",
                list_user_tasks(),
            )

            if task_file is None:
                return

            confirmed = messagebox.askyesno(
                "Подтверждение",
                f"Удалить задачу '{task_file.stem}'?",
            )

            if not confirmed:
                return

            delete_user_task(task_file)

            messagebox.showinfo(
                "Удаление",
                "Задача удалена.",
            )

        except Exception as error:
            messagebox.showerror("Ошибка", str(error))

    def _delete_all_user_tasks(self) -> None:
        try:
            confirmed = messagebox.askyesno(
                "Подтверждение",
                "Удалить все сохранённые пользовательские задачи?",
            )

            if not confirmed:
                return

            delete_all_user_tasks()

            messagebox.showinfo(
                "Удаление",
                "Все сохранённые задачи удалены.",
            )

        except Exception as error:
            messagebox.showerror("Ошибка", str(error))

    def _add_entry(
        self,
        parent: ttk.Frame,
        label: str,
        default: str,
        width: int = 35,
    ) -> ttk.Entry:
        ttk.Label(parent, text=label).pack(anchor="w")
        entry = ttk.Entry(parent, width=width)
        entry.pack(fill=tk.X, pady=(0, 8))
        entry.insert(0, default)
        return entry

    def _read_lines(self, text_widget: tk.Text) -> list[str]:
        content = text_widget.get("1.0", tk.END)
        return [
            line.strip()
            for line in content.splitlines()
            if line.strip()
        ]

    def _read_p0(self, dim: int) -> np.ndarray:
        raw_values = self.p0_entry.get().replace(";", ",").split(",")

        try:
            values = [
                float(value.strip())
                for value in raw_values
                if value.strip()
            ]
        except ValueError as error:
            raise ValueError(
                "Начальное приближение p0 должно содержать числа."
            ) from error

        if len(values) != dim:
            raise ValueError(
                "Количество чисел в p0 должно совпадать "
                "с размерностью системы."
            )

        return np.array(values, dtype=float)

    def _format_number(self, value: float) -> str:
        value = float(value)

        if abs(value) < 1e-8:
            value = 0.0

        return f"{value:.6g}"

    def _format_array(self, values) -> str:
        array = np.asarray(values, dtype=float)

        formatted_values = [
            self._format_number(value)
            for value in array
        ]

        return "[" + ", ".join(formatted_values) + "]"

    def _format_bool(self, value: bool) -> str:
        if value:
            return "Да"

        return "Нет"

    def _format_array(self, values) -> str:
        array = np.asarray(values, dtype=float)
        formatted_values = [
            self._format_number(value)
            for value in array
        ]
        return "[" + ", ".join(formatted_values) + "]"

    def _solve_problem(self) -> None:
        try:
            t0 = float(self.t0_entry.get())
            t1 = float(self.t1_entry.get())
            dim = int(self.dim_entry.get())
            axis_values = [
                f"x{index}"
                for index in range(1, dim + 1)
            ]

            self.phase_x_box["values"] = axis_values
            self.phase_y_box["values"] = axis_values

            if self.phase_x_var.get() not in axis_values:
                self.phase_x_var.set(axis_values[0])

            if self.phase_y_var.get() not in axis_values:
                self.phase_y_var.set(axis_values[min(1, dim - 1)])

            tolerance = float(self.tolerance_entry.get())
            max_iterations = int(self.max_iter_entry.get())
            num_points = int(self.num_points_entry.get())

            rhs_expressions = self._read_lines(self.rhs_text)
            boundary_expressions = self._read_lines(self.boundary_text)
            p0 = self._read_p0(dim)

            rhs, rhs_jacobian = build_rhs_functions(
                expressions=rhs_expressions,
                dim=dim,
            )

            boundary_residual, boundary_jacobian = build_boundary_functions(
                expressions=boundary_expressions,
                dim=dim,
            )

            problem = BVPProblem(
                t0=t0,
                t1=t1,
                dim=dim,
                rhs=rhs,
                rhs_jacobian=rhs_jacobian,
                boundary_residual=boundary_residual,
                boundary_jacobian=boundary_jacobian,
                p0=p0,
                num_points=num_points,
            )

            solution = solve_bvp_continuation(
                problem=problem,
                tolerance=tolerance,
                max_iterations=max_iterations,
            )

            self._show_result(solution)
            self._show_plot(solution)

        except Exception as error:
            messagebox.showerror("Ошибка", str(error))

    def _show_result(self, solution) -> None:
        self.result_text.delete("1.0", tk.END)

        lines = [
            f"Сошлось: {self._format_bool(solution.converged)}",
            f"Количество итераций: {solution.iterations}",
            "",
            "Найденный параметр p:",
            self._format_array(solution.p),
            "",
            "Значение Phi(p) на границе:",
            self._format_array(solution.residual),
            "",
            (
                "Норма ошибки граничных условий ||Phi(p)||: "
                f"{self._format_number(solution.residual_norm)}"
            ),
            "",
            "История нормы ошибки:",
        ]

        for index, value in enumerate(solution.residual_history):
            lines.append(
                f"Итерация {index}: {self._format_number(value)}"
            )

        self.result_text.insert(tk.END, "\n".join(lines))

    def _show_plot(self, solution) -> None:
        if self.canvas is not None:
            self.canvas.get_tk_widget().destroy()

        figure, axis = plt.subplots(figsize=(6, 4))

        plot_type = self.plot_type_var.get()

        if plot_type == "Фазовая плоскость":
            x_axis_name = self.phase_x_var.get()
            y_axis_name = self.phase_y_var.get()

            x_index = int(x_axis_name[1:]) - 1
            y_index = int(y_axis_name[1:]) - 1

            if x_index == y_index:
                raise ValueError(
                    "Для фазового графика нужно выбрать разные оси."
                )

            if (
                x_index >= solution.states.shape[1]
                or y_index >= solution.states.shape[1]
            ):
                raise ValueError(
                    "Выбранные оси не соответствуют размерности системы."
                )

            axis.plot(
                solution.states[:, x_index],
                solution.states[:, y_index],
                label="Траектория",
            )
            axis.scatter(
                solution.states[0, x_index],
                solution.states[0, y_index],
                label="Начало",
            )
            axis.scatter(
                solution.states[-1, x_index],
                solution.states[-1, y_index],
                label="Конец",
            )

            axis.set_xlabel(x_axis_name)
            axis.set_ylabel(y_axis_name)
            axis.set_title(f"Фазовая траектория {x_axis_name}-{y_axis_name}")
            axis.axis("equal")

        elif plot_type == "Сходимость":
            iterations = list(range(len(solution.residual_history)))

            axis.plot(
                iterations,
                solution.residual_history,
                marker="o",
                label="||Phi(p)||",
            )

            axis.set_xlabel("Номер итерации")
            axis.set_ylabel("Норма ошибки")
            axis.set_title("Сходимость метода")
            axis.set_yscale("log")

        else:
            for index in range(solution.states.shape[1]):
                axis.plot(
                    solution.t,
                    solution.states[:, index],
                    label=f"x{index + 1}(t)",
                )

            axis.set_xlabel("t")
            axis.set_ylabel("Значение")
            axis.set_title("Компоненты решения от времени")

        axis.grid(True)
        axis.legend()
        figure.tight_layout()

        self.canvas = FigureCanvasTkAgg(figure, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        plt.close(figure)


def run_app() -> None:
    root = tk.Tk()
    BVPApp(root)
    root.mainloop()
