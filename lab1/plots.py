"""Гистограммы для задачи 4 (matrix_statistics)."""

import matplotlib.pyplot as plt

from grader_contracts.numpy_tasks import RandomMatrixInput
from numpy_tasks import matrix_statistics


def plot_statistics(stats, show=True):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    items = [
        (stats.row_means, "Средние по строкам"),
        (stats.column_means, "Средние по столбцам"),
        (stats.row_variances, "Дисперсии по строкам"),
        (stats.column_variances, "Дисперсии по столбцам"),
    ]
    for ax, (values, title) in zip(axes.ravel(), items):
        ax.hist(values, bins=20, edgecolor="black")
        ax.set_title(title)
    fig.tight_layout()
    if show:
        plt.show()
    return fig


if __name__ == "__main__":
    plot_statistics(matrix_statistics(RandomMatrixInput(rows=500, columns=300, mean=2.0, std=3.0, seed=42)))
