"""Решения задач на NumPy."""

import numpy as np
from grader_contracts.numpy_tasks import (
    BinarizeInput, ChessInput, EllipseInput, MatrixInput, MatrixStatistics,
    MatrixVectorBatchInput, OneHotInput, RandomMatrixInput, RectangleInput,
    TimeSeriesInput, TimeSeriesStatistics,
)


def sum_prod(data: MatrixVectorBatchInput) -> np.ndarray:
    matrices, vectors = data.matrices, data.vectors
    matrices = np.asarray(matrices)   # (p, n, n)
    vectors = np.asarray(vectors)     # (p, n, 1)
    # matmul перемножает пары (n, n) @ (n, 1) по первой оси, затем суммируем
    return np.matmul(matrices, vectors).sum(axis=0)


def binarize(data: BinarizeInput) -> np.ndarray:
    matrix, threshold = data.matrix, data.threshold
    return (np.asarray(matrix) > threshold).astype(int)


def unique_rows(data: MatrixInput) -> list[list[float]]:
    matrix = np.asarray(data.matrix)
    return [np.unique(row).tolist() for row in matrix]


def unique_columns(data: MatrixInput) -> list[list[float]]:
    matrix = np.asarray(data.matrix)
    return [np.unique(col).tolist() for col in matrix.T]


def matrix_statistics(data: RandomMatrixInput) -> MatrixStatistics:
    rows, columns, mean, std, seed = data.rows, data.columns, data.mean, data.std, data.seed
    rng = np.random.default_rng(seed)
    matrix = rng.normal(loc=mean, scale=std, size=(rows, columns))
    return MatrixStatistics(
        matrix=matrix,
        row_means=matrix.mean(axis=1),
        column_means=matrix.mean(axis=0),
        row_variances=matrix.var(axis=1),
        column_variances=matrix.var(axis=0),
    )


def chess(data: ChessInput) -> np.ndarray:
    rows, columns, first, second = data.rows, data.columns, data.first, data.second
    i, j = np.indices((rows, columns))
    return np.where((i + j) % 2 == 0, first, second)


def draw_rectangle(data: RectangleInput) -> np.ndarray:
    width, height = data.width, data.height
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color

    image = np.empty((image_height, image_width, 3), dtype=np.uint8)
    image[:] = background_color

    top = max((image_height - height) // 2, 0)
    left = max((image_width - width) // 2, 0)
    bottom = min(top + height, image_height)
    right = min(left + width, image_width)
    if height > 0 and width > 0:
        image[top:bottom, left:right] = shape_color
    return image


def draw_ellipse(data: EllipseInput) -> np.ndarray:
    semi_axis_x, semi_axis_y = data.semi_axis_x, data.semi_axis_y
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color

    image = np.empty((image_height, image_width, 3), dtype=np.uint8)
    image[:] = background_color
    if semi_axis_x <= 0 or semi_axis_y <= 0:
        return image

    y0 = (image_height - 1) / 2
    x0 = (image_width - 1) / 2
    y, x = np.indices((image_height, image_width))
    inside = ((x - x0) / semi_axis_x) ** 2 + ((y - y0) / semi_axis_y) ** 2 <= 1
    image[inside] = shape_color
    return image


def analyze_time_series(data: TimeSeriesInput) -> TimeSeriesStatistics:
    values, window = data.values, data.window
    values = np.asarray(values, dtype=float)
    n = values.size
    if not 1 <= window <= n:
        raise ValueError("Размер окна должен быть от 1 до длины ряда")

    left, mid, right = values[:-2], values[1:-1], values[2:]
    maxima = np.flatnonzero((mid > left) & (mid > right)) + 1
    minima = np.flatnonzero((mid < left) & (mid < right)) + 1

    moving_average = np.convolve(values, np.ones(window) / window, mode="valid")

    return TimeSeriesStatistics(
        mean=float(values.mean()),
        variance=float(values.var()),
        std=float(values.std()),
        local_maxima_indices=maxima,
        local_minima_indices=minima,
        moving_average=moving_average,
    )


def one_hot(data: OneHotInput) -> np.ndarray:
    labels, class_count = data.labels, data.class_count
    labels = np.asarray(labels, dtype=int)
    if class_count is None:
        class_count = int(labels.max()) + 1 if labels.size else 0
    result = np.zeros((labels.size, class_count), dtype=int)
    result[np.arange(labels.size), labels] = 1
    return result
