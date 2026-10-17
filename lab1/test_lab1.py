import numpy as np
import pytest

from grader_contracts.python_basics import PositiveIntegerInput as P, TextInput as T, VectorPairInput as V
from grader_contracts.numpy_tasks import *
import python_basics as pb
import numpy_tasks as nt


# ---------- Часть 1 ----------
def test_vowels():
    assert pb.count_vowels(T("")) == 0
    assert pb.count_vowels(T("AeIoU")) == 5
    assert pb.count_vowels(T("rhythm")) == 0
    assert pb.count_vowels(T("Hello World")) == 3

def test_unique():
    assert pb.has_unique_characters(T("")) is True
    assert pb.has_unique_characters(T("abc")) is True
    assert pb.has_unique_characters(T("abca")) is False
    assert pb.has_unique_characters(T("aA")) is True

def test_bits():
    assert pb.count_one_bits(P(1)) == 1
    assert pb.count_one_bits(P(7)) == 3
    assert pb.count_one_bits(P(1024)) == 1
    assert pb.count_one_bits(P(2**64 - 1)) == 64

def test_persistence():
    assert pb.multiplicative_persistence(P(39)) == 3
    assert pb.multiplicative_persistence(P(4)) == 0
    assert pb.multiplicative_persistence(P(999)) == 4
    assert pb.multiplicative_persistence(P(10)) == 1
    assert pb.multiplicative_persistence(P(25)) == 2

def test_mse():
    assert pb.mse(V([1, 2, 3], [1, 2, 3])) == 0
    assert pb.mse(V([1, 2], [3, 6])) == 10
    assert pb.mse(V([5], [2])) == 9
    with pytest.raises(ValueError):
        pb.mse(V([1], [1, 2]))

def test_factorization():
    assert pb.prime_factorization(P(86240)) == "(2**5)(5)(7**2)(11)"
    assert pb.prime_factorization(P(2)) == "(2)"
    assert pb.prime_factorization(P(17)) == "(17)"
    assert pb.prime_factorization(P(1024)) == "(2**10)"
    assert pb.prime_factorization(P(9973 * 2)) == "(2)(9973)"
    assert pb.prime_factorization(P(1)) == ""

def test_pyramid():
    assert pb.pyramid(P(1)) == 1
    assert pb.pyramid(P(5)) == 2
    assert pb.pyramid(P(14)) == 3
    assert pb.pyramid(P(30)) == 4
    assert pb.pyramid(P(2)) == "It is impossible"
    assert pb.pyramid(P(15)) == "It is impossible"

def test_balanced():
    assert pb.is_balanced_number(P(1234006)) is True
    assert pb.is_balanced_number(P(123456)) is False
    assert pb.is_balanced_number(P(7)) is True
    assert pb.is_balanced_number(P(12)) is True
    assert pb.is_balanced_number(P(1221)) is True
    assert pb.is_balanced_number(P(13541)) is False
    assert pb.is_balanced_number(P(1111)) is True
    assert pb.is_balanced_number(P(1230)) is False


# ---------- Часть 2 ----------
def test_sum_prod():
    A = np.random.default_rng(0).integers(-5, 5, (4, 3, 3))
    x = np.random.default_rng(1).integers(-5, 5, (4, 3, 1))
    expected = sum(A[i] @ x[i] for i in range(4))
    res = nt.sum_prod(MatrixVectorBatchInput(A, x))
    assert res.shape == (3, 1)
    assert np.array_equal(res, expected)
    assert nt.sum_prod(MatrixVectorBatchInput(A[:1], x[:1])).shape == (3, 1)

def test_binarize():
    M = np.array([[0.1, 0.5], [0.6, 2.0]])
    res = nt.binarize(BinarizeInput(M, 0.5))
    assert np.array_equal(res, [[0, 0], [1, 1]])
    assert np.array_equal(M, [[0.1, 0.5], [0.6, 2.0]])  # исходная не изменена

def test_unique():
    M = np.array([[1, 1, 2], [3, 3, 3], [1, 2, 1]])
    assert nt.unique_rows(MatrixInput(M)) == [[1, 2], [3], [1, 2]]
    assert nt.unique_columns(MatrixInput(M)) == [[1, 3], [1, 2, 3], [1, 2, 3]]

def test_matrix_statistics():
    s = nt.matrix_statistics(RandomMatrixInput(50, 40, 3.0, 2.0, 7))
    assert s.matrix.shape == (50, 40)
    assert s.row_means.shape == (50,) and s.column_means.shape == (40,)
    assert np.allclose(s.row_variances, s.matrix.var(axis=1))
    assert np.allclose(s.column_variances, s.matrix.var(axis=0))
    s2 = nt.matrix_statistics(RandomMatrixInput(50, 40, 3.0, 2.0, 7))
    assert np.array_equal(s.matrix, s2.matrix)  # воспроизводимость
    big = nt.matrix_statistics(RandomMatrixInput(1000, 1000, 3.0, 2.0, 1))
    assert abs(big.matrix.mean() - 3.0) < 0.05 and abs(big.matrix.std() - 2.0) < 0.05

def test_chess():
    res = nt.chess(ChessInput(3, 4, 0, 1))
    assert np.array_equal(res, [[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]])
    assert nt.chess(ChessInput(1, 1, 5, 9)).tolist() == [[5]]
    assert nt.chess(ChessInput(2, 1, 5, 9)).tolist() == [[5], [9]]

def test_rectangle():
    img = nt.draw_rectangle(RectangleInput(4, 2, 6, 8, (255, 0, 0), (0, 0, 0)))
    assert img.shape == (6, 8, 3)
    mask = (img == [255, 0, 0]).all(axis=2)
    assert mask.sum() == 8
    assert mask[2:4, 2:6].all()
    full = nt.draw_rectangle(RectangleInput(8, 6, 6, 8, (1, 2, 3), (0, 0, 0)))
    assert (full == [1, 2, 3]).all()
    big = nt.draw_rectangle(RectangleInput(100, 100, 6, 8, (1, 2, 3), (0, 0, 0)))
    assert big.shape == (6, 8, 3)

def test_ellipse():
    img = nt.draw_ellipse(EllipseInput(3, 2, 9, 11, (0, 255, 0), (10, 10, 10)))
    assert img.shape == (9, 11, 3)
    mask = (img == [0, 255, 0]).all(axis=2)
    assert mask[4, 5]
    assert mask[4, 2] and mask[4, 8]     # края по x: центр ± 3
    assert not mask[0, 0]
    assert (mask == mask[::-1]).all() and (mask == mask[:, ::-1]).all()  # симметрия

def test_time_series():
    r = nt.analyze_time_series(TimeSeriesInput([1, 3, 2, 4, 4, 1, 5], 3))
    assert np.array_equal(r.local_maxima_indices, [1])
    assert np.array_equal(r.local_minima_indices, [2, 5])
    assert np.allclose(r.moving_average, [2, 3, 10/3, 3, 10/3])
    v = np.array([1, 3, 2, 4, 4, 1, 5.])
    assert np.isclose(r.mean, v.mean()) and np.isclose(r.variance, v.var()) and np.isclose(r.std, v.std())
    short = nt.analyze_time_series(TimeSeriesInput([1, 2], 2))
    assert len(short.local_maxima_indices) == 0 and len(short.local_minima_indices) == 0
    assert np.allclose(short.moving_average, [1.5])
    same = nt.analyze_time_series(TimeSeriesInput([1, 2, 3], 1))
    assert np.allclose(same.moving_average, [1, 2, 3])
    with pytest.raises(ValueError):
        nt.analyze_time_series(TimeSeriesInput([1, 2], 5))

def test_one_hot():
    res = nt.one_hot(OneHotInput(np.array([0, 2, 3, 0])))
    assert np.array_equal(res, [[1,0,0,0],[0,0,1,0],[0,0,0,1],[1,0,0,0]])
    assert nt.one_hot(OneHotInput([0, 1], 5)).shape == (2, 5)
    assert nt.one_hot(OneHotInput([0, 0])).tolist() == [[1], [1]]
