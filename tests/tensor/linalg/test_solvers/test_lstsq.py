import numpy as np
import pytest

from pytensor import function
from pytensor.tensor.linalg import lstsq, tensorsolve
from pytensor.tensor.math import _allclose
from pytensor.tensor.type import lmatrix, lscalar, matrix, scalar, tensor4, vector
from tests import unittest_tools as utt


def test_tensorsolve():
    rng = np.random.default_rng(utt.fetch_seed())
    from pytensor.configdefaults import config

    A = tensor4("A", dtype=config.floatX)
    B = matrix("B", dtype=config.floatX)
    X = tensorsolve(A, B)
    fn = function([A, B], [X])

    # slightly modified example from np.linalg.tensorsolve docstring
    a = np.eye(2 * 3 * 4).astype(config.floatX)
    a.shape = (2 * 3, 4, 2, 3 * 4)
    b = rng.random((2 * 3, 4)).astype(config.floatX)

    n_x = np.linalg.tensorsolve(a, b)
    t_x = fn(a, b)
    assert _allclose(n_x, t_x)

    # check the type upcast now
    C = tensor4("C", dtype="float32")
    D = matrix("D", dtype="float64")
    Y = tensorsolve(C, D)
    fn = function([C, D], [Y])

    c = np.eye(2 * 3 * 4, dtype="float32")
    c.shape = (2 * 3, 4, 2, 3 * 4)
    d = rng.random((2 * 3, 4)).astype("float64")
    n_y = np.linalg.tensorsolve(c, d)
    t_y = fn(c, d)
    assert _allclose(n_y, t_y)
    assert n_y.dtype == Y.dtype

    # check the type upcast now
    E = tensor4("E", dtype="int32")
    F = matrix("F", dtype="float64")
    Z = tensorsolve(E, F)
    fn = function([E, F], [Z])

    e = np.eye(2 * 3 * 4, dtype="int32")
    e.shape = (2 * 3, 4, 2, 3 * 4)
    f = rng.random((2 * 3, 4)).astype("float64")
    n_z = np.linalg.tensorsolve(e, f)
    t_z = fn(e, f)
    assert _allclose(n_z, t_z)
    assert n_z.dtype == Z.dtype


class TestLstsq:
    def test_correct_solution(self):
        x = lmatrix()
        y = lmatrix()
        z = lscalar()
        b = lstsq(x, y, z)
        f = function([x, y, z], b)

        TestMatrix1 = np.asarray([[2, 1], [3, 4]])
        TestMatrix2 = np.asarray([[17, 20], [43, 50]])
        TestScalar = np.asarray(1)
        m0, _, rank, _ = f(TestMatrix1, TestMatrix2, TestScalar)
        assert rank.dtype == "int32"
        assert np.allclose(TestMatrix2, np.dot(TestMatrix1, m0))

    def test_wrong_coefficient_matrix(self):
        x = vector()
        y = vector()
        z = scalar()
        b = lstsq(x, y, z)
        f = function([x, y, z], b)
        with pytest.raises(np.linalg.LinAlgError):
            f([2, 1], [2, 1], 1)

    def test_wrong_rcond_dimension(self):
        x = vector()
        y = vector()
        z = vector()
        b = lstsq(x, y, z)
        f = function([x, y, z], b)
        with pytest.raises(np.linalg.LinAlgError):
            f([2, 1], [2, 1], [2, 1])
