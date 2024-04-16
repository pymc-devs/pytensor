import numpy as np

import pytensor
from pytensor.tensor.sort import ArgSortOp, SortOp, argsort, sort
from pytensor.tensor.type import (
    dmatrix,
    dvector,
    float_dtypes,
    integer_dtypes,
    lscalar,
    matrix,
    scalar,
)
from tests import unittest_tools as utt


_all_dtypes = integer_dtypes + float_dtypes


def gen_unique_vector(size, dtype):
    rng = np.random.default_rng(utt.fetch_seed())
    # generate a randomized vector with unique elements
    retval = np.arange(size) * 3.0 + rng.uniform(-1.0, 1.0)
    return (retval[rng.permutation(size)] - size * 1.5).astype(dtype)


class TestSort:
    def setup_method(self):
        self.rng = np.random.default_rng(seed=utt.fetch_seed())
        self.m_val = self.rng.random((3, 2))
        self.v_val = self.rng.random(4)

    def test1(self):
        a = dmatrix()
        w = sort(a)
        f = pytensor.function([a], w)
        utt.assert_allclose(f(self.m_val), np.sort(self.m_val))

    def test2(self):
        a = dmatrix()
        axis = scalar()
        w = sort(a, axis)
        f = pytensor.function([a, axis], w)
        for axis_val in 0, 1:
            gv = f(self.m_val, axis_val)
            gt = np.sort(self.m_val, axis_val)
            utt.assert_allclose(gv, gt)

    def test3(self):
        a = dvector()
        w2 = sort(a)
        f = pytensor.function([a], w2)
        gv = f(self.v_val)
        gt = np.sort(self.v_val)
        utt.assert_allclose(gv, gt)

    def test4(self):
        a = dmatrix()
        axis = scalar()
        l = sort(a, axis, "mergesort")
        f = pytensor.function([a, axis], l)
        for axis_val in 0, 1:
            gv = f(self.m_val, axis_val)
            gt = np.sort(self.m_val, axis_val)
            utt.assert_allclose(gv, gt)

    def test5(self):
        a1 = SortOp("mergesort", [])
        a2 = SortOp("quicksort", [])

        # All the below should give true
        assert a1 != a2
        assert a1 == SortOp("mergesort", [])
        assert a2 == SortOp("quicksort", [])

    def test_None(self):
        a = dmatrix()
        l = sort(a, None)
        f = pytensor.function([a], l)
        gv = f(self.m_val)
        gt = np.sort(self.m_val, None)
        utt.assert_allclose(gv, gt)

    def test_grad_vector(self):
        data = self.rng.random(10).astype(pytensor.config.floatX)
        utt.verify_grad(sort, [data])

    def test_grad_none_axis(self):
        data = self.rng.random(10).astype(pytensor.config.floatX)
        utt.verify_grad(lambda x: sort(x, None), [data])
        utt.verify_grad(lambda x: sort(x, 0), [data])

        data = self.rng.random((2, 3)).astype(pytensor.config.floatX)
        utt.verify_grad(lambda x: sort(x, None), [data])
        data = self.rng.random((2, 3, 4)).astype(pytensor.config.floatX)
        utt.verify_grad(lambda x: sort(x, None), [data])

    def test_grad_negative_axis_2d(self):
        data = self.rng.random((2, 3)).astype(pytensor.config.floatX)
        utt.verify_grad(lambda x: sort(x, -1), [data])
        data = self.rng.random((2, 3)).astype(pytensor.config.floatX)
        utt.verify_grad(lambda x: sort(x, -2), [data])

    def test_grad_negative_axis_3d(self):
        data = self.rng.random((2, 3, 4)).astype(pytensor.config.floatX)
        utt.verify_grad(lambda x: sort(x, -1), [data])
        data = self.rng.random((2, 3, 4)).astype(pytensor.config.floatX)
        utt.verify_grad(lambda x: sort(x, -2), [data])
        data = self.rng.random((2, 3, 4)).astype(pytensor.config.floatX)
        utt.verify_grad(lambda x: sort(x, -3), [data])

    def test_grad_negative_axis_4d(self):
        data = self.rng.random((2, 3, 4, 2)).astype(pytensor.config.floatX)
        utt.verify_grad(lambda x: sort(x, -1), [data])
        data = self.rng.random((2, 3, 4, 2)).astype(pytensor.config.floatX)
        utt.verify_grad(lambda x: sort(x, -2), [data])
        data = self.rng.random((2, 3, 4, 2)).astype(pytensor.config.floatX)
        utt.verify_grad(lambda x: sort(x, -3), [data])
        data = self.rng.random((2, 3, 4, 2)).astype(pytensor.config.floatX)
        utt.verify_grad(lambda x: sort(x, -4), [data])

    def test_grad_nonnegative_axis_2d(self):
        data = self.rng.random((2, 3)).astype(pytensor.config.floatX)
        utt.verify_grad(lambda x: sort(x, 0), [data])
        data = self.rng.random((2, 3)).astype(pytensor.config.floatX)
        utt.verify_grad(lambda x: sort(x, 1), [data])

    def test_grad_nonnegative_axis_3d(self):
        data = self.rng.random((2, 3, 4)).astype(pytensor.config.floatX)
        utt.verify_grad(lambda x: sort(x, 0), [data])
        data = self.rng.random((2, 3, 4)).astype(pytensor.config.floatX)
        utt.verify_grad(lambda x: sort(x, 1), [data])
        data = self.rng.random((2, 3, 4)).astype(pytensor.config.floatX)
        utt.verify_grad(lambda x: sort(x, 2), [data])

    def test_grad_nonnegative_axis_4d(self):
        data = self.rng.random((2, 3, 4, 2)).astype(pytensor.config.floatX)
        utt.verify_grad(lambda x: sort(x, 0), [data])
        data = self.rng.random((2, 3, 4, 2)).astype(pytensor.config.floatX)
        utt.verify_grad(lambda x: sort(x, 1), [data])
        data = self.rng.random((2, 3, 4, 2)).astype(pytensor.config.floatX)
        utt.verify_grad(lambda x: sort(x, 2), [data])
        data = self.rng.random((2, 3, 4, 2)).astype(pytensor.config.floatX)
        utt.verify_grad(lambda x: sort(x, 3), [data])


class TestSortInferShape(utt.InferShapeTester):
    def setup_method(self):
        self.rng = np.random.default_rng(seed=utt.fetch_seed())
        super().setup_method()

    def test_sort(self):
        x = matrix()
        self._compile_and_check(
            [x],
            [sort(x)],
            [self.rng.standard_normal(size=(10, 40)).astype(pytensor.config.floatX)],
            SortOp,
        )
        self._compile_and_check(
            [x],
            [sort(x, axis=None)],
            [self.rng.standard_normal(size=(10, 40)).astype(pytensor.config.floatX)],
            SortOp,
        )


def test_argsort():
    # Set up
    rng = np.random.default_rng(seed=utt.fetch_seed())
    m_val = rng.random((3, 2))
    v_val = rng.random(4)

    # Example 1
    a = dmatrix()
    w = argsort(a)
    f = pytensor.function([a], w)
    gv = f(m_val)
    gt = np.argsort(m_val)
    utt.assert_allclose(gv, gt)

    # Example 2
    a = dmatrix()
    axis = lscalar()
    w = argsort(a, axis)
    f = pytensor.function([a, axis], w)
    for axis_val in 0, 1:
        gv = f(m_val, axis_val)
        gt = np.argsort(m_val, axis_val)
        utt.assert_allclose(gv, gt)

    # Example 3
    a = dvector()
    w2 = argsort(a)
    f = pytensor.function([a], w2)
    gv = f(v_val)
    gt = np.argsort(v_val)
    utt.assert_allclose(gv, gt)

    # Example 4
    a = dmatrix()
    axis = lscalar()
    l = argsort(a, axis, "mergesort")
    f = pytensor.function([a, axis], l)
    for axis_val in 0, 1:
        gv = f(m_val, axis_val)
        gt = np.argsort(m_val, axis_val)
        utt.assert_allclose(gv, gt)

    # Example 5
    a = dmatrix()
    axis = lscalar()
    a1 = ArgSortOp("mergesort", [])
    a2 = ArgSortOp("quicksort", [])
    # All the below should give true
    assert a1 != a2
    assert a1 == ArgSortOp("mergesort", [])
    assert a2 == ArgSortOp("quicksort", [])

    # Example 6: Testing axis=None
    a = dmatrix()
    w2 = argsort(a, None)
    f = pytensor.function([a], w2)
    gv = f(m_val)
    gt = np.argsort(m_val, None)
    utt.assert_allclose(gv, gt)


def test_argsort_grad():
    rng = np.random.default_rng(seed=utt.fetch_seed())
    # Testing grad of argsort
    data = rng.random((2, 3)).astype(pytensor.config.floatX)
    utt.verify_grad(lambda x: argsort(x, axis=-1), [data])

    data = rng.random((2, 3, 4, 5)).astype(pytensor.config.floatX)
    utt.verify_grad(lambda x: argsort(x, axis=-3), [data])

    data = rng.random((2, 3, 3)).astype(pytensor.config.floatX)
    utt.verify_grad(lambda x: argsort(x, axis=2), [data])
