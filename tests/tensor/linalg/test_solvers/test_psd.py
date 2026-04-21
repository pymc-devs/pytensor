import itertools

import numpy as np
import scipy

from pytensor import function
from pytensor.compile import get_default_mode
from pytensor.configdefaults import config
from pytensor.link.numba import NumbaLinker
from pytensor.tensor.linalg import CholeskySolve, cho_solve
from pytensor.tensor.type import matrix, vector
from tests import unittest_tools as utt


class TestCholeskySolve(utt.InferShapeTester):
    def setup_method(self):
        self.op_class = CholeskySolve
        super().setup_method()

    def test_repr(self):
        assert (
            repr(CholeskySolve(lower=True, b_ndim=1))
            == "CholeskySolve(lower=True,b_ndim=1,overwrite_b=False)"
        )

    def test_infer_shape(self):
        rng = np.random.default_rng(utt.fetch_seed())
        A = matrix()
        b = matrix()
        self._compile_and_check(
            [A, b],
            [self.op_class(b_ndim=2)(A, b)],
            [
                np.asarray(rng.random((5, 5)), dtype=config.floatX),
                np.asarray(rng.random((5, 1)), dtype=config.floatX),
            ],
            self.op_class,
            warn=False,
        )
        rng = np.random.default_rng(utt.fetch_seed())
        A = matrix()
        b = vector()
        self._compile_and_check(
            [A, b],
            [self.op_class(b_ndim=1)(A, b)],
            [
                np.asarray(rng.random((5, 5)), dtype=config.floatX),
                np.asarray(rng.random(5), dtype=config.floatX),
            ],
            self.op_class,
            warn=False,
        )

    def test_solve_correctness(self):
        rng = np.random.default_rng(utt.fetch_seed())
        A = matrix()
        b = matrix()
        y = self.op_class(lower=True, b_ndim=2)(A, b)
        cho_solve_lower_func = function([A, b], y)

        y = self.op_class(lower=False, b_ndim=2)(A, b)
        cho_solve_upper_func = function([A, b], y)

        b_val = np.asarray(rng.random((5, 1)), dtype=config.floatX)

        A_val = np.tril(np.asarray(rng.random((5, 5)), dtype=config.floatX))

        assert np.allclose(
            scipy.linalg.cho_solve((A_val, True), b_val),
            cho_solve_lower_func(A_val, b_val),
        )

        A_val = np.triu(np.asarray(rng.random((5, 5)), dtype=config.floatX))
        assert np.allclose(
            scipy.linalg.cho_solve((A_val, False), b_val),
            cho_solve_upper_func(A_val, b_val),
        )

        if config.floatX == "float64":
            M = rng.normal(size=(5, 5)).astype(config.floatX)
            C_val = np.linalg.cholesky(M @ M.T + np.eye(5, dtype=config.floatX))
            for lower, b_shape in itertools.product(
                (True, False), [(5, 1), (5, 5), (5,)]
            ):
                C = C_val if lower else C_val.T
                b_val = rng.normal(size=b_shape).astype(config.floatX)
                utt.verify_grad(
                    lambda c, b: cho_solve((c, lower), b, b_ndim=len(b_shape)),
                    [C, b_val],
                    3,
                    rng,
                    eps=2e-8,
                )

    def test_solve_dtype(self):
        is_numba = isinstance(get_default_mode().linker, NumbaLinker)

        dtypes = [
            "uint8",
            "uint16",
            "uint32",
            "uint64",
            "int8",
            "int16",
            "int32",
            "int64",
            "float16",
            "float32",
            "float64",
        ]

        A_val = np.eye(2)
        b_val = np.ones((2, 1))
        op = self.op_class(b_ndim=2)

        for A_dtype, b_dtype in itertools.product(dtypes, dtypes):
            if is_numba and (A_dtype == "float16" or b_dtype == "float16"):
                continue
            A = matrix(dtype=A_dtype)
            b = matrix(dtype=b_dtype)
            x = op(A, b)
            fn = function([A, b], x)
            x_result = fn(A_val.astype(A_dtype), b_val.astype(b_dtype))

            assert x.dtype == x_result.dtype, (A_dtype, b_dtype)


def test_cho_solve():
    rng = np.random.default_rng(utt.fetch_seed())
    A = matrix()
    b = matrix()
    y = cho_solve((A, True), b)
    cho_solve_lower_func = function([A, b], y)

    b_val = np.asarray(rng.random((5, 1)), dtype=config.floatX)

    A_val = np.tril(np.asarray(rng.random((5, 5)), dtype=config.floatX))

    assert np.allclose(
        scipy.linalg.cho_solve((A_val, True), b_val),
        cho_solve_lower_func(A_val, b_val),
    )


def test_cho_solve_empty():
    rng = np.random.default_rng(utt.fetch_seed())
    A = matrix()
    b = matrix()
    y = cho_solve((A, True), b)
    cho_solve_lower_func = function([A, b], y)

    A_empty = np.tril(np.asarray(rng.random((5, 5)), dtype=config.floatX))
    b_empty = np.empty([5, 0], dtype=config.floatX)

    res = cho_solve_lower_func(A_empty, b_empty)
    assert res.size == 0
    assert res.dtype == config.floatX
