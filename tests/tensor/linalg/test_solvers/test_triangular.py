import functools

import numpy as np
import pytest
import scipy

from pytensor import function
from pytensor import tensor as pt
from pytensor.configdefaults import config
from pytensor.tensor.linalg import SolveTriangular, cholesky, solve_triangular
from pytensor.tensor.type import matrix
from tests import unittest_tools as utt


class TestSolveTriangular(utt.InferShapeTester):
    @staticmethod
    def A_func(x, lower, unit_diagonal):
        x = x @ x.T
        x = cholesky(x, lower=lower)
        if unit_diagonal:
            x = pt.fill_diagonal(x, 1)
        return x

    @staticmethod
    def T(x, trans):
        if trans == 1:
            return x.T
        elif trans == 2:
            return x.conj().T
        return x

    @pytest.mark.parametrize("b_shape", [(5, 1), (5,)])
    def test_infer_shape(self, b_shape):
        rng = np.random.default_rng(utt.fetch_seed())
        A = matrix()
        b_val = rng.random(b_shape).astype(config.floatX)
        b = pt.as_tensor_variable(b_val).type()
        self._compile_and_check(
            [A, b],
            [solve_triangular(A, b)],
            [
                np.asarray(rng.random((5, 5)), dtype=config.floatX),
                b_val,
            ],
            SolveTriangular,
            warn=False,
        )

    @pytest.mark.parametrize(
        "b_shape", [(5, 1), (5,), (5, 5)], ids=["b_col_vec", "b_vec", "b_matrix"]
    )
    @pytest.mark.parametrize("lower", [True, False])
    @pytest.mark.parametrize("trans", [0, 1, 2])
    @pytest.mark.parametrize("unit_diagonal", [True, False])
    def test_correctness(self, b_shape: tuple[int], lower, trans, unit_diagonal):
        rng = np.random.default_rng(utt.fetch_seed())
        A = pt.tensor("A", shape=(5, 5))
        b = pt.tensor("b", shape=b_shape)

        A_val = rng.random((5, 5)).astype(config.floatX)
        b_val = rng.random(b_shape).astype(config.floatX)

        A_func = functools.partial(
            self.A_func, lower=lower, unit_diagonal=unit_diagonal
        )

        x = solve_triangular(
            A_func(A),
            b,
            lower=lower,
            trans=trans,
            unit_diagonal=unit_diagonal,
            b_ndim=len(b_shape),
        )

        f = function([A, b], x)

        x_pt = f(A_val, b_val)
        x_sp = scipy.linalg.solve_triangular(
            A_func(A_val).eval(),
            b_val,
            lower=lower,
            trans=trans,
            unit_diagonal=unit_diagonal,
        )

        np.testing.assert_allclose(
            x_pt,
            x_sp,
            atol=1e-8 if config.floatX == "float64" else 1e-4,
            rtol=1e-8 if config.floatX == "float64" else 1e-4,
        )

    @pytest.mark.parametrize(
        "b_shape", [(5, 1), (5,), (5, 5)], ids=["b_col_vec", "b_vec", "b_matrix"]
    )
    @pytest.mark.parametrize("lower", [True, False])
    @pytest.mark.parametrize("trans", [0, 1])
    @pytest.mark.parametrize("unit_diagonal", [True, False])
    def test_solve_triangular_grad(self, b_shape, lower, trans, unit_diagonal):
        if config.floatX == "float32":
            pytest.skip(reason="Not enough precision in float32 to get a good gradient")

        rng = np.random.default_rng(utt.fetch_seed())
        A_val = rng.normal(size=(5, 5)).astype(config.floatX)
        b_val = rng.normal(size=b_shape).astype(config.floatX)

        A_func = functools.partial(
            self.A_func, lower=lower, unit_diagonal=unit_diagonal
        )

        eps = None
        if config.floatX == "float64":
            eps = 2e-8

        def solve_op(A, b):
            return solve_triangular(
                A_func(A), b, lower=lower, trans=trans, unit_diagonal=unit_diagonal
            )

        utt.verify_grad(solve_op, [A_val, b_val], 3, rng, eps=eps)

    def test_solve_triangular_empty(self):
        rng = np.random.default_rng(utt.fetch_seed())
        A = pt.tensor("A", shape=(5, 5))
        b = pt.tensor("b", shape=(5, 0))

        A_val = rng.random((5, 5)).astype(config.floatX)
        b_empty = np.empty([5, 0], dtype=config.floatX)

        A_func = functools.partial(self.A_func, lower=True, unit_diagonal=True)

        x = solve_triangular(
            A_func(A),
            b,
            lower=True,
            trans=0,
            unit_diagonal=True,
            b_ndim=len((5, 0)),
        )

        f = function([A, b], x)

        res = f(A_val, b_empty)
        assert res.size == 0
        assert res.dtype == config.floatX
