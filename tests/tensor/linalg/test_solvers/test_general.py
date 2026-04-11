import functools

import numpy as np
import pytest
import scipy

from pytensor import function
from pytensor import tensor as pt
from pytensor.configdefaults import config
from pytensor.graph.basic import equal_computations
from pytensor.tensor import TensorVariable
from pytensor.tensor.linalg import (
    Solve,
    SolveBase,
    lu_factor,
    lu_solve,
    solve,
    solve_triangular,
)
from pytensor.tensor.type import matrix, tensor, vector
from tests import unittest_tools as utt


class TestSolveBase:
    class SolveTest(SolveBase):
        def perform(self, node, inputs, outputs):
            A, b = inputs
            outputs[0][0] = scipy.linalg.solve(A, b)

    @pytest.mark.parametrize(
        "A_func, b_func, error_message",
        [
            (vector, matrix, "`A` must be a matrix.*"),
            (
                functools.partial(tensor, dtype="floatX", shape=(None,) * 3),
                matrix,
                "`A` must be a matrix.*",
            ),
            (
                matrix,
                functools.partial(tensor, dtype="floatX", shape=(None,) * 3),
                "`b` must have 2 dims.*",
            ),
        ],
    )
    def test_make_node(self, A_func, b_func, error_message):
        np.random.default_rng(utt.fetch_seed())
        with pytest.raises(ValueError, match=error_message):
            A = A_func()
            b = b_func()
            self.SolveTest(b_ndim=2)(A, b)

    def test__repr__(self):
        np.random.default_rng(utt.fetch_seed())
        A = matrix()
        b = matrix()
        y = self.SolveTest(b_ndim=2)(A, b)
        assert (
            y.__repr__()
            == "SolveTest{lower=False, b_ndim=2, overwrite_a=False, overwrite_b=False}.0"
        )


def test_solve_raises_on_invalid_assume_a():
    with pytest.raises(ValueError, match="Invalid assume_a: test\\. It must be one of"):
        Solve(assume_a="test", b_ndim=2)


solve_test_cases = [
    ("gen", False, False),
    ("gen", False, True),
    ("sym", False, False),
    ("sym", True, False),
    ("sym", True, True),
    ("pos", False, False),
    ("pos", True, False),
    ("pos", True, True),
    ("diagonal", False, False),
    ("diagonal", False, True),
    ("tridiagonal", False, False),
    ("tridiagonal", False, True),
]
solve_test_ids = [
    f"{assume_a}_{'lower' if lower else 'upper'}_{'A^T' if transposed else 'A'}"
    for assume_a, lower, transposed in solve_test_cases
]


class TestSolve(utt.InferShapeTester):
    @staticmethod
    def A_func(x, assume_a):
        if assume_a == "pos":
            return x @ x.T
        elif assume_a == "sym":
            return (x + x.T) / 2
        elif assume_a == "diagonal":
            eye_fn = pt.eye if isinstance(x, TensorVariable) else np.eye
            return x * eye_fn(x.shape[1])
        elif assume_a == "tridiagonal":
            eye_fn = pt.eye if isinstance(x, TensorVariable) else np.eye
            return x * (
                eye_fn(x.shape[1], k=0)
                + eye_fn(x.shape[1], k=-1)
                + eye_fn(x.shape[1], k=1)
            )
        else:
            return x

    @staticmethod
    def T(x, transposed):
        if transposed:
            return x.T
        return x

    @pytest.mark.parametrize("b_shape", [(5, 1), (5,)])
    def test_infer_shape(self, b_shape):
        rng = np.random.default_rng(utt.fetch_seed())
        A = matrix()
        b_val = np.asarray(rng.random(b_shape), dtype=config.floatX)
        b = pt.as_tensor_variable(b_val).type()
        self._compile_and_check(
            [A, b],
            [solve(A, b)],
            [
                np.asarray(rng.random((5, 5)), dtype=config.floatX),
                b_val,
            ],
            Solve,
            warn=False,
        )

    @pytest.mark.parametrize(
        "b_size", [(5, 1), (5, 5), (5,)], ids=["b_col_vec", "b_matrix", "b_vec"]
    )
    @pytest.mark.parametrize(
        "assume_a, lower, transposed", solve_test_cases, ids=solve_test_ids
    )
    def test_solve_correctness(
        self, b_size: tuple[int], assume_a: str, lower: bool, transposed: bool
    ):
        rng = np.random.default_rng(utt.fetch_seed())
        A = pt.tensor("A", shape=(5, 5))
        b = pt.tensor("b", shape=b_size)

        A_val = rng.normal(size=(5, 5)).astype(config.floatX)
        b_val = rng.normal(size=b_size).astype(config.floatX)

        A_func = functools.partial(self.A_func, assume_a=assume_a)
        T = functools.partial(self.T, transposed=transposed)

        y = solve(
            A_func(A),
            b,
            assume_a=assume_a,
            lower=lower,
            transposed=transposed,
            b_ndim=len(b_size),
        )

        solve_func = function([A, b], y)
        X_np = solve_func(A_val.copy(), b_val.copy())

        ATOL = 1e-8 if config.floatX.endswith("64") else 1e-4
        RTOL = 1e-8 if config.floatX.endswith("64") else 1e-4

        np.testing.assert_allclose(
            scipy.linalg.solve(
                A_func(A_val),
                b_val,
                assume_a=assume_a,
                transposed=transposed,
                lower=lower,
            ),
            X_np,
            atol=ATOL,
            rtol=RTOL,
        )

        np.testing.assert_allclose(T(A_func(A_val)) @ X_np, b_val, atol=ATOL, rtol=RTOL)

    @pytest.mark.parametrize(
        "b_size", [(5, 1), (5, 5), (5,)], ids=["b_col_vec", "b_matrix", "b_vec"]
    )
    @pytest.mark.parametrize(
        "assume_a, lower, transposed",
        solve_test_cases,
        ids=solve_test_ids,
    )
    @pytest.mark.skipif(
        config.floatX == "float32", reason="Gradients not numerically stable in float32"
    )
    def test_solve_gradient(
        self, b_size: tuple[int], assume_a: str, lower: bool, transposed: bool
    ):
        rng = np.random.default_rng(utt.fetch_seed())

        eps = 2e-8 if config.floatX == "float64" else None

        A_val = rng.normal(size=(5, 5)).astype(config.floatX)
        b_val = rng.normal(size=b_size).astype(config.floatX)

        solve_op = functools.partial(solve, assume_a=assume_a, b_ndim=len(b_size))
        A_func = functools.partial(self.A_func, assume_a=assume_a)

        utt.verify_grad(
            lambda A, b: solve_op(A_func(A), b), [A_val, b_val], 3, rng, eps=eps
        )

    def test_solve_tringular_indirection(self):
        a = pt.matrix("a")
        b = pt.vector("b")

        indirect = solve(a, b, assume_a="lower triangular")
        direct = solve_triangular(a, b, lower=True, trans=False)
        assert equal_computations([indirect], [direct])

        indirect = solve(a, b, assume_a="upper triangular")
        direct = solve_triangular(a, b, lower=False, trans=False)
        assert equal_computations([indirect], [direct])

        indirect = solve(a, b, assume_a="upper triangular", transposed=True)
        direct = solve_triangular(a, b, lower=False, trans=True)
        assert equal_computations([indirect], [direct])


class TestLUSolve(utt.InferShapeTester):
    @staticmethod
    def factor_and_solve(A, b, sum=False, **lu_kwargs):
        lu_and_pivots = lu_factor(A)
        x = lu_solve(lu_and_pivots, b, **lu_kwargs)
        if not sum:
            return x
        return x.sum()

    @pytest.mark.parametrize("b_shape", [(5,), (5, 5)], ids=["b_vec", "b_matrix"])
    @pytest.mark.parametrize("trans", [True, False], ids=["x_T", "x"])
    def test_lu_solve(self, b_shape: tuple[int], trans):
        rng = np.random.default_rng(utt.fetch_seed())
        A = pt.tensor("A", shape=(5, 5))
        b = pt.tensor("b", shape=b_shape)

        A_val = (
            rng.normal(size=(5, 5)).astype(config.floatX)
            + np.eye(5, dtype=config.floatX) * 0.5
        )
        b_val = rng.normal(size=b_shape).astype(config.floatX)

        x = self.factor_and_solve(A, b, trans=trans, sum=False)

        f = function([A, b], x)
        x_pt = f(A_val.copy(), b_val.copy())
        x_sp = scipy.linalg.lu_solve(
            scipy.linalg.lu_factor(A_val.copy()), b_val.copy(), trans=trans
        )

        np.testing.assert_allclose(x_pt, x_sp)

        def T(x):
            if trans:
                return x.T
            return x

        np.testing.assert_allclose(
            T(A_val) @ x_pt,
            b_val,
            atol=1e-8 if config.floatX == "float64" else 1e-4,
            rtol=1e-8 if config.floatX == "float64" else 1e-4,
        )
        np.testing.assert_allclose(x_pt, x_sp)

    @pytest.mark.parametrize("b_shape", [(5,), (5, 5)], ids=["b_vec", "b_matrix"])
    @pytest.mark.parametrize("trans", [True, False], ids=["x_T", "x"])
    def test_lu_solve_gradient(self, b_shape: tuple[int], trans: bool):
        rng = np.random.default_rng(utt.fetch_seed())

        A_val = rng.normal(size=(5, 5)).astype(config.floatX)
        b_val = rng.normal(size=b_shape).astype(config.floatX)

        test_fn = functools.partial(self.factor_and_solve, sum=True, trans=trans)
        utt.verify_grad(test_fn, [A_val, b_val], 3, rng)

    def test_lu_solve_batch_dims(self):
        A = pt.tensor("A", shape=(3, 1, 5, 5))
        b = pt.tensor("b", shape=(1, 4, 5))
        lu_and_pivots = lu_factor(A)
        x = lu_solve(lu_and_pivots, b, b_ndim=1)
        assert x.type.shape in {(3, 4, None), (3, 4, 5)}

        rng = np.random.default_rng(748)
        A_test = rng.random(A.type.shape).astype(A.type.dtype)
        b_test = rng.random(b.type.shape).astype(b.type.dtype)
        np.testing.assert_allclose(
            x.eval({A: A_test, b: b_test}),
            solve(A, b, b_ndim=1).eval({A: A_test, b: b_test}),
            rtol=1e-9 if config.floatX == "float64" else 1e-5,
        )
