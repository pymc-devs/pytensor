import functools

import numpy as np
import pytest
import scipy

from pytensor import function
from pytensor import tensor as pt
from pytensor.assumptions.specify import assume
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

    @staticmethod
    def _op_names(fn):
        return [
            type(getattr(node.op, "core_op", node.op)).__name__
            for node in fn.maker.fgraph.apply_nodes
        ]

    @pytest.mark.skipif(
        config.mode == "FAST_COMPILE", reason="Consumers rely on rewrites"
    )
    @pytest.mark.parametrize("assume_a", ["sym", "pos", "her"])
    def test_assume_a_records_an_assumption_about_a(self, assume_a):
        """``assume_a`` promises a property of ``a``, so other readers of ``a`` get it.

        Nothing else in the graph tells ``eig`` that ``a`` is symmetric, and ``pos``
        reaches ``eigh`` through the implication that positive definite matrices are.
        """
        a, b = matrix("a"), matrix("b")
        w, _ = pt.linalg.eig(a)
        fn = function([a, b], [solve(a, b, assume_a=assume_a), w])

        op_names = self._op_names(fn)
        assert "Eigh" in op_names
        assert "Eig" not in op_names
        assert "SpecifyAssumptions" not in op_names, (
            "the marker must not outlive the drain"
        )

        rng = np.random.default_rng(31)
        X = rng.normal(size=(6, 6)).astype(config.floatX)
        a_val = X @ X.T + 6 * np.eye(6, dtype=config.floatX)
        b_val = rng.normal(size=(6, 2)).astype(config.floatX)

        ATOL = 1e-8 if config.floatX.endswith("64") else 1e-4
        RTOL = 1e-8 if config.floatX.endswith("64") else 1e-4
        solved, eigenvalues = fn(a_val, b_val)
        np.testing.assert_allclose(
            solved, np.linalg.solve(a_val, b_val), atol=ATOL, rtol=RTOL
        )
        np.testing.assert_allclose(
            np.sort(eigenvalues), np.linalg.eigvalsh(a_val), atol=ATOL, rtol=RTOL
        )

    @pytest.mark.skipif(
        config.mode == "FAST_COMPILE", reason="Consumers rely on rewrites"
    )
    def test_hermitian_is_not_recorded_as_symmetric_for_complex_input(self):
        """A complex Hermitian matrix satisfies ``a.conj().T == a``, not ``a.T == a``.

        Recording it as symmetric would license every rewrite that transposes ``a``
        freely, so the promise stops at the solve for complex dtypes.
        """
        a = matrix("a", dtype="complex128")
        b = matrix("b", dtype="complex128")
        w, _ = pt.linalg.eig(a)
        fn = function([a, b], [solve(a, b, assume_a="her"), w])

        op_names = self._op_names(fn)
        assert "Eig" in op_names
        assert "Eigh" not in op_names

    @pytest.mark.skipif(
        config.mode == "FAST_COMPILE", reason="Consumers rely on rewrites"
    )
    def test_rewrite_built_solve_records_nothing(self):
        """``inv_to_solve`` builds a solve from an assumption it has already read.

        Recording it again would leave a marker behind, as rewriting runs long after
        the pass that resolves them.
        """
        X, r = matrix("X"), matrix("r")
        fn = function([X, r], pt.linalg.inv(assume(X, positive_definite=True)) @ r)

        assert "SpecifyAssumptions" not in self._op_names(fn)

    @pytest.mark.skipif(
        config.mode == "FAST_COMPILE", reason="Consumers rely on rewrites"
    )
    def test_assume_a_diagonal_records_an_assumption_about_a(self):
        """``assume_a='diagonal'`` lowers to a division, erasing the promise.

        Recording it first keeps the property available to every other reader of ``a``,
        each of which drops from a dense op to an elementwise one.
        """
        a, b, c = matrix("a"), matrix("b"), matrix("c")
        fn = function(
            [a, b, c], [solve(a, b, assume_a="diagonal"), a @ c, pt.linalg.det(a)]
        )

        op_names = self._op_names(fn)
        assert "Dot" not in op_names
        assert "Det" not in op_names

        rng = np.random.default_rng(42)
        a_val = np.diag(rng.normal(size=6) + 5.0).astype(config.floatX)
        b_val = rng.normal(size=(6, 2)).astype(config.floatX)
        c_val = rng.normal(size=(6, 3)).astype(config.floatX)

        ATOL = 1e-8 if config.floatX.endswith("64") else 1e-4
        RTOL = 1e-8 if config.floatX.endswith("64") else 1e-4
        solved, product, det = fn(a_val, b_val, c_val)
        np.testing.assert_allclose(
            solved, np.linalg.solve(a_val, b_val), atol=ATOL, rtol=RTOL
        )
        np.testing.assert_allclose(product, a_val @ c_val, atol=ATOL, rtol=RTOL)
        np.testing.assert_allclose(det, np.linalg.det(a_val), atol=ATOL, rtol=RTOL)

    @pytest.mark.skipif(
        config.mode == "FAST_COMPILE", reason="Consumers rely on rewrites"
    )
    def test_assume_a_reaches_an_untagged_solve_of_the_same_matrix(self):
        """A second solve that made no promise of its own still picks the property up.

        The two use different right-hand sides so that they cannot simply merge.
        """
        a, b, c = matrix("a"), matrix("b"), matrix("c")
        fn = function([a, b, c], [solve(a, b, assume_a="pos"), solve(a, c)])

        op_names = self._op_names(fn)
        assert "Solve" not in op_names
        assert op_names.count("Cholesky") == 1
        assert op_names.count("CholeskySolve") == 2

    @pytest.mark.skipif(
        config.mode == "FAST_COMPILE", reason="Consumers rely on rewrites"
    )
    def test_no_assumption_recorded_without_a_promise(self):
        """``assume_a='gen'`` asserts nothing, so nothing is recorded about ``a``."""
        a, b = matrix("a"), matrix("b")
        w, _ = pt.linalg.eig(a)
        fn = function([a, b], [solve(a, b), w])

        op_names = self._op_names(fn)
        assert "Eig" in op_names
        assert "Eigh" not in op_names
        assert "Cholesky" not in op_names

    def test_solve_tringular_indirection(self):
        """The triangular assume_a dispatches to solve_triangular and records itself."""
        a = pt.matrix("a")
        b = pt.vector("b")

        indirect = solve(a, b, assume_a="lower triangular")
        direct = solve_triangular(
            assume(a, lower_triangular=True), b, lower=True, trans=False
        )
        assert equal_computations([indirect], [direct])

        indirect = solve(a, b, assume_a="upper triangular")
        direct = solve_triangular(
            assume(a, upper_triangular=True), b, lower=False, trans=False
        )
        assert equal_computations([indirect], [direct])

        indirect = solve(a, b, assume_a="upper triangular", transposed=True)
        direct = solve_triangular(
            assume(a, upper_triangular=True), b, lower=False, trans=True
        )
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
