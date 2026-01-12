import functools
import itertools
from functools import partial
from typing import Literal

import numpy as np
import pytest
import scipy
from scipy import linalg as scipy_linalg

from pytensor import In, function, grad
from pytensor import tensor as pt
from pytensor.compile import get_default_mode
from pytensor.configdefaults import config
from pytensor.graph.basic import equal_computations
from pytensor.link.numba import NumbaLinker
from pytensor.tensor import TensorVariable
from pytensor.tensor.slinalg import (
    Cholesky,
    CholeskySolve,
    Solve,
    SolveBase,
    SolveTriangular,
    block_diag,
    cho_solve,
    cholesky,
    eigvalsh,
    expm,
    lu,
    lu_factor,
    lu_solve,
    pivot_to_permutation,
    qr,
    schur,
    solve,
    solve_continuous_lyapunov,
    solve_discrete_are,
    solve_discrete_lyapunov,
    solve_sylvester,
    solve_triangular,
)
from pytensor.tensor.type import dmatrix, matrix, tensor, vector
from tests import unittest_tools as utt


def check_lower_triangular(pd, ch_f):
    ch = ch_f(pd)
    assert ch[0, pd.shape[1] - 1] == 0
    assert ch[pd.shape[0] - 1, 0] != 0
    assert np.allclose(np.dot(ch, ch.T), pd)
    assert not np.allclose(np.dot(ch.T, ch), pd)


def check_upper_triangular(pd, ch_f):
    ch = ch_f(pd)
    assert ch[4, 0] == 0
    assert ch[0, 4] != 0
    assert np.allclose(np.dot(ch.T, ch), pd)
    assert not np.allclose(np.dot(ch, ch.T), pd)


def test_cholesky():
    rng = np.random.default_rng(utt.fetch_seed())
    r = rng.standard_normal((5, 5)).astype(config.floatX)
    pd = np.dot(r, r.T)
    x = matrix()
    chol = cholesky(x)
    # Check the default.
    ch_f = function([x], chol)
    check_lower_triangular(pd, ch_f)
    # Explicit lower-triangular.
    chol = Cholesky(lower=True)(x)
    ch_f = function([x], chol)
    check_lower_triangular(pd, ch_f)
    # Explicit upper-triangular.
    chol = Cholesky(lower=False)(x)
    ch_f = function([x], chol)
    check_upper_triangular(pd, ch_f)


def test_cholesky_performance(benchmark):
    rng = np.random.default_rng(utt.fetch_seed())
    r = rng.standard_normal((10, 10)).astype(config.floatX)
    pd = np.dot(r, r.T)
    x = matrix()
    chol = cholesky(x)
    ch_f = function([x], chol)
    benchmark(ch_f, pd)


def test_cholesky_empty():
    empty = np.empty([0, 0], dtype=config.floatX)
    x = matrix()
    chol = cholesky(x)
    ch_f = function([x], chol)
    ch = ch_f(empty)
    assert ch.size == 0
    assert ch.dtype == config.floatX


def test_cholesky_indef():
    x = matrix()
    mat = np.array([[1, 0.2], [0.2, -2]]).astype(config.floatX)

    with pytest.warns(FutureWarning):
        out = cholesky(x, lower=True, on_error="raise")
    chol_f = function([x], out)
    with pytest.raises(scipy.linalg.LinAlgError):
        chol_f(mat)

    out = cholesky(x, lower=True, on_error="nan")
    chol_f = function([x], out)
    assert np.all(np.isnan(chol_f(mat)))


def test_cholesky_grad():
    rng = np.random.default_rng(utt.fetch_seed())
    r = rng.standard_normal((5, 5)).astype(config.floatX)

    # The dots are inside the graph since Cholesky needs separable matrices

    # Check the default.
    utt.verify_grad(lambda r: cholesky(r.dot(r.T)), [r], 3, rng)
    # Explicit lower-triangular.
    utt.verify_grad(
        lambda r: Cholesky(lower=True)(r.dot(r.T)),
        [r],
        3,
        rng,
        abs_tol=0.05,
        rel_tol=0.05,
    )

    # Explicit upper-triangular.
    utt.verify_grad(
        lambda r: Cholesky(lower=False)(r.dot(r.T)),
        [r],
        3,
        rng,
        abs_tol=0.05,
        rel_tol=0.05,
    )


def test_cholesky_grad_indef():
    x = matrix()
    mat = np.array([[1, 0.2], [0.2, -2]]).astype(config.floatX)

    with pytest.warns(FutureWarning):
        out = cholesky(x, lower=True, on_error="raise")
    chol_f = function([x], grad(out.sum(), [x]), mode="FAST_RUN")

    # original cholesky doesn't show up in the grad (if mode="FAST_RUN"), so it does not raise
    assert np.all(np.isnan(chol_f(mat)))

    out = cholesky(x, lower=True, on_error="nan")
    chol_f = function([x], grad(out.sum(), [x]))
    assert np.all(np.isnan(chol_f(mat)))


def test_cholesky_infer_shape():
    x = matrix()
    f_chol = function([x], [cholesky(x).shape, cholesky(x, lower=False).shape])
    if config.mode != "FAST_COMPILE":
        topo_chol = f_chol.maker.fgraph.toposort()
        f_chol.dprint()
        assert not any(
            isinstance(getattr(node.op, "core_op", node.op), Cholesky)
            for node in topo_chol
        )
    for shp in [2, 3, 5]:
        res1, res2 = f_chol(np.eye(shp).astype(x.dtype))
        assert tuple(res1) == (shp, shp)
        assert tuple(res2) == (shp, shp)


def test_eigvalsh():
    A = dmatrix("a")
    B = dmatrix("b")
    f = function([A, B], eigvalsh(A, B))

    rng = np.random.default_rng(utt.fetch_seed())
    a = rng.standard_normal((5, 5))
    a = a + a.T
    for b in [10 * np.eye(5, 5) + rng.standard_normal((5, 5))]:
        w = f(a, b)
        refw = scipy.linalg.eigvalsh(a, b)
        np.testing.assert_array_almost_equal(w, refw)

    # We need to test None separately, as otherwise DebugMode will
    # complain, as this isn't a valid ndarray.
    b = None
    B = pt.NoneConst
    f = function([A], eigvalsh(A, B))
    w = f(a)
    refw = scipy.linalg.eigvalsh(a, b)
    np.testing.assert_array_almost_equal(w, refw)


def test_eigvalsh_grad():
    rng = np.random.default_rng(utt.fetch_seed())
    a = rng.standard_normal((5, 5))
    a = a + a.T
    b = 10 * np.eye(5, 5) + rng.standard_normal((5, 5))
    utt.verify_grad(
        lambda a, b: eigvalsh(a, b).dot([1, 2, 3, 4, 5]), [a, b], rng=np.random
    )


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

        # To correctly check the gradients, we need to include a transformation from the space of unconstrained matrices
        # (A) to a valid input matrix for the given solver. This is done by the A_func function. If this isn't included,
        # the random perturbations used by verify_grad will result in invalid input matrices, and
        # LAPACK will silently do the wrong thing, making the gradients wrong
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


class TestSolveTriangular(utt.InferShapeTester):
    @staticmethod
    def A_func(x, lower, unit_diagonal):
        x = x @ x.T
        x = pt.linalg.cholesky(x, lower=lower)
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
            [A, b],  # function inputs
            [self.op_class(b_ndim=2)(A, b)],  # function outputs
            # A must be square
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
            [A, b],  # function inputs
            [self.op_class(b_ndim=1)(A, b)],  # function outputs
            # A must be square
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

        # try all dtype combinations
        for A_dtype, b_dtype in itertools.product(dtypes, dtypes):
            if is_numba and (A_dtype == "float16" or b_dtype == "float16"):
                # Numba does not support float16
                continue
            A = matrix(dtype=A_dtype)
            b = matrix(dtype=b_dtype)
            x = op(A, b)
            fn = function([A, b], x)
            x_result = fn(A_val.astype(A_dtype), b_val.astype(b_dtype))

            assert x.dtype == x_result.dtype, (A_dtype, b_dtype)


@pytest.mark.parametrize(
    "permute_l, p_indices",
    [(False, True), (True, False), (False, False)],
    ids=["PL", "p_indices", "P"],
)
@pytest.mark.parametrize("complex", [False, True], ids=["real", "complex"])
@pytest.mark.parametrize("shape", [(3, 5, 5), (5, 5)], ids=["batched", "not_batched"])
def test_lu_decomposition(
    permute_l: bool, p_indices: bool, complex: bool, shape: tuple[int]
):
    dtype = config.floatX if not complex else f"complex{int(config.floatX[-2:]) * 2}"

    A = tensor("A", shape=shape, dtype=dtype)
    pt_out = lu(A, permute_l=permute_l, p_indices=p_indices)

    f = function([A], pt_out)

    rng = np.random.default_rng(utt.fetch_seed())
    x = rng.normal(size=shape).astype(config.floatX)
    if complex:
        x = x + 1j * rng.normal(size=shape).astype(config.floatX)

    out = f(x)
    for numerical_out, symbolic_out in zip(out, pt_out):
        assert numerical_out.dtype == symbolic_out.type.dtype

    if permute_l:
        PL, U = out
    elif p_indices:
        p, L, U = out
        if len(shape) == 2:
            P = np.eye(5)[p]
        else:
            P = np.stack([np.eye(5)[idx] for idx in p])
        PL = np.einsum("...nk,...km->...nm", P, L)
    else:
        P, L, U = out
        PL = np.einsum("...nk,...km->...nm", P, L)

    x_rebuilt = np.einsum("...nk,...km->...nm", PL, U)

    np.testing.assert_allclose(
        x,
        x_rebuilt,
        atol=1e-8 if config.floatX == "float64" else 1e-4,
        rtol=1e-8 if config.floatX == "float64" else 1e-4,
    )
    scipy_out = scipy.linalg.lu(x, permute_l=permute_l, p_indices=p_indices)

    for a, b in zip(out, scipy_out, strict=True):
        np.testing.assert_allclose(a, b)


@pytest.mark.parametrize(
    "grad_case", [0, 1, 2], ids=["dU_only", "dL_only", "dU_and_dL"]
)
@pytest.mark.parametrize(
    "permute_l, p_indices",
    [(True, False), (False, True), (False, False)],
    ids=["PL", "p_indices", "P"],
)
@pytest.mark.parametrize("shape", [(3, 5, 5), (5, 5)], ids=["batched", "not_batched"])
def test_lu_grad(grad_case, permute_l, p_indices, shape):
    rng = np.random.default_rng(utt.fetch_seed())
    A_value = rng.normal(size=shape).astype(config.floatX)

    def f_pt(A):
        # lu returns either (P_or_index, L, U) or (PL, U), depending on settings
        out = lu(A, permute_l=permute_l, p_indices=p_indices, check_finite=False)

        match grad_case:
            case 0:
                return out[-1].sum()
            case 1:
                return out[-2].sum()
            case 2:
                return out[-1].sum() + out[-2].sum()

    utt.verify_grad(f_pt, [A_value], rng=rng)


@pytest.mark.parametrize("inverse", [True, False], ids=["inverse", "no_inverse"])
def test_pivot_to_permutation(inverse):
    rng = np.random.default_rng(utt.fetch_seed())
    A_val = rng.normal(size=(5, 5))
    _, pivots = scipy.linalg.lu_factor(A_val)
    perm_idx, *_ = scipy.linalg.lu(A_val, p_indices=True)

    if not inverse:
        perm_idx_pt = pivot_to_permutation(pivots, inverse=False).eval()
        np.testing.assert_array_equal(perm_idx_pt, perm_idx)
    else:
        p_inv_pt = pivot_to_permutation(pivots, inverse=True).eval()
        np.testing.assert_array_equal(p_inv_pt, np.argsort(perm_idx))


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


def test_lu_factor():
    rng = np.random.default_rng(utt.fetch_seed())
    A = matrix()
    A_val = rng.normal(size=(5, 5)).astype(config.floatX)

    f = function([A], lu_factor(A))

    LU, pt_p_idx = f(A_val)
    sp_LU, sp_p_idx = scipy.linalg.lu_factor(A_val)

    np.testing.assert_allclose(LU, sp_LU)
    np.testing.assert_allclose(pt_p_idx, sp_p_idx)

    utt.verify_grad(
        lambda A: lu_factor(A)[0].sum(),
        [A_val],
        rng=rng,
    )


def test_lu_factor_empty():
    A = matrix()
    f = function([A], lu_factor(A))

    A_empty = np.empty([0, 0], dtype=config.floatX)
    LU, pt_p_idx = f(A_empty)

    assert LU.size == 0
    assert LU.dtype == config.floatX
    assert pt_p_idx.size == 0


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


def test_expm():
    rng = np.random.default_rng(utt.fetch_seed())
    A = rng.standard_normal((5, 5)).astype(config.floatX)

    ref = scipy.linalg.expm(A)

    x = matrix()
    m = expm(x)
    expm_f = function([x], m)

    val = expm_f(A)
    np.testing.assert_array_almost_equal(val, ref)


@pytest.mark.parametrize(
    "mode", ["symmetric", "nonsymmetric_real_eig", "nonsymmetric_complex_eig"][-1:]
)
def test_expm_grad(mode):
    rng = np.random.default_rng([898, sum(map(ord, mode))])

    match mode:
        case "symmetric":
            A = rng.standard_normal((5, 5))
            A = A + A.T
        case "nonsymmetric_real_eig":
            A = rng.standard_normal((5, 5))
            w = rng.standard_normal(5) ** 2
            A = (np.diag(w**0.5)).dot(A + A.T).dot(np.diag(w ** (-0.5)))
        case "nonsymmetric_complex_eig":
            A = rng.standard_normal((5, 5))
        case _:
            raise ValueError(f"Invalid mode: {mode}")

    utt.verify_grad(expm, [A], rng=rng, abs_tol=1e-5, rel_tol=1e-5)


@pytest.mark.parametrize(
    "shape, use_complex",
    [((5, 5), False), ((5, 5), True), ((5, 5, 5), False)],
    ids=["float", "complex", "batch_float"],
)
def test_solve_continuous_sylvester(shape: tuple[int], use_complex: bool):
    # batch-complex case got an error from BatchedDot not implemented for complex numbers
    rng = np.random.default_rng()

    dtype = config.floatX
    if use_complex:
        dtype = "complex128" if dtype == "float64" else "complex64"

    A1, A2 = rng.normal(size=(2, *shape))
    B1, B2 = rng.normal(size=(2, *shape))
    Q1, Q2 = rng.normal(size=(2, *shape))

    if use_complex:
        A_val = A1 + 1j * A2
        B_val = B1 + 1j * B2
        Q_val = Q1 + 1j * Q2
    else:
        A_val = A1
        B_val = B1
        Q_val = Q1

    A = pt.tensor("A", shape=shape, dtype=dtype)
    B = pt.tensor("B", shape=shape, dtype=dtype)
    Q = pt.tensor("Q", shape=shape, dtype=dtype)

    X = solve_sylvester(A, B, Q)
    Q_recovered = A @ X + X @ B

    fn = function([A, B, Q], [X, Q_recovered])
    X_val, Q_recovered_val = fn(A_val, B_val, Q_val)

    vec_sylvester = np.vectorize(
        scipy_linalg.solve_sylvester, signature="(m,m),(m,m),(m,m)->(m,m)"
    )
    np.testing.assert_allclose(Q_recovered_val, Q_val, atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(
        X_val, vec_sylvester(A_val, B_val, Q_val), atol=1e-8, rtol=1e-8
    )


@pytest.mark.parametrize("shape", [(5, 5), (5, 5, 5)], ids=["matrix", "batched"])
@pytest.mark.parametrize("use_complex", [False, True], ids=["float", "complex"])
def test_solve_continuous_sylvester_grad(shape: tuple[int], use_complex):
    if config.floatX == "float32":
        pytest.skip(reason="Not enough precision in float32 to get a good gradient")
    if use_complex:
        pytest.skip(reason="Complex numbers are not supported in the gradient test")

    rng = np.random.default_rng(utt.fetch_seed())
    A = rng.normal(size=shape).astype(config.floatX)
    B = rng.normal(size=shape).astype(config.floatX)
    Q = rng.normal(size=shape).astype(config.floatX)

    utt.verify_grad(solve_sylvester, pt=[A, B, Q], rng=rng)


def recover_Q(A, X, continuous=True):
    if continuous:
        return A @ X + X @ A.conj().T
    else:
        return X - A @ X @ A.conj().T


vec_recover_Q = np.vectorize(recover_Q, signature="(m,m),(m,m),()->(m,m)")


@pytest.mark.parametrize("use_complex", [False, True], ids=["float", "complex"])
@pytest.mark.parametrize("shape", [(5, 5), (5, 5, 5)], ids=["matrix", "batch"])
@pytest.mark.parametrize("method", ["direct", "bilinear"])
def test_solve_discrete_lyapunov(
    use_complex, shape: tuple[int], method: Literal["direct", "bilinear"]
):
    rng = np.random.default_rng(utt.fetch_seed())
    dtype = config.floatX
    if use_complex:
        precision = int(dtype[-2:])  # 64 or 32
        dtype = f"complex{int(2 * precision)}"

    A1, A2 = rng.normal(size=(2, *shape))
    Q1, Q2 = rng.normal(size=(2, *shape))

    if use_complex:
        A = A1 + 1j * A2
        Q = Q1 + 1j * Q2
    else:
        A = A1
        Q = Q1

    A, Q = A.astype(dtype), Q.astype(dtype)

    a = pt.tensor(name="a", shape=shape, dtype=dtype)
    q = pt.tensor(name="q", shape=shape, dtype=dtype)

    x = solve_discrete_lyapunov(a, q, method=method)
    f = function([a, q], x)

    X = f(A, Q)
    Q_recovered = vec_recover_Q(A, X, continuous=False)

    atol = rtol = 1e-4 if config.floatX == "float32" else 1e-8
    np.testing.assert_allclose(Q_recovered, Q, atol=atol, rtol=rtol)


@pytest.mark.parametrize("use_complex", [False, True], ids=["float", "complex"])
@pytest.mark.parametrize("shape", [(5, 5), (5, 5, 5)], ids=["matrix", "batch"])
@pytest.mark.parametrize("method", ["direct", "bilinear"])
def test_solve_discrete_lyapunov_gradient(
    use_complex, shape: tuple[int], method: Literal["direct", "bilinear"]
):
    if config.floatX == "float32":
        pytest.skip(reason="Not enough precision in float32 to get a good gradient")
    if use_complex:
        pytest.skip(reason="Complex numbers are not supported in the gradient test")

    rng = np.random.default_rng(utt.fetch_seed())
    A = rng.normal(size=shape).astype(config.floatX)
    Q = rng.normal(size=shape).astype(config.floatX)

    utt.verify_grad(
        functools.partial(solve_discrete_lyapunov, method=method),
        pt=[A, Q],
        rng=rng,
    )


def test_solve_continuous_lyapunov():
    # solve_continuous_lyapunov just calls solve_sylvester, so extensive tests are not needed.
    A = pt.tensor("A", shape=(3, 5, 5))
    Q = pt.tensor("Q", shape=(3, 5, 5))

    X = solve_continuous_lyapunov(A, Q)
    Q_recovered = A @ X + X @ A.conj().mT

    fn = function([A, Q], [X, Q_recovered])

    rng = np.random.default_rng(utt.fetch_seed())
    A_val = rng.normal(size=(3, 5, 5)).astype(config.floatX)
    Q_val = rng.normal(size=(3, 5, 5)).astype(config.floatX)
    _, Q_recovered_val = fn(A_val, Q_val)

    atol = rtol = 1e-2 if config.floatX == "float32" else 1e-8
    np.testing.assert_allclose(Q_recovered_val, Q_val, atol=atol, rtol=rtol)
    utt.verify_grad(solve_continuous_lyapunov, pt=[A_val, Q_val], rng=rng)


@pytest.mark.parametrize("add_batch_dim", [False, True])
def test_solve_discrete_are_forward(add_batch_dim):
    # TEST CASE 4 : darex #1 -- taken from Scipy tests
    a, b, q, r = (
        np.array([[4, 3], [-4.5, -3.5]]),
        np.array([[1], [-1]]),
        np.array([[9, 6], [6, 4]]),
        np.array([[1]]),
    )
    if add_batch_dim:
        a, b, q, r = (np.stack([x] * 5) for x in [a, b, q, r])

    a, b, q, r = (pt.as_tensor_variable(x).astype(config.floatX) for x in [a, b, q, r])

    x = solve_discrete_are(a, b, q, r)

    def eval_fun(a, b, q, r, x):
        term_1 = a.T @ x @ a
        term_2 = a.T @ x @ b
        term_3 = pt.linalg.solve(r + b.T @ x @ b, b.T) @ x @ a

        return term_1 - x - term_2 @ term_3 + q

    res = pt.vectorize(eval_fun, "(m,m),(m,n),(m,m),(n,n),(m,m)->(m,m)")(a, b, q, r, x)
    res_np = res.eval()

    atol = 1e-4 if config.floatX == "float32" else 1e-12
    np.testing.assert_allclose(res_np, np.zeros_like(res_np), atol=atol)


@pytest.mark.parametrize("add_batch_dim", [False, True])
def test_solve_discrete_are_grad(add_batch_dim):
    a, b, q, r = (
        np.array([[4, 3], [-4.5, -3.5]]),
        np.array([[1], [-1]]),
        np.array([[9, 6], [6, 4]]),
        np.array([[1]]),
    )
    if add_batch_dim:
        a, b, q, r = (np.stack([x] * 5) for x in [a, b, q, r])

    a, b, q, r = (x.astype(config.floatX) for x in [a, b, q, r])
    rng = np.random.default_rng(utt.fetch_seed())

    # TODO: Is there a "theoretically motivated" value to use here? I pulled 1e-4 out of a hat
    atol = 1e-4 if config.floatX == "float32" else 1e-12

    utt.verify_grad(
        functools.partial(solve_discrete_are, enforce_Q_symmetric=True),
        pt=[a, b, q, r],
        rng=rng,
        abs_tol=atol,
    )


def test_block_diagonal():
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    B = np.array([[5.0, 6.0], [7.0, 8.0]])
    result = block_diag(A, B)
    assert result.type.shape == (4, 4)
    assert result.owner.op.core_op._props_dict() == {"n_inputs": 2}

    np.testing.assert_allclose(result.eval(), scipy.linalg.block_diag(A, B))


def test_block_diagonal_static_shape():
    A = pt.dmatrix("A", shape=(5, 5))
    B = pt.dmatrix("B", shape=(3, 10))
    result = block_diag(A, B)
    assert result.type.shape == (8, 15)

    A = pt.dmatrix("A", shape=(5, 5))
    B = pt.dmatrix("B", shape=(3, None))
    result = block_diag(A, B)
    assert result.type.shape == (8, None)

    A = pt.dmatrix("A", shape=(None, 5))
    result = block_diag(A, B)
    assert result.type.shape == (None, None)


def test_block_diagonal_grad():
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    B = np.array([[5.0, 6.0], [7.0, 8.0]])

    utt.verify_grad(block_diag, pt=[A, B], rng=np.random.default_rng())


def test_block_diagonal_blockwise():
    batch_size = 5
    A = np.random.normal(size=(batch_size, 2, 2)).astype(config.floatX)
    B = np.random.normal(size=(batch_size, 4, 4)).astype(config.floatX)
    result = block_diag(A, B).eval()
    assert result.shape == (batch_size, 6, 6)
    for i in range(batch_size):
        np.testing.assert_allclose(
            result[i],
            scipy.linalg.block_diag(A[i], B[i]),
            atol=1e-4 if config.floatX == "float32" else 1e-8,
            rtol=1e-4 if config.floatX == "float32" else 1e-8,
        )

    # Test broadcasting
    A = np.random.normal(size=(10, batch_size, 2, 2)).astype(config.floatX)
    B = np.random.normal(size=(1, batch_size, 4, 4)).astype(config.floatX)
    result = block_diag(A, B).eval()
    assert result.shape == (10, batch_size, 6, 6)


@pytest.mark.parametrize(
    "mode, names",
    [
        ("economic", ["Q", "R"]),
        ("full", ["Q", "R"]),
        ("r", ["R"]),
        ("raw", ["H", "tau", "R"]),
    ],
)
@pytest.mark.parametrize("pivoting", [True, False])
def test_qr_modes(mode, names, pivoting):
    rng = np.random.default_rng(utt.fetch_seed())
    A_val = rng.random((4, 4)).astype(config.floatX)

    if pivoting:
        names = [*names, "pivots"]

    A = tensor("A", dtype=config.floatX, shape=(None, None))

    f = function([A], qr(A, mode=mode, pivoting=pivoting))

    outputs_pt = f(A_val)
    outputs_sp = scipy_linalg.qr(A_val, mode=mode, pivoting=pivoting)

    if mode == "raw":
        # The first output of scipy's qr is a tuple when mode is raw; flatten it for easier iteration
        outputs_sp = (*outputs_sp[0], *outputs_sp[1:])
    elif mode == "r" and not pivoting:
        # Here there's only one output from the pytensor function; wrap it in a list for iteration
        outputs_pt = [outputs_pt]

    for out_pt, out_sp, name in zip(outputs_pt, outputs_sp, names):
        np.testing.assert_allclose(out_pt, out_sp, err_msg=f"{name} disagrees")


@pytest.mark.parametrize(
    "shape, gradient_test_case, mode",
    (
        [(s, c, "economic") for s in [(3, 3), (6, 3), (3, 6)] for c in [0, 1, 2]]
        + [(s, c, "full") for s in [(3, 3), (6, 3), (3, 6)] for c in [0, 1, 2]]
        + [(s, 0, "r") for s in [(3, 3), (6, 3), (3, 6)]]
        + [((3, 3), 0, "raw")]
    ),
    ids=(
        [
            f"shape={s}, gradient_test_case={c}, mode=economic"
            for s in [(3, 3), (6, 3), (3, 6)]
            for c in ["Q", "R", "both"]
        ]
        + [
            f"shape={s}, gradient_test_case={c}, mode=full"
            for s in [(3, 3), (6, 3), (3, 6)]
            for c in ["Q", "R", "both"]
        ]
        + [f"shape={s}, gradient_test_case=R, mode=r" for s in [(3, 3), (6, 3), (3, 6)]]
        + ["shape=(3, 3), gradient_test_case=Q, mode=raw"]
    ),
)
@pytest.mark.parametrize("is_complex", [True, False], ids=["complex", "real"])
def test_qr_grad(shape, gradient_test_case, mode, is_complex):
    rng = np.random.default_rng(utt.fetch_seed())

    def _test_fn(x, case=2, mode="reduced"):
        if case == 0:
            return qr(x, mode=mode)[0].sum()
        elif case == 1:
            return qr(x, mode=mode)[1].sum()
        elif case == 2:
            Q, R = qr(x, mode=mode)
            return Q.sum() + R.sum()

    if is_complex:
        pytest.xfail("Complex inputs currently not supported by verify_grad")

    m, n = shape
    a = rng.standard_normal(shape).astype(config.floatX)
    if is_complex:
        a += 1j * rng.standard_normal(shape).astype(config.floatX)

    if mode == "raw":
        with pytest.raises(NotImplementedError):
            utt.verify_grad(
                partial(_test_fn, case=gradient_test_case, mode=mode),
                [a],
                rng=np.random,
            )

    elif mode == "full" and m > n:
        with pytest.raises(AssertionError):
            utt.verify_grad(
                partial(_test_fn, case=gradient_test_case, mode=mode),
                [a],
                rng=np.random,
            )

    else:
        utt.verify_grad(
            partial(_test_fn, case=gradient_test_case, mode=mode), [a], rng=np.random
        )


class TestSchur:
    @pytest.mark.parametrize(
        "shape, output",
        [((5, 5), "real"), ((5, 5), "complex"), ((2, 4, 4), "real")],
        ids=["not_batched_real", "not_batched_complex", "batched_real"],
    )
    @pytest.mark.parametrize("complex", [False, True], ids=["real", "complex"])
    def test_schur_decomposition(self, shape, output, complex):
        dtype = (
            config.floatX if not complex else f"complex{int(config.floatX[-2:]) * 2}"
        )

        A = tensor("A", shape=shape, dtype=dtype)
        T, Z = schur(A, output=output)

        f = function([A], [T, Z])

        rng = np.random.default_rng(utt.fetch_seed())
        x = rng.normal(size=shape).astype(config.floatX)
        if complex:
            x = x + 1j * rng.normal(size=shape).astype(config.floatX)

        T_out, Z_out = f(x)

        # Verify reconstruction
        x_rebuilt = np.einsum("...ij,...jk,...lk->...il", Z_out, T_out, Z_out.conj())

        np.testing.assert_allclose(
            x,
            x_rebuilt,
            atol=1e-6 if config.floatX == "float64" else 1e-3,
            rtol=1e-6 if config.floatX == "float64" else 1e-3,
        )

        vec_schur = np.vectorize(
            lambda a: scipy_linalg.schur(a, output=output),
            signature="(m,m)->(m,m),(m,m)",
        )

        scipy_T, scipy_Z = vec_schur(x)
        np.testing.assert_allclose(T_out, scipy_T, atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(Z_out, scipy_Z, atol=1e-6, rtol=1e-6)

        if len(shape) == 2 and (output == "complex") == complex:
            x_f = np.asfortranarray(x.copy())
            f_mut = function(
                [In(A, mutable=True)],
                [T, Z],
                mode=get_default_mode().including("inplace"),
            )
            f_mut(x_f)
            np.testing.assert_allclose(x_f, scipy_T, atol=1e-6, rtol=1e-6)

    @pytest.mark.parametrize("sort", ["lhp", "rhp", "iuc", "ouc"])
    def test_schur_sort(self, sort):
        rng = np.random.default_rng(utt.fetch_seed())
        x = rng.normal(size=(3, 3)).astype(config.floatX)

        A = matrix("A", dtype=config.floatX)
        T, Z = schur(A, sort=sort)

        f = function([A], [T, Z])
        T_out, Z_out = f(x)

        x_rebuilt = Z_out @ T_out @ Z_out.T

        np.testing.assert_allclose(
            x,
            x_rebuilt,
            atol=1e-6 if config.floatX == "float64" else 1e-3,
            rtol=1e-6 if config.floatX == "float64" else 1e-3,
        )

        scipy_T, scipy_Z, _ = scipy_linalg.schur(x, output="real", sort=sort)
        np.testing.assert_allclose(T_out, scipy_T, atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(Z_out, scipy_Z, atol=1e-6, rtol=1e-6)

    def test_schur_empty(self):
        empty = np.empty([0, 0], dtype=config.floatX)
        A = matrix()
        T, Z = schur(A)
        f = function([A], [T, Z])
        T_out, Z_out = f(empty)
        assert T_out.size == 0
        assert Z_out.size == 0
        assert T_out.dtype == config.floatX
        assert Z_out.dtype == config.floatX
