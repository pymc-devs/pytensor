import functools
import itertools
from typing import Literal

import numpy as np
import pytest
import scipy

import pytensor
from pytensor import function, grad
from pytensor import tensor as pt
from pytensor.configdefaults import config
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
    solve,
    solve_continuous_lyapunov,
    solve_discrete_are,
    solve_discrete_lyapunov,
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
    chol = Cholesky(lower=False, on_error="nan")(x)
    ch_f = function([x], chol)
    check_upper_triangular(pd, ch_f)


def test_cholesky_indef():
    x = matrix()
    mat = np.array([[1, 0.2], [0.2, -2]]).astype(config.floatX)
    cholesky = Cholesky(lower=True, on_error="raise")
    chol_f = function([x], cholesky(x))
    with pytest.raises(scipy.linalg.LinAlgError):
        chol_f(mat)
    cholesky = Cholesky(lower=True, on_error="nan")
    chol_f = function([x], cholesky(x))
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
    cholesky = Cholesky(lower=True, on_error="raise")
    chol_f = function([x], grad(cholesky(x).sum(), [x]))
    with pytest.raises(scipy.linalg.LinAlgError):
        chol_f(mat)
    cholesky = Cholesky(lower=True, on_error="nan")
    chol_f = function([x], grad(cholesky(x).sum(), [x]))
    assert np.all(np.isnan(chol_f(mat)))


@pytest.mark.slow
def test_cholesky_shape():
    rng = np.random.default_rng(utt.fetch_seed())
    x = matrix()
    for l in (cholesky(x), Cholesky(lower=True)(x), Cholesky(lower=False)(x)):
        f_chol = pytensor.function([x], l.shape)
        topo_chol = f_chol.maker.fgraph.toposort()
        if config.mode != "FAST_COMPILE":
            assert sum(node.op.__class__ == Cholesky for node in topo_chol) == 0
        for shp in [2, 3, 5]:
            m = np.cov(rng.standard_normal((shp, shp + 10))).astype(config.floatX)
            np.testing.assert_equal(f_chol(m), (shp, shp))


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


class TestSolveBase(utt.InferShapeTester):
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
            SolveBase(b_ndim=2)(A, b)

    def test__repr__(self):
        np.random.default_rng(utt.fetch_seed())
        A = matrix()
        b = matrix()
        y = SolveBase(b_ndim=2)(A, b)
        assert (
            y.__repr__()
            == "SolveBase{lower=False, check_finite=True, b_ndim=2, overwrite_a=False, overwrite_b=False}.0"
        )


class TestSolve(utt.InferShapeTester):
    def test__init__(self):
        with pytest.raises(ValueError) as excinfo:
            Solve(assume_a="test", b_ndim=2)
        assert "is not a recognized matrix structure" in str(excinfo.value)

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

    def test_correctness(self):
        rng = np.random.default_rng(utt.fetch_seed())
        A = matrix()
        b = matrix()
        y = solve(A, b)
        gen_solve_func = pytensor.function([A, b], y)

        b_val = np.asarray(rng.random((5, 1)), dtype=config.floatX)

        A_val = np.asarray(rng.random((5, 5)), dtype=config.floatX)
        A_val = np.dot(A_val.transpose(), A_val)

        assert np.allclose(
            scipy.linalg.solve(A_val, b_val), gen_solve_func(A_val, b_val)
        )

        A_undef = np.array(
            [
                [1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 1, 1],
                [0, 0, 0, 1, 0],
            ],
            dtype=config.floatX,
        )
        assert np.allclose(
            scipy.linalg.solve(A_undef, b_val), gen_solve_func(A_undef, b_val)
        )

    @pytest.mark.parametrize(
        "m, n, assume_a, lower",
        [
            (5, None, "gen", False),
            (5, None, "gen", True),
            (4, 2, "gen", False),
            (4, 2, "gen", True),
        ],
    )
    def test_solve_grad(self, m, n, assume_a, lower):
        rng = np.random.default_rng(utt.fetch_seed())

        # Ensure diagonal elements of `A` are relatively large to avoid
        # numerical precision issues
        A_val = (rng.normal(size=(m, m)) * 0.5 + np.eye(m)).astype(config.floatX)

        if n is None:
            b_val = rng.normal(size=m).astype(config.floatX)
        else:
            b_val = rng.normal(size=(m, n)).astype(config.floatX)

        eps = None
        if config.floatX == "float64":
            eps = 2e-8

        solve_op = Solve(assume_a=assume_a, lower=lower, b_ndim=1 if n is None else 2)
        utt.verify_grad(solve_op, [A_val, b_val], 3, rng, eps=eps)


class TestSolveTriangular(utt.InferShapeTester):
    @pytest.mark.parametrize("b_shape", [(5, 1), (5,)])
    def test_infer_shape(self, b_shape):
        rng = np.random.default_rng(utt.fetch_seed())
        A = matrix()
        b_val = np.asarray(rng.random(b_shape), dtype=config.floatX)
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

    @pytest.mark.parametrize("lower", [True, False])
    def test_correctness(self, lower):
        rng = np.random.default_rng(utt.fetch_seed())

        b_val = np.asarray(rng.random((5, 1)), dtype=config.floatX)

        A_val = np.asarray(rng.random((5, 5)), dtype=config.floatX)
        A_val = np.dot(A_val.transpose(), A_val)

        C_val = scipy.linalg.cholesky(A_val, lower=lower)

        A = matrix()
        b = matrix()

        cholesky = Cholesky(lower=lower)
        C = cholesky(A)
        y_lower = solve_triangular(C, b, lower=lower)
        lower_solve_func = pytensor.function([C, b], y_lower)

        assert np.allclose(
            scipy.linalg.solve_triangular(C_val, b_val, lower=lower),
            lower_solve_func(C_val, b_val),
        )

    @pytest.mark.parametrize(
        "m, n, lower",
        [
            (5, None, False),
            (5, None, True),
            (4, 2, False),
            (4, 2, True),
        ],
    )
    def test_solve_grad(self, m, n, lower):
        rng = np.random.default_rng(utt.fetch_seed())

        # Ensure diagonal elements of `A` are relatively large to avoid
        # numerical precision issues
        A_val = (rng.normal(size=(m, m)) * 0.5 + np.eye(m)).astype(config.floatX)

        if n is None:
            b_val = rng.normal(size=m).astype(config.floatX)
        else:
            b_val = rng.normal(size=(m, n)).astype(config.floatX)

        eps = None
        if config.floatX == "float64":
            eps = 2e-8

        solve_op = SolveTriangular(lower=lower, b_ndim=1 if n is None else 2)
        utt.verify_grad(solve_op, [A_val, b_val], 3, rng, eps=eps)


class TestCholeskySolve(utt.InferShapeTester):
    def setup_method(self):
        self.op_class = CholeskySolve
        super().setup_method()

    def test_repr(self):
        assert (
            repr(CholeskySolve(lower=True, b_ndim=1))
            == "CholeskySolve(lower=True,check_finite=True,b_ndim=1,overwrite_b=False)"
        )

    def test_infer_shape(self):
        rng = np.random.default_rng(utt.fetch_seed())
        A = matrix()
        b = matrix()
        self._compile_and_check(
            [A, b],  # pytensor.function inputs
            [self.op_class(b_ndim=2)(A, b)],  # pytensor.function outputs
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
            [A, b],  # pytensor.function inputs
            [self.op_class(b_ndim=1)(A, b)],  # pytensor.function outputs
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
        cho_solve_lower_func = pytensor.function([A, b], y)

        y = self.op_class(lower=False, b_ndim=2)(A, b)
        cho_solve_upper_func = pytensor.function([A, b], y)

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
            A = matrix(dtype=A_dtype)
            b = matrix(dtype=b_dtype)
            x = op(A, b)
            fn = function([A, b], x)
            x_result = fn(A_val.astype(A_dtype), b_val.astype(b_dtype))

            assert x.dtype == x_result.dtype


def test_cho_solve():
    rng = np.random.default_rng(utt.fetch_seed())
    A = matrix()
    b = matrix()
    y = cho_solve((A, True), b)
    cho_solve_lower_func = pytensor.function([A, b], y)

    b_val = np.asarray(rng.random((5, 1)), dtype=config.floatX)

    A_val = np.tril(np.asarray(rng.random((5, 5)), dtype=config.floatX))

    assert np.allclose(
        scipy.linalg.cho_solve((A_val, True), b_val),
        cho_solve_lower_func(A_val, b_val),
    )


def test_expm():
    rng = np.random.default_rng(utt.fetch_seed())
    A = rng.standard_normal((5, 5)).astype(config.floatX)

    ref = scipy.linalg.expm(A)

    x = matrix()
    m = expm(x)
    expm_f = function([x], m)

    val = expm_f(A)
    np.testing.assert_array_almost_equal(val, ref)


def test_expm_grad_1():
    # with symmetric matrix (real eigenvectors)
    rng = np.random.default_rng(utt.fetch_seed())
    # Always test in float64 for better numerical stability.
    A = rng.standard_normal((5, 5))
    A = A + A.T

    utt.verify_grad(expm, [A], rng=rng)


def test_expm_grad_2():
    # with non-symmetric matrix with real eigenspecta
    rng = np.random.default_rng(utt.fetch_seed())
    # Always test in float64 for better numerical stability.
    A = rng.standard_normal((5, 5))
    w = rng.standard_normal(5) ** 2
    A = (np.diag(w**0.5)).dot(A + A.T).dot(np.diag(w ** (-0.5)))
    assert not np.allclose(A, A.T)

    utt.verify_grad(expm, [A], rng=rng)


def test_expm_grad_3():
    # with non-symmetric matrix (complex eigenvectors)
    rng = np.random.default_rng(utt.fetch_seed())
    # Always test in float64 for better numerical stability.
    A = rng.standard_normal((5, 5))

    utt.verify_grad(expm, [A], rng=rng)


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


@pytest.mark.parametrize("shape", [(5, 5), (5, 5, 5)], ids=["matrix", "batched"])
@pytest.mark.parametrize("use_complex", [False, True], ids=["float", "complex"])
def test_solve_continuous_lyapunov(shape: tuple[int], use_complex: bool):
    dtype = config.floatX
    if use_complex and dtype == "float32":
        pytest.skip(
            "Not enough precision in complex64 to do schur decomposition "
            "(ill-conditioned matrix errors arise)"
        )
    rng = np.random.default_rng(utt.fetch_seed())

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
    x = solve_continuous_lyapunov(a, q)

    f = function([a, q], x)

    X = f(A, Q)

    Q_recovered = vec_recover_Q(A, X, continuous=True)

    atol = rtol = 1e-2 if config.floatX == "float32" else 1e-8
    np.testing.assert_allclose(Q_recovered.squeeze(), Q, atol=atol, rtol=rtol)


@pytest.mark.parametrize("shape", [(5, 5), (5, 5, 5)], ids=["matrix", "batched"])
@pytest.mark.parametrize("use_complex", [False, True], ids=["float", "complex"])
def test_solve_continuous_lyapunov_grad(shape: tuple[int], use_complex):
    if config.floatX == "float32":
        pytest.skip(reason="Not enough precision in float32 to get a good gradient")
    if use_complex:
        pytest.skip(reason="Complex numbers are not supported in the gradient test")

    rng = np.random.default_rng(utt.fetch_seed())
    A = rng.normal(size=shape).astype(config.floatX)
    Q = rng.normal(size=shape).astype(config.floatX)

    utt.verify_grad(solve_continuous_lyapunov, pt=[A, Q], rng=rng)


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
    assert result.owner.op.core_op._props_dict() == {"n_inputs": 2}

    np.testing.assert_allclose(result.eval(), scipy.linalg.block_diag(A, B))


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
