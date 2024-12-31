import re

import numpy as np
import pytest

import pytensor
import pytensor.tensor as pt
from pytensor import config
from pytensor.graph import FunctionGraph
from tests.link.numba.test_basic import compare_numba_and_py
from tests.unittest_tools import assert_allclose


numba = pytest.importorskip("numba")

ATOL = 0 if config.floatX.endswith("64") else 1e-6
RTOL = 1e-7 if config.floatX.endswith("64") else 1e-6
rng = np.random.default_rng(42849)


def transpose_func(x, trans):
    if trans == 0:
        return x
    if trans == 1:
        return x.conj().T
    if trans == 2:
        return x.T


@pytest.mark.parametrize(
    "b_func, b_size",
    [(pt.matrix, (5, 1)), (pt.matrix, (5, 5)), (pt.vector, (5,))],
    ids=["b_col_vec", "b_matrix", "b_vec"],
)
@pytest.mark.parametrize("lower", [True, False], ids=["lower=True", "lower=False"])
@pytest.mark.parametrize("trans", [0, 1, 2], ids=["trans=N", "trans=C", "trans=T"])
@pytest.mark.parametrize(
    "unit_diag", [True, False], ids=["unit_diag=True", "unit_diag=False"]
)
@pytest.mark.parametrize("complex", [True, False], ids=["complex", "real"])
@pytest.mark.filterwarnings(
    'ignore:Cannot cache compiled function "numba_funcified_fgraph"'
)
def test_solve_triangular(b_func, b_size, lower, trans, unit_diag, complex):
    if complex:
        # TODO: Complex raises ValueError: To change to a dtype of a different size, the last axis must be contiguous,
        #  why?
        pytest.skip("Complex inputs currently not supported to solve_triangular")

    complex_dtype = "complex64" if config.floatX.endswith("32") else "complex128"
    dtype = complex_dtype if complex else config.floatX

    A = pt.matrix("A", dtype=dtype)
    b = b_func("b", dtype=dtype)

    X = pt.linalg.solve_triangular(
        A, b, lower=lower, trans=trans, unit_diagonal=unit_diag
    )
    f = pytensor.function([A, b], X, mode="NUMBA")

    A_val = np.random.normal(size=(5, 5))
    b = np.random.normal(size=b_size)

    if complex:
        A_val = A_val + np.random.normal(size=(5, 5)) * 1j
        b = b + np.random.normal(size=b_size) * 1j
    A_sym = A_val @ A_val.conj().T

    A_tri = np.linalg.cholesky(A_sym).astype(dtype)
    if unit_diag:
        adj_mat = np.ones((5, 5))
        adj_mat[np.diag_indices(5)] = 1 / np.diagonal(A_tri)
        A_tri = A_tri * adj_mat

    A_tri = A_tri.astype(dtype)
    b = b.astype(dtype)

    if not lower:
        A_tri = A_tri.T

    X_np = f(A_tri, b)
    np.testing.assert_allclose(
        transpose_func(A_tri, trans) @ X_np, b, atol=ATOL, rtol=RTOL
    )


@pytest.mark.parametrize("value", [np.nan, np.inf])
@pytest.mark.filterwarnings(
    'ignore:Cannot cache compiled function "numba_funcified_fgraph"'
)
def test_solve_triangular_raises_on_nan_inf(value):
    A = pt.matrix("A")
    b = pt.matrix("b")

    X = pt.linalg.solve_triangular(A, b, check_finite=True)
    f = pytensor.function([A, b], X, mode="NUMBA")
    A_val = np.random.normal(size=(5, 5))
    A_sym = A_val @ A_val.conj().T

    A_tri = np.linalg.cholesky(A_sym).astype(config.floatX)
    b = np.full((5, 1), value)

    with pytest.raises(
        np.linalg.LinAlgError,
        match=re.escape("Non-numeric values"),
    ):
        f(A_tri, b)


@pytest.mark.parametrize("lower", [True, False], ids=["lower=True", "lower=False"])
@pytest.mark.parametrize("trans", [True, False], ids=["trans=True", "trans=False"])
def test_numba_Cholesky(lower, trans):
    cov = pt.matrix("cov")

    if trans:
        cov_ = cov.T
    else:
        cov_ = cov
    chol = pt.linalg.cholesky(cov_, lower=lower)

    fg = FunctionGraph(outputs=[chol])

    x = np.array([0.1, 0.2, 0.3])
    val = np.eye(3) + x[None, :] * x[:, None]

    compare_numba_and_py(fg, [val])


def test_numba_Cholesky_raises_on_nan_input():
    test_value = rng.random(size=(3, 3)).astype(config.floatX)
    test_value[0, 0] = np.nan

    x = pt.tensor(dtype=config.floatX, shape=(3, 3))
    x = x.T.dot(x)
    g = pt.linalg.cholesky(x, check_finite=True)
    f = pytensor.function([x], g, mode="NUMBA")

    with pytest.raises(np.linalg.LinAlgError, match=r"Non-numeric values"):
        f(test_value)


@pytest.mark.parametrize("on_error", ["nan", "raise"])
def test_numba_Cholesky_raise_on(on_error):
    test_value = rng.random(size=(3, 3)).astype(config.floatX)

    x = pt.tensor(dtype=config.floatX, shape=(3, 3))
    g = pt.linalg.cholesky(x, on_error=on_error)
    f = pytensor.function([x], g, mode="NUMBA")

    if on_error == "raise":
        with pytest.raises(
            np.linalg.LinAlgError, match=r"Input to cholesky is not positive definite"
        ):
            f(test_value)
    else:
        assert np.all(np.isnan(f(test_value)))


def test_block_diag():
    A = pt.matrix("A")
    B = pt.matrix("B")
    C = pt.matrix("C")
    D = pt.matrix("D")
    X = pt.linalg.block_diag(A, B, C, D)

    A_val = np.random.normal(size=(5, 5))
    B_val = np.random.normal(size=(3, 3))
    C_val = np.random.normal(size=(2, 2))
    D_val = np.random.normal(size=(4, 4))
    out_fg = pytensor.graph.FunctionGraph([A, B, C, D], [X])
    compare_numba_and_py(out_fg, [A_val, B_val, C_val, D_val])


def test_lamch():
    from scipy.linalg import get_lapack_funcs

    from pytensor.link.numba.dispatch.slinalg import _xlamch

    @numba.njit()
    def xlamch(kind):
        return _xlamch(kind)

    lamch = get_lapack_funcs("lamch", (np.array([0.0], dtype=config.floatX),))

    np.testing.assert_allclose(xlamch("E"), lamch("E"))
    np.testing.assert_allclose(xlamch("S"), lamch("S"))
    np.testing.assert_allclose(xlamch("P"), lamch("P"))
    np.testing.assert_allclose(xlamch("B"), lamch("B"))
    np.testing.assert_allclose(xlamch("R"), lamch("R"))
    np.testing.assert_allclose(xlamch("M"), lamch("M"))


@pytest.mark.parametrize(
    "ord_numba, ord_scipy", [("F", "fro"), ("1", 1), ("I", np.inf)]
)
def test_xlange(ord_numba, ord_scipy):
    # xlange is called internally only, we don't dispatch pt.linalg.norm to it
    from scipy import linalg

    from pytensor.link.numba.dispatch.slinalg import _xlange

    @numba.njit()
    def xlange(x, ord):
        return _xlange(x, ord)

    x = np.random.normal(size=(5, 5))
    np.testing.assert_allclose(xlange(x, ord_numba), linalg.norm(x, ord_scipy))


@pytest.mark.parametrize("ord_numba, ord_scipy", [("1", 1), ("I", np.inf)])
def test_xgecon(ord_numba, ord_scipy):
    # gecon is called internally only, we don't dispatch pt.linalg.norm to it
    from scipy.linalg import get_lapack_funcs

    from pytensor.link.numba.dispatch.slinalg import _xgecon, _xlange

    @numba.njit()
    def gecon(x, norm):
        anorm = _xlange(x, norm)
        cond, info = _xgecon(x, anorm, norm)
        return cond, info

    x = np.random.normal(size=(5, 5))

    rcond, info = gecon(x, norm=ord_numba)

    # Test against direct call to the underlying LAPACK functions
    # Solution does **not** agree with 1 / np.linalg.cond(x) !
    lange, gecon = get_lapack_funcs(("lange", "gecon"), (x,))
    norm = lange(ord_numba, x)
    rcond2, _ = gecon(x, norm, norm=ord_numba)

    assert info == 0
    np.testing.assert_allclose(rcond, rcond2)


def test_getrf():
    from scipy.linalg import lu_factor

    from pytensor.link.numba.dispatch.slinalg import _getrf

    # TODO: Refactor this test to use compare_numba_and_py after we implement lu_factor in pytensor

    @numba.njit()
    def getrf(x):
        return _getrf(x)

    x = np.random.normal(size=(5, 5))
    LU, IPIV, info = getrf(x)
    lu, ipiv = lu_factor(x)

    assert info == 0
    assert_allclose(LU, lu)
    assert_allclose(IPIV, ipiv)


@pytest.mark.parametrize("trans", [0, 1])
def test_getrs(trans):
    from scipy.linalg import lu_factor
    from scipy.linalg import lu_solve as sp_lu_solve

    from pytensor.link.numba.dispatch.slinalg import _getrf, _getrs

    # TODO: Refactor this test to use compare_numba_and_py after we implement lu_solve in pytensor

    @numba.njit()
    def lu_solve(a, b, trans):
        lu, ipiv, info = _getrf(a)
        x, info = _getrs(lu, b, ipiv, trans)
        return x, info

    a = np.random.normal(size=(5, 5))
    b = np.random.normal(size=(5, 3))

    lu_and_piv = lu_factor(a)

    x_sp = sp_lu_solve(lu_and_piv, b, trans)
    x, info = lu_solve(a, b, trans)
    assert info == 0

    assert_allclose(x, x_sp)


@pytest.mark.parametrize(
    "b_func, b_size",
    [(pt.matrix, (5, 1)), (pt.matrix, (5, 5)), (pt.vector, (5,))],
    ids=["b_col_vec", "b_matrix", "b_vec"],
)
@pytest.mark.parametrize("assume_a", ["gen", "sym"], ids=str)
@pytest.mark.parametrize("transposed", [True, False], ids=["trans", "no_trans"])
@pytest.mark.filterwarnings(
    'ignore:Cannot cache compiled function "numba_funcified_fgraph"'
)
def test_solve(b_func, b_size, assume_a, transposed):
    dtype = config.floatX

    A = pt.matrix("A", dtype=dtype)
    b = b_func("b", dtype=dtype)

    X = pt.linalg.solve(
        A, b, lower=False, assume_a=assume_a, transposed=transposed, b_ndim=len(b_size)
    )
    f = pytensor.function([A, b], X, mode="NUMBA")

    A = np.random.normal(size=(5, 5))
    if assume_a == "sym":
        A = A @ A.conj().T
    b = np.random.normal(size=b_size)
    b = b.astype(dtype)

    X_np = f(A, b)
    np.testing.assert_allclose(
        transpose_func(A, transposed) @ X_np, b, atol=ATOL, rtol=RTOL
    )
