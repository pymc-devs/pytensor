import re
from functools import partial
from typing import Literal

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import linalg as scipy_linalg

import pytensor
import pytensor.tensor as pt
from pytensor import config
from tests import unittest_tools as utt
from tests.link.numba.test_basic import compare_numba_and_py


numba = pytest.importorskip("numba")

floatX = config.floatX

rng = np.random.default_rng(42849)


def transpose_func(x, trans):
    if trans == 0:
        return x
    if trans == 1:
        return x.conj().T
    if trans == 2:
        return x.T


@pytest.mark.parametrize(
    "b_shape",
    [(5, 1), (5, 5), (5,)],
    ids=["b_col_vec", "b_matrix", "b_vec"],
)
@pytest.mark.parametrize("lower", [True, False], ids=["lower=True", "lower=False"])
@pytest.mark.parametrize("trans", [0, 1, 2], ids=["trans=N", "trans=C", "trans=T"])
@pytest.mark.parametrize(
    "unit_diag", [True, False], ids=["unit_diag=True", "unit_diag=False"]
)
@pytest.mark.parametrize("is_complex", [True, False], ids=["complex", "real"])
@pytest.mark.filterwarnings(
    'ignore:Cannot cache compiled function "numba_funcified_fgraph"'
)
def test_solve_triangular(b_shape: tuple[int], lower, trans, unit_diag, is_complex):
    if is_complex:
        # TODO: Complex raises ValueError: To change to a dtype of a different size, the last axis must be contiguous,
        #  why?
        pytest.skip("Complex inputs currently not supported to solve_triangular")

    complex_dtype = "complex64" if floatX.endswith("32") else "complex128"
    dtype = complex_dtype if is_complex else floatX

    A = pt.matrix("A", dtype=dtype)
    b = pt.tensor("b", shape=b_shape, dtype=dtype)

    def A_func(x):
        x = x @ x.conj().T
        x_tri = scipy_linalg.cholesky(x, lower=lower).astype(dtype)

        if unit_diag:
            x_tri[np.diag_indices_from(x_tri)] = 1.0

        return x_tri.astype(dtype)

    solve_op = partial(
        pt.linalg.solve_triangular, lower=lower, trans=trans, unit_diagonal=unit_diag
    )

    X = solve_op(A, b)
    f = pytensor.function([A, b], X, mode="NUMBA")

    A_val = np.random.normal(size=(5, 5))
    b_val = np.random.normal(size=b_shape)

    if is_complex:
        A_val = A_val + np.random.normal(size=(5, 5)) * 1j
        b_val = b_val + np.random.normal(size=b_shape) * 1j

    X_np = f(A_func(A_val), b_val)

    test_input = transpose_func(A_func(A_val), trans)

    ATOL = 1e-8 if floatX.endswith("64") else 1e-4
    RTOL = 1e-8 if floatX.endswith("64") else 1e-4

    np.testing.assert_allclose(test_input @ X_np, b_val, atol=ATOL, rtol=RTOL)

    compiled_fgraph = f.maker.fgraph
    compare_numba_and_py(
        compiled_fgraph.inputs,
        compiled_fgraph.outputs,
        [A_func(A_val), b_val],
    )


@pytest.mark.parametrize(
    "lower, unit_diag, trans",
    [(True, True, True), (False, False, False)],
    ids=["lower_unit_trans", "defaults"],
)
def test_solve_triangular_grad(lower, unit_diag, trans):
    A_val = np.random.normal(size=(5, 5)).astype(floatX)
    b_val = np.random.normal(size=(5, 5)).astype(floatX)

    # utt.verify_grad uses small perturbations to the input matrix to calculate the finite difference gradient. When
    # a non-triangular matrix is passed to scipy.linalg.solve_triangular, no error is raise, but the result will be
    # wrong, resulting in wrong gradients. As a result, it is necessary to add a mapping from the space of all matrices
    # to the space of triangular matrices, and test the gradient of that entire graph.
    def A_func_pt(x):
        x = x @ x.conj().T
        x_tri = pt.linalg.cholesky(x, lower=lower).astype(floatX)

        if unit_diag:
            n = A_val.shape[0]
            x_tri = x_tri[np.diag_indices(n)].set(1.0)

        return transpose_func(x_tri.astype(floatX), trans)

    solve_op = partial(
        pt.linalg.solve_triangular, lower=lower, trans=trans, unit_diagonal=unit_diag
    )

    utt.verify_grad(
        lambda A, b: solve_op(A_func_pt(A), b),
        [A_val.copy(), b_val.copy()],
        mode="NUMBA",
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
    A_val = np.random.normal(size=(5, 5)).astype(floatX)
    A_sym = A_val @ A_val.conj().T

    A_tri = np.linalg.cholesky(A_sym).astype(floatX)
    b = np.full((5, 1), value).astype(floatX)

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

    x = np.array([0.1, 0.2, 0.3]).astype(floatX)
    val = np.eye(3).astype(floatX) + x[None, :] * x[:, None]

    compare_numba_and_py([cov], [chol], [val])


def test_numba_Cholesky_raises_on_nan_input():
    test_value = rng.random(size=(3, 3)).astype(floatX)
    test_value[0, 0] = np.nan

    x = pt.tensor(dtype=floatX, shape=(3, 3))
    x = x.T.dot(x)
    g = pt.linalg.cholesky(x)
    f = pytensor.function([x], g, mode="NUMBA")

    with pytest.raises(np.linalg.LinAlgError, match=r"Non-numeric values"):
        f(test_value)


@pytest.mark.parametrize("on_error", ["nan", "raise"])
def test_numba_Cholesky_raise_on(on_error):
    test_value = rng.random(size=(3, 3)).astype(floatX)

    x = pt.tensor(dtype=floatX, shape=(3, 3))
    g = pt.linalg.cholesky(x, on_error=on_error)
    f = pytensor.function([x], g, mode="NUMBA")

    if on_error == "raise":
        with pytest.raises(
            np.linalg.LinAlgError, match=r"Input to cholesky is not positive definite"
        ):
            f(test_value)
    else:
        assert np.all(np.isnan(f(test_value)))


@pytest.mark.parametrize("lower", [True, False], ids=["lower=True", "lower=False"])
def test_numba_Cholesky_grad(lower):
    rng = np.random.default_rng(utt.fetch_seed())
    L = rng.normal(size=(5, 5)).astype(floatX)
    X = L @ L.T

    chol_op = partial(pt.linalg.cholesky, lower=lower)
    utt.verify_grad(chol_op, [X], mode="NUMBA")


def test_block_diag():
    A = pt.matrix("A")
    B = pt.matrix("B")
    C = pt.matrix("C")
    D = pt.matrix("D")
    X = pt.linalg.block_diag(A, B, C, D)

    A_val = np.random.normal(size=(5, 5)).astype(floatX)
    B_val = np.random.normal(size=(3, 3)).astype(floatX)
    C_val = np.random.normal(size=(2, 2)).astype(floatX)
    D_val = np.random.normal(size=(4, 4)).astype(floatX)
    compare_numba_and_py([A, B, C, D], [X], [A_val, B_val, C_val, D_val])


def test_lamch():
    from scipy.linalg import get_lapack_funcs

    from pytensor.link.numba.dispatch.slinalg import _xlamch

    @numba.njit()
    def xlamch(kind):
        return _xlamch(kind)

    lamch = get_lapack_funcs("lamch", (np.array([0.0], dtype=floatX),))

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

    x = np.random.normal(size=(5, 5)).astype(floatX)
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

    x = np.random.normal(size=(5, 5)).astype(floatX)

    rcond, info = gecon(x, norm=ord_numba)

    # Test against direct call to the underlying LAPACK functions
    # Solution does **not** agree with 1 / np.linalg.cond(x) !
    lange, gecon = get_lapack_funcs(("lange", "gecon"), (x,))
    norm = lange(ord_numba, x)
    rcond2, _ = gecon(x, norm, norm=ord_numba)

    assert info == 0
    np.testing.assert_allclose(rcond, rcond2)


@pytest.mark.parametrize("overwrite_a", [True, False])
def test_getrf(overwrite_a):
    from scipy.linalg import lu_factor

    from pytensor.link.numba.dispatch.slinalg import _getrf

    # TODO: Refactor this test to use compare_numba_and_py after we implement lu_factor in pytensor

    @numba.njit()
    def getrf(x, overwrite_a):
        return _getrf(x, overwrite_a=overwrite_a)

    x = np.random.normal(size=(5, 5)).astype(floatX)
    x = np.asfortranarray(
        x
    )  # x needs to be fortran-contiguous going into getrf for the overwrite option to work

    lu, ipiv = lu_factor(x, overwrite_a=False)
    LU, IPIV, info = getrf(x, overwrite_a=overwrite_a)

    assert info == 0
    assert_allclose(LU, lu)

    if overwrite_a:
        assert_allclose(x, LU)

    # TODO: It seems IPIV is 1-indexed in FORTRAN, so we need to subtract 1. I can't find evidence that scipy is doing
    #  this, though.
    assert_allclose(IPIV - 1, ipiv)


@pytest.mark.parametrize("trans", [0, 1])
@pytest.mark.parametrize("overwrite_a", [True, False])
@pytest.mark.parametrize("overwrite_b", [True, False])
@pytest.mark.parametrize("b_shape", [(5,), (5, 3)], ids=["b_1d", "b_2d"])
def test_getrs(trans, overwrite_a, overwrite_b, b_shape):
    from scipy.linalg import lu_factor
    from scipy.linalg import lu_solve as sp_lu_solve

    from pytensor.link.numba.dispatch.slinalg import _getrf, _getrs

    # TODO: Refactor this test to use compare_numba_and_py after we implement lu_solve in pytensor

    @numba.njit()
    def lu_solve(a, b, trans, overwrite_a, overwrite_b):
        lu, ipiv, info = _getrf(a, overwrite_a=overwrite_a)
        x, info = _getrs(lu, b, ipiv, trans=trans, overwrite_b=overwrite_b)
        return x, lu, info

    a = np.random.normal(size=(5, 5)).astype(floatX)
    b = np.random.normal(size=b_shape).astype(floatX)

    # inputs need to be fortran-contiguous going into getrf and getrs for the overwrite option to work
    a = np.asfortranarray(a)
    b = np.asfortranarray(b)

    lu_and_piv = lu_factor(a, overwrite_a=False)
    x_sp = sp_lu_solve(lu_and_piv, b, trans, overwrite_b=False)

    x, lu, info = lu_solve(
        a, b, trans, overwrite_a=overwrite_a, overwrite_b=overwrite_b
    )
    assert info == 0
    if overwrite_a:
        assert_allclose(a, lu)
    if overwrite_b:
        assert_allclose(b, x)

    assert_allclose(x, x_sp)


@pytest.mark.parametrize(
    "b_shape",
    [(5, 1), (5, 5), (5,)],
    ids=["b_col_vec", "b_matrix", "b_vec"],
)
@pytest.mark.parametrize("assume_a", ["gen", "sym", "pos"], ids=str)
@pytest.mark.filterwarnings(
    'ignore:Cannot cache compiled function "numba_funcified_fgraph"'
)
def test_solve(b_shape: tuple[int], assume_a: Literal["gen", "sym", "pos"]):
    A = pt.matrix("A", dtype=floatX)
    b = pt.tensor("b", shape=b_shape, dtype=floatX)

    A_val = np.asfortranarray(np.random.normal(size=(5, 5)).astype(floatX))
    b_val = np.asfortranarray(np.random.normal(size=b_shape).astype(floatX))

    def A_func(x):
        if assume_a == "pos":
            x = x @ x.T
        elif assume_a == "sym":
            x = (x + x.T) / 2
        return x

    X = pt.linalg.solve(
        A_func(A),
        b,
        assume_a=assume_a,
        b_ndim=len(b_shape),
    )
    f = pytensor.function(
        [pytensor.In(A, mutable=True), pytensor.In(b, mutable=True)], X, mode="NUMBA"
    )
    op = f.maker.fgraph.outputs[0].owner.op

    compare_numba_and_py([A, b], [X], test_inputs=[A_val, b_val], inplace=True)

    # Calling this is destructive and will rewrite b_val to be the answer. Store copies of the inputs first.
    A_val_copy = A_val.copy()
    b_val_copy = b_val.copy()

    X_np = f(A_val, b_val)

    # overwrite_b is preferred when both inputs can be destroyed
    assert op.destroy_map == {0: [1]}

    # Confirm inputs were destroyed by checking against the copies
    assert (A_val == A_val_copy).all() == (op.destroy_map.get(0, None) != [0])
    assert (b_val == b_val_copy).all() == (op.destroy_map.get(0, None) != [1])

    ATOL = 1e-8 if floatX.endswith("64") else 1e-4
    RTOL = 1e-8 if floatX.endswith("64") else 1e-4

    # Confirm b_val is used to store to solution
    np.testing.assert_allclose(X_np, b_val, atol=ATOL, rtol=RTOL)
    assert not np.allclose(b_val, b_val_copy)

    # Test that the result is numerically correct. Need to use the unmodified copy
    np.testing.assert_allclose(
        A_func(A_val_copy) @ X_np, b_val_copy, atol=ATOL, rtol=RTOL
    )

    # See the note in tensor/test_slinalg.py::test_solve_correctness for details about the setup here
    utt.verify_grad(
        lambda A, b: pt.linalg.solve(
            A_func(A), b, lower=False, assume_a=assume_a, b_ndim=len(b_shape)
        ),
        [A_val_copy, b_val_copy],
        mode="NUMBA",
    )


@pytest.mark.parametrize(
    "b_func, b_size",
    [(pt.matrix, (5, 1)), (pt.matrix, (5, 5)), (pt.vector, (5,))],
    ids=["b_col_vec", "b_matrix", "b_vec"],
)
@pytest.mark.parametrize("lower", [True, False], ids=lambda x: f"lower = {x}")
def test_cho_solve(b_func, b_size, lower):
    A = pt.matrix("A", dtype=floatX)
    b = b_func("b", dtype=floatX)

    C = pt.linalg.cholesky(A, lower=lower)
    X = pt.linalg.cho_solve((C, lower), b)
    f = pytensor.function([A, b], X, mode="NUMBA")

    A = np.random.normal(size=(5, 5)).astype(floatX)
    A = A @ A.conj().T

    b = np.random.normal(size=b_size)
    b = b.astype(floatX)

    X_np = f(A, b)

    ATOL = 1e-8 if floatX.endswith("64") else 1e-4
    RTOL = 1e-8 if floatX.endswith("64") else 1e-4

    np.testing.assert_allclose(A @ X_np, b, atol=ATOL, rtol=RTOL)
