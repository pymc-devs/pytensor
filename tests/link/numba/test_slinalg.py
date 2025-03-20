import re
from typing import Literal

import numpy as np
import pytest
import scipy

import pytensor
import pytensor.tensor as pt
from pytensor import In, config
from pytensor.tensor.slinalg import Cholesky, CholeskySolve, Solve, SolveTriangular
from tests.link.numba.test_basic import compare_numba_and_py, numba_inplace_mode


numba = pytest.importorskip("numba")

floatX = config.floatX

rng = np.random.default_rng(42849)


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


class TestSolves:
    @pytest.mark.parametrize("lower", [True, False], ids=lambda x: f"lower={x}")
    @pytest.mark.parametrize(
        "overwrite_a, overwrite_b",
        [(False, False), (True, False), (False, True)],
        ids=["no_overwrite", "overwrite_a", "overwrite_b"],
    )
    @pytest.mark.parametrize(
        "b_shape",
        [(5, 1), (5, 5), (5,)],
        ids=["b_col_vec", "b_matrix", "b_vec"],
    )
    @pytest.mark.parametrize("assume_a", ["gen", "sym", "pos"], ids=str)
    def test_solve(
        self,
        b_shape: tuple[int],
        assume_a: Literal["gen", "sym", "pos"],
        lower: bool,
        overwrite_a: bool,
        overwrite_b: bool,
    ):
        if assume_a not in ("sym", "her", "pos") and not lower:
            # Avoid redundant tests with lower=True and lower=False for non symmetric matrices
            pytest.skip("Skipping redundant test already covered by lower=True")

        def A_func(x):
            if assume_a == "pos":
                x = x @ x.T
                x = np.tril(x) if lower else np.triu(x)
            elif assume_a == "sym":
                x = (x + x.T) / 2
                n = x.shape[0]
                # We have to set the unused triangle to something other than zero
                # to see lapack destroying it.
                x[np.triu_indices(n, 1) if lower else np.tril_indices(n, 1)] = np.pi
            return x

        A = pt.matrix("A", dtype=floatX)
        b = pt.tensor("b", shape=b_shape, dtype=floatX)

        rng = np.random.default_rng(418)
        A_val = A_func(rng.normal(size=(5, 5))).astype(floatX)
        b_val = rng.normal(size=b_shape).astype(floatX)

        X = pt.linalg.solve(
            A,
            b,
            assume_a=assume_a,
            b_ndim=len(b_shape),
        )

        f, res = compare_numba_and_py(
            [In(A, mutable=overwrite_a), In(b, mutable=overwrite_b)],
            X,
            test_inputs=[A_val, b_val],
            inplace=True,
            numba_mode=numba_inplace_mode,
        )

        op = f.maker.fgraph.outputs[0].owner.op
        assert isinstance(op, Solve)
        destroy_map = op.destroy_map
        if overwrite_a and overwrite_b:
            raise NotImplementedError(
                "Test not implemented for simultaneous overwrite_a and overwrite_b, as that's not currently supported by PyTensor"
            )
        elif overwrite_a:
            assert destroy_map == {0: [0]}
        elif overwrite_b:
            assert destroy_map == {0: [1]}
        else:
            assert destroy_map == {}

        # Test with F_contiguous inputs
        A_val_f_contig = np.copy(A_val, order="F")
        b_val_f_contig = np.copy(b_val, order="F")
        res_f_contig = f(A_val_f_contig, b_val_f_contig)
        np.testing.assert_allclose(res_f_contig, res)
        # Should always be destroyable
        assert (A_val == A_val_f_contig).all() == (not overwrite_a)
        assert (b_val == b_val_f_contig).all() == (not overwrite_b)

        # Test with C_contiguous inputs
        A_val_c_contig = np.copy(A_val, order="C")
        b_val_c_contig = np.copy(b_val, order="C")
        res_c_contig = f(A_val_c_contig, b_val_c_contig)
        np.testing.assert_allclose(res_c_contig, res)
        # We can destroy C-contiguous A arrays by inverting `tranpose/lower` at runtime
        assert np.allclose(A_val_c_contig, A_val) == (not overwrite_a)
        # b vectors are always f_contiguous if also c_contiguous
        assert np.allclose(b_val_c_contig, b_val) == (
            not (overwrite_b and b_val_c_contig.flags.f_contiguous)
        )

        # Test right results if inputs are not contiguous in either format
        A_val_not_contig = np.repeat(A_val, 2, axis=0)[::2]
        b_val_not_contig = np.repeat(b_val, 2, axis=0)[::2]
        res_not_contig = f(A_val_not_contig, b_val_not_contig)
        np.testing.assert_allclose(res_not_contig, res)
        # Can never destroy non-contiguous inputs
        np.testing.assert_allclose(A_val_not_contig, A_val)
        np.testing.assert_allclose(b_val_not_contig, b_val)

    @pytest.mark.parametrize("lower", [True, False], ids=lambda x: f"lower={x}")
    @pytest.mark.parametrize(
        "transposed", [False, True], ids=lambda x: f"transposed={x}"
    )
    @pytest.mark.parametrize(
        "overwrite_b", [False, True], ids=["no_overwrite", "overwrite_b"]
    )
    @pytest.mark.parametrize(
        "unit_diagonal", [True, False], ids=lambda x: f"unit_diagonal={x}"
    )
    @pytest.mark.parametrize(
        "b_shape",
        [(5, 1), (5, 5), (5,)],
        ids=["b_col_vec", "b_matrix", "b_vec"],
    )
    @pytest.mark.parametrize("is_complex", [True, False], ids=["complex", "real"])
    def test_solve_triangular(
        self,
        b_shape: tuple[int],
        lower: bool,
        transposed: bool,
        unit_diagonal: bool,
        is_complex: bool,
        overwrite_b: bool,
    ):
        if is_complex:
            # TODO: Complex raises ValueError: To change to a dtype of a different size, the last axis must be contiguous,
            #  why?
            pytest.skip("Complex inputs currently not supported to solve_triangular")

        def A_func(x):
            complex_dtype = "complex64" if floatX.endswith("32") else "complex128"
            dtype = complex_dtype if is_complex else floatX

            x = x @ x.conj().T
            x_tri = scipy.linalg.cholesky(x, lower=lower).astype(dtype)

            if unit_diagonal:
                x_tri[np.diag_indices(x_tri.shape[0])] = 1.0

            return x_tri

        A = pt.matrix("A", dtype=floatX)
        b = pt.tensor("b", shape=b_shape, dtype=floatX)

        rng = np.random.default_rng(418)
        A_val = A_func(rng.normal(size=(5, 5))).astype(floatX)
        b_val = rng.normal(size=b_shape).astype(floatX)

        X = pt.linalg.solve_triangular(
            A,
            b,
            lower=lower,
            trans="N" if (not transposed) else ("C" if is_complex else "T"),
            unit_diagonal=unit_diagonal,
            b_ndim=len(b_shape),
        )

        f, res = compare_numba_and_py(
            [A, In(b, mutable=overwrite_b)],
            X,
            test_inputs=[A_val, b_val],
            inplace=True,
            numba_mode=numba_inplace_mode,
        )

        op = f.maker.fgraph.outputs[0].owner.op
        assert isinstance(op, SolveTriangular)
        destroy_map = op.destroy_map
        if overwrite_b:
            assert destroy_map == {0: [1]}
        else:
            assert destroy_map == {}

        # Test with F_contiguous inputs
        A_val_f_contig = np.copy(A_val, order="F")
        b_val_f_contig = np.copy(b_val, order="F")
        res_f_contig = f(A_val_f_contig, b_val_f_contig)
        np.testing.assert_allclose(res_f_contig, res)
        # solve_triangular never destroys A
        np.testing.assert_allclose(A_val, A_val_f_contig)
        # b Should always be destroyable
        assert (b_val == b_val_f_contig).all() == (not overwrite_b)

        # Test with C_contiguous inputs
        A_val_c_contig = np.copy(A_val, order="C")
        b_val_c_contig = np.copy(b_val, order="C")
        res_c_contig = f(A_val_c_contig, b_val_c_contig)
        np.testing.assert_allclose(res_c_contig, res)
        np.testing.assert_allclose(A_val_c_contig, A_val)
        # b c_contiguous vectors are also f_contiguous and destroyable
        assert np.allclose(b_val_c_contig, b_val) == (
            not (overwrite_b and b_val_c_contig.flags.f_contiguous)
        )

        # Test with non-contiguous inputs
        A_val_not_contig = np.repeat(A_val, 2, axis=0)[::2]
        b_val_not_contig = np.repeat(b_val, 2, axis=0)[::2]
        res_not_contig = f(A_val_not_contig, b_val_not_contig)
        np.testing.assert_allclose(res_not_contig, res)
        np.testing.assert_allclose(A_val_not_contig, A_val)
        # Can never destroy non-contiguous inputs
        np.testing.assert_allclose(b_val_not_contig, b_val)

    @pytest.mark.parametrize("value", [np.nan, np.inf])
    def test_solve_triangular_raises_on_nan_inf(self, value):
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

    @pytest.mark.parametrize("lower", [True, False], ids=lambda x: f"lower = {x}")
    @pytest.mark.parametrize(
        "overwrite_b", [False, True], ids=["no_overwrite", "overwrite_b"]
    )
    @pytest.mark.parametrize(
        "b_func, b_shape",
        [(pt.matrix, (5, 1)), (pt.matrix, (5, 5)), (pt.vector, (5,))],
        ids=["b_col_vec", "b_matrix", "b_vec"],
    )
    def test_cho_solve(
        self, b_func, b_shape: tuple[int, ...], lower: bool, overwrite_b: bool
    ):
        def A_func(x):
            x = x @ x.conj().T
            x = scipy.linalg.cholesky(x, lower=lower)
            return x

        A = pt.matrix("A", dtype=floatX)
        b = pt.tensor("b", shape=b_shape, dtype=floatX)

        rng = np.random.default_rng(418)
        A_val = A_func(rng.normal(size=(5, 5))).astype(floatX)
        b_val = rng.normal(size=b_shape).astype(floatX)

        X = pt.linalg.cho_solve(
            (A, lower),
            b,
            b_ndim=len(b_shape),
        )

        f, res = compare_numba_and_py(
            [A, In(b, mutable=overwrite_b)],
            X,
            test_inputs=[A_val, b_val],
            inplace=True,
            numba_mode=numba_inplace_mode,
        )

        op = f.maker.fgraph.outputs[0].owner.op
        assert isinstance(op, CholeskySolve)
        destroy_map = op.destroy_map
        if overwrite_b:
            assert destroy_map == {0: [1]}
        else:
            assert destroy_map == {}

        # Test with F_contiguous inputs
        A_val_f_contig = np.copy(A_val, order="F")
        b_val_f_contig = np.copy(b_val, order="F")
        res_f_contig = f(A_val_f_contig, b_val_f_contig)
        np.testing.assert_allclose(res_f_contig, res)
        # cho_solve never destroys A
        np.testing.assert_allclose(A_val, A_val_f_contig)
        # b Should always be destroyable
        assert (b_val == b_val_f_contig).all() == (not overwrite_b)

        # Test with C_contiguous inputs
        A_val_c_contig = np.copy(A_val, order="C")
        b_val_c_contig = np.copy(b_val, order="C")
        res_c_contig = f(A_val_c_contig, b_val_c_contig)
        np.testing.assert_allclose(res_c_contig, res)
        np.testing.assert_allclose(A_val_c_contig, A_val)
        # b c_contiguous vectors are also f_contiguous and destroyable
        assert np.allclose(b_val_c_contig, b_val) == (
            not (overwrite_b and b_val_c_contig.flags.f_contiguous)
        )

        # Test with non-contiguous inputs
        A_val_not_contig = np.repeat(A_val, 2, axis=0)[::2]
        b_val_not_contig = np.repeat(b_val, 2, axis=0)[::2]
        res_not_contig = f(A_val_not_contig, b_val_not_contig)
        np.testing.assert_allclose(res_not_contig, res)
        np.testing.assert_allclose(A_val_not_contig, A_val)
        # Can never destroy non-contiguous inputs
        np.testing.assert_allclose(b_val_not_contig, b_val)


@pytest.mark.parametrize("lower", [True, False], ids=lambda x: f"lower={x}")
@pytest.mark.parametrize(
    "overwrite_a", [False, True], ids=["no_overwrite", "overwrite_a"]
)
def test_cholesky(lower: bool, overwrite_a: bool):
    cov = pt.matrix("cov")
    chol = pt.linalg.cholesky(cov, lower=lower)

    x = np.array([0.1, 0.2, 0.3]).astype(floatX)
    val = np.eye(3).astype(floatX) + x[None, :] * x[:, None]

    fn, res = compare_numba_and_py(
        [In(cov, mutable=overwrite_a)],
        [chol],
        [val],
        numba_mode=numba_inplace_mode,
        inplace=True,
    )

    op = fn.maker.fgraph.outputs[0].owner.op
    assert isinstance(op, Cholesky)
    destroy_map = op.destroy_map
    if overwrite_a:
        assert destroy_map == {0: [0]}
    else:
        assert destroy_map == {}

    # Test F-contiguous input
    val_f_contig = np.copy(val, order="F")
    res_f_contig = fn(val_f_contig)
    np.testing.assert_allclose(res_f_contig, res)
    # Should always be destroyable
    assert (val == val_f_contig).all() == (not overwrite_a)

    # Test C-contiguous input
    val_c_contig = np.copy(val, order="C")
    res_c_contig = fn(val_c_contig)
    np.testing.assert_allclose(res_c_contig, res)
    # Cannot destroy C-contiguous input
    np.testing.assert_allclose(val_c_contig, val)

    # Test non-contiguous input
    val_not_contig = np.repeat(val, 2, axis=0)[::2]
    res_not_contig = fn(val_not_contig)
    np.testing.assert_allclose(res_not_contig, res)
    # Cannot destroy non-contiguous input
    np.testing.assert_allclose(val_not_contig, val)


def test_cholesky_raises_on_nan_input():
    test_value = rng.random(size=(3, 3)).astype(floatX)
    test_value[0, 0] = np.nan

    x = pt.tensor(dtype=floatX, shape=(3, 3))
    x = x.T.dot(x)
    g = pt.linalg.cholesky(x)
    f = pytensor.function([x], g, mode="NUMBA")

    with pytest.raises(np.linalg.LinAlgError, match=r"Non-numeric values"):
        f(test_value)


@pytest.mark.parametrize("on_error", ["nan", "raise"])
def test_cholesky_raise_on(on_error):
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
