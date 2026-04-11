from typing import Literal

import numpy as np
import pytest
import scipy

import pytensor
import pytensor.tensor as pt
from pytensor import In, config
from pytensor.tensor.linalg.decomposition.lu import lu_factor
from pytensor.tensor.linalg.solvers.general import Solve, lu_solve, solve
from pytensor.tensor.linalg.solvers.psd import CholeskySolve, cho_solve
from pytensor.tensor.linalg.solvers.triangular import SolveTriangular, solve_triangular
from tests.link.numba.test_basic import compare_numba_and_py, numba_inplace_mode


pytestmark = pytest.mark.filterwarnings("error")

numba = pytest.importorskip("numba")

floatX = config.floatX


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
    @pytest.mark.parametrize(
        "assume_a", ["gen", "sym", "her", "pos", "tridiagonal"], ids=str
    )
    @pytest.mark.parametrize("is_complex", [True, False], ids=["complex", "real"])
    def test_solve(
        self,
        b_shape: tuple[int],
        assume_a: Literal["gen", "sym", "pos"],
        lower: bool,
        overwrite_a: bool,
        overwrite_b: bool,
        is_complex: bool,
    ):
        if assume_a not in ("sym", "her", "pos", "tridiagonal") and not lower:
            pytest.skip("Skipping redundant test already covered by lower=True")

        complex_dtype = "complex64" if floatX.endswith("32") else "complex128"
        dtype = complex_dtype if is_complex else floatX

        def A_func(x):
            if assume_a == "pos":
                x = x @ x.conj().T
                x = np.tril(x) if lower else np.triu(x)
            elif assume_a == "sym":
                x = (x + x.T) / 2
                n = x.shape[0]
                x[np.triu_indices(n, 1) if lower else np.tril_indices(n, 1)] = np.pi
            elif assume_a == "her":
                x = (x + x.conj().T) / 2
                n = x.shape[0]
                x[np.triu_indices(n, 1) if lower else np.tril_indices(n, 1)] = np.pi
            elif assume_a == "tridiagonal":
                _x = x
                x = np.zeros_like(x)
                n = x.shape[-1]
                arange_n = np.arange(n)
                x[arange_n[1:], arange_n[:-1]] = np.diag(_x, k=-1)
                x[arange_n, arange_n] = np.diag(_x, k=0)
                x[arange_n[:-1], arange_n[1:]] = np.diag(_x, k=1)
            return x

        A = pt.matrix("A", dtype=dtype)
        b = pt.tensor("b", shape=b_shape, dtype=dtype)

        rng = np.random.default_rng(418)
        A_base = rng.normal(size=(5, 5))
        if is_complex:
            A_base = A_base + 1j * rng.normal(size=(5, 5))
        A_val = A_func(A_base).astype(dtype)
        b_val = rng.normal(size=b_shape).astype(dtype)
        if is_complex:
            b_val = b_val + 1j * rng.normal(size=b_shape).astype(dtype)

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
        assert op.assume_a == assume_a
        destroy_map = op.destroy_map

        if overwrite_a and assume_a == "tridiagonal":
            overwrite_a = False

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

        A_val_f_contig = np.copy(A_val, order="F")
        b_val_f_contig = np.copy(b_val, order="F")
        res_f_contig = f(A_val_f_contig, b_val_f_contig)
        np.testing.assert_allclose(res_f_contig, res)
        assert (A_val == A_val_f_contig).all() == (not overwrite_a)
        assert (b_val == b_val_f_contig).all() == (not overwrite_b)

        A_val_c_contig = np.copy(A_val, order="C")
        b_val_c_contig = np.copy(b_val, order="C")
        res_c_contig = f(A_val_c_contig, b_val_c_contig)
        np.testing.assert_allclose(res_c_contig, res)
        can_destroy_c_contig_A = overwrite_a and not (
            is_complex and assume_a in ("pos", "her")
        )
        assert np.allclose(A_val_c_contig, A_val) == (not can_destroy_c_contig_A)
        assert np.allclose(b_val_c_contig, b_val) == (
            not (overwrite_b and b_val_c_contig.flags.f_contiguous)
        )

        A_val_not_contig = np.repeat(A_val, 2, axis=0)[::2]
        b_val_not_contig = np.repeat(b_val, 2, axis=0)[::2]
        res_not_contig = f(A_val_not_contig, b_val_not_contig)
        np.testing.assert_allclose(res_not_contig, res)
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
        complex_dtype = "complex64" if floatX.endswith("32") else "complex128"
        dtype = complex_dtype if is_complex else floatX

        def A_func(x):
            x = x @ x.conj().T
            x_tri = scipy.linalg.cholesky(x, lower=lower).astype(dtype)
            if unit_diagonal:
                x_tri[np.diag_indices(x_tri.shape[0])] = 1.0
            return x_tri

        A = pt.matrix("A", dtype=dtype)
        b = pt.tensor("b", shape=b_shape, dtype=dtype)

        rng = np.random.default_rng(418)
        A_base = rng.normal(size=(5, 5))
        if is_complex:
            A_base = A_base + 1j * rng.normal(size=(5, 5))
        A_val = A_func(A_base).astype(dtype)
        b_val = rng.normal(size=b_shape).astype(dtype)
        if is_complex:
            b_val = b_val + 1j * rng.normal(size=b_shape).astype(dtype)

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

        A_val_f_contig = np.copy(A_val, order="F")
        b_val_f_contig = np.copy(b_val, order="F")
        res_f_contig = f(A_val_f_contig, b_val_f_contig)
        np.testing.assert_allclose(res_f_contig, res)
        np.testing.assert_allclose(A_val, A_val_f_contig)
        assert (b_val == b_val_f_contig).all() == (not overwrite_b)

        A_val_c_contig = np.copy(A_val, order="C")
        b_val_c_contig = np.copy(b_val, order="C")
        res_c_contig = f(A_val_c_contig, b_val_c_contig)
        np.testing.assert_allclose(res_c_contig, res)
        np.testing.assert_allclose(A_val_c_contig, A_val)
        assert np.allclose(b_val_c_contig, b_val) == (
            not (overwrite_b and b_val_c_contig.flags.f_contiguous)
        )

        A_val_not_contig = np.repeat(A_val, 2, axis=0)[::2]
        b_val_not_contig = np.repeat(b_val, 2, axis=0)[::2]
        res_not_contig = f(A_val_not_contig, b_val_not_contig)
        np.testing.assert_allclose(res_not_contig, res)
        np.testing.assert_allclose(A_val_not_contig, A_val)
        np.testing.assert_allclose(b_val_not_contig, b_val)

    @pytest.mark.parametrize("value", [np.nan, np.inf])
    def test_solve_triangular_does_not_raise_on_nan_inf(self, value):
        A = pt.matrix("A")
        b = pt.matrix("b")

        X = pt.linalg.solve_triangular(A, b, check_finite=True)
        f = pytensor.function([A, b], X, mode="NUMBA")
        A_val = np.random.normal(size=(5, 5)).astype(floatX)
        A_sym = A_val @ A_val.conj().T

        A_tri = np.linalg.cholesky(A_sym).astype(floatX)
        b = np.full((5, 1), value).astype(floatX)

        assert not np.isfinite(f(A_tri, b)).any()

    @pytest.mark.parametrize("lower", [True, False], ids=lambda x: f"lower = {x}")
    @pytest.mark.parametrize(
        "overwrite_b", [False, True], ids=["no_overwrite", "overwrite_b"]
    )
    @pytest.mark.parametrize(
        "b_func, b_shape",
        [(pt.matrix, (5, 1)), (pt.matrix, (5, 5)), (pt.vector, (5,))],
        ids=["b_col_vec", "b_matrix", "b_vec"],
    )
    @pytest.mark.parametrize("is_complex", [True, False], ids=["complex", "real"])
    def test_cho_solve(
        self,
        b_func,
        b_shape: tuple[int, ...],
        lower: bool,
        overwrite_b: bool,
        is_complex: bool,
    ):
        complex_dtype = "complex64" if floatX.endswith("32") else "complex128"
        dtype = complex_dtype if is_complex else floatX

        def A_func(x):
            x = x @ x.conj().T
            x = scipy.linalg.cholesky(x, lower=lower)
            return x

        A = pt.matrix("A", dtype=dtype)
        b = pt.tensor("b", shape=b_shape, dtype=dtype)

        rng = np.random.default_rng(418)
        A_base = rng.normal(size=(5, 5))
        if is_complex:
            A_base = A_base + 1j * rng.normal(size=(5, 5))
        A_val = A_func(A_base).astype(dtype)
        b_val = rng.normal(size=b_shape).astype(dtype)
        if is_complex:
            b_val = b_val + 1j * rng.normal(size=b_shape).astype(dtype)

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

        A_val_f_contig = np.copy(A_val, order="F")
        b_val_f_contig = np.copy(b_val, order="F")
        res_f_contig = f(A_val_f_contig, b_val_f_contig)
        np.testing.assert_allclose(res_f_contig, res)
        np.testing.assert_allclose(A_val, A_val_f_contig)
        assert (b_val == b_val_f_contig).all() == (not overwrite_b)

        A_val_c_contig = np.copy(A_val, order="C")
        b_val_c_contig = np.copy(b_val, order="C")
        res_c_contig = f(A_val_c_contig, b_val_c_contig)
        np.testing.assert_allclose(res_c_contig, res)
        np.testing.assert_allclose(A_val_c_contig, A_val)
        assert np.allclose(b_val_c_contig, b_val) == (
            not (overwrite_b and b_val_c_contig.flags.f_contiguous)
        )

        A_val_not_contig = np.repeat(A_val, 2, axis=0)[::2]
        b_val_not_contig = np.repeat(b_val, 2, axis=0)[::2]
        res_not_contig = f(A_val_not_contig, b_val_not_contig)
        np.testing.assert_allclose(res_not_contig, res)
        np.testing.assert_allclose(A_val_not_contig, A_val)
        np.testing.assert_allclose(b_val_not_contig, b_val)

    @pytest.mark.parametrize("trans", [True, False], ids=lambda x: f"trans = {x}")
    @pytest.mark.parametrize(
        "overwrite_b", [False, True], ids=["no_overwrite", "overwrite_b"]
    )
    @pytest.mark.parametrize(
        "b_func, b_shape",
        [(pt.matrix, (5, 1)), (pt.matrix, (5, 5)), (pt.vector, (5,))],
        ids=["b_col_vec", "b_matrix", "b_vec"],
    )
    def test_lu_solve(
        self, b_func, b_shape: tuple[int, ...], trans: bool, overwrite_b: bool
    ):
        A = pt.matrix("A", dtype=floatX)
        b = pt.tensor("b", shape=b_shape, dtype=floatX)

        rng = np.random.default_rng(418)
        A_val = rng.normal(size=(5, 5)).astype(floatX)
        b_val = rng.normal(size=b_shape).astype(floatX)

        lu_and_piv = lu_factor(A)
        X = pt.linalg.lu_solve(
            lu_and_piv,
            b,
            b_ndim=len(b_shape),
            trans=trans,
        )

        f, res = compare_numba_and_py(
            [A, In(b, mutable=overwrite_b)],
            X,
            test_inputs=[A_val, b_val],
            inplace=True,
            numba_mode=numba_inplace_mode,
            eval_obj_mode=False,
        )

        A_val_f_contig = np.copy(A_val, order="F")
        b_val_f_contig = np.copy(b_val, order="F")
        res_f_contig = f(A_val_f_contig, b_val_f_contig)
        np.testing.assert_allclose(res_f_contig, res)

        all_equal = (b_val == b_val_f_contig).all()
        should_destroy = overwrite_b and trans

        if should_destroy:
            assert not all_equal
        else:
            assert all_equal

        A_val_c_contig = np.copy(A_val, order="C")
        b_val_c_contig = np.copy(b_val, order="C")
        res_c_contig = f(A_val_c_contig, b_val_c_contig)

        np.testing.assert_allclose(res_c_contig, res)
        np.testing.assert_allclose(A_val_c_contig, A_val)

        assert not (
            should_destroy and b_val_c_contig.flags.f_contiguous
        ) == np.allclose(b_val_c_contig, b_val)

        A_val_not_contig = np.repeat(A_val, 2, axis=0)[::2]
        b_val_not_contig = np.repeat(b_val, 2, axis=0)[::2]
        res_not_contig = f(A_val_not_contig, b_val_not_contig)
        np.testing.assert_allclose(res_not_contig, res)
        np.testing.assert_allclose(A_val_not_contig, A_val)

        np.testing.assert_allclose(b_val_not_contig, b_val)

    @pytest.mark.parametrize(
        "solve_op",
        [solve, solve_triangular, cho_solve, lu_solve],
        ids=lambda x: x.__name__,
    )
    def test_empty(self, solve_op):
        a = pt.matrix("x")
        b = pt.vector("b")
        if solve_op is cho_solve:
            out = solve_op((a, True), b)
        elif solve_op is lu_solve:
            out = solve_op((a, b.astype("int32")), b)
        else:
            out = solve_op(a, b)
        compare_numba_and_py(
            [a, b],
            [out],
            [np.zeros((0, 0)), np.zeros(0)],
            eval_obj_mode=False,
        )
