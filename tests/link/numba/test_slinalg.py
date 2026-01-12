from typing import Literal

import numpy as np
import pytest
import scipy

import pytensor
import pytensor.tensor as pt
from pytensor import In, config
from pytensor.tensor.slinalg import (
    LU,
    QR,
    Cholesky,
    CholeskySolve,
    LUFactor,
    Solve,
    SolveTriangular,
    cho_solve,
    cholesky,
    lu,
    lu_factor,
    lu_solve,
    schur,
    solve,
    solve_triangular,
)
from tests.link.numba.test_basic import compare_numba_and_py, numba_inplace_mode


pytestmark = pytest.mark.filterwarnings("error")

numba = pytest.importorskip("numba")

floatX = config.floatX

rng = np.random.default_rng(42849)


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
    @pytest.mark.parametrize("assume_a", ["gen", "sym", "pos", "tridiagonal"], ids=str)
    def test_solve(
        self,
        b_shape: tuple[int],
        assume_a: Literal["gen", "sym", "pos"],
        lower: bool,
        overwrite_a: bool,
        overwrite_b: bool,
    ):
        if assume_a not in ("sym", "her", "pos", "tridiagonal") and not lower:
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
            elif assume_a == "tridiagonal":
                _x = x
                x = np.zeros_like(x)
                n = x.shape[-1]
                arange_n = np.arange(n)
                x[arange_n[1:], arange_n[:-1]] = np.diag(_x, k=-1)
                x[arange_n, arange_n] = np.diag(_x, k=0)
                x[arange_n[:-1], arange_n[1:]] = np.diag(_x, k=1)
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
        assert op.assume_a == assume_a
        destroy_map = op.destroy_map

        if overwrite_a and assume_a == "tridiagonal":
            # Tridiagonal solve never destroys the A matrix
            # Treat test from here as if overwrite_a is False
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
    def test_solve_triangular_does_not_raise_on_nan_inf(self, value):
        A = pt.matrix("A")
        b = pt.matrix("b")

        X = pt.linalg.solve_triangular(A, b, check_finite=True)
        f = pytensor.function([A, b], X, mode="NUMBA")
        A_val = np.random.normal(size=(5, 5)).astype(floatX)
        A_sym = A_val @ A_val.conj().T

        A_tri = np.linalg.cholesky(A_sym).astype(floatX)
        b = np.full((5, 1), value).astype(floatX)

        # Not checking everything is nan, because, with inf, LAPACK returns a mix of inf/nan, but does not set info != 0
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

        lu_and_piv = pt.linalg.lu_factor(A)
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

        # Test with F_contiguous inputs
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

        # Test with C_contiguous inputs
        A_val_c_contig = np.copy(A_val, order="C")
        b_val_c_contig = np.copy(b_val, order="C")
        res_c_contig = f(A_val_c_contig, b_val_c_contig)

        np.testing.assert_allclose(res_c_contig, res)
        np.testing.assert_allclose(A_val_c_contig, A_val)

        # b c_contiguous vectors are also f_contiguous and destroyable
        assert not (
            should_destroy and b_val_c_contig.flags.f_contiguous
        ) == np.allclose(b_val_c_contig, b_val)

        # Test with non-contiguous inputs
        A_val_not_contig = np.repeat(A_val, 2, axis=0)[::2]
        b_val_not_contig = np.repeat(b_val, 2, axis=0)[::2]
        res_not_contig = f(A_val_not_contig, b_val_not_contig)
        np.testing.assert_allclose(res_not_contig, res)
        np.testing.assert_allclose(A_val_not_contig, A_val)

        # Can never destroy non-contiguous inputs
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
            eval_obj_mode=False,  # pivot_to_permutation seems to still be jitted despite the monkey patching
        )


class TestDecompositions:
    @pytest.mark.parametrize("lower", [True, False], ids=lambda x: f"lower={x}")
    @pytest.mark.parametrize(
        "overwrite_a", [False, True], ids=["no_overwrite", "overwrite_a"]
    )
    def test_cholesky(self, lower: bool, overwrite_a: bool):
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
        # Should always be destroyable
        assert (val == val_c_contig).all() == (not overwrite_a)

        # Test non-contiguous input
        val_not_contig = np.repeat(val, 2, axis=0)[::2]
        res_not_contig = fn(val_not_contig)
        np.testing.assert_allclose(res_not_contig, res)
        # Cannot destroy non-contiguous input
        np.testing.assert_allclose(val_not_contig, val)

    def test_cholesky_raises_on_nan_input(self):
        test_value = rng.random(size=(3, 3)).astype(floatX)
        test_value[0, 0] = np.nan

        x = pt.tensor(dtype=floatX, shape=(3, 3))
        x = x.T.dot(x)
        with pytest.warns(FutureWarning):
            g = pt.linalg.cholesky(x, check_finite=True, on_error="raise")
        f = pytensor.function([x], g, mode="NUMBA")

        with pytest.raises(
            np.linalg.LinAlgError, match=r"Matrix is not positive definite"
        ):
            f(test_value)

    @pytest.mark.parametrize("on_error", ["nan", "raise"])
    def test_cholesky_raise_on(self, on_error):
        test_value = rng.random(size=(3, 3)).astype(floatX)

        x = pt.tensor(dtype=floatX, shape=(3, 3))
        if on_error == "raise":
            with pytest.warns(FutureWarning):
                g = pt.linalg.cholesky(x, on_error=on_error)
        else:
            g = pt.linalg.cholesky(x, on_error=on_error)
        f = pytensor.function([x], g, mode="NUMBA")

        if on_error == "raise":
            with pytest.raises(
                np.linalg.LinAlgError,
                match=r"Matrix is not positive definite",
            ):
                f(test_value)
        else:
            assert np.all(np.isnan(f(test_value)))

    @pytest.mark.parametrize(
        "permute_l, p_indices",
        [(True, False), (False, True), (False, False)],
        ids=["PL", "p_indices", "P"],
    )
    @pytest.mark.parametrize(
        "overwrite_a", [True, False], ids=["overwrite_a", "no_overwrite"]
    )
    def test_lu(self, permute_l, p_indices, overwrite_a):
        shape = (5, 5)
        rng = np.random.default_rng()
        A = pt.tensor(
            "A",
            shape=shape,
            dtype=config.floatX,
        )
        A_val = rng.normal(size=shape).astype(config.floatX)

        lu_outputs = pt.linalg.lu(A, permute_l=permute_l, p_indices=p_indices)

        fn, res = compare_numba_and_py(
            [In(A, mutable=overwrite_a)],
            lu_outputs,
            [A_val],
            numba_mode=numba_inplace_mode,
            inplace=True,
        )

        op = fn.maker.fgraph.outputs[0].owner.op
        assert isinstance(op, LU)

        destroy_map = op.destroy_map

        if overwrite_a and permute_l:
            assert destroy_map == {0: [0]}
        elif overwrite_a:
            assert destroy_map == {1: [0]}
        else:
            assert destroy_map == {}

        # Test F-contiguous input
        val_f_contig = np.copy(A_val, order="F")
        res_f_contig = fn(val_f_contig)

        for x, x_f_contig in zip(res, res_f_contig, strict=True):
            np.testing.assert_allclose(x, x_f_contig)

        # Should always be destroyable
        assert (A_val == val_f_contig).all() == (not overwrite_a)

        # Test C-contiguous input
        val_c_contig = np.copy(A_val, order="C")
        res_c_contig = fn(val_c_contig)
        for x, x_c_contig in zip(res, res_c_contig, strict=True):
            np.testing.assert_allclose(x, x_c_contig)

        # Cannot destroy C-contiguous input
        np.testing.assert_allclose(val_c_contig, A_val)

        # Test non-contiguous input
        val_not_contig = np.repeat(A_val, 2, axis=0)[::2]
        res_not_contig = fn(val_not_contig)
        for x, x_not_contig in zip(res, res_not_contig, strict=True):
            np.testing.assert_allclose(x, x_not_contig)

        # Cannot destroy non-contiguous input
        np.testing.assert_allclose(val_not_contig, A_val)

    @pytest.mark.parametrize(
        "overwrite_a", [True, False], ids=["overwrite_a", "no_overwrite"]
    )
    def test_lu_factor(self, overwrite_a):
        shape = (5, 5)
        rng = np.random.default_rng()

        A = pt.tensor("A", shape=shape, dtype=config.floatX)
        A_val = rng.normal(size=shape).astype(config.floatX)

        LU, piv = pt.linalg.lu_factor(A)

        fn, res = compare_numba_and_py(
            [In(A, mutable=overwrite_a)],
            [LU, piv],
            [A_val],
            numba_mode=numba_inplace_mode,
            inplace=True,
        )

        op = fn.maker.fgraph.outputs[0].owner.op
        assert isinstance(op, LUFactor)

        if overwrite_a:
            assert op.destroy_map == {1: [0]}

        # Test F-contiguous input
        val_f_contig = np.copy(A_val, order="F")
        res_f_contig = fn(val_f_contig)

        for x, x_f_contig in zip(res, res_f_contig, strict=True):
            np.testing.assert_allclose(x, x_f_contig)

        # Should always be destroyable
        assert (A_val == val_f_contig).all() == (not overwrite_a)

        # Test C-contiguous input
        val_c_contig = np.copy(A_val, order="C")
        res_c_contig = fn(val_c_contig)
        for x, x_c_contig in zip(res, res_c_contig, strict=True):
            np.testing.assert_allclose(x, x_c_contig)

        # Cannot destroy C-contiguous input
        np.testing.assert_allclose(val_c_contig, A_val)

        # Test non-contiguous input
        val_not_contig = np.repeat(A_val, 2, axis=0)[::2]
        res_not_contig = fn(val_not_contig)
        for x, x_not_contig in zip(res, res_not_contig, strict=True):
            np.testing.assert_allclose(x, x_not_contig)

        # Cannot destroy non-contiguous input
        np.testing.assert_allclose(val_not_contig, A_val)

    @pytest.mark.parametrize(
        "mode, pivoting",
        [("economic", False), ("full", True), ("r", False), ("raw", True)],
        ids=["economic", "full_pivot", "r", "raw_pivot"],
    )
    @pytest.mark.parametrize(
        "overwrite_a", [False, True], ids=["overwrite_a", "no_overwrite"]
    )
    @pytest.mark.parametrize("complex", (False, True))
    def test_qr(self, mode, pivoting, overwrite_a, complex):
        shape = (5, 5)
        rng = np.random.default_rng()
        A = pt.tensor(
            "A",
            shape=shape,
            dtype="complex128" if complex else "float64",
        )
        if complex:
            A_val = rng.normal(size=(*shape, 2)).view(dtype=A.dtype).squeeze(-1)
        else:
            A_val = rng.normal(size=shape).astype(A.dtype)

        qr_outputs = pt.linalg.qr(A, mode=mode, pivoting=pivoting)

        fn, res = compare_numba_and_py(
            [In(A, mutable=overwrite_a)],
            qr_outputs,
            [A_val],
            numba_mode=numba_inplace_mode,
            inplace=True,
        )

        op = fn.maker.fgraph.outputs[0].owner.op
        assert isinstance(op, QR)

        destroy_map = op.destroy_map

        if overwrite_a:
            assert destroy_map == {0: [0]}
        else:
            assert destroy_map == {}

        # Test F-contiguous input
        val_f_contig = np.copy(A_val, order="F")
        res_f_contig = fn(val_f_contig)

        for x, x_f_contig in zip(res, res_f_contig, strict=True):
            np.testing.assert_allclose(x, x_f_contig)

        # Should always be destroyable
        assert (A_val == val_f_contig).all() == (not overwrite_a)

        # Test C-contiguous input
        val_c_contig = np.copy(A_val, order="C")
        res_c_contig = fn(val_c_contig)
        for x, x_c_contig in zip(res, res_c_contig, strict=True):
            np.testing.assert_allclose(x, x_c_contig)

        # Cannot destroy C-contiguous input
        np.testing.assert_allclose(val_c_contig, A_val)

        # Test non-contiguous input
        val_not_contig = np.repeat(A_val, 2, axis=0)[::2]
        res_not_contig = fn(val_not_contig)
        for x, x_not_contig in zip(res, res_not_contig, strict=True):
            np.testing.assert_allclose(x, x_not_contig)

        # Cannot destroy non-contiguous input
        np.testing.assert_allclose(val_not_contig, A_val)

    @pytest.mark.parametrize(
        "decomp_op", (cholesky, lu, lu_factor), ids=lambda x: x.__name__
    )
    def test_empty(self, decomp_op):
        x = pt.matrix("x")
        outs = decomp_op(x)
        if not isinstance(outs, tuple | list):
            outs = [outs]
        compare_numba_and_py(
            [x],
            outs,
            [np.zeros((0, 0))],
        )

    @pytest.mark.parametrize("output", ["real", "complex"], ids=lambda x: f"output_{x}")
    @pytest.mark.parametrize(
        "input_type", ["real", "complex"], ids=lambda x: f"input_{x}"
    )
    @pytest.mark.parametrize(
        "overwrite_a", [False, True], ids=["no_overwrite", "overwrite_a"]
    )
    def test_schur(self, output, input_type, overwrite_a):
        shape = (5, 5)
        # Scipy only respects output parameter for real inputs
        # Complex inputs always produce complex output
        requires_casting = input_type == "real" and output == "complex"

        dtype = (
            config.floatX
            if input_type == "real"
            else ("complex64" if config.floatX.endswith("32") else "complex128")
        )
        A = pt.tensor("A", shape=shape, dtype=dtype)
        T, Z = schur(A, output=output)

        rng = np.random.default_rng()
        A_val = rng.normal(size=shape).astype(dtype)

        fn, (T_res, Z_res) = compare_numba_and_py(
            [In(A, mutable=overwrite_a)],
            [T, Z],
            [A_val],
            numba_mode=numba_inplace_mode,
            inplace=True,
        )

        expected_complex_output = input_type == "complex" or output == "complex"
        assert (
            np.iscomplexobj(T_res) and np.iscomplexobj(Z_res)
        ) == expected_complex_output

        # Verify reconstruction
        A_rebuilt = Z_res @ T_res @ Z_res.conj().T
        np.testing.assert_allclose(A_val, A_rebuilt, atol=1e-6, rtol=1e-6)

        # Test F-contiguous input
        val_f_contig = np.copy(A_val, order="F")
        T_f, Z_f = fn(val_f_contig)
        np.testing.assert_allclose(T_f, T_res, atol=1e-6)
        np.testing.assert_allclose(Z_f, Z_res, atol=1e-6)

        expect_destroy = overwrite_a and not requires_casting
        assert (A_val == val_f_contig).all() == (not expect_destroy)

        # Test C-contiguous input (cannot destroy)
        val_c_contig = np.copy(A_val, order="C")
        T_c, Z_c = fn(val_c_contig)
        np.testing.assert_allclose(T_c, T_res, atol=1e-6)
        np.testing.assert_allclose(Z_c, Z_res, atol=1e-6)
        np.testing.assert_allclose(val_c_contig, A_val)


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


def test_block_diag_with_read_only_inp():
    # Regression test where numba would complain a about *args containing both read-only and regular inputs
    # Currently, constants are read-only for numba, but for future-proofing we add an explicitly read-only input as well
    x = pt.tensor("x", shape=(2, 2))
    x_read_only = pt.tensor("x", shape=(2, 2))
    x_const = pt.constant(np.ones((2, 2), dtype=x.type.dtype), name="x_read_only")
    out = pt.linalg.block_diag(x, x_read_only, x_const)

    x_test = np.ones((2, 2), dtype=x.type.dtype)
    x_read_only_test = x_test.copy()
    x_read_only_test.flags.writeable = False
    compare_numba_and_py(
        [x, x_read_only],
        [out],
        [x_test, x_read_only_test],
    )


@pytest.mark.parametrize("inverse", [True, False], ids=["p_inv", "p"])
def test_pivot_to_permutation(inverse):
    from pytensor.tensor.slinalg import pivot_to_permutation

    rng = np.random.default_rng(123)
    A = rng.normal(size=(5, 5)).astype(floatX)

    perm_pt = pt.vector("p", dtype="int32")
    piv_pt = pivot_to_permutation(perm_pt, inverse=inverse)
    f = pytensor.function([perm_pt], piv_pt, mode="NUMBA")

    _, piv = scipy.linalg.lu_factor(A)

    if inverse:
        p = np.arange(len(piv))
        for i in range(len(piv)):
            p[i], p[piv[i]] = p[piv[i]], p[i]
        np.testing.assert_allclose(f(piv), p)
    else:
        p, *_ = scipy.linalg.lu(A, p_indices=True)
        np.testing.assert_allclose(f(piv), p)
