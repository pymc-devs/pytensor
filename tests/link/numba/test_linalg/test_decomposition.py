import contextlib

import numpy as np
import pytest
import scipy

import pytensor
import pytensor.tensor as pt
from pytensor import In, config
from pytensor.tensor._linalg.decomposition import svd
from pytensor.tensor._linalg.decomposition.cholesky import Cholesky, cholesky
from pytensor.tensor._linalg.decomposition.eigen import Eigh, eig
from pytensor.tensor._linalg.decomposition.lu import (
    LU,
    LUFactor,
    lu,
    lu_factor,
    pivot_to_permutation,
)
from pytensor.tensor._linalg.decomposition.qr import QR, qr
from pytensor.tensor._linalg.decomposition.schur import qz, schur
from tests.link.numba.test_basic import compare_numba_and_py, numba_inplace_mode


pytestmark = pytest.mark.filterwarnings("error")

numba = pytest.importorskip("numba")

floatX = config.floatX

rng = np.random.default_rng(42849)


# --- From test_nlinalg.py ---


@pytest.mark.parametrize("input_dtype", ["int64", "float64", "complex128"])
@pytest.mark.parametrize("symmetric", [True, False], ids=["symmetric", "general"])
def test_Eig(input_dtype, symmetric):
    x = pt.matrix("x", dtype=input_dtype)
    if x.type.numpy_dtype.kind in "fc":
        x_val = rng.normal(size=(3, 3)).astype(input_dtype)
    else:
        x_val = rng.integers(1, 10, size=(3, 3)).astype("int64")

    if symmetric:
        x_val = x_val + x_val.T

    def assert_fn(x, y):
        # eig can return equivalent values with some sign flips depending on impl, allow for that
        np.testing.assert_allclose(np.abs(x), np.abs(y), strict=True)

    g = eig(x)
    _, [eigen_values, eigen_vectors] = compare_numba_and_py(
        graph_inputs=[x],
        graph_outputs=g,
        test_inputs=[x_val],
        assert_fn=assert_fn,
    )
    # Check eig is correct
    np.testing.assert_allclose(
        x_val @ eigen_vectors,
        eigen_vectors @ np.diag(eigen_values),
        atol=1e-7,
        rtol=1e-5,
    )


@pytest.mark.parametrize(
    "x, uplo, exc",
    [
        (
            (
                pt.dmatrix(),
                (lambda x: x.T.dot(x))(rng.random(size=(3, 3)).astype("float64")),
            ),
            "L",
            None,
        ),
        (
            (
                pt.lmatrix(),
                (lambda x: x.T.dot(x))(
                    rng.integers(1, 10, size=(3, 3)).astype("int64")
                ),
            ),
            "U",
            UserWarning,
        ),
    ],
)
def test_Eigh(x, uplo, exc):
    x, test_x = x
    g = Eigh(uplo)(x)

    cm = contextlib.suppress() if exc is None else pytest.warns(exc)
    with cm:
        compare_numba_and_py([x], g, [test_x])


@pytest.mark.parametrize(
    "x, full_matrices, compute_uv, exc",
    [
        (
            (
                pt.dmatrix(),
                (lambda x: x.T.dot(x))(rng.random(size=(3, 3)).astype("float64")),
            ),
            True,
            True,
            None,
        ),
        (
            (
                pt.dmatrix(),
                (lambda x: x.T.dot(x))(rng.random(size=(3, 3)).astype("float64")),
            ),
            False,
            True,
            None,
        ),
        (
            (
                pt.lmatrix(),
                (lambda x: x.T.dot(x))(
                    rng.integers(1, 10, size=(3, 3)).astype("int64")
                ),
            ),
            True,
            True,
            None,
        ),
        (
            (
                pt.lmatrix(),
                (lambda x: x.T.dot(x))(
                    rng.integers(1, 10, size=(3, 3)).astype("int64")
                ),
            ),
            True,
            False,
            None,
        ),
    ],
)
def test_SVD(x, full_matrices, compute_uv, exc):
    x, test_x = x
    g = svd.SVD(full_matrices, compute_uv)(x)

    cm = contextlib.suppress() if exc is None else pytest.warns(exc)
    with cm:
        compare_numba_and_py([x], g, [test_x])


# --- From test_slinalg.py TestDecompositions ---


class TestDecompositions:
    @pytest.mark.parametrize("lower", [True, False], ids=lambda x: f"lower={x}")
    @pytest.mark.parametrize(
        "overwrite_a", [False, True], ids=["no_overwrite", "overwrite_a"]
    )
    @pytest.mark.parametrize("is_complex", [True, False], ids=["complex", "real"])
    def test_cholesky(self, lower: bool, overwrite_a: bool, is_complex: bool):
        complex_dtype = "complex64" if floatX.endswith("32") else "complex128"
        dtype = complex_dtype if is_complex else floatX

        cov = pt.matrix("cov", dtype=dtype)
        chol = cholesky(cov, lower=lower)

        rng = np.random.default_rng(42)
        x = rng.normal(size=(3, 3))
        if is_complex:
            x = x + 1j * rng.normal(size=(3, 3))
        x = x.astype(dtype)
        val = np.eye(3, dtype=dtype) + x @ x.conj().T

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

        val_f_contig = np.copy(val, order="F")
        res_f_contig = fn(val_f_contig)
        np.testing.assert_allclose(res_f_contig, res)
        assert (val == val_f_contig).all() == (not overwrite_a)

        val_c_contig = np.copy(val, order="C")
        res_c_contig = fn(val_c_contig)
        np.testing.assert_allclose(res_c_contig, res)
        assert (val == val_c_contig).all() == (not overwrite_a)

        val_not_contig = np.repeat(val, 2, axis=0)[::2]
        res_not_contig = fn(val_not_contig)
        np.testing.assert_allclose(res_not_contig, res)
        np.testing.assert_allclose(val_not_contig, val)

    def test_cholesky_raises_on_nan_input(self):
        test_value = rng.random(size=(3, 3)).astype(floatX)
        test_value[0, 0] = np.nan

        x = pt.tensor(dtype=floatX, shape=(3, 3))
        x = x.T.dot(x)
        with pytest.warns(FutureWarning):
            g = cholesky(x, check_finite=True, on_error="raise")
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
                g = cholesky(x, on_error=on_error)
        else:
            g = cholesky(x, on_error=on_error)
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
        A = pt.tensor("A", shape=shape, dtype=config.floatX)
        A_val = rng.normal(size=shape).astype(config.floatX)

        lu_outputs = lu(A, permute_l=permute_l, p_indices=p_indices)

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

        val_f_contig = np.copy(A_val, order="F")
        res_f_contig = fn(val_f_contig)
        for x, x_f_contig in zip(res, res_f_contig, strict=True):
            np.testing.assert_allclose(x, x_f_contig)
        assert (A_val == val_f_contig).all() == (not overwrite_a)

        val_c_contig = np.copy(A_val, order="C")
        res_c_contig = fn(val_c_contig)
        for x, x_c_contig in zip(res, res_c_contig, strict=True):
            np.testing.assert_allclose(x, x_c_contig)
        np.testing.assert_allclose(val_c_contig, A_val)

        val_not_contig = np.repeat(A_val, 2, axis=0)[::2]
        res_not_contig = fn(val_not_contig)
        for x, x_not_contig in zip(res, res_not_contig, strict=True):
            np.testing.assert_allclose(x, x_not_contig)
        np.testing.assert_allclose(val_not_contig, A_val)

    @pytest.mark.parametrize(
        "overwrite_a", [True, False], ids=["overwrite_a", "no_overwrite"]
    )
    def test_lu_factor(self, overwrite_a):
        shape = (5, 5)
        rng = np.random.default_rng()

        A = pt.tensor("A", shape=shape, dtype=config.floatX)
        A_val = rng.normal(size=shape).astype(config.floatX)

        LU_out, piv = lu_factor(A)

        fn, res = compare_numba_and_py(
            [In(A, mutable=overwrite_a)],
            [LU_out, piv],
            [A_val],
            numba_mode=numba_inplace_mode,
            inplace=True,
        )

        op = fn.maker.fgraph.outputs[0].owner.op
        assert isinstance(op, LUFactor)

        if overwrite_a:
            assert op.destroy_map == {1: [0]}

        val_f_contig = np.copy(A_val, order="F")
        res_f_contig = fn(val_f_contig)
        for x, x_f_contig in zip(res, res_f_contig, strict=True):
            np.testing.assert_allclose(x, x_f_contig)
        assert (A_val == val_f_contig).all() == (not overwrite_a)

        val_c_contig = np.copy(A_val, order="C")
        res_c_contig = fn(val_c_contig)
        for x, x_c_contig in zip(res, res_c_contig, strict=True):
            np.testing.assert_allclose(x, x_c_contig)
        np.testing.assert_allclose(val_c_contig, A_val)

        val_not_contig = np.repeat(A_val, 2, axis=0)[::2]
        res_not_contig = fn(val_not_contig)
        for x, x_not_contig in zip(res, res_not_contig, strict=True):
            np.testing.assert_allclose(x, x_not_contig)
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

        qr_outputs = qr(A, mode=mode, pivoting=pivoting)

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

        val_f_contig = np.copy(A_val, order="F")
        res_f_contig = fn(val_f_contig)
        for x, x_f_contig in zip(res, res_f_contig, strict=True):
            np.testing.assert_allclose(x, x_f_contig)
        assert (A_val == val_f_contig).all() == (not overwrite_a)

        val_c_contig = np.copy(A_val, order="C")
        res_c_contig = fn(val_c_contig)
        for x, x_c_contig in zip(res, res_c_contig, strict=True):
            np.testing.assert_allclose(x, x_c_contig)
        np.testing.assert_allclose(val_c_contig, A_val)

        val_not_contig = np.repeat(A_val, 2, axis=0)[::2]
        res_not_contig = fn(val_not_contig)
        for x, x_not_contig in zip(res, res_not_contig, strict=True):
            np.testing.assert_allclose(x, x_not_contig)
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

        A_rebuilt = Z_res @ T_res @ Z_res.conj().T
        np.testing.assert_allclose(A_val, A_rebuilt, atol=1e-6, rtol=1e-6)

        val_f_contig = np.copy(A_val, order="F")
        T_f, Z_f = fn(val_f_contig)
        np.testing.assert_allclose(T_f, T_res, atol=1e-6)
        np.testing.assert_allclose(Z_f, Z_res, atol=1e-6)

        expect_destroy = overwrite_a and not requires_casting
        assert (A_val == val_f_contig).all() == (not expect_destroy)

        val_c_contig = np.copy(A_val, order="C")
        T_c, Z_c = fn(val_c_contig)
        np.testing.assert_allclose(T_c, T_res, atol=1e-6)
        np.testing.assert_allclose(Z_c, Z_res, atol=1e-6)
        np.testing.assert_allclose(val_c_contig, A_val)

    @pytest.mark.parametrize(
        "output, input_type, sort, return_eigenvalues",
        [
            ("real", "real", None, False),
            ("complex", "real", "lhp", True),
            ("real", "complex", "ouc", False),
            ("complex", "complex", None, True),
            ("real", "real", "iuc", True),
        ],
        ids=[
            "real_nosort",
            "real_to_complex_sort",
            "complex_sort",
            "complex_nosort_eig",
            "real_sort_eig",
        ],
    )
    def test_qz(self, output, input_type, sort, return_eigenvalues):
        shape = (5, 5)

        dtype = (
            config.floatX
            if input_type == "real"
            else ("complex64" if config.floatX.endswith("32") else "complex128")
        )
        A = pt.tensor("A", shape=shape, dtype=dtype)
        B = pt.tensor("B", shape=shape, dtype=dtype)
        outputs = qz(
            A, B, output=output, sort=sort, return_eigenvalues=return_eigenvalues
        )

        if return_eigenvalues:
            AA, BB, alpha, beta, Q, Z = outputs
            output_list = [AA, BB, alpha, beta, Q, Z]
        else:
            AA, BB, Q, Z = outputs
            output_list = [AA, BB, Q, Z]

        rng = np.random.default_rng()
        A_val = rng.normal(size=shape).astype(dtype)
        B_val = rng.normal(size=shape).astype(dtype)

        fn, res = compare_numba_and_py(
            [A, B],
            output_list,
            [A_val, B_val],
            numba_mode=numba_inplace_mode,
            inplace=True,
        )

        if return_eigenvalues:
            AA_res, BB_res, alpha_res, beta_res, Q_res, Z_res = res
        else:
            AA_res, BB_res, Q_res, Z_res = res

        expected_complex_output = input_type == "complex" or output == "complex"
        assert np.iscomplexobj(AA_res) == expected_complex_output
        assert np.iscomplexobj(BB_res) == expected_complex_output
        assert np.iscomplexobj(Q_res) == expected_complex_output
        assert np.iscomplexobj(Z_res) == expected_complex_output

        A_rebuilt = Q_res @ AA_res @ Z_res.conj().T
        B_rebuilt = Q_res @ BB_res @ Z_res.conj().T
        np.testing.assert_allclose(A_val, A_rebuilt, atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(B_val, B_rebuilt, atol=1e-5, rtol=1e-5)

        A_val_f_contig = np.copy(A_val, order="F")
        B_val_f_contig = np.copy(B_val, order="F")
        res_f = fn(A_val_f_contig, B_val_f_contig)
        if return_eigenvalues:
            AA_f, BB_f, alpha_f, beta_f, Q_f, Z_f = res_f
            np.testing.assert_allclose(alpha_f, alpha_res, atol=1e-6)
            np.testing.assert_allclose(beta_f, beta_res, atol=1e-6)
        else:
            AA_f, BB_f, Q_f, Z_f = res_f
        np.testing.assert_allclose(AA_f, AA_res, atol=1e-6)
        np.testing.assert_allclose(BB_f, BB_res, atol=1e-6)
        np.testing.assert_allclose(Q_f, Q_res, atol=1e-6)
        np.testing.assert_allclose(Z_f, Z_res, atol=1e-6)

        A_val_c_contig = np.copy(A_val, order="C")
        B_val_c_contig = np.copy(B_val, order="C")
        res_c = fn(A_val_c_contig, B_val_c_contig)
        if return_eigenvalues:
            AA_c, BB_c, alpha_c, beta_c, Q_c, Z_c = res_c
            np.testing.assert_allclose(alpha_c, alpha_res, atol=1e-6)
            np.testing.assert_allclose(beta_c, beta_res, atol=1e-6)
        else:
            AA_c, BB_c, Q_c, Z_c = res_c
        np.testing.assert_allclose(AA_c, AA_res, atol=1e-6)
        np.testing.assert_allclose(BB_c, BB_res, atol=1e-6)
        np.testing.assert_allclose(Q_c, Q_res, atol=1e-6)
        np.testing.assert_allclose(Z_c, Z_res, atol=1e-6)


@pytest.mark.parametrize("inverse", [True, False], ids=["p_inv", "p"])
def test_pivot_to_permutation(inverse):
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
