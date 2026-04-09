import functools

import numpy as np
import pytest
from scipy import linalg as scipy_linalg

from pytensor import In, function
from pytensor.compile import get_default_mode
from pytensor.configdefaults import config
from pytensor.tensor.linalg import qz, schur
from pytensor.tensor.type import matrix, tensor
from tests import unittest_tools as utt


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


class TestQZ:
    @pytest.mark.parametrize(
        "shape, output",
        [((5, 5), "real"), ((5, 5), "complex"), ((2, 4, 4), "real")],
        ids=["not_batched_real", "not_batched_complex", "batched_real"],
    )
    @pytest.mark.parametrize("complex", [False, True], ids=["real", "complex"])
    @pytest.mark.parametrize("sort", [None, "lhp", "rhp", "iuc", "ouc"])
    def test_qz_decomposition(self, shape, output, complex, sort):
        dtype = (
            config.floatX if not complex else f"complex{int(config.floatX[-2:]) * 2}"
        )

        A = tensor("A", shape=shape, dtype=dtype)
        B = tensor("B", shape=shape, dtype=dtype)
        outputs = qz(
            A, B, output=output, sort=sort, return_eigenvalues=sort is not None
        )

        f = function([A, B], outputs)

        rng = np.random.default_rng(utt.fetch_seed())
        A_val, B_val = rng.normal(size=(2, *shape))
        A_val = A_val.astype(config.floatX)
        B_val = B_val.astype(config.floatX)

        if complex:
            A_val = A_val + 1j * rng.normal(size=shape).astype(config.floatX)
            B_val = B_val + 1j * rng.normal(size=shape).astype(config.floatX)

        output_values = f(A_val, B_val)
        if sort is None:
            AA_val, BB_val, Q_val, Z_val = output_values
        else:
            AA_val, BB_val, alpha_val, beta_val, Q_val, Z_val = output_values

        A_rebuilt = np.einsum("...ij,...jk,...lk->...il", Q_val, AA_val, Z_val.conj())
        B_rebuilt = np.einsum("...ij,...jk,...lk->...il", Q_val, BB_val, Z_val.conj())

        np.testing.assert_allclose(
            A_val,
            A_rebuilt,
            atol=1e-6 if config.floatX == "float64" else 1e-3,
            rtol=1e-6 if config.floatX == "float64" else 1e-3,
        )

        np.testing.assert_allclose(
            B_val,
            B_rebuilt,
            atol=1e-6 if config.floatX == "float64" else 1e-3,
            rtol=1e-6 if config.floatX == "float64" else 1e-3,
        )

        scipy_fn = (
            scipy_linalg.qz
            if sort is None
            else functools.partial(scipy_linalg.ordqz, sort=sort)
        )
        scipy_signature = (
            "(m,m),(m,m)->(m,m),(m,m),(m,m),(m,m)"
            if sort is None
            else ("(m,m),(m,m)->(m,m),(m,m),(m),(m),(m,m),(m,m)")
        )

        vec_qz = np.vectorize(
            lambda a, b: scipy_fn(a, b, output=output),
            signature=scipy_signature,
        )

        scipy_result = vec_qz(A_val, B_val)
        if sort is None:
            scipy_AA, scipy_BB, scipy_Q, scipy_Z = scipy_result
        else:
            scipy_AA, scipy_BB, scipy_alpha, scipy_beta, scipy_Q, scipy_Z = scipy_result

        np.testing.assert_allclose(AA_val, scipy_AA, atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(BB_val, scipy_BB, atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(Q_val, scipy_Q, atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(Z_val, scipy_Z, atol=1e-6, rtol=1e-6)

        if sort is not None:
            np.testing.assert_allclose(alpha_val, scipy_alpha, atol=1e-6, rtol=1e-6)
            np.testing.assert_allclose(beta_val, scipy_beta, atol=1e-6, rtol=1e-6)

        if len(shape) == 2 and (output == "complex") == complex:
            A_f = np.asfortranarray(A_val.copy())
            B_f = np.asfortranarray(B_val.copy())

            f_mut = function(
                [In(A, mutable=True), In(B, mutable=True)],
                outputs,
                mode=get_default_mode().including("inplace"),
            )
            f_mut(A_f, B_f)

            np.testing.assert_allclose(A_f, scipy_AA, atol=1e-6, rtol=1e-6)
            np.testing.assert_allclose(B_f, scipy_BB, atol=1e-6, rtol=1e-6)
