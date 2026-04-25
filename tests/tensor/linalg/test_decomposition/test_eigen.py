import numpy as np
import pytest
import scipy

import pytensor.tensor as pt
from pytensor import function
from pytensor.tensor.linalg import Eig, Eigh, Eigvalsh, eig, eigh, eigvalsh
from pytensor.tensor.type import matrix
from tests import unittest_tools as utt


class TestEig(utt.InferShapeTester):
    op_class = Eig
    dtype = "float64"
    op = staticmethod(eig)

    def setup_method(self):
        super().setup_method()

        self.rng = np.random.default_rng(utt.fetch_seed())
        self.A = matrix(dtype=self.dtype)
        self.X = self.rng.normal(size=(5, 5)).astype(self.dtype)
        self.S = self.X.dot(self.X.T)

    def test_infer_shape(self):
        A = self.A
        S = self.S
        self._compile_and_check(
            [A],
            self.op(A),
            [S],
            self.op_class,
            warn=False,
        )

    def test_eval(self):
        A = matrix(dtype=self.dtype)
        fn = function([A], self.op(A))

        # Symmetric input (real eigenvalues)
        A_val = self.rng.normal(size=(5, 5)).astype(self.dtype)
        A_val = A_val + A_val.T

        w, v = fn(A_val)
        w_np, v_np = np.linalg.eig(A_val)
        np.testing.assert_allclose(np.abs(w), np.abs(w_np))
        np.testing.assert_allclose(np.abs(v), np.abs(v_np))
        np.testing.assert_array_almost_equal(np.dot(A_val, v), w * v)

        # Asymmetric input (real eigenvalues)
        z = self.rng.normal(size=(5,)) ** 2
        A_val = (np.diag(z**0.5)).dot(A_val).dot(np.diag(z ** (-0.5)))

        w, v = fn(A_val)
        w_np, v_np = np.linalg.eig(A_val)
        np.testing.assert_allclose(np.abs(w), np.abs(w_np))
        np.testing.assert_allclose(np.abs(v), np.abs(v_np))
        np.testing.assert_array_almost_equal(np.dot(A_val, v), w * v)

        # Asymmetric input (complex eigenvalues)
        A_val = self.rng.normal(size=(5, 5))
        w, v = fn(A_val)
        w_np, v_np = np.linalg.eig(A_val)
        np.testing.assert_allclose(np.abs(w), np.abs(w_np))
        np.testing.assert_allclose(np.abs(v), np.abs(v_np))
        np.testing.assert_array_almost_equal(np.dot(A_val, v), w * v)


class TestEigh(TestEig):
    op = staticmethod(eigh)
    op_class = Eigh

    @pytest.mark.parametrize("is_complex", [True, False], ids=["complex", "real"])
    def test_eval(self, is_complex):
        if is_complex:
            dtype = "complex128" if self.dtype == "float64" else "complex64"
        else:
            dtype = self.dtype

        A = matrix(dtype=dtype)
        fn = function([A], self.op(A))

        A_val = self.rng.normal(size=(5, 5)).astype(dtype)
        if is_complex:
            A_val = A_val + 1j * self.rng.normal(size=(5, 5)).astype(dtype)
            A_val = A_val + A_val.T.conj()  # Hermitian
        else:
            A_val = A_val + A_val.T

        w, v = fn(A_val)
        w_np, v_np = np.linalg.eigh(A_val)
        rtol = 1e-2 if np.finfo(dtype).bits <= 32 else 1e-7
        np.testing.assert_allclose(np.dot(A_val, v), w * v, rtol=rtol)

        np.testing.assert_allclose(np.abs(w), np.abs(w_np), rtol=rtol)
        np.testing.assert_allclose(np.abs(v), np.abs(v_np), rtol=rtol)

    @pytest.mark.parametrize("is_complex", [True, False], ids=["complex", "real"])
    def test_eval_generalized(self, is_complex):
        if is_complex:
            dtype = "complex128" if self.dtype == "float64" else "complex64"
        else:
            dtype = self.dtype

        A = matrix(dtype=dtype)
        B = matrix(dtype=dtype)
        fn = function([A, B], self.op(A, B))

        A_val = self.rng.normal(size=(5, 5)).astype(dtype)
        if is_complex:
            A_val = A_val + 1j * self.rng.normal(size=(5, 5)).astype(dtype)
            A_val = A_val + A_val.T.conj()
        else:
            A_val = A_val + A_val.T

        # Posdef input (add diagonal for better conditioning)
        B_val = self.rng.normal(size=(5, 5)).astype(dtype)
        if is_complex:
            B_val = B_val + 1j * self.rng.normal(size=(5, 5)).astype(dtype)
        B_val = B_val @ B_val.T.conj() + 50 * np.eye(5, dtype=dtype)

        w, v = fn(A_val, B_val)
        w_np, v_np = scipy.linalg.eigh(A_val, B_val)
        rtol = 5e-2 if np.finfo(dtype).bits <= 32 else 1e-7
        np.testing.assert_allclose(np.dot(A_val, v), B_val @ (w * v), rtol=rtol)

        np.testing.assert_allclose(np.abs(w), np.abs(w_np), rtol=rtol)
        np.testing.assert_allclose(np.abs(v), np.abs(v_np), rtol=rtol)

    @pytest.mark.parametrize("lower", [True, False], ids=["lower", "upper"])
    def test_grad_basic(self, lower):
        rng = self.rng

        def make_spd(x):
            return x @ x.T + 10 * pt.eye(x.shape[0])

        x_val = rng.normal(size=(5, 5)).astype(self.dtype)

        # Eigenvalue gradient
        utt.verify_grad(
            lambda x: self.op(make_spd(x), lower=lower)[0], [x_val], rng=rng
        )
        # Eigenvector gradient: use sign-invariant cost (v**2) because
        # eigenvector signs can flip under finite-difference perturbation
        utt.verify_grad(
            lambda x: self.op(make_spd(x), lower=lower)[1] ** 2, [x_val], rng=rng
        )

    @pytest.mark.parametrize("lower", [True, False], ids=["lower", "upper"])
    def test_grad_generalized(self, lower):
        rng = self.rng

        def make_spd(x):
            return x @ x.T + 50 * pt.eye(x.shape[0])

        a_val = rng.normal(size=(5, 5)).astype(self.dtype)
        b_val = rng.normal(size=(5, 5)).astype(self.dtype)

        # Gradients w.r.t. A
        # Eigenvalues
        utt.verify_grad(
            lambda a, b: eigh(a + a.T, make_spd(b), lower=lower)[0],
            [a_val, b_val],
            rng=rng,
        )
        # Eigenvectors
        utt.verify_grad(
            lambda a, b: eigh(a + a.T, make_spd(b), lower=lower)[1],
            [a_val, b_val],
            rng=rng,
        )

        # Gradients w.r.t B
        # Eigenvalues
        utt.verify_grad(
            lambda a, b: eigh(a + a.T, make_spd(b), lower=lower)[1],
            [a_val, b_val],
            rng=rng,
        )
        # Eigenvectors
        utt.verify_grad(
            lambda a, b: eigh(a + a.T, make_spd(b), lower=lower)[1],
            [a_val, b_val],
            rng=rng,
        )
        S = self.S
        a = matrix(dtype=self.dtype)
        wu, vu = (out.eval({a: S}) for out in self.op(a, "U"))
        wl, vl = (out.eval({a: S}) for out in self.op(a, "L"))
        atol = 1e-14 if np.dtype(self.dtype).itemsize == 8 else 1e-5
        rtol = 1e-14 if np.dtype(self.dtype).itemsize == 8 else 1e-3
        np.testing.assert_allclose(wu, wl, atol=atol, rtol=rtol)
        np.testing.assert_allclose(
            vu * np.sign(vu[0, :]), vl * np.sign(vl[0, :]), atol=atol, rtol=rtol
        )

    def test_grad(self):
        X = self.X
        utt.verify_grad(lambda x: self.op(x.dot(x.T))[0], [X], rng=self.rng)
        utt.verify_grad(lambda x: self.op(x.dot(x.T))[1], [X], rng=self.rng)
        utt.verify_grad(lambda x: self.op(x.dot(x.T), "U")[0], [X], rng=self.rng)
        utt.verify_grad(lambda x: self.op(x.dot(x.T), "U")[1], [X], rng=self.rng)


class TestEighFloat32(TestEigh):
    dtype = "float32"

    def test_uplo(self):
        super().test_uplo()

    def test_grad(self):
        super().test_grad()
