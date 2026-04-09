import numpy as np
import scipy

import pytensor.tensor as pt
from pytensor import function
from pytensor.tensor.linalg import Eig, eig, eigh, eigvalsh
from pytensor.tensor.type import dmatrix, matrix
from tests import unittest_tools as utt


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


class TestEig(utt.InferShapeTester):
    op_class = Eig
    dtype = "float64"
    op = staticmethod(eig)

    def setup_method(self):
        super().setup_method()

        self.rng = np.random.default_rng(utt.fetch_seed())
        self.A = matrix(dtype=self.dtype)
        self.X = np.asarray(self.rng.random((5, 5)), dtype=self.dtype)
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

    def test_eval(self):
        A = matrix(dtype=self.dtype)
        fn = function([A], self.op(A))

        # Symmetric input (real eigenvalues)
        A_val = self.rng.normal(size=(5, 5)).astype(self.dtype)
        A_val = A_val + A_val.T

        w, v = fn(A_val)
        w_np, v_np = np.linalg.eigh(A_val)
        rtol = 1e-2 if self.dtype == "float32" else 1e-7
        np.testing.assert_allclose(np.dot(A_val, v), w * v, rtol=rtol)

        np.testing.assert_allclose(np.abs(w), np.abs(w_np), rtol=rtol)
        np.testing.assert_allclose(np.abs(v), np.abs(v_np), rtol=rtol)

    def test_uplo(self):
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
