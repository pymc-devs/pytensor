import numpy as np
import pytest
import scipy

import pytensor.tensor as pt
from pytensor import function
from pytensor.tensor.linalg import Eig, Eigh, Eigvalsh, eig, eigh, eigvalsh
from pytensor.tensor.type import matrix
from tests import unittest_tools as utt


def _random_hermitian_pd(rng, n, shift):
    M = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    return M @ M.conj().T + shift * np.eye(n)


def _complex_numeric_grad(f, x, eps=1e-6):
    """Central-difference gradient dL/dRe + i dL/dIm of a real loss of complex x."""
    grad = np.zeros_like(x)
    for idx in np.ndindex(x.shape):
        step = np.zeros_like(x)
        step[idx] = eps
        d_real = (f(x + step) - f(x - step)) / (2 * eps)
        d_imag = (f(x + 1j * step) - f(x - 1j * step)) / (2 * eps)
        grad[idx] = d_real + 1j * d_imag
    return grad


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
        w_sym, v_sym = self.op(A)
        assert w_sym.type.dtype == np.finfo(dtype).dtype
        assert v_sym.type.dtype == dtype
        fn = function([A], [w_sym, v_sym])

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

    @pytest.mark.parametrize(
        "generalized", [False, True], ids=["standard", "generalized"]
    )
    @pytest.mark.parametrize("lower", [True, False], ids=["lower", "upper"])
    def test_pullback_complex(self, lower, generalized):
        # pytensor.grad rejects complex gradients, so the pullback is called directly
        rng = np.random.default_rng(utt.fetch_seed())
        vals = [_random_hermitian_pd(rng, 4, shift) for shift in (1, 5)]
        vals = vals[: 1 + generalized]
        inputs = [pt.zmatrix(f"x{i}") for i in range(len(vals))]
        op = Eigh(lower=lower)
        w, v = op(*inputs)
        fwd = function(inputs, [w, v])

        w_weights = rng.normal(size=4)
        v_weights = rng.normal(size=(4, 4))

        def loss(*args):
            w_val, v_val = fwd(*args)
            return w_val @ w_weights + (np.abs(v_val) ** 2 * v_weights).sum()

        gw, gv = pt.dvector("gw"), pt.zmatrix("gv")
        pullback = function([*inputs, gw, gv], op.pullback(inputs, [w, v], [gw, gv]))
        # Cotangents of loss, which does not depend on the phase of each eigenvector
        grads = pullback(*vals, w_weights, 2 * v_weights * fwd(*vals)[1])

        for k, grad in enumerate(grads):
            expected = _complex_numeric_grad(
                lambda x, k=k: loss(*vals[:k], x, *vals[k + 1 :]), vals[k]
            )
            np.testing.assert_allclose(grad, expected, atol=1e-7)


class TestEigvalsh(TestEig):
    op = staticmethod(eigvalsh)
    op_class = Eigvalsh

    def test_infer_shape(self):
        A = self.A
        S = self.S
        self._compile_and_check(
            [A],
            [self.op(A)],
            [S],
            self.op_class,
            warn=False,
        )

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
            A_val = A_val + A_val.T.conj()
        else:
            A_val = A_val + A_val.T

        w = fn(A_val)
        w_np = scipy.linalg.eigvalsh(A_val)
        rtol = 1e-2 if np.finfo(dtype).bits <= 32 else 1e-7
        np.testing.assert_allclose(np.abs(w), np.abs(w_np), rtol=rtol)

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

        B_val = self.rng.normal(size=(5, 5)).astype(dtype)
        if is_complex:
            B_val = B_val + 1j * self.rng.normal(size=(5, 5)).astype(dtype)
        B_val = B_val @ B_val.T.conj() + 50 * np.eye(5, dtype=dtype)

        w = fn(A_val, B_val)
        w_np = scipy.linalg.eigvalsh(A_val, B_val)
        rtol = 5e-2 if np.finfo(dtype).bits <= 32 else 1e-7
        np.testing.assert_allclose(np.abs(w), np.abs(w_np), rtol=rtol)

    @pytest.mark.parametrize("lower", [True, False], ids=["lower", "upper"])
    def test_grad_basic(self, lower):
        X = self.X
        utt.verify_grad(lambda x: self.op(x.dot(x.T), lower=lower), [X], rng=self.rng)

    @pytest.mark.parametrize("lower", [True, False], ids=["lower", "upper"])
    def test_grad_generalized(self, lower):
        rng = self.rng

        def make_spd(x):
            return x @ x.T + 50 * pt.eye(x.shape[0])

        a_val = rng.normal(size=(5, 5)).astype(self.dtype)
        b_val = rng.normal(size=(5, 5)).astype(self.dtype)

        # Gradients w.r.t. A
        utt.verify_grad(
            lambda a, b: eigh(a + a.T, make_spd(b), lower=lower)[0],
            [a_val, b_val],
            rng=rng,
        )

        # Gradients w.r.t B
        utt.verify_grad(
            lambda a, b: eigh(a + a.T, make_spd(b), lower=lower)[1],
            [a_val, b_val],
            rng=rng,
        )

    @pytest.mark.parametrize(
        "generalized", [False, True], ids=["standard", "generalized"]
    )
    @pytest.mark.parametrize("lower", [True, False], ids=["lower", "upper"])
    def test_pullback_complex(self, lower, generalized):
        # pytensor.grad rejects complex gradients, so the pullback is called directly
        rng = np.random.default_rng(utt.fetch_seed())
        vals = [_random_hermitian_pd(rng, 4, shift) for shift in (1, 5)]
        vals = vals[: 1 + generalized]
        inputs = [pt.zmatrix(f"x{i}") for i in range(len(vals))]
        op = Eigvalsh(lower=lower)
        w = op(*inputs)
        fwd = function(inputs, w)
        w_weights = rng.normal(size=4)

        gw = pt.dvector("gw")
        pullback = function([*inputs, gw], op.pullback(inputs, [w], [gw]))
        grads = pullback(*vals, w_weights)

        for k, grad in enumerate(grads):
            expected = _complex_numeric_grad(
                lambda x, k=k: fwd(*vals[:k], x, *vals[k + 1 :]) @ w_weights, vals[k]
            )
            np.testing.assert_allclose(grad, expected, atol=1e-7)
