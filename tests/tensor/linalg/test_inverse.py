import numpy as np
import pytest

import pytensor
from pytensor import function
from pytensor.configdefaults import config
from pytensor.graph.replace import clone_replace
from pytensor.tensor.basic import arange
from pytensor.tensor.linalg import (
    MatrixInverse,
    TensorInv,
    matrix_inverse,
    pinv,
    tensorinv,
)
from pytensor.tensor.math import _allclose
from pytensor.tensor.type import matrix, tensor3, tensor4, vector
from tests import unittest_tools as utt
from tests.test_rop import break_op


def test_pseudoinverse_correctness():
    rng = np.random.default_rng(utt.fetch_seed())
    d1 = rng.integers(4) + 2
    d2 = rng.integers(4) + 2
    r = rng.standard_normal((d1, d2)).astype(config.floatX)

    x = matrix()
    xi = pinv(x)

    ri = function([x], xi)(r)
    assert ri.shape[0] == r.shape[1]
    assert ri.shape[1] == r.shape[0]
    assert ri.dtype == r.dtype
    # Note that pseudoinverse can be quite imprecise so I prefer to compare
    # the result with what np.linalg returns
    assert _allclose(ri, np.linalg.pinv(r))


def test_pseudoinverse_grad():
    rng = np.random.default_rng(utt.fetch_seed())
    d1 = rng.integers(4) + 2
    d2 = rng.integers(4) + 2
    r = rng.standard_normal((d1, d2)).astype(config.floatX)

    utt.verify_grad(pinv, [r])


def test_pseudoinverse_static_shape():
    x = matrix(shape=(3, 5))
    z = pinv(x)
    assert z.type.shape == (5, 3)

    g = pytensor.grad(z.sum(), x)
    f = function([x], g)

    rng = np.random.default_rng(utt.fetch_seed())
    r = rng.standard_normal((3, 5)).astype(config.floatX)
    assert f(r).shape == (3, 5)

    utt.verify_grad(pinv, [r])


class TestMatrixInverse(utt.InferShapeTester):
    def setup_method(self):
        super().setup_method()
        self.op_class = MatrixInverse
        self.op = matrix_inverse
        self.rng = np.random.default_rng(utt.fetch_seed())

    def test_inverse_correctness(self):
        r = self.rng.standard_normal((4, 4)).astype(config.floatX)

        x = matrix()
        xi = self.op(x)

        ri = function([x], xi)(r)
        assert ri.shape == r.shape
        assert ri.dtype == r.dtype

        rir = np.dot(ri, r)
        rri = np.dot(r, ri)

        assert _allclose(np.identity(4), rir), rir
        assert _allclose(np.identity(4), rri), rri

    def test_infer_shape(self):
        r = self.rng.standard_normal((4, 4)).astype(config.floatX)

        x = matrix()
        xi = self.op(x)

        self._compile_and_check([x], [xi], [r], self.op_class, warn=False)

    def test_pushforward_pullback(self):
        rtol = 1e-7 if config.floatX == "float64" else 1e-5
        mx = matrix("mx")
        mv = matrix("mv")
        v = vector("v")
        y = MatrixInverse()(mx).sum(axis=0)

        yv = pytensor.gradient.pushforward(y, mx, mv, use_op_pushforward=True)
        pushforward_f = function([mx, mv], yv)

        yv_via_lop = pytensor.gradient.pushforward(y, mx, mv, use_op_pushforward=False)
        pushforward_via_pullback_f = function([mx, mv], yv_via_lop)

        sy, _ = pytensor.scan(
            lambda i, y, x, v: (pytensor.gradient.grad(y[i], x) * v).sum(),
            sequences=arange(y.shape[0]),
            non_sequences=[y, mx, mv],
        )
        scan_f = function([mx, mv], sy)

        rng = np.random.default_rng(utt.fetch_seed())
        vx = np.asarray(rng.standard_normal((4, 4)), pytensor.config.floatX)
        vv = np.asarray(rng.standard_normal((4, 4)), pytensor.config.floatX)

        v_ref = scan_f(vx, vv)
        np.testing.assert_allclose(pushforward_f(vx, vv), v_ref, rtol=rtol)
        np.testing.assert_allclose(pushforward_via_pullback_f(vx, vv), v_ref, rtol=rtol)

        with pytest.raises(ValueError):
            pytensor.gradient.pushforward(
                clone_replace(y, replace={mx: break_op(mx)}),
                mx,
                mv,
                use_op_pushforward=True,
            )

        vv = np.asarray(rng.uniform(size=(4,)), pytensor.config.floatX)
        yv = pytensor.gradient.pullback(y, mx, v)
        pullback_f = function([mx, v], yv)

        sy = pytensor.gradient.grad((v * y).sum(), mx)
        scan_f = function([mx, v], sy)

        v_ref = scan_f(vx, vv)
        v = pullback_f(vx, vv)
        np.testing.assert_allclose(v, v_ref, rtol=rtol)


def test_inverse_singular():
    singular = np.array([[1, 0, 0]] + [[0, 1, 0]] * 2, dtype=config.floatX)
    a = matrix()
    f = function([a], matrix_inverse(a))
    with pytest.raises(np.linalg.LinAlgError):
        f(singular)


def test_inverse_grad():
    rng = np.random.default_rng(utt.fetch_seed())
    r = rng.standard_normal((4, 4))
    utt.verify_grad(matrix_inverse, [r], rng=np.random)

    rng = np.random.default_rng(utt.fetch_seed())

    r = rng.standard_normal((4, 4))
    utt.verify_grad(matrix_inverse, [r], rng=np.random)


class TestTensorInv(utt.InferShapeTester):
    def setup_method(self):
        super().setup_method()
        self.A = tensor4("A", dtype=config.floatX)
        self.B = tensor3("B", dtype=config.floatX)
        self.a = np.random.random((4, 6, 8, 3)).astype(config.floatX)
        self.b = np.random.random((2, 15, 30)).astype(config.floatX)
        self.b1 = np.random.random((30, 2, 15)).astype(
            config.floatX
        )  # for ind=1 since we need prod(b1.shape[:ind]) == prod(b1.shape[ind:])

    def test_infer_shape(self):
        A = self.A
        Ai = tensorinv(A)
        self._compile_and_check(
            [A],  # pytensor.function inputs
            [Ai],  # pytensor.function outputs
            [self.a],  # value to substitute
            TensorInv,
        )

    def test_eval(self):
        A = self.A
        Ai = tensorinv(A)
        n_ainv = np.linalg.tensorinv(self.a)
        tf_a = function([A], [Ai])
        t_ainv = tf_a(self.a)
        assert _allclose(n_ainv, t_ainv)

        B = self.B
        Bi = tensorinv(B)
        Bi1 = tensorinv(B, ind=1)
        n_binv = np.linalg.tensorinv(self.b)
        n_binv1 = np.linalg.tensorinv(self.b1, ind=1)
        tf_b = function([B], [Bi])
        tf_b1 = function([B], [Bi1])
        t_binv = tf_b(self.b)
        t_binv1 = tf_b1(self.b1)
        assert _allclose(t_binv, n_binv)
        assert _allclose(t_binv1, n_binv1)
