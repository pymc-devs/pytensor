import numpy as np
import pytest
from scipy.special import beta as scipy_beta
from scipy.special import factorial as scipy_factorial
from scipy.special import log_softmax as scipy_log_softmax
from scipy.special import logit as scipy_logit
from scipy.special import poch as scipy_poch
from scipy.special import softmax as scipy_softmax
from scipy.special import xlog1py as scipy_xlog1py
from scipy.special import xlogy as scipy_xlogy

from pytensor import grad
from pytensor.compile.maker import function
from pytensor.configdefaults import config
from pytensor.graph.replace import vectorize_graph
from pytensor.tensor.special import (
    LogSoftmax,
    Softmax,
    beta,
    betaln,
    factorial,
    log_softmax,
    logit,
    poch,
    softmax,
    xlog1py,
    xlogy,
)
from pytensor.tensor.type import matrix, tensor, tensor3, tensor4, vector, vectors
from tests import unittest_tools as utt
from tests.tensor.utils import random_ranged


class TestSoftmax(utt.InferShapeTester):
    @pytest.mark.parametrize("axis", [None, 0, 1, 2, 3, -1, -2])
    def test_perform(self, axis):
        x = tensor4("x")
        rng = np.random.default_rng(utt.fetch_seed())
        xv = rng.standard_normal((2, 3, 4, 5)).astype(config.floatX)

        f = function([x], softmax(x, axis=axis))
        assert np.allclose(f(xv), scipy_softmax(xv, axis=axis))

    @pytest.mark.parametrize("column", [0, 1, 2, 3])
    @pytest.mark.parametrize("axis", [None, 0, 1])
    def test_grad(self, axis, column):
        def f(a):
            return softmax(a, axis=axis)[:, column]

        rng = np.random.default_rng(utt.fetch_seed())
        utt.verify_grad(f, [rng.random((3, 4, 2))])

    def test_infer_shape(self):
        admat = matrix()
        rng = np.random.default_rng(utt.fetch_seed())
        admat_val = rng.random((3, 4)).astype(config.floatX)
        self._compile_and_check(
            [admat], [Softmax(axis=-1)(admat)], [admat_val], Softmax
        )

    def test_vector_perform(self):
        x = vector()
        f = function([x], softmax(x, axis=None))

        rng = np.random.default_rng(utt.fetch_seed())
        xv = rng.standard_normal((6,)).astype(config.floatX)
        assert np.allclose(f(xv), scipy_softmax(xv))

    def test_vector_grad(self):
        def f(a):
            return softmax(a, axis=None)

        rng = np.random.default_rng(utt.fetch_seed())
        utt.verify_grad(f, [rng.random(4)])

    def test_raw_input(self):
        out = softmax([1.0, 2.0, 3.0], axis=-1)
        assert out.type.ndim == 1
        assert out.type.dtype == "float64"

    def test_valid_axis(self):
        with pytest.raises(TypeError):
            Softmax(1.5)

        x = tensor3()
        Softmax(axis=2)(x)
        Softmax(axis=-3)(x)

        with pytest.raises(ValueError):
            Softmax(axis=3)(x)

        with pytest.raises(ValueError):
            Softmax(axis=-4)(x)


class TestLogSoftmax(utt.InferShapeTester):
    @pytest.mark.parametrize("column", [0, 1, 2, 3])
    @pytest.mark.parametrize("axis", [None, 0, 1])
    def test_matrix_grad(self, axis, column):
        def f(a):
            return log_softmax(a, axis=axis)[:, column]

        rng = np.random.default_rng(utt.fetch_seed())
        utt.verify_grad(f, [rng.random((3, 4))])

    def test_vector_perform(self):
        x = vector()
        f = function([x], log_softmax(x, axis=None))

        rng = np.random.default_rng(utt.fetch_seed())
        xv = rng.standard_normal((6,)).astype(config.floatX)
        assert np.allclose(f(xv), scipy_log_softmax(xv))

    def test_vector_grad(self):
        def f(a):
            return log_softmax(a, axis=None)

        rng = np.random.default_rng(utt.fetch_seed())
        utt.verify_grad(f, [rng.random((4,))])

    def test_raw_input(self):
        out = log_softmax([1.0, 2.0, 3.0], axis=-1)
        assert out.type.ndim == 1
        assert out.type.dtype == "float64"

    def test_valid_axis(self):
        with pytest.raises(TypeError):
            LogSoftmax(1.5)

        x = tensor3()
        LogSoftmax(axis=2)(x)
        LogSoftmax(axis=-3)(x)

        with pytest.raises(ValueError):
            LogSoftmax(axis=3)(x)

        with pytest.raises(ValueError):
            LogSoftmax(axis=-4)(x)


@pytest.mark.parametrize(
    "core_axis, batch_axis",
    [
        (None, (1, 2, 3, 4)),
        (0, (1,)),
    ],
)
@pytest.mark.parametrize(
    "op, constructor", [(Softmax, softmax), (LogSoftmax, log_softmax)]
)
def test_vectorize_softmax(op, constructor, core_axis, batch_axis):
    x = tensor(shape=(5, 5, 5, 5))
    batch_x = tensor(shape=(3, 5, 5, 5, 5))

    out = constructor(x, axis=core_axis)
    assert isinstance(out.owner.op, op)

    new_out = vectorize_graph(out, {x: batch_x})
    assert isinstance(new_out.owner.op, op)
    assert new_out.owner.op.axis == batch_axis


@pytest.mark.parametrize("mode", ["FAST_COMPILE", "FAST_RUN"])
@pytest.mark.parametrize(
    "constructor, scipy_fn",
    [(softmax, scipy_softmax), (log_softmax, scipy_log_softmax)],
)
def test_softmax_stability(constructor, scipy_fn, mode):
    """The helpers must be stable in any mode, whatever graph they end up emitting."""
    x = matrix("x")
    x_val = np.array([[800.0, 805.0]], dtype=config.floatX)

    f = function([x], constructor(x, axis=-1), mode=mode)

    np.testing.assert_allclose(f(x_val), scipy_fn(x_val, axis=-1), rtol=1e-6)


def test_poch():
    _z, _m = vectors("z", "m")
    actual_fn = function([_z, _m], poch(_z, _m))

    z = random_ranged(0, 5, (2,))
    m = random_ranged(0, 5, (2,))
    actual = actual_fn(z, m)
    expected = scipy_poch(z, m)
    np.testing.assert_allclose(
        actual, expected, rtol=1e-7 if config.floatX == "float64" else 1e-5
    )


@pytest.mark.parametrize("n", random_ranged(0, 5, (1,)))
def test_factorial(n):
    _n = vector("n")
    actual_fn = function([_n], factorial(_n))

    n = random_ranged(0, 5, (2,))
    actual = actual_fn(n)
    expected = scipy_factorial(n)
    np.testing.assert_allclose(
        actual, expected, rtol=1e-7 if config.floatX == "float64" else 1e-5
    )


def test_logit():
    x = vector("x")
    actual_fn = function([x], logit(x), allow_input_downcast=True)

    x_test = np.linspace(0, 1)
    actual = actual_fn(x_test)
    expected = scipy_logit(x_test)
    np.testing.assert_allclose(
        actual, expected, rtol=1e-7 if config.floatX == "float64" else 1e-5
    )


def test_beta():
    _a, _b = vectors("a", "b")
    actual_fn = function([_a, _b], beta(_a, _b))

    a = random_ranged(0, 5, (2,))
    b = random_ranged(0, 5, (2,))
    actual = actual_fn(a, b)
    expected = scipy_beta(a, b)
    np.testing.assert_allclose(
        actual, expected, rtol=1e-7 if config.floatX == "float64" else 1e-5
    )


def test_betaln():
    _a, _b = vectors("a", "b")
    actual_fn = function([_a, _b], betaln(_a, _b))

    a = random_ranged(0, 5, (2,))
    b = random_ranged(0, 5, (2,))
    actual = np.exp(actual_fn(a, b))
    expected = scipy_beta(a, b)
    np.testing.assert_allclose(
        actual, expected, rtol=1e-7 if config.floatX == "float64" else 1e-5
    )


def test_xlogy():
    x, y = vectors("x", "y")
    out = xlogy(x, y)

    f = function([x, y], out)
    x_test = np.array([0.0, 0.5, 1.0, 2.0])
    y_test = np.array([0.0, 0.5, 1.0, 2.0])
    np.testing.assert_allclose(f(x_test, y_test), scipy_xlogy(x_test, y_test))

    # test grad edge cases
    gx, gy = grad(out.sum(), [x, y])
    gf = function([x, y], [gx, gy])
    # x=0, y=0: grad_x = log(0) = -inf, grad_y = 0/0 = nan
    [gxv], [gyv] = gf(np.array([0.0]), np.array([0.0]))
    assert gxv == -np.inf
    assert np.isnan(gyv)
    # x=0, y=1: grad_x = log(1) = 0, grad_y = 0/1 = 0
    [gxv], [gyv] = gf(np.array([0.0]), np.array([1.0]))
    np.testing.assert_allclose(gxv, 0.0)
    np.testing.assert_allclose(gyv, 0.0)

    rng = np.random.default_rng(239)
    utt.verify_grad(xlogy, [rng.random((3, 4)), rng.random((3, 4))])


def test_xlogy_as_xlogx():
    x = vector("x")
    out = xlogy(x, x)

    f = function([x], out)
    x_test = np.array([0.0, 0.5, 1.0, 2.0])
    np.testing.assert_allclose(f(x_test), scipy_xlogy(x_test, x_test))

    # test grad edge cases
    gx = grad(out.sum(), x)
    gf = function([x], gx)
    # x=0: 1+log(0) = -inf
    np.testing.assert_equal(gf(np.array([0.0]))[0], -np.inf)
    # x=1: 1+log(1) = 1
    np.testing.assert_allclose(gf(np.array([1.0]))[0], 1.0)

    rng = np.random.default_rng(260)
    utt.verify_grad(lambda x: xlogy(x, x), [rng.random((3, 4))])


def test_xlog1py():
    x, y = vectors("x", "y")
    out = xlog1py(x, y)

    f = function([x, y], out)

    x_test = np.array([0.0, 0.5, 1.0, 2.0])
    y_test = np.array([-1.0, 0.0, 1.0, 2.0])
    np.testing.assert_allclose(f(x_test, y_test), scipy_xlog1py(x_test, y_test))

    # test grad edge cases
    gx, gy = grad(out.sum(), [x, y])
    gf = function([x, y], [gx, gy])
    # x=0, y=-1: grad_x = log1p(-1) = -inf, grad_y = 0/(1+(-1)) = nan
    [gxv], [gyv] = gf(np.array([0.0]), np.array([-1.0]))
    assert gxv == -np.inf
    assert np.isnan(gyv)
    # x=0, y=0: grad_x = log1p(0) = 0, grad_y = 0/1 = 0
    [gxv], [gyv] = gf(np.array([0.0]), np.array([0.0]))
    np.testing.assert_allclose(gxv, 0.0)
    np.testing.assert_allclose(gyv, 0.0)

    rng = np.random.default_rng(286)
    utt.verify_grad(xlog1py, [rng.random((3, 4)), rng.random((3, 4))])


def test_xlogy_no_distribute_at_boundary():
    """Regression test: ``xlogy((a - 1), y)`` must not be algebraically
    distributed into ``a*log(y) - log(y)`` when canonicalize/stabilize run.

    The distribution is mathematically valid for finite values but breaks at
    the boundary where ``log(y) = -inf``: ``a*(-inf) - (-inf) = nan``.

    This pattern shows up in the chi-squared log-pdf
    ``xlogy(nu/2 - 1, x)`` at ``x = 0`` with ``nu > 2``, where the answer
    must be ``-inf`` (so the pdf at 0 is 0).

    Achieved by keeping ``XLogY.inline = False``, which hides the inner
    ``x * log(y)`` from `local_greedy_distributor`.
    """
    nu = tensor("nu", shape=(), dtype="int64")
    x = vector("x")
    f = function([nu, x], xlogy(nu / 2 - 1, x))
    out = f(np.int64(3), np.array([0.0, 1.0, 2.0]))
    np.testing.assert_array_equal(out, np.array([-np.inf, 0.0, 0.5 * np.log(2.0)]))


def test_xlog1py_no_distribute_at_boundary():
    """See ``test_xlogy_no_distribute_at_boundary``. Same hazard for
    ``xlog1py`` at ``y = -1`` where ``log1p(y) = -inf``.
    """
    a = tensor("a", shape=(), dtype="int64")
    y = vector("y")
    f = function([a, y], xlog1py(a / 2 - 1, y))
    out = f(np.int64(3), np.array([-1.0, 0.0, 1.0]))
    np.testing.assert_array_equal(out, np.array([-np.inf, 0.0, 0.5 * np.log(2.0)]))
