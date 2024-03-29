import numpy as np
import pytest
from scipy.special import beta as scipy_beta
from scipy.special import factorial as scipy_factorial
from scipy.special import log_softmax as scipy_log_softmax
from scipy.special import logit as scipy_logit
from scipy.special import poch as scipy_poch
from scipy.special import softmax as scipy_softmax

from pytensor.compile.function import function
from pytensor.configdefaults import config
from pytensor.graph.replace import vectorize_node
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.special import (
    LogSoftmax,
    Softmax,
    SoftmaxGrad,
    beta,
    betaln,
    factorial,
    log_softmax,
    logit,
    poch,
    softmax,
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

    def test_valid_axis(self):
        with pytest.raises(TypeError):
            Softmax(1.5)

        x = [tensor3()] * LogSoftmax.nin
        Softmax(2)(*x)
        Softmax(-3)(*x)

        with pytest.raises(ValueError):
            Softmax(3)(*x)

        with pytest.raises(ValueError):
            Softmax(-4)(*x)


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

    def test_valid_axis(self):
        with pytest.raises(TypeError):
            LogSoftmax(1.5)

        x = [tensor3()] * LogSoftmax.nin
        LogSoftmax(2)(*x)
        LogSoftmax(-3)(*x)

        with pytest.raises(ValueError):
            LogSoftmax(3)(*x)

        with pytest.raises(ValueError):
            LogSoftmax(-4)(*x)


class TestSoftmaxGrad(utt.InferShapeTester):
    def test_infer_shape(self):
        admat = matrix()
        bdmat = matrix()
        rng = np.random.default_rng(utt.fetch_seed())
        admat_val = rng.random((3, 4)).astype(config.floatX)
        bdmat_val = rng.random((3, 4)).astype(config.floatX)
        self._compile_and_check(
            [admat, bdmat],
            [SoftmaxGrad(axis=-1)(admat, bdmat)],
            [admat_val, bdmat_val],
            SoftmaxGrad,
        )

    def test_valid_axis(self):
        with pytest.raises(TypeError):
            SoftmaxGrad(1.5)

        x = [tensor3()] * SoftmaxGrad.nin
        SoftmaxGrad(2)(*x)
        SoftmaxGrad(-3)(*x)

        with pytest.raises(ValueError):
            SoftmaxGrad(3)(*x)

        with pytest.raises(ValueError):
            SoftmaxGrad(-4)(*x)


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

    node = constructor(x, axis=core_axis).owner
    assert isinstance(node.op, op)

    new_node = vectorize_node(node, batch_x)
    if len(batch_axis) == 1:
        assert isinstance(new_node.op, op)
        assert (new_node.op.axis,) == batch_axis
    else:
        assert isinstance(new_node.op, Blockwise) and isinstance(
            new_node.op.core_op, op
        )
        assert new_node.op.core_op.axis == core_axis


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
