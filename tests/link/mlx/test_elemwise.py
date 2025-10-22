import numpy as np
import pytest
import scipy

from pytensor import config, function
from pytensor.tensor.basic import switch
from pytensor.tensor.math import (
    add,
    cos,
    eq,
    exp,
    ge,
    gt,
    int_div,
    isinf,
    le,
    log,
    lt,
    mul,
    neq,
    power,
    prod,
    sigmoid,
    sin,
    sub,
    true_div,
)
from pytensor.tensor.math import all as pt_all
from pytensor.tensor.math import any as pt_any
from pytensor.tensor.math import max as pt_max
from pytensor.tensor.math import min as pt_min
from pytensor.tensor.math import sum as pt_sum
from pytensor.tensor.special import SoftmaxGrad, log_softmax, softmax
from pytensor.tensor.type import matrix, vector, vectors
from tests.link.mlx.test_basic import compare_mlx_and_py


mx = pytest.importorskip("mlx.core")


@pytest.mark.parametrize("op", [pt_any, pt_all, pt_max, pt_min])
def test_input(op) -> None:
    x = vector("x")
    out = op(x > 0)
    x_test = mx.array([1.0, 2.0, 3.0])

    compare_mlx_and_py([x], out, [x_test])


def test_mlx_CAReduce():
    a_pt = vector("a")
    a_pt.tag.test_value = np.r_[1, 2, 3].astype(config.floatX)

    x = pt_sum(a_pt, axis=None)

    compare_mlx_and_py([a_pt], [x], [np.r_[1, 2, 3].astype(config.floatX)])

    a_pt = matrix("a")
    a_pt.tag.test_value = np.c_[[1, 2, 3], [1, 2, 3]].astype(config.floatX)

    x = pt_sum(a_pt, axis=0)

    compare_mlx_and_py([a_pt], [x], [np.c_[[1, 2, 3], [1, 2, 3]].astype(config.floatX)])

    x = pt_sum(a_pt, axis=1)

    compare_mlx_and_py([a_pt], [x], [np.c_[[1, 2, 3], [1, 2, 3]].astype(config.floatX)])

    a_pt = matrix("a")
    a_pt.tag.test_value = np.c_[[1, 2, 3], [1, 2, 3]].astype(config.floatX)

    x = prod(a_pt, axis=0)

    compare_mlx_and_py([a_pt], [x], [np.c_[[1, 2, 3], [1, 2, 3]].astype(config.floatX)])

    x = pt_all(a_pt)

    compare_mlx_and_py([a_pt], [x], [np.c_[[1, 2, 3], [1, 2, 3]].astype(config.floatX)])


@pytest.mark.parametrize("axis", [None, 0, 1])
def test_softmax(axis):
    x = matrix("x")
    x_test_value = np.arange(6, dtype=config.floatX).reshape(2, 3)
    out = softmax(x, axis=axis)
    compare_mlx_and_py([x], [out], [x_test_value])


@pytest.mark.parametrize("axis", [None, 0, 1])
def test_softmax_grad(axis):
    dy = matrix("dy")
    dy_test_value = np.array([[1, 1, 1], [0, 0, 0]], dtype=config.floatX)
    sm = matrix("sm")
    sm_test_value = np.arange(6, dtype=config.floatX).reshape(2, 3)
    out = SoftmaxGrad(axis=axis)(dy, sm)

    compare_mlx_and_py([dy, sm], [out], [dy_test_value, sm_test_value])


@pytest.mark.parametrize("axis", [None, 0, 1])
def test_logsoftmax(axis):
    x = matrix("x")
    x_test_value = np.arange(6, dtype=config.floatX).reshape(2, 3)
    out = log_softmax(x, axis=axis)

    compare_mlx_and_py([x], [out], [x_test_value])


@pytest.mark.parametrize("size", [(10, 10), (1000, 1000)])
@pytest.mark.parametrize("axis", [0, 1])
def test_logsumexp_benchmark(size, axis, benchmark):
    X = matrix("X")
    X_max = pt_max(X, axis=axis, keepdims=True)
    X_max = switch(isinf(X_max), 0, X_max)
    X_lse = log(pt_sum(exp(X - X_max), axis=axis, keepdims=True)) + X_max

    rng = np.random.default_rng(23920)
    X_val = rng.normal(size=size)

    X_lse_fn = function([X], X_lse, mode="MLX")

    # JIT compile first
    _ = X_lse_fn(X_val)

    res = benchmark(X_lse_fn, X_val)

    exp_res = scipy.special.logsumexp(X_val, axis=axis, keepdims=True)
    np.testing.assert_array_almost_equal(res, exp_res)


def test_multiple_input_multiply():
    x, y, z = vectors("xyz")
    out = mul(x, y, z)
    compare_mlx_and_py([x, y, z], [out], test_inputs=[[1.5], [2.5], [3.5]])


@pytest.mark.parametrize(
    "op",
    [
        pytest.param(exp, id="exp"),
        pytest.param(log, id="log"),
        pytest.param(sin, id="sin"),
        pytest.param(cos, id="cos"),
        pytest.param(sigmoid, id="sigmoid"),
    ],
)
def test_elemwise_one_input(op) -> None:
    x = vector("x")
    out = op(x)
    x_test = mx.array([1.0, 2.0, 3.0])
    compare_mlx_and_py([x], out, [x_test])


@pytest.mark.parametrize(
    "op",
    [
        add,
        sub,
        mul,
        power,
        le,
        lt,
        ge,
        gt,
        eq,
        neq,
        true_div,
        int_div,
    ],
    ids=[
        "add",
        "sub",
        "mul",
        "power",
        "le",
        "lt",
        "ge",
        "gt",
        "eq",
        "neq",
        "true_div",
        "int_div",
    ],
)
def test_elemwise_two_inputs(op) -> None:
    x = vector("x")
    y = vector("y")
    out = op(x, y)
    x_test = mx.array([1.0, 2.0, 3.0])
    y_test = mx.array([4.0, 5.0, 6.0])
    compare_mlx_and_py([x, y], out, [x_test, y_test])
