import numpy as np
import pytest
import scipy.special

import pytensor
import pytensor.tensor as pt
from pytensor.compile import get_mode
from pytensor.configdefaults import config
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.op import get_test_value
from pytensor.tensor import elemwise as pt_elemwise
from pytensor.tensor.math import all as pt_all
from pytensor.tensor.math import prod
from pytensor.tensor.math import sum as pt_sum
from pytensor.tensor.special import SoftmaxGrad, log_softmax, softmax
from pytensor.tensor.type import matrix, tensor, vector, vectors
from tests.link.jax.test_basic import compare_jax_and_py
from tests.tensor.test_elemwise import TestElemwise


def test_elemwise_runtime_broadcast():
    TestElemwise.check_runtime_broadcast(get_mode("JAX"))


def test_jax_Dimshuffle():
    a_pt = matrix("a")

    x = a_pt.T
    x_fg = FunctionGraph([a_pt], [x])
    compare_jax_and_py(x_fg, [np.c_[[1.0, 2.0], [3.0, 4.0]].astype(config.floatX)])

    x = a_pt.dimshuffle([0, 1, "x"])
    x_fg = FunctionGraph([a_pt], [x])
    compare_jax_and_py(x_fg, [np.c_[[1.0, 2.0], [3.0, 4.0]].astype(config.floatX)])

    a_pt = tensor(dtype=config.floatX, shape=(None, 1))
    x = a_pt.dimshuffle((0,))
    x_fg = FunctionGraph([a_pt], [x])
    compare_jax_and_py(x_fg, [np.c_[[1.0, 2.0, 3.0, 4.0]].astype(config.floatX)])

    a_pt = tensor(dtype=config.floatX, shape=(None, 1))
    x = pt_elemwise.DimShuffle(input_ndim=2, new_order=(0,))(a_pt)
    x_fg = FunctionGraph([a_pt], [x])
    compare_jax_and_py(x_fg, [np.c_[[1.0, 2.0, 3.0, 4.0]].astype(config.floatX)])


def test_jax_CAReduce():
    a_pt = vector("a")
    a_pt.tag.test_value = np.r_[1, 2, 3].astype(config.floatX)

    x = pt_sum(a_pt, axis=None)
    x_fg = FunctionGraph([a_pt], [x])

    compare_jax_and_py(x_fg, [np.r_[1, 2, 3].astype(config.floatX)])

    a_pt = matrix("a")
    a_pt.tag.test_value = np.c_[[1, 2, 3], [1, 2, 3]].astype(config.floatX)

    x = pt_sum(a_pt, axis=0)
    x_fg = FunctionGraph([a_pt], [x])

    compare_jax_and_py(x_fg, [np.c_[[1, 2, 3], [1, 2, 3]].astype(config.floatX)])

    x = pt_sum(a_pt, axis=1)
    x_fg = FunctionGraph([a_pt], [x])

    compare_jax_and_py(x_fg, [np.c_[[1, 2, 3], [1, 2, 3]].astype(config.floatX)])

    a_pt = matrix("a")
    a_pt.tag.test_value = np.c_[[1, 2, 3], [1, 2, 3]].astype(config.floatX)

    x = prod(a_pt, axis=0)
    x_fg = FunctionGraph([a_pt], [x])

    compare_jax_and_py(x_fg, [np.c_[[1, 2, 3], [1, 2, 3]].astype(config.floatX)])

    x = pt_all(a_pt)
    x_fg = FunctionGraph([a_pt], [x])

    compare_jax_and_py(x_fg, [np.c_[[1, 2, 3], [1, 2, 3]].astype(config.floatX)])


@pytest.mark.parametrize("axis", [None, 0, 1])
def test_softmax(axis):
    x = matrix("x")
    x.tag.test_value = np.arange(6, dtype=config.floatX).reshape(2, 3)
    out = softmax(x, axis=axis)
    fgraph = FunctionGraph([x], [out])
    compare_jax_and_py(fgraph, [get_test_value(i) for i in fgraph.inputs])


@pytest.mark.parametrize("axis", [None, 0, 1])
def test_logsoftmax(axis):
    x = matrix("x")
    x.tag.test_value = np.arange(6, dtype=config.floatX).reshape(2, 3)
    out = log_softmax(x, axis=axis)
    fgraph = FunctionGraph([x], [out])
    compare_jax_and_py(fgraph, [get_test_value(i) for i in fgraph.inputs])


@pytest.mark.parametrize("axis", [None, 0, 1])
def test_softmax_grad(axis):
    dy = matrix("dy")
    dy.tag.test_value = np.array([[1, 1, 1], [0, 0, 0]], dtype=config.floatX)
    sm = matrix("sm")
    sm.tag.test_value = np.arange(6, dtype=config.floatX).reshape(2, 3)
    out = SoftmaxGrad(axis=axis)(dy, sm)
    fgraph = FunctionGraph([dy, sm], [out])
    compare_jax_and_py(fgraph, [get_test_value(i) for i in fgraph.inputs])


@pytest.mark.parametrize("size", [(10, 10), (1000, 1000)])
@pytest.mark.parametrize("axis", [0, 1])
def test_logsumexp_benchmark(size, axis, benchmark):
    X = pt.matrix("X")
    X_max = pt.max(X, axis=axis, keepdims=True)
    X_max = pt.switch(pt.isinf(X_max), 0, X_max)
    X_lse = pt.log(pt.sum(pt.exp(X - X_max), axis=axis, keepdims=True)) + X_max

    rng = np.random.default_rng(23920)
    X_val = rng.normal(size=size)

    X_lse_fn = pytensor.function([X], X_lse, mode="JAX")

    # JIT compile first
    _ = X_lse_fn(X_val)

    res = benchmark(X_lse_fn, X_val)

    exp_res = scipy.special.logsumexp(X_val, axis=axis, keepdims=True)
    np.testing.assert_array_almost_equal(res, exp_res)


def test_multiple_input_multiply():
    x, y, z = vectors("xyz")
    out = pt.mul(x, y, z)

    fg = FunctionGraph(outputs=[out], clone=False)
    compare_jax_and_py(fg, [[1.5], [2.5], [3.5]])
