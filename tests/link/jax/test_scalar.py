import numpy as np
import pytest

import pytensor.scalar.basic as ps
import pytensor.tensor as pt
from pytensor.configdefaults import config
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.op import get_test_value
from pytensor.scalar.basic import Composite
from pytensor.tensor import as_tensor
from pytensor.tensor.elemwise import Elemwise
from pytensor.tensor.math import all as pt_all
from pytensor.tensor.math import (
    betaincinv,
    cosh,
    erf,
    erfc,
    erfcinv,
    erfcx,
    erfinv,
    gammainccinv,
    gammaincinv,
    iv,
    kve,
    log,
    log1mexp,
    polygamma,
    psi,
    sigmoid,
    softplus,
    tri_gamma,
)
from pytensor.tensor.type import matrix, scalar, vector
from tests.link.jax.test_basic import compare_jax_and_py


jax = pytest.importorskip("jax")
from pytensor.link.jax.dispatch import jax_funcify


try:
    pass

    TFP_INSTALLED = True
except ModuleNotFoundError:
    TFP_INSTALLED = False


def test_second():
    a0 = scalar("a0")
    b = scalar("b")

    out = ps.second(a0, b)
    fgraph = FunctionGraph([a0, b], [out])
    compare_jax_and_py(fgraph, [10.0, 5.0])

    a1 = vector("a1")
    out = pt.second(a1, b)
    fgraph = FunctionGraph([a1, b], [out])
    compare_jax_and_py(fgraph, [np.zeros([5], dtype=config.floatX), 5.0])

    a2 = matrix("a2", shape=(1, None), dtype="float64")
    b2 = matrix("b2", shape=(None, 1), dtype="int32")
    out = pt.second(a2, b2)
    fgraph = FunctionGraph([a2, b2], [out])
    compare_jax_and_py(
        fgraph, [np.zeros((1, 3), dtype="float64"), np.ones((5, 1), dtype="int32")]
    )


def test_second_constant_scalar():
    b = scalar("b", dtype="int")
    out = pt.second(0.0, b)
    fgraph = FunctionGraph([b], [out])
    # Test dispatch directly as useless second is removed during compilation
    fn = jax_funcify(fgraph)
    [res] = fn(1)
    assert res == 1
    assert res.dtype == out.dtype


def test_identity():
    a = scalar("a")
    a.tag.test_value = 10

    out = ps.identity(a)
    fgraph = FunctionGraph([a], [out])
    compare_jax_and_py(fgraph, [get_test_value(i) for i in fgraph.inputs])


@pytest.mark.parametrize(
    "x, y, x_val, y_val",
    [
        (scalar("x"), scalar("y"), np.array(10), np.array(20)),
        (scalar("x"), vector("y"), np.array(10), np.arange(10, 20)),
        (
            matrix("x"),
            vector("y"),
            np.arange(10 * 20).reshape((20, 10)),
            np.arange(10, 20),
        ),
    ],
)
def test_jax_Composite_singe_output(x, y, x_val, y_val):
    x_s = ps.float64("x")
    y_s = ps.float64("y")

    comp_op = Elemwise(Composite([x_s, y_s], [x_s + y_s * 2 + ps.exp(x_s - y_s)]))

    out = comp_op(x, y)

    out_fg = FunctionGraph([x, y], [out])

    test_input_vals = [
        x_val.astype(config.floatX),
        y_val.astype(config.floatX),
    ]
    _ = compare_jax_and_py(out_fg, test_input_vals)


def test_jax_Composite_multi_output():
    x = vector("x")

    x_s = ps.float64("xs")
    outs = Elemwise(Composite(inputs=[x_s], outputs=[x_s + 1, x_s - 1]))(x)

    fgraph = FunctionGraph([x], outs)
    compare_jax_and_py(fgraph, [np.arange(10, dtype=config.floatX)])


def test_erf():
    x = scalar("x")
    out = erf(x)
    fg = FunctionGraph([x], [out])

    compare_jax_and_py(fg, [1.0])


def test_erfc():
    x = scalar("x")
    out = erfc(x)
    fg = FunctionGraph([x], [out])

    compare_jax_and_py(fg, [1.0])


def test_erfinv():
    x = scalar("x")
    out = erfinv(x)
    fg = FunctionGraph([x], [out])

    compare_jax_and_py(fg, [0.95])


@pytest.mark.parametrize(
    "op, test_values",
    [
        (erfcx, (0.7,)),
        (erfcinv, (0.7,)),
        (iv, (0.3, 0.7)),
        (kve, (-2.5, 2.0)),
    ],
)
@pytest.mark.skipif(not TFP_INSTALLED, reason="Test requires tensorflow-probability")
def test_tfp_ops(op, test_values):
    inputs = [as_tensor(test_value).type() for test_value in test_values]
    output = op(*inputs)

    fg = FunctionGraph(inputs, [output])
    compare_jax_and_py(fg, test_values)


def test_betaincinv():
    a = vector("a", dtype="float64")
    b = vector("b", dtype="float64")
    x = vector("x", dtype="float64")
    out = betaincinv(a, b, x)
    fg = FunctionGraph([a, b, x], [out])
    compare_jax_and_py(
        fg,
        [
            np.array([5.5, 7.0]),
            np.array([5.5, 7.0]),
            np.array([0.25, 0.7]),
        ],
    )


def test_gammaincinv():
    k = vector("k", dtype="float64")
    x = vector("x", dtype="float64")
    out = gammaincinv(k, x)
    fg = FunctionGraph([k, x], [out])
    compare_jax_and_py(fg, [np.array([5.5, 7.0]), np.array([0.25, 0.7])])


def test_gammainccinv():
    k = vector("k", dtype="float64")
    x = vector("x", dtype="float64")
    out = gammainccinv(k, x)
    fg = FunctionGraph([k, x], [out])
    compare_jax_and_py(fg, [np.array([5.5, 7.0]), np.array([0.25, 0.7])])


def test_psi():
    x = scalar("x")
    out = psi(x)
    fg = FunctionGraph([x], [out])
    compare_jax_and_py(fg, [3.0])


def test_tri_gamma():
    x = vector("x", dtype="float64")
    out = tri_gamma(x)
    fg = FunctionGraph([x], [out])
    compare_jax_and_py(fg, [np.array([3.0, 5.0])])


def test_polygamma():
    n = vector("n", dtype="int32")
    x = vector("x", dtype="float32")
    out = polygamma(n, x)
    fg = FunctionGraph([n, x], [out])
    compare_jax_and_py(
        fg,
        [
            np.array([0, 1, 2]).astype("int32"),
            np.array([0.5, 0.9, 2.5]).astype("float32"),
        ],
    )


def test_log1mexp():
    x = vector("x")
    out = log1mexp(x)
    fg = FunctionGraph([x], [out])

    compare_jax_and_py(fg, [[-1.0, -0.75, -0.5, -0.25]])


def test_nnet():
    x = vector("x")
    x.tag.test_value = np.r_[1.0, 2.0].astype(config.floatX)

    out = sigmoid(x)
    fgraph = FunctionGraph([x], [out])
    compare_jax_and_py(fgraph, [get_test_value(i) for i in fgraph.inputs])

    out = softplus(x)
    fgraph = FunctionGraph([x], [out])
    compare_jax_and_py(fgraph, [get_test_value(i) for i in fgraph.inputs])


def test_jax_variadic_Scalar():
    mu = vector("mu", dtype=config.floatX)
    mu.tag.test_value = np.r_[0.1, 1.1].astype(config.floatX)
    tau = vector("tau", dtype=config.floatX)
    tau.tag.test_value = np.r_[1.0, 2.0].astype(config.floatX)

    res = -tau * mu

    fgraph = FunctionGraph([mu, tau], [res])

    compare_jax_and_py(fgraph, [get_test_value(i) for i in fgraph.inputs])

    res = -tau * (tau - mu) ** 2

    fgraph = FunctionGraph([mu, tau], [res])

    compare_jax_and_py(fgraph, [get_test_value(i) for i in fgraph.inputs])


def test_add_scalars():
    x = pt.matrix("x")
    size = x.shape[0] + x.shape[0] + x.shape[1]
    out = pt.ones(size).astype(config.floatX)

    out_fg = FunctionGraph([x], [out])
    compare_jax_and_py(out_fg, [np.ones((2, 3)).astype(config.floatX)])


def test_mul_scalars():
    x = pt.matrix("x")
    size = x.shape[0] * x.shape[0] * x.shape[1]
    out = pt.ones(size).astype(config.floatX)

    out_fg = FunctionGraph([x], [out])
    compare_jax_and_py(out_fg, [np.ones((2, 3)).astype(config.floatX)])


def test_div_scalars():
    x = pt.matrix("x")
    size = x.shape[0] // x.shape[1]
    out = pt.ones(size).astype(config.floatX)

    out_fg = FunctionGraph([x], [out])
    compare_jax_and_py(out_fg, [np.ones((12, 3)).astype(config.floatX)])


def test_mod_scalars():
    x = pt.matrix("x")
    size = x.shape[0] % x.shape[1]
    out = pt.ones(size).astype(config.floatX)

    out_fg = FunctionGraph([x], [out])
    compare_jax_and_py(out_fg, [np.ones((12, 3)).astype(config.floatX)])


def test_jax_multioutput():
    x = vector("x")
    x.tag.test_value = np.r_[1.0, 2.0].astype(config.floatX)
    y = vector("y")
    y.tag.test_value = np.r_[3.0, 4.0].astype(config.floatX)

    w = cosh(x**2 + y / 3.0)
    v = cosh(x / 3.0 + y**2)

    fgraph = FunctionGraph([x, y], [w, v])

    compare_jax_and_py(fgraph, [get_test_value(i) for i in fgraph.inputs])


def test_jax_logp():
    mu = vector("mu")
    mu.tag.test_value = np.r_[0.0, 0.0].astype(config.floatX)
    tau = vector("tau")
    tau.tag.test_value = np.r_[1.0, 1.0].astype(config.floatX)
    sigma = vector("sigma")
    sigma.tag.test_value = (1.0 / get_test_value(tau)).astype(config.floatX)
    value = vector("value")
    value.tag.test_value = np.r_[0.1, -10].astype(config.floatX)

    logp = (-tau * (value - mu) ** 2 + log(tau / np.pi / 2.0)) / 2.0
    conditions = [sigma > 0]
    alltrue = pt_all([pt_all(1 * val) for val in conditions])
    normal_logp = pt.switch(alltrue, logp, -np.inf)

    fgraph = FunctionGraph([mu, tau, sigma, value], [normal_logp])

    compare_jax_and_py(fgraph, [get_test_value(i) for i in fgraph.inputs])
