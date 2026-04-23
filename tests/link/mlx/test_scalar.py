import numpy as np
import pytest

import pytensor.scalar.basic as ps
import pytensor.tensor as pt
from pytensor.compile.io import In
from pytensor.configdefaults import config
from pytensor.graph.fg import FunctionGraph
from pytensor.scalar.basic import Composite
from pytensor.tensor.elemwise import Elemwise
from pytensor.tensor.math import all as pt_all
from pytensor.tensor.math import (
    cosh,
    erf,
    erfc,
    erfcx,
    erfinv,
    log,
    log1mexp,
    sigmoid,
    softplus,
)
from pytensor.tensor.type import matrix, scalar, vector
from tests.link.mlx.test_basic import compare_mlx_and_py


mlx = pytest.importorskip("mlx.core")
from pytensor.link.mlx.dispatch import mlx_funcify


def test_second():
    a0 = scalar("a0")
    b = scalar("b")

    out = ps.second(a0, b)
    compare_mlx_and_py([a0, b], [out], [10.0, 5.0])

    a1 = vector("a1")
    out = pt.second(a1, b)
    compare_mlx_and_py([a1, b], [out], [np.zeros([5], dtype=config.floatX), 5.0])

    a2 = matrix("a2", shape=(1, None), dtype="float64")
    b2 = matrix("b2", shape=(None, 1), dtype="int32")
    out = pt.second(a2, b2)
    compare_mlx_and_py(
        [a2, b2],
        [out],
        [np.zeros((1, 3), dtype="float64"), np.ones((5, 1), dtype="int32")],
    )


def test_second_constant_scalar():
    b = scalar("b", dtype="int32")
    out = pt.second(0.0, b)
    fgraph = FunctionGraph([b], [out])
    # Test dispatch directly as useless second is removed during compilation
    fn = mlx_funcify(fgraph)
    [res] = fn(1)
    assert res == 1

    # Cast to numpy to get compariable dtypes
    assert np.array(res).dtype == out.dtype


def test_identity():
    a = scalar("a")
    a_test_value = 10

    out = ps.identity(a)
    compare_mlx_and_py([a], [out], [a_test_value])


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
def test_mlx_Composite_singe_output(x, y, x_val, y_val):
    x_s = ps.float64("x")
    y_s = ps.float64("y")

    # Change exp -> cos relative to the JAX test, because exp overflows in float32 mode (MLX default)
    comp_op = Elemwise(Composite([x_s, y_s], [x_s + y_s * 2 + ps.cos(x_s - y_s)]))

    out = comp_op(x, y)

    test_input_vals = [
        x_val.astype(config.floatX),
        y_val.astype(config.floatX),
    ]

    _ = compare_mlx_and_py([x, y], [out], test_input_vals)


def test_mlx_Composite_multi_output():
    x = vector("x")

    x_s = ps.float64("xs")
    outs = Elemwise(Composite(inputs=[x_s], outputs=[x_s + 1, x_s - 1]))(x)

    compare_mlx_and_py(
        [x],
        outs,
        [np.arange(10, dtype=config.floatX)],
    )


def test_erf():
    x = scalar("x")
    out = erf(x)
    compare_mlx_and_py([x], [out], [1.0])


def test_erfc():
    x = scalar("x")
    out = erfc(x)
    compare_mlx_and_py([x], [out], [1.0])


def test_erfinv():
    x = scalar("x")
    out = erfinv(x)
    compare_mlx_and_py([x], [out], [0.95])


def test_erfcx():
    x = scalar("x")
    out = erfcx(x)
    compare_mlx_and_py([x], [out], [0.7])


def test_log1mexp():
    x = vector("x")
    out = log1mexp(x)

    compare_mlx_and_py([x], [out], [[-1.0, -0.75, -0.5, -0.25]])


def test_nnet():
    x = vector("x")
    x_test_value = np.r_[1.0, 2.0].astype(config.floatX)

    out = sigmoid(x)
    compare_mlx_and_py([x], [out], [x_test_value])

    out = softplus(x)
    compare_mlx_and_py([x], [out], [x_test_value])


def test_mlx_variadic_Scalar():
    mu = vector("mu", dtype=config.floatX)
    mu_test_value = np.r_[0.1, 1.1].astype(config.floatX)
    tau = vector("tau", dtype=config.floatX)
    tau_test_value = np.r_[1.0, 2.0].astype(config.floatX)

    res = -tau * mu

    compare_mlx_and_py([mu, tau], [res], [mu_test_value, tau_test_value])

    res = -tau * (tau - mu) ** 2

    compare_mlx_and_py([mu, tau], [res], [mu_test_value, tau_test_value])


def test_add_scalars():
    x = pt.matrix("x")
    size = x.shape[0] + x.shape[0] + x.shape[1]
    out = pt.ones(size).astype(config.floatX)

    compare_mlx_and_py([x], [out], [np.ones((2, 3)).astype(config.floatX)])


def test_mul_scalars():
    x = pt.matrix("x")
    size = x.shape[0] * x.shape[0] * x.shape[1]
    out = pt.ones(size).astype(config.floatX)

    compare_mlx_and_py([x], [out], [np.ones((2, 3)).astype(config.floatX)])


def test_div_scalars():
    x = pt.matrix("x")
    size = x.shape[0] // x.shape[1]
    out = pt.ones(size).astype(config.floatX)

    compare_mlx_and_py([x], [out], [np.ones((12, 3)).astype(config.floatX)])


def test_mod_scalars():
    x = pt.matrix("x")
    size = x.shape[0] % x.shape[1]
    out = pt.ones(size).astype(config.floatX)

    compare_mlx_and_py([x], [out], [np.ones((12, 3)).astype(config.floatX)])


def test_mlx_multioutput():
    x = vector("x")
    x_test_value = np.r_[1.0, 2.0].astype(config.floatX)
    y = vector("y")
    y_test_value = np.r_[3.0, 4.0].astype(config.floatX)

    w = cosh(x**2 + y / 3.0)
    v = cosh(x / 3.0 + y**2)

    compare_mlx_and_py([x, y], [w, v], [x_test_value, y_test_value])


def test_mlx_logp():
    mu = vector("mu")
    mu_test_value = np.r_[0.0, 0.0].astype(config.floatX)
    tau = vector("tau")
    tau_test_value = np.r_[1.0, 1.0].astype(config.floatX)
    sigma = vector("sigma")
    sigma_test_value = (1.0 / tau_test_value).astype(config.floatX)
    value = vector("value")
    value_test_value = np.r_[0.1, -10].astype(config.floatX)

    logp = (-tau * (value - mu) ** 2 + log(tau / np.pi / 2.0)) / 2.0
    conditions = [sigma > 0]
    alltrue = pt_all([pt_all(1 * val) for val in conditions])
    normal_logp = pt.switch(alltrue, logp, -np.inf)

    compare_mlx_and_py(
        [mu, tau, sigma, value],
        [normal_logp],
        [
            mu_test_value,
            tau_test_value,
            sigma_test_value,
            value_test_value,
        ],
    )


def test_mlx_real():
    x = pt.tensor("x", dtype="complex64", shape=(None,))
    out = pt.real(x)[0].set(np.float32(99.0))
    x_val = np.array([1 + 2j, 3 + 4j], dtype="complex64")
    _, output = compare_mlx_and_py([In(x, mutable=True)], [out], [x_val])

    # Verify that the real Op does not return a view, resulting in mutation of the input
    assert output[0][0].item() != x_val.real[0].item()
