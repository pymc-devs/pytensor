import numpy as np
import pytest

import pytensor.scalar.basic as ps
import pytensor.tensor as pt
from pytensor.configdefaults import config
from pytensor.graph.fg import FunctionGraph
from pytensor.scalar.basic import Composite
from pytensor.tensor.elemwise import Elemwise
from pytensor.tensor.math import (
    erf,
    erfc,
    erfcx,
    erfinv,
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
