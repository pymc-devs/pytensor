from functools import partial

import numpy as np
import pytest

from pytensor import config
from pytensor.tensor.basic import switch
from pytensor.tensor.math import (
    add,
    cos,
    eq,
    erfc,
    erfcx,
    exp,
    ge,
    gt,
    int_div,
    isnan,
    le,
    log,
    lt,
    mul,
    neq,
    power,
    prod,
    sigmoid,
    sin,
    softplus,
    sub,
    true_div,
)
from pytensor.tensor.math import all as pt_all
from pytensor.tensor.math import any as pt_any
from pytensor.tensor.math import max as pt_max
from pytensor.tensor.math import min as pt_min
from pytensor.tensor.math import sum as pt_sum
from pytensor.tensor.special import log_softmax, softmax
from pytensor.tensor.type import matrix, tensor, vector, vectors
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

    x = pt_sum(a_pt, axis=None)

    compare_mlx_and_py([a_pt], [x], [np.r_[1, 2, 3].astype(config.floatX)])

    a_pt = matrix("a")

    x = pt_sum(a_pt, axis=0)

    compare_mlx_and_py([a_pt], [x], [np.c_[[1, 2, 3], [1, 2, 3]].astype(config.floatX)])

    x = pt_sum(a_pt, axis=1)

    compare_mlx_and_py([a_pt], [x], [np.c_[[1, 2, 3], [1, 2, 3]].astype(config.floatX)])

    a_pt = matrix("a")

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
def test_logsoftmax(axis):
    x = matrix("x")
    x_test_value = np.arange(6, dtype=config.floatX).reshape(2, 3)
    out = log_softmax(x, axis=axis)

    compare_mlx_and_py([x], [out], [x_test_value])


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


def test_switch() -> None:
    x = vector("x")
    y = vector("y")

    out = switch(x > 0, y, x)

    x_test = mx.array([-1.0, 2.0, 3.0])
    y_test = mx.array([4.0, 5.0, 6.0])

    compare_mlx_and_py([x, y], out, [x_test, y_test])


def test_int_div_specific() -> None:
    x = vector("x")
    y = vector("y")
    out = int_div(x, y)

    # Test with integers that demonstrate floor division behavior
    x_test = mx.array([7.0, 8.0, 9.0, -7.0, -8.0])
    y_test = mx.array([3.0, 3.0, 3.0, 3.0, 3.0])

    compare_mlx_and_py([x, y], out, [x_test, y_test])


def test_isnan() -> None:
    x = vector("x")
    out = isnan(x)

    x_test = mx.array([1.0, np.nan, 3.0, np.inf, -np.nan, 0.0, -np.inf])

    compare_mlx_and_py([x], out, [x_test])


def test_isnan_edge_cases() -> None:
    from pytensor.tensor.type import scalar

    x = scalar("x")
    out = isnan(x)

    # Test individual cases
    test_cases = [0.0, np.nan, np.inf, -np.inf, 1e-10, 1e10]

    for test_val in test_cases:
        x_test = test_val
        compare_mlx_and_py([x], out, [x_test])


def test_erfc() -> None:
    """Test complementary error function"""
    x = vector("x")
    out = erfc(x)

    # Test with various values including negative, positive, and zero
    x_test = mx.array([0.0, 0.5, 1.0, -0.5, -1.0, 2.0, -2.0, 0.1])

    compare_mlx_and_py([x], out, [x_test])


def test_erfc_extreme_values() -> None:
    """Test erfc with extreme values"""
    x = vector("x")
    out = erfc(x)

    # Test with larger values where erfc approaches 0 or 2
    x_test = mx.array([-3.0, -2.5, 2.5, 3.0])

    # Use relaxed tolerance for extreme values due to numerical precision differences
    relaxed_assert = partial(np.testing.assert_allclose, rtol=1e-3, atol=1e-6)

    compare_mlx_and_py([x], out, [x_test], assert_fn=relaxed_assert)


def test_erfcx() -> None:
    """Test scaled complementary error function"""
    x = vector("x")
    out = erfcx(x)

    # Test with positive values where erfcx is most numerically stable
    x_test = mx.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])

    relaxed_assert = partial(np.testing.assert_allclose, rtol=1e-3)

    compare_mlx_and_py([x], out, [x_test], assert_fn=relaxed_assert)


def test_erfcx_small_values() -> None:
    """Test erfcx with small values"""
    x = vector("x")
    out = erfcx(x)

    # Test with small values
    x_test = mx.array([0.001, 0.01, 0.1, 0.2])

    compare_mlx_and_py([x], out, [x_test])


def test_softplus() -> None:
    """Test softplus (log(1 + exp(x))) function"""
    x = vector("x")
    out = softplus(x)

    # Test with normal range values
    x_test = mx.array([0.0, 1.0, 2.0, -1.0, -2.0, 10.0])

    compare_mlx_and_py([x], out, [x_test])


def test_softplus_extreme_values() -> None:
    """Test softplus with extreme values to verify numerical stability"""
    x = vector("x")
    out = softplus(x)

    # Test with extreme values where different branches of the implementation are used
    x_test = mx.array([-40.0, -50.0, 20.0, 30.0, 35.0, 50.0])

    # Use relaxed tolerance for extreme values due to numerical precision differences
    relaxed_assert = partial(np.testing.assert_allclose, rtol=1e-4, atol=1e-8)

    compare_mlx_and_py([x], out, [x_test], assert_fn=relaxed_assert)


@pytest.mark.parametrize("op", [add, mul], ids=["add", "mul"])
def test_variadic_broadcast(op):
    # Regression #2086: a 3-input Elemwise with broadcasting operands used to crash
    # because the variadic path stacked the operands, which requires equal shapes.
    x = tensor("x", shape=(3, 4))
    y = tensor("y", shape=(1, 4))
    z = tensor("z", shape=(3, 1))
    out = op(x, y, z)
    assert len(out.owner.inputs) == 3  # single 3-input Elemwise, no fusion needed

    x_test = np.arange(12, dtype=config.floatX).reshape(3, 4) + 1
    y_test = np.arange(4, dtype=config.floatX).reshape(1, 4) + 1
    z_test = np.arange(3, dtype=config.floatX).reshape(3, 1) + 1
    compare_mlx_and_py([x, y, z], [out], [x_test, y_test, z_test])


@pytest.mark.parametrize("dtype", ["bool", "int8"], ids=["bool", "int8"])
def test_variadic_add_dtype(dtype):
    # Regression #2086: the variadic add upcast bool/int operands (e.g. int8 ->
    # int32) and broke bool OR-semantics. Folding the binary op preserves dtype.
    x = tensor("x", shape=(3,), dtype=dtype)
    y = tensor("y", shape=(3,), dtype=dtype)
    z = tensor("z", shape=(3,), dtype=dtype)
    out = add(x, y, z)

    def assert_fn(mlx_res, py_res):
        np.testing.assert_allclose(mlx_res, py_res)
        assert np.asarray(mlx_res).dtype == np.asarray(py_res).dtype

    vals = (
        np.array([True, False, True])
        if dtype == "bool"
        else np.array([1, 2, 3], dtype=dtype)
    )
    compare_mlx_and_py([x, y, z], [out], [vals, vals, vals], assert_fn=assert_fn)
