import numpy as np
import pytest

import pytensor
import pytensor.tensor as pt
from pytensor.tensor.math import Argmax, Max
from tests.link.mlx.test_basic import compare_mlx_and_py, mx


def test_dot():
    x = pt.matrix("x")
    y = pt.matrix("y")

    out = x.dot(y)
    fn = pytensor.function([x, y], out, mode="MLX")

    seed = sum(map(ord, "test_mlx_dot"))
    rng = np.random.default_rng(seed)

    test_x = rng.normal(size=(3, 2))
    test_y = rng.normal(size=(2, 4))

    actual = fn(test_x, test_y)
    assert isinstance(actual, mx.array)
    expected = np.dot(test_x, test_y)
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


@pytest.mark.parametrize(
    "op",
    [
        pytest.param(pt.exp, id="exp"),
        pytest.param(pt.log, id="log"),
        pytest.param(pt.sin, id="sin"),
        pytest.param(pt.cos, id="cos"),
        pytest.param(pt.sigmoid, id="sigmoid"),
    ],
)
def test_elemwise_one_input(op) -> None:
    x = pt.vector("x")
    out = op(x)
    x_test = mx.array([1.0, 2.0, 3.0])
    compare_mlx_and_py([x], out, [x_test])


def test_switch() -> None:
    x = pt.vector("x")
    y = pt.vector("y")

    out = pt.switch(x > 0, y, x)

    x_test = mx.array([-1.0, 2.0, 3.0])
    y_test = mx.array([4.0, 5.0, 6.0])

    compare_mlx_and_py([x, y], out, [x_test, y_test])


@pytest.mark.parametrize("op", [pt.sum, pt.prod])
def test_input(op) -> None:
    x = pt.vector("x")
    y = pt.vector("y")
    out = op([x, y, x + y])
    x_test = mx.array([1.0, 2.0, 3.0])
    y_test = mx.array([4.0, 5.0, 6.0])
    compare_mlx_and_py([x, y], out, [x_test, y_test])


@pytest.mark.parametrize(
    "op",
    [
        pytest.param(pt.add, id="add"),
        pytest.param(pt.sub, id="sub"),
        pytest.param(pt.mul, id="mul"),
        pytest.param(pt.power, id="power"),
        pytest.param(pt.le, id="le"),
        pytest.param(pt.lt, id="lt"),
        pytest.param(pt.ge, id="ge"),
        pytest.param(pt.gt, id="gt"),
        pytest.param(pt.eq, id="eq"),
        pytest.param(pt.neq, id="neq"),
        pytest.param(pt.true_div, id="true_div"),
        pytest.param(pt.int_div, id="int_div"),
    ],
)
def test_elemwise_two_inputs(op) -> None:
    x = pt.vector("x")
    y = pt.vector("y")
    out = op(x, y)
    x_test = mx.array([1.0, 2.0, 3.0])
    y_test = mx.array([4.0, 5.0, 6.0])
    compare_mlx_and_py([x, y], out, [x_test, y_test])


def test_int_div_specific() -> None:
    """Test integer division with specific test cases"""
    x = pt.vector("x")
    y = pt.vector("y")
    out = pt.int_div(x, y)

    # Test with integers that demonstrate floor division behavior
    x_test = mx.array([7.0, 8.0, 9.0, -7.0, -8.0])
    y_test = mx.array([3.0, 3.0, 3.0, 3.0, 3.0])

    compare_mlx_and_py([x, y], out, [x_test, y_test])


def test_isnan() -> None:
    """Test IsNan operation with various inputs including NaN values"""
    x = pt.vector("x")
    out = pt.isnan(x)

    # Test with mix of normal values, NaN, and infinity
    x_test = mx.array([1.0, np.nan, 3.0, np.inf, -np.nan, 0.0, -np.inf])

    compare_mlx_and_py([x], out, [x_test])


def test_isnan_edge_cases() -> None:
    """Test IsNan with edge cases"""
    x = pt.scalar("x")
    out = pt.isnan(x)

    # Test individual cases
    test_cases = [0.0, np.nan, np.inf, -np.inf, 1e-10, 1e10]

    for test_val in test_cases:
        x_test = test_val
        compare_mlx_and_py([x], out, [x_test])


def test_erfc() -> None:
    """Test complementary error function"""
    x = pt.vector("x")
    out = pt.erfc(x)

    # Test with various values including negative, positive, and zero
    x_test = mx.array([0.0, 0.5, 1.0, -0.5, -1.0, 2.0, -2.0, 0.1])

    compare_mlx_and_py([x], out, [x_test])


def test_erfc_extreme_values() -> None:
    """Test erfc with extreme values"""
    x = pt.vector("x")
    out = pt.erfc(x)

    # Test with larger values where erfc approaches 0 or 2
    x_test = mx.array([-3.0, -2.5, 2.5, 3.0])

    # Use relaxed tolerance for extreme values due to numerical precision differences
    from functools import partial

    relaxed_assert = partial(np.testing.assert_allclose, rtol=1e-3, atol=1e-6)

    compare_mlx_and_py([x], out, [x_test], assert_fn=relaxed_assert)


def test_erfcx() -> None:
    """Test scaled complementary error function"""
    x = pt.vector("x")
    out = pt.erfcx(x)

    # Test with positive values where erfcx is most numerically stable
    x_test = mx.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])

    compare_mlx_and_py([x], out, [x_test])


def test_erfcx_small_values() -> None:
    """Test erfcx with small values"""
    x = pt.vector("x")
    out = pt.erfcx(x)

    # Test with small values
    x_test = mx.array([0.001, 0.01, 0.1, 0.2])

    compare_mlx_and_py([x], out, [x_test])


def test_softplus() -> None:
    """Test softplus (log(1 + exp(x))) function"""
    x = pt.vector("x")
    out = pt.softplus(x)

    # Test with normal range values
    x_test = mx.array([0.0, 1.0, 2.0, -1.0, -2.0, 10.0])

    compare_mlx_and_py([x], out, [x_test])


def test_softplus_extreme_values() -> None:
    """Test softplus with extreme values to verify numerical stability"""
    x = pt.vector("x")
    out = pt.softplus(x)

    # Test with extreme values where different branches of the implementation are used
    x_test = mx.array([-40.0, -50.0, 20.0, 30.0, 35.0, 50.0])

    # Use relaxed tolerance for extreme values due to numerical precision differences
    from functools import partial

    relaxed_assert = partial(np.testing.assert_allclose, rtol=1e-4, atol=1e-8)

    compare_mlx_and_py([x], out, [x_test], assert_fn=relaxed_assert)


@pytest.mark.xfail(reason="Argmax not implemented yet")
def test_mlx_max_and_argmax():
    # Test that a single output of a multi-output `Op` can be used as input to
    # another `Op`
    x = pt.dvector()
    mx = Max([0])(x)
    amx = Argmax([0])(x)
    out = mx * amx
    compare_mlx_and_py([x], [out], [np.r_[1, 2]])
