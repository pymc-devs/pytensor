import numpy as np
import pytest

import pytensor
import pytensor.tensor as pt
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
    ],
)
def test_elemwise_two_inputs(op) -> None:
    x = pt.vector("x")
    y = pt.vector("y")
    out = op(x, y)
    x_test = mx.array([1.0, 2.0, 3.0])
    y_test = mx.array([4.0, 5.0, 6.0])
    compare_mlx_and_py([x, y], out, [x_test, y_test])
