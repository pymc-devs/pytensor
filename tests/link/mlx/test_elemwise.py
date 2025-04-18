import pytest

import pytensor.tensor as pt
from tests.link.mlx.test_basic import compare_mlx_and_py, mx


@pytest.mark.parametrize("op", [pt.any, pt.all, pt.max, pt.min])
def test_input(op) -> None:
    x = pt.vector("x")
    out = op(x > 0)
    x_test = mx.array([1.0, 2.0, 3.0])

    compare_mlx_and_py([x], out, [x_test])
