import pytensor.tensor as pt
from tests.link.mlx.test_basic import compare_mlx_and_py, mx


def test_all() -> None:
    x = pt.vector("x")

    out = pt.all(x > 0)

    x_test = mx.array([-1.0, 2.0, 3.0])

    compare_mlx_and_py([x], out, [x_test])
