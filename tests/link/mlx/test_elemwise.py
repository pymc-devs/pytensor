import numpy as np
import pytest

import pytensor.tensor as pt
from tests.link.mlx.test_basic import compare_mlx_and_py


mx = pytest.importorskip("mlx.core")


@pytest.mark.parametrize("op", [pt.any, pt.all, pt.max, pt.min])
def test_input(op) -> None:
    x = pt.vector("x")
    out = op(x > 0)
    x_test = mx.array([1.0, 2.0, 3.0])

    compare_mlx_and_py([x], out, [x_test])


def test_elemwise_operations() -> None:
    """Test elemwise operations (IntDiv, IsNan, Erfc, Erfcx, Softplus) in elemwise context"""
    x = pt.vector("x")
    y = pt.vector("y")

    # Test int_div in an elemwise expression
    out_int_div = pt.int_div(x, y) + 1
    x_test = mx.array([10.0, 15.0, 20.0])
    y_test = mx.array([3.0, 4.0, 6.0])
    compare_mlx_and_py([x, y], out_int_div, [x_test, y_test])

    # Test isnan in an elemwise expression
    z = pt.vector("z")
    out_isnan = pt.isnan(z).astype("float32") * 10
    z_test = mx.array([1.0, np.nan, 3.0])
    compare_mlx_and_py([z], out_isnan, [z_test])

    # Test erfc in an elemwise expression
    w = pt.vector("w")
    out_erfc = pt.erfc(w) * 2.0
    w_test = mx.array([0.0, 0.5, 1.0])
    compare_mlx_and_py([w], out_erfc, [w_test])

    # Test erfcx in an elemwise expression
    v = pt.vector("v")
    out_erfcx = pt.erfcx(v) + 0.1
    v_test = mx.array([0.0, 1.0, 2.0])
    compare_mlx_and_py([v], out_erfcx, [v_test])

    # Test softplus in an elemwise expression
    u = pt.vector("u")
    out_softplus = pt.softplus(u) - 0.5
    u_test = mx.array([0.0, 1.0, -1.0])
    compare_mlx_and_py([u], out_softplus, [u_test])
