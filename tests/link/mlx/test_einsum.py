import numpy as np
import pytest

import pytensor.tensor as pt
from tests.link.mlx.test_basic import compare_mlx_and_py


mx = pytest.importorskip("mlx.core")


def test_mlx_einsum():
    subscripts = "ij, jk, kl -> il"
    x = np.random.rand(3, 5)
    y = np.random.rand(5, 2)
    z = np.random.rand(2, 4)

    shapes = {
        "x": (3, 5),
        "y": (5, 2),
        "z": (2, 4),
    }
    x_pt, y_pt, z_pt = (pt.tensor(name, shape=shape) for name, shape in shapes.items())
    out = pt.einsum(subscripts, x_pt, y_pt, z_pt)
    compare_mlx_and_py([x_pt, y_pt, z_pt], [out], [x, y, z])


def test_ellipsis_einsum():
    subscripts = "...i,...i->..."
    x = np.random.rand(2, 5)
    y = np.random.rand(2, 5)

    x_pt = pt.tensor("x", shape=x.shape)
    y_pt = pt.tensor("y", shape=y.shape)
    out = pt.einsum(subscripts, x_pt, y_pt)
    compare_mlx_and_py([x_pt, y_pt], [out], [x, y])


def test_einsum_trace():
    subscripts = "ii->"
    x_pt = pt.matrix("x")
    x_val = np.random.rand(5, 5)
    out = pt.einsum(subscripts, x_pt)
    compare_mlx_and_py([x_pt], [out], [x_val])


def test_einsum_batched_outer_product():
    a = pt.matrix("a", dtype="float32")
    b = pt.matrix("b", dtype="float32")
    out = pt.einsum("bi,bj->bij", a, b)

    a_val = np.random.normal(size=(5, 3)).astype("float32")
    b_val = np.random.normal(size=(5, 2)).astype("float32")

    compare_mlx_and_py([a, b], [out], [a_val, b_val])
