import numpy as np
import pytest

import pytensor.tensor as pt
from pytensor.graph import FunctionGraph
from tests.link.jax.test_basic import compare_jax_and_py


jax = pytest.importorskip("jax")


def test_jax_einsum():
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
    fg = FunctionGraph([x_pt, y_pt, z_pt], [out])
    compare_jax_and_py(fg, [x, y, z])


def test_ellipsis_einsum():
    subscripts = "...i,...i->..."
    x = np.random.rand(2, 5)
    y = np.random.rand(2, 5)

    x_pt = pt.tensor("x", shape=x.shape)
    y_pt = pt.tensor("y", shape=y.shape)
    out = pt.einsum(subscripts, x_pt, y_pt)
    fg = FunctionGraph([x_pt, y_pt], [out])
    compare_jax_and_py(fg, [x, y])
