import numpy as np
import pytest

import pytensor
import pytensor.tensor as pt


jax = pytest.importorskip("jax")


def test_jax_einsum():
    subscripts = "ij, jk, kl -> il"
    x = np.random.rand(3, 5)
    y = np.random.rand(5, 2)
    z = np.random.rand(2, 4)

    shapes = ((3, 5), (5, 2), (2, 4))
    x_pt, y_pt, z_pt = (
        pt.tensor(name, shape=shape) for name, shape in zip("xyz", shapes)
    )
    out = pt.einsum(subscripts, x_pt, y_pt, z_pt)
    f = pytensor.function([x_pt, y_pt, z_pt], out, mode="JAX")

    np.testing.assert_allclose(f(x, y, z), np.einsum(subscripts, x, y, z))


@pytest.mark.xfail(raises=NotImplementedError)
def test_ellipsis_einsum():
    subscripts = "...i,...i->..."
    x = np.random.rand(2, 5)
    y = np.random.rand(2, 5)

    x_pt = pt.tensor("x", shape=x.shape)
    y_pt = pt.tensor("y", shape=y.shape)
    out = pt.einsum(subscripts, x_pt, y_pt)
    f = pytensor.function([x_pt, y_pt], out, mode="JAX")

    np.testing.assert_allclose(f(x, y), np.einsum(subscripts, x, y))
