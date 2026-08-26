import numpy as np
import pytest

import pytensor.tensor as pt
from pytensor import config
from pytensor.compile.mode import get_mode
from tests.link.mlx.test_basic import compare_mlx_and_py


mx = pytest.importorskip("mlx.core")


def test_mlx_det():
    rng = np.random.default_rng(15)

    A = pt.matrix(name="A")
    A_val = rng.normal(size=(3, 3)).astype(config.floatX)

    out = pt.linalg.det(A)

    compare_mlx_and_py([A], [out], [A_val])


def test_mlx_slogdet():
    rng = np.random.default_rng(15)

    A = pt.matrix(name="A")
    A_val = rng.normal(size=(3, 3)).astype(config.floatX)

    sign, logabsdet = pt.linalg.slogdet(A)

    compare_mlx_and_py([A], [sign, logabsdet], [A_val], mlx_mode=get_mode("MLX"))


@pytest.mark.parametrize("batch_shape", [(4,), (2, 3)], ids=["batch", "batch_2d"])
def test_mlx_det_batched(batch_shape):
    """`Det` factorizes with `mx.linalg.lu_factor`, which `mx.vmap` cannot batch (#2385)."""
    rng = np.random.default_rng(15)
    n = 3

    A = pt.tensor("A", shape=(*batch_shape, n, n))
    A_val = rng.normal(size=(*batch_shape, n, n)).astype(config.floatX)

    compare_mlx_and_py([A], [pt.linalg.det(A)], [A_val])
