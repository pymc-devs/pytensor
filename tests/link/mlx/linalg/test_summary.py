import numpy as np
import pytest

import pytensor.tensor as pt
from pytensor import config
from pytensor.compile.mode import get_mode
from tests.link.mlx.test_basic import compare_mlx_and_py


mx = pytest.importorskip("mlx.core")


@pytest.mark.parametrize("batch_shape", [(), (3,)], ids=["core", "batched"])
def test_mlx_det(batch_shape):
    rng = np.random.default_rng(15)

    A = pt.tensor("A", shape=(*batch_shape, 3, 3))
    A_val = rng.normal(size=(*batch_shape, 3, 3)).astype(config.floatX)

    out = pt.linalg.det(A)

    compare_mlx_and_py([A], [out], [A_val])


@pytest.mark.parametrize("batch_shape", [(), (3,)], ids=["core", "batched"])
def test_mlx_slogdet(batch_shape):
    rng = np.random.default_rng(15)

    A = pt.tensor("A", shape=(*batch_shape, 3, 3))
    A_val = rng.normal(size=(*batch_shape, 3, 3)).astype(config.floatX)

    sign, logabsdet = pt.linalg.slogdet(A)

    compare_mlx_and_py([A], [sign, logabsdet], [A_val], mlx_mode=get_mode("MLX"))
