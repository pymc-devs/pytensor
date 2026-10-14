from functools import partial

import numpy as np
import pytest

import pytensor.tensor as pt
from pytensor import config
from tests.link.mlx.test_basic import compare_mlx_and_py, mlx_mode


@pytest.mark.parametrize("batch_shape", [(), (3,)], ids=["core", "batched"])
@pytest.mark.parametrize("op", [pt.linalg.inv, pt.linalg.pinv], ids=["inv", "pinv"])
def test_mlx_inv(op, batch_shape):
    rng = np.random.default_rng(15)
    n = 3

    A = pt.tensor("A", shape=(*batch_shape, n, n))
    A_val = rng.normal(size=(*batch_shape, n, n))
    A_val = (A_val @ np.swapaxes(A_val, -1, -2)).astype(config.floatX)

    out = op(A)

    compare_mlx_and_py(
        [A],
        [out],
        [A_val],
        mlx_mode=mlx_mode,
        assert_fn=partial(
            np.testing.assert_allclose, atol=1e-6, rtol=1e-6, strict=True
        ),
    )
