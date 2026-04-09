from functools import partial

import numpy as np

import pytensor.tensor as pt
from pytensor import config
from tests.link.mlx.test_basic import compare_mlx_and_py, mlx_mode


def test_mlx_kron():
    rng = np.random.default_rng(15)

    A = pt.matrix(name="A")
    B = pt.matrix(name="B")
    A_val, B_val = rng.normal(scale=0.1, size=(2, 3, 3)).astype(config.floatX)
    out = pt.linalg.kron(A, B)

    compare_mlx_and_py(
        [A, B],
        [out],
        [A_val, B_val],
        mlx_mode=mlx_mode,
        assert_fn=partial(np.testing.assert_allclose, atol=1e-6, strict=True),
    )
