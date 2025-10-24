from functools import partial

import numpy as np
import pytest

import pytensor.tensor as pt
from pytensor import config
from tests.link.mlx.test_basic import compare_mlx_and_py, mlx_mode


mlx_linalg_mode = mlx_mode.including("blockwise")


@pytest.mark.parametrize("compute_uv", [True, False])
def test_mlx_svd(compute_uv):
    rng = np.random.default_rng()

    A = pt.matrix(name="X")
    A_val = rng.normal(size=(3, 3)).astype(config.floatX)
    A_val = A_val @ A_val.T

    out = pt.linalg.svd(A, compute_uv=compute_uv)

    compare_mlx_and_py(
        [A],
        out,
        [A_val],
        mlx_mode=mlx_linalg_mode,
        assert_fn=partial(np.testing.assert_allclose, atol=1e-4, strict=True),
    )


def test_mlx_kron():
    rng = np.random.default_rng()

    A = pt.matrix(name="A")
    B = pt.matrix(name="B")
    A_val, B_val = rng.normal(scale=0.1, size=(2, 3, 3)).astype(config.floatX)
    out = pt.linalg.kron(A, B)

    compare_mlx_and_py(
        [A, B],
        [out],
        [A_val, B_val],
        mlx_mode=mlx_linalg_mode,
        assert_fn=partial(np.testing.assert_allclose, atol=1e-4, strict=True),
    )


@pytest.mark.parametrize("op", [pt.linalg.inv, pt.linalg.pinv], ids=["inv", "pinv"])
def test_mlx_inv(op):
    rng = np.random.default_rng()
    n = 3

    A = pt.matrix(name="A")
    A_val = rng.normal(size=(n, n))
    A_val = (A_val @ A_val.T).astype(config.floatX)

    out = op(A)

    compare_mlx_and_py(
        [A],
        [out],
        [A_val],
        mlx_mode=mlx_linalg_mode,
        assert_fn=partial(
            np.testing.assert_allclose, atol=1e-3, rtol=1e-3, strict=True
        ),
    )
