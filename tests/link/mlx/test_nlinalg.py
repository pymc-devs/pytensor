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
    )


def test_mlx_kron():
    rng = np.random.default_rng()

    A = pt.matrix(name="A")
    B = pt.matrix(name="B")
    A_val, B_val = rng.normal(size=(2, 3, 3)).astype(config.floatX)
    out = pt.linalg.kron(A, B)

    compare_mlx_and_py(
        [A, B],
        [out],
        [A_val, B_val],
        mlx_mode=mlx_linalg_mode,
    )


@pytest.mark.parametrize("op", [pt.linalg.inv, pt.linalg.pinv], ids=["inv", "pinv"])
def test_mlx_inv(op):
    rng = np.random.default_rng()

    A = pt.matrix(name="A")
    A_val = rng.normal(size=(3, 3)).astype(config.floatX)
    A_val = A_val @ A_val.T

    out = op(A)

    compare_mlx_and_py(
        [A],
        [out],
        [A_val],
        mlx_mode=mlx_linalg_mode,
    )
