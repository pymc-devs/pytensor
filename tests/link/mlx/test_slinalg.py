import contextlib

import numpy as np
import pytest

import pytensor.tensor as pt
from pytensor import config
from tests.link.mlx.test_basic import compare_mlx_and_py, mlx_mode


# mlx complains about useless vmap (when there are no batch dims), so we need to include
# local_remove_useless_blockwise rewrite for these tests
mlx_linalg_mode = mlx_mode.including("blockwise")


@pytest.mark.parametrize("lower", [True, False])
def test_mlx_cholesky(lower):
    rng = np.random.default_rng()

    A = pt.tensor("A", shape=(5, 5))
    A_val = rng.normal(size=(5, 5)).astype(config.floatX)
    A_val = A_val @ A_val.T

    out = pt.linalg.cholesky(A, lower=lower)

    compare_mlx_and_py(
        [A],
        [out],
        [A_val],
        mlx_mode=mlx_linalg_mode,
    )


@pytest.mark.parametrize("assume_a", ["gen", "pos"])
def test_mlx_solve(assume_a):
    rng = np.random.default_rng()

    A = pt.tensor("A", shape=(5, 5))
    b = pt.tensor("B", shape=(5, 5))

    out = pt.linalg.solve(A, b, b_ndim=2, assume_a=assume_a)

    A_val = rng.normal(size=(5, 5)).astype(config.floatX)
    A_val = A_val @ A_val.T

    b_val = rng.normal(size=(5, 5)).astype(config.floatX)

    context = (
        contextlib.suppress()
        if assume_a == "gen"
        else pytest.warns(
            UserWarning, match=f"MLX solve does not support assume_a={assume_a}"
        )
    )

    with context:
        compare_mlx_and_py([A, b], [out], [A_val, b_val], mlx_mode=mlx_linalg_mode)
