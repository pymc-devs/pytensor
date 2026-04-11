from functools import partial

import numpy as np
import pytest

import pytensor.tensor as pt
from pytensor import config
from pytensor.tensor.linalg.decomposition import lu, svd
from pytensor.tensor.linalg.decomposition.cholesky import cholesky
from tests.link.mlx.test_basic import compare_mlx_and_py, mlx_mode


@pytest.mark.parametrize("compute_uv", [True, False])
def test_mlx_svd(compute_uv):
    rng = np.random.default_rng(15)

    A = pt.matrix(name="X")
    A_val = rng.normal(size=(3, 3)).astype(config.floatX)
    A_val = A_val @ A_val.T

    out = svd.svd(A, compute_uv=compute_uv)

    compare_mlx_and_py(
        [A],
        out,
        [A_val],
        mlx_mode=mlx_mode,
        assert_fn=partial(np.testing.assert_allclose, atol=1e-6, strict=True),
    )


@pytest.mark.parametrize("lower", [True, False])
def test_mlx_cholesky(lower):
    rng = np.random.default_rng(15)
    n = 3

    A = pt.tensor("A", shape=(n, n))
    A_val = rng.normal(size=(n, n))
    A_val = (A_val @ A_val.T).astype(config.floatX)

    out = cholesky(A, lower=lower)

    compare_mlx_and_py(
        [A],
        [out],
        [A_val],
        mlx_mode=mlx_mode,
        assert_fn=partial(np.testing.assert_allclose, atol=1e-6, strict=True),
    )


def test_mlx_LU():
    rng = np.random.default_rng(15)

    A = pt.tensor("A", shape=(5, 5))
    out = lu.lu(A, permute_l=False, p_indices=True)

    A_val = rng.normal(size=(5, 5)).astype(config.floatX)

    compare_mlx_and_py(
        [A],
        out,
        [A_val],
        mlx_mode=mlx_mode,
        assert_fn=partial(np.testing.assert_allclose, atol=1e-6, strict=True),
    )
