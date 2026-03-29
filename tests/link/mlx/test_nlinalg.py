from functools import partial

import numpy as np
import pytest

import pytensor.tensor as pt
from pytensor import config
from pytensor.compile.mode import get_mode
from tests.link.mlx.test_basic import compare_mlx_and_py, mlx_mode


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


def test_mlx_eig():
    rng = np.random.default_rng(15)

    M = rng.normal(size=(3, 3))
    A_val = (M @ M.T).astype(config.floatX)

    A = pt.matrix(name="A")
    outs = pt.linalg.eig(A)

    compare_mlx_and_py([A], outs, [A_val])


@pytest.mark.parametrize("UPLO", ["L", "U"])
def test_mlx_eigh(UPLO):
    rng = np.random.default_rng(15)

    M = rng.normal(size=(3, 3))
    A_val = (M @ M.T).astype(config.floatX)

    A = pt.matrix(name="A")
    outs = pt.linalg.eigh(A, UPLO=UPLO)

    compare_mlx_and_py([A], outs, [A_val])


@pytest.mark.parametrize("compute_uv", [True, False])
def test_mlx_svd(compute_uv):
    rng = np.random.default_rng(15)

    A = pt.matrix(name="X")
    A_val = rng.normal(size=(3, 3)).astype(config.floatX)
    A_val = A_val @ A_val.T

    out = pt.linalg.svd(A, compute_uv=compute_uv)

    compare_mlx_and_py(
        [A],
        out,
        [A_val],
        mlx_mode=mlx_mode,
        assert_fn=partial(np.testing.assert_allclose, atol=1e-6, strict=True),
    )


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


@pytest.mark.parametrize("op", [pt.linalg.inv, pt.linalg.pinv], ids=["inv", "pinv"])
def test_mlx_inv(op):
    rng = np.random.default_rng(15)
    n = 3

    A = pt.matrix(name="A")
    A_val = rng.normal(size=(n, n))
    A_val = (A_val @ A_val.T).astype(config.floatX)

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
