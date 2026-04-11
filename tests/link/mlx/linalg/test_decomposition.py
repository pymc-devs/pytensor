from functools import partial

import numpy as np
import pytest
from packaging.version import parse as V

import pytensor.tensor as pt
from pytensor import config
from pytensor.tensor.linalg.decomposition import lu, svd
from pytensor.tensor.linalg.decomposition.cholesky import cholesky
from pytensor.tensor.type_other import NoneConst
from tests.link.mlx.test_basic import compare_mlx_and_py, mlx_mode


mx = pytest.importorskip("mlx.core")


@pytest.mark.skipif(
    V(mx.__version__) < V("0.30.1"),
    reason="mx.linalg.eig causes a Fatal Python error (Abort trap) on MLX <0.30.1 "
    "(maybe -- the exact version cutoff is unknown)",
)
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


@pytest.mark.parametrize("lower", [True, False])
def test_mlx_eigvalsh(lower):
    rng = np.random.default_rng(15)

    M = rng.normal(size=(3, 3))
    A_val = (M @ M.T).astype(config.floatX)

    A = pt.matrix(name="A")
    B = pt.matrix(name="B")

    out_with_b = pt.linalg.eigvalsh(A, B, lower=lower)
    with pytest.raises(NotImplementedError):
        compare_mlx_and_py([A, B], [out_with_b], [A_val, A_val])

    out_no_b = pt.linalg.eigvalsh(A, NoneConst, lower=lower)
    compare_mlx_and_py([A], [out_no_b], [A_val])


def test_mlx_lu_factor():
    rng = np.random.default_rng(15)

    A = pt.matrix(name="A")
    A_val = rng.normal(size=(5, 5)).astype(config.floatX)

    out = pt.linalg.lu_factor(A)

    compare_mlx_and_py([A], out, [A_val])


def test_mlx_pivot_to_permutations():
    rng = np.random.default_rng(15)

    A = pt.matrix(name="A")
    A_val = rng.normal(size=(5, 5)).astype(config.floatX)

    from pytensor.tensor.linalg.decomposition.lu import pivot_to_permutation

    lu_and_pivots = pt.linalg.lu_factor(A)
    out = pivot_to_permutation(lu_and_pivots[1])

    compare_mlx_and_py([A], [out], [A_val])


@pytest.mark.parametrize("mode", ["economic", "r"])
def test_mlx_qr(mode):
    rng = np.random.default_rng(15)

    A = pt.matrix(name="A")
    A_val = rng.normal(size=(5, 3)).astype(config.floatX)

    out = pt.linalg.qr(A, mode=mode)

    compare_mlx_and_py([A], out, [A_val])
