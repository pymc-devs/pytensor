from functools import partial

import numpy as np
import pytest
from packaging.version import parse as V

import pytensor.tensor as pt
from pytensor import config
from pytensor.tensor.linalg.decomposition import lu, svd
from pytensor.tensor.linalg.decomposition.cholesky import cholesky
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


@pytest.mark.parametrize("lower", [True, False])
def test_mlx_eigh(lower):
    rng = np.random.default_rng(15)

    M = rng.normal(size=(3, 3))
    A_val = (M @ M.T).astype(config.floatX)

    A = pt.matrix(name="A")
    outs = pt.linalg.eigh(A, lower=lower, driver="evd")

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

    out_no_b = pt.linalg.eigvalsh(A, None, lower=lower)

    # Pytensor uses d
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


@pytest.mark.parametrize("batch_shape", [(4,), (2, 3)], ids=["batch", "batch_2d"])
def test_mlx_lu_factor_batched(batch_shape):
    """`mx.vmap` has no rule for the `LUF` primitive `lu_factor` builds (#2385)."""
    rng = np.random.default_rng(15)
    n = 5

    A = pt.tensor("A", shape=(*batch_shape, n, n))
    A_val = rng.normal(size=(*batch_shape, n, n)).astype(config.floatX)

    compare_mlx_and_py([A], pt.linalg.lu_factor(A), [A_val], mlx_mode=mlx_mode)


@pytest.mark.parametrize("inverse", [True, False], ids=["inverse", "forward"])
@pytest.mark.parametrize("batch_shape", [(4,), (2, 3)], ids=["batch", "batch_2d"])
def test_mlx_pivot_to_permutations_batched(batch_shape, inverse):
    """Coverage for the batched pivot scan reached by a batched `lu_solve`.

    The core dispatch loops in Python over ``pivots.shape[0]``, which is the
    *core* length under `mx.vmap`, so this path is already correct -- pinning it
    so the batched `lu_solve` fix cannot regress it. Built directly because
    `pivot_to_permutation` itself only accepts a 1-d input.
    """
    from pytensor.tensor.blockwise import Blockwise
    from pytensor.tensor.linalg.decomposition.lu import PivotToPermutations

    rng = np.random.default_rng(15)
    n = 5

    pivots = pt.tensor("pivots", shape=(*batch_shape, n), dtype="int32")
    # LAPACK-style pivots: entry i may only reference row i or later.
    pivots_val = (
        np.stack(
            [
                np.array([rng.integers(i, n) for i in range(n)])
                for _ in range(int(np.prod(batch_shape)))
            ]
        )
        .reshape(*batch_shape, n)
        .astype("int32")
    )

    out = Blockwise(PivotToPermutations(inverse=inverse))(pivots)

    compare_mlx_and_py([pivots], [out], [pivots_val], mlx_mode=mlx_mode)
