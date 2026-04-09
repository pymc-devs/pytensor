import numpy as np
import pytest
import scipy

from pytensor import function
from pytensor.configdefaults import config
from pytensor.tensor.linalg import lu, lu_factor, pivot_to_permutation
from pytensor.tensor.type import matrix, tensor
from tests import unittest_tools as utt


@pytest.mark.parametrize(
    "permute_l, p_indices",
    [(False, True), (True, False), (False, False)],
    ids=["PL", "p_indices", "P"],
)
@pytest.mark.parametrize("complex", [False, True], ids=["real", "complex"])
@pytest.mark.parametrize("shape", [(3, 5, 5), (5, 5)], ids=["batched", "not_batched"])
def test_lu_decomposition(
    permute_l: bool, p_indices: bool, complex: bool, shape: tuple[int]
):
    dtype = config.floatX if not complex else f"complex{int(config.floatX[-2:]) * 2}"

    A = tensor("A", shape=shape, dtype=dtype)
    pt_out = lu(A, permute_l=permute_l, p_indices=p_indices)

    f = function([A], pt_out)

    rng = np.random.default_rng(utt.fetch_seed())
    x = rng.normal(size=shape).astype(config.floatX)
    if complex:
        x = x + 1j * rng.normal(size=shape).astype(config.floatX)

    out = f(x)
    for numerical_out, symbolic_out in zip(out, pt_out):
        assert numerical_out.dtype == symbolic_out.type.dtype

    if permute_l:
        PL, U = out
    elif p_indices:
        p, L, U = out
        if len(shape) == 2:
            P = np.eye(5)[p]
        else:
            P = np.stack([np.eye(5)[idx] for idx in p])
        PL = np.einsum("...nk,...km->...nm", P, L)
    else:
        P, L, U = out
        PL = np.einsum("...nk,...km->...nm", P, L)

    x_rebuilt = np.einsum("...nk,...km->...nm", PL, U)

    np.testing.assert_allclose(
        x,
        x_rebuilt,
        atol=1e-8 if config.floatX == "float64" else 1e-4,
        rtol=1e-8 if config.floatX == "float64" else 1e-4,
    )
    scipy_out = scipy.linalg.lu(x, permute_l=permute_l, p_indices=p_indices)

    for a, b in zip(out, scipy_out, strict=True):
        np.testing.assert_allclose(a, b)


@pytest.mark.parametrize(
    "grad_case", [0, 1, 2], ids=["dU_only", "dL_only", "dU_and_dL"]
)
@pytest.mark.parametrize(
    "permute_l, p_indices",
    [(True, False), (False, True), (False, False)],
    ids=["PL", "p_indices", "P"],
)
@pytest.mark.parametrize("shape", [(3, 5, 5), (5, 5)], ids=["batched", "not_batched"])
def test_lu_grad(grad_case, permute_l, p_indices, shape):
    rng = np.random.default_rng(utt.fetch_seed())
    A_value = rng.normal(size=shape).astype(config.floatX)

    def f_pt(A):
        # lu returns either (P_or_index, L, U) or (PL, U), depending on settings
        out = lu(A, permute_l=permute_l, p_indices=p_indices, check_finite=False)

        match grad_case:
            case 0:
                return out[-1].sum()
            case 1:
                return out[-2].sum()
            case 2:
                return out[-1].sum() + out[-2].sum()

    utt.verify_grad(f_pt, [A_value], rng=rng)


@pytest.mark.parametrize("inverse", [True, False], ids=["inverse", "no_inverse"])
def test_pivot_to_permutation(inverse):
    rng = np.random.default_rng(utt.fetch_seed())
    A_val = rng.normal(size=(5, 5))
    _, pivots = scipy.linalg.lu_factor(A_val)
    perm_idx, *_ = scipy.linalg.lu(A_val, p_indices=True)

    if not inverse:
        perm_idx_pt = pivot_to_permutation(pivots, inverse=False).eval()
        np.testing.assert_array_equal(perm_idx_pt, perm_idx)
    else:
        p_inv_pt = pivot_to_permutation(pivots, inverse=True).eval()
        np.testing.assert_array_equal(p_inv_pt, np.argsort(perm_idx))


def test_lu_factor():
    rng = np.random.default_rng(utt.fetch_seed())
    A = matrix()
    A_val = rng.normal(size=(5, 5)).astype(config.floatX)

    f = function([A], lu_factor(A))

    LU, pt_p_idx = f(A_val)
    sp_LU, sp_p_idx = scipy.linalg.lu_factor(A_val)

    np.testing.assert_allclose(LU, sp_LU)
    np.testing.assert_allclose(pt_p_idx, sp_p_idx)

    utt.verify_grad(
        lambda A: lu_factor(A)[0].sum(),
        [A_val],
        rng=rng,
    )


def test_lu_factor_empty():
    A = matrix()
    f = function([A], lu_factor(A))

    A_empty = np.empty([0, 0], dtype=config.floatX)
    LU, pt_p_idx = f(A_empty)

    assert LU.size == 0
    assert LU.dtype == config.floatX
    assert pt_p_idx.size == 0
