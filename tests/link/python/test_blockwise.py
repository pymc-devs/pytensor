import numpy as np
import pytest

import pytensor.tensor as pt
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.linalg.decomposition.cholesky import Cholesky
from tests.link.python.test_basic import compare_python_and_perform


def _has_cholesky(fgraph):
    # The useless-Blockwise rewrite leaves a bare Cholesky for unbatched inputs.
    for node in fgraph.apply_nodes:
        op = node.op
        if isinstance(op, Cholesky) or (
            isinstance(op, Blockwise) and isinstance(op.core_op, Cholesky)
        ):
            return True
    return False


def _pd_matrices(shape, seed=0):
    rng = np.random.default_rng(seed)
    B = rng.standard_normal(shape)
    A = B @ np.swapaxes(B, -1, -2) + shape[-1] * np.eye(shape[-1])
    return A.astype("float64")


@pytest.mark.parametrize("lower", [True, False])
@pytest.mark.parametrize("shape", [(3, 3), (4, 3, 3), (2, 5, 3, 3)])
def test_cholesky_dispatch(shape, lower):
    A = pt.tensor("A", shape=(None,) * len(shape))
    out = pt.linalg.cholesky(A, lower=lower)
    fn, _ = compare_python_and_perform([A], out, [_pd_matrices(shape)])
    assert _has_cholesky(fn.maker.fgraph)


@pytest.mark.parametrize("lower", [True, False])
@pytest.mark.parametrize("b_shape", [(4,), (4, 2)])
@pytest.mark.parametrize("batch", [(), (3,)])
def test_solve_triangular_dispatch(batch, b_shape, lower):
    rng = np.random.default_rng(1)
    Av = np.tril(rng.standard_normal((*batch, 4, 4))) + 4 * np.eye(4)
    if not lower:
        Av = np.swapaxes(Av, -1, -2)
    bv = rng.standard_normal((*batch, *b_shape))
    A = pt.tensor("A", shape=(None,) * Av.ndim)
    b = pt.tensor("b", shape=(None,) * bv.ndim)
    out = pt.linalg.solve_triangular(A, b, lower=lower, b_ndim=len(b_shape))
    compare_python_and_perform(
        [A, b], out, [Av.astype("float64"), bv.astype("float64")]
    )


@pytest.mark.parametrize("pivoting", [False, True])
@pytest.mark.parametrize("mode", ["full", "economic", "r", "raw"])
@pytest.mark.parametrize("shape", [(4, 3), (3, 4), (5, 3, 4)])
def test_qr_dispatch(shape, mode, pivoting):
    rng = np.random.default_rng(0)
    A = pt.tensor("A", shape=(None,) * len(shape))
    out = pt.linalg.qr(A, mode=mode, pivoting=pivoting)
    compare_python_and_perform([A], out, [rng.standard_normal(shape)])


def test_blockwise_falls_back_without_core_dispatch():
    # The general Solve has no python_funcify dispatch, so Blockwise must fall
    # back to its (vectorized) perform and still match the reference.
    A = pt.matrix("A")
    b = pt.vector("b")
    out = pt.linalg.solve(A, b)
    rng = np.random.default_rng(2)
    Av = rng.standard_normal((4, 4)) + 4 * np.eye(4)
    bv = rng.standard_normal(4)
    compare_python_and_perform([A, b], out, [Av, bv])
