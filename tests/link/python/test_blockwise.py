import numpy as np
import pytest
import scipy.linalg as sla

import pytensor
import pytensor.tensor as pt
from pytensor.link.python.dispatch.basic import python_funcify
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.linalg.decomposition.cholesky import Cholesky
from tests.link.python.test_basic import compare_py_and_pyjit


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
    Av = _pd_matrices(shape)
    fn, factor = compare_py_and_pyjit([A], out, [Av])
    assert _has_cholesky(fn.maker.fgraph)

    assert factor.shape == Av.shape
    assert factor.dtype == out.type.dtype
    if lower:
        np.testing.assert_allclose(factor, np.tril(factor), atol=1e-12)
        np.testing.assert_allclose(factor @ factor.mT, Av)
    else:
        np.testing.assert_allclose(factor, np.triu(factor), atol=1e-12)
        np.testing.assert_allclose(factor.mT @ factor, Av)


@pytest.mark.parametrize("lower", [True, False])
@pytest.mark.parametrize("b_shape", [(4,), (4, 2)])
@pytest.mark.parametrize("batch", [(), (3,)])
def test_solve_triangular_dispatch(batch, b_shape, lower):
    rng = np.random.default_rng(1)
    Av = np.tril(rng.standard_normal((*batch, 4, 4))) + 4 * np.eye(4)
    if not lower:
        Av = Av.mT
    bv = rng.standard_normal((*batch, *b_shape))
    A = pt.tensor("A", shape=(None,) * Av.ndim)
    b = pt.tensor("b", shape=(None,) * bv.ndim)
    out = pt.linalg.solve_triangular(A, b, lower=lower, b_ndim=len(b_shape))
    _, x = compare_py_and_pyjit([A, b], out, [Av, bv])

    assert x.shape == bv.shape
    assert x.dtype == out.type.dtype
    if len(b_shape) == 1:
        np.testing.assert_allclose(np.einsum("...ij,...j->...i", Av, x), bv)
    else:
        np.testing.assert_allclose(Av @ x, bv)


@pytest.mark.parametrize("pivoting", [False, True])
@pytest.mark.parametrize("mode", ["full", "economic", "r"])
@pytest.mark.parametrize("shape", [(4, 3), (3, 4), (5, 3, 4)])
def test_qr_dispatch(shape, mode, pivoting):
    rng = np.random.default_rng(0)
    A = pt.tensor("A", shape=(None,) * len(shape))
    outputs = pt.linalg.qr(A, mode=mode, pivoting=pivoting)
    if not isinstance(outputs, list):
        outputs = [outputs]
    Av = rng.standard_normal(shape)
    _, results = compare_py_and_pyjit([A], outputs, [Av])

    if pivoting:
        *factors, jpvt = results
        assert jpvt.shape == (*shape[:-2], shape[-1])
        # A permutation, not just any index vector.
        np.testing.assert_array_equal(
            np.sort(jpvt, axis=-1),
            np.broadcast_to(np.arange(shape[-1]), jpvt.shape),
        )
        Av = np.take_along_axis(Av, jpvt[..., None, :], axis=-1)
    else:
        factors = results

    if mode == "r":
        [R] = factors
        np.testing.assert_allclose(R, np.triu(R), atol=1e-12)
        # Q drops out of the normal equations, so R alone is pinned by them.
        np.testing.assert_allclose(R.mT @ R, Av.mT @ Av, atol=1e-10)
        return

    Q, R = factors
    np.testing.assert_allclose(R, np.triu(R), atol=1e-12)
    np.testing.assert_allclose(Q @ R, Av, atol=1e-10)
    gram = Q.mT @ Q
    np.testing.assert_allclose(
        gram, np.broadcast_to(np.eye(Q.shape[-1]), gram.shape), atol=1e-10
    )


@pytest.mark.parametrize("pivoting", [False, True])
@pytest.mark.parametrize("shape", [(4, 3), (3, 4), (5, 3, 4)])
def test_qr_raw_dispatch(shape, pivoting):
    # `raw` returns the packed Householder reflectors, whose layout is a scipy
    # convention rather than a mathematical property, so scipy is the reference.
    rng = np.random.default_rng(0)
    A = pt.tensor("A", shape=(None,) * len(shape))
    out = pt.linalg.qr(A, mode="raw", pivoting=pivoting)
    Av = rng.standard_normal(shape)
    _, results = compare_py_and_pyjit([A], out, [Av])

    for index in np.ndindex(shape[:-2]):
        (factor, tau), *rest = sla.qr(Av[index], mode="raw", pivoting=pivoting)
        for got, expected in zip(results, [factor, tau, *rest], strict=True):
            np.testing.assert_allclose(got[index], expected)


@pytest.mark.parametrize("mode", ["PYTHON", "PYJIT"])
def test_blockwise_rejects_runtime_broadcast(mode):
    # A batch dimension declared non-broadcastable must not be broadcast at
    # runtime, even though `np.vectorize` would happily do it.
    A = pt.tensor("A", shape=(None, 3, 3))
    b = pt.tensor("b", shape=(None, 3))
    out = pt.linalg.solve_triangular(A, b, lower=True, b_ndim=1)
    Av = (np.tril(np.ones((3, 3))) + 3 * np.eye(3))[None]
    bv = np.ones((4, 3))

    fn = pytensor.function([A, b], out, mode=mode)
    with pytest.raises(ValueError, match="Runtime broadcasting not allowed"):
        fn(Av, bv)

    # The same graph still solves once the batch lengths agree.
    Av_batched = np.repeat(Av, 4, axis=0)
    x = fn(Av_batched, bv)
    np.testing.assert_allclose(np.einsum("...ij,...j->...i", Av_batched, x), bv)


@pytest.mark.parametrize("mode", ["PYTHON", "PYJIT"])
def test_empty_input_shortcut(mode):
    # Cholesky upcasts integer input; the empty-input shortcut must honour that.
    x = pt.matrix("x", dtype="int64")
    factor_out = pt.linalg.cholesky(x)
    factor = pytensor.function([x], factor_out, mode=mode)(
        np.zeros((0, 0), dtype="int64")
    )
    assert factor.dtype == factor_out.type.dtype
    assert factor.shape == (0, 0)

    A = pt.matrix("A", dtype="int64")
    b = pt.matrix("b", dtype="int64")
    solved_out = pt.linalg.solve_triangular(A, b, lower=True)
    solved = pytensor.function([A, b], solved_out, mode=mode)(
        np.zeros((0, 0), dtype="int64"), np.zeros((0, 2), dtype="int64")
    )
    assert solved.dtype == solved_out.type.dtype
    assert solved.shape == (0, 2)


@pytest.mark.parametrize("mode", ["PYTHON", "PYJIT"])
def test_cholesky_returns_nan_when_not_positive_definite(mode):
    x = pt.matrix("x")
    out = pt.linalg.cholesky(x)
    not_pd = np.array([[1.0, 2.0], [2.0, 1.0]])
    factor = pytensor.function([x], out, mode=mode)(not_pd)
    assert np.isnan(factor).all()


@pytest.mark.parametrize("mode", ["PYTHON", "PYJIT"])
def test_batched_cholesky_nans_only_the_failing_matrix(mode):
    # `np.vectorize` writes each core result into a shared output buffer, so a
    # single failing matrix must not poison its neighbours.
    x = pt.tensor("x", shape=(None, None, None))
    out = pt.linalg.cholesky(x, lower=True)
    batch = np.stack([np.eye(2) * 4.0, np.array([[1.0, 2.0], [2.0, 1.0]])])
    factor = pytensor.function([x], out, mode=mode)(batch)

    np.testing.assert_allclose(factor[0], np.eye(2) * 2.0)
    assert np.isnan(factor[1]).all()


@pytest.mark.parametrize("mode", ["PYTHON", "PYJIT"])
def test_solve_triangular_returns_nan_when_singular(mode):
    A = pt.matrix("A")
    b = pt.vector("b")
    out = pt.linalg.solve_triangular(A, b, lower=True)
    singular = np.array([[0.0, 0.0], [1.0, 1.0]])
    x = pytensor.function([A, b], out, mode=mode)(singular, np.array([1.0, 2.0]))
    assert np.isnan(x).all()


def test_blockwise_falls_back_without_core_dispatch():
    # The general Solve has no python_funcify dispatch, so Blockwise must fall
    # back to its (vectorized) perform.
    A = pt.tensor("A", shape=(None, None, None))
    b = pt.tensor("b", shape=(None, None))
    out = pt.linalg.solve(A, b, b_ndim=1)
    rng = np.random.default_rng(2)
    Av = rng.standard_normal((3, 4, 4)) + 4 * np.eye(4)
    bv = rng.standard_normal((3, 4))
    fn, x = compare_py_and_pyjit([A, b], out, [Av, bv])

    [node] = [
        node for node in fn.maker.fgraph.apply_nodes if isinstance(node.op, Blockwise)
    ]
    with pytest.raises(NotImplementedError):
        python_funcify(node.op, node=node)

    np.testing.assert_allclose(np.einsum("...ij,...j->...i", Av, x), bv)
