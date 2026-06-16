import numpy as np
import pytest

import pytensor
import pytensor.tensor as pt
from pytensor.tensor import tensor
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.math import Dot
from tests.link.mlx.test_basic import compare_mlx_and_py, mlx_mode, py_mode


matmul = Blockwise(Dot(), signature="(i,j),(j,k)->(i,k)")
odd_matmul = Blockwise(Dot(), signature="(i00,i01),(i10,i11)->(o00,o01)")


def _spd_batch(rng, batch):
    """A batch of symmetric positive-definite matrices for Cholesky."""
    a = rng.standard_normal((*batch, 3, 3))
    return a @ np.swapaxes(a, -1, -2) + 3 * np.eye(3)


# Core ops with distinct gufunc signatures, each built for a leading batch shape:
# two rank-2 inputs, one rank-2 input, two rank-1 inputs.
def _matmul_graph(rng, batch):
    a = tensor("a", shape=(*batch, 2, 3))
    b = tensor("b", shape=(*batch, 3, 4))
    values = [rng.standard_normal((*batch, 2, 3)), rng.standard_normal((*batch, 3, 4))]
    return [a, b], matmul(a, b), values


def _cholesky_graph(rng, batch):
    m = tensor("m", shape=(*batch, 3, 3))
    return [m], pt.linalg.cholesky(m), [_spd_batch(rng, batch)]


def _convolve_graph(rng, batch):
    v = tensor("v", shape=(*batch, 16))
    k = tensor("k", shape=(*batch, 5))
    values = [rng.standard_normal((*batch, 16)), rng.standard_normal((*batch, 5))]
    return [v, k], pt.signal.convolve1d(v, k, mode="valid"), values


@pytest.mark.parametrize(
    "build",
    [_matmul_graph, _cholesky_graph, _convolve_graph],
    ids=["matmul", "cholesky", "convolve1d"],
)
@pytest.mark.parametrize("batch", [(5,), (2, 3)], ids=["single_batch", "nested_batch"])
def test_blockwise_signatures(build, batch):
    rng = np.random.default_rng(7)
    inputs, out, values = build(rng, batch)

    assert isinstance(out.owner.op, Blockwise)
    compare_mlx_and_py(inputs, [out], values)


@pytest.mark.parametrize(
    "a_shape, b_shape",
    [
        ((2, 3), (3, 4)),  # no batch dims -> core function, no vmap
        ((5, 2, 3), (3, 4)),  # one input unbatched -> broadcast over batch
        ((2, 1, 2, 3), (1, 3, 3, 4)),  # size-1 batch dims on different axes
        ((1, 2, 3), (1, 3, 4)),  # all batch dims size-1 -> squeeze + expand
    ],
    ids=["no_batch", "broadcast_unbatched", "cross_broadcast", "all_broadcast"],
)
def test_blockwise_batch_broadcasting(a_shape, b_shape):
    rng = np.random.default_rng(7)
    a = tensor("a", shape=a_shape)
    b = tensor("b", shape=b_shape)
    out = matmul(a, b)

    assert isinstance(out.owner.op, Blockwise)
    compare_mlx_and_py(
        [a, b], [out], [rng.standard_normal(a_shape), rng.standard_normal(b_shape)]
    )


def test_blockwise_no_runtime_broadcast():
    rng = np.random.default_rng(7)
    a = tensor("a", shape=(None, 2, 3))
    b = tensor("b", shape=(5, 3, 4))
    out = matmul(a, b)

    assert isinstance(out.owner.op, Blockwise)
    values = [rng.standard_normal((1, 2, 3)), rng.standard_normal((5, 3, 4))]

    py_fn = pytensor.function([a, b], out, mode=py_mode)
    with pytest.raises(ValueError, match="Runtime broadcasting not allowed"):
        py_fn(*values)

    mlx_fn = pytensor.function([a, b], out, mode=mlx_mode)
    with pytest.raises(ValueError, match="Runtime broadcasting not allowed"):
        mlx_fn(*values)


@pytest.mark.parametrize("batch", [(), (5,)], ids=["no_batch", "single_batch"])
def test_blockwise_fallback_signature(batch):
    rng = np.random.default_rng(7)
    a = tensor("a", shape=(*batch, 2, 3))
    b = tensor("b", shape=(*batch, 3, 4))
    out = odd_matmul(a, b)

    assert isinstance(out.owner.op, Blockwise)
    compare_mlx_and_py(
        [a, b],
        [out],
        [rng.standard_normal((*batch, 2, 3)), rng.standard_normal((*batch, 3, 4))],
    )


def test_blockwise_multi_output():
    rng = np.random.default_rng(7)
    x = tensor("x", shape=(1, 4, 4))
    q, r = pt.linalg.qr(x, mode="economic")

    assert isinstance(q.owner.op, Blockwise)
    compare_mlx_and_py(
        graph_inputs=[x],
        graph_outputs=[q, r],
        test_inputs=[rng.standard_normal((1, 4, 4))],
    )
