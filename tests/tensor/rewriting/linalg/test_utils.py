from pytensor import tensor as pt
from pytensor.graph.fg import FunctionGraph
from pytensor.tensor.rewriting.linalg.utils import (
    clients_through_expand_dims,
    rebroadcast_like,
    strip_left_expand_dims_and_transpose,
)
from pytensor.tensor.type import matrix, tensor
from tests.unittest_tools import assert_equal_computations


def test_strip_left_expand_dims_and_transpose():
    X = matrix("X")

    assert strip_left_expand_dims_and_transpose(X) == (X, False)
    assert strip_left_expand_dims_and_transpose(pt.expand_dims(X, 0)) == (X, False)
    assert strip_left_expand_dims_and_transpose(pt.expand_dims(X, (0, 1))) == (X, False)

    assert strip_left_expand_dims_and_transpose(X.mT) == (X, True)

    # The single peel handles the fused expand_dims + transpose DimShuffle that
    # local_dimshuffle_lift leaves behind after merging adjacent DimShuffles
    fused_expand_and_transpose = X.dimshuffle("x", 1, 0)
    assert strip_left_expand_dims_and_transpose(fused_expand_and_transpose) == (X, True)

    right_expanded = pt.expand_dims(X, -1)
    assert strip_left_expand_dims_and_transpose(right_expanded) == (
        right_expanded,
        False,
    )


def test_clients_through_expand_dims():
    X = matrix("X")
    y = tensor("y", shape=(None, None, None))

    direct = pt.exp(X)
    once = y - pt.expand_dims(X, 0)
    twice = pt.sqrt(pt.expand_dims(pt.expand_dims(X, 0), 0))
    right_expanded = pt.expand_dims(X, -1)
    fgraph = FunctionGraph(outputs=[direct, once, twice, right_expanded], clone=False)

    clients = {
        (client.outputs[0], idx)
        for client, idx in clients_through_expand_dims(fgraph, X)
    }
    assert clients == {(direct, 0), (once, 1), (twice, 0), (right_expanded, 0)}


def test_rebroadcast_like():
    X = matrix("X")

    assert rebroadcast_like(X, X) is X

    expanded = pt.expand_dims(X, 0)
    assert_equal_computations([rebroadcast_like(X, expanded)], [expanded])
    assert_equal_computations([rebroadcast_like(expanded, X)], [expanded.squeeze(0)])

    batched = tensor("batched", shape=(5, None, None))
    recovered = rebroadcast_like(X, batched)
    assert recovered.type.ndim == 3
    assert batched.type.is_super(recovered.type)

    int_X = matrix("int_X", dtype="int64")
    assert rebroadcast_like(int_X, X).type.dtype == X.type.dtype
