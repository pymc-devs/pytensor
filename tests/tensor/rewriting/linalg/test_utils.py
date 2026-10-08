from pytensor import tensor as pt
from pytensor.graph.fg import FunctionGraph
from pytensor.tensor.rewriting.linalg.utils import (
    clients_through_padding,
    rebroadcast_like,
    strip_left_expand_dims,
)
from pytensor.tensor.type import matrix, tensor
from tests.unittest_tools import assert_equal_computations


def test_strip_left_expand_dims():
    X = matrix("X")

    assert strip_left_expand_dims(X) == (X, False)
    assert strip_left_expand_dims(pt.expand_dims(X, 0)) == (X, False)
    assert strip_left_expand_dims(pt.expand_dims(X, (0, 1))) == (X, False)
    assert strip_left_expand_dims(pt.expand_dims(pt.expand_dims(X, 0), 0)) == (X, False)

    assert strip_left_expand_dims(X.mT) == (X, True)
    assert strip_left_expand_dims(pt.expand_dims(X.mT, 0)) == (X, True)
    assert strip_left_expand_dims(pt.expand_dims(X, 0).mT) == (X, True)
    assert strip_left_expand_dims(pt.expand_dims(X.mT, 0).mT) == (X, False)

    fused_pad_and_transpose = X.dimshuffle("x", 1, 0)
    assert strip_left_expand_dims(fused_pad_and_transpose) == (X, True)

    right_padded = pt.expand_dims(X, -1)
    assert strip_left_expand_dims(right_padded) == (right_padded, False)


def test_clients_through_padding():
    X = matrix("X")
    y = tensor("y", shape=(None, None, None))

    direct = pt.exp(X)
    once = y - pt.expand_dims(X, 0)
    twice = pt.sqrt(pt.expand_dims(pt.expand_dims(X, 0), 0))
    right_padded = pt.expand_dims(X, -1)
    fgraph = FunctionGraph(outputs=[direct, once, twice, right_padded], clone=False)

    clients = {
        (client.outputs[0], idx) for client, idx in clients_through_padding(fgraph, X)
    }
    assert clients == {(direct, 0), (once, 1), (twice, 0), (right_padded, 0)}


def test_rebroadcast_like():
    X = matrix("X")

    assert rebroadcast_like(X, X) is X

    padded = pt.expand_dims(X, 0)
    assert_equal_computations([rebroadcast_like(X, padded)], [padded])
    assert_equal_computations([rebroadcast_like(padded, X)], [padded.squeeze(0)])

    batched = tensor("batched", shape=(5, None, None))
    recovered = rebroadcast_like(X, batched)
    assert recovered.type.ndim == 3
    assert batched.type.is_super(recovered.type)

    int_X = matrix("int_X", dtype="int64")
    assert rebroadcast_like(int_X, X).type.dtype == X.type.dtype
