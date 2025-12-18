from pytensor.graph import FunctionGraph, rewrite_graph
from pytensor.tensor.reshape import JoinDims, SplitDims, join_dims, split_dims
from pytensor.tensor.shape import Reshape
from pytensor.tensor.type import tensor


def test_local_split_dims_to_reshape():
    x = tensor("x", shape=(2, 10, 3))
    x_split = split_dims(x, axis=1, shape=(2, 5, 1))

    fg = FunctionGraph(inputs=[x], outputs=[x_split])

    assert sum([1 for node in fg.toposort() if isinstance(node.op, SplitDims)]) == 1

    rewrite_graph(fg, include=("canonicalize",))

    assert sum([1 for node in fg.toposort() if isinstance(node.op, SplitDims)]) == 0
    assert sum([1 for node in fg.toposort() if isinstance(node.op, Reshape)]) == 1
    assert fg.outputs[0].type.shape == (2, 2, 5, 1, 3)


def test_local_join_dims_to_reshape():
    x = tensor("x", shape=(2, 2, 5, 1, 3))
    x_join = join_dims(x, axis=(1, 2, 3))

    fg = FunctionGraph(inputs=[x], outputs=[x_join])

    assert sum([1 for node in fg.toposort() if isinstance(node.op, JoinDims)]) == 1

    rewrite_graph(fg, include=("canonicalize",))

    assert sum([1 for node in fg.toposort() if isinstance(node.op, JoinDims)]) == 0
    assert sum([1 for node in fg.toposort() if isinstance(node.op, Reshape)]) == 1
    assert fg.outputs[0].type.shape == (2, 10, 3)
