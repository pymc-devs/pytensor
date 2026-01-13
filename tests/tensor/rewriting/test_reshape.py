import numpy as np

from pytensor.graph import FunctionGraph, rewrite_graph
from pytensor.tensor.elemwise import DimShuffle
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
    x_join = join_dims(x, start_axis=1, n_axes=3)

    fg = FunctionGraph(inputs=[x], outputs=[x_join])

    assert sum([1 for node in fg.toposort() if isinstance(node.op, JoinDims)]) == 1

    rewrite_graph(fg, include=("canonicalize",))

    assert sum([1 for node in fg.toposort() if isinstance(node.op, JoinDims)]) == 0
    assert sum([1 for node in fg.toposort() if isinstance(node.op, Reshape)]) == 1
    assert fg.outputs[0].type.shape == (2, 10, 3)


def test_local_join_dims_noop():
    """Test that join_dims with n_axes=1 becomes identity (no-op)."""
    x = tensor("x", shape=(2, 3, 4))
    x_join = join_dims(x, start_axis=1, n_axes=1)

    fg = FunctionGraph(inputs=[x], outputs=[x_join])

    # Before rewrite: should have 1 JoinDims node
    assert sum([1 for node in fg.toposort() if isinstance(node.op, JoinDims)]) == 1

    rewrite_graph(fg, include=("canonicalize",))

    # After rewrite: should have 0 JoinDims nodes
    assert sum([1 for node in fg.toposort() if isinstance(node.op, JoinDims)]) == 0
    # Output should be equivalent to input (identity rewrite)
    # The rewrite returns the input variable, so output should match input shape/type
    assert fg.outputs[0].type.shape == x.type.shape
    assert fg.outputs[0].type.dtype == x.type.dtype
    assert fg.outputs[0].type.ndim == x.type.ndim


def test_local_join_dims_expand():
    """Test that join_dims with n_axes=0 becomes expand_dims."""
    x = tensor("x", shape=(2, 3, 4))
    x_join = join_dims(x, start_axis=1, n_axes=0)

    fg = FunctionGraph(inputs=[x], outputs=[x_join])

    # Before rewrite: should have 1 JoinDims node
    assert sum([1 for node in fg.toposort() if isinstance(node.op, JoinDims)]) == 1

    rewrite_graph(fg, include=("canonicalize",))

    # After rewrite: should have 0 JoinDims nodes
    assert sum([1 for node in fg.toposort() if isinstance(node.op, JoinDims)]) == 0
    # Should have 1 DimShuffle node with is_expand_dims=True
    expand_nodes = [
        node
        for node in fg.toposort()
        if isinstance(node.op, DimShuffle) and node.op.is_expand_dims
    ]
    assert len(expand_nodes) == 1
    # Output shape should be (2, 1, 3, 4) - new dimension of size 1 inserted at axis 1
    assert fg.outputs[0].type.shape == (2, 1, 3, 4)


def test_local_split_dims_squeeze():
    """Test that split_dims with shape=() becomes squeeze."""
    x = tensor("x", shape=(2, 1, 3, 4))
    # Create a constant empty shape array - split_dims will convert it to a tensor
    empty_shape = np.array([], dtype="int32")
    x_split = split_dims(x, axis=1, shape=empty_shape)

    fg = FunctionGraph(inputs=[x], outputs=[x_split])

    # Before rewrite: should have 1 SplitDims node
    assert sum([1 for node in fg.toposort() if isinstance(node.op, SplitDims)]) == 1

    rewrite_graph(fg, include=("canonicalize",))

    # After rewrite: should have 0 SplitDims nodes
    assert sum([1 for node in fg.toposort() if isinstance(node.op, SplitDims)]) == 0
    # Should have 1 DimShuffle node with is_squeeze=True
    squeeze_nodes = [
        node
        for node in fg.toposort()
        if isinstance(node.op, DimShuffle) and node.op.is_squeeze
    ]
    assert len(squeeze_nodes) == 1
    # Output shape should be (2, 3, 4) - dimension 1 removed
    assert fg.outputs[0].type.shape == (2, 3, 4)


# def test_local_split_dims_specify_shape():
#     """Test that split_dims with shape=(dim,) becomes specify_shape (when input shape is None)."""
#     # Create input with unknown shape at axis 1
#     x = tensor("x", shape=(2, None, 4))
#     # Create a constant shape with single dimension - split_dims will convert it to a tensor
#     dim_shape = np.array([5], dtype="int32")
#     x_split = split_dims(x, axis=1, shape=dim_shape)
#
#     fg = FunctionGraph(inputs=[x], outputs=[x_split])
#
#     # Before rewrite: should have 1 SplitDims node
#     assert sum([1 for node in fg.toposort() if isinstance(node.op, SplitDims)]) == 1
#
#     rewrite_graph(fg, include=("canonicalize",))
#
#     # After rewrite: should have 0 SplitDims nodes
#     assert sum([1 for node in fg.toposort() if isinstance(node.op, SplitDims)]) == 0
#     # Should have 1 SpecifyShape node
#     assert sum([1 for node in fg.toposort() if isinstance(node.op, SpecifyShape)]) == 1
#     # Output shape should be (2, 5, 4) - dimension 1 specified as 5
#     assert fg.outputs[0].type.shape == (2, 5, 4)
