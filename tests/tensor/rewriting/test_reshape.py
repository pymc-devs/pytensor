import numpy as np

import pytensor as pt
from pytensor.graph import FunctionGraph, rewrite_graph
from pytensor.tensor import shape
from pytensor.tensor.basic import expand_dims
from pytensor.tensor.extra_ops import squeeze
from pytensor.tensor.reshape import JoinDims, SplitDims, join_dims, split_dims
from pytensor.tensor.shape import Reshape, specify_shape
from pytensor.tensor.type import tensor
from tests.unittest_tools import assert_equal_computations


def test_local_split_dims():
    """Test that split_dims converts to reshape for general shapes."""
    x = tensor("x", shape=(2, 10, 3))
    x_split = split_dims(x, axis=1, shape=(2, 5, 1))

    fg = FunctionGraph(inputs=[x], outputs=[x_split])

    rewrite_graph(
        fg,
        include=("canonicalize",),
        exclude=(
            "local_subtensor_merge",
            "local_subtensor_remove_broadcastable_index",
        ),
    )

    # Build the expected computation manually
    x_shape = shape(x)

    # 1. Build shape vector for reshape: (x.shape[0], 2, 5, x.shape[2]) = (2, 2, 5, 3)
    # The split shape (2, 5, 1) has the 1 removed for reshape, then expand_dims adds it back
    shape_vector = pt.tensor.stack([x_shape[0], np.int64(2), np.int64(5), x_shape[2]])

    # 2. Replicate the Reshape and ExpandDims
    reshaped = Reshape(4)(x, shape_vector)
    expanded = expand_dims(reshaped, axis=3)

    # 3. SpecifyShape to lock in the output shape
    expected_shape_tuple = (2, 2, 5, 1, 3)
    expected = specify_shape(expanded, expected_shape_tuple)

    assert_equal_computations(
        [fg.outputs[0]], [expected], in_xs=[fg.outputs[0]], in_ys=[expected]
    )


def test_local_join_dims():
    x = tensor("x", shape=(2, 2, 5, 1, 3))
    x_join = join_dims(x, start_axis=1, n_axes=3)

    fg = FunctionGraph(inputs=[x], outputs=[x_join])

    rewrite_graph(
        fg,
        include=("canonicalize",),
        exclude=(
            "local_subtensor_merge",
            "local_subtensor_remove_broadcastable_index",
        ),
    )

    expected = x.reshape((2, 10, 3))
    assert_equal_computations(
        [fg.outputs[0]], [expected], in_xs=[fg.outputs[0]], in_ys=[expected]
    )
    # expected = x.reshape((2, 10, 3))
    # assert_equal_computations([fg.outputs[0]], [expected], in_xs=[fg.outputs[0]], in_ys=[expected])


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
    assert_equal_computations([fg.outputs[0]], [x], in_xs=[fg.outputs[0]], in_ys=[x])


def test_local_join_dims_expand():
    """Test that join_dims with n_axes=0 becomes expand_dims."""
    x = tensor("x", shape=(2, 3, 4))
    x_join = join_dims(x, start_axis=1, n_axes=0)

    fg = FunctionGraph(inputs=[x], outputs=[x_join])

    # Before rewrite: should have 1 JoinDims node
    assert sum([1 for node in fg.toposort() if isinstance(node.op, JoinDims)]) == 1

    rewrite_graph(fg, include=("canonicalize",))

    # Output shape should be (2, 1, 3, 4) - new dimension of size 1 inserted at axis 1
    expected = expand_dims(x, axis=1)
    assert_equal_computations(
        [fg.outputs[0]], [expected], in_xs=[fg.outputs[0]], in_ys=[expected]
    )


def test_local_split_dims_to_reshape_squeeze_case():
    """Test that split_dims with shape tensor of static shape (0,) becomes squeeze via merged rewrite."""
    x = tensor("x", shape=(2, 1, 3, 4))
    # Create a tensor variable with static shape (0,)
    empty_shape_var = tensor("empty_shape", shape=(0,), dtype="int32")
    x_split = split_dims(x, axis=1, shape=empty_shape_var)

    fg = FunctionGraph(inputs=[x, empty_shape_var], outputs=[x_split])

    # Before rewrite: should have 1 SplitDims node
    assert sum([1 for node in fg.toposort() if isinstance(node.op, SplitDims)]) == 1

    rewrite_graph(fg, include=("canonicalize",))

    # Output shape should be (2, 3, 4) - dimension 1 removed
    expected = squeeze(x, axis=1)
    assert_equal_computations(
        [fg.outputs[0]], [expected], in_xs=[fg.outputs[0]], in_ys=[expected]
    )
