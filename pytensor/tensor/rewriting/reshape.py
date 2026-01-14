from pytensor.graph import node_rewriter
from pytensor.graph.rewriting.basic import copy_stack_trace
from pytensor.tensor.basic import expand_dims
from pytensor.tensor.extra_ops import squeeze
from pytensor.tensor.reshape import JoinDims, SplitDims
from pytensor.tensor.rewriting.basic import register_canonicalize


@register_canonicalize
@node_rewriter([SplitDims])
def local_split_dims_to_reshape(fgraph, node):
    """
    Canonicalize SplitDims Ops to Reshape Ops for further graph reasoning (and dispatch to other backends).
    Special case: if shape is (0,), converts to squeeze instead.
    """

    x, shape = node.inputs
    axis = node.op.axis

    # Special case: empty shape -> squeeze
    if shape.type.shape == (0,):
        squeezed_x = squeeze(x, axis=axis)
        copy_stack_trace(x, squeezed_x)
        return [squeezed_x]

    # Special case: empty shape -> squeeze
    if shape.type.shape == (0,):
        squeezed_x = squeeze(x, axis=axis)
        copy_stack_trace(x, squeezed_x)
        return [squeezed_x]

    output_shape = [
        *x.shape[:axis],
        *shape,
        *x.shape[axis + 1 :],
    ]

    new_x = x.reshape(output_shape)
    copy_stack_trace(x, new_x)

    return [new_x]


@register_canonicalize
@node_rewriter([JoinDims])
def local_join_dims_noop(fgraph, node):
    """
    Canonicalize JoinDims Ops to identity if n_axes=1.
    join_dims(x, axis=axis, n_axes=1) -> x
    """
    (x,) = node.inputs
    op = node.op

    if op.n_axes == 1:
        copy_stack_trace(x, x)
        return [x]
    return None


@register_canonicalize
@node_rewriter([JoinDims])
def local_join_dims_expand(fgraph, node):
    """
    Canonicalize JoinDims Ops to expand dims if n_axes=0.
    join_dims(x, axis=axis, n_axes=0) -> expand_dims(x, axis)
    """
    (x,) = node.inputs
    op = node.op

    if op.n_axes == 0:
        expanded_x = expand_dims(x, axis=node.op.start_axis)
        copy_stack_trace(x, expanded_x)
        return [expanded_x]
    return None


@register_canonicalize
@node_rewriter([JoinDims])
def local_join_dims_to_reshape(fgraph, node):
    """
    Canonicalize JoinDims Ops to Reshape Ops for further graph reasoning (and dispatch to other backends).
    """

    (x,) = node.inputs
    op = node.op
    start_axis = op.start_axis
    n_axes = op.n_axes

    output_shape = [
        *x.shape[:start_axis],
        -1,
        *x.shape[start_axis + n_axes :],
    ]

    new_x = x.reshape(output_shape)

    copy_stack_trace(x, new_x)
    return [new_x]
