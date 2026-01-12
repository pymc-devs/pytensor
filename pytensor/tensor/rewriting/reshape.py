from pytensor.graph import node_rewriter
from pytensor.graph.rewriting.basic import copy_stack_trace
from pytensor.tensor.reshape import JoinDims, SplitDims
from pytensor.tensor.rewriting.basic import register_canonicalize


@register_canonicalize
@node_rewriter([SplitDims])
def local_split_dims_to_reshape(fgraph, node):
    """
    Canonicalize SplitDims Ops to Reshape Ops for further graph reasoning (and dispatch to other backends).
    """

    x, shape = node.inputs
    axis = node.op.axis

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
