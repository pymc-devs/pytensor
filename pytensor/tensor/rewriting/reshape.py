from pytensor.graph import node_rewriter
from pytensor.graph.rewriting.basic import copy_stack_trace
from pytensor.tensor.basic import expand_dims
from pytensor.tensor.extra_ops import squeeze
from pytensor.tensor.reshape import JoinDims, SplitDims
from pytensor.tensor.rewriting.basic import register_canonicalize
from pytensor.tensor.shape import specify_shape


@register_canonicalize
@node_rewriter([JoinDims])
def local_join_dims_degenerate(fgraph, node):
    """Canonicalize the size-1/degenerate ``JoinDims`` forms out to DimShuffle.

    A join of zero axes inserts a size-1 dimension (``expand_dims``); a join of
    a single axis is the identity. The general join stays a ``JoinDims`` Op and
    is dispatched natively by each backend.
    """
    (x,) = node.inputs
    op = node.op

    if op.n_axes == 0:
        expanded_x = expand_dims(x, axis=op.start_axis)
        copy_stack_trace(x, expanded_x)
        return [expanded_x]

    if op.n_axes == 1:
        return [x]

    return None


@register_canonicalize
@node_rewriter([SplitDims])
def local_split_dims_degenerate(fgraph, node):
    """Canonicalize the size-1/degenerate ``SplitDims`` forms out to DimShuffle.

    Splitting into zero dims squeezes the axis (which must be size 1); splitting
    into a single dim is a ``specify_shape`` on that axis. The general split
    stays a ``SplitDims`` Op and is dispatched natively by each backend.
    """
    x, shape = node.inputs
    axis = node.op.axis

    if shape.type.shape == (0,):
        squeezed_x = squeeze(x, axis=axis)
        copy_stack_trace(x, squeezed_x)
        return [squeezed_x]

    if shape.type.shape == (1,):
        specified_shape = [None] * x.type.ndim
        specified_shape[axis] = shape
        specified_x = specify_shape(x, specified_shape)
        copy_stack_trace(x, specified_x)
        return [specified_x]

    return None
