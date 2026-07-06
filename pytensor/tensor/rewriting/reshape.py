from pytensor.graph import node_rewriter
from pytensor.graph.rewriting.basic import copy_stack_trace
from pytensor.tensor.basic import MakeVector, expand_dims
from pytensor.tensor.extra_ops import squeeze
from pytensor.tensor.reshape import JoinDims, SplitDims
from pytensor.tensor.rewriting.basic import register_canonicalize
from pytensor.tensor.rewriting.subtensor import _is_shape_of_x_at
from pytensor.tensor.shape import specify_shape


def _split_count(split_node):
    """Number of dims a ``SplitDims`` node produces from its split axis."""
    return split_node.outputs[0].type.ndim - split_node.inputs[0].type.ndim + 1


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


@register_canonicalize
@node_rewriter([JoinDims])
def local_join_of_split(fgraph, node):
    """``JoinDims(SplitDims(x, s)) -> x`` when the join re-merges the split span.

    Unconditional: ``SplitDims`` guarantees ``prod(s) == x.shape[axis]`` at
    runtime, so re-joining exactly those dims restores ``x``.
    """
    (inp,) = node.inputs
    split_node = inp.owner
    if split_node is None or not isinstance(split_node.op, SplitDims):
        return None

    x = split_node.inputs[0]
    op = node.op
    if op.start_axis == split_node.op.axis and op.n_axes == _split_count(split_node):
        copy_stack_trace(node.outputs[0], x)
        return [x]

    return None


@register_canonicalize
@node_rewriter([JoinDims])
def local_join_of_join(fgraph, node):
    """``JoinDims(JoinDims(x)) -> JoinDims(x)`` when the two spans are contiguous.

    Merges only when the outer join span covers the inner's joined axis, so the
    two joins collapse into a single contiguous join in ``x`` coordinates
    (``n1 + n2 - 1`` axes from the outer start). Disjoint spans are left as two
    joins.
    """
    (inp,) = node.inputs
    inner = inp.owner
    if inner is None or not isinstance(inner.op, JoinDims):
        return None

    (x,) = inner.inputs
    start1, n1 = inner.op.start_axis, inner.op.n_axes
    start2, n2 = node.op.start_axis, node.op.n_axes

    if start2 <= start1 < start2 + n2:
        merged = JoinDims(start2, n1 + n2 - 1)(x)
        copy_stack_trace(node.outputs[0], merged)
        return [merged]

    return None


@register_canonicalize
@node_rewriter([SplitDims])
def local_split_of_join(fgraph, node):
    """``SplitDims(JoinDims(x), s) -> x`` when ``s`` restores the joined dims.

    The one place needing a static proof: each ``s[i]`` must provably equal the
    corresponding pre-join dim ``x.shape[start + i]``.
    """
    x, shape = node.inputs
    inner = x.owner
    if inner is None or not isinstance(inner.op, JoinDims):
        return None

    (x0,) = inner.inputs
    start1, n1 = inner.op.start_axis, inner.op.n_axes
    if node.op.axis != start1 or _split_count(node) != n1:
        return None

    if shape.owner is None or not isinstance(shape.owner.op, MakeVector):
        return None
    elems = shape.owner.inputs
    if len(elems) != n1:
        return None

    if all(_is_shape_of_x_at(e, x0, start1 + i) for i, e in enumerate(elems)):
        copy_stack_trace(node.outputs[0], x0)
        return [x0]

    return None
