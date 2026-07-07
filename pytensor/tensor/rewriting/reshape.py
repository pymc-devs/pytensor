from pytensor.graph import node_rewriter
from pytensor.graph.rewriting.basic import copy_stack_trace
from pytensor.tensor.basic import MakeVector, expand_dims, join, stack
from pytensor.tensor.elemwise import Elemwise
from pytensor.tensor.extra_ops import squeeze
from pytensor.tensor.math import prod
from pytensor.tensor.reshape import JoinDims, SplitDims, join_dims, split_dims
from pytensor.tensor.rewriting.basic import register_canonicalize, register_specialize
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
def local_join_dims_squeeze(fgraph, node):
    """Squeeze provably size-1 dims out of a ``JoinDims`` span.

    A size-1 dim contributes a factor of 1 to the join, so it can be squeezed
    out before joining. Keeping ``JoinDims`` free of size-1 dims is the canonical
    form (mirrors ``local_reshape_to_dimshuffle``): it isolates the genuine
    dimension-merge from the expand/squeeze that DimShuffle-targeting rewrites
    can then simplify (e.g. dissolving a broadcast behind a ``repeat``).
    """
    (x,) = node.inputs
    op = node.op
    static_shape = x.type.shape
    drop = [i for i in op.axis_range if static_shape[i] == 1]
    if not drop:
        return None

    squeezed = squeeze(x, axis=drop)
    out = join_dims(squeezed, start_axis=op.start_axis, n_axes=op.n_axes - len(drop))
    copy_stack_trace(node.outputs[0], out)
    return [out]


@register_canonicalize
@node_rewriter([SplitDims])
def local_split_dims_expand(fgraph, node):
    """Emit ``expand_dims`` for provably size-1 factors of a ``SplitDims``.

    A size-1 split factor is an inert dim, so it is expressed as an
    ``expand_dims`` around a split of the genuine (non-1) factors rather than
    carried inside the ``SplitDims`` (canonical form; mirror of
    ``local_join_dims_squeeze``). Fires only in the mixed case (at least one
    size-1 and one non-1 factor); the split of the kept factors preserves the
    ``prod(sizes) == x.shape[axis]`` runtime check.
    """
    x, shape = node.inputs
    axis = node.op.axis
    m = _split_count(node)
    [output] = node.outputs

    split_static = output.type.shape[axis : axis + m]
    keep = [j for j in range(m) if split_static[j] != 1]
    drop = [axis + j for j in range(m) if split_static[j] == 1]
    if not drop or not keep:
        return None

    inner = split_dims(x, shape=stack([shape[j] for j in keep]), axis=axis)
    out = expand_dims(inner, axis=drop)
    copy_stack_trace(node.outputs[0], out)
    return [out]


@register_canonicalize
@node_rewriter([JoinDims])
def local_join_of_split(fgraph, node):
    """Collapse ``JoinDims(SplitDims(x, s))`` when the spans are boundary-aligned.

    - Join span ⊇ split span → a single ``JoinDims`` over the original axes: the
      split group recombines and joins with the covered pass-through axes (the
      exact-span case degenerates to ``x``).
    - Join span ⊆ split span → a single ``SplitDims`` of ``x`` with the joined
      sub-run of sizes multiplied into one dim.
    - A join straddling a split boundary is a genuine dimension-mix and is left.
    """
    (inp,) = node.inputs
    split_node = inp.owner
    if split_node is None or not isinstance(split_node.op, SplitDims):
        return None

    x, s = split_node.inputs
    a = split_node.op.axis
    m = _split_count(split_node)
    j, nj = node.op.start_axis, node.op.n_axes

    if j <= a and j + nj >= a + m:
        merged = join_dims(x, start_axis=j, n_axes=nj - m + 1)
    elif a <= j and j + nj <= a + m:
        lo, hi = j - a, j - a + nj
        merged_size = prod(s[lo:hi], keepdims=True)
        parts = [
            *([s[:lo]] if lo > 0 else []),
            merged_size,
            *([s[hi:]] if hi < m else []),
        ]
        new_sizes = parts[0] if len(parts) == 1 else join(0, *parts)
        merged = split_dims(x, shape=new_sizes, axis=a)
    else:
        return None

    copy_stack_trace(node.outputs[0], merged)
    return [merged]


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


@register_canonicalize
@node_rewriter([SplitDims])
def local_split_of_split(fgraph, node):
    """``SplitDims(SplitDims(x, s1), s2) -> SplitDims(x, spliced)``.

    When the outer split refines an axis the inner split produced, the two
    collapse into a single split of the original axis, splicing ``s2`` into
    ``s1`` in place of the refined dim. Splits of untouched pass-through axes are
    left as two independent splits.
    """
    x, s2 = node.inputs
    inner = x.owner
    if inner is None or not isinstance(inner.op, SplitDims):
        return None

    x0, s1 = inner.inputs
    n1 = _split_count(inner)
    k = node.op.axis - inner.op.axis
    if not 0 <= k < n1:
        return None

    parts = [*([s1[:k]] if k > 0 else []), s2, *([s1[k + 1 :]] if k + 1 < n1 else [])]
    spliced = parts[0] if len(parts) == 1 else join(0, *parts)
    merged = split_dims(x0, shape=spliced, axis=inner.op.axis)
    copy_stack_trace(node.outputs[0], merged)
    return [merged]


@register_canonicalize
@register_specialize
@node_rewriter([JoinDims, SplitDims])
def local_join_split_dims_lift(fgraph, node):
    """Lift a ``JoinDims``/``SplitDims`` through a unary ``Elemwise``.

    ``JoinDims(UnaryElemwise(x)) -> UnaryElemwise(JoinDims(x))`` (and likewise for
    ``SplitDims``). Mirrors ``local_reshape_lift`` so rewrites that match on the
    Elemwise (e.g. ``log1msigm_to_softplus``) still fire when a reshape-as-view op
    sits between them.
    """
    inner = node.inputs[0].owner
    if inner is None or not isinstance(inner.op, Elemwise) or len(inner.inputs) != 1:
        return None

    (elem_input,) = inner.inputs
    lifted = node.op(elem_input, *node.inputs[1:])
    copy_stack_trace(node.outputs, lifted)
    out = inner.op(lifted)
    copy_stack_trace(node.outputs + node.inputs, out)
    return [out]
