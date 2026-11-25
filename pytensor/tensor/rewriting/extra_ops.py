import pytensor.scalar.basic as ps
from pytensor.graph.rewriting.basic import node_rewriter
from pytensor.tensor.basic import Alloc, alloc, arange, as_tensor_variable
from pytensor.tensor.elemwise import Elemwise
from pytensor.tensor.extra_ops import CumOp, Repeat, Unique
from pytensor.tensor.rewriting.basic import (
    register_canonicalize,
    register_specialize,
    register_useless,
)


@register_useless
@register_canonicalize
@node_rewriter([Unique])
def local_Unique_scalar(fgraph, node):
    """Convert ``unique(x)`` to ``x`` when ``x`` is a scalar."""
    if node.op.return_index or node.op.return_inverse or node.op.return_counts:
        return False

    uniqued_var = node.inputs[0]

    if uniqued_var.ndim != 0:
        return False

    old_out = node.outputs[0]
    res = as_tensor_variable(uniqued_var, ndim=old_out.ndim, dtype=old_out.dtype)
    return [res]


@register_useless
@register_canonicalize
@node_rewriter([Unique])
def local_Unique_lift(fgraph, node):
    """Convert ``unique(f(x, ...), axis=None)`` to ``unique(x, axis=None)``.

    ``Alloc``, ``Repeat`` and ``second`` only broadcast/tile their input, so
    they don't change the set of unique values and can be dropped from the
    input of an axis-less ``unique`` (a "reduction/consumption" rather than a
    true lift).
    """
    if (
        node.op.return_index
        or node.op.return_inverse
        or node.op.return_counts
        or node.op.axis is not None
    ):
        return False

    var = node.inputs[0]
    owner = var.owner
    if owner is None:
        return False

    if isinstance(owner.op, Alloc | Repeat):
        # The value being broadcast/tiled is the first input.
        inner_var = owner.inputs[0]
    elif isinstance(owner.op, Elemwise) and isinstance(owner.op.scalar_op, ps.Second):
        # ``second(shape, x)`` fills the shape with the second input.
        inner_var = owner.inputs[1]
    else:
        return False

    new_unique, *_ = node.op.make_node(inner_var).outputs

    old_out = node.outputs[0]
    new_x = as_tensor_variable(new_unique, ndim=old_out.ndim, dtype=old_out.dtype)
    return [new_x]


@register_useless
@register_canonicalize
@register_specialize
@node_rewriter([CumOp])
def local_useless_cumop(fgraph, node):
    """Remove a CumOp along an axis of length 1."""
    [x] = node.inputs
    if x.type.broadcastable[node.op.axis]:
        return [x]


@register_canonicalize
@register_specialize
@node_rewriter([CumOp])
def local_cumop_of_alloc(fgraph, node):
    """Apply CumOp before Alloc to avoid accumulating repeated copies.

    If Alloc keeps the cum axis, accumulate the value instead::

        cumop(alloc(x, *shape), axis) -> alloc(cumop(x, value_axis), *shape)

    If Alloc broadcasts the cum axis, all entries along it are equal, so with
    ``k = arange(1, n + 1)``::

        cumsum(alloc(x, *shape), axis) -> alloc(x * k, *shape)

    cumprod along a broadcast axis would become ``x ** k``, which is more
    expensive than the cumulative product it replaces, so it is left alone.
    """
    [x] = node.inputs
    match x.owner_op_and_inputs:
        case Alloc(), value, *shape:
            axis = node.op.axis
            if x.type.broadcastable[axis]:
                # Left to local_useless_cumop
                return None
            value_axis = axis - (len(shape) - value.ndim)
            if value_axis >= 0 and not value.type.broadcastable[value_axis]:
                # cum axis is not broadcast
                new_value = CumOp(axis=value_axis, mode=node.op.mode)(value)
            elif node.op.mode == "add":
                # cum axis is broadcast
                pattern = ["x"] * len(shape)
                pattern[axis] = 0
                k = arange(1, shape[axis] + 1, dtype=x.dtype).dimshuffle(pattern)
                new_value = value * k
            else:
                return None
            return [alloc(new_value, *shape)]
