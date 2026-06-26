import pytensor.scalar.basic as ps
from pytensor.graph.rewriting.basic import node_rewriter
from pytensor.tensor.basic import Alloc, as_tensor_variable
from pytensor.tensor.elemwise import Elemwise
from pytensor.tensor.extra_ops import Repeat, Unique
from pytensor.tensor.rewriting.basic import register_canonicalize, register_useless


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
