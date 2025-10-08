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
    if not isinstance(node.op, Unique):
        return False

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
def local_Unique_Alloc_lift(fgraph, node):
    """Convert ``unique(alloc(x, ...), axis=None)`` to ``unique(x, axis=None)``.

    This isn't really so much a lift as a "reduction/consumption".
    """
    if not isinstance(node.op, Unique):
        return False

    if (
        node.op.return_index
        or node.op.return_inverse
        or node.op.return_counts
        or node.op.axis is not None
    ):
        return False

    alloc_var = node.inputs[0]

    if not (alloc_var.owner and isinstance(alloc_var.owner.op, Alloc)):
        return False

    alloced_var, *_alloc_shape = alloc_var.owner.inputs

    new_unique, *_ = node.op.make_node(alloced_var).outputs

    old_out = node.outputs[0]
    new_x = as_tensor_variable(new_unique, ndim=old_out.ndim, dtype=old_out.dtype)
    return [new_x]


@register_useless
@register_canonicalize
@node_rewriter([Unique])
def local_Unique_Repeat_lift(fgraph, node):
    """Convert ``unique(repeat(x, ...), axis=None)`` to ``unique(x, axis=None)``.

    This isn't really so much a lift as a "reduction/consumption".
    """
    if not isinstance(node.op, Unique):
        return False

    if (
        node.op.return_index
        or node.op.return_inverse
        or node.op.return_counts
        or node.op.axis is not None
    ):
        return False

    repeat_var = node.inputs[0]

    if not (repeat_var.owner and isinstance(repeat_var.owner.op, Repeat)):
        return False

    repeated_var, *_repeat_shape = repeat_var.owner.inputs

    new_unique, *_ = node.op.make_node(repeated_var).outputs

    old_out = node.outputs[0]
    new_x = as_tensor_variable(new_unique, ndim=old_out.ndim, dtype=old_out.dtype)
    return [new_x]


@register_useless
@register_canonicalize
@node_rewriter([Unique])
def local_Unique_second(fgraph, node):
    """Convert ``unique(second(x, ...), axis=None)`` to ``second(x, axis=None)``.

    This isn't really so much a lift as a "reduction/consumption".
    """
    if not isinstance(node.op, Unique):
        return False

    if (
        node.op.return_index
        or node.op.return_inverse
        or node.op.return_counts
        or node.op.axis is not None
    ):
        return False

    second_var = node.inputs[0]

    if not (
        second_var.owner
        and isinstance(second_var.owner.op, Elemwise)
        and isinstance(second_var.owner.op.scalar_op, ps.Second)
    ):
        return False

    _shape_var, seconded_var = second_var.owner.inputs

    new_unique, *_ = node.op.make_node(seconded_var).outputs

    old_out = node.outputs[0]
    new_x = as_tensor_variable(new_unique, ndim=old_out.ndim, dtype=old_out.dtype)
    return [new_x]
