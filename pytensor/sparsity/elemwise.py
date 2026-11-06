from functools import reduce

from pytensor import scalar as ps
from pytensor import tensor as pt
from pytensor.configdefaults import config
from pytensor.graph.basic import Constant
from pytensor.graph.rewriting.unify import OpPattern
from pytensor.sparsity.core import (
    local_sparsity,
    local_zeros,
    register_pattern_rule,
)
from pytensor.tensor.elemwise import Elemwise
from pytensor.tensor.rewriting.basic import elemwise_of


def output_shaped_zeros(zeros, variable, output):
    """Return the known-zero mask of ``variable`` broadcast to the shape of ``output``
    and flattened, or None when it has none."""
    if zeros[variable] is None:
        return None
    # specify_shape restores the static shape that reshape drops, so that broadcasting
    # a length-1 axis stays legal.
    mask = pt.specify_shape(
        zeros[variable].reshape(variable.shape), variable.type.shape
    )
    return pt.broadcast_to(mask, output.shape).ravel()


def constant_condition(node):
    """Return the flat, output-shaped condition of a Switch, or None when it is not a
    ``Constant``."""
    condition = node.inputs[0]
    if not isinstance(condition, Constant):
        return None
    return pt.broadcast_to(
        pt.constant(condition.data.astype(bool)), node.outputs[0].shape
    ).ravel()


def elemwise_blocks(node, keep):
    """Return one broadcast block per input, the same for every output.

    ``keep`` holds, per input, False to drop its block, None to keep every output
    element, or a mask of the output elements to keep.
    """
    output = node.outputs[0]
    row = []
    for variable, input_keep in zip(node.inputs, keep, strict=True):
        if input_keep is False:
            row.append(None)
            continue

        # An input with the output's broadcast pattern has the output's shape, so it
        # needs no gather.
        if variable.type.broadcastable == output.type.broadcastable:
            chain = ()
        else:
            element_ids = pt.specify_shape(
                pt.arange(variable.size).reshape(variable.shape), variable.type.shape
            )
            source_ids = pt.broadcast_to(element_ids, output.shape).ravel()
            chain = (pt.eye(variable.size, dtype=config.floatX)[:, source_ids].T,)

        if input_keep is not None:
            chain = (pt.diag(pt.cast(input_keep, config.floatX)), *chain)
        row.append(chain)
    return [row for _ in node.outputs]


@local_sparsity.register(Elemwise)
def local_sparsity_Elemwise(op, node, zeros):
    return elemwise_blocks(node=node, keep=[None] * len(node.inputs))


@register_pattern_rule(
    local_sparsity,
    elemwise_of(OpPattern(ps.ScalarOp, piecewise_constant=True)),
)
def local_sparsity_piecewise_constant(op, node, zeros):
    """A piecewise-constant scalar Op has a zero derivative almost everywhere."""
    return [[None] * len(node.inputs) for _ in node.outputs]


@register_pattern_rule(local_sparsity, elemwise_of(ps.Switch))
def local_sparsity_Switch(op, node, zeros):
    """The condition picks the branch, but the derivative does not depend on it."""
    take_true = constant_condition(node)
    if take_true is None:
        return elemwise_blocks(node=node, keep=[False, None, None])
    return elemwise_blocks(node=node, keep=[False, take_true, ~take_true])


@register_pattern_rule(local_sparsity, elemwise_of(ps.Mul))
def local_sparsity_Mul(op, node, zeros):
    """The product is zero, for every value of x_i, wherever another factor is a known
    zero, so its derivative in x_i is zero there."""
    output = node.outputs[0]
    nonzero = [
        None
        if (mask := output_shaped_zeros(zeros=zeros, variable=variable, output=output))
        is None
        else ~mask
        for variable in node.inputs
    ]
    keep = []
    for own in range(len(node.inputs)):
        others = [
            mask
            for position, mask in enumerate(nonzero)
            if position != own and mask is not None
        ]
        keep.append(reduce(pt.and_, others) if others else None)
    return elemwise_blocks(node=node, keep=keep)


@register_pattern_rule(local_sparsity, elemwise_of(ps.TrueDiv))
def local_sparsity_TrueDiv(op, node, zeros):
    """x / y is zero, for every nonzero y, wherever x is a known zero."""
    numerator_zeros = output_shaped_zeros(
        zeros=zeros, variable=node.inputs[0], output=node.outputs[0]
    )
    keep = [None, None if numerator_zeros is None else ~numerator_zeros]
    return elemwise_blocks(node=node, keep=keep)


@register_pattern_rule(local_zeros, elemwise_of(ps.Mul))
def local_zeros_Mul(op, node, zeros):
    """The product is zero wherever any factor is."""
    known = [
        mask
        for variable in node.inputs
        if (
            mask := output_shaped_zeros(
                zeros=zeros, variable=variable, output=node.outputs[0]
            )
        )
        is not None
    ]
    return [reduce(pt.or_, known) if known else None]


@register_pattern_rule(local_zeros, elemwise_of(ps.Add))
@register_pattern_rule(local_zeros, elemwise_of(ps.Sub))
def local_zeros_Add_Sub(op, node, zeros):
    """A sum or difference is zero wherever every term is."""
    input_zeros = [
        output_shaped_zeros(zeros=zeros, variable=variable, output=node.outputs[0])
        for variable in node.inputs
    ]
    if any(mask is None for mask in input_zeros):
        return [None]
    return [reduce(pt.and_, input_zeros)]


@register_pattern_rule(local_zeros, elemwise_of(ps.TrueDiv))
def local_zeros_TrueDiv(op, node, zeros):
    """x / y is zero wherever x is."""
    return [
        output_shaped_zeros(
            zeros=zeros, variable=node.inputs[0], output=node.outputs[0]
        )
    ]


@register_pattern_rule(local_zeros, elemwise_of(ps.Second))
def local_zeros_Second(op, node, zeros):
    """``second(x, y)`` is ``y`` broadcast to the shape of ``x``."""
    return [
        output_shaped_zeros(
            zeros=zeros, variable=node.inputs[1], output=node.outputs[0]
        )
    ]


@register_pattern_rule(local_zeros, elemwise_of(ps.Switch))
def local_zeros_Switch(op, node, zeros):
    """A Switch is zero where the branch it can take is, or both branches are when the
    condition is not a ``Constant``."""
    output = node.outputs[0]
    _, if_true, if_false = node.inputs
    true_zeros = output_shaped_zeros(zeros=zeros, variable=if_true, output=output)
    false_zeros = output_shaped_zeros(zeros=zeros, variable=if_false, output=output)
    take_true = constant_condition(node)

    if take_true is None:
        if true_zeros is None or false_zeros is None:
            return [None]
        return [true_zeros & false_zeros]
    if true_zeros is None and false_zeros is None:
        return [None]

    no_zeros = pt.zeros(output.size, dtype=bool)
    return [
        pt.where(
            take_true,
            no_zeros if true_zeros is None else true_zeros,
            no_zeros if false_zeros is None else false_zeros,
        )
    ]


@register_pattern_rule(
    local_zeros,
    elemwise_of(OpPattern(ps.UnaryScalarOp, preserves_zero=True)),
)
def local_zeros_preserves_zero(op, node, zeros):
    """A zero-preserving unary Op is zero wherever its input is."""
    [variable] = node.inputs
    return [output_shaped_zeros(zeros=zeros, variable=variable, output=node.outputs[0])]
