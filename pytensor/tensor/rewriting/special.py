from pytensor.graph.rewriting.basic import copy_stack_trace, node_rewriter
from pytensor.scalar.basic import Exp
from pytensor.tensor.elemwise import DimShuffle, Elemwise
from pytensor.tensor.math import Sum, log, true_div
from pytensor.tensor.rewriting.basic import register_stabilize
from pytensor.tensor.special import Softmax, log_softmax
from pytensor.tensor.subtensor import (
    AdvancedSubtensor,
    Subtensor,
)
from pytensor.tensor.type import values_eq_approx_remove_inf


subtensor_ops = (
    Subtensor,
    AdvancedSubtensor,
)


@register_stabilize
@node_rewriter([log])
def local_logsoftmax(fgraph, node):
    """
    Detect Log(Softmax(x)) and replace it with LogSoftmax(x)

    This also lifts Subtensor or Dimshuffle operations that could be in between log and softmax

    Note: only forward pass is affected
    """

    def find_softmax_under_lifteable_ops(inp_node, ops_to_lift):
        if inp_node is None:
            return

        if isinstance(inp_node.op, Softmax):
            return inp_node

        if isinstance(inp_node.op, subtensor_ops):
            ops_to_lift.append((inp_node.op, inp_node.inputs[1:]))
            return find_softmax_under_lifteable_ops(
                inp_node.inputs[0].owner, ops_to_lift
            )

        if isinstance(inp_node.op, DimShuffle):
            ops_to_lift.append((inp_node.op, ()))
            return find_softmax_under_lifteable_ops(
                inp_node.inputs[0].owner, ops_to_lift
            )

    ops_to_lift = []
    softmax_node = find_softmax_under_lifteable_ops(node.inputs[0].owner, ops_to_lift)

    if softmax_node is None:
        return

    ret = log_softmax(softmax_node.inputs[0], axis=softmax_node.op.axis)
    ret.tag.values_eq_approx = values_eq_approx_remove_inf

    # Lift ops that used to be between log and softmax
    for op_to_lift, parameters in reversed(ops_to_lift):
        ret = op_to_lift(ret, *parameters)

    copy_stack_trace(node.outputs, ret)
    return [ret]


@register_stabilize("symbolic_op_recognition")
@node_rewriter([true_div])
def local_softmax_stabilize(fgraph, node):
    """Detect exp(x) / sum(exp(x), keepdims=True) and replace with Softmax(x)."""
    numerator, denominator = node.inputs

    if not numerator.type.dtype.startswith("float"):
        return

    match numerator.owner_op_and_inputs:
        case Elemwise(Exp()), x:
            pass
        case _:
            return None

    # Denominator may be wrapped in a DimShuffle (from keepdims=True)
    match denominator.owner_op_and_inputs:
        case DimShuffle(), sum_var:
            pass
        case _:
            sum_var = denominator

    match sum_var.owner_op_and_inputs:
        case (Sum(axis=axis), exp_x) if exp_x is numerator:
            pass
        case _:
            return None

    ret = Softmax(axis=axis)(x)
    copy_stack_trace(node.outputs, ret)
    return [ret]
