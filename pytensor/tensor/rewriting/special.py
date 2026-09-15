from pytensor.graph.rewriting.basic import (
    PatternNodeRewriter,
    copy_stack_trace,
    node_rewriter,
)
from pytensor.graph.rewriting.unify import OpPattern
from pytensor.scalar.basic import Exp
from pytensor.tensor.elemwise import DimShuffle, Elemwise
from pytensor.tensor.math import Sum, add, exp, log, sub, true_div
from pytensor.tensor.rewriting.basic import register_stabilize
from pytensor.tensor.special import (
    LogSoftmax,
    LogSumExp,
    Softmax,
    log_softmax,
    logaddexp,
    logsumexp,
)
from pytensor.tensor.subtensor import (
    AdvancedSubtensor,
    Subtensor,
)
from pytensor.tensor.type import values_eq_approx_remove_inf
from pytensor.tensor.utils import normalize_reduce_axis


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


# Exp(LogSoftmax(x)) -> Softmax(x)
local_exp_log_softmax = PatternNodeRewriter(
    (exp, (OpPattern(LogSoftmax, axis="axis"), "x")),
    (OpPattern(Softmax, axis="axis"), "x"),
    name="local_exp_log_softmax",
)
register_stabilize(local_exp_log_softmax)


# x - logsumexp(x, axis, keepdims=True) -> LogSoftmax(x)
# The shared "axis" makes the DimShuffle match only when it re-expands the reduced axes
local_log_softmax_from_logsumexp = PatternNodeRewriter(
    (
        sub,
        "x",
        (
            OpPattern(DimShuffle, is_expand_dims=True, augment="axis"),
            (OpPattern(LogSumExp, axis="axis"), "x"),
        ),
    ),
    (OpPattern(LogSoftmax, axis="axis"), "x"),
    name="local_log_softmax_from_logsumexp",
)
register_stabilize(local_log_softmax_from_logsumexp)


@register_stabilize("symbolic_op_recognition", "fast_compile")
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

    ret = Softmax(axis=normalize_reduce_axis(axis, x.type.ndim, normalize_none=True))(x)
    copy_stack_trace(node.outputs, ret)
    return [ret]


@register_stabilize("symbolic_op_recognition", "fast_compile")
@node_rewriter([log])
def local_log_add_exp(fgraph, node):
    """``log(exp(x) + exp(y) + exp(z)) -> logaddexp(x, y, z)``.

    TODO: in canonicalize, change log10 and log2 -> log
    """
    z = node.inputs[0]
    if z.owner and z.owner.op == add:
        zi = z.owner.inputs
        pre_exp = [x.owner.inputs[0] for x in zi if x.owner and x.owner.op == exp]
        # all arguments to add are exp(<something>)
        if len(pre_exp) == len(zi):
            return [logaddexp(*pre_exp)]


@register_stabilize("symbolic_op_recognition", "fast_compile")
@node_rewriter([log])
def local_log_sum_exp(fgraph, node):
    """``log(sum_i(exp(x_i))) -> logsumexp(x)``."""
    sum_node = node.inputs[0].owner
    # If the sum has keepdims=True, there might be a dimshuffle
    if sum_node and isinstance(sum_node.op, DimShuffle):
        dimshuffle_op = sum_node.op
        sum_node = sum_node.inputs[0].owner
    else:
        dimshuffle_op = None

    if not (sum_node and isinstance(sum_node.op, Sum)):
        return

    exp_node, axis = sum_node.inputs[0].owner, sum_node.op.axis
    if not (
        exp_node
        and isinstance(exp_node.op, Elemwise)
        and isinstance(exp_node.op.scalar_op, Exp)
    ):
        return

    ret = logsumexp(exp_node.inputs[0], axis=axis)

    # Restore the dimshuffle op, if any.
    if dimshuffle_op:
        ret = dimshuffle_op(ret)

    return [ret]
