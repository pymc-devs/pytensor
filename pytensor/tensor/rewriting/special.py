import numpy as np

from pytensor.graph.basic import Constant
from pytensor.graph.rewriting.basic import (
    PatternNodeRewriter,
    copy_stack_trace,
    node_rewriter,
)
from pytensor.graph.rewriting.unify import OpPattern
from pytensor.scalar.basic import Add, Exp
from pytensor.tensor.basic import cast
from pytensor.tensor.elemwise import DimShuffle, Elemwise
from pytensor.tensor.math import Sum, exp, log, sigmoid, softplus, sub, true_div
from pytensor.tensor.rewriting.basic import register_specialize, register_stabilize
from pytensor.tensor.special import (
    LogAddExp,
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
    _is_provably_positive,
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


def _addends_in_log_form(terms):
    """Map each addend to its logarithm, or return ``None`` when that is unsafe.

    Addends that are ``exp(x)`` map to ``x``; the rest must be provably positive
    (e.g. positive constants) and map to ``log(term)``, which constant folding
    evaluates for constants. At least one addend must be an ``exp`` for the sum
    to be at risk of overflow, otherwise ``None`` is returned.
    """
    log_terms = []
    any_exp = False
    for term in terms:
        match term.owner_op_and_inputs:
            case Elemwise(Exp()), x:
                log_terms.append(x)
                any_exp = True
            case _ if _is_provably_positive(term):
                log_terms.append(log(term))
            case _:
                return None
    return log_terms if any_exp else None


@register_stabilize("symbolic_op_recognition", "fast_compile")
@node_rewriter([log])
def local_log_add_exp(fgraph, node):
    """``log(exp(x) + exp(y) + c) -> logaddexp(x, y, log(c))``.

    Every addend must be an ``exp`` or provably positive (e.g. a positive constant),
    and at least one must be an ``exp``.

    TODO: in canonicalize, change log10 and log2 -> log
    """
    match node.inputs[0].owner_op_and_inputs:
        case Elemwise(Add()), *terms:
            pass
        case _:
            return None

    log_terms = _addends_in_log_form(terms)
    if log_terms is None:
        return None

    ret = logaddexp(*log_terms)
    if ret.dtype != node.outputs[0].dtype:
        ret = cast(ret, node.outputs[0].dtype)
    copy_stack_trace(node.outputs, ret)
    return [ret]


@register_stabilize("symbolic_op_recognition", "fast_compile")
@node_rewriter([true_div])
def local_sigmoid_stabilize(fgraph, node):
    """Detect ``exp(x) / (exp(x) + exp(y) + c)`` and replace it with
    ``sigmoid(x - logaddexp(y, log(c)))``.

    The numerator must be one of the denominator addends and the same restrictions of
    `local_log_add_exp` apply to the addends. With a two-addend denominator the
    ``logaddexp`` collapses to the log form of the other addend, so
    ``c / (c + exp(x)) -> sigmoid(log(c) - x)`` and
    ``exp(x) / (exp(x) + exp(y)) -> sigmoid(x - y)``.
    """
    num, denom = node.inputs
    match denom.owner_op_and_inputs:
        case Elemwise(Add()), *terms:
            pass
        case _:
            return None

    num_idx = next((i for i, term in enumerate(terms) if term is num), None)
    if num_idx is None:
        return None

    log_terms = _addends_in_log_form(terms)
    if log_terms is None:
        return None

    others = [term for i, term in enumerate(log_terms) if i != num_idx]
    ret = sigmoid(
        log_terms[num_idx] - (logaddexp(*others) if len(others) > 1 else others[0])
    )
    if ret.dtype != node.outputs[0].dtype:
        ret = cast(ret, node.outputs[0].dtype)
    copy_stack_trace(node.outputs, ret)
    return [ret]


@register_stabilize
@register_specialize
@node_rewriter([LogAddExp])
def local_logaddexp_const_to_softplus(fgraph, node):
    """``logaddexp(c, x) -> c + softplus(x - c)`` for a finite constant ``c``.

    A single ``softplus`` replaces the max-subtraction inner graph of `LogAddExp`, and
    its gradient is a bare ``sigmoid``. Restricted to finite constants: with
    ``c = -inf`` this form would give ``nan`` where `LogAddExp` correctly returns
    ``x``.

    Registered in stabilize so it sees the `LogAddExp` emitted by `local_log_add_exp`
    (whose ``log(c)`` input end-of-pass constant folding collapses) before the
    specialize-time `OpFromGraph` inliner gets to it.
    """
    if len(node.inputs) != 2:
        return None
    for const, x in (node.inputs, node.inputs[::-1]):
        if isinstance(const, Constant) and np.isfinite(const.data).all():
            ret = const + softplus(x - const)
            if ret.dtype != node.outputs[0].dtype:
                ret = cast(ret, node.outputs[0].dtype)
            copy_stack_trace(node.outputs, ret)
            return [ret]
    return None


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
