from pytensor.graph.rewriting.basic import copy_stack_trace, node_rewriter
from pytensor.tensor.elemwise import DimShuffle
from pytensor.tensor.math import Sum, exp, log
from pytensor.tensor.math import sum as at_sum
from pytensor.tensor.math import true_div
from pytensor.tensor.rewriting.basic import register_stabilize
from pytensor.tensor.rewriting.math import local_mul_canonizer
from pytensor.tensor.special import Softmax, SoftmaxGrad, log_softmax
from pytensor.tensor.subtensor import (
    AdvancedIncSubtensor,
    AdvancedSubtensor,
    AdvancedSubtensor1,
    Subtensor,
)
from pytensor.tensor.type import (
    values_eq_approx_remove_inf,
    values_eq_approx_remove_nan,
)


subtensor_ops = (
    Subtensor,
    AdvancedSubtensor,
    AdvancedSubtensor1,
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


@register_stabilize
@node_rewriter([SoftmaxGrad])
def local_logsoftmax_grad(fgraph, node):
    """
    Detect Log(Softmax(x))'s grad and replace it with LogSoftmax(x)'s grad

    Note: only grad is affected
    """
    if (
        node.inputs[0].owner is not None
        and node.inputs[0].owner.op == true_div
        and len(node.inputs[0].owner.inputs) >= 2
        and node.inputs[0].owner.inputs[1].owner is not None
        and isinstance(node.inputs[0].owner.inputs[1].owner.op, Softmax)
        and node.inputs[1] == node.inputs[0].owner.inputs[1]
        and not (
            # skip if it will be optimized by
            # local_advanced_indexing_crossentropy_onehot_grad
            node.inputs[0].owner.op == true_div
            and node.inputs[0].owner.inputs[0].owner is not None
            and isinstance(
                node.inputs[0].owner.inputs[0].owner.op, AdvancedIncSubtensor
            )
            # the rewrite only applies to legacy SoftmaxGrad
            and node.op == SoftmaxGrad(axis=-1)
            and node.inputs[0].owner.inputs[1].ndim == 2
        )
    ):
        # get parameters from unoptimized op
        grads, sm = node.inputs[0].owner.inputs
        ret = grads - at_sum(grads, axis=sm.owner.op.axis, keepdims=True) * sm
        ret.tag.values_eq_approx = values_eq_approx_remove_nan
        copy_stack_trace(node.outputs[0], ret)
        return [ret]


def softmax_simplifier(numerators, denominators):
    for numerator in list(numerators):
        if not numerator.type.dtype.startswith("float"):
            continue

        if not (numerator.owner and numerator.owner.op == exp):
            continue

        matching_denom = None

        for denominator in denominators:
            # Division with dimshuffle
            if denominator.owner and isinstance(denominator.owner.op, DimShuffle):
                ds_order = denominator.owner.op.new_order
                # Check that at most only one dimension is being reintroduced by
                # a dimshuffle. The cases where all dimensions are reintroduced
                # after a complete sum reduction end up in the else branch
                if ds_order.count("x") != 1:
                    continue
                # Check that dimshuffle does not change order of original dims
                ds_order_without_x = tuple(dim for dim in ds_order if dim != "x")
                if tuple(sorted(ds_order_without_x)) != ds_order_without_x:
                    continue
                new_dim = ds_order.index("x")
                z = denominator.owner.inputs[0]
                if z.owner and isinstance(z.owner.op, Sum):
                    sum_axis = z.owner.op.axis
                    # Check that reintroduced dim was the one reduced
                    if (
                        (sum_axis is not None)
                        and (len(sum_axis) == 1)
                        and (sum_axis[0] == new_dim)
                    ):
                        if z.owner.inputs[0] is numerator:
                            (sum_axis,) = sum_axis
                            matching_denom = denominator
                            break

            # Division without dimshuffle
            else:
                z = denominator
                if z.owner and isinstance(z.owner.op, Sum):
                    sum_axis = z.owner.op.axis
                    # Filter out partial summations over more than one axis
                    # The cases where all axis of summation are explicitly given
                    # as in `sum(matrix, axis=(0, 1))` are eventually rewritten
                    # to `sum(matrix)` and this branch is not a blocker
                    if sum_axis is not None and len(sum_axis) != 1:
                        continue
                    if z.owner.inputs[0] is numerator:
                        if sum_axis is not None:
                            (sum_axis,) = sum_axis
                        matching_denom = denominator
                        break

        if matching_denom:
            softmax = Softmax(axis=sum_axis)(numerator.owner.inputs[0])
            copy_stack_trace(numerator, softmax)
            numerators.remove(numerator)
            denominators.remove(matching_denom)
            numerators.append(softmax)

    return numerators, denominators


local_mul_canonizer.add_simplifier(softmax_simplifier, "softmax_simplifier")
