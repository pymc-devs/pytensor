"""Scalar fusion rewrites for the JavaScript emitter."""

from collections import Counter

import numpy as np

from pytensor.graph.basic import Constant
from pytensor.graph.rewriting.basic import MergeOptimizer, in2out, node_rewriter
from pytensor.scalar.basic import (
    EQ,
    GE,
    GT,
    LE,
    LT,
    NEQ,
    Composite,
    Log,
    Mul,
    ScalarOp,
    Switch,
    and_,
    cast,
    eq,
    upgrade_to_float,
)


class MulLog(ScalarOp):
    """Fuse multiplication and log, allowing zero factors to bypass finite logs."""

    __props__ = ()
    nin = 2
    nout = 1

    def impl(self, x, y):
        return x * np.log(y)


mul_log = MulLog(upgrade_to_float)


def switch_branch(variable, truth):
    branch = variable.owner.inputs[1 if truth else 2]
    return cast(branch, variable.dtype) if branch.type != variable.type else branch


@node_rewriter([Switch])
def local_prune_nested_switches(fgraph, node):
    condition, *branches = node.inputs
    if condition.dtype != "bool":
        return

    def prune(variable, truth, memo):
        if variable in memo:
            return memo[variable]
        owner = variable.owner
        if owner is None:
            return variable
        if isinstance(owner.op, Switch) and owner.inputs[0] is condition:
            result = prune(switch_branch(variable, truth), truth, memo)
        else:
            inputs = [prune(arg, truth, memo) for arg in owner.inputs]
            if all(new is old for new, old in zip(inputs, owner.inputs, strict=True)):
                return variable
            outputs = owner.op(*inputs, return_list=True)
            memo.update(zip(owner.outputs, outputs, strict=True))
            result = memo[variable]
        memo[variable] = result
        return result

    replacements = [
        prune(branch, truth, {})
        for branch, truth in zip(branches, (True, False), strict=True)
    ]
    if any(new is not old for new, old in zip(replacements, branches, strict=True)):
        return [node.op(condition, *replacements)]


@node_rewriter([ScalarOp])
def local_merge_switches(fgraph, node):
    if isinstance(node.op, Switch) or len(node.outputs) != 1:
        return
    switches = [
        arg for arg in node.inputs if arg.owner and isinstance(arg.owner.op, Switch)
    ]
    conditions = Counter(arg.owner.inputs[0] for arg in switches)
    if not conditions:
        return
    condition, count = conditions.most_common(1)[0]
    if count < 2 or condition.dtype != "bool":
        return
    branches = [
        node.op(
            *[
                switch_branch(arg, truth)
                if arg.owner
                and isinstance(arg.owner.op, Switch)
                and arg.owner.inputs[0] is condition
                else arg
                for arg in node.inputs
            ]
        )
        for truth in (True, False)
    ]
    return [Switch()(condition, *branches)]


@node_rewriter([Mul])
def local_bool_mul(fgraph, node):
    if len(node.inputs) == 2 and node.outputs[0].dtype == "bool":
        return [and_(*node.inputs)]


@node_rewriter([EQ])
def local_log_zero_check(fgraph, node):
    for logarithm, constant in (node.inputs, node.inputs[::-1]):
        if (
            isinstance(constant, Constant)
            and np.isneginf(constant.data)
            and logarithm.owner
            and isinstance(logarithm.owner.op, Log)
        ):
            argument = logarithm.owner.inputs[0]
            return [eq(argument, argument.type.make_constant(0))]


@node_rewriter([EQ, NEQ, LT, LE, GT, GE])
def local_exact_comparison_constant(fgraph, node):
    for position, constant in enumerate(node.inputs):
        other = node.inputs[1 - position]
        if (
            isinstance(constant, Constant)
            and np.dtype(constant.dtype).kind in "fiub"
            and other.dtype.startswith("float")
            and constant.dtype != other.dtype
        ):
            with np.errstate(over="ignore", invalid="ignore"):
                value = np.asarray(constant.data).astype(other.dtype).item()
            exact = (
                value == float(constant.data)
                if np.dtype(constant.dtype).kind == "f"
                else np.isfinite(value)
                and value.is_integer()
                and int(value) == int(constant.data)
            )
            if exact:
                inputs = list(node.inputs)
                inputs[position] = other.type.make_constant(value)
                return [node.op(*inputs)]


def specialize_scalar(op, inputs):
    if not isinstance(op, Composite):
        return op
    graph = op.fgraph.unfreeze()
    zero_inputs = {
        scalar
        for scalar, tensor in zip(graph.inputs, inputs, strict=True)
        if isinstance(tensor, Constant)
        and tensor.data.size
        and np.count_nonzero(tensor.data) * 2 < tensor.data.size
    }

    @node_rewriter([Mul])
    def local_mul_log(fgraph, node):
        if len(node.inputs) != 2:
            return
        for factor, logarithm in (node.inputs, node.inputs[::-1]):
            if (
                factor in zero_inputs
                and logarithm.owner
                and isinstance(logarithm.owner.op, Log)
            ):
                return [mul_log(factor, logarithm.owner.inputs[0])]

    in2out(
        local_log_zero_check,
        local_bool_mul,
        local_exact_comparison_constant,
        local_mul_log,
    ).rewrite(graph)
    MergeOptimizer().rewrite(graph)
    in2out(local_prune_nested_switches, local_merge_switches).rewrite(graph)
    MergeOptimizer().rewrite(graph)
    return Composite(graph.inputs, graph.outputs)
