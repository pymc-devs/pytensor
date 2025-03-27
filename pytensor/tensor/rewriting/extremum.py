import operator
from collections import deque

import numpy as np

from pytensor.compile import optdb
from pytensor.graph import Constant, node_rewriter
from pytensor.graph.rewriting.basic import (
    copy_stack_trace,
    out2in,
)
from pytensor.scalar import (
    GE,
    GT,
    LE,
    LT,
    Abs,
    Add,
    Cast,
    Exp,
    Log,
    Log1p,
    Maximum,
    Minimum,
    Sqr,
    Sub,
    discrete_dtypes,
)
from pytensor.tensor.basic import atleast_Nd
from pytensor.tensor.elemwise import Elemwise
from pytensor.tensor.extra_ops import broadcast_arrays
from pytensor.tensor.math import add, maximum, minimum, switch, variadic_add
from pytensor.tensor.rewriting.basic import register_canonicalize
from pytensor.tensor.shape import Shape_i
from pytensor.tensor.type import uint_dtypes
from pytensor.tensor.utils import import_func_from_string
from pytensor.tensor.variable import TensorConstant


EXTREMUM_OPS = Minimum | Maximum
DIRECTIONAL_COMPARISON_OPS = GE | GT | LE | LT


@register_canonicalize
@node_rewriter([switch])
def local_switch_to_extremum(fgraph, node):
    """Rewrite switch(x >= y, x, y) -> maximum(x, y)."""
    [out] = node.outputs

    if not all(out.type.broadcastable):
        # Only do this for scalar graphs
        return None

    if out.dtype not in discrete_dtypes:
        # Switch ignores `nan` values so it is not equivalent to maximum in that case
        return None

    cond, x, y = node.inputs
    cond_node = cond.owner
    if not (
        cond_node is not None
        and isinstance(cond_node.op, Elemwise)
        and isinstance(cond_node.op.scalar_op, DIRECTIONAL_COMPARISON_OPS)
    ):
        return None

    cond_x, cond_y = cond_node.inputs
    logical_op = cond_node.op.scalar_op
    if cond_x is x and cond_y is y:
        if isinstance(logical_op, GT | GE):
            return [maximum(x, y)]
        else:
            return [minimum(x, y)]
    elif cond_x is y and cond_y is x:
        # Flipped meaning
        if isinstance(logical_op, GT | GE):
            return [minimum(x, y)]
        else:
            return [maximum(x, y)]


@register_canonicalize
@node_rewriter([add])
def local_extremum_plus_x(fgraph, node):
    """Rewrite maximum(y, z) + x  -> maximum(y+x, z+x).

    Only do this for scalar graphs and when x is a root variable
    """
    if not all(node.out.type.broadcastable):
        return None

    minmax_terms = [
        t
        for t in node.inputs
        if t.owner
        and isinstance(t.owner.op, Elemwise)
        and isinstance(t.owner.op.scalar_op, EXTREMUM_OPS)
    ]
    if len(minmax_terms) != 1:
        return None
    [minmax_term] = minmax_terms
    other_terms = [t for t in node.inputs if t is not minmax_term]
    if not all(t.owner is None for t in other_terms):
        # Keep it to simple additions
        return None
    c = variadic_add(*other_terms)

    if isinstance(c, Constant) and c.unique_value == 0:
        # Eager optimization if c is zero, to reduce number of passes
        return [minmax_term]

    # To reduce passes we do c + t, as c is likely to be a constant and this is where the add_canonizer would put them next.
    return [minmax_term.owner.op(*[c + t for t in minmax_term.owner.inputs])]


@register_canonicalize
@node_rewriter([minimum, maximum])
def local_flatten_extremum(fgraph, node):
    """Rewrite maximum(maximum(x, y), ..., maximum(w, z)) -> maximum(x, y, ..., w, z).

    This makes it easier to remove useless branches that don't seem to talk to each other.

    Also remove duplicated variables or multiple constants.

    Restricted to scalar graphs only.
    """
    if not all(node.out.type.broadcastable):
        return None

    scalar_op = node.op.scalar_op
    inputs = node.inputs

    # Quick exit circuit
    if not (
        # Repeated inputs
        len(inputs) != len(set(inputs))
        # There's a nested Op that is the same as the outer one
        or any(
            inp.owner is not None
            and isinstance(inp.owner.op, Elemwise)
            and inp.owner.op.scalar_op == scalar_op
            for inp in inputs
        )
        # There are multiple constants
        or sum(isinstance(inp, Constant) for inp in inputs) > 1
    ):
        return None

    old_inputs = deque(inputs)
    new_inputs = []
    new_inputs_set = set()  # For faster comparison, but we don't want random ordering
    is_maximum = isinstance(scalar_op, Maximum)
    extremum_const = None
    while old_inputs:
        old_inp = old_inputs.popleft()
        if old_inp in new_inputs_set:
            # duplicate inputs
            continue

        if (
            old_inp.owner
            and isinstance(old_inp.owner.op, Elemwise)
            and old_inp.owner.op.scalar_op == scalar_op
        ):
            # Add to the queue to be flatten out
            old_inputs.extend(old_inp.owner.inputs)
            continue

        if isinstance(old_inp, Constant):
            if extremum_const is None:
                extremum_const = old_inp
            else:
                # Either discard this constant or the previous one
                # TODO: We could apply this logic to non-scalars as well
                data = old_inp.data.item()
                extremum_data = extremum_const.data.item()
                if (is_maximum and data <= extremum_data) or (
                    not is_maximum and data >= extremum_data
                ):
                    continue  # discard this constant

                new_inputs.remove(extremum_const)
                new_inputs_set.remove(extremum_const)
                extremum_const = old_inp

        new_inputs.append(old_inp)
        new_inputs_set.add(old_inp)

    if len(new_inputs) > 1:
        new_out = node.op(*new_inputs)
        copy_stack_trace(new_inputs, new_out)
    else:
        [new_out] = new_inputs

    # Removed constants may have broadcast or upcast the output
    if new_out.dtype != node.out.type.dtype:
        new_out = new_out.astype(node.out.type.dtype)
    if new_out.ndim != node.out.type.ndim:
        new_out = atleast_Nd(new_out, node.out.type.ndim)
    return [new_out]


@register_canonicalize
@node_rewriter([maximum, minimum])
def local_useless_extremum_x_plus_offset(fgraph, node):
    """Rewrite maximum(x, x + 1) -> x + 1."""
    variables, constants = [], []
    for inp in node.inputs:
        if (
            inp.owner is not None
            and isinstance(inp.owner.op, Elemwise)
            and isinstance(inp.owner.op.scalar_op, Add)
        ):
            if len(inp.owner.inputs) > 2:
                # Addition with too many terms for us to reason about
                return
            x, y = inp.owner.inputs
            if isinstance(x, TensorConstant) and x.unique_value is not None:
                variables.append(y)
                constants.append(x.unique_value)
            elif isinstance(y, TensorConstant) and y.unique_value is not None:
                variables.append(x)
                constants.append(y.unique_value)
            else:
                return None
        else:
            variables.append(inp)
            constants.append(0)

    if len(set(variables)) != 1:
        # TODO: Implement logic for multiple subsets of variables
        return None

    # Find the branch with the highest constant
    if node.op == maximum:
        new_out = node.inputs[np.argmax(constants)]
    else:
        new_out = node.inputs[np.argmin(constants)]

    # Removed branch may have broadcast or upcast the output
    if new_out.dtype != node.out.type.dtype:
        new_out = new_out.astype(node.out.type.dtype)
    if new_out.type.broadcastable != node.out.type.broadcastable:
        new_out = broadcast_arrays(new_out, *node.inputs)[0]
    return [new_out]


def _estimate_upper_bound(var, atleast=None) -> float:
    if atleast is not None and getattr(var.tag, "upper_bound", np.inf) <= atleast:
        # We already proved an upper bound as low as atleast
        return atleast  # type: ignore

    ub = np.inf

    if var.owner is None:
        if isinstance(var, Constant):
            ub = var.data.item()
        else:
            if var.dtype == "bool":
                ub = 1

    elif isinstance(var.owner.op, Elemwise):
        scalar_op = var.owner.op.scalar_op

        if isinstance(scalar_op, Minimum):
            for min_var in var.owner.inputs:
                ub = min(ub, _estimate_upper_bound(min_var, atleast=atleast))
                if ub == atleast:
                    break  # This is enough for us

        elif isinstance(scalar_op, Maximum):
            ub = -np.inf
            for max_var in var.owner.inputs:
                ub = max(ub, _estimate_upper_bound(max_var))
                if ub == np.inf:
                    break  # Don't bother with other inputs

        elif isinstance(scalar_op, Add):
            ub = 0
            for inp in var.owner.inputs:
                ub += _estimate_upper_bound(inp)
                if ub == np.inf:
                    # Don't bother with other inputs
                    break

        elif isinstance(scalar_op, Sub):
            left, right = var.owner.inputs
            ub = _estimate_upper_bound(left)
            if ub != np.inf:
                ub -= _estimate_lower_bound(right)

        elif isinstance(scalar_op, Cast):
            # Trivial case
            if var.type.dtype == "bool":
                ub = 1

            if atleast is None or ub > atleast:
                # We are not satisfied with the trivial upper bound of 1
                [bef_cast] = var.owner.inputs
                bef_ub = _estimate_upper_bound(bef_cast, atleast=atleast)
                if bef_ub != np.inf:
                    # If we actually got a bound, we can cast it
                    bef_ub = np.array(bef_ub).astype(var.dtype).item()
                ub = min(ub, bef_ub)

    var.tag.upper_bound = ub
    return ub


def _estimate_lower_bound(var, atleast=None) -> float:
    if atleast is not None and getattr(var.tag, "lower_bound", -np.inf) >= atleast:
        # We already proved a lower bound as high as atleast
        return atleast  # type: ignore

    lb = -np.inf

    if var.owner is None:
        if isinstance(var, Constant):
            lb = var.data.item()
        else:
            # We can't reason about the lower bound of a root variable besides from dtypes
            if var.dtype == "bool":
                lb = 0
            elif var.dtype in uint_dtypes:
                lb = 0

    elif isinstance(var.owner.op, Shape_i):
        lb = 0

    elif isinstance(var.owner.op, Elemwise):
        scalar_op = var.owner.op.scalar_op

        if isinstance(scalar_op, Minimum):
            lb = np.inf
            for min_var in var.owner.inputs:
                lb = min(lb, _estimate_lower_bound(min_var, atleast=atleast))
                if lb == -np.inf:
                    # Don't bother with other inputs
                    break

        elif isinstance(scalar_op, Maximum):
            for max_var in var.owner.inputs:
                lb = max(lb, _estimate_lower_bound(max_var))
                if lb == atleast:
                    break  # This is enough for us

        elif isinstance(scalar_op, Add):
            lb = 0
            for inp in var.owner.inputs:
                lb += _estimate_lower_bound(inp)
                if lb == -np.inf:
                    # Don't bother with other inputs
                    break

        elif isinstance(scalar_op, Sub):
            left, right = var.owner.inputs
            lb = _estimate_lower_bound(left)
            if lb != -np.inf:
                lb -= _estimate_upper_bound(right)

        elif isinstance(scalar_op, Abs):
            lb = 0  # Guaranteed by abs

            atleast = 3
            # lb=(-5, inf) -> lb(abs)=(0, inf)  -> not enough
            # lb=(3, inf) -> lb(abs)=(0, 5) -> not enough
            # up=(-inf, -3) -> lb(abs) = (3, inf) -> maybe enough

            if atleast is None or lb < atleast:
                # We are not satisfied with the trivial lower bound of 0
                [abs_var] = var.owner.inputs
                lb = max(lb, _estimate_lower_bound(abs_var, atleast=atleast))
                if atleast is None or lb < atleast:
                    # If we are still not satisfied, we can try to estimate the upper bound
                    # if upper bound is smaller than the negative of the requested value we're good
                    ub_negative = _estimate_upper_bound(
                        abs_var, atleast=-atleast if atleast is not None else None
                    )
                    if ub_negative < -lb:
                        # We learned something more precise
                        assert ub_negative < 0
                        lb = abs(ub_negative)

        elif isinstance(scalar_op, Exp | Sqr | Log | Log1p):
            # Monotonic functions
            if atleast is not None:
                if isinstance(scalar_op, Exp):
                    atleast = np.log(atleast)
                elif isinstance(scalar_op, Sqr):
                    atleast = np.sqrt(atleast)
                elif isinstance(scalar_op, Log):
                    atleast = np.log(atleast)
                elif isinstance(scalar_op, Log1p):
                    atleast = np.expm1(atleast)

            np_func = import_func_from_string(scalar_op.nfunc_spec[0])
            lb = np_func(_estimate_lower_bound(var, atleast))

        elif isinstance(scalar_op, Cast):
            # Some trivial cases for casts that round to zero
            if var.type.dtype == "bool" or var.type.dtype in uint_dtypes:
                lb = 0

            if atleast is None or lb < atleast:
                # We are not satisfied with the trivial lower bound of 0
                [bef_cast] = var.owner.inputs
                bef_lb = _estimate_lower_bound(bef_cast, atleast=atleast)
                if bef_lb != -np.inf:
                    # If we actually got a bound, we can cast it
                    bef_lb = np.array(bef_lb).astype(var.dtype).item()
                lb = max(lb, bef_lb)

    var.tag.lower_bound = lb
    return lb


# registered as a graph rewrite below to avoid too many calls
@node_rewriter([switch])
def local_useless_switch_branches(fgraph, node):
    if node.out.dtype not in discrete_dtypes:
        return None

    cond, true_branch, false_branch = node.inputs
    if not (
        cond.owner is not None
        and isinstance(cond.owner.op, Elemwise)
        and isinstance(cond.owner.op.scalar_op, DIRECTIONAL_COMPARISON_OPS)
    ):
        return None

    left, right = cond.owner.inputs

    scalar_op = cond.owner.op.scalar_op
    if isinstance(scalar_op, LE):
        # Same as GE, but with left and right swapped
        scalar_op = GE
        left, right = right, left
    elif isinstance(scalar_op, LT):
        # Same as GT, but with left and right swapped
        scalar_op = GT
        left, right = right, left

    if isinstance(scalar_op, GE):
        # left >= right is useless when lower bound of left >= upper bound of right
        # (5, inf) >= (-inf, 5) is always True
        left_lb = _estimate_lower_bound(left)
        if left_lb != -np.inf and left_lb >= _estimate_upper_bound(
            right, atleast=left_lb
        ):
            return [true_branch]
        # or upper bound of left < lower bound of right
        # (-inf, 5) >= (5+eps, inf) is always false
        left_ub = _estimate_upper_bound(left)
        if left_ub != np.inf and left_ub < _estimate_lower_bound(
            left, atleast=left_ub + 1e-5
        ):
            return [false_branch]

    elif isinstance(scalar_op, GT):
        # left > right is useless when lower bound of left > upper bound of right
        # (5, inf) > (-inf, 5-eps) is always True
        left_lb = _estimate_lower_bound(left)
        if left_lb != -np.inf and left_lb > _estimate_upper_bound(
            right, atleast=left_lb - 1e-5
        ):
            return [true_branch]
        # or upper bound of left <= lower bound of right
        # (-inf, 5) > (5, inf) is always false
        left_ub = _estimate_upper_bound(left)
        if left_ub != np.inf and left_ub <= _estimate_lower_bound(
            left, atleast=left_ub
        ):
            return [false_branch]


# registered as a graph rewrite below to avoid too many calls
@node_rewriter([minimum, maximum])
def local_useless_extremum_branches(fgraph, node):
    """Rewrite useless branches in a maximum/minimum based on lower-upper bound reasoning.

    maximum(x, y, z) -> if any xyz's upper bound <= yzx' lower bound, i can be discarded.

    Example:
        maximum(0, shape(x), y) -> maximum(shape(x), y)  since shape(x) is already lower bounded by zero
        maximum(2, minimum(x, 1)) -> 2, since minimum(x, y) is already upper bounded by 1
        maximum(1, minimum(x, 1-shape(y)) -> 1
    """
    [old_out] = node.outputs
    if not all(old_out.type.broadcastable):
        return None

    if isinstance(node.op.scalar_op, Minimum):
        informative_bound = _estimate_lower_bound
        uninformative_bound_value = -np.inf
        reverse = True
        reverse_bound = _estimate_upper_bound
        logical_comp = operator.le
    else:
        informative_bound = _estimate_upper_bound
        uninformative_bound_value = np.inf
        reverse = False
        reverse_bound = _estimate_lower_bound
        logical_comp = operator.ge

    inputs, bounds = zip(
        *sorted(
            ((inp, informative_bound(inp)) for inp in node.inputs),
            key=lambda x: x[1],
            reverse=reverse,
        ),
        strict=False,  # useless
    )

    while len(bounds) > 1 and bounds[0] != uninformative_bound_value:
        most_restricted_bound = bounds[0]

        # If any other branch as a lower bound >= upper_bound, they can be discarded
        for other_inp in inputs[1:]:
            if logical_comp(
                reverse_bound(other_inp, atleast=most_restricted_bound),
                most_restricted_bound,
            ):
                # We can remove the restricted bound input
                inputs = inputs[1:]
                bounds = bounds[1:]
                break
        else:  # no break
            break

    if len(inputs) == 1:
        [new_out] = inputs
    elif len(inputs) < len(node.inputs):
        new_out = minimum(*inputs)
        copy_stack_trace(old_out, new_out)
    else:
        return None

    # Removed branches may have broadcast or upcast the output
    if new_out.dtype != old_out.type.dtype:
        new_out = new_out.astype(old_out.type.dtype)
    if new_out.type.ndim != old_out.type.ndim:
        new_out = atleast_Nd(new_out, old_out.type.ndim)

    return [new_out]


# This rewrite can be expensive, call it once going from out to in
# After all the local rewrites in canonicalize have been applied
# out2in is preferrable because we truncate more on the outputs, and any
# domain bound analysis that go up to the inputs are cached anyway.
optdb["canonicalize"].register(
    local_useless_extremum_branches.__name__,
    out2in(local_useless_switch_branches, local_useless_extremum_branches),
    "fast_run",
)
