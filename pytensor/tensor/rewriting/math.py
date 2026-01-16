r"""Rewrites for the `Op`\s in :mod:`pytensor.tensor.math`."""

import itertools
import operator
from collections import defaultdict
from functools import partial, reduce

import numpy as np

import pytensor.scalar.basic as ps
import pytensor.scalar.math as ps_math
from pytensor.graph.basic import Constant, Variable
from pytensor.graph.rewriting.basic import (
    NodeRewriter,
    PatternNodeRewriter,
    SequentialNodeRewriter,
    copy_stack_trace,
    in2out,
    node_rewriter,
)
from pytensor.graph.rewriting.utils import get_clients_at_depth
from pytensor.tensor.basic import (
    Alloc,
    Join,
    MakeVector,
    alloc,
    as_tensor_variable,
    cast,
    constant,
    expand_dims,
    get_underlying_scalar_constant_value,
    moveaxis,
    ones_like,
    register_infer_shape,
    split,
    switch,
    zeros,
    zeros_like,
)
from pytensor.tensor.elemwise import CAReduce, DimShuffle, Elemwise
from pytensor.tensor.exceptions import NotScalarConstantError
from pytensor.tensor.extra_ops import broadcast_arrays, concat_with_broadcast
from pytensor.tensor.math import (
    Dot,
    Prod,
    Sum,
    _conj,
    _dot,
    _matmul,
    add,
    arccosh,
    arcsinh,
    arctanh,
    cosh,
    deg2rad,
    digamma,
    dot,
    erf,
    erfc,
    exp,
    expm1,
    ge,
    int_div,
    isinf,
    kve,
    le,
    log,
    log1mexp,
    log1p,
    log1pexp,
    makeKeepDims,
    maximum,
    mul,
    neg,
    polygamma,
    prod,
    rad2deg,
    reciprocal,
    sigmoid,
    sign,
    sinh,
    softplus,
    sqr,
    sqrt,
    sub,
    tanh,
    tri_gamma,
    true_div,
    variadic_add,
    variadic_mul,
)
from pytensor.tensor.math import abs as pt_abs
from pytensor.tensor.math import max as pt_max
from pytensor.tensor.math import pow as pt_pow
from pytensor.tensor.math import sum as pt_sum
from pytensor.tensor.rewriting.basic import (
    alloc_like,
    broadcasted_by,
    local_fill_sink,
    register_canonicalize,
    register_specialize,
    register_stabilize,
    register_uncanonicalize,
    register_useless,
)
from pytensor.tensor.rewriting.blockwise import blockwise_of
from pytensor.tensor.rewriting.elemwise import apply_local_dimshuffle_lift
from pytensor.tensor.rewriting.linalg import is_matrix_transpose
from pytensor.tensor.shape import Shape, Shape_i, specify_shape
from pytensor.tensor.slinalg import BlockDiagonal
from pytensor.tensor.subtensor import Subtensor
from pytensor.tensor.type import (
    complex_dtypes,
    uint_dtypes,
    values_eq_approx_remove_inf,
    values_eq_approx_remove_inf_nan,
    values_eq_approx_remove_nan,
)
from pytensor.tensor.variable import (
    TensorConstant,
    TensorVariable,
)


def scalarconsts_rest(inputs, elemwise=True, only_process_constants=False):
    """Partition a list of variables into two kinds:
    scalar constants, and the rest."""
    consts = []
    origconsts = []
    nonconsts = []
    for i in inputs:
        try:
            v = get_underlying_scalar_constant_value(
                i, elemwise=elemwise, only_process_constants=only_process_constants
            )
            consts.append(v)
            origconsts.append(i)
        except NotScalarConstantError:
            nonconsts.append(i)
    return consts, origconsts, nonconsts


@register_canonicalize("shape_unsafe")
@register_stabilize("shape_unsafe")
@node_rewriter([Dot])
def local_0_dot_x(fgraph, node):
    x, y = node.inputs
    if (
        get_underlying_scalar_constant_value(
            x, only_process_constants=False, raise_not_constant=False
        )
        == 0
        or get_underlying_scalar_constant_value(
            y, only_process_constants=False, raise_not_constant=False
        )
        == 0
    ):
        return [zeros((x.shape[0], y.shape[1]), dtype=node.outputs[0].type.dtype)]


@register_stabilize
@node_rewriter([blockwise_of(BlockDiagonal)])
def local_block_diag_dot_to_dot_block_diag(fgraph, node):
    r"""
    Perform the rewrite ``dot(block_diag(A, B), C) -> concat(dot(A, C), dot(B, C))``

    BlockDiag results in the creation of a matrix of shape ``(n1 * n2, m1 * m2)``. Because dot has complexity
    of approximately O(n^3), it's always better to perform two dot products on the smaller matrices, rather than
    a single dot on the larger matrix.
    """
    # Check that the BlockDiagonal is an input to a Dot node:
    for client in itertools.chain.from_iterable(
        get_clients_at_depth(fgraph, node, depth=i) for i in [1, 2]
    ):
        if client.op not in (_dot, _matmul):
            continue

        [blockdiag_result] = node.outputs
        blockdiag_inputs = node.inputs

        dot_op = client.op

        try:
            client_idx = client.inputs.index(blockdiag_result)
        except ValueError:
            # If the blockdiag result is not an input to the dot, there is at least one Op between them.
            # We allow left expand_dims (DimShuffle), which is introduced automatically by Blockwise to equalize number of batch dims,
            # But does not change the semantics of the graph
            for ancestor in client.inputs:
                if (
                    ancestor.owner is not None
                    and (
                        isinstance(ancestor.owner.op, DimShuffle)
                        and ancestor.owner.op.is_left_expand_dims
                    )
                    and blockdiag_result in ancestor.owner.inputs
                ):
                    client_idx = client.inputs.index(ancestor)
                    break
            else:  # no-break
                # Not a simple left expand_dims between dot and block_diag
                return None

        other_input = client.inputs[1 - client_idx]

        split_axis = -2 if client_idx == 0 else -1
        split_size_axis = -1 if client_idx == 0 else -2

        other_dot_input_split = split(
            other_input,
            splits_size=[
                component.shape[split_size_axis] for component in blockdiag_inputs
            ],
            n_splits=len(blockdiag_inputs),
            axis=split_axis,
        )

        split_dot_results = [
            dot_op(component, other_split)
            if client_idx == 0
            else dot_op(other_split, component)
            for component, other_split in zip(blockdiag_inputs, other_dot_input_split)
        ]
        new_output = concat_with_broadcast(split_dot_results, axis=split_axis)

        copy_stack_trace(node.outputs[0], new_output)
        return {client.outputs[0]: new_output}


@register_canonicalize
@node_rewriter([Dot, _matmul])
def local_lift_transpose_through_dot(fgraph, node):
    r"""Perform the rewrite ``dot(x,y).T -> dot(y.T, x.T)``.

    These rewrites "lift" (propagate towards the inputs) `DimShuffle`
    through dot product.  It allows to put the graph in a more standard shape,
    and to later merge consecutive `DimShuffle`\s.
    """

    clients = fgraph.clients[node.out]

    if len(clients) != 1:
        # If the dot is used in more than one place, we don't want to duplicate it
        return None

    [(client, _)] = clients

    if not (isinstance(client.op, DimShuffle) and is_matrix_transpose(client.out)):
        return None

    x, y = node.inputs
    # Output is dot product of transposed inputs in reverse order
    ret = node.op(y.mT, x.mT)

    # Copy over stack trace to output from result of dot-product
    copy_stack_trace(node.out, ret)

    return {client.out: ret}


def _batched_matmul_to_core_matmul(fgraph, node, allow_reshape: bool):
    """Move batch dimensions of matmul operands to core matmul

    Example, if x has batch dimensions that don't overlap with batch dimensions of y
    x @ y -> (x.reshape(-1, x.shape[-1]) @ y).reshape(*x.shape[:-1], y.shape[-1])

    It also works for batch dimensions of y that don't overlap with batch dimensions of x

    The rewrite only uses reshape when mixing dimensions, and it can refuse to apply if `allow_reshape=False`
    """

    x, y = node.inputs
    batch_ndim = node.op.batch_ndim(node)

    x_axis_to_merge = [
        i
        for i, (bcast_x, bcast_y) in enumerate(
            zip(x.type.broadcastable[:-2], y.type.broadcastable[:-2])
        )
        if bcast_y and not bcast_x
    ]

    y_axis_to_merge = [
        i
        for i, (bcast_x, bcast_y) in enumerate(
            zip(x.type.broadcastable[:-2], y.type.broadcastable[:-2])
        )
        if bcast_x and not bcast_y
    ]

    if not (x_axis_to_merge or y_axis_to_merge):
        return None

    x_shape = tuple(x.shape)
    y_shape = tuple(y.shape)
    x_is_row = x.type.broadcastable[-2]
    y_is_col = y.type.broadcastable[-1]
    n_x_axis_to_merge = len(x_axis_to_merge)
    n_y_axis_to_merge = len(y_axis_to_merge)
    n_axis_to_merge = n_x_axis_to_merge + n_y_axis_to_merge

    x_stacked, y_stacked = x, y
    dims_were_merged = False

    if n_x_axis_to_merge:
        # ravel batch dimensions of x on the core (m) axis
        x_axis_destination = tuple(range(-n_x_axis_to_merge - 2, -2))
        x_stacked = moveaxis(x, x_axis_to_merge, x_axis_destination)
        if x_is_row:
            # x was a row matrix, squeeze it to clean up the graph
            x_stacked = x_stacked.squeeze(-2)
        if n_x_axis_to_merge > 1 or not x_is_row:
            if not allow_reshape:
                # TODO: We could allow the y rewrite to go on
                # Or just move one axis (the largest) if x is row
                return None

            # Ravel moved batch dims together with (m) if needed
            x_stacked_shape = tuple(x_stacked.shape)
            x_stacked = x_stacked.reshape(
                (*x_stacked_shape[: batch_ndim - n_x_axis_to_merge], -1, x_shape[-1])
            )
            dims_were_merged = True

    if n_y_axis_to_merge:
        # ravel batch dimensions of y on the core (n) axis
        y_axis_destination = tuple(range(-n_y_axis_to_merge - 1, -1))
        y_stacked = moveaxis(y, y_axis_to_merge, y_axis_destination)
        if y_is_col:
            # y was a column matrix, squeeze it to clean up the graph
            y_stacked = y_stacked.squeeze(-1)
        if n_y_axis_to_merge > 1 or not y_is_col:
            if not allow_reshape:
                # TODO: We could allow the x rewrite to go on
                # Or just move one axis (the largest) if y is col
                return None
            # Ravel moved batch dims together with (n) if needed
            y_stacked_shape = tuple(y_stacked.shape)
            y_stacked = y_stacked.reshape(
                (*y_stacked_shape[: batch_ndim - n_y_axis_to_merge], y_shape[-2], -1)
            )
            dims_were_merged = True

    # Squeeze x_dims corresponding to merged dimensions of y
    x_axis_to_squeeze = np.array(y_axis_to_merge)
    for i in reversed(x_axis_to_merge):
        # The corresponding dimensions of y may have shifted when we merged dimensions of x
        x_axis_to_squeeze[x_axis_to_squeeze > i] -= 1
    x_stacked = x_stacked.squeeze(tuple(x_axis_to_squeeze))

    # Same for y
    y_axis_to_squeeze = np.array(x_axis_to_merge)
    for i in reversed(y_axis_to_merge):
        y_axis_to_squeeze[y_axis_to_squeeze > i] -= 1
    y_stacked = y_stacked.squeeze(tuple(y_axis_to_squeeze))

    out_stacked = x_stacked @ y_stacked

    # Split back any merged dimensions
    if dims_were_merged:
        x_merged_shapes = [x_shape[i] for i in x_axis_to_merge]
        if not x_is_row:
            # Otherwise we handle that later with expand_dims, which is cleaner
            x_merged_shapes.append(x_shape[-2])
        y_merged_shapes = [y_shape[i] for i in y_axis_to_merge]
        if not y_is_col:
            # Otherwise we handle that later with expand_dims, which is cleaner
            y_merged_shapes.append(y_shape[-1])
        out_stacked_shape = tuple(out_stacked.shape)
        out_unstacked = out_stacked.reshape(
            (
                *out_stacked_shape[: batch_ndim - n_axis_to_merge],
                *x_merged_shapes,
                *y_merged_shapes,
            )
        )
    else:
        out_unstacked = out_stacked

    # Add back dummy row, col axis
    # We do this separately to avoid the reshape as much as we can
    if y_is_col and (n_y_axis_to_merge or dims_were_merged):
        out_unstacked = expand_dims(out_unstacked, -1)
    if x_is_row and (n_x_axis_to_merge or dims_were_merged):
        out_unstacked = expand_dims(out_unstacked, -n_y_axis_to_merge - 2)

    # Move batch axis back to their original location
    source = range(-n_axis_to_merge - 2, 0)
    destination = (*x_axis_to_merge, -2, *y_axis_to_merge, -1)
    out = moveaxis(out_unstacked, source, destination)
    return [out]


@register_canonicalize
@node_rewriter(tracks=[_matmul])
def local_batched_matmul_to_core_matmul(fgraph, node):
    # Allow passing batch dimensions of matmul to core vector / column matrices
    return _batched_matmul_to_core_matmul(fgraph, node, allow_reshape=False)


@register_specialize
@node_rewriter(tracks=[_matmul])
def local_batched_matmul_to_core_matmul_with_reshape(fgraph, node):
    # Allow stacking batch dimensions of matmul with core dimensions, with a reshape operation
    # We only apply this in specialize, because grahs with reshape are hard to work with
    return _batched_matmul_to_core_matmul(fgraph, node, allow_reshape=True)


@register_canonicalize
@register_specialize
@node_rewriter([_matmul, Dot])
def local_dot_to_mul(fgraph, node):
    """Rewrite dots that correspond to multiplication without summation.

    We don't touch outer product without batch-dimensions, to allow rewriting into GER,
    which seems more performant in that case.

    # TODO: Once we blockwise Blas operations we shouldn't do it for outer product with batch-dimensions either
    # TODO: We may still want to canonicalize outer dot as mul, and detect that for GER.
    """
    a, b = node.inputs
    a_static_shape = a.type.shape
    b_static_shape = b.type.shape

    # Check if we have matrix-matrix product: (..., m, 1) * (..., 1, n) -> (..., m, n)
    if not (a_static_shape[-1] == 1 or b_static_shape[-2] == 1):
        return None

    # If it's a core Dot we only rewrite if there's no outer product
    # (1, 1) * (1, n) or (m, 1) * (1, 1)
    # Otherwise we leave as is, so GER can be used instead
    if isinstance(node.op, Dot) and not (
        a_static_shape[-2] == 1 or b_static_shape[-1] == 1
    ):
        return None

    # Add specify_shape for unknown dimensions that must be 1
    # To avoid runtime broadcast error by multiply
    if a.type.shape[-1] != 1:
        a = specify_shape(a, (..., None, 1))
    if b.type.shape[-2] != 1:
        b = specify_shape(b, (..., 1, None))

    new_out = mul(a, b)
    copy_stack_trace(node.out, new_out)
    return [new_out]


for pair in (
    (deg2rad, rad2deg),
    (cosh, arccosh),
    (tanh, arctanh),
    (sinh, arcsinh),
    (_conj, _conj),
    (neg, neg),
    (reciprocal, reciprocal),
):
    # Create a simple PatternNodeRewriter for each pair of opposite ops
    # instead of a general Op that is called to often for very few hits
    for op, inv_op in (pair, reversed(pair)):
        rewrite = PatternNodeRewriter(
            (op, (inv_op, "x")),
            "x",
            allow_multiple_clients=True,
            allow_cast=True,
            name=f"useless_{op}_of_{inv_op}",
        )
        register_canonicalize(rewrite)
        register_specialize(rewrite)

        if op is inv_op:
            break  # Same Op, no need to define two rewrites


@register_canonicalize
@register_specialize
@node_rewriter([log, log1p, exp, expm1])
def local_exp_log(fgraph, node):
    x = node.inputs[0]

    if not (x.owner and isinstance(x.owner.op, Elemwise)):
        return

    prev_op = x.owner.op.scalar_op
    node_op = node.op.scalar_op

    # Case for log(exp(x)) -> x
    if isinstance(prev_op, ps.Exp) and isinstance(node_op, ps.Log):
        new_out = x.owner.inputs[0]
        old_out = node.outputs[0]
        # Exp may have cast integer input to float
        if new_out.dtype != old_out.dtype:
            new_out = cast(new_out, old_out.dtype)
        return [new_out]

    # Case for log1p(expm1(x)) -> x
    if isinstance(prev_op, ps.Expm1) and isinstance(node_op, ps.Log1p):
        new_out = x.owner.inputs[0]
        old_out = node.outputs[0]
        # Expm1 may have cast integer input to float
        if new_out.dtype != old_out.dtype:
            new_out = cast(new_out, old_out.dtype)
        return [new_out]

    # Case for exp(softplus(x)) aka exp(log1pexp) -> 1 + exp(x)
    if isinstance(prev_op, ps_math.Softplus) and isinstance(node_op, ps.Exp):
        x = x.owner.inputs[0]
        return [add(1, exp(x))]

    # Case for expm1(softplus(x)) aka expm1(log1pexp) -> exp(x)
    if isinstance(prev_op, ps_math.Softplus) and isinstance(node_op, ps.Expm1):
        x = x.owner.inputs[0]
        return [exp(x)]


@register_canonicalize
@register_specialize
@node_rewriter([sqrt, sqr])
def local_sqrt_sqr(fgraph, node):
    x = node.inputs[0]

    if not (x.owner and isinstance(x.owner.op, Elemwise)):
        return

    prev_op = x.owner.op.scalar_op
    node_op = node.op.scalar_op

    # Case for sqrt(sqr(x)) -> |x|
    if isinstance(prev_op, ps.Sqrt) and isinstance(node_op, ps.Sqr):
        new_out = pt_abs(x.owner.inputs[0])
        old_out = node.outputs[0]

        # Handle potential integer to float cast by sqr
        if new_out.dtype != old_out.dtype:
            new_out = cast(new_out, old_out.dtype)
        return [new_out]

    # Case for sqr(sqrt(x)) -> x
    if isinstance(prev_op, ps.Sqr) and isinstance(node_op, ps.Sqrt):
        x = x.owner.inputs[0]
        old_out = node.outputs[0]
        new_out = switch(ge(x, 0), x, np.asarray(np.nan, old_out.dtype))

        return [new_out]


@register_specialize
@node_rewriter([log])
def local_log_sqrt(fgraph, node):
    x = node.inputs[0]

    if (
        not x.owner
        or not isinstance(x.owner.op, Elemwise)
        or not isinstance(x.owner.op.scalar_op, ps.Sqrt)
    ):
        return

    # Case for log(sqrt(x)) -> 0.5 * log(x)
    x = x.owner.inputs[0]
    old_out = node.outputs[0]
    new_out = mul(as_tensor_variable(0.5, dtype=x.dtype), log(x))
    if new_out.dtype != old_out.dtype:
        new_out = cast(new_out, old_out.dtype)

    copy_stack_trace(node.out, new_out)
    return [new_out]


@register_specialize
@node_rewriter([exp, expm1, log1pexp, log1mexp])
def local_exp_log_nan_switch(fgraph, node):
    # Rewrites of the kind exp(log...(x)) that require a `nan` switch
    x = node.inputs[0]

    if not (x.owner and isinstance(x.owner.op, Elemwise)):
        return

    prev_op = x.owner.op.scalar_op
    node_op = node.op.scalar_op

    # Case for exp(log(x)) -> x
    if isinstance(prev_op, ps.Log) and isinstance(node_op, ps.Exp):
        x = x.owner.inputs[0]
        old_out = node.outputs[0]
        new_out = switch(ge(x, 0), x, np.asarray(np.nan, old_out.dtype))
        return [new_out]

    # Case for exp(log1p(x)) -> x + 1
    if isinstance(prev_op, ps.Log1p) and isinstance(node_op, ps.Exp):
        x = x.owner.inputs[0]
        old_out = node.outputs[0]
        new_out = switch(ge(x, -1), add(1, x), np.asarray(np.nan, old_out.dtype))
        return [new_out]

    # Case for expm1(log(x)) -> x - 1
    if isinstance(prev_op, ps.Log) and isinstance(node_op, ps.Expm1):
        x = x.owner.inputs[0]
        old_out = node.outputs[0]
        new_out = switch(ge(x, 0), sub(x, 1), np.asarray(np.nan, old_out.dtype))
        return [new_out]

    # Case for expm1(log1p(x)) -> x
    if isinstance(prev_op, ps.Log1p) and isinstance(node_op, ps.Expm1):
        x = x.owner.inputs[0]
        old_out = node.outputs[0]
        new_out = switch(ge(x, -1), x, np.asarray(np.nan, old_out.dtype))
        return [new_out]

    # Case for exp(log1mexp(x)) -> 1 - exp(x)
    if isinstance(prev_op, ps_math.Log1mexp) and isinstance(node_op, ps.Exp):
        x = x.owner.inputs[0]
        old_out = node.outputs[0]
        new_out = switch(le(x, 0), sub(1, exp(x)), np.asarray(np.nan, old_out.dtype))
        return [new_out]

    # Case for expm1(log1mexp(x)) -> -exp(x)
    if isinstance(prev_op, ps_math.Log1mexp) and isinstance(node_op, ps.Expm1):
        x = x.owner.inputs[0]
        old_out = node.outputs[0]
        new_out = switch(le(x, 0), neg(exp(x)), np.asarray(np.nan, old_out.dtype))
        return [new_out]

    # Case for log1pexp(log(x)) -> log1p(x)   (log1pexp aka softplus)
    if isinstance(prev_op, ps.Log) and isinstance(node_op, ps_math.Softplus):
        x = x.owner.inputs[0]
        old_out = node.outputs[0]
        new_out = switch(ge(x, 0), log1p(x), np.asarray(np.nan, old_out.dtype))
        return [new_out]

    # Case for log1mexp(log(x)) -> log1p(-x)
    if isinstance(prev_op, ps.Log) and isinstance(node_op, ps_math.Log1mexp):
        x = x.owner.inputs[0]
        old_out = node.outputs[0]
        new_out = switch(ge(x, 0), log1p(-x), np.asarray(np.nan, old_out.dtype))
        return [new_out]

    # Case for log1mexp(log1mexp(x)) -> x
    if isinstance(prev_op, ps_math.Log1mexp) and isinstance(node_op, ps_math.Log1mexp):
        x = x.owner.inputs[0]
        old_out = node.outputs[0]
        new_out = switch(le(x, 0), x, np.asarray(np.nan, old_out.dtype))
        return [new_out]


@register_canonicalize
@register_specialize
@node_rewriter([Sum])
def local_sumsqr2dot(fgraph, node):
    """
    This rewrite detects
    ``pt.sqr(W.dimshuffle("x", 0, 1) * G.dimshuffle(0, "x", 1) ).sum(axis=(1, 2))``
    and converts it to ``pt.dot(pt.sqr(G), pt.sqr(W).sum(axis=0))``.
    """
    if node.op.axis == (1, 2):
        in1 = node.inputs[0]
        out = node.outputs[0]

        if (
            in1.owner
            and isinstance(in1.owner.op, Elemwise)
            and isinstance(in1.owner.op.scalar_op, ps.Sqr)
        ):
            in_sqr = in1.owner.inputs[0]
            if (
                in_sqr.owner
                and isinstance(in_sqr.owner.op, Elemwise)
                and isinstance(in_sqr.owner.op.scalar_op, ps.Mul)
                and len(in_sqr.owner.inputs) == 2
            ):
                in_mul1, in_mul2 = in_sqr.owner.inputs

                if (
                    isinstance(in_mul1.owner.op, DimShuffle)
                    and in_mul1.owner.op.new_order == ("x", 0, 1)
                    and isinstance(in_mul2.owner.op, DimShuffle)
                    and in_mul2.owner.op.new_order == (0, "x", 1)
                ):
                    W = in_mul1.owner.inputs[0]
                    G = in_mul2.owner.inputs[0]

                    new_out = dot(sqr(G), sqr(W).sum(axis=0))
                    if new_out.dtype != out.dtype:
                        new_out = cast(new_out, dtype=out.dtype)
                    return [new_out]


@register_specialize
@node_rewriter([mul, true_div])
def local_mul_exp_to_exp_add(fgraph, node):
    """
    This rewrite detects e^x * e^y and converts it to e^(x+y).
    Similarly, e^x / e^y becomes e^(x-y).
    """
    exps = [
        n.owner.inputs[0]
        for n in node.inputs
        if n.owner
        and isinstance(n.owner.op, Elemwise)
        and isinstance(n.owner.op.scalar_op, ps.Exp)
    ]
    # Can only do any rewrite if there are at least two exp-s
    if len(exps) >= 2:
        # Mul -> add; TrueDiv -> sub
        orig_op, new_op = mul, add
        if isinstance(node.op.scalar_op, ps.TrueDiv):
            orig_op, new_op = true_div, sub
        new_out = exp(new_op(*exps))
        if new_out.dtype != node.outputs[0].dtype:
            new_out = cast(new_out, dtype=node.outputs[0].dtype)
        # The original Mul may have more than two factors, some of which may not be exp nodes.
        # If so, we keep multiplying them with the new exp(sum) node.
        # E.g.: e^x * y * e^z * w --> e^(x+z) * y * w
        rest = [
            n
            for n in node.inputs
            if not (
                n.owner
                and isinstance(n.owner.op, Elemwise)
                and isinstance(n.owner.op.scalar_op, ps.Exp)
            )
        ]
        if len(rest) > 0:
            new_out = orig_op(new_out, *rest)
            if new_out.dtype != node.outputs[0].dtype:
                new_out = cast(new_out, dtype=node.outputs[0].dtype)
        return [new_out]


@register_specialize
@node_rewriter([mul, true_div])
def local_mul_pow_to_pow_add(fgraph, node):
    """
    This rewrite detects a^x * a^y and converts it to a^(x+y).
    Similarly, a^x / a^y becomes a^(x-y).
    """
    # search for pow-s and group them by their bases
    pow_nodes = defaultdict(list)
    rest = []
    for n in node.inputs:
        if (
            n.owner
            and isinstance(n.owner.op, Elemwise)
            and isinstance(n.owner.op.scalar_op, ps.Pow)
        ):
            base_node = n.owner.inputs[0]
            # exponent is at n.owner.inputs[1], but we need to store the full node
            # in case this particular power node remains alone and can't be rewritten
            pow_nodes[base_node].append(n)
        else:
            rest.append(n)

    # Can only do any rewrite if there are at least two pow-s with the same base
    can_rewrite = [k for k, v in pow_nodes.items() if len(v) >= 2]
    if len(can_rewrite) >= 1:
        # Mul -> add; TrueDiv -> sub
        orig_op, new_op = mul, add
        if isinstance(node.op.scalar_op, ps.TrueDiv):
            orig_op, new_op = true_div, sub
        pow_factors = []
        # Rewrite pow-s having the same base for each different base
        # E.g.: a^x * a^y --> a^(x+y)
        for base in can_rewrite:
            exponents = [n.owner.inputs[1] for n in pow_nodes[base]]
            new_node = base ** new_op(*exponents)
            if new_node.dtype != node.outputs[0].dtype:
                new_node = cast(new_node, dtype=node.outputs[0].dtype)
            pow_factors.append(new_node)
        # Don't forget about those sole pow-s that couldn't be rewriten
        sole_pows = [v[0] for k, v in pow_nodes.items() if k not in can_rewrite]
        # Combine the rewritten pow-s and other, non-pow factors of the original Mul
        # E.g.: a^x * y * b^z * a^w * v * b^t --> a^(x+z) * b^(z+t) * y * v
        if len(pow_factors) > 1 or len(sole_pows) > 0 or len(rest) > 0:
            new_out = orig_op(*pow_factors, *sole_pows, *rest)
            if new_out.dtype != node.outputs[0].dtype:
                new_out = cast(new_out, dtype=node.outputs[0].dtype)
        else:
            # if all factors of the original mul were pows-s with the same base,
            # we can get rid of the mul completely.
            new_out = pow_factors[0]
        return [new_out]


@register_stabilize
@register_specialize
@register_canonicalize
@node_rewriter([add, sub])
def local_expm1(fgraph, node):
    """Detect ``exp(a) - 1`` or ``-1 + exp(a)`` and convert them to ``expm1(a)``."""
    if len(node.inputs) != 2:
        # TODO: handle more than two inputs in add
        return None

    if isinstance(node.op.scalar_op, ps.Sub):
        exp_x, other_inp = node.inputs
        if not (
            exp_x.owner
            and isinstance(exp_x.owner.op, Elemwise)
            and isinstance(exp_x.owner.op.scalar_op, ps.Exp)
            and get_underlying_scalar_constant_value(
                other_inp, raise_not_constant=False
            )
            == 1
        ):
            return None
    else:
        # Try both orders
        other_inp, exp_x = node.inputs
        for i in range(2):
            if i == 1:
                other_inp, exp_x = exp_x, other_inp
            if (
                exp_x.owner
                and isinstance(exp_x.owner.op, Elemwise)
                and isinstance(exp_x.owner.op.scalar_op, ps.Exp)
                and get_underlying_scalar_constant_value(
                    other_inp, raise_not_constant=False
                )
                == -1
            ):
                break
        else:  # no break
            return None

    [old_out] = node.outputs

    [x] = exp_x.owner.inputs
    if x.type.broadcastable != old_out.type.broadcastable:
        x = broadcast_arrays(x, other_inp)[0]

    new_out = expm1(x)

    if new_out.dtype != old_out.dtype:
        new_out = cast(new_out, dtype=old_out.dtype)

    if not old_out.type.is_super(new_out.type):
        return None

    return [new_out]


@register_specialize
@register_stabilize
@register_canonicalize
@node_rewriter([mul])
def local_mul_switch_sink(fgraph, node):
    """
    This rewrite makes the following changes in the graph:

        pt.mul(A, pt.switch(cond, 0, iff), B) -> pt.switch(cond, 0, pt.mul(A, B, iff))
        pt.mul(A, pt.switch(cond, ift, 0), B) -> pt.switch(cond, pt.mul(A, B, ift), 0)

    ``A`` and ``B`` being several (or none) symbolic variables.
    This is useful because ``A`` and ``B`` may not be numerically stable and give
    NaN or inf values for cases where the switch returns 0.
    With this rewrite ``pt.grad(pt.switch(...))`` has the right behavior.

    Examples
    --------

        x -> f(x)
        x -> g(x)
        y = pt.switch(cond, f(x), g(x))

    without the rewrite:

        pt.grad(y, x) -> grad(f(x), x) * grad(y, f(x)) + grad(g(x), x) * grad(y, g(x))

    with the rewrite

        pt.grad(y, x) -> switch(cond, grad(f(x), x), 0) + switch(cond, 0, grad(g(x), x))

    This will be particularly useful for the lazy ``if`` because we skip an entire
    part of the graph.

    """
    for mul_inp_idx, mul_inp in enumerate(node.inputs):
        if mul_inp.owner and mul_inp.owner.op == switch:
            switch_node = mul_inp.owner
            # Look for a zero as the first or second branch of the switch
            for branch in range(2):
                zero_inp = underlying_zero = switch_node.inputs[1 + branch]

                # Allow zero inside a DimShuffle or Alloc
                if zero_inp.owner is not None and isinstance(
                    zero_inp.owner.op, DimShuffle | Alloc
                ):
                    underlying_zero = zero_inp.owner.inputs[0]

                if not (
                    isinstance(underlying_zero, TensorConstant)
                    and underlying_zero.unique_value == 0
                ):
                    continue

                switch_cond = switch_node.inputs[0]
                other_switch_input = switch_node.inputs[1 + (1 - branch)]

                listmul = list(node.inputs)
                listmul[mul_inp_idx] = other_switch_input
                fmul = mul(*listmul)

                # Copy over stacktrace for elementwise multiplication op
                # from previous elementwise multiplication op.
                # An error in the multiplication (e.g. errors due to
                # inconsistent shapes), will point to the
                # multiplication op.
                copy_stack_trace(node.outputs, fmul)

                if branch == 0:
                    fct = switch(switch_cond, zero_inp, fmul)
                else:
                    fct = switch(switch_cond, fmul, zero_inp)

                # Tell debug_mode than the output is correct, even if nan disappear
                fct.tag.values_eq_approx = values_eq_approx_remove_nan

                # Copy over stacktrace for switch op from both previous
                #  elementwise multiplication op and previous switch op,
                # because an error in this part can be caused by either
                # of the two previous ops.
                copy_stack_trace(node.outputs + switch_node.outputs, fct)
                return [fct]


@register_canonicalize
@node_rewriter([true_div, int_div])
def local_div_switch_sink(fgraph, node):
    """
    This rewrite makes the following changes in the graph:

        pt.div(pt.switch(cond, 0, iff), A) -> pt.switch(cond, 0, pt.div(iff, A))
        pt.div(pt.switch(cond, ift, 0), A) -> pt.switch(cond, pt.div(ift, A), 0)

    where ``A`` is a symbolic variable.

    This is useful because ``A`` may not be numerically stable and give
    ``nan`` or ``inf`` values for cases where the switch returns 0.

    See `local_mul_switch_sink` for more details.

    """
    num, denom = node.inputs

    if num.owner and num.owner.op == switch:
        switch_node = num.owner
        # Look for a zero as the first or second branch of the switch
        for branch in range(2):
            zero_inp = underlying_zero = switch_node.inputs[1 + branch]

            # Allow zero inside a DimShuffle or Alloc
            if zero_inp.owner is not None and isinstance(
                zero_inp.owner.op, DimShuffle | Alloc
            ):
                underlying_zero = zero_inp.owner.inputs[0]

            if not (
                isinstance(underlying_zero, TensorConstant)
                and underlying_zero.unique_value == 0
            ):
                continue

            switch_cond = switch_node.inputs[0]
            other_switch_input = switch_node.inputs[1 + (1 - branch)]

            fdiv = node.op(other_switch_input, denom)

            # Copy over stacktrace for elementwise division op
            # from previous elementwise multiplication op.
            # An error in the division (e.g. errors due to
            # inconsistent shapes or division by zero),
            # will point to the new division op.
            copy_stack_trace(node.outputs, fdiv)

            if branch == 0:
                fct = switch(switch_cond, zero_inp, fdiv)
            else:
                fct = switch(switch_cond, fdiv, zero_inp)

            # Tell debug_mode than the output is correct, even if nan disappear
            fct.tag.values_eq_approx = values_eq_approx_remove_nan

            # Copy over stacktrace for switch op from both previous
            # elementwise division op and previous switch op,
            # because an error in this part can be caused by either
            # of the two previous ops.
            copy_stack_trace(node.outputs + switch_node.outputs, fct)
            return [fct]


class AlgebraicCanonizer(NodeRewriter):
    r"""A `Rewriter` that rewrites algebraic expressions.

    The variable is a `node_rewriter`. It is best used
    with a `WalkingGraphRewriter` in in-to-out order.

    Usage: ``AlgebraicCanonizer(main, inverse, reciprocal, calculate)``

    Parameters
    ----------
    main
        A suitable `Op` class that is commutative, associative and
        takes one to an arbitrary number of inputs, e.g. add or
        mul
    inverse
        An `Op` class such that ``inverse(main(x, y), y) == x``
        (e.g. `sub` or `true_div`).
    reciprocal
        A function such that ``main(x, reciprocal(y)) == inverse(x, y)``
        (e.g. `neg` or `reciprocal`).
    calculate
        Function that takes a list of `numpy.ndarray` instances
        for the numerator, another list for the denumerator,
        and calculates ``inverse(main(\*num), main(\*denum))``. It
        takes a keyword argument, `aslist`. If ``True``, the value
        should be returned as a list of one element, unless
        the value is such that ``value = main()``. In that case,
        the return value should be an empty list.

    Examples
    --------
    >>> import pytensor.tensor as pt
    >>> from pytensor.tensor.rewriting.math import AlgebraicCanonizer
    >>> add_canonizer = AlgebraicCanonizer(add, sub, neg, \
    ...                                    lambda n, d: sum(n) - sum(d))
    >>> mul_canonizer = AlgebraicCanonizer(mul, true_div, reciprocal, \
    ...                                    lambda n, d: prod(n) / prod(d))

    Examples of rewrites `mul_canonizer` can perform:

        | x / x -> 1
        | (x * y) / x -> y
        | x / y / x -> 1 / y
        | x / y / z -> x / (y * z)
        | x / (y / z) -> (x * z) / y
        | (a / b) * (b / c) * (c / d) -> a / d
        | (2.0 * x) / (4.0 * y) -> (0.5 * x) / y
        | 2 * x / 2 -> x
        | x * y * z -> Elemwise(mul){x,y,z} #only one pass over the memory.
        |           !-> Elemwise(mul){x,Elemwise(mul){y,z}}

    """

    def __init__(self, main, inverse_fn, reciprocal_fn, calculate, use_reciprocal=True):
        self.main = main
        self.inverse = inverse_fn
        self.reciprocal = reciprocal_fn
        self.calculate = calculate
        self.use_reciprocal = use_reciprocal

        self.external_simplifiers = []

    def add_simplifier(self, simplifier, reason):
        self.external_simplifiers.append((reason, simplifier))

    def tracks(self):
        return [self.main, self.inverse, self.reciprocal]

    def get_num_denum(self, inp):
        r"""
        This extract two lists, ``num`` and ``denum``, such that the input is:
        ``self.inverse(self.main(\*num), self.main(\*denum))``. It returns
        the two lists in a ``(num, denum)`` pair.

        For example, for main, inverse and ``reciprocal = \*, / and inv()``,

        | input -> returned value (num, denum)

        | x*y -> ([x, y], [])
        | inv(x) -> ([], [x])
        | inv(x) * inv(y) -> ([], [x, y])
        | x*y/z -> ([x, y], [z])
        | log(x) / y * (z + x) / y -> ([log(x), z + x], [y, y])
        | (((a / b) * c) / d) -> ([a, c], [b, d])
        | a / (b / c) -> ([a, c], [b])
        | log(x) -> ([log(x)], [])
        | x**y -> ([x**y], [])
        | x * y * z -> ([x, y, z], [])

        """
        # This function is recursive.  The idea is that there is a
        # get_num_denum recursion in which the internal ops are all
        # one of (main, inverse, reciprocal, DimShuffle) and the
        # internal data nodes all have the dtype of the 'input'
        # argument. The leaf-Variables of the graph covered by the
        # recursion may be of any Variable type.

        if inp.owner is None or inp.owner.op not in [
            self.main,
            self.inverse,
            self.reciprocal,
        ]:
            if inp.owner and isinstance(inp.owner.op, DimShuffle):
                # If input is a DimShuffle of some input which does
                # something like this:

                # * change a vector of length N into a 1xN row matrix
                # * change a scalar into a 1x1x1 tensor
                # * in general, complete the shape of a tensor
                #   with broadcastable 1s to the *left*
                # Then we will simply discard the DimShuffle and return
                # the num/denum of its input
                dsn = inp.owner  # dimshuffle node
                dsop = dsn.op  # dimshuffle op

                # the first input of the dimshuffle i.e. the ndarray to redim
                dsi0 = dsn.inputs[0]

                # The compatible order is a DimShuffle "new_order" of the form:
                # ('x', ..., 'x', 0, 1, 2, ..., dimshuffle_input.type.ndim)

                # That kind of DimShuffle only adds broadcastable
                # dimensions on the left, without discarding any
                # existing broadcastable dimension and is inserted
                # automatically by Elemwise when the inputs have
                # different numbers of dimensions (hence why we can
                # discard its information - we know we can retrieve it
                # later on).
                compatible_order = ("x",) * (inp.type.ndim - dsi0.type.ndim) + tuple(
                    range(dsi0.type.ndim)
                )
                if dsop.new_order == compatible_order:
                    # If the "new_order" is the one we recognize,
                    # we return the num_denum of the dimshuffled input.
                    return self.get_num_denum(inp.owner.inputs[0])
                else:
                    # This is when the input isn't produced by main,
                    # inverse or reciprocal.
                    return [inp], []
            else:
                return [inp], []
        num = []
        denum = []
        parent = inp.owner

        # We get the (num, denum) pairs for each input
        # pairs = [self.get_num_denum(input2) if input2.type.dtype ==
        # input.type.dtype else ([input2], []) for input2 in
        # parent.inputs]
        pairs = [self.get_num_denum(input2) for input2 in parent.inputs]

        if parent.op == self.main:
            # If we have main(x, y, ...), numx, denumx, numy, denumy, ...
            # then num is concat(numx, numy, num...) and denum is
            # concat(denumx, denumy, denum...) note that main() can have any
            # number of arguments >= 0 concat is list concatenation
            num = reduce(list.__iadd__, map(operator.itemgetter(0), pairs))
            denum = reduce(list.__iadd__, map(operator.itemgetter(1), pairs))
        elif parent.op == self.inverse:
            # If we have inverse(x, y), numx, denumx, numy and denumy
            # then num is concat(numx, denumy) and denum is
            # concat(denumx, numy) note that inverse() is binary
            num = pairs[0][0] + pairs[1][1]
            denum = pairs[0][1] + pairs[1][0]
        elif parent.op == self.reciprocal:
            # If we have reciprocal(x), numx, denumx
            # then num is denumx and denum is numx
            # note that reciprocal() is unary
            num = pairs[0][1]
            denum = pairs[0][0]
        return num, denum

    def merge_num_denum(self, num, denum):
        r"""
        Utility function which takes two lists, num and denum, and
        returns something which is equivalent to inverse(main(\*num),
        main(\*denum)), but depends on the length of num and the length
        of denum (in order to minimize the number of operations).

        Let n = len(num) and d = len(denum):

        | n=0, d=0: neutral element (given by self.calculate([], []))
        |           (for example, this would be 0 if main is addition
        |           and 1 if main is multiplication)
        | n=1, d=0: num[0]
        | n=0, d=1: reciprocal(denum[0])
        | n=1, d=1: inverse(num[0], denum[0])
        | n=0, d>1: reciprocal(main(\*denum))
        | n>1, d=0: main(\*num)
        | n=1, d>1: inverse(num[0], main(\*denum))
        | n>1, d=1: inverse(main(\*num), denum[0])
        | n>1, d>1: inverse(main(\*num), main(\*denum))

        Given the values of n and d to which they are associated, all
        of the above are equivalent to:
        inverse(main(\*num), main(\*denum))

        """

        ln, ld = len(num), len(denum)
        if not ln and not ld:
            return as_tensor_variable(self.calculate([], []))
        if not ln:
            if self.use_reciprocal:
                return self.reciprocal(self.merge_num_denum(denum, []))
            else:
                ln = [self.calculate([], [], aslist=False)]
        if not ld:
            if ln == 1:
                # num[0] should always be a variable
                assert isinstance(num[0], Variable)
                return num[0]
            else:
                return self.main(*num)
        return self.inverse(
            self.merge_num_denum(num, []), self.merge_num_denum(denum, [])
        )

    def simplify(self, num, denum, out_type):
        """
        Shorthand for:

        .. code-block:: python

            self.simplify_constants(*self.simplify_factors(num, denum))

        """
        rval = self.simplify_constants(
            *self.simplify_factors(num, denum), out_type=out_type
        )
        for reason, simplifier in self.external_simplifiers:
            # TODO: document that 'reason' is associated with this
            #       simplification to help auditing when things go
            #       wrong
            rval = simplifier(*rval)
        return rval

    def simplify_factors(self, num, denum):
        """
        For any Variable r which is both in num and denum, removes it
        from both lists. Modifies the lists inplace. Returns the
        modified lists. For example:

        | [x], [x] -> [], []
        | [x, y], [x] -> [y], []
        | [a, b], [c, d] -> [a, b], [c, d]

        """
        ln = len(num)
        ld = len(denum)
        if ld > 2 and ln > 2:
            # Faster version for "big" inputs.
            while True:
                s = set(num)
                # Inputs can appear multiple times
                redo = len(s) != len(num)
                inter = s.intersection(denum)
                for v in inter:
                    num.remove(v)
                    denum.remove(v)
                if not (redo and inter):
                    break
        else:
            for v in list(num):
                if v in denum:
                    num.remove(v)
                    denum.remove(v)
        return num, denum

    def simplify_constants(self, orig_num, orig_denum, out_type=None):
        """
        Find all constants and put them together into a single constant.

        Finds all constants in orig_num and orig_denum
        and puts them together into a single
        constant. The constant is inserted as the first element of the
        numerator. If the constant is the neutral element, it is
        removed from the numerator.

        Examples
        --------
        Let main be multiplication:

        | [2, 3, x], [] -> [6, x], []
        | [x, y, 2], [4, z] -> [0.5, x, y], [z]
        | [x, 2, y], [z, 2] -> [x, y], [z]

        """
        # Lists representing the numerator and denumerator
        num, denum = [], []

        # Lists representing the *constant* elements of num and denum
        numct, denumct = [], []

        for v in orig_num:
            if isinstance(v, TensorConstant) and v.unique_value is not None:
                # We found a constant in the numerator!
                # We add it to numct
                numct.append(v.unique_value)
            else:
                num.append(v)
        for v in orig_denum:
            if isinstance(v, TensorConstant) and v.unique_value is not None:
                denumct.append(v.unique_value)
            else:
                denum.append(v)

        if self.use_reciprocal or num:
            # This will calculate either:
            # [inverse(main(*numct), main(*denumct))]
            # [] - if inverse(main(*numct), main(*denumct)) is the
            # neutral element
            ct = self.calculate(numct, denumct, aslist=True, out_type=out_type)
        else:
            # This happens if we don't allow the reciprocal and the
            # numerator is empty. That means we will need to represent
            # reciprocal(x) like inverse(neutral_element, x) so
            # we can't allow ct == []
            # TODO: why is this branch needed when merge_num_denum
            # does it for us?
            ct = [self.calculate(numct, denumct, aslist=False, out_type=out_type)]

        # Wrapping ct in a Constant with the right dtype
        ct = [constant(c, dtype=out_type.dtype) for c in ct]

        if orig_num and len(numct) == 1 and len(denumct) == 0 and ct:
            # In that case we should only have one constant in `ct`.
            [var_ct] = ct
            first_num_var = orig_num[0]
            first_num_ct = (
                first_num_var.unique_value
                if isinstance(first_num_var, TensorConstant)
                else None
            )
            if first_num_ct is not None and var_ct.type.values_eq(
                var_ct.data, first_num_ct
            ):
                # This is an important trick :( if it so happens that:
                # * there's exactly one constant on the numerator and none on
                #   the denominator
                # * it's not the neutral element (ct is an empty list in that
                #   case)
                # * the constant is the same as the first argument in the
                #   numerator (we only check the first argument because the
                #   canonizer puts the computed constants first)
                # -> then we return very exactly the original num/denum.
                # If we don't do that the rewrite will just loop
                # infinitely because it will not catch on that there are
                # no changes to be made and every time it will want to
                # replace something by the same thing...
                # Note that it is important to use `values_eq` instead of
                # the == operator, to handle NaN values correctly.
                return orig_num, orig_denum

        return ct + num, denum

    def transform(self, fgraph, node, enforce_tracks=True):
        op = node.op
        if enforce_tracks and (op not in {self.main, self.inverse, self.reciprocal}):
            return False

        assert len(node.outputs) == 1
        out = node.outputs[0]

        out_clients = fgraph.clients.get(out)

        if not out_clients:
            return False

        # check if any of the clients of this node would be part of
        # this canonized graph...  if so, we do nothing and wait for
        # them to be transformed.
        for c, c_idx in out_clients:
            while (
                isinstance(c.op, DimShuffle) and len(fgraph.clients[c.outputs[0]]) <= 1
            ):
                c = fgraph.clients[c.outputs[0]][0][0]
            if c.op in [self.main, self.inverse, self.reciprocal]:
                return False

        # Here we make the canonical version of the graph around this node
        # See the documentation of get_num_denum and simplify
        orig_num, orig_denum = self.get_num_denum(node.outputs[0])
        num, denum = self.simplify(list(orig_num), list(orig_denum), out.type)

        def same(x, y):
            return len(x) == len(y) and all(
                np.all(xe == ye) for xe, ye in zip(x, y, strict=True)
            )

        if (
            same(orig_num, num)
            and same(orig_denum, denum)
            and
            # Check to see if we've collapsed some nested ops.
            not (
                len(orig_denum) == 0
                and
                # Make sure this change would increase the number of vector
                # arguments--decreasing the number of unnecessary `self.main`
                # nodes.
                len(node.inputs) < len(orig_num)
            )
            and
            # Do a similar check for the reciprocal op.
            not (
                self.use_reciprocal
                and node.op == self.reciprocal
                and len(orig_num) == 0
                and node.inputs[0].owner
                and len(node.inputs[0].owner.inputs) < len(orig_denum)
            )
        ):
            return False

        new = self.merge_num_denum(num, denum)
        if new.type.dtype != out.type.dtype:
            new = cast(new, out.type.dtype)

        if new.type.broadcastable != out.type.broadcastable:
            new = broadcast_arrays(new, *node.inputs)[0]

        if (new.type.dtype == out.type.dtype) and (
            new.type.broadcastable == out.type.broadcastable
        ):
            new.tag.values_eq_approx = values_eq_approx_remove_inf_nan
            copy_stack_trace(out, new)
            return [new]
        else:
            return False

    def __str__(self):
        return getattr(
            self,
            "name",
            f"AlgebraicCanonizer({self.main}, {self.inverse}, {self.reciprocal})",
        )


def mul_calculate(num, denum, aslist=False, out_type=None):
    if not num and not denum:
        # Smallest 1 possible.
        if aslist:
            return []
        else:
            return np.int8(1)

    # Make sure we do not accidentally upcast data types.
    if out_type is None:
        out_dtype = ps.upcast(*[v.dtype for v in (num + denum)])
    else:
        out_dtype = out_type.dtype
    one = np.asarray(1, dtype=out_dtype)

    v = reduce(np.multiply, num, one) / reduce(np.multiply, denum, one)
    if aslist:
        if np.all(v == 1):
            return []
        else:
            return [v]
    return v


local_mul_canonizer = AlgebraicCanonizer(
    mul, true_div, reciprocal, mul_calculate, False
)
register_canonicalize(local_mul_canonizer, "shape_unsafe", name="local_mul_canonizer")


@register_canonicalize
@node_rewriter([neg])
def local_neg_to_mul(fgraph, node):
    return [mul(np.array(-1, dtype=node.inputs[0].dtype), node.inputs[0])]


@register_specialize
@node_rewriter([Sum, Prod])
def local_sum_prod_of_mul_or_div(fgraph, node):
    """
    sum(a * X) -> a * sum(X), when a is broadcasted along the sum dimensions

    or

    prod(a * X) -> (a ** size(X)) * prod(X)

    It also applies to reduction of X / a,
    but not a / X, as that would still require inverting every value in X before the reduction

    TODO: In the case where not all axis overlap with broadcast dimensions,
     consider introducing an outer reduction after factoring out the compatible reduced dimensions
     E.g. sum(arange(5) * X, axis=(0, 2)) -> sum(sum(X, axis=0) * arange(5), axis=1)
    """

    [node_inps] = node.inputs
    if node_inps.owner is None:
        return None

    inner_op = node_inps.owner.op
    if not (inner_op == mul or inner_op == true_div):
        return None

    reduced_axes = node.op.axis
    if reduced_axes is None:
        reduced_axes = tuple(range(node_inps.type.ndim))

    # Separate terms that can be moved out of the Sum/Prod and those that cannot
    if inner_op == mul:
        # Mul accepts arbitrary inputs, so we need to separate into two groups
        outer_terms = []
        inner_terms = []
        for term in node_inps.owner.inputs:
            term_bcast = term.type.broadcastable
            if all(term_bcast[i] for i in reduced_axes):
                outer_terms.append(term.squeeze(reduced_axes))
            else:
                inner_terms.append(term)

        if not outer_terms:
            return None
        else:
            outer_term = variadic_mul(*outer_terms)

        if not inner_terms:
            inner_term = None
        else:
            inner_term = variadic_mul(*inner_terms)

    else:  # true_div
        # We only care about removing the denominator out of the reduction
        numerator, denominator = node_inps.owner.inputs
        denominator_bcast = denominator.type.broadcastable
        if all(denominator_bcast[i] for i in reduced_axes):
            outer_term = denominator.squeeze(reduced_axes)
            inner_term = numerator
        else:
            return None

    # If we have a `Prod`, then the outside terms need to be raised to the power of the number of elements
    # that were contracted in the input
    if isinstance(node.op, Prod) and inner_term is not None:
        dtype = inner_term.dtype
        n_reduced_elements = prod(
            [inner_term.shape[i].astype(dtype) for i in reduced_axes]
        )
        outer_term = outer_term**n_reduced_elements

    if inner_term is None:
        # Sum/Prod is useless, just return the outer_term
        # (This can only happen for mul, not division)
        new_out = outer_term
    else:
        reduced_inner_term = node.op(inner_term)
        if inner_op == mul:
            new_out = outer_term * reduced_inner_term
        else:
            new_out = reduced_inner_term / outer_term
        copy_stack_trace(node.outputs, [inner_term, reduced_inner_term, outer_term])

    copy_stack_trace(node.outputs, new_out)
    return [new_out]


@register_specialize
@node_rewriter([Sum])
def local_sum_of_neg_to_neg_of_sum(fgraph, node):
    """Rewrite sum(-X) -> -sum(X)."""
    [node_inps] = node.inputs
    if node_inps.owner and node_inps.owner.op == neg:
        s = node.op(node_inps.owner.inputs[0])
        ret = neg(s)
        # There are never errors in the negative op, thus
        # we need only to copy over stacktrace from previous output node to
        # the two new ops.
        copy_stack_trace(node.outputs, [s, ret])

        return [ret]


@register_specialize
@node_rewriter([sub])
def local_elemwise_sub_zeros(fgraph, node):
    """
    Elemwise{sub}(X,X) -> zeros_like(X)
    """
    if node.inputs[0] == node.inputs[1]:
        res = zeros_like(node.inputs[0])
        # Copy over stacktrace from previous output.
        # This could help for failures due to out-of-memory.
        copy_stack_trace(node.outputs, res)
        return [res]


@register_useless
@register_specialize
@register_stabilize
@register_canonicalize
@node_rewriter([Elemwise])
def local_useless_elemwise_comparison(fgraph, node):
    """...

    # Comparing to itself is constant
    Elemwise[{LT,GT}](X, X) -> Elemwise[zeros](X)
    Elemwise[{LE,GE}](X, X) -> Elemwise[ones](X)
    Elemwise[{minimum,maximum}](X, X) -> X

    # Comparing shape to 0 can be constant
    Elemwise[LT](X.shape[i], 0) -> Elemwise[zeros](X)
    Elemwise[GE](X.shape[i], 0) -> Elemwise[ones](X)
    Elemwise[maximum](X.shape[i], 0) -> X.shape[i]
    Elemwise[maximum](0, X.shape[i]) -> X.shape[i]
    Elemwise[minimum](X.shape[i], 0) -> 0
    Elemwise[minimum](0, X.shape[i]) -> 0

    # The shape can be replaced with sum of shapes
    Elemwise[LT](add([anything that is shapes]), 0) -> Elemwise[zeros](X)
    Elemwise[GE](add([anything that is shapes]), 0) -> Elemwise[ones](X)

    # Shapes are never negative
    # Needed by Reshape.infer_shape
    Elemwise[EQ](Subtensor(Shape(x)), -N) -> Elemwise[zeros](X)

    Notes
    -----
    These cases appear in the graph generated by scan. These rewrites will make
    the graph easier to read.

    """
    # TODO: Refactor this function. So much repeated code!

    if node.op.scalar_op.nin != 2:
        return

    dtype = node.outputs[0].type.dtype
    out_bcast = node.outputs[0].type.broadcastable

    # Elemwise[{LT,GT}](X, X) -> Elemwise[zeros](X)
    if (
        isinstance(node.op.scalar_op, ps.LT | ps.GT)
        and node.inputs[0] is node.inputs[1]
    ):
        res = zeros_like(node.inputs[0], dtype=dtype, opt=True)
        # Copy over stacktrace from previous output.
        copy_stack_trace(node.outputs, res)
        return [res]

    # Elemwise[{LE,GE}](X, X) -> Elemwise[ones](X)
    if (
        isinstance(node.op.scalar_op, ps.LE | ps.GE)
        and node.inputs[0] is node.inputs[1]
    ):
        res = ones_like(node.inputs[0], dtype=dtype, opt=True)

        # Copy over stacktrace from previous output.
        copy_stack_trace(node.outputs, res)
        return [res]

    # Elemwise[{minimum,maximum}](X, X) -> X
    if (
        isinstance(node.op.scalar_op, ps.ScalarMinimum | ps.ScalarMaximum)
        and node.inputs[0] is node.inputs[1]
    ):
        res = node.inputs[0]
        # Copy over stacktrace from previous output.
        copy_stack_trace(node.outputs, res)
        return [res]

    # Elemwise[LT](X.shape[i], 0) -> Elemwise[zeros](X)
    if (
        isinstance(node.op.scalar_op, ps.LT)
        and node.inputs[0].owner
        and isinstance(node.inputs[0].owner.op, Shape_i)
        and get_underlying_scalar_constant_value(
            node.inputs[1], only_process_constants=True, raise_not_constant=False
        )
        == 0
    ):
        res = zeros_like(node.inputs[0], dtype=dtype, opt=True)
        if res.type.broadcastable != out_bcast:
            res = broadcast_arrays(res, node.inputs[1])[0]
        # Copy over stacktrace from previous output.
        copy_stack_trace(node.outputs, res)
        return [res]

    # Elemwise[GE](X.shape[i], 0) -> Elemwise[ones](X)
    if (
        isinstance(node.op.scalar_op, ps.GE)
        and node.inputs[0].owner
        and isinstance(node.inputs[0].owner.op, Shape_i)
        and get_underlying_scalar_constant_value(
            node.inputs[1], only_process_constants=True, raise_not_constant=False
        )
        == 0
    ):
        res = ones_like(node.inputs[0], dtype=dtype, opt=True)
        if res.type.broadcastable != out_bcast:
            res = broadcast_arrays(res, node.inputs[1])[0]
        # Copy over stacktrace from previous output.
        copy_stack_trace(node.outputs, res)
        return [res]

    # Elemwise[maximum](X.shape[i], 0) -> X.shape[i]
    if isinstance(node.op.scalar_op, ps.ScalarMaximum):
        for idx in range(2):
            if (
                node.inputs[idx].owner
                and isinstance(node.inputs[idx].owner.op, Shape_i)
                and get_underlying_scalar_constant_value(
                    node.inputs[1 - idx],
                    only_process_constants=True,
                    raise_not_constant=False,
                )
                == 0
            ):
                res = node.inputs[idx]
                if res.type.broadcastable != out_bcast:
                    res = broadcast_arrays(res, node.inputs[1 - idx])[0]
                # No need to copy over stacktrace.
                return [res]

    # Elemwise[minimum](X.shape[i], 0) -> 0
    if isinstance(node.op.scalar_op, ps.ScalarMinimum):
        for idx in range(2):
            if (
                node.inputs[idx].owner
                and isinstance(node.inputs[idx].owner.op, Shape_i)
                and get_underlying_scalar_constant_value(
                    node.inputs[1 - idx],
                    only_process_constants=True,
                    raise_not_constant=False,
                )
                == 0
            ):
                res = zeros_like(node.inputs[idx], dtype=dtype, opt=True)
                if res.type.broadcastable != out_bcast:
                    res = broadcast_arrays(res, node.inputs[1 - idx])[0]
                # No need to copy over stacktrace.
                return [res]

    # Elemwise[LT](add([anything that is shapes]), 0) -> Elemwise[zeros](X)
    if (
        isinstance(node.op.scalar_op, ps.LT)
        and node.inputs[0].owner
        and isinstance(node.inputs[0].owner.op, Elemwise)
        and isinstance(node.inputs[0].owner.op.scalar_op, ps.Add)
        and all(
            isinstance(var.owner and var.owner.op, Shape_i)
            for var in node.inputs[0].owner.inputs
        )
        and get_underlying_scalar_constant_value(
            node.inputs[1], only_process_constants=True, raise_not_constant=False
        )
        == 0
    ):
        res = zeros_like(node.inputs[0], dtype=dtype, opt=True)
        if res.type.broadcastable != out_bcast:
            res = broadcast_arrays(res, node.inputs[1])[0]
        # Copy over stacktrace from previous output.
        copy_stack_trace(node.outputs, res)
        return [res]

    # Elemwise[GE](add([anything that is shapes]), 0) -> Elemwise[ones](X)
    if (
        isinstance(node.op.scalar_op, ps.GE)
        and node.inputs[0].owner
        and isinstance(node.inputs[0].owner.op, Elemwise)
        and isinstance(node.inputs[0].owner.op.scalar_op, ps.Add)
        and all(
            isinstance(var.owner and var.owner.op, Shape_i)
            for var in node.inputs[0].owner.inputs
        )
        and get_underlying_scalar_constant_value(
            node.inputs[1], only_process_constants=True, raise_not_constant=False
        )
        == 0
    ):
        res = ones_like(node.inputs[0], dtype=dtype, opt=True)
        if res.type.broadcastable != out_bcast:
            res = broadcast_arrays(res, node.inputs[1])[0]
        # Copy over stacktrace from previous output.
        copy_stack_trace(node.outputs, res)
        return [res]

    # Elemwise[EQ](Subtensor(Shape(x)), -N)
    # Elemwise[EQ](somegraph that only depend of shape, -N)
    # TODO: handle the case where the -N is on either side
    """
|Elemwise{eq,no_inplace} [id B] ''
| |Subtensor{int64} [id C] ''
| | |Join [id D] ''
| | | |TensorConstant{0} [id E]
| | | |Subtensor{int64:int64:} [id F] ''
| | | | |Shape [id G] ''
    """

    def investigate_if_shape(node) -> bool:
        "Return True if values will be shapes, so >= 0"
        if isinstance(node.op, Shape | Shape_i):
            return True
        elif isinstance(node.op, Subtensor) and node.inputs[0].owner:
            return investigate_if_shape(node.inputs[0].owner)
        elif isinstance(node.op, Join):
            return all(
                v.owner and investigate_if_shape(v.owner) for v in node.inputs[1:]
            )
        elif isinstance(node.op, MakeVector):
            return all(v.owner and investigate_if_shape(v.owner) for v in node.inputs)
        return False

    if (
        isinstance(node.op.scalar_op, ps.EQ)
        and node.inputs[0].owner
        and investigate_if_shape(node.inputs[0].owner)
        and (
            isinstance(node.inputs[1], TensorConstant)
            and node.inputs[1].unique_value is not None
            and node.inputs[1].unique_value < 0
        )
    ):
        res = zeros_like(node.inputs[0], dtype=dtype, opt=True)
        if res.type.broadcastable != out_bcast:
            res = broadcast_arrays(res, node.inputs[1])[0]
        # Copy over stacktrace from previous output.
        copy_stack_trace(node.outputs, res)
        return [res]

    return


@register_canonicalize
@node_rewriter([Sum, Prod])
def local_sum_prod_all_to_none(fgraph, node):
    """
    Sum{0,1,...N} -> Sum{} or
    Prod{0,1,...N} -> Prod{}

    """
    op_type = Sum if isinstance(node.op, Sum) else Prod
    # if all the axes are named, then use None as a shorthand
    # this permits more merging
    if node.op.axis is None:
        return
    if set(node.op.axis) == set(range(node.inputs[0].type.ndim)):
        return [op_type(axis=None, dtype=node.op.dtype)(node.inputs[0])]


@register_canonicalize
@node_rewriter([CAReduce])
def local_reduce_chain(fgraph, node) -> list[TensorVariable] | None:
    """
    Sum(Sum()) -> single Sum()
    or any CAReduce(Careduce(x)) of the same type

    """
    [inner_reduce] = node.inputs
    if not (inner_reduce.owner and isinstance(inner_reduce.owner.op, CAReduce)):
        return None

    # Don't apply rewrite if inner_reduce is used elsewhere
    if len(fgraph.clients[inner_reduce]) > 1:
        return None

    # Check if CAReduces have the same scalar op
    outer_op: CAReduce = node.op
    inner_op = inner_reduce.owner.op

    if outer_op.scalar_op != inner_op.scalar_op:
        return None

    outer_axis = outer_op.axis
    inner_axis = inner_op.axis
    [x] = inner_reduce.owner.inputs
    # check to see either the inner or outer prod is doing a
    # product over all axis, in which case we can remove it
    if outer_axis is None or inner_axis is None:
        return [outer_op.clone(axis=None)(x)]

    # Merge axis
    newaxis = list(inner_axis)
    for i in outer_axis:
        new_i = i
        for ii in inner_axis:
            if new_i >= ii:
                new_i += 1
        assert new_i not in newaxis
        newaxis.append(new_i)

    assert len(newaxis) == len(inner_axis) + len(outer_axis)
    return [outer_op.clone(axis=sorted(newaxis))(x)]


@register_canonicalize
@register_uncanonicalize  # Needed for MaxAndArgmax -> CAReduce
@node_rewriter([CAReduce])
def local_reduce_join(fgraph, node):
    """
    CAReduce{scalar.op}(Join(axis=x, a, b), axis=x) -> Elemwise{scalar.op}(a, b)

    When a, b have a dim length of 1 along the join axis

    """
    if not (node.inputs[0].owner and isinstance(node.inputs[0].owner.op, Join)):
        return None

    [joined_out] = node.inputs
    joined_node = joined_out.owner
    join_axis_tensor, *joined_inputs = joined_node.inputs

    n_joined_inputs = len(joined_inputs)
    if n_joined_inputs < 2:
        # Let some other rewrite get rid of this useless Join
        return None
    if n_joined_inputs > 2 and not isinstance(node.op.scalar_op, ps.Add | ps.Mul):
        # We don't rewrite if a single Elemwise cannot take all inputs at once
        return None

    if not isinstance(join_axis_tensor, Constant):
        return None
    join_axis = join_axis_tensor.data

    # Check whether reduction happens on joined axis
    reduce_op = node.op
    reduce_axis = reduce_op.axis
    if reduce_axis is None:
        if joined_out.type.ndim > 1:
            return None
    elif reduce_axis != (join_axis,):
        return None

    # Check all inputs are broadcastable along the join axis and squeeze those dims away
    new_inputs = []
    for inp in joined_inputs:
        if not inp.type.broadcastable[join_axis]:
            return None
        # Most times inputs to join have an expand_dims, we eagerly clean up those here
        new_input = apply_local_dimshuffle_lift(fgraph, inp.squeeze(join_axis))
        new_inputs.append(new_input)

    ret = Elemwise(node.op.scalar_op)(*new_inputs)

    if ret.dtype != node.outputs[0].dtype:
        # The reduction do something about the dtype.
        return None

    return [ret]


@register_infer_shape
@register_canonicalize("fast_compile", "local_cut_useless_reduce")
@register_useless("local_cut_useless_reduce")
@node_rewriter([CAReduce])
def local_useless_reduce(fgraph, node):
    """Sum(a, axis=[]) -> a"""
    (summed,) = node.inputs
    # if reduce were doing anything, the output ndim would be reduced
    if summed.type == node.outputs[0].type:
        return [summed]


@register_canonicalize
@register_uncanonicalize
@register_specialize
@node_rewriter([CAReduce])
def local_reduce_broadcastable(fgraph, node):
    """Remove reduction over broadcastable dimensions."""
    (reduced,) = node.inputs
    odtype = node.outputs[0].dtype
    if node.op.axis is None:
        if all(reduced.broadcastable):
            return [reduced.dimshuffle().astype(odtype)]
    else:
        axis = list(node.op.axis)
        cuttable = [a for a in axis if reduced.broadcastable[a]]
        if cuttable:
            # -- we can remove some axes of summation.
            new_axis = []
            pattern = []
            ii = 0
            for p in range(reduced.ndim):
                if p not in cuttable:
                    if p in axis:
                        new_axis.append(ii)
                    pattern.append(p)
                    ii += 1
            new_reduced = reduced.dimshuffle(*pattern)
            if new_axis:
                if type(node.op) is CAReduce:
                    # This case handles `CAReduce` instances
                    # (e.g. generated by `scalar_elemwise`), and not the
                    # scalar `Op`-specific subclasses
                    # TODO FIXME: This highlights a major design flaw in
                    # `CAReduce` (or at least our use of it), and it needs
                    # to be fixed
                    new_op = node.op.__class__(node.op.scalar_op, axis=new_axis)
                else:
                    new_op = node.op.__class__(axis=new_axis)
                return [new_op(new_reduced)]
            else:
                # -- in this case we can remove the reduction completely
                return [new_reduced.astype(odtype)]


@register_specialize
@node_rewriter([Sum, Prod])
def local_opt_alloc(fgraph, node):
    """
    sum(alloc(constant,shapes...)) => constant*prod(shapes)
    or
    prod(alloc(constant,shapes...)) => constant**prod(shapes)

    """
    (node_inps,) = node.inputs
    if node_inps.owner and isinstance(node_inps.owner.op, Alloc):
        inp = node_inps.owner.inputs[0]
        shapes = node_inps.owner.inputs[1:]
        try:
            val = get_underlying_scalar_constant_value(inp, only_process_constants=True)
            assert val.size == 1
            val = val.reshape(1)[0]
            # check which type of op
            size = mul(*shapes)
            if inp.dtype in ("float16", "float32"):
                # shapes are ints and normally int64.
                # We don't want to have a float64 upcast
                # We don't want to downcast to float16
                # as we fear it could loose too much precision
                # that will be amplified by the mul/pow below.
                size = size.astype("float32")
            if node.op.axis is None or node.op.axis == tuple(range(inp.ndim)):
                if isinstance(node.op, Sum):
                    val = val * size
                else:
                    val = val**size
                # Sum can change the input dtype (upcast or bool
                # -> float32) by default or by user request.
                # We can ignore the acc_dtype, as there is only 1
                # elemwise we will do and not a sequence, so there is no
                # accumulation of errors.
                # So mostly, we just need to cast the output to the old
                # dtype.
                val = val.astype(node.outputs[0].dtype)
                return [val]
            to_prod = [shapes[i] for i in range(len(shapes)) if i in node.op.axis]
            if to_prod:
                size = mul(*to_prod)
                if isinstance(node.op, Sum):
                    val *= size
                else:
                    val = val**size
            # See comments above.
            val = val.astype(node.outputs[0].dtype)
            return [
                alloc(
                    val,
                    *[shapes[i] for i in range(len(shapes)) if i not in node.op.axis],
                )
            ]
        except NotScalarConstantError:
            pass


@register_specialize
@node_rewriter([neg])
def local_neg_div_neg(fgraph, node):
    """
    - (-a / b) -> a / b

    Also performs - (c / b) -> ((-c) / b) when c is a scalar constant.

    """
    if node.inputs[0].owner and node.inputs[0].owner.op == true_div:
        frac = node.inputs[0]
        num, denom = frac.owner.inputs
        if num.owner and num.owner.op == neg:
            if len(fgraph.clients[frac]) == 1:
                # No other clients of the original division
                new_num = num.owner.inputs[0]
                return [true_div(new_num, denom)]
        elif all(num.broadcastable) and isinstance(num, Constant):
            if len(fgraph.clients[frac]) == 1:
                new_num = -num.data
                return [true_div(new_num, denom)]


@register_canonicalize
@register_specialize
@node_rewriter([sub])
def local_sub_neg_to_add(fgraph, node):
    """
    x - (-y) -> x + y

    """
    minuend, subtrahend = node.inputs

    if subtrahend.owner:
        if subtrahend.owner.op == neg:
            pre_neg = subtrahend.owner.inputs[0]
            new_out = add(minuend, pre_neg)
            return [new_out]


@register_specialize
@node_rewriter([add])
def local_add_neg_to_sub(fgraph, node):
    """
    -x + y -> y - x
    x + (-y) -> x - y

    """
    # This rewrite is only registered during specialization, because the
    # `local_neg_to_mul` rewrite modifies the relevant pattern during canonicalization

    # Rewrite is only applicable when there are two inputs to add
    if len(node.inputs) == 2:
        # Look for pattern with either input order
        for first, second in (node.inputs, reversed(node.inputs)):
            if second.owner:
                if second.owner.op == neg:
                    pre_neg = second.owner.inputs[0]
                    new_out = sub(first, pre_neg)
                    return [new_out]


@register_canonicalize
@node_rewriter([mul])
def local_mul_zero(fgraph, node):
    """
    As part of canonicalization, we replace multiplication by zero
    with zero.

    """
    otype = node.outputs[0].type

    for i in node.inputs:
        try:
            value = get_underlying_scalar_constant_value(i)
        except NotScalarConstantError:
            continue
        # print 'MUL by value', value, node.inputs
        if value == 0:
            # print '... returning zeros'
            return [broadcast_arrays(np.asarray(0, dtype=otype.dtype), *node.inputs)[0]]


# TODO: Add this to the canonicalization to reduce redundancy.
@register_specialize
@node_rewriter([true_div])
def local_div_to_reciprocal(fgraph, node):
    if (
        get_underlying_scalar_constant_value(
            node.inputs[0], only_process_constants=True, raise_not_constant=False
        )
        == 1.0
    ):
        out = node.outputs[0]
        new_out = reciprocal(local_mul_canonizer.merge_num_denum(node.inputs[1:], []))
        # The ones could have forced upcasting
        if new_out.dtype != out.dtype:
            new_out = cast(new_out, dtype=out.dtype)
        # The ones could have forced a specific length
        if not out.type.is_super(new_out.type):
            new_out = alloc_like(new_out, out, fgraph)
        return [new_out]


@register_canonicalize
@node_rewriter([reciprocal])
def local_reciprocal_canon(fgraph, node):
    return [pt_pow(node.inputs[0], -1.0)]


@register_canonicalize
@node_rewriter([pt_pow])
def local_pow_canonicalize(fgraph, node):
    """
    Rewrites for exponential functions with straight-forward simplifications:
    1. x ** 0 -> 1
    2. x ** 1 -> x
    3. 1 ** x -> 1

    In all cases, the shape of the output is the result of broadcasting the shapes of the inputs.
    """
    cst_base = get_underlying_scalar_constant_value(
        node.inputs[0], only_process_constants=True, raise_not_constant=False
    )
    cst_exponent = get_underlying_scalar_constant_value(
        node.inputs[1], only_process_constants=True, raise_not_constant=False
    )

    new_out = None

    if cst_base == 1:
        # 1 ** x = 1
        new_out = broadcast_arrays(*node.inputs)[0]
    elif cst_exponent == 0:
        # x ** 0 = 1
        new_out = broadcast_arrays(ones_like(node.inputs[0]), node.inputs[1])[0]
    elif cst_exponent == 1:
        # x ** 1 = x
        new_out = broadcast_arrays(*node.inputs)[0]

    if new_out is None:
        return

    if new_out.dtype != node.out.dtype:
        new_out = cast(new_out, dtype=node.out.dtype)

    return [new_out]


@register_specialize
@node_rewriter([mul])
def local_mul_to_sqr(fgraph, node):
    """
    x*x -> sqr(x)
    """
    if len(node.inputs) == 2:
        if node.inputs[0] is node.inputs[1]:
            return [sqr(node.inputs[0])]


@register_canonicalize
@node_rewriter([int_div])
def local_intdiv_by_one(fgraph, node):
    """x // 1 -> x"""
    if isinstance(node.inputs[1], TensorConstant) and np.all(node.inputs[1].value == 1):
        return [node.inputs[0].astype(node.outputs[0].dtype)]


@register_canonicalize
@register_specialize
@node_rewriter([int_div, true_div])
def local_zero_div(fgraph, node):
    """0 / x -> 0"""
    if (
        get_underlying_scalar_constant_value(
            node.inputs[0], only_process_constants=True, raise_not_constant=False
        )
        == 0
    ):
        ret = alloc_like(0, node.outputs[0], fgraph)
        ret.tag.values_eq_approx = values_eq_approx_remove_nan
        return [ret]


@register_specialize
@node_rewriter([pt_pow])
def local_pow_specialize(fgraph, node):
    # the idea here is that we have pow(x, y)
    odtype = node.outputs[0].dtype
    xsym = node.inputs[0]
    ysym = node.inputs[1]
    try:
        y = get_underlying_scalar_constant_value(ysym, only_process_constants=True)
    except NotScalarConstantError:
        return

    if not broadcasted_by(xsym, ysym):
        rval = None

        if np.all(y == 2):
            rval = [sqr(xsym)]
        if np.all(y == 1):
            rval = [xsym]
        if np.all(y == 0):
            rval = [alloc_like(1, xsym, fgraph)]
        if np.all(y == 0.5):
            rval = [sqrt(xsym)]
        if np.all(y == -0.5):
            rval = [reciprocal(sqrt(xsym))]
        if np.all(y == -1):
            rval = [reciprocal(xsym)]
        if np.all(y == -2):
            rval = [reciprocal(sqr(xsym))]
        if rval:
            if not rval[0].type.broadcastable == node.outputs[0].type.broadcastable:
                return None
            rval[0] = cast(rval[0], odtype)
            assert rval[0].type.dtype == node.outputs[0].type.dtype
            return rval


@register_specialize
@node_rewriter([pt_pow])
def local_pow_to_nested_squaring(fgraph, node):
    """Convert a large power exponent to multiple squaring operations.

    Note: This sounds like the kind of thing any half-decent compiler can do by itself?
    """

    # the idea here is that we have pow(x, y)
    xsym, ysym = node.inputs

    try:
        y = get_underlying_scalar_constant_value(ysym, only_process_constants=True)
    except NotScalarConstantError:
        return

    odtype = node.outputs[0].dtype

    # the next line is needed to fix a strange case that I don't
    # know how to make a separate test.
    # That happen in the `test_log_erfc` test.
    # y is a ndarray with dtype int8 and value 2,4 or 6. This make
    # the abs(y) <= 512 fail!
    # taking the value outside ndarray solve the problem.
    # it could be that in that case, numpy make the comparison
    # into the wrong type(do in int8 that overflow.)
    if isinstance(y, np.ndarray):
        assert y.size == 1
        try:
            y = y[0]
        except IndexError:
            pass
    if not broadcasted_by(xsym, ysym):
        rval = None
        # 512 is too small for the cpu and too big for some gpu!
        if abs(y) == int(abs(y)) and abs(y) <= 512:
            pow2 = [xsym]
            pow2_scal = [ps.get_scalar_type(xsym.dtype)()]
            y_to_do = abs(y)
            for i in range(int(np.log2(y_to_do))):
                pow2.append(sqr(pow2[i]))
                pow2_scal.append(ps.sqr(pow2_scal[i]))
            rval1 = None
            rval1_scal = None
            while y_to_do > 0:
                log_to_do = int(np.log2(y_to_do))
                if rval1 is not None:
                    rval1 *= pow2[log_to_do]
                    rval1_scal *= pow2_scal[log_to_do]
                else:
                    rval1 = pow2[log_to_do]
                    rval1_scal = pow2_scal[log_to_do]
                y_to_do -= 2**log_to_do

            if abs(y) > 2:
                # We fuse all the pow together here to make
                # compilation faster
                rval1 = Elemwise(ps.Composite([pow2_scal[0]], [rval1_scal])).make_node(
                    xsym
                )
            if y < 0:
                rval = [reciprocal(rval1)]
            else:
                rval = [rval1]
        if rval is not None:
            rval[0] = cast(rval[0], odtype)
            return rval


@register_specialize
@node_rewriter([mul])
def local_mul_specialize(fgraph, node):
    """
    Remove special-case constants from mul arguments and useless neg in inputs.

    mul(-1, x) -> neg(x)
    mul(1, x, y) -> mul(x, y)
    mul(0, ...) -> alloc(0, shapes...)

    This is not done if we would add more nodes in the graph, like with:

    mul(-1, x, y) -/-> neg(mul(x, y))

    """

    # at this point [post canonicalize], mul() may have many inputs.
    # the idea here is that we have pow(x, y)
    has_neg = False
    new_inputs = []
    nb_neg_node = 0
    nb_cst = 0
    for inp in node.inputs:
        # remove any neg arguments
        while inp.owner and inp.owner.op == neg:
            has_neg ^= True
            inp = inp.owner.inputs[0]
            nb_neg_node += 1

        # remove special case arguments of 1, -1 or 0
        y = get_underlying_scalar_constant_value(
            inp, only_process_constants=True, raise_not_constant=False
        )
        if y == 1.0:
            nb_cst += 1
        elif y == -1.0:
            nb_cst += 1
            has_neg ^= True  # toggles
        elif y == 0.0:
            # if we find any zero, we just return right away
            return [alloc_like(0, node.outputs[0], fgraph)]
        else:
            new_inputs.append(inp)

    if new_inputs != node.inputs:
        if new_inputs:
            if len(new_inputs) == 1:
                if has_neg:
                    if new_inputs[0].dtype in ([*uint_dtypes, "bool"]):
                        return
                    else:
                        rval = -new_inputs[0]
                else:
                    rval = new_inputs[0]
            else:
                # The next case would cause a replace by an equivalent case.
                if has_neg and nb_neg_node == 0 and nb_cst == 1:
                    return
                elif has_neg:
                    # Don't add an extra neg node as we can't
                    # fully replace this mul by a neg.
                    m1 = np.asarray(-1, dtype=node.outputs[0].dtype)
                    new_inputs = [m1, *new_inputs]
                rval = mul(*new_inputs)

            return [alloc_like(rval, node.outputs[0], fgraph)]
        else:
            # there are no variable inputs to mul
            # N.B. this could have been constant-folded...
            if has_neg:
                return [alloc_like(-1, node.outputs[0], fgraph)]
            else:
                return [alloc_like(1, node.outputs[0], fgraph)]


@register_specialize
@node_rewriter([add])
def local_add_remove_zeros(fgraph, node):
    new_inputs = []
    for inp in node.inputs:
        try:
            y = get_underlying_scalar_constant_value(inp)
        except NotScalarConstantError:
            y = inp
        if y == 0.0:
            continue
        new_inputs.append(inp)

    if len(new_inputs) == len(node.inputs):
        return False

    node_output = node.outputs[0]
    dtype = node_output.type.dtype

    if len(new_inputs) == 0:
        # we got rid of the entire expression!
        ndim = node_output.type.ndim
        # Reuse call to constant for cache()
        cst = constant(np.zeros((1,) * ndim, dtype=dtype))
        assert cst.type.broadcastable == (True,) * ndim
        return [alloc_like(cst, node_output, fgraph)]

    ret = [alloc_like(variadic_add(*new_inputs), node_output, fgraph)]

    # The dtype should not be changed. It can happen if the input
    # that was forcing upcasting was equal to 0.
    if ret[0].dtype != dtype:
        ret = [cast(ret[0], dtype)]

    return ret


mul_canonizer = in2out(
    SequentialNodeRewriter(
        local_mul_canonizer, local_fill_sink, apply_all_rewrites=True
    ),
    name="mul_canonizer_groups",
)


def check_for_x_over_absX(numerators, denominators):
    """Convert x/abs(x) into sign(x)."""
    # TODO: this function should dig/search through dimshuffles
    # This won't catch a dimshuffled absolute value
    for den in list(denominators):
        if den.owner and den.owner.op == pt_abs and den.owner.inputs[0] in numerators:
            if den.owner.inputs[0].type.dtype.startswith("complex"):
                # TODO: Make an Op that projects a complex number to
                #      have unit length but projects 0 to 0.  That
                #      would be a weird Op, but consistent with the
                #      special case below.  I heard there's some
                #      convention in Matlab that is similar to
                #      this... but not sure.
                pass
            else:
                denominators.remove(den)
                numerators.remove(den.owner.inputs[0])
                numerators.append(sign(den.owner.inputs[0]))
    return numerators, denominators


local_mul_canonizer.add_simplifier(check_for_x_over_absX, "X_over_absX")


@register_canonicalize
@node_rewriter([pt_abs])
def local_abs_lift(fgraph, node):
    """
    Move the abs toward the input.

    This is needed for check_for_x_over_absX to apply in more case.

    """
    if node.inputs[0].owner:
        assert node.nin == 1
        if node.inputs[0].owner.op == mul:
            return [mul(*[pt_abs(i) for i in node.inputs[0].owner.inputs])]
        if node.inputs[0].owner.op == true_div:
            i = node.inputs[0].owner.inputs
            return [true_div(pt_abs(i[0]), pt_abs(i[1]))]


@register_specialize
@node_rewriter([mul, true_div])
def local_abs_merge(fgraph, node):
    """
    Merge abs generated by local_abs_lift when the canonizer don't
    need it anymore

    """
    if node.op == mul and sum(i.owner.op == pt_abs for i in node.inputs if i.owner) > 1:
        inputs = []
        for i in node.inputs:
            if i.owner and i.owner.op == pt_abs:
                inputs.append(i.owner.inputs[0])
            elif isinstance(i, Constant):
                try:
                    const = get_underlying_scalar_constant_value(
                        i, only_process_constants=True
                    )
                except NotScalarConstantError:
                    return False
                if not const >= 0:
                    return False
                inputs.append(i)
            else:
                return False
        return [pt_abs(mul(*inputs))]
    if (
        node.op == true_div
        and sum(i.owner.op == pt_abs for i in node.inputs if i.owner) == 2
    ):
        return [
            pt_abs(
                true_div(node.inputs[0].owner.inputs[0], node.inputs[1].owner.inputs[0])
            )
        ]


@register_stabilize
@register_specialize
@node_rewriter([log])
def local_log1p(fgraph, node):
    # log(1+x) -> log1p(x)
    # log(1-x) -> log1p(-x)
    (log_arg,) = node.inputs
    if log_arg.owner and log_arg.owner.op == add:
        scalars, _scalar_inputs, nonconsts = scalarconsts_rest(
            log_arg.owner.inputs, only_process_constants=True
        )
        # scalar_inputs are potentially dimshuffled and fill'd scalars
        if scalars and isclose(np.sum(scalars), 1):
            if nonconsts:
                ninp = variadic_add(*nonconsts)
                if ninp.dtype != log_arg.type.dtype:
                    ninp = ninp.astype(node.outputs[0].dtype)
                return [alloc_like(log1p(ninp), node.outputs[0], fgraph)]

    elif log_arg.owner and log_arg.owner.op == sub:
        one, other = log_arg.owner.inputs
        try:
            one = get_underlying_scalar_constant_value(one, only_process_constants=True)
        except NotScalarConstantError:
            return

        if one != 1:
            return

        if other.type.broadcastable != log_arg.type.broadcastable:
            other = broadcast_arrays(other, one)[0]

        if other.type.dtype != log_arg.type.dtype:
            other = other.astype(log_arg.dtype)

        return [log1p(neg(other))]


@register_stabilize
@register_specialize
@node_rewriter([log])
def local_log_add_exp(fgraph, node):
    """
    ``log(exp(x)+exp(y)+exp(z)) = max + log(x-max, y-max, z-max)``

    TODO: in canonicalize, change log10 and log2 -> log
    """

    z = node.inputs[0]
    if z.owner and z.owner.op == add:
        zi = z.owner.inputs
        pre_exp = [x.owner.inputs[0] for x in zi if x.owner and x.owner.op == exp]
        # all arguments to add are exp(<something>)
        if len(pre_exp) == len(zi):
            # Do not offset when max_pre = -np.inf, to avoid nan in the output
            # Switch statement is placed directly inside add to break the self-symmetry
            # of the returned output (otherwise the rewrite would not stabilize)
            max_pre = reduce(maximum, pre_exp)
            ret = max_pre + log(
                add(
                    *[
                        switch(isinf(max_pre), exp(max_pre), exp(p - max_pre))
                        for p in pre_exp
                    ]
                )
            )
            return [ret]


@register_stabilize
@register_specialize
@node_rewriter([log])
def local_log_sum_exp(fgraph, node):
    # log(sum_i(exp(x_i))) = x_max + log(sum_i(exp(x_i - x_max)))

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
        and isinstance(exp_node.op.scalar_op, ps.Exp)
    ):
        return

    pre_exp = exp_node.inputs[0]
    max_pre_exp = pt_max(pre_exp, axis=axis)
    max_pre_exp_keepdims = makeKeepDims(pre_exp, max_pre_exp, axis)

    # Do not offset when max_pre = -np.inf, to avoid nan in the output
    # Switch statement is placed directly inside sum to break the self-symmetry
    # of the returned output (otherwise the rewrite would not stabilize)
    ret = max_pre_exp + log(
        pt_sum(
            switch(
                isinf(max_pre_exp_keepdims),
                exp(max_pre_exp_keepdims),
                exp(pre_exp - max_pre_exp_keepdims),
            ),
            axis=axis,
        ),
    )

    # Restore the dimshuffle op, if any.
    if dimshuffle_op:
        ret = dimshuffle_op(ret)

    return [ret]


def add_calculate(num, denum, aslist=False, out_type=None):
    # TODO: make sure that this function and mul_calculate are similar
    if out_type is None:
        zero = 0.0
    else:
        zero = np.asarray(0, dtype=out_type.dtype)
    # zero = 0.0 if out_type is None else np.asarray(0,
    # dtype=out_type.dtype)
    if out_type and out_type.dtype == "bool":
        if len(denum) == 0:
            # NumPy 1.14 do not accept to do "bool - bool"
            v = reduce(np.add, num, zero)
        else:
            raise Exception(
                "bool subtraction not supported. This should not happen as"
                " an earlier error should have been raised"
            )
    else:
        v = reduce(np.add, num, zero) - reduce(np.add, denum, zero)
    if aslist:
        if np.all(v == 0):
            return []
        else:
            return [v]
    return v


local_add_canonizer = AlgebraicCanonizer(add, sub, neg, add_calculate)
add_canonizer = in2out(
    SequentialNodeRewriter(
        local_add_canonizer, local_fill_sink, apply_all_rewrites=True
    ),
    name="add_canonizer_group",
)

register_canonicalize(local_add_canonizer, "shape_unsafe", name="local_add_canonizer")


def distribute_greedy(pos_pairs, neg_pairs, num, denum, out_type, minscore=0):
    # each pair in pos_pairs and neg_pairs is a num/denum pair. this
    # function attempts to add num and denum to the corresponding parts
    # of each pair, and counts how many multiplications/divisions can
    # be saved in that way.

    # each division is counted like div_cost multiplications
    # (typically, division costs more so we are willing to multiply more
    # in order to divide less)
    # 1.5 was obtained through an informal test and may very well be
    # platform dependent
    div_cost = 1.5

    # score is number of operations saved, higher is better
    score = len(num) + div_cost * len(denum)
    new_pos_pairs = list(
        itertools.starmap(
            local_mul_canonizer.simplify,
            [(n + num, d + denum, out_type) for (n, d) in pos_pairs],
        )
    )
    new_neg_pairs = list(
        itertools.starmap(
            local_mul_canonizer.simplify,
            [(n + num, d + denum, out_type) for (n, d) in neg_pairs],
        )
    )
    for (n, d), (nn, dd) in zip(
        pos_pairs + neg_pairs, new_pos_pairs + new_neg_pairs, strict=True
    ):
        # We calculate how many operations we are saving with the new
        # num and denum
        score += len(n) + div_cost * len(d) - len(nn) - div_cost * len(dd)
    if score <= minscore:
        # the change is not applied because it adds too many operations
        return False, pos_pairs, neg_pairs
    return True, new_pos_pairs, new_neg_pairs


def attempt_distribution(factor, num, denum, out_type):
    """Try to insert each `num` and each `denum` in the factor?

    Returns
    -------
    changes?, new_factor, new_num, new_denum
        If there are changes, `new_num` and `new_denum` contain all the
        numerators and denominators that could not be distributed in the factor

    """
    pos_terms, neg_terms = local_add_canonizer.get_num_denum(factor)
    if len(pos_terms) == 1 and not neg_terms:
        return False, factor, num, denum
    pos_pairs = list(map(local_mul_canonizer.get_num_denum, pos_terms))
    neg_pairs = list(map(local_mul_canonizer.get_num_denum, neg_terms))
    change = False
    for n in list(num):
        success, pos_pairs, neg_pairs = distribute_greedy(
            pos_pairs, neg_pairs, [n], [], out_type
        )
        if success:
            change = True
            num.remove(n)
    for d in list(denum):
        success, pos_pairs, neg_pairs = distribute_greedy(
            pos_pairs, neg_pairs, [], [d], out_type
        )
        if success:
            change = True
            denum.remove(d)
    if not change:
        return change, factor, num, denum
    else:
        return (
            change,
            local_add_canonizer.merge_num_denum(
                list(itertools.starmap(local_mul_canonizer.merge_num_denum, pos_pairs)),
                list(itertools.starmap(local_mul_canonizer.merge_num_denum, neg_pairs)),
            ),
            num,
            denum,
        )


@register_canonicalize
@register_stabilize
@node_rewriter([mul, true_div, reciprocal])
def local_greedy_distributor(fgraph, node):
    """Reduce the number of multiplications and/or divisions.

    This rewrite tries to apply distributivity of multiplication
    to addition in order to reduce the number of multiplications
    and/or divisions that must be done. The algorithm weighs division
    more than multiplication to account for the former's slightly
    greater computational cost.

    The following expressions are simplified:
    1. ``((a/x + b/y) * x * y) -> a*y + b*x``
    2. ``((a/x + b) * x) -> a + b*x``
    3. There are other forms too where node is a true_div.

    The following expressions are not simplified:
    4. ``((a + b) * x) /> a*x + b*x``

    This rewrite aims to reduce computational cost. It may also
    increase numerical stability, e.g. when ``x`` and/or ``y`` tend to ``0`` in
    Example 1.

    """

    out = node.outputs[0]
    num, denum = local_mul_canonizer.get_num_denum(out)
    if len(num) == 1 and not denum:
        return False

    new_num, new_denum = [], []

    change = False

    out_type = out.type
    for candidate in list(num):
        if candidate not in num:
            continue
        num.remove(candidate)
        _change, candidate, num, denum = attempt_distribution(
            candidate,
            num,
            denum,
            out_type,
        )

        change |= _change
        new_num.append(candidate)

    for candidate in list(denum):
        if candidate not in denum:
            continue
        denum.remove(candidate)
        _change, candidate, denum, num = attempt_distribution(
            candidate, denum, num, out_type
        )
        change |= _change
        new_denum.append(candidate)
    if not change:
        return False

    new_num += num
    new_denum += denum

    rval = local_mul_canonizer.merge_num_denum(new_num, new_denum)

    if rval.type != out.type:
        # WHY DOES THIS HAPPEN?
        return False

    return [rval]


get_clients_at_depth1 = partial(get_clients_at_depth, depth=1)
get_clients_at_depth2 = partial(get_clients_at_depth, depth=2)

# 1+erf(x)=>erfc(-x)
local_one_plus_erf = PatternNodeRewriter(
    (add, 1, (erf, "x")),
    (erfc, (neg, "x")),
    allow_multiple_clients=True,
    name="local_one_plus_erf",
    tracks=[erf],
    get_nodes=get_clients_at_depth1,
)
register_canonicalize(local_one_plus_erf)
register_stabilize(local_one_plus_erf)
register_specialize(local_one_plus_erf)

# 1-erf(x)=>erfc(x)
local_one_minus_erf = PatternNodeRewriter(
    (sub, 1, (erf, "x")),
    (erfc, "x"),
    allow_multiple_clients=True,
    name="local_one_minus_erf",
    tracks=[erf],
    get_nodes=get_clients_at_depth1,
)
register_canonicalize(local_one_minus_erf)
register_stabilize(local_one_minus_erf)
register_specialize(local_one_minus_erf)

# (-1)+erf(x) => -erfc(x)
# There is no need for erf(x)+(-1) nor erf(x) - 1, as the `local_add_mul`
# canonicalize will convert those to the matched pattern
local_erf_minus_one = PatternNodeRewriter(
    (add, -1, (erf, "x")),
    (neg, (erfc, "x")),
    allow_multiple_clients=True,
    name="local_erf_minus_one",
    tracks=[erf],
    get_nodes=get_clients_at_depth1,
)
register_canonicalize(local_erf_minus_one)
register_stabilize(local_erf_minus_one)
register_specialize(local_erf_minus_one)

# 1-erfc(x) => erf(x)
local_one_minus_erfc = PatternNodeRewriter(
    (sub, 1, (erfc, "x")),
    (erf, "x"),
    allow_multiple_clients=True,
    name="local_one_minus_erfc",
    tracks=[erfc],
    get_nodes=get_clients_at_depth1,
)
register_canonicalize(local_one_minus_erfc)
register_stabilize(local_one_minus_erfc)
register_specialize(local_one_minus_erfc)

# -1 + erfc(-x)=>erf(x)
local_erf_neg_minus_one = PatternNodeRewriter(
    (add, -1, (erfc, (neg, "x"))),
    (erf, "x"),
    allow_multiple_clients=True,
    name="local_erf_neg_minus_one",
    tracks=[erfc],
    get_nodes=get_clients_at_depth1,
)
register_canonicalize(local_erf_neg_minus_one)
register_stabilize(local_erf_neg_minus_one)
register_specialize(local_erf_neg_minus_one)


@register_stabilize
@register_specialize
@node_rewriter([log])
def local_log_erfc(fgraph, node):
    """Stability rewrite for ``log(erfc(x))``.

    Notes
    -----
        log(erfc(x)) => when x>threshold,
                    -x**2-log(x)-.5*log(pi)+log(1-1/(2*x**2)+3/(4*x**4)-15/(8*x**6))
        for float64: threshold=26.641747557 was chosen with:
        [(i,numpy.log(scipy.special.erfc(numpy.asarray([i],dtype='float64'))))
        for i in numpy.arange(26.641747557,26.6417475571,.00000000001)]
        for float32: threshold=10.0541949, [(i,numpy.log(scipy.special.erfc(
            numpy.asarray([i],dtype='float32')))) for i in numpy.arange(
            10.0541948,10.0541951,.0000001)]
    """

    if not (node.inputs[0].owner and node.inputs[0].owner.op == erfc):
        return False

    if hasattr(node.tag, "local_log_erfc_applied"):
        # We use that flag to don't apply the rewrite recursively
        # TODO FIXME: We shouldn't need to use tags for this.
        return False

    node.tag.local_log_erfc_applied = True

    x = node.inputs[0].owner.inputs[0]
    stab_value = (
        -(x**2)
        - log(x)
        - 0.5 * log(np.pi)
        + log(1 - 1 / (2 * x**2) + 3 / (4 * x**4) - 15 / (8 * x**6))
    )

    if node.outputs[0].dtype == "float32" or node.outputs[0].dtype == "float16":
        threshold = 10.0541949
    elif node.outputs[0].dtype == "float64":
        threshold = 26.641747557

    ret = switch(x < threshold, node.outputs[0], stab_value)
    ret.tag.values_eq_approx = values_eq_approx_remove_inf
    return [ret]


@register_stabilize
@register_specialize
@node_rewriter([true_div])
def local_grad_log_erfc_neg(fgraph, node):
    """Stability rewrite for the grad of ``log(erfc(x))``.

    Notes
    -----
        ([y*]exp(-(x**2)))/erfc(x)  # The y* is optional
        ([y*]exp(x**2))/erfc(-x) => [y*](when x > threshold,
                                sqrt(pi)*-x/(1-1/(2*x**2)+3/(4*x**4)-15/(8*x**6)))

        for float64: threshold=26.63 see at the end of the fct for the explanation
        for float32: threshold=9.3 see at the end of the fct for the explanation

    TODO: remove the constraint that there are only 2 inputs to exp(x**2)
    is the second.

    TODO: at the test point 10 in float32, there is instability in the original
    value. The original gives -30.0, the stab -20.1 and in float64 -18.1.
    Make it so that the test does not generate an error in that case!

    """
    if not (node.inputs[1].owner and node.inputs[1].owner.op == erfc):
        return False

    erfc_in = node.inputs[1]
    erfc_x = erfc_in.owner.inputs[0]

    if node.inputs[0].owner is None:
        return False

    # TODO: All of this should be replaced with a single, simple unification
    # The mul is optional.
    if node.inputs[0].owner.op != mul:
        mul_in = None
        y = []
        if not (node.inputs[0].owner and node.inputs[0].owner.op == exp):
            return False
        exp_in = node.inputs[0]
    else:
        mul_in = node.inputs[0]
        exp_in = None
        for idx, inp in enumerate(mul_in.owner.inputs):
            if inp.owner and inp.owner.op == exp:
                exp_in = inp
                break
        else:
            return False

        if len(mul_in.owner.inputs) == 2:
            y = [mul_in.owner.inputs[1 - idx]]
        else:
            y = mul_in.owner.inputs[:]
            del y[idx]

    if exp_in.owner.inputs[0].owner is None:
        return False

    if exp_in.owner.inputs[0].owner.op == neg:
        neg_in = exp_in.owner.inputs[0]
        if not (
            neg_in.owner.inputs[0].owner and neg_in.owner.inputs[0].owner.op == sqr
        ):
            return False
        sqr_in = neg_in.owner.inputs[0]
        x = sqr_in.owner.inputs[0]
    elif exp_in.owner.inputs[0].owner.op == mul:
        # We should compare that -(erfc_x**2) is equivalent to mul_neg.
        # There is currently no easy way to do this in the general case,
        # so we implement some common case for now.

        # In many cases the neg are replaced by mul in the graph.
        # This also allows to stabilize log(erfc(cst*x)).
        mul_neg = exp_in.owner.inputs[0]

        # In case that multiple mul are not fused together, we do it here.
        def check_input(inputs):
            new_inputs = []
            for i in inputs:
                if i.owner and i.owner.op == mul:
                    new_inputs.extend(check_input(i.owner.inputs))
                else:
                    new_inputs.append(i)
            return new_inputs

        mul_inputs = check_input(mul_neg.owner.inputs)

        # Put the constant first.
        for i in range(len(mul_inputs)):
            if isinstance(i, Constant):
                if i == 0:
                    break
                else:
                    tmp = mul_inputs[0]
                    mul_inputs[0] = mul_inputs[i]
                    mul_inputs[i] = tmp
                    break
        mul_neg = mul(*mul_inputs)

        try:
            cst2 = get_underlying_scalar_constant_value(
                mul_neg.owner.inputs[0], only_process_constants=True
            )
        except NotScalarConstantError:
            return False

        if len(mul_neg.owner.inputs) == 2:
            if not (
                mul_neg.owner.inputs[1].owner
                and mul_neg.owner.inputs[1].owner.op == sqr
            ):
                return False
            sqr_in = mul_neg.owner.inputs[1]
            x = sqr_in.owner.inputs[0]
        elif len(mul_neg.owner.inputs) == 3:
            if mul_neg.owner.inputs[1] is not mul_neg.owner.inputs[2]:
                return False
            x = mul_neg.owner.inputs[1]
        else:
            return False

        if cst2 != -1:
            if not (
                erfc_x.owner
                and erfc_x.owner.op == mul
                and len(erfc_x.owner.inputs) == 2
            ):
                # todo implement that case
                return False
            if erfc_x.owner.inputs[1] is not mul_neg.owner.inputs[1]:
                return False

            x = erfc_x
            try:
                cst = get_underlying_scalar_constant_value(
                    erfc_x.owner.inputs[0], only_process_constants=True
                )
            except NotScalarConstantError:
                return False
            if cst2 != -cst * 2:
                return False

            # The constant is valid. Must check that the
        elif erfc_x is not x:
            return False

    else:
        return False

    if hasattr(node.tag, "local_grad_log_erfc_neg"):
        # We use that flag to don't apply the rewrite recursively
        # TODO FIXME: We shouldn't need to use tags for this.
        return False

    if erfc_x is not x:
        return None

    # we move the y outside the div.
    true_div_no_mul = true_div(exp_in, erfc_in)
    true_div_no_mul.owner.tag.local_grad_log_erfc_neg = True

    # aaron value
    stab_value = (
        x
        * pt_pow(1 - 1 / (2 * (x**2)) + 3 / (4 * (x**4)) - 15 / (8 * (x**6)), -1)
        * cast(sqrt(np.pi), dtype=x.dtype)
    )

    if x.dtype == "float32" or x.dtype == "float16":
        threshold = 9.3
        # threshold = 10.1
    elif x.dtype == "float64":
        threshold = 26.641747557

    ret = switch(x < threshold, true_div_no_mul, stab_value)

    if y:
        ret = mul(ret, *y)

    ret.tag.values_eq_approx = values_eq_approx_remove_inf_nan

    return [ret]


def isclose(x, ref, rtol=0, atol=0, num_ulps=10):
    """

    Returns
    -------
    bool
        True iff x is a constant close to ref (by default 10 ULPs).

    """
    x = np.asarray(x)
    if np.issubdtype(x.dtype, np.floating):
        atol = atol + num_ulps * np.abs(np.spacing(x.dtype.type(ref)))
    return np.allclose(x, ref, rtol=rtol, atol=atol)


def _is_1(expr):
    """

    Returns
    -------
    bool
        True iff expr is a constant close to 1.

    """
    try:
        v = get_underlying_scalar_constant_value(expr)
        return isclose(v, 1)
    except NotScalarConstantError:
        return False


logsigm_to_softplus = PatternNodeRewriter(
    (log, (sigmoid, "x")),
    (neg, (softplus, (neg, "x"))),
    allow_multiple_clients=True,
    values_eq_approx=values_eq_approx_remove_inf,
    tracks=[sigmoid],
    get_nodes=get_clients_at_depth1,
)
log1msigm_to_softplus = PatternNodeRewriter(
    (log, (sub, dict(pattern="y", constraint=_is_1), (sigmoid, "x"))),
    (neg, (softplus, "x")),
    allow_multiple_clients=True,
    values_eq_approx=values_eq_approx_remove_inf,
    tracks=[sigmoid],
    get_nodes=get_clients_at_depth2,
)
log1p_neg_sigmoid = PatternNodeRewriter(
    (log1p, (neg, (sigmoid, "x"))),
    (neg, (softplus, "x")),
    values_eq_approx=values_eq_approx_remove_inf,
    allow_multiple_clients=True,
    tracks=[sigmoid],
    get_nodes=get_clients_at_depth2,
)

register_stabilize(logsigm_to_softplus, name="logsigm_to_softplus")
register_stabilize(log1msigm_to_softplus, name="log1msigm_to_softplus")
register_stabilize(log1p_neg_sigmoid, name="log1p_neg_sigmoid")
register_specialize(log1p_neg_sigmoid, name="log1p_neg_sigmoid")


def is_1pexp(t, only_process_constants=True):
    """

    Returns
    -------
    object
        If 't' is of the form (1+exp(x)), return (False, x).
        Else return None.

    """
    if t.owner and t.owner.op == add:
        scalars, _scalar_inputs, nonconsts = scalarconsts_rest(
            t.owner.inputs, only_process_constants=only_process_constants
        )
        # scalar_inputs are potentially dimshuffled and filled with scalars
        if len(nonconsts) == 1:
            maybe_exp = nonconsts[0]
            if maybe_exp.owner and maybe_exp.owner.op == exp:
                # Verify that the constant terms sum to 1.
                if scalars:
                    scal_sum = scalars[0]
                    for s in scalars[1:]:
                        scal_sum = scal_sum + s
                    if isclose(scal_sum, 1):
                        return False, maybe_exp.owner.inputs[0]
    return None


def is_exp(var):
    """
    Match a variable with either of the `exp(x)` or `-exp(x)` patterns.

    Parameters
    ----------
    var
        The Variable to analyze.

    Returns
    -------
    tuple
        A pair (b, x) with `b` a boolean set to True if `var` is of the
        form `-exp(x)` and False if `var` is of the form `exp(x)`. If `var`
        cannot be cast into either form, then return `None`.

    """
    _neg = False
    neg_info = is_neg(var)
    if neg_info is not None:
        _neg = True
        var = neg_info
    if var.owner and var.owner.op == exp:
        return _neg, var.owner.inputs[0]


def is_mul(var):
    """
    Match a variable with `x * y * z * ...`.

    Parameters
    ----------
    var
        The Variable to analyze.

    Returns
    -------
    object
        A list [x, y, z, ...] if `var` is of the form `x * y * z * ...`,
        or None if `var` cannot be cast into this form.

    """
    if var.owner and var.owner.op == mul:
        return var.owner.inputs
    else:
        return None


def partition_num_or_denom(r, f):
    if r.owner and r.owner.op == mul:
        a = r.owner.inputs
    else:
        a = [r]

    # ugly 2.4-compatible thing
    f_terms = []
    _neg = False
    rest = []
    for t in a:
        f_t = f(t)
        if f_t is None:
            rest.append(t)
        else:
            neg_t, f_t = f_t
            f_terms.append(f_t)
            _neg ^= neg_t  # bit flip if neg_t is true
    return f_terms, rest, _neg


def is_neg(var):
    """
    Match a variable with the `-x` pattern.

    Parameters
    ----------
    var
        The Variable to analyze.

    Returns
    -------
    object
        `x` if `var` is of the form `-x`, or None otherwise.

    """
    var_node = var.owner
    if not var_node:
        return None
    # First match against `neg`.
    if var_node.op == neg:
        return var_node.inputs[0]
    # Then match against a multiplication by -1.
    if var_node.op == mul and len(var_node.inputs) >= 2:
        for idx, mul_input in enumerate(var_node.inputs):
            try:
                constant = get_underlying_scalar_constant_value(mul_input)
                is_minus_1 = isclose(constant, -1)
            except NotScalarConstantError:
                is_minus_1 = False
            if is_minus_1:
                # Found a multiplication by -1.
                if len(var_node.inputs) == 2:
                    # Only return the other input.
                    return var_node.inputs[1 - idx]
                else:
                    # Return the multiplication of all other inputs.
                    return mul(*(var_node.inputs[0:idx] + var_node.inputs[idx + 1 :]))
    # No match.
    return None


@register_stabilize
@node_rewriter([true_div])
def local_exp_over_1_plus_exp(fgraph, node):
    """

    exp(x)/(1+exp(x)) -> sigm(x)
    c/(1+exp(x)) -> c*sigm(-x)

    """
    # This rewrite should be done for numerical stability
    # so we don't care to check client counts
    # find all the exp() terms in the numerator
    num, denom = node.inputs
    num_exp_x, num_rest, num_neg = partition_num_or_denom(num, is_exp)
    denom_1pexp, denom_rest, denom_neg = partition_num_or_denom(denom, is_1pexp)

    sigmoids = []
    for t in denom_1pexp:
        if t in num_exp_x:
            # case: exp(x) /(1+exp(x))
            sigmoids.append(sigmoid(t))
            del num_exp_x[num_exp_x.index(t)]
        else:
            # case: 1/(1+exp(x))
            sigmoids.append(sigmoid(-t))
        copy_stack_trace(node.outputs[0], sigmoids[-1])

    if not sigmoids:  # we didn't find any.  abort
        return
    # put the new numerator together
    new_num = sigmoids + [exp(t) for t in num_exp_x] + num_rest
    new_num = variadic_mul(*new_num)

    if num_neg ^ denom_neg:
        new_num = -new_num

    copy_stack_trace(num, new_num)

    if len(denom_rest) == 0:
        return [new_num]
    elif len(denom_rest) == 1:
        out = new_num / denom_rest[0]
    else:
        out = new_num / mul(*denom_rest)

    copy_stack_trace(node.outputs[0], out)
    return [out]


def parse_mul_tree(root):
    """
    Parse a tree of multiplications starting at the given root.

    Parameters
    ----------
    root
        The variable at the root of the tree.

    Returns
    -------
    object
        A tree where each non-leaf node corresponds to a multiplication
        in the computation of `root`, represented by the list of its inputs.
        Each input is a pair [n, x] with `n` a boolean value indicating whether
        sub-tree `x` should be negated.

    Examples
    --------

    .. code-block:: python

        x * y               -> [False, [[False, x], [False, y]]]
        -(x * y)            -> [True, [[False, x], [False, y]]]
        -x * y              -> [False, [[True, x], [False, y]]]
        -x                  -> [True, x]
        (x * y) * -z        -> [False, [[False, [[False, x], [False, y]]],
                                        [True, z]]]

    """
    # Is it a multiplication?
    mul_info = is_mul(root)
    if mul_info is None:
        # Is it a negation?
        neg_info = is_neg(root)
        if neg_info is None:
            # Keep the root "as is".
            return [False, root]
        else:
            # Recurse, inverting the negation.
            neg, sub_tree = parse_mul_tree(neg_info)
            return [not neg, sub_tree]
    else:
        # Recurse into inputs.
        return [False, list(map(parse_mul_tree, mul_info))]


def replace_leaf(arg, leaves, new_leaves, op, neg):
    """
    Attempt to replace a leaf of a multiplication tree.

    We search for a leaf in `leaves` whose argument is `arg`, and if we find
    one, we remove it from `leaves` and add to `new_leaves` a leaf with
    argument `arg` and variable `op(arg)`.

    Parameters
    ----------
    arg
        The argument of the leaf we are looking for.
    leaves
        List of leaves to look into. Each leaf should be a pair
        (x, l) with `x` the argument of the Op found in the leaf, and `l` the
        actual leaf as found in a multiplication tree output by `parse_mul_tree`
        (i.e. a pair [boolean, variable]).
    new_leaves
        If a replacement occurred, then the leaf is removed from `leaves`
        and added to the list `new_leaves` (after being modified by `op`).
    op
        A function that, when applied to `arg`, returns the Variable
        we want to replace the original leaf variable with.
    neg : bool
        If True, then the boolean value associated to the leaf should
        be swapped. If False, then this value should remain unchanged.

    Returns
    -------
    bool
        True if a replacement occurred, or False otherwise.

    """
    for idx, x in enumerate(leaves):
        if x[0] == arg:
            x[1][0] ^= neg
            x[1][1] = op(arg)
            leaves.pop(idx)
            new_leaves.append(x)
            return True
    return False


def simplify_mul(tree):
    """
    Simplify a multiplication tree.

    Parameters
    ----------
    tree
        A multiplication tree (as output by `parse_mul_tree`).

    Returns
    -------
    object
        A multiplication tree computing the same output as `tree` but without
        useless multiplications by 1 nor -1 (identified by leaves of the form
        [False, None] or [True, None] respectively). Useless multiplications
        (with less than two inputs) are also removed from the tree.

    """
    neg, inputs = tree
    if isinstance(inputs, list):
        # Recurse through inputs.
        s_inputs = []
        for s_i in map(simplify_mul, inputs):
            if s_i[1] is None:
                # Multiplication by +/-1.
                neg ^= s_i[0]
            else:
                s_inputs.append(s_i)
        if not s_inputs:
            # The multiplication is empty.
            rval = [neg, None]
        elif len(s_inputs) == 1:
            # The multiplication has a single input.
            s_inputs[0][0] ^= neg
            rval = s_inputs[0]
        else:
            rval = [neg, s_inputs]
    else:
        rval = tree
    # print(f"simplify_mul: {tree} -> {rval}")
    return rval


def compute_mul(tree):
    """
    Compute the Variable that is the output of a multiplication tree.

    This is the inverse of the operation performed by `parse_mul_tree`, i.e.
    compute_mul(parse_mul_tree(tree)) == tree.

    Parameters
    ----------
    tree
        A multiplication tree (as output by `parse_mul_tree`).

    Returns
    -------
    object
        A Variable that computes the multiplication represented by the tree.

    """
    neg, inputs = tree
    if inputs is None:
        raise AssertionError(
            "Function `compute_mul` found a missing leaf, did you forget to "
            "call `simplify_mul` on the tree first?"
        )
    elif isinstance(inputs, list):
        # Recurse through inputs.
        rval = mul(*map(compute_mul, inputs))
    else:
        rval = inputs
    if neg:
        rval = -rval
    return rval


def perform_sigm_times_exp(
    tree,
    exp_x=None,
    exp_minus_x=None,
    sigm_x=None,
    sigm_minus_x=None,
    parent=None,
    child_idx=None,
    full_tree=None,
):
    """
    Core processing of the `local_sigm_times_exp` rewrite.

    This recursive function operates on a multiplication tree as output by
    `parse_mul_tree`. It walks through the tree and modifies it in-place
    by replacing matching pairs (exp, sigmoid) with the desired version.

    Parameters
    ----------
    tree
        The sub-tree to operate on.
    exp_x
        List of arguments ``x`` so that ``exp(x)`` exists somewhere in the whole
        multiplication tree. Each argument is a pair ``(x, leaf)`` with ``x`` the
        argument of the exponential, and ``leaf`` the corresponding leaf in the
        multiplication tree (of the form ``[n, exp(x)]`` -- see `parse_mul_tree`).
        If ``None``, this argument is initialized to an empty list.
    exp_minus_x
        Similar to `exp_x`, but for ``exp(-x)``.
    sigm_x
        Similar to `exp_x`, but for ``sigmoid(x)``.
    sigm_minus_x
        Similar to `exp_x`, but for ``sigmoid(-x)``.
    parent
        Parent of `tree` (``None`` if `tree` is the global root).
    child_idx
        Index of `tree` in its parent's inputs (``None`` if `tree` is the global
        root).
    full_tree
        The global multiplication tree (should not be set except by recursive
        calls to this function). Used for debugging only.

    Returns
    -------
    bool
        ``True`` if a modification was performed somewhere in the whole multiplication
        tree, or ``False`` otherwise.

    """
    if exp_x is None:
        exp_x = []
    if exp_minus_x is None:
        exp_minus_x = []
    if sigm_x is None:
        sigm_x = []
    if sigm_minus_x is None:
        sigm_minus_x = []
    if full_tree is None:
        full_tree = tree
    # if False:  # Debug code.
    #     print("<perform_sigm_times_exp>")
    #     print(f"  full_tree   = {full_tree}")
    #     print(f"  tree        = {tree}")
    #     print(f"  exp_x       = {exp_x}")
    #     print(f"  exp_minus_x = {exp_minus_x}")
    #     print(f"  sigm_x      = {sigm_x}")
    #     print(f"  sigm_minus_x= {sigm_minus_x}")
    neg, inputs = tree
    if isinstance(inputs, list):
        # Recurse through inputs of the multiplication.
        rval = False
        for sub_idx, sub_tree in enumerate(inputs):
            rval |= perform_sigm_times_exp(
                tree=sub_tree,
                parent=tree,
                child_idx=sub_idx,
                exp_x=exp_x,
                exp_minus_x=exp_minus_x,
                sigm_x=sigm_x,
                sigm_minus_x=sigm_minus_x,
                full_tree=full_tree,
            )
        return rval
    else:
        # Reached a leaf: if it is an exponential or a sigmoid, then we
        # first attempt to find a match in leaves already visited.
        # If there is such a match, we modify the already-visited leaf
        # accordingly: for instance if we visited a leaf sigmoid(x), then
        # find later a -exp(-x), we replace the previous leaf by
        # -sigmoid(-x) and remove the -exp(-x) from the tree.
        # If no match is found, then we register this leaf so that it can
        # be found later while walking the tree.
        var = inputs
        keep_it = False
        exp_info = is_exp(var)
        if exp_info is not None:
            exp_neg, exp_arg = exp_info
            neg ^= exp_neg
            neg_arg = is_neg(exp_arg)
            if neg_arg is None:
                if not replace_leaf(exp_arg, sigm_minus_x, sigm_x, sigmoid, neg):
                    exp_x.append((exp_arg, tree))
                    keep_it = True
            else:
                if not replace_leaf(
                    neg_arg, sigm_x, sigm_minus_x, lambda x: sigmoid(-x), neg
                ):
                    exp_minus_x.append((neg_arg, tree))
                    keep_it = True
        elif var.owner and var.owner.op == sigmoid:
            sigm_arg = var.owner.inputs[0]
            neg_arg = is_neg(sigm_arg)
            if neg_arg is None:
                if not replace_leaf(
                    sigm_arg, exp_minus_x, sigm_minus_x, lambda x: sigmoid(-x), neg
                ):
                    sigm_x.append((sigm_arg, tree))
                    keep_it = True
            else:
                if not replace_leaf(neg_arg, exp_x, sigm_x, sigmoid, neg):
                    sigm_minus_x.append((neg_arg, tree))
                    keep_it = True
        else:
            # It is not an exponential nor a sigmoid.
            keep_it = True
        if not keep_it:
            # Delete this leaf, i.e. replace it by [False, None] (corresponding
            # to a multiplication by 1).
            assert parent is not None
            parent[1][child_idx] = [False, None]
        return not keep_it


@register_stabilize
@node_rewriter([mul])
def local_sigm_times_exp(fgraph, node):
    """
    exp(x) * sigm(-x) -> sigm(x)
    exp(-x) * sigm(x) -> sigm(-x)

    todo: add stack traces to the intermediate variables
    """
    # Obtain tree of multiplications starting at this node.
    mul_tree = parse_mul_tree(node.outputs[0])
    did_something = perform_sigm_times_exp(mul_tree)
    if not did_something:
        # No change.
        return None
    # The rewrite may have introduced multiplications by 1 in the tree:
    # get rid of them.
    mul_tree = simplify_mul(mul_tree)
    # Recompute final output based on the updated tree.
    out = compute_mul(mul_tree)
    # keep the stack trace
    copy_stack_trace(node.outputs[0], out)
    return [out]


@register_stabilize
@node_rewriter([reciprocal])
def local_reciprocal_1_plus_exp(fgraph, node):
    """``reciprocal(1+exp(x)) -> sigm(-x)``

    TODO: This is redundant; we can just decided on *one* canonical form
    for division (e.g. either `true_div` or `reciprocal`) and have this
    taken care of with existing rewrites.
    """
    # This Rewrite should be done for numerical stability
    # so we don't care to check client counts
    reciprocal_arg = node.inputs[0]
    if reciprocal_arg.owner and reciprocal_arg.owner.op == add:
        scalars_, _scalar_inputs, nonconsts = scalarconsts_rest(
            reciprocal_arg.owner.inputs, only_process_constants=True
        )
        # scalar_inputs are potentially dimshuffled and fill'd scalars
        if len(nonconsts) == 1:
            if nonconsts[0].owner and nonconsts[0].owner.op == exp:
                if scalars_ and isclose(np.sum(scalars_), 1):
                    out = [
                        alloc_like(
                            sigmoid(neg(nonconsts[0].owner.inputs[0])),
                            node.outputs[0],
                            fgraph,
                        )
                    ]
                    # keep combined stack traces of
                    #     exp(x):           nonconsts[0],
                    #     1 + exp(x):       reciprocal_arg,
                    #     1 / (1 + exp(x)): node.outputs[0]
                    copy_stack_trace(
                        [nonconsts[0], reciprocal_arg, node.outputs[0]], out
                    )
                    return out


# 1 - sigmoid(x) -> sigmoid(-x)
local_1msigmoid = PatternNodeRewriter(
    (sub, dict(pattern="y", constraint=_is_1), (sigmoid, "x")),
    (sigmoid, (neg, "x")),
    tracks=[sigmoid],
    get_nodes=get_clients_at_depth1,
    name="local_1msigmoid",
)
register_stabilize(local_1msigmoid)
register_specialize(local_1msigmoid)


@register_stabilize
@node_rewriter([log1p])
def local_log1p_plusminus_exp(fgraph, node):
    """Transforms log1p of ±exp(x) into log1pexp (aka softplus) / log1mexp
    ``log1p(exp(x))  -> log1pexp(x)``
    ``log1p(-exp(x)) -> log1mexp(x)``
    where "-" can be "neg" or any other expression detected by "is_neg"
    """
    (log1p_arg,) = node.inputs
    exp_info = is_exp(log1p_arg)
    if exp_info is not None:
        exp_neg, exp_arg = exp_info
        if exp_neg:
            return [log1mexp(exp_arg)]
        else:
            return [log1pexp(exp_arg)]  # aka softplus


@register_stabilize
@node_rewriter([expm1])
def logmexpm1_to_log1mexp(fgraph, node):
    """``log(-expm1(x)) -> log1mexp(x)``
    where "-" can be "neg" or any other expression detected by "is_neg"
    """
    rewrites = {}
    for node in get_clients_at_depth(fgraph, node, depth=2):
        if node.op == log:
            (log_arg,) = node.inputs
            neg_arg = is_neg(log_arg)
            if neg_arg is not None and neg_arg.owner and neg_arg.owner.op == expm1:
                (expm1_arg,) = neg_arg.owner.inputs
                rewrites[node.outputs[0]] = log1mexp(expm1_arg)
    return rewrites


# log(exp(a) - exp(b)) -> a + log1mexp(b - a)
logdiffexp_to_log1mexpdiff = PatternNodeRewriter(
    (log, (sub, (exp, "x"), (exp, "y"))),
    (add, "x", (log1mexp, (sub, "y", "x"))),
    allow_multiple_clients=True,
)
register_stabilize(logdiffexp_to_log1mexpdiff, name="logdiffexp_to_log1mexpdiff")

# log(sigmoid(x) / (1 - sigmoid(x))) -> x
# i.e logit(sigmoid(x)) -> x
local_logit_sigmoid = PatternNodeRewriter(
    (log, (true_div, (sigmoid, "x"), (sub, 1, (sigmoid, "x")))),
    "x",
    tracks=[sigmoid],
    get_nodes=get_clients_at_depth2,
    allow_multiple_clients=True,
    name="local_logit_sigmoid",
)
register_canonicalize(local_logit_sigmoid)
register_specialize(local_logit_sigmoid)

# sigmoid(log(x / (1-x)) -> x
# i.e., sigmoid(logit(x)) -> x
local_sigmoid_logit = PatternNodeRewriter(
    (sigmoid, (log, (true_div, "x", (sub, 1, "x")))),
    "x",
    allow_multiple_clients=True,
    name="local_sigmoid_logit",
)
register_canonicalize(local_sigmoid_logit)
register_specialize(local_sigmoid_logit)


@register_canonicalize
@register_useless
@node_rewriter([_conj])
def local_useless_conj(fgraph, node):
    r"""Remove `conj` `Op`\s applied to non-imaginary variable types."""
    x = node.inputs[0]
    if x.type.dtype not in complex_dtypes:
        return [x]


local_polygamma_to_digamma = PatternNodeRewriter(
    (polygamma, 0, "x"),
    (digamma, "x"),
    allow_multiple_clients=True,
    name="local_polygamma_to_digamma",
)

register_specialize(local_polygamma_to_digamma)

local_polygamma_to_tri_gamma = PatternNodeRewriter(
    (polygamma, 1, "x"),
    (tri_gamma, "x"),
    allow_multiple_clients=True,
    name="local_polygamma_to_tri_gamma",
)

register_specialize(local_polygamma_to_tri_gamma)

local_log_kv = PatternNodeRewriter(
    # Rewrite log(kv(v, x)) = log(kve(v, x) * exp(-x)) -> log(kve(v, x)) - x
    # During stabilize -x is converted to -1.0 * x
    (log, (mul, (kve, "v", "x"), (exp, (mul, -1.0, "x")))),
    (sub, (log, (kve, "v", "x")), "x"),
    allow_multiple_clients=True,
    name="local_log_kv",
    # Start the rewrite from the less likely kve node
    tracks=[kve],
    get_nodes=get_clients_at_depth2,
)

register_stabilize(local_log_kv)
