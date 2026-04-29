from collections.abc import Iterable, Sequence
from typing import cast

import numpy as np
from numpy.lib.array_utils import normalize_axis_index, normalize_axis_tuple

from pytensor.compile import optdb
from pytensor.graph import (
    Constant,
    FunctionGraph,
    Variable,
    node_rewriter,
    vectorize_graph,
)
from pytensor.graph.rewriting.basic import (
    NodeRewriter,
    SequentialGraphRewriter,
    copy_stack_trace,
    out2in,
)
from pytensor.tensor.basic import (
    Alloc,
    AllocDiag,
    ARange,
    ExtractDiag,
    Eye,
    Join,
    MakeVector,
    alloc,
    arange,
    as_tensor,
    expand_dims,
    get_underlying_scalar_constant_value,
    join,
    register_infer_shape,
)
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.elemwise import CAReduce, DimShuffle, Elemwise
from pytensor.tensor.exceptions import NotScalarConstantError
from pytensor.tensor.extra_ops import squeeze
from pytensor.tensor.math import Dot, dot, minimum
from pytensor.tensor.rewriting.basic import (
    register_canonicalize,
    register_specialize,
    register_stabilize,
)
from pytensor.tensor.rewriting.elemwise import local_dimshuffle_lift
from pytensor.tensor.rewriting.subtensor import (
    _constant_has_unique_indices,
    local_adv_idx_to_diagonal,
    local_adv_idx_to_slice,
    local_advanced_read_of_write_constant_indices,
    local_slice_read_of_write,
    local_useless_slice,
    register_useless,
)
from pytensor.tensor.shape import (
    Reshape,
    Shape,
    SpecifyShape,
    specify_shape,
)
from pytensor.tensor.special import Softmax, softmax
from pytensor.tensor.subtensor import (
    AdvancedIncSubtensor,
    AdvancedSubtensor,
    AdvancedSubtensor1,
    Subtensor,
    _non_consecutive_adv_indexing,
    as_index_literal,
    get_constant_idx,
    get_idx_list,
    indexed_result_shape,
    indices_from_subtensor,
)
from pytensor.tensor.type import TensorType
from pytensor.tensor.variable import TensorConstant, TensorVariable


def _canonical_indexing(var, indices):
    """Index ``var``, squeezing indexed broadcast dims whose index has size 1.

    On a length-1 dim only zero is a valid index, so the index is
    redundant — squeezing is equivalent and simpler.  If squeezed
    indices contributed unique output dimensions, those are reinserted
    via ``expand_dims`` after indexing.
    """
    squeeze_axes = []
    kept_indices = []
    max_drop_ndim = 0
    max_kept_ndim = 0
    first_adv_axis = None
    for axis, (bcast, idx) in enumerate(
        zip(var.type.broadcastable, indices, strict=False)
    ):
        if isinstance(idx, slice):
            kept_indices.append(idx)
        else:
            if first_adv_axis is None:
                first_adv_axis = axis

            # np.ndim works for all supported cases: int, numpy arrays, pytensor variables
            idx_ndim = np.ndim(idx)
            if bcast:
                match idx:
                    case Variable():
                        idx_size1 = all(idx.type.broadcastable)
                    case np.ndarray():
                        idx_size1 = idx.size == 1
                    case int() | np.integer():
                        idx_size1 = True
                    case _:
                        raise AssertionError

                # idx only contributes dummy dimensions (if any), not actual shape
                # It doesn't really matter what the index was, only valid values are zeros.
                if idx_size1:
                    max_drop_ndim = max(max_drop_ndim, idx_ndim)
                    squeeze_axes.append(axis)
                    continue

            max_kept_ndim = max(max_kept_ndim, idx_ndim)
            kept_indices.append(idx)

    # Remove useless trailing slice(None) indices
    while kept_indices and kept_indices[-1] == slice(None):
        kept_indices.pop()

    result = var

    if squeeze_axes:
        result = result.squeeze(axis=tuple(squeeze_axes))

    if kept_indices:
        result = result[tuple(kept_indices)]

    if (lost := max_drop_ndim - max_kept_ndim) > 0:
        assert first_adv_axis is not None
        result = expand_dims(
            result, tuple(range(first_adv_axis, first_adv_axis + lost))
        )

    return result


def _dims_dropped_by_basic_index(idxs: Sequence[slice | int]) -> tuple[int, ...]:
    # Inputs can be slice or integer indexes
    # Slices keep the dimensions, integers collapse them
    return tuple(i for i, idx in enumerate(idxs) if not isinstance(idx, slice))


def _ndim_dropped_left_of_axis_by_basic_index(
    idxs: Sequence[slice | int], axis: int
) -> int:
    return len(_dims_dropped_by_basic_index(idxs[:axis]))


def _axis_is_indexed_by_basic_index(
    idxs: Sequence[slice | int], axis: int | Sequence[int]
) -> bool:
    if isinstance(axis, int):
        axis = (axis,)
    return any(ax < len(idxs) and not idxs[ax] == slice(None) for ax in axis)


def _lift_subtensor_non_axis(
    local_subtensor_lift_rewrite: NodeRewriter,
    fgraph: FunctionGraph,
    variable: TensorVariable,
    idx_tuple: tuple[int | slice],
    axis: int,
    old_subtensor_variable: TensorVariable,
) -> None | list[TensorVariable]:
    # Apply generic subtensor lift rewrite along "non-axis" dimensions
    real_indices = [idx for idx in idx_tuple if not idx == slice(None)]
    if len(real_indices) > 1 and variable.type.ndim > 1:
        # Split the subtensor
        idx_to_keep = idx_tuple[axis]
        idxs_to_lift = (*idx_tuple[:axis], slice(None), *idx_tuple[axis + 1 :])

        # Lift the non-axis indexes by calling the rewrite itself
        indexed_variable = variable[idxs_to_lift]
        [indexed_variable] = cast(
            list[TensorVariable],
            local_subtensor_lift_rewrite.transform(fgraph, indexed_variable.owner),
        )
        copy_stack_trace([old_subtensor_variable, indexed_variable], indexed_variable)

        # Then reintroduce the axis index
        ndim_reduced_left = _ndim_dropped_left_of_axis_by_basic_index(idx_tuple, axis)
        new_axis = axis - ndim_reduced_left
        idxs_to_keep = (*(slice(None),) * new_axis, idx_to_keep)
        new_out = indexed_variable[idxs_to_keep]
        copy_stack_trace(old_subtensor_variable, new_out)
        return [new_out]

    else:
        return None


def _index_provably_smaller(idx, val_static_dim) -> bool:
    # Per-axis check: non-repeating indices can't expand a single axis.
    # Does not account for cross-axis broadcast expansion from outer indexing.
    if isinstance(idx, slice) or idx.ndim == 0:
        return True
    if all(idx.type.broadcastable):
        return True
    if idx.type.dtype == "bool":
        return True
    if _constant_has_unique_indices(idx):
        return True
    if isinstance(idx.owner_op, ARange):
        return True
    if isinstance(idx.owner_op, Reshape | DimShuffle):
        # Views that don't add dimensions
        if _index_provably_smaller(idx.owner.inputs[0], val_static_dim):
            return True

    # Fallback to static shape analysis
    if val_static_dim is None:
        return False
    idx_static_shape = idx.type.shape
    if any(d is None for d in idx_static_shape):
        return False
    return bool(np.prod(idx_static_shape) < val_static_dim)


@register_canonicalize
@register_stabilize
@register_specialize
@node_rewriter([Subtensor])
def local_subtensor_of_dot(fgraph, node):
    """Rewrite ``at.dot(A, B)[idxs]`` into ``at.dot(A[idxs_a], B[idxs_b])``.
    ``idxs_a`` is the first ``A.ndim-1`` entries of ``idxs``, and ``idxs_b`` is
    the remaining entries of ``idxs`` (if any), modified to skip the
    second-to-last dimension of ``B`` (because dot sums over this dimension).
    """
    x, *idx_vars = node.inputs
    if not (
        x.owner is not None
        and (
            isinstance(x.owner.op, Dot)
            or (
                isinstance(x.owner.op, Blockwise)
                and isinstance(x.owner.op.core_op, Dot)
            )
        )
    ):
        return
    # If there is other node that use the outputs of the dot
    # We don't want to compute twice the sub part.
    if len(fgraph.clients[x]) > 1:
        return

    a = x.owner.inputs[0]
    b = x.owner.inputs[1]
    idx_list = indices_from_subtensor(idx_vars, node.op.idx_list)

    if not idx_list:
        # Nothing to do, `local_useless_slice` will handle this case
        return None

    batch_ndim = (
        x.owner.op.batch_ndim(x.owner) if isinstance(x.owner.op, Blockwise) else 0
    )

    if batch_ndim:
        batch_idx_list, idx_list = idx_list[:batch_ndim], idx_list[batch_ndim:]
        if not idx_list:
            # Indexing only over batch dimensions of Blockwise, nothing to do here
            # This will be handled by `local_subtensor_of_batch_dims`
            return None
        # We perform the rest of the rewrite on dummy a, b that correspond to the core case
        a = a.type.clone(shape=a.type.shape[batch_ndim:])()
        b = b.type.clone(shape=b.type.shape[batch_ndim:])()

    a_indices = idx_list[:1]
    b_indices = (slice(None), *idx_list[1:])

    a_sub = a[tuple(a_indices)]
    b_sub = b[tuple(b_indices)]
    r = dot(a_sub, b_sub)

    if batch_ndim:
        # Replace dummy inputs by the original batch ones
        r = vectorize_graph(r, replace={a: x.owner.inputs[0], b: x.owner.inputs[1]})
        r = r[tuple(batch_idx_list)]

    copy_stack_trace([node.outputs[0], node.inputs[0]], r)

    return [r]


@register_canonicalize("shape_unsafe")
@register_specialize("shape_unsafe")
@node_rewriter([Subtensor, AdvancedSubtensor, AdvancedSubtensor1])
def local_subtensor_of_batch_dims(fgraph, node):
    """Lift a (basic or advanced) Subtensor through the batch dims of an Elemwise or Blockwise.

    exp(x)[:, 0] -> exp(x[:, 0])
    add(x, y)[0] -> add(x[0], y[0])
    add(x[None], y)[2] -> add(x, y[2])
    add(x, y)[arange(d), arange(d)] -> add(x[arange(d), arange(d)], y[arange(d), arange(d)])

    Bail on boolean masks and non-consecutive advanced indexing — numpy hoists
    those advanced groups to position 0, which would misalign the lifted
    indices. On a broadcast (length-1) axis of an input, replace the advanced
    index with length-1 zeros so the lifted input still broadcasts correctly.
    """
    elem, *idx = node.inputs

    if not (elem.owner and isinstance(elem.owner.op, Elemwise | Blockwise)):
        return None

    if len(fgraph.clients[elem]) > 1:
        # Elemwise output is used beyond the Subtensor.
        # Get out to avoid repeated computations
        return None

    idx_tuple = indices_from_subtensor(idx, node.op.idx_list)

    if any(isinstance(i, TensorVariable) and i.type.dtype == "bool" for i in idx_tuple):
        # Boolean masks have data-dependent shape.
        return None
    if _non_consecutive_adv_indexing(idx_tuple):
        return None

    # Skip when lifting would expand a gather past a non-broadcast input's size.
    for inp in elem.owner.inputs:
        for axis, idx in enumerate(idx_tuple):
            if axis >= inp.type.ndim:
                break
            if not isinstance(idx, TensorVariable) or idx.type.ndim == 0:
                continue
            if inp.type.broadcastable[axis]:
                continue
            if not _index_provably_smaller(idx, inp.type.shape[axis]):
                return None

    batch_ndim = (
        elem.owner.op.batch_ndim(elem.owner)
        if isinstance(elem.owner.op, Blockwise)
        else elem.ndim
    )

    if len(idx_tuple) > batch_ndim:
        # Indexing on core dimensions of Blockwise. We split the indices and lift the batch ones only
        batch_indices, core_indices = idx_tuple[:batch_ndim], idx_tuple[batch_ndim:]
        if all(idx == slice(None) for idx in batch_indices):
            # No batch indices, nothing to do
            return None
        if any(not isinstance(i, slice) for i in batch_indices) and any(
            isinstance(i, TensorVariable) for i in core_indices
        ):
            # Splitting advanced batch from advanced core indices would hoist
            # the lifted batch indices to position 0.
            return None
        elem_with_batch_indices = elem[batch_indices]
        [elem_with_batch_indices_lifted] = local_subtensor_of_batch_dims.transform(
            fgraph, elem_with_batch_indices.owner
        )
        # Reapply the core_indices
        core_ndim = elem.type.ndim - batch_ndim
        # Number of batch dims may have changed with the lifting of indices, so we recompute
        new_batch_ndim = elem_with_batch_indices_lifted.type.ndim - core_ndim
        new_indices = (*(slice(None),) * new_batch_ndim, *core_indices)
        new_elem = elem_with_batch_indices_lifted[new_indices]
        copy_stack_trace(node.outputs[0], new_elem)
        return [new_elem]

    elem_inputs = elem.owner.inputs
    elem_bcast = elem.type.broadcastable[:batch_ndim]
    if all(inp.type.broadcastable[:batch_ndim] == elem_bcast for inp in elem_inputs):
        # No need to worry about implicit broadcasting.
        indexed_inputs = [inp[idx_tuple] for inp in elem_inputs]

    else:
        # The original indices may not make sense on some of the broadcasted dimensions
        new_idxs = [list(idx_tuple) for _ in elem_inputs]
        for dim, (dim_idx, dim_bcast_out, *dim_bcast_inputs) in enumerate(
            zip(
                idx_tuple,
                elem_bcast,
                *(inp.type.broadcastable[:batch_ndim] for inp in elem_inputs),
                # Indices can be shorter than input ndims
                strict=False,
            )
        ):
            if isinstance(dim_idx, slice) and dim_idx == slice(None):
                # Full slice can be safely applied to all inputs
                continue

            if all(dim_bcast_inp == elem_bcast for dim_bcast_inp in dim_bcast_inputs):
                # This dim is not broadcasted for any of the inputs, original index can be applied to all inputs
                continue

            # Slices stay; advanced indices become length-1 zeros
            # that _canonical_indexing will squeeze.
            if isinstance(dim_idx, slice):
                safe_bcast_dim_idx = slice(None)
            else:
                safe_bcast_dim_idx = np.zeros((1,) * dim_idx.type.ndim, dtype="int64")
            for inp_idx, dim_bcast_inp in zip(new_idxs, dim_bcast_inputs, strict=True):
                if dim_bcast_inp:
                    inp_idx[dim] = safe_bcast_dim_idx

        indexed_inputs = [
            _canonical_indexing(inp, tuple(new_idx))
            for inp, new_idx in zip(elem_inputs, new_idxs, strict=True)
        ]

    [old_out] = node.outputs

    # Copy stack trace to new inputs
    [copy_stack_trace(old_out, new_inp) for new_inp in indexed_inputs]

    # Define elemwise operation on indexed inputs
    new_out = elem.owner.op(*indexed_inputs)

    # Copy stack trace to new output
    copy_stack_trace([old_out, *node.inputs], new_out)

    return [new_out]


@register_canonicalize
@register_specialize
@node_rewriter([Subtensor])
def local_subtensor_of_reduce(fgraph, node):
    """Lift a Subtensor through a CAReduce Op.

    For now rewrite is restricted to single axis of reduction, for simplicity.

    sum(x, axis=1)[0] -> sum(x[0], axis=0)
    sum(x, axis=1)[1:] -> sum(x[1:], axis=1)
    sum(x, axis=0)[0] -> sum(x[:, 0], axis=0)
    sum(x, axis=0)[1:] -> sum(x[:, 1:], axis=0)

    """
    red, *idx = node.inputs

    if not (red.owner and isinstance(red.owner.op, CAReduce)):
        return None

    if len(fgraph.clients[red]) > 1:
        # Don't apply rewrite if another node requires the full reduction
        return None

    [x] = red.owner.inputs
    axis = red.owner.op.axis

    if axis is None:
        axis = tuple(range(x.type.ndim))

    # TODO: Allow reduction across multiple axis
    if len(axis) != 1:
        return None

    [axis] = normalize_axis_tuple(axis, x.ndim)
    idx_tuple = indices_from_subtensor(idx, node.op.idx_list)

    # Index input of reduction.
    new_idxs = list(idx_tuple)
    if axis < len(idx_tuple):
        # When there are indexes beyond the axis of reduction, we need to shift them with None slices.
        new_idxs.insert(axis, slice(None))
    x_sub = x[tuple(new_idxs)]

    [old_out] = node.outputs
    copy_stack_trace(old_out, x_sub)

    # Adjust axis of reduction when indexing drops dimensions (integer indexing as apposed to slice indexing)
    axis -= len(
        [idx_item for idx_item in idx_tuple[:axis] if not isinstance(idx_item, slice)]
    )

    # Apply reduction to indexed input
    out = type(red.owner.op)(axis=axis)(x_sub)
    copy_stack_trace(old_out, out)
    return [out]


@register_canonicalize
@register_specialize
@node_rewriter([Subtensor])
def local_subtensor_of_softmax(fgraph, node):
    """Lift a Subtensor through a Softmax.

    softmax(x, axis=1)[0] -> softmax(x[0], axis=0)
    softmax(x, axis=1)[:, :, 0] -> softmax(x[:, :, 0], axis=1)

    If part of the indexing acts on the axis of reduction, we split it
    softmax(x, axis=1)[:, 0, 1:] -> softmax(x[:, :, 1:], axis=1)[0]

    """
    sm, *idx = node.inputs

    if not (sm.owner and isinstance(sm.owner.op, Softmax)):
        return None

    if len(fgraph.clients[sm]) > 1:
        return None

    [x] = sm.owner.inputs
    axis = sm.owner.op.axis

    if axis is None:
        if x.type.ndim == 1:
            axis = 0
        else:
            # All dimensions are mixed, we can't lift the subtensor
            return None
    elif len(axis) == 1:
        axis = normalize_axis_index(axis[0], sm.ndim)
    else:
        return None

    [old_out] = node.outputs
    idx_tuple = indices_from_subtensor(idx, node.op.idx_list)

    if _axis_is_indexed_by_basic_index(idx_tuple, axis):
        # If there are more dimensions being indexed, we can split them
        # And lift the non-axis indexes while keeping the axis index
        return _lift_subtensor_non_axis(
            local_subtensor_lift_rewrite=local_subtensor_of_softmax,
            fgraph=fgraph,
            variable=sm,
            idx_tuple=idx_tuple,
            axis=axis,
            old_subtensor_variable=old_out,
        )

    # Index input to softmax
    x_sub = x[idx_tuple]

    # Adjust axis of reduction when indexing drops dimensions (integer indexing as apposed to slice indexing)
    axis -= len(
        [idx_item for idx_item in idx_tuple[:axis] if not isinstance(idx_item, slice)]
    )

    out = softmax(x_sub, axis=axis)
    copy_stack_trace(old_out, out)
    return [out]


@register_canonicalize("shape_unsafe")
@register_specialize("shape_unsafe")
@node_rewriter([Subtensor])
def local_subtensor_of_expand_dims(fgraph, node):
    """Lift a Subtensor through a DimShuffle that only expands dims.

    expand_dims(x, axis=0)[0] -> x
    expand_dims(x, axis=0)[:, 0] -> expand_dims(x[0], axis=0)
    expand_dims(x, axis=2)[0] -> expand_dims(x[0], axis=1)

    This goes beyond `local_subtensor_remove_broadcastable_index` which
    simply removes useless subtensors on broadcastable dimensions.
    """
    ds, *idx = node.inputs

    if not (ds.owner and isinstance(ds.owner.op, DimShuffle)):
        return None

    ds_op = ds.owner.op

    if not ds_op.is_expand_dims:
        return None

    expanded_axes = ds_op.augment
    [x] = ds.owner.inputs

    idx_tuple = indices_from_subtensor(idx, node.op.idx_list)

    # Keep indexes for the original dimensions, and drop indexes for the expanded dimensions when safe
    new_idxs = []
    for i, idx_item in enumerate(idx_tuple):
        if i in expanded_axes:
            if isinstance(idx_item, slice):
                # Slice could be keeping or dropping this dimension
                if idx_item == slice(None):
                    # A None slice, always keeps the dimension.
                    # We skip the index, and later introduce the needed expand_dim
                    continue
                else:
                    # Other slices could keep or drop the dimension.
                    # Get out instead o trying to figure out which case it is
                    return None
            else:
                # Integer indexing can only drop the dimension (if it's a valid graph)
                # We can just drop the index and avoid expanding the dimension
                # This is why this rewrite is tagged with "shape_unsafe"
                continue
        else:
            # Keep indexes for non-expanded dimensions
            new_idxs.append(idx_item)

    [old_out] = node.outputs
    out = x[tuple(new_idxs)]
    copy_stack_trace(old_out, out)

    if out.type.broadcastable != old_out.type.broadcastable:
        # Re-introduce needed new dimensions (corresponding to full slices on the original expanded dimensions)
        # If out.type.broadcastable == (False) and old_out.type.broadcastable == (True, False, True)
        # then axis = (0, 2)
        old_bcast = list(old_out.type.broadcastable)
        expanded_bcast = list(out.type.broadcastable)
        axis = []
        i = 0
        while i < len(old_bcast):
            if i == len(expanded_bcast) or expanded_bcast[i] != old_bcast[i]:
                expanded_bcast.insert(i, True)
                axis.append(i)
            i += 1
        out = expand_dims(out, axis=axis)
        copy_stack_trace(old_out, out)

    return [out]


@register_canonicalize
@register_specialize
@node_rewriter([Subtensor])
def local_subtensor_of_squeeze(fgraph, node):
    """Lift subtensor through a squeeze operation"""
    x, *idxs_vars = node.inputs
    if not (
        x.owner is not None
        and isinstance(x.owner.op, DimShuffle)
        and x.owner.op.is_squeeze
    ):
        return None

    [x_before_squeeze] = x.owner.inputs
    idxs = indices_from_subtensor(idxs_vars, node.op.idx_list)
    dropped_dims = x.owner.op.drop

    # Apply indices directly on x
    # Add empty slices on the axis that squeeze would have removed
    new_idxs = list(idxs)
    for d in sorted(dropped_dims):
        new_idxs.insert(d, slice(None))
    new_idxs = np.array(new_idxs, dtype=object)

    x_indexed = x_before_squeeze[tuple(new_idxs)]

    # Reapply squeeze
    # Indexing may have squeezed some dimensions, so we need to recalculate dropped_dims
    new_dropped_dims = np.array(dropped_dims)
    for i, new_idx in reversed(tuple(enumerate(new_idxs))):
        if not isinstance(new_idx, slice):
            # If it's not a slice, it's an integer which drops the dimension
            new_dropped_dims[new_dropped_dims > i] -= 1
    new_x = x_indexed.squeeze(tuple(new_dropped_dims))

    copy_stack_trace(x, new_x)
    return [new_x]


@register_canonicalize
@register_specialize
@node_rewriter([Subtensor])
def local_subtensor_of_transpose(fgraph, node):
    """Lift a Subtensor through a DimShuffle that only transposes.

    transpose(x, (1, 0, 2))[i:, j:, k:] -> transpose(x[j:, i:, k:], (1, 0, 2))
    """
    ds, *idx = node.inputs

    if not (ds.owner and isinstance(ds.owner.op, DimShuffle)):
        return None

    ds_op = ds.owner.op
    if not ds_op.is_transpose:
        return None

    transposition = ds_op._transposition
    [x] = ds.owner.inputs

    idx_tuple = indices_from_subtensor(idx, node.op.idx_list)

    # Apply the transposition to the indexes
    ndim = x.type.ndim
    n_implicit_idxs = ndim - len(idx_tuple)
    idx_tuple = idx_tuple + (slice(None),) * n_implicit_idxs
    new_idxs = [idx_tuple[transposition.index(i)] for i in range(ndim)]
    new_x = x[tuple(new_idxs)]

    # Reintroduce any dims dropped by indexing so the original transpose still works
    dims_dropped_by_new_idx = _dims_dropped_by_basic_index(new_idxs)
    if dims_dropped_by_new_idx:
        new_x = expand_dims(new_x, axis=dims_dropped_by_new_idx)

    # Apply the transpose
    new_out = ds_op(new_x)

    # Squeeze dims again now that the transpose is done
    if dims_dropped_by_new_idx:
        dims_dropped_by_original_idx = _dims_dropped_by_basic_index(idx_tuple)
        new_out = squeeze(new_out, axis=dims_dropped_by_original_idx)

    # Cleanup consecutive expand_dims / transpose / squeeze (if any)
    if dims_dropped_by_new_idx:
        [new_out] = local_dimshuffle_lift.transform(fgraph, new_out.owner)

    return [new_out]


def lift_subtensor_through_alloc(fgraph, node):
    """``alloc(val, *shape)[idx]`` -> ``alloc(val[idx_on_kept_dims], *out_shape)``.

    Push the read past Alloc so the broadcast happens at most once and
    indexing operates on ``val`` (smaller) where possible. Covers basic
    ``Subtensor``, ``AdvancedSubtensor``, and ``AdvancedSubtensor1`` reads.

    On non-broadcast ``val`` dims an advanced index could expand ``val[idx]``
    past ``val.size``; only fire when the index is provably smaller or when
    the resulting Alloc is dropped.

    Bail on boolean masks and non-consecutive advanced indexing.
    """
    src = node.inputs[0]
    match src.owner_op_and_inputs:
        case (Alloc(), val, *alloc_dims):
            pass
        case _:
            return None
    n_added_dims = src.type.ndim - val.type.ndim

    indices = list(get_idx_list(node.inputs, node.op.idx_list))
    indices += [slice(None)] * (src.type.ndim - len(indices))

    if any(
        isinstance(idx, TensorVariable) and idx.type.dtype == "bool" for idx in indices
    ):
        return None
    # Non-consecutive advanced indices get hoisted to position 0 in the result
    # but stay in place inside ``val[val_indexer]``, misaligning the Alloc shape.
    if _non_consecutive_adv_indexing(indices):
        return None

    val_indexer: list = []
    dangerous_index_reaches_val = False
    for axis, idx in enumerate(indices):
        if axis < n_added_dims:
            # Axis was added by Alloc; index doesn't reach val.
            continue
        val_static_dim = val.type.shape[axis - n_added_dims]
        if val_static_dim == 1:
            # Broadcast val dim: slices stay (Alloc broadcasts on top);
            # advanced indices become length-1 zeros for squeeze.
            if isinstance(idx, slice):
                val_indexer.append(slice(None))
            else:
                val_indexer.append(np.zeros((1,) * idx.type.ndim, dtype=np.int64))
            continue
        val_indexer.append(idx)
        if not _index_provably_smaller(idx, val_static_dim):
            # Per-axis check; doesn't account for net effect across all axes.
            dangerous_index_reaches_val = True

    nw_val = _canonical_indexing(val, val_indexer)
    new_shape = indexed_result_shape(alloc_dims, indices)
    drops_alloc = nw_val.type.broadcastable == node.outputs[0].type.broadcastable

    if dangerous_index_reaches_val and not drops_alloc:
        return None

    if drops_alloc:
        result = nw_val
    else:
        result = alloc(nw_val, *new_shape)

    copy_stack_trace(node.outputs[0], result)
    return [result]


@register_infer_shape
@node_rewriter([Subtensor])
def local_basic_subtensor_of_alloc(fgraph, node):
    return lift_subtensor_through_alloc(fgraph, node)


@register_useless
@register_canonicalize
@register_specialize
@node_rewriter([Subtensor, AdvancedSubtensor, AdvancedSubtensor1])
def local_subtensor_of_alloc(fgraph, node):
    return lift_subtensor_through_alloc(fgraph, node)


def _diag_indices(ndim, a1, a2, d, row_off, col_off):
    """``[slice(None)] * ndim`` with axes ``(a1, a2)`` set to paired aranges
    sharing one ``arange(d)`` node so ``indexed_result_shape``'s same-node
    fast path skips ``broadcast_shape``.
    """
    ar = arange(d, dtype="int64")
    rows = ar + row_off if row_off else ar
    cols = ar + col_off if col_off else ar
    idxs: list = [slice(None)] * ndim
    idxs[a1] = rows
    idxs[a2] = cols
    return idxs


@node_rewriter([ExtractDiag])
def local_extract_diag_of_alloc_diag(fgraph, node):
    """Short-circuit ``extract_diag(alloc_diag(v, ..., k_alloc), offset)``.

    Diagonals at different offsets never cross:

    - ``offset == k_alloc``: full match ``->`` ``v``
    - ``offset != k_alloc``: no overlap ``->`` ``alloc(0, ..., d)`` along the
      read diagonal of the synthesized ``(L+|k_alloc|, L+|k_alloc|)`` matrix
      where ``d = L + |k_alloc| - |offset|``
    """
    inner = node.inputs[0]
    if not isinstance(inner.owner_op, AllocDiag):
        return None
    op = node.op
    diag_op = inner.owner.op
    if (op.axis1, op.axis2) != (diag_op.axis1, diag_op.axis2):
        return None  # cross-axis case is rare; bail
    [v] = inner.owner.inputs
    if op.offset == diag_op.offset:
        copy_stack_trace(node.outputs[0], v)
        return [v]
    v_shape = tuple(
        s if s is not None else v.shape[i] for i, s in enumerate(v.type.shape)
    )
    d = v_shape[-1] + abs(diag_op.offset) - abs(op.offset)
    out = alloc(np.asarray(0, dtype=v.dtype), *v_shape[:-1], d)
    copy_stack_trace(node.outputs[0], out)
    return [out]


@node_rewriter([ExtractDiag])
def local_extract_diag_of_eye(fgraph, node):
    """Short-circuit ``extract_diag(eye(n, m, k_eye), offset)``.

    The result is an ``alloc`` of ``1`` (matching offset) or ``0`` (mismatched
    offset) along the diagonal length ``min(n - row_off, m - col_off)``.
    """
    inner = node.inputs[0]
    if not isinstance(inner.owner_op, Eye):
        return None
    op = node.op
    n, m, k_inp = inner.owner.inputs
    if not isinstance(k_inp, TensorConstant):
        return None
    val = np.asarray(1 if op.offset == int(k_inp.data) else 0, dtype=inner.dtype)
    row_off, col_off = max(0, -op.offset), max(0, op.offset)
    a = n if row_off == 0 else n - row_off
    b = m if col_off == 0 else m - col_off
    d = a if a is b else minimum(a, b)
    out = alloc(val, d)
    copy_stack_trace(node.outputs[0], out)
    return [out]


@node_rewriter([ExtractDiag])
def local_extract_diag_lift(fgraph, node):
    """Lower ``ExtractDiag(X)`` to ``X[..., arange(d)+r, arange(d)+c, ...]``
    and commit only when the immediate next-step lift would consume it.

    Each branch builds the hypothetical ``AdvancedSubtensor`` off-fgraph and
    tests via the gating rewriter's ``.fn`` — no commit if the gate misses,
    so we never leave an unhelpful paired-arange gather behind.

    - ``Alloc`` parent — gated on ``local_subtensor_of_alloc``; rebuilds the
      outer Alloc shape with the diag length ``d`` so ``Shape(arange(d))[0]``
      doesn't survive.
    - ``Elemwise`` / ``Blockwise`` parent — gated on
      ``local_subtensor_of_batch_dims``. Blockwise core dims bail (the lift
      can't push past core ndim).
    - ``AdvancedIncSubtensor`` write-chain parent — gated on
      ``local_advanced_read_of_write_constant_indices``.
    """
    inner = node.inputs[0]
    op = node.op
    a1, a2 = op.axis1, op.axis2
    parent_op = inner.owner_op

    if isinstance(parent_op, Alloc):
        shape_inputs = inner.owner.inputs[1:]
    elif isinstance(parent_op, Elemwise | Blockwise):
        if isinstance(parent_op, Blockwise):
            batch_ndim = inner.owner.op.batch_ndim(inner.owner)
            if a1 >= batch_ndim or a2 >= batch_ndim:
                return None
        shape_inputs = inner.shape
    elif isinstance(parent_op, AdvancedIncSubtensor):
        shape_inputs = inner.shape
    else:
        return None

    row_off, col_off = max(0, -op.offset), max(0, op.offset)

    def _diag_dim(ax, off):
        s = inner.type.shape[ax]
        if s is not None:
            return s - off
        return shape_inputs[ax] - off if off else shape_inputs[ax]

    a_term, b_term = _diag_dim(a1, row_off), _diag_dim(a2, col_off)
    if isinstance(a_term, int) and isinstance(b_term, int):
        diag_len = min(a_term, b_term)
    else:
        diag_len = a_term if a_term is b_term else minimum(a_term, b_term)
    idxs = _diag_indices(inner.type.ndim, a1, a2, diag_len, row_off, col_off)
    hypothetical = inner[tuple(idxs)].owner

    if isinstance(parent_op, Alloc):
        return local_subtensor_of_alloc.fn(fgraph, hypothetical)
    elif isinstance(parent_op, Elemwise | Blockwise):
        return local_subtensor_of_batch_dims.fn(fgraph, hypothetical)
    else:
        # AdvancedIncSubtensor write chain
        return local_advanced_read_of_write_constant_indices.fn(fgraph, hypothetical)


extract_diag_lift_pass = SequentialGraphRewriter(
    out2in(
        local_extract_diag_of_alloc_diag,
        local_extract_diag_of_eye,
        local_extract_diag_lift,
        local_subtensor_of_alloc,
        local_subtensor_of_batch_dims,
        local_slice_read_of_write,
        local_advanced_read_of_write_constant_indices,
        name="extract_diag_lift_walker",
    ),
    out2in(
        local_adv_idx_to_diagonal,
        local_adv_idx_to_slice,
        local_useless_slice,
        name="extract_diag_cleanup",
    ),
)
extract_diag_lift_pass.__name__ = "extract_diag_lift_pass"  # type: ignore[attr-defined]
register_specialize(extract_diag_lift_pass, "shape_unsafe")  # type: ignore[arg-type]


@register_canonicalize
@node_rewriter([Subtensor])
def local_subtensor_SpecifyShape_lift(fgraph, node):
    """Lift ``specify_shape(x, s)[i_1, ..., i_n]`` to ``specify_shape(x[i1, ... , i_n], s[n:])``."""

    if not isinstance(node.op, Subtensor):
        return False

    specify_shape_node = node.inputs[0]

    if not (
        specify_shape_node.owner
        and isinstance(specify_shape_node.owner.op, SpecifyShape)
    ):
        return False

    obj_arg = specify_shape_node.owner.inputs[0]
    shape_arg = specify_shape_node.owner.inputs[1:]

    indices = get_idx_list(node.inputs, node.op.idx_list)

    if any(isinstance(index, slice) for index in indices):
        return False

    new_obj_arg = obj_arg[indices]
    # No need to specify shape for scalar outputs
    if new_obj_arg.ndim == 0:
        return [new_obj_arg]
    return [specify_shape(new_obj_arg, shape_arg[len(indices) :])]


@register_infer_shape
@register_specialize
@register_canonicalize("fast_compile")
@register_useless
@node_rewriter([Subtensor, AdvancedSubtensor1])
def local_subtensor_make_vector(fgraph, node):
    """Perform ``*Subtensor*`` operations on ``MakeVector`` outputs when the indices are constant.

    Replace all ``Subtensor`` and ``MakeVector`` cases like:
        [a,b,c][0] -> a
        [a,b,c][0:2] -> [a,b]

    Replace all ``AdvancedSubtensor1`` and ``MakeVector`` cases like:
        [a,b,c][[0,2]] -> [a,c]

    We can do this for constant indexes.

    .. note:

        This optimization implicitly relies on shape optimizations.

    TODO: This only applies to a single indexed dimension; we should have
    something more general for constant ``*Subtensor*`` graphs (or perhaps
    include this kind of work in the constant folding).
    """
    x = node.inputs[0]

    if not (x.owner and isinstance(x.owner.op, MakeVector)):
        return False

    make_vector_op = x.owner.op

    if isinstance(node.op, Subtensor):
        idxs = node.op.idx_list

        # Subtensor has no indexes, return make_vector
        if not idxs:
            return [x]

        (idx,) = idxs

        if isinstance(idx, int):
            idx = node.inputs[1]
    elif isinstance(node.op, AdvancedSubtensor1):
        idx = node.inputs[1]

    if isinstance(idx, Variable):
        if idx.ndim == 0:
            try:
                v = get_underlying_scalar_constant_value(
                    idx, only_process_constants=True
                )
                try:
                    ret = [x.owner.inputs[v]]
                except IndexError:
                    raise NotScalarConstantError("Bad user graph!")
                return ret
            except NotScalarConstantError:
                pass
        elif idx.ndim == 1 and isinstance(idx, Constant):
            values = list(map(int, list(idx.value)))
            ret = make_vector_op(*[x.owner.inputs[v] for v in values])
            copy_stack_trace(node.outputs[0], ret)
            return [ret]
    elif isinstance(idx, slice):
        # The index is a slice.  If it's a constant slice, we can perform the
        # index operation here.
        try:
            const_slice = get_constant_idx(
                node.op.idx_list, node.inputs, allow_partial=False
            )[0]
            sliced_inputs = x.owner.inputs[const_slice]
            if not sliced_inputs:
                return False
            if len(sliced_inputs) == 1:
                ret = expand_dims(sliced_inputs[0], axis=0)
            else:
                ret = make_vector_op(*sliced_inputs)
            copy_stack_trace(node.outputs, ret)
            return [ret]
        except NotScalarConstantError:
            pass


@register_canonicalize
@register_specialize
@node_rewriter([Subtensor])
def local_subtensor_of_join(fgraph, node):
    """Lift a Subtensor through a Join.

    join(axis=1, x, y)[0] -> join(axis=0, x[0], y[0])
    join(axis=1, x, y)[:, 0, -1] -> join(axis=1, x[:, :, -1], y[:, :, -1])[:, 0]

    """
    join_var, *idx = node.inputs

    if not (join_var.owner and isinstance(join_var.owner.op, Join)):
        return None

    if len(fgraph.clients[join_var]) > 1:
        # Join involves a full_copy, so we don't want to do it twice
        return None

    join_axis, *join_components = join_var.owner.inputs

    # Rewrite only works when the join axis is a constant along a non-indexed dimension
    if not isinstance(join_axis, Constant):
        return None

    [old_out] = node.outputs
    axis = normalize_axis_index(join_axis.data, join_components[0].type.ndim)
    idx_tuple = indices_from_subtensor(idx, node.op.idx_list)
    if _axis_is_indexed_by_basic_index(idx_tuple, axis):
        return _lift_subtensor_non_axis(
            local_subtensor_lift_rewrite=local_subtensor_of_join,
            fgraph=fgraph,
            variable=join_var,
            idx_tuple=idx_tuple,
            axis=axis,
            old_subtensor_variable=old_out,
        )

    # Lift index to the Join components
    indexed_components = [component[idx_tuple] for component in join_components]
    new_axis = axis - _ndim_dropped_left_of_axis_by_basic_index(idx_tuple, axis)
    out = join(new_axis, *indexed_components)

    return [out]


@register_specialize
@register_canonicalize
@node_rewriter([Subtensor])
def local_subtensor_shape_constant(fgraph, node):
    r"""Simplify constant `Subtensor`\s on `Shape`\s dimensions that are known.

    We want to convert graphs like

        Subtensor{int64} [id A] ''
         |Shape [id B] ''
         | |<TensorType(float64, row)> [id C]
         |ScalarConstant{0} [id D]

    into

        TensorConstant{1}

    TODO: Something like `local_shape_to_shape_i` should be a general
    canonicalization, and not a `ShapeFeature`-dependent rewrite.  If that were
    the case, we could change this to only operate on `Shape_i`\s.
    Currently, we're not handling them because they should only appear when
    `ShapeFeature` is present, and it will also simplify/remove them.

    """
    if not isinstance(node.op, Subtensor):
        return False

    shape = node.inputs[0]

    if not (shape.owner and isinstance(shape.owner.op, Shape)):
        return False

    shape_arg = shape.owner.inputs[0]

    (idx,) = get_idx_list(node.inputs, node.op.idx_list)

    try:
        idx_val = as_index_literal(idx)
    except NotScalarConstantError:
        return False

    if not isinstance(shape_arg.type, TensorType):
        return False

    try:
        shape_parts = shape_arg.type.broadcastable[idx_val]
    except IndexError:
        # An out-of-bounds index here is an error in the source graph
        # (e.g. ``scalar.shape[0]``), but it should fail at runtime rather
        # than abort the rewrite pass.
        return False

    if isinstance(shape_parts, Iterable):
        if all(shape_parts):
            return [as_tensor([1] * len(shape_parts), dtype=np.int64, ndim=1)]
    elif shape_parts:
        return [as_tensor(1, dtype=np.int64)]


@node_rewriter([Subtensor])
def local_subtensor_of_adv_subtensor(fgraph, node):
    """Lift a simple Subtensor through an AdvancedSubtensor, when basic index dimensions are to the left of any advanced ones.

    x[:, :, vec_idx][i, j] -> x[i, j][vec_idx]
    x[:, vec_idx][i, j, k] -> x[i][vec_idx][j, k]

    Restricted to a single advanced indexing dimension.

    An alternative approach could have fused the basic and advanced indices,
    so it is not clear this rewrite should be canonical or a specialization.
    Users must include it manually if it fits their use case.
    """
    adv_subtensor, *idxs = node.inputs

    if not (
        adv_subtensor.owner and isinstance(adv_subtensor.owner.op, AdvancedSubtensor)
    ):
        return None

    if len(fgraph.clients[adv_subtensor]) > 1:
        # AdvancedSubtensor involves a full_copy, so we don't want to do it twice
        return None

    x, *adv_index_vars = adv_subtensor.owner.inputs
    adv_idxs = indices_from_subtensor(adv_index_vars, adv_subtensor.owner.op.idx_list)

    # Advanced indexing is a minefield, avoid all cases except for consecutive integer indices
    if (
        not all(
            (
                (isinstance(adv_idx, TensorVariable) and adv_idx.type.dtype != "bool")
                or (isinstance(adv_idx, slice) and adv_idx == slice(None))
            )
            for adv_idx in adv_idxs
        )
    ) or _non_consecutive_adv_indexing(adv_idxs):
        return None

    for first_adv_idx_dim, adv_idx in enumerate(adv_idxs):
        # We already made sure there were only None slices besides integer indexes
        if isinstance(adv_idx, TensorVariable):
            break
    else:  # no-break
        # Not sure if this should ever happen, but better safe than sorry
        return None

    basic_idxs = indices_from_subtensor(idxs, node.op.idx_list)
    basic_idxs_lifted = basic_idxs[:first_adv_idx_dim]
    basic_idxs_kept = ((slice(None),) * len(basic_idxs_lifted)) + basic_idxs[
        first_adv_idx_dim:
    ]

    if all(basic_idx == slice(None) for basic_idx in basic_idxs_lifted):
        # All basic indices happen to the right of the advanced indices
        return None

    [basic_subtensor] = node.outputs
    dropped_dims = _dims_dropped_by_basic_index(basic_idxs_lifted)

    x_indexed = x[basic_idxs_lifted]
    copy_stack_trace([basic_subtensor, adv_subtensor], x_indexed)

    x_after_index_lift = expand_dims(x_indexed, dropped_dims)
    x_after_adv_idx = adv_subtensor.owner.op(x_after_index_lift, *adv_index_vars)
    copy_stack_trace([basic_subtensor, adv_subtensor], x_after_adv_idx)

    new_out = squeeze(x_after_adv_idx[basic_idxs_kept], dropped_dims)
    return [new_out]


# Rewrite will only be included if tagged by name
r = local_subtensor_of_adv_subtensor
optdb["canonicalize"].register(r.__name__, r, use_db_name_as_tag=False)
optdb["specialize"].register(r.__name__, r, use_db_name_as_tag=False)
del r
