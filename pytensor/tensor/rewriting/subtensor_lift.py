from collections.abc import Iterable, Sequence
from typing import cast

import numpy as np
from numpy.lib.array_utils import normalize_axis_index, normalize_axis_tuple

from pytensor import Variable
from pytensor.compile import optdb
from pytensor.graph import Constant, FunctionGraph, node_rewriter, vectorize_graph
from pytensor.graph.rewriting.basic import NodeRewriter, copy_stack_trace
from pytensor.scalar import basic as ps
from pytensor.tensor.basic import (
    Alloc,
    Join,
    MakeVector,
    alloc,
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
from pytensor.tensor.math import Dot, ceil_intdiv, dot
from pytensor.tensor.rewriting.basic import (
    register_canonicalize,
    register_specialize,
    register_stabilize,
)
from pytensor.tensor.rewriting.elemwise import local_dimshuffle_lift
from pytensor.tensor.rewriting.subtensor import is_full_slice, register_useless
from pytensor.tensor.shape import (
    Shape,
    SpecifyShape,
    specify_shape,
)
from pytensor.tensor.special import Softmax, softmax
from pytensor.tensor.subtensor import (
    AdvancedSubtensor,
    AdvancedSubtensor1,
    Subtensor,
    _non_consecutive_adv_indexing,
    as_index_literal,
    get_canonical_form_slice,
    get_constant_idx,
    get_idx_list,
    indices_from_subtensor,
)
from pytensor.tensor.type import TensorType
from pytensor.tensor.type_other import NoneTypeT, SliceType
from pytensor.tensor.variable import TensorVariable


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
    return any(ax < len(idxs) and not is_full_slice(idxs[ax]) for ax in axis)


def _lift_subtensor_non_axis(
    local_subtensor_lift_rewrite: NodeRewriter,
    fgraph: FunctionGraph,
    variable: TensorVariable,
    idx_tuple: tuple[int | slice],
    axis: int,
    old_subtensor_variable: TensorVariable,
) -> None | list[TensorVariable]:
    # Apply generic subtensor lift rewrite along "non-axis" dimensions
    real_indices = [idx for idx in idx_tuple if not is_full_slice(idx)]
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
@node_rewriter([Subtensor])
def local_subtensor_of_batch_dims(fgraph, node):
    """Lift a Subtensor through the batch dims of an (Elemwise or Blockwise) operation and its implicit broadcasting behavior.

    exp(x)[:, 0] -> exp(x[:, 0])
    add(x, y)[0] -> add(x[0], y[0])
    add(x[None], y)[2] -> add(x, y[2])
    """
    elem, *idx = node.inputs

    if not (elem.owner and isinstance(elem.owner.op, Elemwise | Blockwise)):
        return None

    if len(fgraph.clients[elem]) > 1:
        # Elemwise output is used beyond the Subtensor.
        # Get out to avoid repeated computations
        return None

    idx_tuple = indices_from_subtensor(idx, node.op.idx_list)

    batch_ndim = (
        elem.owner.op.batch_ndim(elem.owner)
        if isinstance(elem.owner.op, Blockwise)
        else elem.ndim
    )

    if len(idx_tuple) > batch_ndim:
        # Indexing on core dimensions of Blockwise. We split the indices and lift the batch ones only
        batch_indices, core_indices = idx_tuple[:batch_ndim], idx_tuple[batch_ndim:]
        if all(is_full_slice(idx) for idx in batch_indices):
            # No batch indices, nothing to do
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
            if is_full_slice(dim_idx):
                # Full slice can be safely applied to all inputs
                continue

            if all(dim_bcast_inp == elem_bcast for dim_bcast_inp in dim_bcast_inputs):
                # This dim is not broadcasted for any of the inputs, original index can be applied to all inputs
                continue

            # Some dims are broadcasted, so we need to adapt their indices
            # Slice indexing keeps the dimension, so we use a full slice for broadcasted inputs
            # Integer indexing drops the dimension, so we index by zero for the broadcsated inputs
            safe_bcast_dim_idx = slice(None) if isinstance(dim_idx, slice) else 0
            for inp_idx, dim_bcast_inp in zip(new_idxs, dim_bcast_inputs, strict=True):
                if dim_bcast_inp:
                    inp_idx[dim] = safe_bcast_dim_idx

        indexed_inputs = [
            inp[tuple(new_idx)]
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
    else:
        # Softmax currently only allows None or a single integer axis
        # Unlike CAReduce it does not normalize negative indices
        axis = normalize_axis_index(axis, sm.ndim)

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
                if is_full_slice(idx_item):
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

    transposition = ds_op.transposition
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


@register_infer_shape
@register_useless
@register_canonicalize
@register_specialize
@node_rewriter([Subtensor])
def local_subtensor_of_alloc(fgraph, node):
    """

    alloc(val)[x:y] -> alloc(val[...])
    alloc(val)[x:y] -> alloc(val)
    This can be seen as a lift, but it also reduce the number of computation/memory.

    """
    if not isinstance(node.op, Subtensor):
        return False
    u = node.inputs[0]
    if u.owner is None:
        return False
    if not isinstance(u.owner.op, Alloc):
        return False
    slices = get_idx_list(node.inputs, node.op.idx_list)
    val = u.owner.inputs[0]
    dims = u.owner.inputs[1:]
    assert len(slices) <= len(dims)

    # Number of dimensions added to val
    n_added_dims = u.ndim - val.ndim
    # Dimensions of the returned alloc
    nw_dims = []
    # Slices to take from val
    val_slices = []

    for i, (sl, dim) in enumerate(zip(slices, dims, strict=False)):
        # If val was not copied over that dim,
        # we need to take the appropriate subtensor on it.
        if i >= n_added_dims:
            # We check that the corresponding val dimensions was
            # not a broadcasted dimensions.
            if (
                val.type.ndim > (i - n_added_dims)
                and val.type.broadcastable[i - n_added_dims]
            ):
                val_slices.append(slice(None))
            else:
                val_slices.append(sl)

        csl, _ = get_canonical_form_slice(sl, dim)
        if type(csl) is not slice:
            # That dimension is removed.
            pass
        else:
            nw_dim = csl.stop - csl.start

            if csl.step != 1:
                # Do not add the ceil_intdiv() graphs in the graphs
                # when this is not needed as it prevent detecting the
                # correct broadcast pattern.
                nw_dim = ceil_intdiv(nw_dim, csl.step)
            nw_dims += [nw_dim]

    nw_val = val[tuple(val_slices)]
    nw_dims += dims[len(slices) :]
    if nw_val.ndim > len(nw_dims):
        return False
    rval = alloc(nw_val, *nw_dims)
    if not isinstance(rval, list | tuple):
        rval = [rval]
    return rval


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

    if any(
        isinstance(index, slice) or isinstance(getattr(index, "type", None), SliceType)
        for index in indices
    ):
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

        if isinstance(idx, ps.ScalarType | TensorType):
            old_idx, idx = idx, node.inputs[1]
            assert idx.type.is_super(old_idx)
    elif isinstance(node.op, AdvancedSubtensor1):
        idx = node.inputs[1]

    if isinstance(idx, int | np.integer):
        return [x.owner.inputs[idx]]
    elif isinstance(idx, Variable):
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

    assert idx_val != np.newaxis

    if not isinstance(shape_arg.type, TensorType):
        return False

    shape_parts = shape_arg.type.broadcastable[idx_val]

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

    x, *adv_idxs = adv_subtensor.owner.inputs

    # Advanced indexing is a minefield, avoid all cases except for consecutive integer indices
    if any(
        (
            isinstance(adv_idx.type, NoneTypeT)
            or (isinstance(adv_idx.type, TensorType) and adv_idx.type.dtype == "bool")
            or (isinstance(adv_idx.type, SliceType) and not is_full_slice(adv_idx))
        )
        for adv_idx in adv_idxs
    ) or _non_consecutive_adv_indexing(adv_idxs):
        return None

    for first_adv_idx_dim, adv_idx in enumerate(adv_idxs):
        # We already made sure there were only None slices besides integer indexes
        if isinstance(adv_idx.type, TensorType):
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
    x_after_adv_idx = adv_subtensor.owner.op(x_after_index_lift, *adv_idxs)
    copy_stack_trace([basic_subtensor, adv_subtensor], x_after_adv_idx)

    new_out = squeeze(x_after_adv_idx[basic_idxs_kept], dropped_dims)
    return [new_out]


# Rewrite will only be included if tagged by name
r = local_subtensor_of_adv_subtensor
optdb["canonicalize"].register(r.__name__, r, use_db_name_as_tag=False)
optdb["specialize"].register(r.__name__, r, use_db_name_as_tag=False)
del r
