from collections.abc import Iterable, Sequence

import numpy as np

from pytensor import Variable
from pytensor.graph import Constant, node_rewriter
from pytensor.graph.rewriting.basic import copy_stack_trace
from pytensor.scalar import basic as ps
from pytensor.tensor.basic import (
    Alloc,
    MakeVector,
    alloc,
    as_tensor,
    expand_dims,
    get_underlying_scalar_constant_value,
    register_infer_shape,
)
from pytensor.tensor.elemwise import DimShuffle, Elemwise
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
from pytensor.tensor.subtensor import (
    AdvancedSubtensor1,
    Subtensor,
    as_index_literal,
    get_canonical_form_slice,
    get_constant_idx,
    get_idx_list,
    indices_from_subtensor,
)
from pytensor.tensor.type import TensorType
from pytensor.tensor.type_other import SliceType


def _dims_dropped_by_basic_index(idxs: Sequence[slice | int]) -> tuple[int, ...]:
    # Inputs can be slice or integer indexes
    # Slices keep the dimensions, integers collapse them
    return tuple(i for i, idx in enumerate(idxs) if not isinstance(idx, slice))


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
    if not isinstance(node.op, Subtensor):
        return
    if not (node.inputs[0].owner and isinstance(node.inputs[0].owner.op, Dot)):
        return
    # If there is other node that use the outputs of the dot
    # We don't want to compute twice the sub part.
    if len(fgraph.clients[node.inputs[0]]) > 1:
        return

    a = node.inputs[0].owner.inputs[0]
    b = node.inputs[0].owner.inputs[1]

    idx_list = get_idx_list(node.inputs, node.op.idx_list)

    num_a_indices = min(a.ndim - 1, len(idx_list))
    a_indices = idx_list[:num_a_indices]
    b_indices = idx_list[num_a_indices:]

    # This is necessary because np.dot sums the last index of a with the second to last of b
    # so we want to skip the second-to-last index into b.
    # This wasn't necessary for a, because we just omitted the last index.
    # We skip this if b.ndim = 1, since then we just want b_sub = b, not b_sub = b[:]
    # (dot also handles b.ndim < 2 as a special case)
    if b.ndim > 1 and len(b_indices) >= b.ndim - 1:
        b_indices = (
            b_indices[: b.ndim - 2]
            + (slice(None, None, None),)
            + b_indices[b.ndim - 2 :]
        )

    a_sub = a.__getitem__(tuple(a_indices))
    b_sub = b.__getitem__(tuple(b_indices)) if b_indices else b

    # Copy over previous output stacktrace to a_sub and b_sub,
    # because an error in the subtensor operation (e.g. an index error)
    # on either a or b must correspond to an error in the
    # subtensor operation on their dot product.
    copy_stack_trace(node.outputs[0], [a_sub, b_sub])

    # Copy over previous output stacktrace and previous dot product stacktrace,
    # because an error here may correspond to an either in either the original
    # dot product, or in the dot product after the subtensor operation.
    r = dot(a_sub, b_sub)
    copy_stack_trace([node.outputs[0], node.inputs[0]], r)

    return [r]


@register_canonicalize("shape_unsafe")
@register_specialize("shape_unsafe")
@node_rewriter([Subtensor])
def local_subtensor_of_elemwise(fgraph, node):
    """Lift a Subtensor through an Elemwise and its implicit broadcasting behavior.

    exp(x)[:, 0] -> exp(x[:, 0])
    add(x, y)[0] -> add(x[0], y[0])
    add(x[None], y)[2] -> add(x, y[2])
    """
    elem, *idx = node.inputs

    if not (elem.owner and isinstance(elem.owner.op, Elemwise)):
        return None

    if len(fgraph.clients[elem]) > 1:
        # Elemwise output is used beyond the Subtensor.
        # Get out to avoid repeated computations
        return None

    idx_tuple = indices_from_subtensor(idx, node.op.idx_list)

    elem_inputs = elem.owner.inputs
    elem_bcast = elem.type.broadcastable
    if all(inp.type.broadcastable == elem_bcast for inp in elem_inputs):
        # No need to worry about implicit broadcasting.
        indexed_inputs = [inp[idx_tuple] for inp in elem_inputs]

    else:
        # The original indices may not make sense on some of the broadcasted dimensions
        new_idxs = [list(idx_tuple) for _ in elem_inputs]
        for dim, (dim_idx, dim_bcast_out, *dim_bcast_inputs) in enumerate(
            zip(
                idx_tuple,
                elem_bcast,
                *(inp.type.broadcastable for inp in elem_inputs),
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

    if not isinstance(node.op, Subtensor | AdvancedSubtensor1):
        return False

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
            ret = make_vector_op(*x.owner.inputs[const_slice])
            copy_stack_trace(node.outputs, ret)
            return [ret]
        except NotScalarConstantError:
            pass


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
