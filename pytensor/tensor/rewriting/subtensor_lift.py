from collections.abc import Iterable

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
from pytensor.tensor.math import Dot, ceil_intdiv, dot
from pytensor.tensor.rewriting.basic import (
    register_canonicalize,
    register_specialize,
    register_stabilize,
)
from pytensor.tensor.rewriting.subtensor import is_full_slice, register_useless
from pytensor.tensor.shape import (
    Shape,
    SpecifyShape,
    Unbroadcast,
    specify_shape,
    unbroadcast,
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


# fast_compile to allow opt subtensor(cast{float32}(make_vector))
@register_canonicalize("fast_compile")
@node_rewriter([Subtensor])
def local_subtensor_lift(fgraph, node):
    """
    unary(x)[idx] -> unary(x[idx])#any broadcast pattern.

    Handles the following unary ops:
    elemwise(x,...)[idx] -> elemwise(x[idx],...)
      when x,... are broadcasted scalar or not broadcasted at all
    Unbroadcast(x)[idx] => Unbroadcast(x[idx])

    """
    if isinstance(node.op, Subtensor):
        u = node.inputs[0]
        if u.owner is None or len(fgraph.clients[u]) > 1:
            return False

        if isinstance(u.owner.op, Elemwise) and len(u.owner.inputs) == 1:
            idx = node.inputs[1:]
            x_idx = node.op(u.owner.inputs[0], *idx)
            # Copy over previous output stacktrace
            copy_stack_trace(node.outputs, x_idx)
            ret = u.owner.op(x_idx)
            # Copy over previous output stacktrace
            # and stacktrace from previous unary operation
            copy_stack_trace([node.outputs[0], node.inputs[0]], ret)
            return [ret]

        if isinstance(u.owner.op, Elemwise):
            new_inputs = []
            if all(sum(i.type.broadcastable) == 0 for i in u.owner.inputs):
                # There is no broadcastable in the inputs
                idx = node.inputs[1:]
                new_inputs = [node.op(i, *idx) for i in u.owner.inputs]
                # Copy over previous output stacktrace
                copy_stack_trace(node.outputs[0], new_inputs)

                ret = u.owner.op(*new_inputs)
                # Copy over previous output stacktrace
                # and stacktrace from previous unary operation
                copy_stack_trace([node.outputs[0], node.inputs[0]], ret)
                return [ret]
            elif all(sum(i.type.broadcastable) in [i.ndim, 0] for i in u.owner.inputs):
                # There is no broadcastable in the inputs or it is scalar
                idx = node.inputs[1:]
                new_inputs = []
                for i in u.owner.inputs:
                    if sum(i.type.broadcastable) == 0:
                        new_inputs.append(node.op(i, *idx))
                    else:
                        # If the subtensor remove some dims, we must
                        # lower the number of dimensions of this scalar.
                        if node.outputs[0].ndim == i.ndim:
                            new_inputs.append(i)
                        else:
                            new_inputs.append(
                                i.dimshuffle(["x"] * node.outputs[0].ndim)
                            )

                # Copy over previous output stacktrace
                copy_stack_trace(node.outputs[0], new_inputs)

                ret = u.owner.op(*new_inputs)
                # Copy over previous output stacktrace
                # and stacktrace from previous unary operation
                copy_stack_trace([node.outputs[0], node.inputs[0]], ret)
                return [ret]

        if isinstance(u.owner.op, Unbroadcast):
            # Subtensor might reduce dim., adapt broadcast pattern accordingly
            old_axes = u.owner.op.axes
            new_axes = []

            # loop through indices being subtensor-ed
            # i indexes broadcastable pattern before subtensor
            # j indexes broadcastable pattern after subtensor
            j = 0
            for i, x in enumerate(node.op.idx_list):
                # if it is not a slice, it will reduce the dimension, should
                # not appear in the broascastable dimensions
                if isinstance(x, slice):
                    if i in old_axes:
                        new_axes.append(j)
                    j += 1
            # now keep the broadcastable pattern of all
            # items not appearing in subtensor list
            for i in range(len(node.op.idx_list), len(u.broadcastable)):
                if i in old_axes:
                    new_axes.append(j)
                j += 1

            subt_x = node.op(u.owner.inputs[0], *node.inputs[1:])
            # Copy over previous output stacktrace
            copy_stack_trace(node.outputs[0], subt_x)

            rbcast_subt_x = unbroadcast(subt_x, *new_axes)
            # Copy over previous output stacktrace
            # and stacktrace from previous unary operation
            copy_stack_trace([node.outputs[0], node.inputs[0]], rbcast_subt_x)

            return [rbcast_subt_x]


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
