import itertools
import sys
import warnings

import numpy as np

import pytensor
from pytensor import compile
from pytensor.compile import optdb
from pytensor.graph.basic import Constant, Variable
from pytensor.graph.rewriting.basic import (
    WalkingGraphRewriter,
    copy_stack_trace,
    dfs_rewriter,
    in2out,
    node_rewriter,
)
from pytensor.raise_op import Assert
from pytensor.scalar import Add, ScalarConstant
from pytensor.scalar import constant as scalar_constant
from pytensor.tensor.basic import (
    Alloc,
    ExtractDiag,
    Join,
    ScalarFromTensor,
    TensorFromScalar,
    alloc,
    arange,
    cast,
    concatenate,
    expand_dims,
    get_scalar_constant_value,
    get_underlying_scalar_constant_value,
    moveaxis,
    register_infer_shape,
    switch,
)
from pytensor.tensor.basic import constant as tensor_constant
from pytensor.tensor.blockwise import _squeeze_left
from pytensor.tensor.elemwise import Elemwise
from pytensor.tensor.exceptions import NotScalarConstantError
from pytensor.tensor.extra_ops import broadcast_to
from pytensor.tensor.math import (
    add,
    and_,
    eq,
    ge,
    gt,
    le,
    lt,
    maximum,
    minimum,
    or_,
    variadic_add,
)
from pytensor.tensor.rewriting.basic import (
    register_canonicalize,
    register_specialize,
    register_stabilize,
)
from pytensor.tensor.rewriting.blockwise import blockwise_of
from pytensor.tensor.shape import (
    shape_padleft,
    shape_padright,
    shape_tuple,
)
from pytensor.tensor.sharedvar import TensorSharedVariable
from pytensor.tensor.subtensor import (
    AdvancedIncSubtensor,
    AdvancedIncSubtensor1,
    AdvancedSubtensor,
    AdvancedSubtensor1,
    IncSubtensor,
    Subtensor,
    _non_consecutive_adv_indexing,
    advanced_inc_subtensor1,
    advanced_subtensor1,
    as_index_constant,
    basic_subtensor,
    flatten_index_variables,
    get_canonical_form_slice,
    get_constant_idx,
    get_idx_list,
    get_slice_elements,
    inc_subtensor,
    indices_from_subtensor,
)
from pytensor.tensor.type import TensorType
from pytensor.tensor.variable import TensorConstant, TensorVariable


def register_useless(lopt, *tags, **kwargs):
    if isinstance(lopt, str):

        def register(inner_lopt):
            return register_useless(inner_lopt, lopt, *tags, **kwargs)

        return register
    else:
        name = kwargs.pop("name", None) or lopt.__name__

        compile.mode.local_useless.register(
            name, lopt, "fast_run", *tags, position="last", **kwargs
        )
        return lopt


def transform_take(a, indices, axis):
    r"""Transform ``arr[:,:,:,indices,...]``-like operations into single-dimensional, vector index operations.

    This effectively converts certain `AdvancedSubtensor` `Op`\s into a
    combination of `AdvancedSubtensor1`, `Dimshuffle`, and `Reshape` `Op`\s,
    which can be more efficient.

    Parameters
    ----------
    a : TensorVariable
        The source array.
    indices : TensorVariable, ndarray, list, tuple
        The indices of the values to extract.
    axis : int
        The axis over which to select values. By default, the flattened
        input array is used.

    """
    a = pytensor.tensor.as_tensor_variable(a)
    indices = pytensor.tensor.as_tensor_variable(indices)
    # We can use the more efficient `AdvancedSubtensor1` if `indices` is a vector
    if indices.ndim == 1:
        if axis == 0:
            return advanced_subtensor1(a, indices)
        else:
            shuffle = list(range(a.ndim))
            shuffle[0] = axis
            shuffle[axis] = 0
            res = advanced_subtensor1(a.dimshuffle(shuffle), indices).dimshuffle(
                shuffle
            )
            return res

    # We can reshape and flatten the indices in order to use an
    # `AdvancedSubtensor1` `Op` per the above
    indices_shape = shape_tuple(indices)
    a_shape = shape_tuple(a)

    shape_parts = [
        a_shape[:axis],
        indices_shape,
        a_shape[axis + 1 :],
    ]

    shape_parts = [sp for sp in shape_parts if len(sp) > 0]

    assert len(shape_parts) > 0

    if len(shape_parts) > 1:
        shape = pytensor.tensor.concatenate(shape_parts)
    elif len(shape_parts) == 1:
        shape = shape_parts[0]
    else:
        shape = ()

    ndim = a.ndim + indices.ndim - 1

    return transform_take(a, indices.flatten(), axis).reshape(shape, ndim=ndim)


def is_full_slice(x):
    warnings.warn(
        "The function is deprecated, use x==slice(None) instead.",
        DeprecationWarning,
    )
    return x == slice(None)


def get_advsubtensor_axis(indices):
    """Determine the axis at which an array index is applied.

    This only works for ``take``-like indices: e.g. ``x[:, :, idx, ...]``.  For
    the above example, `get_advsubtensor_axis` would return ``2``.  If it
    encounters anything other than a set of `indices` containing full slices
    and an array/tensor index, it will return ``None``.

    """
    found_idx = False
    axis = 0
    for idx in indices:
        if not found_idx and idx == slice(None):
            # Preceding full slices
            axis += 1
        elif found_idx and not idx == slice(None):
            # We don't handle multiple indices
            return
        elif found_idx and idx == slice(None):
            # Trailing full slices
            continue
        else:
            found_idx = True

    if isinstance(
        indices[axis], TensorConstant | TensorVariable | TensorSharedVariable
    ):
        return axis


@register_specialize
@node_rewriter([AdvancedSubtensor])
def local_replace_AdvancedSubtensor(fgraph, node):
    r"""
    This rewrite converts expressions like ``X[..., y]`` into ``X.T[y].T``, for
    a vector ``y``, and ``X[z, ...]`` into ``X[z.flatten()].reshape(...)``, for a
    matrix ``z``.

    These rewrites replace `AdvancedSubtensor`\s with the more efficient
    `AdvancedSubtensor1` and `Subtensor` `Op`\s.
    """

    if not isinstance(node.op, AdvancedSubtensor):
        return

    indexed_var, *index_variables = node.inputs
    indices = indices_from_subtensor(index_variables, node.op.idx_list)
    axis = get_advsubtensor_axis(indices)

    if axis is None or indices[axis].dtype == "bool":
        # Booleans aren't handled
        return

    new_res = transform_take(indexed_var, indices[axis], axis)
    copy_stack_trace(node.outputs[0], new_res)
    return [new_res]


@register_specialize
@node_rewriter([AdvancedIncSubtensor])
def local_AdvancedIncSubtensor_to_AdvancedIncSubtensor1(fgraph, node):
    r"""Replace `AdvancedIncSubtensor`\s with `AdvancedIncSubtensor1`\s.

    This is only done when there's a single vector index.
    """

    if node.op.ignore_duplicates:
        # `AdvancedIncSubtensor1` does not ignore duplicate index values
        return

    res, val, *index_variables = node.inputs
    indices = indices_from_subtensor(index_variables, node.op.idx_list)

    axis = get_advsubtensor_axis(indices)

    if axis is None or indices[axis].dtype == "bool":
        # Booleans aren't currently handled by `AdvancedIncSubtensor1`
        return

    new_subtensor = transform_take(res, indices[axis], axis)

    new_res = inc_subtensor(
        new_subtensor,
        val,
        inplace=node.op.inplace,
        set_instead_of_inc=node.op.set_instead_of_inc,
        ignore_duplicates=False,
    )
    copy_stack_trace(node.outputs[0], new_res)
    return [new_res]


@register_infer_shape
@register_useless
@register_canonicalize
@register_specialize
@register_stabilize
@node_rewriter([Subtensor, IncSubtensor, AdvancedSubtensor, AdvancedIncSubtensor])
def local_useless_slice(fgraph, node):
    """Remove useless slices and canonicalize redundant slice bounds to ``None``.

    Applies to all Subtensor Ops with slices (basic and advanced, get and set).

    - ``X[0, :]`` → ``X[0]`` (trailing full slices dropped)
    - ``X[:]`` → ``X``
    - ``X[0:7:1]`` → ``X[:]`` when ``X.shape[0] <= 7``
    - ``X[-1:-8:-1]`` → ``X[::-1]`` when ``X.shape[0] <= 7``
    """
    op = node.op
    idx_list = op.idx_list
    if not idx_list:
        if isinstance(op, Subtensor | AdvancedSubtensor):
            return [node.inputs[0]]
        else:
            # We let local_useless_inc_subtensor handle these
            return None

    if is_inc_subtensor := isinstance(op, IncSubtensor | AdvancedIncSubtensor):
        x, y, *idx_vars = node.inputs
    else:
        x, *idx_vars = node.inputs

    new_idxs = list(indices_from_subtensor(idx_vars, idx_list))
    change_flag = False
    last_useful_idx = -1
    for dim, s in enumerate(new_idxs):
        if not isinstance(s, slice):
            last_useful_idx = dim
            continue

        if s == slice(None):
            continue

        step = s.step

        if step is None:
            positive_step = True
        elif isinstance(step, Constant):
            step_value = step.data
            positive_step = step.data > 0
            if step_value == 1:
                change_flag = True
                step = None
        else:
            # We can only canonicalize start and stop if we know the sign of step
            last_useful_idx = dim
            continue

        start = s.start
        stop = s.stop

        dim_length = x.type.shape[dim] if dim < x.type.ndim else None
        if start is not None and isinstance(start, Constant):
            start_val = start.data
            if positive_step:
                if (
                    start_val == 0
                    # Negative start that wraps to or before index 0
                    or (dim_length is not None and -start_val >= dim_length)
                ):
                    change_flag = True
                    start = None
            else:
                if (
                    start_val == -1
                    # Positive start at or beyond the last index
                    or (dim_length is not None and start_val >= dim_length - 1)
                ):
                    change_flag = True
                    start = None

        if dim_length is not None and stop is not None and isinstance(stop, Constant):
            stop_val = stop.data
            if positive_step:
                # Positive stop at or beyond the length
                if stop_val >= dim_length:
                    change_flag = True
                    stop = None
            else:
                # Negative stop that wraps to or before index 0
                if -stop_val > dim_length:
                    change_flag = True
                    stop = None

        if start is not None or stop is not None or step is not None:
            last_useful_idx = dim

        new_idxs[dim] = slice(start, stop, step)

    if change_flag or (last_useful_idx + 1) < len(idx_list):
        new_idxs = new_idxs[: last_useful_idx + 1]
        new_idx_list, new_flat_vars = flatten_index_variables(new_idxs)
        props = op._props_dict() | {"idx_list": new_idx_list}
        if is_inc_subtensor:
            new_node = type(op)(**props)(x, y, *new_flat_vars).owner
            if not new_idx_list:
                ret = local_useless_inc_subtensor.fn(fgraph, new_node)
                if ret:
                    copy_stack_trace(node.outputs, ret)
                    return ret
            out = new_node.outputs[0]
        else:
            out = type(op)(**props)(x, *new_flat_vars) if new_idx_list else x
        copy_stack_trace(node.outputs, out)
        return [out]


@register_canonicalize
@register_specialize
@node_rewriter([Subtensor])
def local_subtensor_merge(fgraph, node):
    """
    Refactored optimization to deal with all cases of tensor merging.
    Given a subgraph of the form Subtensor(Subtensor(u)), the optimization
    expresses all slices in a canonical form, and then merges them together.

    """
    from pytensor.scan.op import Scan

    u = node.inputs[0]
    if not (u.owner is not None and isinstance(u.owner.op, Subtensor)):
        return None

    # We can merge :)
    # x actual tensor on which we are picking slices
    x = u.owner.inputs[0]
    # slices of the first applied subtensor
    slices1 = get_idx_list(u.owner.inputs, u.owner.op.idx_list)
    slices2 = get_idx_list(node.inputs, node.op.idx_list)

    # Don't try to do the optimization on do-while scan outputs,
    # as it will create a dependency on the shape of the outputs
    if (
        x.owner is not None
        and isinstance(x.owner.op, Scan)
        and x.owner.op.info.as_while
    ):
        return None

    # Get the shapes of the vectors !
    try:
        # try not to introduce new shape into the graph
        xshape = fgraph.shape_feature.shape_of[x]
        ushape = fgraph.shape_feature.shape_of[u]
    except AttributeError:
        # Following the suggested use of shape_feature which should
        # consider the case when the compilation mode doesn't
        # include the ShapeFeature
        xshape = x.shape
        ushape = u.shape

    merged_slices = []
    pos_2 = 0
    pos_1 = 0
    while (pos_1 < len(slices1)) and (pos_2 < len(slices2)):
        slice1 = slices1[pos_1]
        if isinstance(slice1, slice):
            merged_slices.append(
                merge_two_slices(
                    fgraph, slice1, xshape[pos_1], slices2[pos_2], ushape[pos_2]
                )
            )
            pos_2 += 1
        else:
            merged_slices.append(slice1)
        pos_1 += 1

    if pos_2 < len(slices2):
        merged_slices += slices2[pos_2:]
    else:
        merged_slices += slices1[pos_1:]

    merged_slices = tuple(as_index_constant(s) for s in merged_slices)
    out = basic_subtensor(x, *merged_slices)

    # Copy over previous output stacktrace
    # and stacktrace from previous slicing operation.
    # Why? Because, the merged slicing operation could have failed
    # because of either of the two original slicing operations
    orig_out = node.outputs[0]
    copy_stack_trace([orig_out, node.inputs[0]], out)
    return [out]


@register_specialize
@register_canonicalize
@node_rewriter([Subtensor])
def local_subtensor_remove_broadcastable_index(fgraph, node):
    """
    Remove broadcastable dimension with index 0 or -1
    a[:,:,:,0] -> a.dimshuffle(0,1,2), when
        a.broadcastable = (False, False, False, True)
    a[0,:,-1,:] -> a.dimshuffle(1,3), when
        a.broadcastable = (True, False, True, False)

    """
    if isinstance(node.op, Subtensor):
        idx = node.op.idx_list
    else:
        return

    remove_dim = []
    node_inputs_idx = 1
    for dim, elem in enumerate(idx):
        if isinstance(elem, int):
            # The idx is a integer position.
            dim_index = node.inputs[node_inputs_idx]
            if isinstance(dim_index, ScalarConstant):
                dim_index = dim_index.value
            if dim_index in (0, -1) and node.inputs[0].broadcastable[dim]:
                remove_dim.append(dim)
                node_inputs_idx += 1
            else:
                return
        elif isinstance(elem, slice):
            if elem != slice(None):
                return
        else:
            raise TypeError("case not expected")

    if len(remove_dim) == 0:
        return
    else:
        all_dim = range(node.inputs[0].ndim)
        remain_dim = [x for x in all_dim if x not in remove_dim]
        return [node.inputs[0].dimshuffle(tuple(remain_dim))]


@register_canonicalize
@register_specialize
@node_rewriter([IncSubtensor, AdvancedIncSubtensor])
def local_useless_inc_subtensor(fgraph, node):
    r"""Remove redundant `IncSubtensor`\s.

    - ``x[full_slices].set(y)`` → ``y``  (broadcast/cast to x's shape)
    - ``zeros[full_slices].inc(y)`` → ``y``  (broadcast/cast to x's shape)
    - ``x[full_slices].inc(y)`` → ``x + y``
    """

    x, y, *index_vars = node.inputs

    indices = indices_from_subtensor(index_vars, node.op.idx_list)

    # Check that all indices are full slices or full reversals
    if not all(
        isinstance(e, slice)
        and e.start is None
        and e.stop is None
        and (
            e.step is None
            or get_scalar_constant_value(
                e.step, only_process_constants=True, raise_not_constant=False
            )
            == -1
        )
        for e in indices
    ):
        return

    is_inc = not node.op.set_instead_of_inc
    x_is_zero = False
    if is_inc:
        try:
            x_is_zero = get_underlying_scalar_constant_value(x) == 0
        except NotScalarConstantError:
            pass

    # IncSubtensor casts y to x's dtype and broadcasts y onto x's shape
    out_dtype = node.outputs[0].type.dtype

    static_same = x.type.shape == y.type.shape and all(
        s is not None for s in x.type.shape
    )
    if not static_same:
        if hasattr(fgraph, "shape_feature") and fgraph.shape_feature.same_shape(x, y):
            static_same = True

    if y.type.dtype != out_dtype:
        y = cast(y, out_dtype)

    if not static_same:
        y = alloc(y, *x.shape)
        copy_stack_trace(node.outputs[0], y)

    if not all(e.step is None for e in node.op.idx_list):
        y = Subtensor(node.op.idx_list)(y, *index_vars)

    if not is_inc or x_is_zero:
        return [y]

    r = add(x, y)
    copy_stack_trace(node.outputs[0], r)
    return [r]


@register_canonicalize
@register_specialize
@node_rewriter([AdvancedIncSubtensor1])
def local_set_to_inc_subtensor(fgraph, node):
    r"""
    AdvancedIncSubtensor1(x, x[ilist]+other, ilist, set_instead_of_inc=True) ->
    AdvancedIncSubtensor1(x, other, ilist, set_instead_of_inc=False)

    TODO FIXME: Why doesn't this apply to all `*IncSubtensor*` `Op`\s?  If it
    did this wouldn't need to also be included in the "specialize" pass.

    """
    if (
        isinstance(node.op, AdvancedIncSubtensor1)
        and node.op.set_instead_of_inc
        and node.inputs[1].owner
        and isinstance(node.inputs[1].owner.op, Elemwise)
        and isinstance(node.inputs[1].owner.op.scalar_op, Add)
    ):
        addn = node.inputs[1].owner
        subn = None
        other = None

        if addn.inputs[0].owner and isinstance(
            addn.inputs[0].owner.op, AdvancedSubtensor1
        ):
            subn = addn.inputs[0].owner
            other = addn.inputs[1]
        elif addn.inputs[1].owner and isinstance(
            addn.inputs[1].owner.op, AdvancedSubtensor1
        ):
            subn = addn.inputs[1].owner
            other = addn.inputs[0]
        else:
            return
        if subn.inputs[1] != node.inputs[2] or subn.inputs[0] != node.inputs[0]:
            return
        ret = advanced_inc_subtensor1(node.inputs[0], other, node.inputs[2])

        copy_stack_trace(node.outputs, ret)

        return [ret]


@register_canonicalize
@register_stabilize
@register_specialize
@node_rewriter([add])
def local_add_of_sparse_write(fgraph, node):
    """Absorb a sparse write into a surrounding add: ``x + zeros[idx].set(v) -> x[idx].inc(v)``.

    A set into a zero-filled base is just the dense form of a sparse update.
    Adding it to another tensor is equivalent to incrementing in place, which
    avoids materialising the dense sparse representation.

    Also handles ``zeros[idx].inc(v)`` when ``idx`` is duplicate-free, since
    with unique indices inc is semantically equivalent to set.
    """
    for i, sparse_candidate in enumerate(node.inputs):
        if not (
            sparse_candidate.owner
            and isinstance(
                sparse_candidate.owner.op,
                IncSubtensor | AdvancedIncSubtensor1 | AdvancedIncSubtensor,
            )
        ):
            continue

        inner_op = sparse_candidate.owner.op
        base, v, *idx_vars = sparse_candidate.owner.inputs

        if (
            get_underlying_scalar_constant_value(
                base, elemwise=False, raise_not_constant=False
            )
            != 0
        ):
            continue

        # An inc into zeros is only equivalent to a set when indices are
        # duplicate-free. Basic (slice/scalar) indexing is always unique;
        # advanced integer-array indices must be checked.
        if not inner_op.set_instead_of_inc and not isinstance(inner_op, IncSubtensor):
            if not all(_constant_has_unique_indices(idx) for idx in idx_vars):
                continue

        others = [node.inputs[j] for j in range(len(node.inputs)) if j != i]
        other = variadic_add(*others)

        if inner_op.set_instead_of_inc:
            new_op = type(inner_op)(
                **(inner_op._props_dict() | {"set_instead_of_inc": False})
            )
        else:
            new_op = inner_op
        r = new_op(other, v, *idx_vars)
        copy_stack_trace(node.outputs[0], r)
        return [r]

    return None


@register_canonicalize
@register_specialize
@node_rewriter([Subtensor])
def local_useless_subtensor(fgraph, node):
    """Remove `Subtensor` if it takes the full input."""

    if not node.op.idx_list:
        return [node.inputs[0]]

    # The more elaborate optimization needs ShapeOpt and fgraph.shape_feature
    if not hasattr(fgraph, "shape_feature"):
        return

    shape_of = fgraph.shape_feature.shape_of

    cdata = get_constant_idx(
        node.op.idx_list,
        node.inputs,
        allow_partial=True,
        only_process_constants=True,
    )
    for pos, idx in enumerate(cdata):
        if not isinstance(idx, slice):
            # If idx is not a slice, this means we remove this dimension
            # from the output, so the subtensor is not useless
            return False
        if idx.start is not None and idx.start != 0:
            # If the start of the slice is different from 0, or is a
            # variable, then we assume the subtensor is not useless
            return False
        if idx.step is not None and idx.step != 1:
            # If we are going backwards, or skipping elements, then this
            # is not a useless subtensor
            return False

        length_pos = shape_of[node.inputs[0]][pos]

        if isinstance(idx.stop, int | np.integer):
            length_pos_data = sys.maxsize
            try:
                length_pos_data = get_scalar_constant_value(
                    length_pos, only_process_constants=True
                )
            except NotScalarConstantError:
                pass

            if idx.stop < length_pos_data:
                return False
        elif isinstance(idx.stop, Variable):
            length_pos_shape_i = idx.stop
            # length_pos is a tensor variable, but length_pos_shape_i
            # is a scalar variable. We try to see if they represent
            # the same underlying variable.
            if length_pos_shape_i.owner and isinstance(
                length_pos_shape_i.owner.op, ScalarFromTensor
            ):
                length_pos_shape_i = length_pos_shape_i.owner.inputs[0]
            elif length_pos.owner and isinstance(length_pos.owner.op, TensorFromScalar):
                length_pos = length_pos.owner.inputs[0]
            else:
                # We did not find underlying variables of the same type
                return False

            # The type can be different: int32 vs int64. length_pos
            # should always be int64 as that is what the shape
            # tracker keep. Subtensor accept any scalar int{8,16,32,64}
            # as index type.
            assert str(length_pos.type.dtype) == "int64"
            assert str(length_pos_shape_i.type.dtype) in [
                "int8",
                "int16",
                "int32",
                "int64",
            ]

            # length_pos_shape_i cannot be None
            if length_pos_shape_i != length_pos:
                return False
        elif idx.stop is None:
            continue
        else:
            return False

    return [node.inputs[0]]


@register_canonicalize
@node_rewriter([Subtensor])
def local_convert_negative_indices(fgraph, node):
    """Convert negative indices in `Subtensor` with static length to positive indices."""
    x, *raw_idxs = node.inputs
    idxs = indices_from_subtensor(raw_idxs, node.op.idx_list)

    new_idxs = None
    for i, (dim_length, idx) in enumerate(zip(x.type.shape, idxs)):
        if (
            dim_length is None
            or isinstance(idx, slice)
            or not isinstance(idx, Constant)
        ):
            continue

        val = idx.data
        if val >= 0:
            continue

        new_val = val + dim_length
        if new_val < 0:
            # This is an invalid index, keep original to not confuse the user
            return None

        if new_idxs is None:
            new_idxs = list(idxs)
        new_idxs[i] = new_val

    if new_idxs is None:
        # No negative indices to convert
        return None

    new_subtensor = x[tuple(new_idxs)]
    copy_stack_trace(node.outputs, new_subtensor)
    return [new_subtensor]


@register_canonicalize
@register_specialize
@node_rewriter([AdvancedSubtensor1])
def local_useless_AdvancedSubtensor1(fgraph, node):
    """Remove `AdvancedSubtensor1` if it takes the full input.

    In the `AdvancedSubtensor1` case, the full input is taken when the indices
    are equivalent to ``arange(0, input.shape[0], 1)`` using either an explicit
    list/vector or the `ARange` `Op`.

    """
    # This optimization needs ShapeOpt and fgraph.shape_feature
    if not hasattr(fgraph, "shape_feature"):
        return

    shape_of = fgraph.shape_feature.shape_of

    # get length of the indexed tensor along the first axis
    try:
        length = get_scalar_constant_value(
            shape_of[node.inputs[0]][0], only_process_constants=True
        )
    except NotScalarConstantError:
        return False

    # get index (which must be a vector by definition)
    idx = node.inputs[1]

    # `idx` must be equivalent to [0,1,...,shape[0] - 1] to qualify for
    # this optimization
    if isinstance(idx, Constant):
        idx = idx.value
        if len(idx) != length:
            return False
        if np.any(idx != np.arange(length)):
            return False
    else:
        return False

    return [node.inputs[0]]


def merge_two_slices(fgraph, slice1, len1, slice2, len2):
    """
     This function merges two slices into a single slice. The code works on
     the assumption that:

     a) slice1 is actually a slice and not an index, while slice2
        can be just an index.

     b) the two slices **have been applied consecutively** on the same
        tensor

    The output slice is **not** in canonical form, but actually just a slice
    that can be applied to a tensor to produce the same output as applying
    the two consecutive slices.
    ``len1`` is the length of the tensor **before** applying the first slice,
    while ``len2`` is the length **after** applying the first slice.
    """

    if not isinstance(slice1, slice):
        raise ValueError("slice1 should be of type `slice`")

    # Simple case where one of the slices is useless
    if slice1 == slice(None):
        return slice2
    elif slice2 == slice(None):
        return slice1

    sl1, reverse1 = get_canonical_form_slice(slice1, len1)
    sl2, reverse2 = get_canonical_form_slice(slice2, len2)

    if not isinstance(sl2, slice):
        if reverse1 is None:
            # The first slice is not in reverse, which makes things a lot
            # more clear.
            # In this case we need to take care only of the special cases:
            # len2 <=0    -> throw index error regardless of sl2
            # sl2 > len2  -> throw index error
            # sl2 < -len2 -> throw index error
            # To get a index error we simply use len1+1 to indicate we are
            # out of bounds, because passing this index through the formula
            # of getting the mixed slice is not guaranteed to result in an
            # index error. The **issue though** if that the error will
            # complain about accessing element len1+1 which is probably not
            # too intuitive for the user
            val = sl1.start + sl2 * sl1.step
            val = switch(le(len2, 0), len1 + 1, val)
            val = switch(ge(sl2, len2), len1 + 1, val)
            val = switch(lt(sl2, 0), -len1 - 1, val)
            if sl1.step:
                val = switch(eq(sl1.step, 0), len1 + 1, val)
            return val
        else:
            # We are in the more complex case when we do not actually know
            # if the first slice was in reverse or not.
            # in case it was not in reverse:
            p_val = sl1.start + sl2 * sl1.step
            # case it was in reverse we need to realize that we do not want
            # the k-th element from sl.start but the k-th element from
            # sl.stop backwards
            n_val = sl1.stop - 1 - sl2 * sl1.step
            # we need to pick either n_val or p_val and then follow same
            # steps as above for covering the index error cases
            val = switch(lt(reverse1, 0), n_val, p_val)
            val = switch(le(len2, 0), len1 + 1, val)
            val = switch(ge(sl2, len2), len1 + 1, val)
            val = switch(lt(sl2, 0), -len1 - 1, val)
            if sl1.step is not None:
                val = switch(eq(sl1.step, 0), len1 + 1, val)
            return val
    else:
        # We are deleaing with two slices that need to be put together
        # according to the two steps we have 4 different combinations of
        # positive/negative. I will denote the case I'm looking at by
        # suffixes to the variables (nn,np,pn,pp):
        flen = sl2.stop - sl2.start
        p_step = sl1.step * sl2.step
        n_step = sl1.step * sl2.step * -1

        pp_start = minimum(sl1.start + sl2.start * sl1.step, sl1.stop)
        pp_stop = minimum(sl1.start + sl2.stop * sl1.step, sl1.stop)

        pn_stop = sl1.start + (sl2.start - 1) * sl1.step
        pn_stop = switch(
            and_(lt(pn_stop, 0), gt(flen, 0)),
            -len1 - 1,
            minimum(pn_stop, sl1.stop),
        )
        pn_start = sl1.start + (sl2.stop - 1) * sl1.step
        pn_start = minimum(pn_start, sl1.stop)
        pn_start = maximum(pn_start, 0)

        np_stop = sl1.stop - sl2.stop * sl1.step - 1
        np_stop = switch(
            and_(lt(np_stop, 0), gt(flen, 0)),
            -len1 - 1,
            maximum(sl1.start - 1, np_stop),
        )
        np_start = maximum(sl1.start, sl1.stop - sl2.start * sl1.step - 1)

        nn_start = maximum(sl1.start, (sl1.stop - 1) - (sl2.stop - 1) * sl1.step)
        nn_stop = maximum(sl1.start, sl1.stop - sl2.start * sl1.step)

        start = switch(
            lt(reverse2 * reverse1, 0),
            switch(lt(reverse1, 0), np_start, pn_start),
            switch(lt(reverse1, 0), nn_start, pp_start),
        )

        stop = switch(
            lt(reverse2 * reverse1, 0),
            switch(lt(reverse1, 0), np_stop, pn_stop),
            switch(lt(reverse1, 0), nn_stop, pp_stop),
        )

        step = switch(lt(reverse2 * reverse1, 0), n_step, p_step)
        start = switch(le(flen, 0), 0, start)
        stop = switch(le(flen, 0), 0, stop)

        return slice(start, stop, step)


@register_canonicalize
@node_rewriter([add])
def local_IncSubtensor_serialize(fgraph, node):
    """
    When using Subtensor, gradient graphs can be ugly.

    If we ask for grad(f(a[0]), a), we are going to get something like

        IncSubtensor(Elemwise{second}(a, 0), g(f(a[0])), [0])

    This might be ugly, but at least it's as fast as you could want.
    If we ask for grad(f(a[0], a[1], a[2]), a), it's much worse...

        Elemwise{Add}
            IncSubtensor(Elemwise{second}(a, 0), g(f(a[0])), [0])
            IncSubtensor(Elemwise{second}(a, 0), g(f(a[1])), [1])
            IncSubtensor(Elemwise{second}(a, 0), g(f(a[2])), [2])

    This is much worse because this time we have to produce 3 matrices
    the size of 'a', just so we can add them together.

    This Op rearranges IncSubtensor's that all work on the same
    initial argument (here, Elemwise{second}(a,0)) into a chain.  The
    advantage of the chain structure is that each one can be optimized
    later in the pipeline to operate inplace.

    Ideally, the op will do something like this:

    #
    #  add(x, incsubtensor(b, c), incsubtensor(b, d))
    #  -> incsubtensor(incsubtensor(add(x,b,b), c), d)

    """

    def movable(i):
        # Return True iff this is a incsubtensor that we can move
        return (
            i.owner
            and isinstance(
                i.owner.op,
                IncSubtensor | AdvancedIncSubtensor1 | AdvancedIncSubtensor,
            )
            and i.type.is_super(o_type)
            and len(fgraph.clients[i]) == 1
            and not i.owner.op.set_instead_of_inc
        )

    o_type = node.outputs[0].type

    movable_inputs = [i for i in node.inputs if movable(i)]

    if movable_inputs:
        new_inputs = [i for i in node.inputs if not movable(i)] + [
            mi.owner.inputs[0] for mi in movable_inputs
        ]
        new_add = variadic_add(*new_inputs)
        # Copy over stacktrace from original output, as an error
        # (e.g. an index error) in this add operation should
        # correspond to an error in the original add operation.
        copy_stack_trace(node.outputs[0], new_add)

        # stack up the new incsubtensors
        tip = new_add
        for mi in movable_inputs:
            assert o_type.is_super(tip.type)
            tip = mi.owner.op(tip, *mi.owner.inputs[1:])
            # Copy over stacktrace from outputs of the original
            # "movable" operation to the new operation.
            copy_stack_trace(node.outputs + mi.owner.outputs, tip)

        return [tip]


# We register it in a WalkingGraphRewriter inside the canonizer EQ optimizer.
# Otherwise in some cases it was making the EQ optimizer use 45. In
# the WalkingGraphRewriter, the EQ only use 5 passes.
compile.optdb.register(
    "pre_local_IncSubtensor_serialize",
    in2out(local_IncSubtensor_serialize),
    "fast_run",
    # Just before canonizer
    position=0.99,
)


# after priority 50 Destructive inplace operations
# gemm is the first one now, at priority 70


@node_rewriter([IncSubtensor], inplace=True)
def local_inplace_setsubtensor(fgraph, node):
    if isinstance(node.op, IncSubtensor) and not node.op.inplace:
        dta = node.op.destroyhandler_tolerate_aliased
        new_op = node.op.__class__(
            node.op.idx_list,
            inplace=True,
            set_instead_of_inc=node.op.set_instead_of_inc,
            destroyhandler_tolerate_aliased=dta,
        )
        new_node = new_op(*node.inputs)
        val = getattr(node.outputs[0].tag, "nan_guard_mode_check", True)
        new_node.tag.nan_guard_mode_check = val

        # Copy stacktrace from original outputs to new outputs.
        # This is sensible, because the new operation is the
        # same as the old one, but now with different attributes.
        copy_stack_trace(node.outputs, new_node)
        return [new_node]
    return False


compile.optdb.register(
    "local_inplace_setsubtensor",
    WalkingGraphRewriter(
        local_inplace_setsubtensor, failure_callback=WalkingGraphRewriter.warn_inplace
    ),
    "fast_run",
    "inplace",
    position=50.1,
)


@node_rewriter([AdvancedIncSubtensor1], inplace=True)
def local_inplace_AdvancedIncSubtensor1(fgraph, node):
    if node.op.inplace:
        return

    x, y, idx = node.inputs
    if fgraph.has_destroyers([x]):
        # In this case we can't operate inplace, but if x is just an alloc of zeros
        # We're better off duplicating it and then acting on it inplace.
        if (
            x.owner is not None
            and isinstance(x.owner.op, Alloc)
            and x.owner.op.value_is_scalar_zero(x.owner.inputs[0])
        ):
            x = x.owner.clone().outputs[0]
        else:
            return None  # Inplace isn't valid

    new_op = node.op.clone_inplace()
    new_node = new_op(x, y, idx)
    copy_stack_trace(node.outputs, new_node)
    return [new_node]


compile.optdb.register(
    "local_inplace_AdvancedIncSubtensor1",
    WalkingGraphRewriter(
        local_inplace_AdvancedIncSubtensor1,
        failure_callback=WalkingGraphRewriter.warn_inplace,
    ),
    "fast_run",
    "inplace",
    position=70.6,
)


@node_rewriter([AdvancedIncSubtensor], inplace=True)
def local_inplace_AdvancedIncSubtensor(fgraph, node):
    if isinstance(node.op, AdvancedIncSubtensor) and not node.op.inplace:
        new_op = type(node.op)(
            node.op.idx_list,
            inplace=True,
            set_instead_of_inc=node.op.set_instead_of_inc,
            ignore_duplicates=node.op.ignore_duplicates,
        )
        new_node = new_op(*node.inputs)
        copy_stack_trace(node.outputs, new_node)
        return [new_node]
    return False


compile.optdb.register(
    "local_inplace_AdvancedIncSubtensor",
    WalkingGraphRewriter(
        local_inplace_AdvancedIncSubtensor,
        failure_callback=WalkingGraphRewriter.warn_inplace,
    ),
    "fast_run",
    "inplace",
    position=70.6,
)


# Register old name
@register_canonicalize("local_incsubtensor_of_allocs")
@register_stabilize("local_incsubtensor_of_allocs")
@node_rewriter([IncSubtensor, AdvancedIncSubtensor, AdvancedIncSubtensor1])
def local_incsubtensor_of_zeros(fgraph, node):
    """
    IncSubtensor(x, zeros, idx) -> x

    """
    if (
        isinstance(node.op, IncSubtensor | AdvancedIncSubtensor | AdvancedIncSubtensor1)
        and not node.op.set_instead_of_inc
    ):
        x = node.inputs[0]
        y = node.inputs[1]
        try:
            # Don't use only_process_constants=True. We need to
            # investigate Alloc of 0s but with non constant shape.
            if get_underlying_scalar_constant_value(y, elemwise=False) == 0:
                # No need to copy over the stacktrace,
                # because x should already have a stacktrace
                return [x]
        except NotScalarConstantError:
            return


@register_canonicalize
@register_specialize
@node_rewriter([IncSubtensor])
def local_incsubtensor_of_zeros_to_setsubtensor(fgraph, node):
    """
    IncSubtensor(zeros, x, ...) -> SetSubtensor(zeros, x, ...)
    """
    if isinstance(node.op, (IncSubtensor)) and not node.op.set_instead_of_inc:
        x = node.inputs[0]

        if isinstance(x, Constant) and not np.any(x.data):
            return [
                IncSubtensor(
                    node.op.idx_list,
                    node.op.inplace,
                    set_instead_of_inc=True,
                    destroyhandler_tolerate_aliased=node.op.destroyhandler_tolerate_aliased,
                )(*node.inputs)
            ]


@register_canonicalize("local_setsubtensor_of_allocs")
@register_stabilize("local_setsubtensor_of_allocs")
@node_rewriter([IncSubtensor])
def local_setsubtensor_of_constants(fgraph, node):
    """
    SetSubtensor(x, x[idx], idx) -> x

    when x is constant or alloc.

    """
    if isinstance(node.op, IncSubtensor) and node.op.set_instead_of_inc:
        x = node.inputs[0]
        y = node.inputs[1]

        # Don't use only_process_constants=True. We need to
        # investigate Alloc of 0s but with non constant shape.
        try:
            replace_x = get_underlying_scalar_constant_value(x, elemwise=False)
        except NotScalarConstantError:
            return

        try:
            replace_y = get_underlying_scalar_constant_value(y, elemwise=False)
        except NotScalarConstantError:
            return

        if replace_x == replace_y:
            # No need to copy over the stacktrace,
            # because x should already have a stacktrace
            return [x]
        else:
            return False


def _constant_has_unique_indices(idx) -> bool:
    """Check whether a constant index has no duplicate entries.

    Boolean indices, scalars, and single-element arrays are trivially unique.
    For larger integer arrays, indices that mix positive and negative values
    may alias, so those are treated as potentially duplicated.  The result
    is cached on ``idx.tag``.
    """
    if not isinstance(idx, Constant):
        return False
    cached = getattr(idx.tag, "unique_indices", None)
    if cached is not None:
        return bool(cached)
    idx_val = np.asarray(idx.data)
    if idx_val.dtype == bool:
        result = True
    elif idx_val.size <= 1:
        result = True
    else:
        has_pos = (idx_val >= 0).any()
        has_neg = (idx_val < 0).any()
        result = not (has_pos and has_neg) and np.unique(idx_val).size == idx_val.size
    idx.tag.unique_indices = result
    return result


@register_canonicalize("shape_unsafe")
@register_specialize("shape_unsafe")
@node_rewriter([Subtensor, AdvancedSubtensor1, AdvancedSubtensor])
def local_read_of_write_same_indices(fgraph, node):
    """Read of a write at the same indices: ``x[idx].set/inc(v)[idx]``.

    .. code::

        x[idx].set(v)[idx] -> v
        x[idx].inc(v)[idx] -> x[idx] + v   (idx must be duplicate-free)

    Applies when the outer read and inner write share identical index
    variables (``is`` check) and the same ``idx_list``.  The inc case
    additionally requires duplicate-free indices: slices and scalar indices
    are trivially unique, while integer-array indices must be constant with
    no repeated entries (mixing positive and negative values counts as
    potentially duplicated since they may alias).

    Companion rewrites:

    - ``local_advanced_read_of_write_constant_indices`` handles the multi-axis
      case when read and write indices differ but are both constant.
    - ``local_write_of_write_same_indices`` folds nested write chains.
    """
    if isinstance(node.op, Subtensor):
        write_type = IncSubtensor
    elif isinstance(node.op, AdvancedSubtensor1):
        write_type = AdvancedIncSubtensor1
    else:
        write_type = AdvancedIncSubtensor

    inner = node.inputs[0]
    if not (inner.owner and isinstance(inner.owner.op, write_type)):
        return None

    if node.op.idx_list != inner.owner.op.idx_list:
        return None

    x, v, *inner_idx_vars = inner.owner.inputs
    outer_idx_vars = node.inputs[1:]

    if not all(o is i for o, i in zip(outer_idx_vars, inner_idx_vars, strict=True)):
        return None

    out = node.outputs[0]

    if inner.owner.op.set_instead_of_inc:
        r = cast(v, out.dtype)
        if not r.type.is_super(out.type):
            r = alloc(r, *out.shape)
        copy_stack_trace(out, r)
        return [r]
    else:
        # Inc case: advanced integer-array indices must be duplicate-free;
        # slices and scalar indices are trivially unique.
        indices = indices_from_subtensor(outer_idx_vars, node.op.idx_list)
        for idx in indices:
            if isinstance(idx, TensorVariable) and idx.type.ndim > 0:
                if not _constant_has_unique_indices(idx):
                    return None

        x_at_idx = x[tuple(indices)]
        copy_stack_trace(out, x_at_idx)
        r = x_at_idx + v
        copy_stack_trace(out, r)
        return [r]


@register_canonicalize("shape_unsafe")
@register_specialize("shape_unsafe")
@node_rewriter([AdvancedSubtensor])
def local_advanced_read_of_write_constant_indices(fgraph, node):
    """Read of a write at possibly-different constant indices.

    .. code::

        x[w_idx].set(v)[r_idx] ->
            v[lookup]                     (full coverage — base irrelevant)
            x[r_idx]                      (no coverage — set irrelevant)
            x[r_idx].set(v[k])[...]       (partial — mix base and v)

        x[w_idx].inc(v)[r_idx] ->
            x[r_idx] + v[lookup]          (full coverage, unique writes)
            x[r_idx]                      (no coverage)
            x[r_idx].inc(v[k])[...]       (partial, unique writes)

    Fires only when all advanced indices on both sides are 1-D integer
    constants with matching ``idx_list``.  The inc case additionally requires
    unique joint write coords so each read coord matches at most one write.

    Companion rewrites:

    - ``local_read_of_write_same_indices`` handles the identity-check case
      (symbolic indices allowed) for basic and advanced subtensors.
    - ``local_write_of_write_same_indices`` folds nested write chains.
    """
    inner = node.inputs[0]
    if not (inner.owner and isinstance(inner.owner.op, AdvancedIncSubtensor)):
        return None

    inner_op = inner.owner.op
    is_set = inner_op.set_instead_of_inc

    # Both must have the same idx_list structure (same axes advanced-indexed).
    if node.op.idx_list != inner_op.idx_list:
        return None

    base, v, *write_idx_inputs = inner.owner.inputs
    read_idx_inputs = node.inputs[1:]

    write_indices = indices_from_subtensor(write_idx_inputs, inner_op.idx_list)
    read_indices = indices_from_subtensor(read_idx_inputs, node.op.idx_list)

    # Collect advanced (integer) axes; other axes must be identical slices.
    # Cross-sign indices are rejected since negatives can alias positives
    # (a normalisation rewrite handles those separately).
    write_arrs = []
    read_arrs = []
    for w, r in zip(write_indices, read_indices, strict=True):
        if isinstance(w, TensorVariable) and isinstance(r, TensorVariable):
            if not (isinstance(w, TensorConstant) and isinstance(r, TensorConstant)):
                return None
            # Require proper 1-D array indices, not broadcastable length-1.
            if w.type.broadcastable != (False,) or r.type.broadcastable != (False,):
                return None
            w_arr = np.asarray(w.data)
            r_arr = np.asarray(r.data)
            # Reject only cross-sign within an axis — negatives can alias
            # positives on the same axis, but uniformly negative (or
            # uniformly non-negative) indices compare correctly as raw values.
            # Short-circuit so the common all-non-negative case skips most checks.
            if ((w_arr < 0).any() or (r_arr < 0).any()) and (
                (w_arr >= 0).any() or (r_arr >= 0).any()
            ):
                return None
            write_arrs.append(w_arr)
            read_arrs.append(r_arr)
        elif isinstance(w, slice) and isinstance(r, slice):
            if w != r:
                return None
        else:
            return None

    if not write_arrs:
        return None

    n_write = len(write_arrs[0])

    # Extend indices with implicit trailing slices so axis bookkeeping is
    # uniform regardless of whether the subtensor indexed all base dims.
    n_trailing = base.type.ndim - len(write_indices)
    full_write = list(write_indices) + [slice(None)] * n_trailing

    # Compute where the advanced axis lands in the result of x[indices], per
    # numpy semantics: hoisted to position 0 if the adv indices are split by
    # slices, otherwise kept in place at the position of the first adv axis
    # (counting only slice axes that come before it, since collapsed adv
    # axes share one output dim).
    adv_axes = [
        i for i, idx in enumerate(full_write) if isinstance(idx, TensorVariable)
    ]
    if _non_consecutive_adv_indexing(full_write):
        adv_pos = 0
        slice_shapes = [
            base.shape[i] for i in range(base.type.ndim) if i not in set(adv_axes)
        ]
    else:
        first_adv = min(adv_axes)
        last_adv = max(adv_axes)
        pre = [
            base.shape[i] for i in range(first_adv) if isinstance(full_write[i], slice)
        ]
        post = [
            base.shape[i]
            for i in range(last_adv + 1, base.type.ndim)
            if isinstance(full_write[i], slice)
        ]
        adv_pos = len(pre)
        slice_shapes = pre + post

    # Bring v to its full natural shape so we can index the adv axis directly.
    natural_shape_v = [*slice_shapes[:adv_pos], n_write, *slice_shapes[adv_pos:]]
    v = alloc(v, *natural_shape_v)
    # set_subtensor/inc_subtensor cast v to the buffer dtype internally; we need
    # to do it explicitly so v[lookup] (and subsequent ops) match the output dtype.
    out_dtype = node.outputs[0].type.dtype
    if v.type.dtype != out_dtype:
        v = cast(v, out_dtype)

    write_coords = np.column_stack(write_arrs)  # (n_write, n_axes)
    read_coords = np.column_stack(read_arrs)  # (n_read, n_axes)

    if is_set:
        # Set: last-write-wins; uncovered positions need the base.
        write_dict: dict[tuple, int] = {}
        for k in range(len(write_coords)):
            write_dict[tuple(write_coords[k])] = k
    else:
        # Inc: require unique write coords so each read matches at most one
        # write.  With duplicates we'd need a scatter-add at write positions,
        # which generally isn't simpler than the original inc.
        write_dict = {}
        for k in range(len(write_coords)):
            coord = tuple(write_coords[k])
            if coord in write_dict:
                return None
            write_dict[coord] = k

    lookup = np.array(
        [write_dict.get(tuple(rc), -1) for rc in read_coords], dtype=np.int64
    )
    covered = lookup >= 0

    def constant_idx(idx, merge_feature=getattr(fgraph, "merge_feature", None)):
        # Build (or reuse) the TensorConstant that indexes the adv axis.
        # Read-write graphs on the same idx require structural identity
        # To not be at the mercy of MergeOptimizer firing in time,
        # we eagerly reuse index variables if they already exist in the graph
        # (which is the case in which those rewrites would need)
        idx = tensor_constant(idx)
        if merge_feature is None:
            return idx
        else:
            return merge_feature.atomic_sig_inv.get(idx.signature(), idx)

    def index_adv(t, positions):
        # Index axis `adv_pos` of t with `positions`. Skip if identity.
        if len(positions) == n_write and np.array_equal(positions, np.arange(n_write)):
            return t
        indexer = [slice(None)] * t.type.ndim
        indexer[adv_pos] = constant_idx(positions)
        return t[tuple(indexer)]

    if is_set:
        if covered.all():
            # Every read position is overwritten; base is irrelevant.
            out = index_adv(v, lookup)
        elif not covered.any():
            # No read position is overwritten; the set is irrelevant.
            out = base[tuple(read_indices)]
        else:
            # Mix: read base, then overwrite covered positions with v values.
            base_part = base[tuple(read_indices)]
            covered_read = np.flatnonzero(covered)
            covered_write = lookup[covered]
            indexer = [slice(None)] * base_part.type.ndim
            indexer[adv_pos] = constant_idx(covered_read)
            out = base_part[tuple(indexer)].set(index_adv(v, covered_write))
    else:
        # Inc case (write coords are unique by construction above).
        base_part = base[tuple(read_indices)]
        copy_stack_trace(node.outputs[0], base_part)
        if not covered.any():
            return [base_part]

        if covered.all():
            out = base_part + index_adv(v, lookup)
        else:
            covered_read = np.flatnonzero(covered)
            covered_write = lookup[covered]
            indexer = [slice(None)] * base_part.type.ndim
            indexer[adv_pos] = constant_idx(covered_read)
            out = base_part[tuple(indexer)].inc(index_adv(v, covered_write))

    copy_stack_trace(node.outputs[0], out)
    return [out]


@register_canonicalize
@register_stabilize
@register_specialize
@node_rewriter([IncSubtensor, AdvancedIncSubtensor1, AdvancedIncSubtensor])
def local_write_of_write_same_indices(fgraph, node):
    """Fold nested write ops that share the same indices.

    .. code::

        x[idx].set/inc(a)[idx].set(b) -> x[idx].set(b)
        x[idx].inc(a)[idx].inc(b)     -> x[idx].inc(a + b)
        x[idx].set(a)[idx].inc(b)     -> x[idx].set(a + b)   [unique idx]

    Outer-set always applies (it shadows the inner write).  Inc+inc is safe
    because addition is associative.  Inc-of-set requires duplicate-free
    indices: slices are trivially unique, advanced indices are checked
    per-axis (conservative — joint-tuple uniqueness would be exact).

    If the inc-of-set base is zero-filled the result is emitted as an
    ``inc`` so downstream zero-aware rewrites can still fire.

    Typically arises from gradient accumulation or user code that writes
    then updates the same slice (e.g. Scan updates).

    Companion rewrites:

    - ``local_read_of_write_same_indices`` simplifies a read following a
      write at the same indices.
    - ``local_advanced_read_of_write_constant_indices`` handles multi-axis
      reads with differing constant indices.
    """
    # AdvancedIncSubtensor.ignore_duplicates is not a concern here:
    # the outer-set and inc+inc cases are valid regardless of duplicates,
    # and the inc-of-set case requires verified-unique indices so there
    # are no duplicates for the flag to affect.
    inner_x, b, *outer_idx_vars = node.inputs
    if not (inner_x.owner and isinstance(inner_x.owner.op, type(node.op))):
        return

    base, a, *inner_idx_vars = inner_x.owner.inputs

    # Same indices: idx_list (slice specs) must match and all index
    # variables must be identical.  AdvancedIncSubtensor1 has a fixed
    # class-level idx_list = (0,) so the comparison is trivially true.
    if node.op.idx_list != inner_x.owner.op.idx_list:
        return
    if not all(o is i for o, i in zip(outer_idx_vars, inner_idx_vars, strict=True)):
        return

    outer_is_set = node.op.set_instead_of_inc
    inner_is_set = inner_x.owner.op.set_instead_of_inc

    if outer_is_set:
        # Outer set shadows inner completely.
        new_val = b
        use_set = True
    elif inner_is_set:
        # x[idx].set(a)[idx].inc(b) — needs unique indices.
        # Basic indexing (slices/scalars) is always duplicate-free.
        # For advanced indexing, per-axis uniqueness is conservative but
        # sufficient: it guarantees no duplicates in the joint cross-product
        # after broadcasting.
        if not isinstance(node.op, IncSubtensor):
            if not all(_constant_has_unique_indices(v) for v in outer_idx_vars):
                return
        new_val = a + b
        if (
            get_underlying_scalar_constant_value(
                base, elemwise=False, raise_not_constant=False
            )
            == 0
        ):
            use_set = False
        else:
            use_set = True
    else:
        # x[idx].inc(a)[idx].inc(b) — always safe (addition is associative).
        new_val = a + b
        use_set = False

    if isinstance(node.op, AdvancedIncSubtensor1):
        new_op = AdvancedIncSubtensor1(set_instead_of_inc=use_set)
    else:
        # ignore_duplicates is deliberately not propagated: the merged op
        # should use the safe np.add.at path (the default).
        new_op = type(node.op)(idx_list=node.op.idx_list, set_instead_of_inc=use_set)
    r = new_op(base, new_val, *outer_idx_vars)
    copy_stack_trace(node.outputs[0], r)
    return [r]


@register_specialize
@register_stabilize
@register_canonicalize
@register_useless
@node_rewriter([IncSubtensor, AdvancedIncSubtensor, AdvancedIncSubtensor1])
def local_useless_inc_subtensor_alloc(fgraph, node):
    """
    Replaces an [Advanced]IncSubtensor[1], whose increment is an `alloc` of
    a fully or partially broadcastable variable, by one that skips the
    intermediate `alloc` where possible.

    """
    if isinstance(node.op, IncSubtensor | AdvancedIncSubtensor | AdvancedIncSubtensor1):
        x, y, *index_variables = node.inputs

        if y.owner is not None and isinstance(y.owner.op, Alloc):
            # `z` is the input of the Alloc op, i.e. at.alloc(z, <shape>)
            z = y.owner.inputs[0]

            try:
                shape_feature = fgraph.shape_feature
            except AttributeError:
                # The shape feature may not be available in some mode, but we
                # need it for this optimization, so don't continue.
                return False

            shape_of = shape_feature.shape_of
            same_shape = shape_feature.same_shape

            # Get the subtensor of `x` indexed by `i` in order to compare
            # shapes later.
            if isinstance(node.op, IncSubtensor):
                xi = Subtensor(node.op.idx_list)(x, *index_variables)
            elif isinstance(node.op, AdvancedIncSubtensor):
                xi = AdvancedSubtensor(node.op.idx_list)(x, *index_variables)
            elif isinstance(node.op, AdvancedIncSubtensor1):
                xi = advanced_subtensor1(x, *index_variables)
            else:
                raise Exception("Should never happen!")

            reason = "local_useless_incsubtensor_alloc"

            # Add `xi` to the shape feature `fgraph`. This is important for
            # shape inference later because the variable must be part of the
            # function graph in order to call `same_shape` on it.
            if xi not in shape_of:
                shape_feature.on_import(fgraph, xi.owner, f"{reason}: add `xi`")

            # `xi` may have more dimensions than `y` since the subtensor ops
            # do automatic broadcasting of the increment internally. Thus, we
            # need to make the leading implicitly broadcasted dimensions
            # explicit for shape comparison later.
            if xi.ndim > y.ndim:
                y = shape_padleft(y, xi.ndim - y.ndim)
                if y not in shape_of:
                    shape_feature.on_import(fgraph, y.owner, f"{reason}: add `y`")

            # Build `z_broad` explicitly to include extra implicit dimensions.
            z_broad = (True,) * (xi.ndim - z.ndim) + z.broadcastable

            cond = [
                # The shapes of `y` and `xi` must either agree or `y` may
                # also have shape equal to 1 which may be treated as a
                # broadcastable dimension by the subtensor op.
                or_(eq(y.shape[k], 1), eq(y.shape[k], xi.shape[k]))
                # Loop over all dimensions.
                for k in range(xi.ndim)
                # We need to check the above shapes, if
                # * the pre-alloc increment `z` is broadcastable in
                # dimension `k` (if it isn't, then the shapes of `z` and
                # `y` are the same by the definition of the `Alloc` op in
                # this dimension and replacing `y` by `z` will not hide a
                # shape error), and
                # * `xi` and `y` do not have the same shape in dimension
                # `k` or we cannot infer the shape statically (if the
                # shapes of `xi` and `y` are not the same, then replacing
                # `y` by `z` will hide the shape error of `y`), and
                # * the shape of `y` is not equal to 1 or we cannot infer
                # the shape statically (if the shape of `y` is equal to
                # 1, then `y` is broadcasted by the inc_subtensor op
                # internally, so the shapes of `xi` and `y` do not need
                # to match in dimension `k`; else we need to check at
                # runtime that the shape of `y` is either 1 or the same
                # as `xi` or otherwise replacing `y` by `z` will hide a
                # shape error).
                if (
                    z_broad[k]
                    and not same_shape(xi, y, dim_x=k, dim_y=k)
                    and shape_of[y][k] != 1
                )
            ]

            if len(cond) > 0:
                msg = "`x[i]` and `y` do not have the same shape."
                z = Assert(msg)(z, *cond)

            r = node.op(x, z, *index_variables)
            # Copy over stacktrace from previous output, since
            # we don't expect problems when removing the intermediate
            # alloc operation and so we still want to point at the line
            # of the inc_subtensor operation.
            copy_stack_trace(node.outputs, r)

            return [r]


@register_specialize
@node_rewriter([Join])
def local_join_subtensors(fgraph, node):
    r"""Simplify contiguous :class:`Subtensor`\s inside a :class:`Join`.

    `join((x[:3], x[3:5]), axis=0) -> x[:5]`
    """
    # TODO: Generalize to AdvancedSubtensors

    axis, tensors = node.inputs[0], node.inputs[1:]

    try:
        axis = get_scalar_constant_value(axis)
    except NotScalarConstantError:
        return

    for subtensor1_idx, (subtensor1, subtensor2) in enumerate(
        itertools.pairwise(tensors)
    ):
        # Check that two consecutive Subtensors are operating on the same base tensor
        if not (
            (
                subtensor1.owner is not None
                and isinstance(subtensor1.owner.op, Subtensor)
            )
            and (
                subtensor2.owner is not None
                and isinstance(subtensor2.owner.op, Subtensor)
            )
            and (subtensor1.owner.inputs[0] is subtensor2.owner.inputs[0])
        ):
            continue

        # Check that subtensors have consecutive indexes across the join axis
        idxs_subtensor1 = indices_from_subtensor(
            subtensor1.owner.inputs[1:], subtensor1.owner.op.idx_list
        )
        idxs_subtensor2 = indices_from_subtensor(
            subtensor2.owner.inputs[1:], subtensor2.owner.op.idx_list
        )
        try:
            idxs_axis_subtensor1 = idxs_subtensor1[axis]
            idxs_axis_subtensor2 = idxs_subtensor2[axis]
        except IndexError:
            continue
        if not (
            isinstance(idxs_axis_subtensor1, slice)
            and isinstance(idxs_axis_subtensor2, slice)
        ):
            continue
        start_subtensor1, stop_subtensor1, step_subtensor1 = (
            idxs_axis_subtensor1.start,
            idxs_axis_subtensor1.stop,
            idxs_axis_subtensor1.step,
        )
        start_subtensor2, stop_subtensor2, step_subtensor2 = (
            idxs_axis_subtensor2.start,
            idxs_axis_subtensor2.stop,
            idxs_axis_subtensor2.step,
        )
        if not (
            (stop_subtensor1 is not None and start_subtensor2 is not None)
            and (stop_subtensor1 == start_subtensor2)
        ):
            continue

        # Check that step is None or 1
        # For non-unit steps (perhaps except for -1) we would need to know the
        # exact values of start and stop to know if they can be merged
        for step in (step_subtensor1, step_subtensor2):
            if step is None:
                continue
            try:
                if get_scalar_constant_value(step, only_process_constants=True) != 1:
                    return None
            except NotScalarConstantError:
                return None

        # Check that all other idxs of subtensor are the same
        if all(
            idxs_nonaxis_subtensor1 == idxs_nonaxis_subtensor2
            for i, (idxs_nonaxis_subtensor1, idxs_nonaxis_subtensor2) in enumerate(
                zip(idxs_subtensor1, idxs_subtensor2, strict=True)
            )
            if i != axis
        ):
            base_tensor = subtensor1.owner.inputs[0]
            new_idxs = list(idxs_subtensor1)
            new_idxs[axis] = slice(start_subtensor1, stop_subtensor2, step_subtensor1)
            merged_subtensors = base_tensor[new_idxs]

            new_joined_tensors = [
                *tensors[:subtensor1_idx],
                merged_subtensors,
                *tensors[subtensor1_idx + 2 :],
            ]
            if len(new_joined_tensors) > 1:
                return [concatenate(new_joined_tensors, axis=axis)]
            else:
                return [merged_subtensors]


@node_rewriter(
    [
        Subtensor,
        AdvancedSubtensor1,
        AdvancedSubtensor,
        IncSubtensor,
        AdvancedIncSubtensor,
        AdvancedIncSubtensor1,
    ]
)
def local_uint_constant_indices(fgraph, node):
    """Convert constant indices to unsigned dtypes."""

    op = node.op
    if isinstance(op, IncSubtensor | AdvancedIncSubtensor | AdvancedIncSubtensor1):
        x, y, *indices = node.inputs
    else:
        x, *indices = node.inputs
        y = None

    new_indices = list(indices_from_subtensor(indices, node.op.idx_list))
    has_new_index = False

    for i, index in enumerate(new_indices):
        if not isinstance(index, Constant):
            continue

        index_val = index.data

        if index_val is None or isinstance(index_val, slice):
            # TODO: If slice index dtypes matter, we can consider converting
            # those, as well.
            continue

        assert isinstance(index_val, np.generic | np.ndarray)

        if index_val.size == 0:
            continue

        if index_val.dtype == bool:
            continue

        if np.ndim(index_val) > 0:
            minval = index_val.min()
        else:
            minval = index_val

        if minval >= 0:
            maxval = index_val.max()
            dtype = np.min_scalar_type(maxval)
        else:
            # If we can't convert to unsigned, then don't attempt to minimize
            # the type size either--at least not for now.
            # dtype = np.min_scalar_type(-max(-minval, maxval))
            continue

        if dtype == index_val.dtype:
            continue

        if isinstance(index.type, TensorType):
            new_index = tensor_constant(index_val.astype(dtype), dtype=dtype)
        else:
            new_index = scalar_constant(index_val.astype(dtype), dtype=dtype)

        new_indices[i] = new_index
        has_new_index = True

    if not has_new_index:
        return False

    new_indices = get_slice_elements(new_indices)
    new_args = (x, *new_indices) if y is None else (x, y, *new_indices)
    new_out = op(*new_args)
    copy_stack_trace(node.outputs[0], new_out)
    return [new_out]


compile.optdb.register(
    local_uint_constant_indices.__name__,
    dfs_rewriter(local_uint_constant_indices),
    # We don't include in the Python / C because those always cast indices to int64 internally.
    "numba",
    "jax",
    # After specialization and uncanonicalization
    # Other rewrites don't worry about the dtype of the indices
    # And can cause unnecessary passes of this optimization
    # Such as x.shape[np.int(0)] -> x.shape[np.uint(0)]
    position=4,
)


@register_stabilize
@register_specialize
@node_rewriter([blockwise_of(Subtensor)])
def local_blockwise_of_subtensor(fgraph, node):
    """Rewrite Blockwise of Subtensor, where the only batch input is the indexed tensor.

    Blockwise(Subtensor{a: b})(x, a, b) -> x[:, a:b] when x has one batch dimension, and a/b none

    TODO: Handle batched indices like we do with blockwise of inc_subtensor
    TODO: Extend to AdvanceSubtensor
    """
    x, *idxs = node.inputs
    if not all(all(idx.type.broadcastable) for idx in idxs):
        return

    core_idxs = indices_from_subtensor(
        [idx.squeeze() for idx in idxs], node.op.core_op.idx_list
    )
    # Add empty slices for the batch dims
    none_slices = (slice(None),) * node.op.batch_ndim(node)
    return [x[(*none_slices, *core_idxs)]]


@register_canonicalize("shape_unsafe")
@register_stabilize("shape_unsafe")
@register_specialize("shape_unsafe")
@node_rewriter([blockwise_of(IncSubtensor | AdvancedIncSubtensor)])
def local_blockwise_inc_subtensor(fgraph, node):
    """Rewrite blockwised inc_subtensors.

    Note: The reason we don't apply this rewrite eagerly in the `_vectorize_node` dispatch
    Is that we often have batch dimensions from alloc of shapes/reshape that can be removed by rewrites

    such as x[:vectorized(w.shape[0])].set(y), that will later be rewritten as x[:w.shape[1]].set(y),
    and can be safely rewritten without Blockwise.
    """
    core_op = node.op.core_op
    x, y, *idxs = node.inputs
    [out] = node.outputs
    advanced = isinstance(core_op, AdvancedIncSubtensor)

    if advanced and any(idx.type.dtype == "bool" for idx in idxs):
        # Get out if we have boolean indices as they cross dimension boundaries
        # / can't be safely broadcasted depending on their runtime content
        return None

    batch_ndim = node.op.batch_ndim(node)
    idxs_core_ndim = [len(inp_sig) for inp_sig in node.op.inputs_sig[2:]]
    max_idx_core_ndim = max(idxs_core_ndim, default=0)

    # Broadcast buffer to batch_shape
    if x.type.broadcastable != out.type.broadcastable:
        batch_shape = [1] * batch_ndim
        for inp in node.inputs:
            for i, (broadcastable, batch_dim) in enumerate(
                zip(inp.type.broadcastable[:batch_ndim], tuple(inp.shape)[:batch_ndim])
            ):
                if broadcastable:
                    # This dimension is broadcastable, it doesn't provide shape information
                    continue
                if batch_shape[i] != 1:
                    # We already found a source of shape for this batch dimension
                    continue
                batch_shape[i] = batch_dim
        x = broadcast_to(x, (*batch_shape, *x.shape[batch_ndim:]))
        assert x.type.broadcastable == out.type.broadcastable

    # Massage indices so they respect blockwise semantics while using regular indexing
    core_idxs = []
    for idx_entry in core_op.idx_list:
        if isinstance(idx_entry, slice):
            # Squeeze away dummy dimensions so we can convert to slice
            new_entries = [None, None, None]
            for i, slice_idx_entry in enumerate(
                (idx_entry.start, idx_entry.stop, idx_entry.step)
            ):
                if slice_idx_entry is None:
                    continue
                else:
                    new_entries[i] = new_entry = idxs[slice_idx_entry].squeeze()
                    if new_entry.ndim > 0:
                        # If the slice entry has dimensions after the squeeze we can't convert it to a slice
                        # We could try to convert to equivalent integer indices, but nothing guarantees
                        # that the slice is "square".
                        return None
            squeezed_index = slice(*new_entries)
        else:
            if advanced:
                # For AdvancedIncSubtensor we have tensor integer indices,
                # We need to expand batch indexes on the right, so they don't interact with core index dimensions
                # We still squeeze on the left in case that allows us to use simpler indices
                squeezed_index = _squeeze_left(
                    shape_padright(
                        idxs[idx_entry], max_idx_core_ndim - idxs_core_ndim[idx_entry]
                    ),
                    stop_at_dim=batch_ndim,
                )
            else:
                # For basic IncSubtensor integers indices can be used as is, but we try to squeeze away dummy
                # batch dimensions in case we can end up with a basic IncSubtensor again
                squeezed_index = _squeeze_left(idxs[idx_entry])

        core_idxs.append(squeezed_index)

    # Create new indices for the batch dimensions
    has_batched_indices = not all(
        all(idx.type.broadcastable[:batch_ndim])
        for idx in idxs
        if not isinstance(idx, slice)
    )
    if has_batched_indices:
        # If indices have batch dimensions, we need to align them element-wise with the respective batch dimensions of x
        # We achieve this by creating `arange` indices and adding expand_dims for correct broadcasting.
        # Example:
        # x = pt.zeros(5); idx = [0, 1, 0]; out = x[idx].set(y)
        # batch_x = pt.zeros((2, 5)); batch_idx = [[0, 1, 0], [1, 1, 2]]
        # batch_out = batch_x[[0, 1][:, None], batch_idx].set(y)
        # If instead batch_x = pt.zeros((2, 2, 5))
        # batch_out = batch_x[[0, 1][:, None, None], [0, 1][None, 1, None], batch_idx]

        # Note: For simplicity we use arange for all batch dimensions of x,
        # even if not all may have corresponding batch index dimensions
        batch_slices = [
            shape_padright(arange(x_batch_shape, dtype="int64"), n)
            for (x_batch_shape, n) in zip(
                tuple(x.shape)[:batch_ndim],
                reversed(range(max_idx_core_ndim, max_idx_core_ndim + batch_ndim)),
            )
        ]
    else:
        # In the case we don't have batch indices,
        # we can use slice(None) to broadcast the core indices to each new batch dimension of x / y
        batch_slices = [slice(None)] * batch_ndim

    new_idxs = (*batch_slices, *core_idxs)
    x_view = x[new_idxs]

    # Introduce any implicit expand_dims on core dimension of y
    missing_y_core_ndim = x_view.type.ndim - y.type.ndim
    implicit_axes = tuple(range(batch_ndim, batch_ndim + missing_y_core_ndim))
    y = expand_dims(y, implicit_axes)

    # Transpose y if needed
    if has_batched_indices:
        # By introducing arange slices we may caused a transposition of the advanced group to the front
        # If this was not already happening in the core graph, we'll need to transpose y to align it correctly
        if max_idx_core_ndim and not (
            advanced and _non_consecutive_adv_indexing(core_idxs)
        ):
            integer_pos = [
                i for i, entry in enumerate(core_op.idx_list) if isinstance(entry, int)
            ]
            slice_pos = [
                i
                for i, entry in enumerate(core_op.idx_list)
                if isinstance(entry, slice)
            ]
            if slice_pos and integer_pos and (slice_pos[0] < integer_pos[-1]):
                y = moveaxis(
                    y,
                    [batch_ndim + integer_pos[0] + i for i in range(max_idx_core_ndim)],
                    [batch_ndim + i for i in range(max_idx_core_ndim)],
                )
    else:
        # Conversely if we tried to use `slice(None)` for the batch dimensions but there was already transposition
        # in the core case, we'll need to move the batch slices of y to after the advanced indexing group
        if advanced and _non_consecutive_adv_indexing(core_idxs):
            y = moveaxis(
                y,
                [i for i in range(batch_ndim)],  # noqa: C416
                [max_idx_core_ndim + i for i in range(batch_ndim)],
            )

    # Remove useless left-batch dimensions of y (if any)
    y = _squeeze_left(y, stop_at_dim=batch_ndim)

    if core_op.set_instead_of_inc:
        new_out = x[new_idxs].set(y)
    else:
        new_out = x[new_idxs].inc(y)

    copy_stack_trace(out, new_out)
    return [new_out]


@node_rewriter(tracks=[AdvancedSubtensor, AdvancedIncSubtensor])
def bool_idx_to_nonzero(fgraph, node):
    """Convert boolean indexing into equivalent vector boolean index, supported by our dispatch

    x[1:, eye(3, dtype=bool), 1:] -> x[1:, *eye(3).nonzero()]
    """
    if isinstance(node.op, AdvancedSubtensor):
        x, *idxs = node.inputs
    else:
        x, y, *idxs = node.inputs

    idxs = indices_from_subtensor(idxs, node.op.idx_list)

    bool_pos = {
        i
        for i, idx in enumerate(idxs)
        if isinstance(idx, TensorVariable) and idx.dtype == "bool"
    }

    if not bool_pos:
        return None

    new_idxs = []
    for i, idx in enumerate(idxs):
        if i in bool_pos:
            new_idxs.extend(idx.nonzero())
        else:
            new_idxs.append(idx)

    if isinstance(node.op, AdvancedSubtensor):
        new_out = x[tuple(new_idxs)]
    else:
        new_out = (
            x[tuple(new_idxs)].set(y)
            if node.op.set_instead_of_inc
            else x[tuple(new_idxs)].inc(y)
        )

    return [copy_stack_trace(node.outputs[0], new_out)]


optdb["specialize"].register(
    bool_idx_to_nonzero.__name__,
    bool_idx_to_nonzero,
    "numba",
    "shape_unsafe",  # It can mask invalid mask sizes
    use_db_name_as_tag=False,  # Not included if only "specialize" is requested
)


@register_stabilize("shape_unsafe")
@register_specialize("shape_unsafe")
@node_rewriter([ExtractDiag])
def local_extract_diag_of_write(fgraph, node):
    """Delegate ``extract_diag(advanced_inc_subtensor(...))`` to the constant-indices rewrite.

    Rewrites ``extract_diag(x, offset=k)`` as the equivalent
    ``x[..., arange(d) + max(0, -k), arange(d) + max(0, k), ...]`` and
    calls ``local_advanced_read_of_write_constant_indices`` to do the
    work.  Since ``extract_diag`` is a zero-copy view, we only commit the
    replacement when the downstream rewrite eliminates the gather.

    Requires statically-known sizes on the two diagonal axes.
    """
    op = node.op

    inner = node.inputs[0]
    # AdvancedIncSubtensor1 is intentionally not accepted: it writes whole
    # rows/slices on a single axis, not specific (i, j) positions, so it
    # can't express "write the diagonal" the way two paired index arrays can.
    if not (inner.owner and isinstance(inner.owner.op, AdvancedIncSubtensor)):
        return None

    # Need static sizes on the two diagonal axes to build constant indices.
    dim_a = inner.type.shape[op.axis1]
    dim_b = inner.type.shape[op.axis2]
    if dim_a is None or dim_b is None:
        return None

    k = op.offset
    row_offset = max(0, -k)
    col_offset = max(0, k)
    d = min(dim_a - row_offset, dim_b - col_offset)
    if d <= 0:
        return None

    # Build equivalent AdvancedSubtensor: inner[..., arange(d) + row_offset, ..., arange(d) + col_offset, ...]
    base_arange = np.arange(d, dtype=np.int64)
    rows = pytensor.tensor.as_tensor_variable(base_arange + row_offset)
    cols = pytensor.tensor.as_tensor_variable(base_arange + col_offset)
    idxs = [slice(None)] * inner.type.ndim
    idxs[op.axis1] = rows
    idxs[op.axis2] = cols
    equiv = inner[tuple(idxs)]

    if not (equiv.owner and isinstance(equiv.owner.op, AdvancedSubtensor)):
        return None

    # Delegate to the general read-after-write rewrite.
    result = local_advanced_read_of_write_constant_indices.fn(fgraph, equiv.owner)
    if not result:
        return None

    # Stay zero-copy where possible: when the simplification reduced to a
    # gather of the inner write's base at our diagonal-arange pattern (i.e.
    # the no-coverage case where the write is irrelevant for this read),
    # re-emit as ExtractDiag so we keep the view semantics of the original.
    base = inner.owner.inputs[0]
    [result_var] = result
    if (
        result_var.owner
        and isinstance(result_var.owner.op, AdvancedSubtensor)
        and result_var.owner.inputs[0] is base
    ):
        out = base.diagonal(offset=k, axis1=op.axis1, axis2=op.axis2)
        copy_stack_trace(node.outputs[0], out)
        return [out]

    copy_stack_trace(node.outputs[0], result)
    return result
