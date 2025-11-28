import itertools
import sys

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
from pytensor.graph.type import Type
from pytensor.raise_op import Assert
from pytensor.scalar import Add, ScalarConstant, ScalarType
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
    full,
    get_scalar_constant_value,
    get_underlying_scalar_constant_value,
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
from pytensor.tensor.math import all as pt_all
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
    advanced_inc_subtensor1,
    advanced_subtensor,
    advanced_subtensor1,
    as_index_constant,
    get_canonical_form_slice,
    get_constant_idx,
    get_idx_list,
    get_slice_elements,
    inc_subtensor,
    indices_from_subtensor,
)
from pytensor.tensor.type import TensorType, integer_dtypes
from pytensor.tensor.type_other import NoneTypeT, SliceType
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
    else:
        shape = shape_parts[0]

    ndim = a.ndim + indices.ndim - 1

    return transform_take(a, indices.flatten(), axis).reshape(shape, ndim=ndim)


def is_full_slice(x):
    """Determine if `x` is a ``slice(None)`` or a symbolic equivalent."""
    if isinstance(x, slice):
        return x == slice(None)

    if isinstance(x, Variable) and isinstance(x.type, SliceType):
        if x.owner is None:
            if isinstance(x, Constant):
                return x.data == slice(None)
            else:
                # Root slice variable
                return False

        # Symbolic MakeSlice
        # Ignores start = 0, step = 1 cases
        return all(isinstance(i.type, NoneTypeT) for i in x.owner.inputs)

    return False


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
        if not found_idx and is_full_slice(idx):
            # Preceding full slices
            axis += 1
        elif found_idx and not is_full_slice(idx):
            # We don't handle multiple indices
            return
        elif found_idx and is_full_slice(idx):
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

    indexed_var = node.inputs[0]
    tensor_inputs = node.inputs[1:]

    # Reconstruct indices from idx_list and tensor inputs
    indices = []
    input_idx = 0
    for entry in node.op.idx_list:
        if isinstance(entry, slice):
            indices.append(entry)
        elif isinstance(entry, Type):
            if input_idx < len(tensor_inputs):
                indices.append(tensor_inputs[input_idx])
                input_idx += 1

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

    res = node.inputs[0]
    val = node.inputs[1]
    tensor_inputs = node.inputs[2:]

    # Reconstruct indices from idx_list and tensor inputs
    indices = []
    input_idx = 0
    for entry in node.op.idx_list:
        if isinstance(entry, slice):
            indices.append(entry)
        elif isinstance(entry, Type):
            if input_idx < len(tensor_inputs):
                indices.append(tensor_inputs[input_idx])
                input_idx += 1

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
@node_rewriter([Subtensor])
def local_useless_slice(fgraph, node):
    """
    Remove useless slice(None) of the form:
        1. X[0, :] -> X[0]
        2. X[:] -> X

    Also, canonicalize slices of the form:
        X[0:7:1] -> X[None:None:None]
        where X is a vector of length 7

    And:
        X[-1:-8:-1] -> X[::-1]
        where x is a vector of length 7

    """
    idxs = get_idx_list(node.inputs, node.op.idx_list)
    x = node.inputs[0]

    if not idxs:
        return [node.inputs[0]]

    new_idxs = list(idxs)
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

        if start is not None and get_scalar_constant_value(
            start, only_process_constants=True, raise_not_constant=False
        ) == (0 if positive_step else -1):
            change_flag = True
            start = None

        if (
            stop is not None
            and x.type.shape[dim] is not None
            and get_scalar_constant_value(
                stop, only_process_constants=True, raise_not_constant=False
            )
            == (x.type.shape[dim] if positive_step else -x.type.shape[dim] - 1)
        ):
            change_flag = True
            stop = None

        if start is not None or stop is not None or step is not None:
            last_useful_idx = dim

        new_idxs[dim] = slice(start, stop, step)

    if change_flag or ((last_useful_idx + 1) < len(idxs)):
        new_idxs = tuple(new_idxs[: last_useful_idx + 1])
        out = x[new_idxs] if new_idxs else x
        # Copy over previous output stacktrace
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
    subtens = Subtensor(merged_slices)

    sl_ins = get_slice_elements(merged_slices, lambda x: isinstance(x, Variable))
    # Do not call make_node for test_value
    out = subtens(x, *sl_ins)

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
        if isinstance(elem, ScalarType):
            # The idx is a ScalarType, ie a Type. This means the actual index
            # is contained in node.inputs[1]
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
        elif isinstance(elem, int | np.integer):
            if elem in (0, -1) and node.inputs[0].broadcastable[dim]:
                remove_dim.append(dim)
        else:
            raise TypeError("case not expected")

    if len(remove_dim) == 0:
        return
    else:
        all_dim = range(node.inputs[0].ndim)
        remain_dim = [x for x in all_dim if x not in remove_dim]
        return [node.inputs[0].dimshuffle(tuple(remain_dim))]


@register_specialize
@register_canonicalize
@node_rewriter([Subtensor])
def local_subtensor_inc_subtensor(fgraph, node):
    """
    Subtensor(SetSubtensor(x, y, idx), idx) -> y

    """
    if isinstance(node.op, Subtensor):
        x = node.inputs[0]
        if not (x.owner and isinstance(x.owner.op, IncSubtensor)):
            return
        if not x.owner.op.set_instead_of_inc:
            return

        if x.owner.inputs[2:] == node.inputs[1:] and tuple(
            x.owner.op.idx_list
        ) == tuple(node.op.idx_list):
            out = node.outputs[0]
            y = x.owner.inputs[1]
            # If the dtypes differ, cast y into x.dtype
            if x.dtype != y.dtype:
                y = y.astype(x.dtype)
            if (
                out.type.dtype == y.type.dtype
                and out.type.broadcastable == y.type.broadcastable
            ):
                # if x[idx] and y have the same type, directly return y
                return [y]
            else:
                # The difference is related to broadcasting pattern
                assert out.broadcastable != y.broadcastable
                # We have to alloc y to the shape of x[idx]
                x_subtensor = node.op(x.owner.inputs[0], *x.owner.inputs[2:])
                return [alloc(y, *x_subtensor.shape)]
        else:
            return


@register_useless
@register_canonicalize
@register_specialize
@node_rewriter([IncSubtensor])
def local_useless_inc_subtensor(fgraph, node):
    r"""Remove redundant `IncSubtensor`\s.

    More specifically, ``set_subtensor(x[indices], y)`` is replaced by
    ``y[indices]`` when ``indices`` are full `slice`\s and ``y``'s shape is
    equal to ``x[indices]``, and ``inc_subtensor(x[indices], y)`` is replaced
    by ``y[indices]`` when ``x[indices]`` is some array of ``0``\s, ``indices``
    are full slices, and the shapes are equal.
    """
    if not isinstance(node.op, IncSubtensor):
        return

    if not hasattr(fgraph, "shape_feature"):
        return

    x, y, *index_inputs = node.inputs

    if node.op.set_instead_of_inc is False:
        # This is an increment operation, so the array being incremented must
        # consist of all zeros in order for the entire operation to be useless
        try:
            c = get_underlying_scalar_constant_value(x)
            if c != 0:
                return
        except NotScalarConstantError:
            return

    idx_cst = indices_from_subtensor(list(index_inputs), node.op.idx_list)

    # Check that all indices are full slices with only reversals and no step
    # sizes
    # TODO: It seems like there should be a basic `IncSubtensor`
    # canonicalization that removes these redundant slices.
    if all(
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
        for e in idx_cst
    ):
        # `IncSubtensor` broadcasts `x` on `y` based on run-time shapes, so we
        # must check that they are the same
        if not fgraph.shape_feature.same_shape(x, y):
            return

        # There are no reversals, so we don't need a replacement.
        if all(e.step is None for e in node.op.idx_list):
            # They are exactly the same shapes, so we can remove this `IncSubtensor`
            return [y]

        new_node = Subtensor(node.op.idx_list).make_node(y, *index_inputs)
        new_out = new_node.outputs[0]
        copy_stack_trace(node.outputs, new_out)

        return [new_out]


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
    if is_full_slice(slice1):
        return slice2
    elif is_full_slice(slice2):
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


@register_canonicalize
@register_specialize
@node_rewriter([AdvancedSubtensor1])
def local_adv_sub1_adv_inc_sub1(fgraph, node):
    """Rewrite graphs like ``AdvancedSubtensor1(AdvancedSetSubtensor1(...), ...)``.

    .. code::

        AdvancedSubtensor1(AdvancedSetSubtensor1(x, y, idx), idx) -> y


    Notes
    -----
    This rewrite adds an `AssertOp`; otherwise, it would remove shape and index
    error. If you want to get rid of them, see the :ref:`unsafe_rewrites`
    section.

    A previous version of this rewrite also matched
    ``AdvancedSubtensor1(AdvancedIncSubtensor1(x, y, idx), idx)``.
    This is incorrect when there are duplicate indices.
    The current version warns the user about potential issues.

    """
    if not isinstance(node.op, AdvancedSubtensor1):
        return
    inp = node.inputs[0]
    if not (inp.owner and isinstance(inp.owner.op, AdvancedIncSubtensor1)):
        return
    idx = node.inputs[1]
    idx2 = inp.owner.inputs[2]
    x = inp.owner.inputs[0]
    y = inp.owner.inputs[1]
    if idx is not idx2:
        return
    if (
        not inp.owner.op.set_instead_of_inc
        and
        # Don't use only_process_constants=True. We need to
        # investigate Alloc of 0s but with non constant shape.
        get_underlying_scalar_constant_value(
            x, elemwise=False, raise_not_constant=False
        )
        != 0
    ):
        return

    if not inp.owner.op.set_instead_of_inc:
        return

    cond = [pt_all(and_(lt(idx, x.shape[0]), ge(idx, -x.shape[0])))]
    if not fgraph.shape_feature.same_shape(idx, y, 0, 0):
        cond.append(eq(idx.shape[0], y.shape[0]))
    r = Assert(
        "Bad indexing or shapes in a AdvancedIncSubtensor1 that was optimized away"
    )(y, *cond)
    copy_stack_trace(y, r)

    if r.dtype == node.outputs[0].dtype:
        return [r]
    # It is possible that y is upcast or downcast to x.dtype.
    # In all case, as we set or add with 0, we can just cast y.
    r2 = cast(r, node.outputs[0].dtype)

    # Copy over stacktrace from before casting, since
    # we don't expect problems in the casting operation,
    # and any problems in the indexing would have been spotted above.
    copy_stack_trace(r, r2)
    return [r2]


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
        x = node.inputs[0]
        y = node.inputs[1]
        i = node.inputs[2:]

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
                xi = Subtensor(node.op.idx_list)(x, *i)
            elif isinstance(node.op, AdvancedIncSubtensor):
                xi = advanced_subtensor(x, *i)
            elif isinstance(node.op, AdvancedIncSubtensor1):
                xi = advanced_subtensor1(x, *i)
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
                    and shape_of[xi][k] == 1
                )
            ]

            if len(cond) > 0:
                msg = "`x[i]` and `y` do not have the same shape."
                z = Assert(msg)(z, *cond)

            r = node.op(x, z, *i)
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

    idx_list = getattr(node.op, "idx_list", None)
    new_indices = list(indices_from_subtensor(indices, idx_list))
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

    if isinstance(op, Subtensor | IncSubtensor):
        # Basic index Ops contain information about the dtype of the indices, so wee have to recreate them
        props = op._props_dict()
        props["idx_list"] = new_indices
        op = type(op)(**props)
        # Basic index Ops don't expect slices, but the respective start/step/stop
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

    Note: The reason we don't apply this rewrite eagerly in the `vectorize_node` dispatch
    Is that we often have batch dimensions from alloc of shapes/reshape that can be removed by rewrites

    such as x[:vectorized(w.shape[0])].set(y), that will later be rewritten as x[:w.shape[1]].set(y),
    and can be safely rewritten without Blockwise.
    """
    core_op = node.op.core_op
    x, y, *idxs = node.inputs
    [out] = node.outputs
    if isinstance(core_op, AdvancedIncSubtensor):
        if any(
            (
                # Blockwise requires all inputs to be tensors so it is not possible
                # to wrap an AdvancedIncSubtensor with slice / newaxis inputs, but we check again just in case
                # If this is ever supported we need to pay attention to special behavior of numpy when advanced indices
                # are separated by basic indices
                isinstance(idx, SliceType | NoneTypeT)
                # Also get out if we have boolean indices as they cross dimension boundaries
                # / can't be safely broadcasted depending on their runtime content
                or (idx.type.dtype == "bool")
            )
            for idx in idxs
        ):
            return None

    batch_ndim = node.op.batch_ndim(node)
    idxs_core_ndim = [len(inp_sig) for inp_sig in node.op.inputs_sig[2:]]
    max_idx_core_ndim = max(idxs_core_ndim, default=0)

    # Step 1. Broadcast buffer to batch_shape
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

    # Step 2. Massage indices so they respect blockwise semantics
    if isinstance(core_op, IncSubtensor):
        # For basic IncSubtensor there are two cases:
        # 1. Slice entries -> We need to squeeze away dummy dimensions so we can convert back to slice
        # 2. Integers -> Can be used as is, but we try to squeeze away dummy batch dimensions
        #   in case we can end up with a basic IncSubtensor again
        core_idxs = []
        counter = 0
        for idx in core_op.idx_list:
            if isinstance(idx, slice):
                # Squeeze away dummy dimensions so we can convert to slice
                new_entries = [None, None, None]
                for i, entry in enumerate((idx.start, idx.stop, idx.step)):
                    if entry is None:
                        continue
                    else:
                        new_entries[i] = new_entry = idxs[counter].squeeze()
                        counter += 1
                        if new_entry.ndim > 0:
                            # If the slice entry has dimensions after the squeeze we can't convert it to a slice
                            # We could try to convert to equivalent integer indices, but nothing guarantees
                            # that the slice is "square".
                            return None
                core_idxs.append(slice(*new_entries))
            else:
                core_idxs.append(_squeeze_left(idxs[counter]))
                counter += 1
    else:
        # For AdvancedIncSubtensor we have tensor integer indices,
        # We need to expand batch indexes on the right, so they don't interact with core index dimensions
        # We still squeeze on the left in case that allows us to use simpler indices
        core_idxs = [
            _squeeze_left(
                shape_padright(idx, max_idx_core_ndim - idx_core_ndim),
                stop_at_dim=batch_ndim,
            )
            for idx, idx_core_ndim in zip(idxs, idxs_core_ndim)
        ]

    # Step 3. Create new indices for the new batch dimension of x
    if not all(
        all(idx.type.broadcastable[:batch_ndim])
        for idx in idxs
        if not isinstance(idx, slice)
    ):
        # If indices have batch dimensions in the indices, they will interact with the new dimensions of x
        # We build vectorized indexing with new arange indices that do not interact with core indices or each other
        # (i.e., they broadcast)

        # Note: due to how numpy handles non-consecutive advanced indexing (transposing it to the front),
        # we don't want to create a mix of slice(None), and arange() indices for the new batch dimension,
        # even if not all batch dimensions have corresponding batch indices.
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

    # Step 4. Introduce any implicit expand_dims on core dimension of y
    missing_y_core_ndim = x_view.type.ndim - y.type.ndim
    implicit_axes = tuple(range(batch_ndim, batch_ndim + missing_y_core_ndim))
    y = _squeeze_left(expand_dims(y, implicit_axes), stop_at_dim=batch_ndim)

    if isinstance(core_op, IncSubtensor):
        # Check if we can still use a basic IncSubtensor
        if isinstance(x_view.owner.op, Subtensor):
            new_props = core_op._props_dict()
            new_props["idx_list"] = x_view.owner.op.idx_list
            new_core_op = type(core_op)(**new_props)
            symbolic_idxs = x_view.owner.inputs[1:]
            new_out = new_core_op(x, y, *symbolic_idxs)
        else:
            # We need to use AdvancedSet/IncSubtensor
            if core_op.set_instead_of_inc:
                new_out = x[new_idxs].set(y)
            else:
                new_out = x[new_idxs].inc(y)
    else:
        # AdvancedIncSubtensor takes symbolic indices/slices directly, no need to create a new op
        symbolic_idxs = x_view.owner.inputs[1:]
        new_out = core_op(x, y, *symbolic_idxs)

    copy_stack_trace(out, new_out)
    return [new_out]


@node_rewriter(tracks=[AdvancedSubtensor, AdvancedIncSubtensor])
def ravel_multidimensional_bool_idx(fgraph, node):
    """Convert multidimensional boolean indexing into equivalent vector boolean index, supported by Numba

    x[eye(3, dtype=bool)] -> x.ravel()[eye(3).ravel()]
    x[eye(3, dtype=bool)].set(y) -> x.ravel()[eye(3).ravel()].set(y).reshape(x.shape)
    """
    if isinstance(node.op, AdvancedSubtensor):
        x = node.inputs[0]
        tensor_inputs = node.inputs[1:]
    else:
        x, y = node.inputs[0], node.inputs[1]
        tensor_inputs = node.inputs[2:]

    # Reconstruct indices from idx_list and tensor inputs
    idxs = []
    input_idx = 0
    for entry in node.op.idx_list:
        if isinstance(entry, slice):
            idxs.append(entry)
        elif isinstance(entry, Type):
            if input_idx < len(tensor_inputs):
                idxs.append(tensor_inputs[input_idx])
                input_idx += 1

    if any(
        (
            (isinstance(idx.type, TensorType) and idx.type.dtype in integer_dtypes)
            or isinstance(idx.type, NoneTypeT)
        )
        for idx in idxs
    ):
        # Get out if there are any other advanced indexes or np.newaxis
        return None

    bool_idxs = [
        (i, idx)
        for i, idx in enumerate(idxs)
        if (isinstance(idx.type, TensorType) and idx.dtype == "bool")
    ]

    if len(bool_idxs) != 1:
        # Get out if there are no or multiple boolean idxs
        return None

    [(bool_idx_pos, bool_idx)] = bool_idxs
    bool_idx_ndim = bool_idx.type.ndim
    if bool_idx.type.ndim < 2:
        # No need to do anything if it's a vector or scalar, as it's already supported by Numba
        return None

    x_shape = x.shape
    raveled_x = x.reshape(
        (*x_shape[:bool_idx_pos], -1, *x_shape[bool_idx_pos + bool_idx_ndim :])
    )

    raveled_bool_idx = bool_idx.ravel()
    new_idxs = list(idxs)
    new_idxs[bool_idx_pos] = raveled_bool_idx

    if isinstance(node.op, AdvancedSubtensor):
        # Create new AdvancedSubtensor with updated idx_list
        new_idx_list = list(node.op.idx_list)
        new_tensor_inputs = list(tensor_inputs)

        # Update the idx_list and tensor_inputs for the raveled boolean index
        input_idx = 0
        for i, entry in enumerate(node.op.idx_list):
            if isinstance(entry, Type):
                if input_idx == bool_idx_pos:
                    new_tensor_inputs[input_idx] = raveled_bool_idx
                input_idx += 1

        new_out = AdvancedSubtensor(new_idx_list)(raveled_x, *new_tensor_inputs)
    else:
        # Create new AdvancedIncSubtensor with updated idx_list
        new_idx_list = list(node.op.idx_list)
        new_tensor_inputs = list(tensor_inputs)

        # Update the tensor_inputs for the raveled boolean index
        input_idx = 0
        for i, entry in enumerate(node.op.idx_list):
            if isinstance(entry, Type):
                if input_idx == bool_idx_pos:
                    new_tensor_inputs[input_idx] = raveled_bool_idx
                input_idx += 1

        # The dimensions of y that correspond to the boolean indices
        # must already be raveled in the original graph, so we don't need to do anything to it
        new_out = AdvancedIncSubtensor(
            new_idx_list,
            inplace=node.op.inplace,
            set_instead_of_inc=node.op.set_instead_of_inc,
            ignore_duplicates=node.op.ignore_duplicates,
        )(raveled_x, y, *new_tensor_inputs)
        # But we must reshape the output to match the original shape
        new_out = new_out.reshape(x_shape)

    return [copy_stack_trace(node.outputs[0], new_out)]


@node_rewriter(tracks=[AdvancedSubtensor, AdvancedIncSubtensor])
def ravel_multidimensional_int_idx(fgraph, node):
    """Convert multidimensional integer indexing into equivalent consecutive vector integer index,
    supported by Numba or by our specialized dispatchers

        x[eye(3)] -> x[eye(3).ravel()].reshape((3, 3))

    NOTE: This is very similar to the rewrite `local_replace_AdvancedSubtensor` except it also handles non-full slices

        x[eye(3), 2:] -> x[eye(3).ravel(), 2:].reshape((3, 3, ...)), where ... are the remaining output shapes

    It also handles multiple integer indices, but only if they don't broadcast

        x[eye(3,), 2:, eye(3)] -> x[eye(3).ravel(), eye(3).ravel(), 2:].reshape((3, 3, ...)), where ... are the remaining output shapes

    Also handles AdvancedIncSubtensor, but only if the advanced indices are consecutive and neither indices nor y broadcast

        x[eye(3), 2:].set(y) -> x[eye(3).ravel(), 2:].set(y.reshape(-1, y.shape[1:]))

    """
    op = node.op
    non_consecutive_adv_indexing = op.non_consecutive_adv_indexing(node)
    is_inc_subtensor = isinstance(op, AdvancedIncSubtensor)

    if is_inc_subtensor:
        x, y, *idxs = node.inputs
        # Inc/SetSubtensor is harder to reason about due to y
        # We get out if it's broadcasting or if the advanced indices are non-consecutive
        if non_consecutive_adv_indexing or (
            y.type.broadcastable != x[tuple(idxs)].type.broadcastable
        ):
            return None

    else:
        x, *idxs = node.inputs

    if any(
        (
            (isinstance(idx.type, TensorType) and idx.type.dtype == "bool")
            or isinstance(idx.type, NoneTypeT)
        )
        for idx in idxs
    ):
        # Get out if there are any other advanced indices or np.newaxis
        return None

    int_idxs_and_pos = [
        (i, idx)
        for i, idx in enumerate(idxs)
        if (isinstance(idx.type, TensorType) and idx.dtype in integer_dtypes)
    ]

    if not int_idxs_and_pos:
        return None

    int_idxs_pos, int_idxs = zip(
        *int_idxs_and_pos, strict=False
    )  # strict=False because by definition it's true

    first_int_idx_pos = int_idxs_pos[0]
    first_int_idx = int_idxs[0]
    first_int_idx_bcast = first_int_idx.type.broadcastable

    if any(int_idx.type.broadcastable != first_int_idx_bcast for int_idx in int_idxs):
        # We don't have a view-only broadcasting operation
        # Explicitly broadcasting the indices can incur a memory / copy overhead
        return None

    int_idxs_ndim = len(first_int_idx_bcast)
    if (
        int_idxs_ndim == 0
    ):  # This should be a basic indexing operation, rewrite elsewhere
        return None

    int_idxs_need_raveling = int_idxs_ndim > 1
    if not (int_idxs_need_raveling or non_consecutive_adv_indexing):
        # Numba or our dispatch natively supports consecutive vector indices, nothing needs to be done
        return None

    # Reorder non-consecutive indices
    if non_consecutive_adv_indexing:
        assert not is_inc_subtensor  # Sanity check that we got out if this was the case
        # This case works as if all the advanced indices were on the front
        transposition = list(int_idxs_pos) + [
            i for i in range(len(idxs)) if i not in int_idxs_pos
        ]
        idxs = tuple(idxs[a] for a in transposition)
        x = x.transpose(transposition)
        first_int_idx_pos = 0
        del int_idxs_pos  # Make sure they are not wrongly used

    # Ravel multidimensional indices
    if int_idxs_need_raveling:
        idxs = list(idxs)
        for idx_pos, int_idx in enumerate(int_idxs, start=first_int_idx_pos):
            idxs[idx_pos] = int_idx.ravel()

    # Index with reordered and/or raveled indices
    new_subtensor = x[tuple(idxs)]

    if is_inc_subtensor:
        y_shape = tuple(y.shape)
        y_raveled_shape = (
            *y_shape[:first_int_idx_pos],
            -1,
            *y_shape[first_int_idx_pos + int_idxs_ndim :],
        )
        y_raveled = y.reshape(y_raveled_shape)

        new_out = inc_subtensor(
            new_subtensor,
            y_raveled,
            set_instead_of_inc=op.set_instead_of_inc,
            ignore_duplicates=op.ignore_duplicates,
            inplace=op.inplace,
        )

    else:
        # Unravel advanced indexing dimensions
        raveled_shape = tuple(new_subtensor.shape)
        unraveled_shape = (
            *raveled_shape[:first_int_idx_pos],
            *first_int_idx.shape,
            *raveled_shape[first_int_idx_pos + 1 :],
        )
        new_out = new_subtensor.reshape(unraveled_shape)

    return [copy_stack_trace(node.outputs[0], new_out)]


optdb["specialize"].register(
    ravel_multidimensional_bool_idx.__name__,
    ravel_multidimensional_bool_idx,
    "numba",
    use_db_name_as_tag=False,  # Not included if only "specialize" is requested
)

optdb["specialize"].register(
    ravel_multidimensional_int_idx.__name__,
    ravel_multidimensional_int_idx,
    "numba",
    use_db_name_as_tag=False,  # Not included if only "specialize" is requested
)


@register_canonicalize
@register_stabilize
@register_specialize
@node_rewriter([ExtractDiag])
def extract_diag_of_diagonal_set_subtensor(fgraph, node):
    """Undo extract diagonal from a set diagonal

    This rewrites the following pattern:
        y = write_diagonal*(x, x_diag, offset=k1)
        z = extract_diag(y, offset=k2)

    as:
        z = diag_x, if k1 == k2
        z = x if k1 != k2

    * write_diagonal is not an atomic operation, but a sequence of Arange/SetSubtensor operations.

    """

    def is_cosntant_arange(var) -> bool:
        if not (isinstance(var, TensorConstant) and var.type.ndim == 1):
            return False

        data = var.data
        start, stop = data[0], data[-1] + 1
        return data.size == (stop - start) and (data == np.arange(start, stop)).all()  # type: ignore

    [diag_x] = node.inputs
    if not (
        diag_x.owner is not None
        and isinstance(diag_x.owner.op, AdvancedIncSubtensor)
        and diag_x.owner.op.set_instead_of_inc
    ):
        return None

    x, y, *idxs = diag_x.owner.inputs

    if not (
        x.type.ndim >= 2
        and None not in x.type.shape[-2:]
        and x.type.shape[-2] == x.type.shape[-1]
    ):
        # TODO: for now we only support rewrite with static square shape for x
        return None

    op = node.op
    if op.axis2 > len(idxs):
        return None

    # Check all non-axis indices are full slices
    axis = {op.axis1, op.axis2}
    if not all(is_full_slice(idx) for i, idx in enumerate(idxs) if i not in axis):
        return None

    # Check axis indices are arange we would expect from setting on the diagonal
    axis1_idx = idxs[op.axis1]
    axis2_idx = idxs[op.axis2]
    if not (is_cosntant_arange(axis1_idx) and is_cosntant_arange(axis2_idx)):
        return None

    dim_length = x.type.shape[-1]
    offset = op.offset
    start_stop1 = (axis1_idx.data[0], axis1_idx.data[-1] + 1)
    start_stop2 = (axis2_idx.data[0], axis2_idx.data[-1] + 1)
    orig_start1, orig_start2 = start_stop1[0], start_stop2[0]

    if offset < 0:
        # The logic for checking if we are selecting or not a diagonal for negative offset is the same
        # as the one with positive offset but swapped axis
        start_stop1, start_stop2 = start_stop2, start_stop1
        offset = -offset

    start1, stop1 = start_stop1
    start2, stop2 = start_stop2
    if (
        start1 == 0
        and start2 == offset
        and stop1 == dim_length - offset
        and stop2 == dim_length
    ):
        # We are extracting the just written diagonal
        if y.type.ndim == 0 or y.type.shape[-1] == 1:
            # We may need to broadcast y
            y = full((*x.shape[:-2], dim_length - offset), y, dtype=x.type.dtype)
        return [y]
    elif (orig_start2 - orig_start1) != op.offset:
        # Some other diagonal was written, ignore it
        return [op(x)]
    else:
        # A portion, but no the whole diagonal was written, don't do anything
        return None
