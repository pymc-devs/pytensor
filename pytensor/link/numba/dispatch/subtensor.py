import operator
import sys
from hashlib import sha256

import numba
import numpy as np
from llvmlite import ir
from numba import types
from numba.core.pythonapi import box

import pytensor.link.numba.dispatch.basic as numba_basic
from pytensor.graph import Type
from pytensor.link.numba.cache import (
    compile_numba_function_src,
)
from pytensor.link.numba.dispatch.basic import (
    generate_fallback_impl,
    register_funcify_and_cache_key,
    register_funcify_default_op_cache_key,
)
from pytensor.link.utils import unique_name_generator
from pytensor.tensor import TensorType
from pytensor.tensor.rewriting.subtensor import is_full_slice
from pytensor.tensor.subtensor import (
    AdvancedIncSubtensor,
    AdvancedIncSubtensor1,
    AdvancedSubtensor,
    AdvancedSubtensor1,
    IncSubtensor,
    Subtensor,
)
from pytensor.tensor.type_other import MakeSlice, NoneTypeT, SliceType


def slice_new(self, start, stop, step):
    fnty = ir.FunctionType(self.pyobj, [self.pyobj, self.pyobj, self.pyobj])
    fn = self._get_function(fnty, name="PySlice_New")
    return self.builder.call(fn, [start, stop, step])


def enable_slice_boxing():
    """Enable boxing for Numba's native ``slice``s.

    TODO: this can be removed when https://github.com/numba/numba/pull/6939 is
    merged and a release is made.
    """

    @box(types.SliceType)
    def box_slice(typ, val, c):
        """Implement boxing for ``slice`` objects in Numba.

        This makes it possible to return an Numba's internal representation of a
        ``slice`` object as a proper ``slice`` to Python.
        """
        start = c.builder.extract_value(val, 0)
        stop = c.builder.extract_value(val, 1)
        step = c.builder.extract_value(val, 2) if typ.has_step else None

        # Numba uses sys.maxsize and -sys.maxsize-1 to represent None
        # We want to use None in the Python representation
        none_val = ir.Constant(ir.IntType(64), sys.maxsize)
        neg_none_val = ir.Constant(ir.IntType(64), -sys.maxsize - 1)
        none_obj = c.pyapi.get_null_object()

        start = c.builder.select(
            c.builder.icmp_signed("==", start, none_val),
            none_obj,
            c.box(types.int64, start),
        )

        # None stop is represented as neg_none_val when step is negative
        if step is not None:
            stop_none_val = c.builder.select(
                c.builder.icmp_signed(">", step, ir.Constant(ir.IntType(64), 0)),
                none_val,
                neg_none_val,
            )
        else:
            stop_none_val = none_val
        stop = c.builder.select(
            c.builder.icmp_signed("==", stop, stop_none_val),
            none_obj,
            c.box(types.int64, stop),
        )

        if step is not None:
            step = c.builder.select(
                c.builder.icmp_signed("==", step, none_val),
                none_obj,
                c.box(types.int64, step),
            )
        else:
            step = none_obj

        slice_val = slice_new(c.pyapi, start, stop, step)

        return slice_val

    @numba.extending.overload(operator.contains)
    def in_seq_empty_tuple(x, y):
        if isinstance(x, types.Tuple) and not x.types:
            return lambda x, y: False


enable_slice_boxing()


@register_funcify_default_op_cache_key(MakeSlice)
def numba_funcify_MakeSlice(op, **kwargs):
    @numba_basic.numba_njit
    def makeslice(*x):
        return slice(*x)

    return makeslice


def subtensor_op_cache_key(op, **extra_fields):
    key_parts = [type(op), tuple(extra_fields.items())]
    if hasattr(op, "idx_list"):
        idx_parts = []
        for idx in op.idx_list:
            if isinstance(idx, slice):
                idx_parts.append(
                    (
                        idx.start is None,
                        idx.stop is None,
                        idx.step is None,
                    )
                )
            else:
                idx_parts.append("i")
        key_parts.append(tuple(idx_parts))
    if isinstance(op, IncSubtensor | AdvancedIncSubtensor | AdvancedIncSubtensor1):
        key_parts.append((op.inplace, op.set_instead_of_inc))
    if isinstance(op, AdvancedIncSubtensor):
        key_parts.append(op.ignore_duplicates)
    return sha256(str(tuple(key_parts)).encode()).hexdigest()


@register_funcify_and_cache_key(Subtensor)
@register_funcify_and_cache_key(IncSubtensor)
@register_funcify_and_cache_key(AdvancedSubtensor1)
def numba_funcify_default_subtensor(op, node, **kwargs):
    """Create a Python function that assembles and uses an index on an array."""

    unique_names = unique_name_generator(
        ["subtensor", "incsubtensor", "z"], suffix_sep="_"
    )

    def convert_indices(indices, entry):
        if indices and isinstance(entry, Type):
            rval = indices.pop(0)
            return unique_names(rval)
        elif isinstance(entry, slice):
            return (
                f"slice({convert_indices(indices, entry.start)}, "
                f"{convert_indices(indices, entry.stop)}, "
                f"{convert_indices(indices, entry.step)})"
            )
        elif isinstance(entry, type(None)):
            return "None"
        else:
            raise ValueError()

    set_or_inc = isinstance(
        op, IncSubtensor | AdvancedIncSubtensor1 | AdvancedIncSubtensor
    )
    index_start_idx = 1 + int(set_or_inc)

    input_names = [unique_names(v, force_unique=True) for v in node.inputs]
    op_indices = list(node.inputs[index_start_idx:])
    idx_list = getattr(op, "idx_list", None)

    indices_creation_src = (
        tuple(convert_indices(op_indices, idx) for idx in idx_list)
        if idx_list
        else tuple(input_names[index_start_idx:])
    )

    if len(indices_creation_src) == 1:
        indices_creation_src = f"indices = ({indices_creation_src[0]},)"
    else:
        indices_creation_src = ", ".join(indices_creation_src)
        indices_creation_src = f"indices = ({indices_creation_src})"

    if set_or_inc:
        if op.inplace:
            index_prologue = f"z = {input_names[0]}"
        else:
            index_prologue = f"z = np.copy({input_names[0]})"

        if node.inputs[1].ndim == 0:
            # TODO FIXME: This is a hack to get around a weird Numba typing
            # issue.  See https://github.com/numba/numba/issues/6000
            y_name = f"{input_names[1]}.item()"
        else:
            y_name = input_names[1]

        if op.set_instead_of_inc:
            function_name = "set_subtensor"
            index_body = f"z[indices] = {y_name}"
        else:
            function_name = "inc_subtensor"
            index_body = f"z[indices] += {y_name}"
    else:
        function_name = "subtensor"
        index_prologue = ""
        index_body = f"z = {input_names[0]}[indices]"

    subtensor_def_src = f"""
def {function_name}({", ".join(input_names)}):
    {index_prologue}
    {indices_creation_src}
    {index_body}
    return np.asarray(z)
    """

    func = compile_numba_function_src(
        subtensor_def_src,
        function_name=function_name,
        global_env=globals() | {"np": np},
    )
    cache_key = subtensor_op_cache_key(op, func="numba_funcify_default_subtensor")
    return numba_basic.numba_njit(func, boundscheck=True), cache_key


@register_funcify_and_cache_key(AdvancedSubtensor)
@register_funcify_and_cache_key(AdvancedIncSubtensor)
def numba_funcify_AdvancedSubtensor(op, node, **kwargs):
    if isinstance(op, AdvancedSubtensor):
        _x, _y, idxs = node.inputs[0], None, node.inputs[1:]
    else:
        _x, _y, *idxs = node.inputs

    basic_idxs = [
        idx
        for idx in idxs
        if (
            isinstance(idx.type, NoneTypeT)
            or (isinstance(idx.type, SliceType) and not is_full_slice(idx))
        )
    ]
    adv_idxs = [
        {
            "axis": i,
            "dtype": idx.type.dtype,
            "bcast": idx.type.broadcastable,
            "ndim": idx.type.ndim,
        }
        for i, idx in enumerate(idxs)
        if isinstance(idx.type, TensorType)
    ]

    # Special implementation for consecutive integer vector indices
    if (
        not basic_idxs
        and len(adv_idxs) >= 2
        # Must be integer vectors
        # Todo: we could allow shape=(1,) if this is the shape of x
        and all(
            (adv_idx["bcast"] == (False,) and adv_idx["dtype"] != "bool")
            for adv_idx in adv_idxs
        )
        # Must be consecutive
        and not op.non_consecutive_adv_indexing(node)
    ):
        return numba_funcify_multiple_integer_vector_indexing(op, node, **kwargs)

    # Other cases not natively supported by Numba (fallback to obj-mode)
    if (
        # Numba does not support indexes with more than one dimension
        any(idx["ndim"] > 1 for idx in adv_idxs)
        # Nor multiple vector indexes
        or sum(idx["ndim"] > 0 for idx in adv_idxs) > 1
        # The default PyTensor implementation does not handle duplicate indices correctly
        or (
            isinstance(op, AdvancedIncSubtensor)
            and not op.set_instead_of_inc
            and not (
                op.ignore_duplicates
                # Only vector integer indices can have "duplicates", not scalars or boolean vectors
                or all(
                    adv_idx["ndim"] == 0 or adv_idx["dtype"] == "bool"
                    for adv_idx in adv_idxs
                )
            )
        )
    ):
        return generate_fallback_impl(op, node, **kwargs), subtensor_op_cache_key(
            op, func="fallback_impl"
        )

    # What's left should all be supported natively by numba
    return numba_funcify_default_subtensor(op, node, **kwargs)


def _broadcasted_to(x_bcast: tuple[bool, ...], to_bcast: tuple[bool, ...]):
    # Check that x is not broadcasted to y based on broadcastable info
    if len(x_bcast) < len(to_bcast):
        return True
    for x_bcast_dim, to_bcast_dim in zip(x_bcast, to_bcast, strict=True):
        if x_bcast_dim and not to_bcast_dim:
            return True
    return False


def numba_funcify_multiple_integer_vector_indexing(
    op: AdvancedSubtensor | AdvancedIncSubtensor, node, **kwargs
):
    # Special-case implementation for multiple consecutive vector integer indices (and set/incsubtensor)
    if isinstance(op, AdvancedSubtensor):
        idxs = node.inputs[1:]
    else:
        idxs = node.inputs[2:]

    first_axis = next(
        i for i, idx in enumerate(idxs) if isinstance(idx.type, TensorType)
    )
    try:
        after_last_axis = next(
            i
            for i, idx in enumerate(idxs[first_axis:], start=first_axis)
            if not isinstance(idx.type, TensorType)
        )
    except StopIteration:
        after_last_axis = len(idxs)
    last_axis = after_last_axis - 1

    vector_indices = idxs[first_axis:after_last_axis]
    assert all(v.type.broadcastable == (False,) for v in vector_indices)
    y_is_broadcasted = False

    if isinstance(op, AdvancedSubtensor):

        @numba_basic.numba_njit
        def advanced_subtensor_multiple_vector(x, *idxs):
            none_slices = idxs[:first_axis]
            vec_idxs = idxs[first_axis:after_last_axis]

            x_shape = x.shape
            idx_shape = vec_idxs[0].shape
            shape_bef = x_shape[:first_axis]
            shape_aft = x_shape[after_last_axis:]
            out_shape = (*shape_bef, *idx_shape, *shape_aft)
            out_buffer = np.empty(out_shape, dtype=x.dtype)
            for i, scalar_idxs in enumerate(zip(*vec_idxs)):
                out_buffer[(*none_slices, i)] = x[(*none_slices, *scalar_idxs)]
            return out_buffer

        ret_func = advanced_subtensor_multiple_vector

    else:
        inplace = op.inplace

        # Check if y must be broadcasted
        # Includes the last integer vector index,
        x, y = node.inputs[:2]
        indexed_bcast_dims = (
            *x.type.broadcastable[:first_axis],
            *x.type.broadcastable[last_axis:],
        )
        y_is_broadcasted = _broadcasted_to(y.type.broadcastable, indexed_bcast_dims)

        if op.set_instead_of_inc:

            @numba_basic.numba_njit
            def advanced_set_subtensor_multiple_vector(x, y, *idxs):
                vec_idxs = idxs[first_axis:after_last_axis]
                x_shape = x.shape

                if inplace:
                    out = x
                else:
                    out = x.copy()

                if y_is_broadcasted:
                    y = np.broadcast_to(y, x_shape[:first_axis] + x_shape[last_axis:])

                for outer in np.ndindex(x_shape[:first_axis]):
                    for i, scalar_idxs in enumerate(zip(*vec_idxs)):
                        out[(*outer, *scalar_idxs)] = y[(*outer, i)]
                return out

            ret_func = advanced_set_subtensor_multiple_vector

        else:

            @numba_basic.numba_njit
            def advanced_inc_subtensor_multiple_vector(x, y, *idxs):
                vec_idxs = idxs[first_axis:after_last_axis]
                x_shape = x.shape

                if inplace:
                    out = x
                else:
                    out = x.copy()

                if y_is_broadcasted:
                    y = np.broadcast_to(y, x_shape[:first_axis] + x_shape[last_axis:])

                for outer in np.ndindex(x_shape[:first_axis]):
                    for i, scalar_idxs in enumerate(zip(*vec_idxs)):
                        out[(*outer, *scalar_idxs)] += y[(*outer, i)]
                return out

            ret_func = advanced_inc_subtensor_multiple_vector

    cache_key = subtensor_op_cache_key(
        op,
        func="multiple_integer_vector_indexing",
        y_is_broadcasted=y_is_broadcasted,
    )
    return ret_func, cache_key


@register_funcify_and_cache_key(AdvancedIncSubtensor1)
def numba_funcify_AdvancedIncSubtensor1(op, node, **kwargs):
    inplace = op.inplace
    set_instead_of_inc = op.set_instead_of_inc
    x, vals, _idxs = node.inputs
    broadcast_with_index = vals.type.ndim < x.type.ndim or vals.type.broadcastable[0]
    # TODO: Add runtime_broadcast check

    if set_instead_of_inc:
        if broadcast_with_index:

            @numba_basic.numba_njit(boundscheck=True)
            def advancedincsubtensor1_inplace(x, val, idxs):
                if val.ndim == x.ndim:
                    core_val = val[0]
                elif val.ndim == 0:
                    # Workaround for https://github.com/numba/numba/issues/9573
                    core_val = val.item()
                else:
                    core_val = val

                for idx in idxs:
                    x[idx] = core_val
                return x

        else:

            @numba_basic.numba_njit(boundscheck=True)
            def advancedincsubtensor1_inplace(x, vals, idxs):
                if not len(idxs) == len(vals):
                    raise ValueError("The number of indices and values must match.")
                # no strict argument because incompatible with numba
                for idx, val in zip(idxs, vals):
                    x[idx] = val
                return x
    else:
        if broadcast_with_index:

            @numba_basic.numba_njit(boundscheck=True)
            def advancedincsubtensor1_inplace(x, val, idxs):
                if val.ndim == x.ndim:
                    core_val = val[0]
                elif val.ndim == 0:
                    # Workaround for https://github.com/numba/numba/issues/9573
                    core_val = val.item()
                else:
                    core_val = val

                for idx in idxs:
                    x[idx] += core_val
                return x

        else:

            @numba_basic.numba_njit(boundscheck=True)
            def advancedincsubtensor1_inplace(x, vals, idxs):
                if not len(idxs) == len(vals):
                    raise ValueError("The number of indices and values must match.")
                # no strict argument because unsupported by numba
                # TODO: this doesn't come up in tests
                for idx, val in zip(idxs, vals):
                    x[idx] += val
                return x

    cache_key = subtensor_op_cache_key(
        op,
        func="numba_funcify_advancedincsubtensor1",
        broadcast_with_index=broadcast_with_index,
    )

    if inplace:
        return advancedincsubtensor1_inplace, cache_key

    else:

        @numba_basic.numba_njit
        def advancedincsubtensor1(x, vals, idxs):
            x = x.copy()
            return advancedincsubtensor1_inplace(x, vals, idxs)

        return advancedincsubtensor1, cache_key
