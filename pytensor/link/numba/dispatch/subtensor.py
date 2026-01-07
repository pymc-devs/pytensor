import operator
import sys
from hashlib import sha256
from textwrap import dedent, indent

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
from pytensor.link.numba.dispatch.compile_ops import numba_deepcopy
from pytensor.link.numba.dispatch.string_codegen import create_tuple_string
from pytensor.tensor import TensorType
from pytensor.tensor.subtensor import (
    AdvancedIncSubtensor,
    AdvancedIncSubtensor1,
    AdvancedSubtensor,
    AdvancedSubtensor1,
    IncSubtensor,
    Subtensor,
)
from pytensor.tensor.type_other import MakeSlice, NoneTypeT


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


@numba.extending.overload(numba_deepcopy)
def numba_deepcopy_slice(x):
    if isinstance(x, types.SliceType):

        def deepcopy_slice(x):
            return slice(
                numba_deepcopy(x.start), numba_deepcopy(x.stop), numba_deepcopy(x.step)
            )

        return deepcopy_slice


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

    def convert_indices(indice_names, entry):
        if indice_names and isinstance(entry, Type):
            return next(indice_names)
        elif isinstance(entry, slice):
            return (
                f"slice({convert_indices(indice_names, entry.start)}, "
                f"{convert_indices(indice_names, entry.stop)}, "
                f"{convert_indices(indice_names, entry.step)})"
            )
        elif isinstance(entry, type(None)):
            return "None"
        else:
            raise ValueError()

    set_or_inc = isinstance(
        op, IncSubtensor | AdvancedIncSubtensor1 | AdvancedIncSubtensor
    )
    index_start_idx = 1 + int(set_or_inc)
    op_indices = list(node.inputs[index_start_idx:])
    idx_list = getattr(op, "idx_list", None)
    idx_names = [f"idx_{i}" for i in range(len(op_indices))]

    input_names = ["x", "y", *idx_names] if set_or_inc else ["x", *idx_names]

    idx_names_iterator = iter(idx_names)
    indices_creation_src = (
        tuple(convert_indices(idx_names_iterator, idx) for idx in idx_list)
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
    cache_key = subtensor_op_cache_key(
        op, func="numba_funcify_default_subtensor", version=1
    )
    return numba_basic.numba_njit(func, boundscheck=True), cache_key


@register_funcify_and_cache_key(AdvancedSubtensor)
@register_funcify_and_cache_key(AdvancedIncSubtensor)
def numba_funcify_AdvancedSubtensor(op, node, **kwargs):
    if isinstance(op, AdvancedSubtensor):
        _x, _y, idxs = node.inputs[0], None, node.inputs[1:]
    else:
        _x, _y, *idxs = node.inputs

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

    must_ignore_duplicates = (
        isinstance(op, AdvancedIncSubtensor)
        and not op.set_instead_of_inc
        and op.ignore_duplicates
        # Only vector integer indices can have "duplicates", not scalars or boolean vectors
        and not all(
            adv_idx["ndim"] == 0 or adv_idx["dtype"] == "bool" for adv_idx in adv_idxs
        )
    )

    # Special implementation for integer indices that respects duplicates
    if (
        not must_ignore_duplicates
        and len(adv_idxs) >= 1
        and all(adv_idx["dtype"] != "bool" for adv_idx in adv_idxs)
        # Implementation does not support newaxis
        and not any(isinstance(idx.type, NoneTypeT) for idx in idxs)
    ):
        return vector_integer_advanced_indexing(op, node, **kwargs)

    must_respect_duplicates = (
        isinstance(op, AdvancedIncSubtensor)
        and not op.set_instead_of_inc
        and not op.ignore_duplicates
        # Only vector integer indices can have "duplicates", not scalars or boolean vectors
        and not all(
            adv_idx["ndim"] == 0 or adv_idx["dtype"] == "bool" for adv_idx in adv_idxs
        )
    )

    # Cases natively supported by Numba
    if (
        # Numba indexing, like Numpy, ignores duplicates in update
        not must_respect_duplicates
        # Numba does not support indexes with more than one dimension
        and not any(idx["ndim"] > 1 for idx in adv_idxs)
        # Nor multiple vector indexes
        and not sum(idx["ndim"] > 0 for idx in adv_idxs) > 1
    ):
        return numba_funcify_default_subtensor(op, node, **kwargs)

    # Otherwise fallback to obj_mode
    return generate_fallback_impl(op, node, **kwargs), subtensor_op_cache_key(
        op, func="fallback_impl"
    )


@register_funcify_and_cache_key(AdvancedIncSubtensor1)
def numba_funcify_AdvancedIncSubtensor1(op, node, **kwargs):
    return vector_integer_advanced_indexing(op, node=node, **kwargs)


def vector_integer_advanced_indexing(
    op: AdvancedSubtensor1 | AdvancedSubtensor | AdvancedIncSubtensor, node, **kwargs
):
    """Implement all forms of advanced indexing (and assignment) that combine basic and vector integer indices.

        It does not support `newaxis` in basic indices

        It handles += like `np.add.at` would, accumulating add for duplicate indices.

    Examples
    --------

    Codegen for an AdvancedSubtensor, with non-consecutive matrix indices, and a slice(1, None) basic index

    .. code-block:: python

        # AdvancedSubtensor [id A] <Tensor3(int64, shape=(2, 2, 3))>
        #  ├─ <Tensor3(int64, shape=(3, 4, 5))> [id B] <Tensor3(int64, shape=(3, 4, 5))>
        #  ├─ [[1 2] [2 1]] [id C] <Matrix(uint8, shape=(2, 2))>
        #  ├─ SliceConstant{1, None, None} [id D] <slice>
        #  └─ [[0 0] [0 0]] [id E] <Matrix(uint8, shape=(2, 2))>


        def advanced_integer_vector_indexing(x, idx0, idx1, idx2):
            # Move advanced indexed dims to the front (if needed)
            x_adv_dims_front = x.transpose((0, 2, 1))

            # Perform basic indexing once (if needed)
            basic_indexed_x = x_adv_dims_front[:, :, idx1]

            # Broadcast indices
            adv_idx_shape = np.broadcast_shapes(idx0.shape, idx2.shape)
            (idx0, idx2) = (
                np.broadcast_to(idx0, adv_idx_shape),
                np.broadcast_to(idx2, adv_idx_shape),
            )

            # Create output buffer
            adv_idx_size = idx0.size
            basic_idx_shape = basic_indexed_x.shape[2:]
            out_buffer = np.empty((adv_idx_size, *basic_idx_shape), dtype=x.dtype)

            # Index over tuples of raveled advanced indices and write to output buffer
            for i, scalar_idxs in enumerate(zip(idx0.ravel(), idx2.ravel())):
                out_buffer[i] = basic_indexed_x[scalar_idxs]

            # Unravel out_buffer (if needed)
            out_buffer = out_buffer.reshape((*adv_idx_shape, *basic_idx_shape))

            # Move advanced output indexing group to its final position (if needed) and return
            return out_buffer


    Codegen for similar AdvancedSetSubtensor

    .. code-block::python

        AdvancedSetSubtensor [id A] <Tensor3(int64, shape=(3, 4, 5))>
         ├─ x [id B] <Tensor3(int64, shape=(3, 4, 5))>
         ├─ y [id C] <Matrix(int64, shape=(2, 4))>
         ├─ [1 2] [id D] <Vector(uint8, shape=(2,))>
         ├─ SliceConstant{None, None, None} [id E] <slice>
         └─ [3 4] [id F] <Vector(uint8, shape=(2,))>

        def set_advanced_integer_vector_indexing(x, y, idx0, idx1, idx2):
            # Expand dims of y explicitly (if needed)
            y = y

            # Copy x (if not inplace)
            x = x.copy()

            # Move advanced indexed dims to the front (if needed)
            # This will remain a view of x
            x_adv_dims_front = x.transpose((0, 2, 1))

            # Perform basic indexing once (if needed)
            # This will remain a view of x
            basic_indexed_x = x_adv_dims_front[:, :, idx1]

            # Broadcast indices
            adv_idx_shape = np.broadcast_shapes(idx0.shape, idx2.shape)
            (idx0, idx2) = (np.broadcast_to(idx0, adv_idx_shape), np.broadcast_to(idx2, adv_idx_shape))

            # Move advanced indexed dims to the front (if needed)
            y_adv_dims_front = y

            # Broadcast y to the shape of each assignment/update
            adv_idx_shape = idx0.shape
            basic_idx_shape = basic_indexed_x.shape[2:]
            y_bcast = np.broadcast_to(y_adv_dims_front, (*adv_idx_shape, *basic_idx_shape))

            # Ravel the advanced dims (if needed)
            # Note that numba reshape only supports C-arrays, so we ravel before reshape
            y_bcast = y_bcast

            # Index over tuples of raveled advanced indices and update buffer
            for i, scalar_idxs in enumerate(zip(idx0, idx2)):
                basic_indexed_x[scalar_idxs] = y_bcast[i]

            # Return the original x, with the entries updated
            return x


    Codegen for an AdvancedIncSubtensor, with two contiguous advanced groups not in the leading axis

    .. code-block::python

        AdvancedIncSubtensor [id A] <Tensor3(int64, shape=(3, 4, 5))>
         ├─ x [id B] <Tensor3(int64, shape=(3, 4, 5))>
         ├─ y [id C] <Matrix(int64, shape=(2, 2))>
         ├─ SliceConstant{1, None, None} [id D] <slice>
         ├─ [1 2] [id E] <Vector(uint8, shape=(2,))>
         └─ [3 4] [id F] <Vector(uint8, shape=(2,))>

        def inc_advanced_integer_vector_indexing(x, y, idx0, idx1, idx2):
            # Expand dims of y explicitly (if needed)
            y = y

            # Copy x (if not inplace)
            x = x.copy()

            # Move advanced indexed dims to the front (if needed)
            # This will remain a view of x
            x_adv_dims_front = x.transpose((1, 2, 0))

            # Perform basic indexing once (if needed)
            # This will remain a view of x
            basic_indexed_x = x_adv_dims_front[:, :, idx0]

            # Broadcast indices
            adv_idx_shape = np.broadcast_shapes(idx1.shape, idx2.shape)
            (idx1, idx2) = (np.broadcast_to(idx1, adv_idx_shape), np.broadcast_to(idx2, adv_idx_shape))

            # Move advanced indexed dims to the front (if needed)
            y_adv_dims_front = y.transpose((1, 0))

            # Broadcast y to the shape of each assignment/update
            adv_idx_shape = idx1.shape
            basic_idx_shape = basic_indexed_x.shape[2:]
            y_bcast = np.broadcast_to(y_adv_dims_front, (*adv_idx_shape, *basic_idx_shape))

            # Ravel the advanced dims (if needed)
            # Note that numba reshape only supports C-arrays, so we ravel before reshape
            y_bcast = y_bcast

            # Index over tuples of raveled advanced indices and update buffer
            for i, scalar_idxs in enumerate(zip(idx1, idx2)):
                basic_indexed_x[scalar_idxs] += y_bcast[i]

            # Return the original x, with the entries updated
            return x

    """
    if isinstance(op, AdvancedSubtensor1 | AdvancedSubtensor):
        x, *idxs = node.inputs
    else:
        x, y, *idxs = node.inputs
    [out] = node.outputs

    adv_indices_pos = tuple(
        i for i, idx in enumerate(idxs) if isinstance(idx.type, TensorType)
    )
    assert adv_indices_pos  # Otherwise it's just basic indexing
    basic_indices_pos = tuple(
        i for i, idx in enumerate(idxs) if not isinstance(idx.type, TensorType)
    )
    explicit_basic_indices_pos = (*basic_indices_pos, *range(len(idxs), x.type.ndim))

    # Create index signature and split them among basic and advanced
    idx_signature = ", ".join(f"idx{i}" for i in range(len(idxs)))
    adv_indices = [f"idx{i}" for i in adv_indices_pos]
    basic_indices = [f"idx{i}" for i in basic_indices_pos]

    # Define transpose axis so that advanced indexing dims are on the front
    adv_axis_front_order = (*adv_indices_pos, *explicit_basic_indices_pos)
    adv_axis_front_transpose_needed = adv_axis_front_order != tuple(range(x.ndim))
    adv_idx_ndim = max(idxs[i].ndim for i in adv_indices_pos)

    # Helper needed for basic indexing after moving advanced indices to the front
    basic_indices_with_none_slices = ", ".join(
        (*((":",) * len(adv_indices)), *basic_indices)
    )

    # Position of the first advanced index dimension after indexing the array
    if (np.diff(adv_indices_pos) > 1).any():
        # If not consecutive, it's always at the front
        out_adv_axis_pos = 0
    else:
        # Otherwise wherever the first advanced index is located
        out_adv_axis_pos = adv_indices_pos[0]

    to_tuple = create_tuple_string  # alias to make code more readable below

    if isinstance(op, AdvancedSubtensor1 | AdvancedSubtensor):
        # Define transpose axis on the output to restore original meaning
        # After (potentially) having transposed advanced indexing dims to the front unlike numpy
        _final_axis_order = list(range(adv_idx_ndim, out.type.ndim))
        for i in range(adv_idx_ndim):
            _final_axis_order.insert(out_adv_axis_pos + i, i)
        final_axis_order = tuple(_final_axis_order)
        del _final_axis_order
        final_axis_transpose_needed = final_axis_order != tuple(range(out.type.ndim))

        func_name = "advanced_integer_vector_indexing"
        codegen = dedent(
            f"""
            def {func_name}(x, {idx_signature}):
                # Move advanced indexed dims to the front (if needed)
                x_adv_dims_front = {f"x.transpose({adv_axis_front_order})" if adv_axis_front_transpose_needed else "x"}

                # Perform basic indexing once (if needed)
                basic_indexed_x = {f"x_adv_dims_front[{basic_indices_with_none_slices}]" if basic_indices else "x_adv_dims_front"}
            """
        )
        if len(adv_indices) > 1:
            codegen += indent(
                dedent(
                    f"""
                    # Broadcast indices
                    adv_idx_shape = np.broadcast_shapes{to_tuple([f"{idx}.shape" for idx in adv_indices])}
                    {to_tuple(adv_indices)} = {to_tuple([f"np.broadcast_to({idx}, adv_idx_shape)" for idx in adv_indices])}
                    """
                ),
                " " * 4,
            )
        codegen += indent(
            dedent(
                f"""
                # Create output buffer
                adv_idx_size = {adv_indices[0]}.size
                basic_idx_shape = basic_indexed_x.shape[{len(adv_indices)}:]
                out_buffer = np.empty((adv_idx_size, *basic_idx_shape), dtype=x.dtype)

                # Index over tuples of raveled advanced indices and write to output buffer
                for i, scalar_idxs in enumerate(zip{to_tuple([f"{idx}.ravel()" for idx in adv_indices] if adv_idx_ndim != 1 else adv_indices)}):
                    out_buffer[i] = basic_indexed_x[scalar_idxs]

                # Unravel out_buffer (if needed)
                out_buffer = {f"out_buffer.reshape((*{adv_indices[0]}.shape, *basic_idx_shape))" if adv_idx_ndim != 1 else "out_buffer"}

                # Move advanced output indexing group to its final position (if needed) and return
                return {f"out_buffer.transpose({final_axis_order})" if final_axis_transpose_needed else "out_buffer"}
                """
            ),
            " " * 4,
        )

    else:
        # Make implicit dims of y explicit to simplify code
        # Numba doesn't support `np.expand_dims` with multiple axis, so we use indexing with newaxis
        indexed_ndim = x[tuple(idxs)].type.ndim
        y_expand_dims = [":"] * y.type.ndim
        y_implicit_dims = range(indexed_ndim - y.type.ndim)
        for axis in y_implicit_dims:
            y_expand_dims.insert(axis, "None")

        # We transpose the advanced dimensions of x to the front for indexing
        # We may have to do the same for y
        # Note that if there are non-contiguous advanced indices,
        # y must already be aligned with the indices jumping to the front
        y_adv_axis_front_order = tuple(
            range(
                # Position of the first advanced axis after indexing
                out_adv_axis_pos,
                # Position of the last advanced axis after indexing
                out_adv_axis_pos + adv_idx_ndim,
            )
        )
        y_order = tuple(range(indexed_ndim))
        y_adv_axis_front_order = (
            *y_adv_axis_front_order,
            # Basic indices, after explicit_expand_dims
            *(o for o in y_order if o not in y_adv_axis_front_order),
        )
        y_adv_axis_front_transpose_needed = y_adv_axis_front_order != y_order

        func_name = f"{'set' if op.set_instead_of_inc else 'inc'}_advanced_integer_vector_indexing"
        codegen = dedent(
            f"""
            def {func_name}(x, y, {idx_signature}):
                # Expand dims of y explicitly (if needed)
                y = {f"y[{', '.join(y_expand_dims)},]" if y_implicit_dims else "y"}

                # Copy x (if not inplace)
                x = {"x" if op.inplace else "x.copy()"}

                # Move advanced indexed dims to the front (if needed)
                # This will remain a view of x
                x_adv_dims_front = {f"x.transpose({adv_axis_front_order})" if adv_axis_front_transpose_needed else "x"}

                # Perform basic indexing once (if needed)
                # This will remain a view of x
                basic_indexed_x = {f"x_adv_dims_front[{basic_indices_with_none_slices}]" if basic_indices else "x_adv_dims_front"}
            """
        )
        if len(adv_indices) > 1:
            codegen += indent(
                dedent(
                    f"""
                    # Broadcast indices
                    adv_idx_shape = np.broadcast_shapes{to_tuple([f"{idx}.shape" for idx in adv_indices])}
                    {to_tuple(adv_indices)} = {to_tuple([f"np.broadcast_to({idx}, adv_idx_shape)" for idx in adv_indices])}
                    """
                ),
                " " * 4,
            )
        codegen += indent(
            dedent(
                f"""
                # Move advanced indexed dims to the front (if needed)
                y_adv_dims_front = {f"y.transpose({y_adv_axis_front_order})" if y_adv_axis_front_transpose_needed else "y"}

                # Broadcast y to the shape of each assignment/update
                adv_idx_shape = {adv_indices[0]}.shape
                basic_idx_shape = basic_indexed_x.shape[{len(adv_indices)}:]
                y_bcast = np.broadcast_to(y_adv_dims_front, (*adv_idx_shape, *basic_idx_shape))

                # Ravel the advanced dims (if needed)
                # Note that numba reshape only supports C-arrays, so we ravel before reshape
                y_bcast = {"y_bcast.ravel().reshape((-1, *basic_idx_shape))" if adv_idx_ndim != 1 else "y_bcast"}

                # Index over tuples of raveled advanced indices and update buffer
                for i, scalar_idxs in enumerate(zip{to_tuple([f"{idx}.ravel()" for idx in adv_indices] if adv_idx_ndim != 1 else adv_indices)}):
                    basic_indexed_x[scalar_idxs] {"=" if op.set_instead_of_inc else "+="} y_bcast[i]

                # Return the original x, with the entries updated
                return x
                """
            ),
            " " * 4,
        )

    cache_key = subtensor_op_cache_key(
        op,
        codegen=codegen,
    )

    ret_func = numba_basic.numba_njit(
        compile_numba_function_src(
            codegen,
            function_name=func_name,
            global_env=globals(),
        )
    )
    return ret_func, cache_key
