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
from pytensor.graph import Variable
from pytensor.link.numba.cache import (
    compile_numba_function_src,
)
from pytensor.link.numba.dispatch.basic import (
    generate_fallback_impl,
    register_funcify_and_cache_key,
)
from pytensor.link.numba.dispatch.compile_ops import numba_deepcopy
from pytensor.link.numba.dispatch.string_codegen import (
    CODE_TOKEN,
    build_source_code,
    create_tuple_string,
)
from pytensor.tensor import TensorType, TensorVariable
from pytensor.tensor.rewriting.scalarize import ScalarSubtensor
from pytensor.tensor.subtensor import (
    AdvancedIncSubtensor,
    AdvancedSubtensor,
    IncSubtensor,
    Subtensor,
    indices_from_subtensor,
)


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
    if isinstance(op, IncSubtensor | AdvancedIncSubtensor):
        key_parts.append((op.inplace, op.set_instead_of_inc))
    if isinstance(op, AdvancedIncSubtensor):
        key_parts.append(op.ignore_duplicates)
    return sha256(str(tuple(key_parts)).encode()).hexdigest()


def _scalar_loop_nest(
    body: str,
    loop_indices: list[str],
    shape: str,
    *,
    first_axis: int = 0,
) -> str:
    """Wrap body in one loop per dimension, from outermost to innermost.

    ``first_axis`` selects the first dimension of ``shape`` to loop over.
    For indices ``i, j`` and shape ``out.shape``, the emitted loop nest is:

    .. code-block:: python

        for i in range(out.shape[0]):
            for j in range(out.shape[1]):
                out[i, j] = value
    """
    code: list[str | CODE_TOKEN] = []
    for axis, index_name in enumerate(loop_indices, start=first_axis):
        code.extend([f"for {index_name} in range({shape}[{axis}]):", CODE_TOKEN.INDENT])
    code.extend(body.splitlines())
    code.extend([CODE_TOKEN.DEDENT] * len(loop_indices))
    return build_source_code(code)


def _update_value(
    name: str, static_shape: tuple[int | None, ...], indices: list[str]
) -> str:
    """Index an update at its original rank, specializing its known dimensions.

    For output indices ``i, j``, the emitted reads depend on the update's static
    shape. Unknown dimensions retain Subtensor's runtime broadcasting behavior:

    .. code-block:: python

        y[(0, j)]  # Static shape (1, 5)
        y[(j,)]  # Static shape (5,)
        y[(0 if y.shape[0] == 1 else i, 0)]  # Static shape (None, 1)
    """
    offset = len(indices) - len(static_shape)
    source_indices = []
    for axis, (length, index) in enumerate(
        zip(static_shape, indices[offset:], strict=True)
    ):
        if length == 1:
            source_indices.append("0")
        elif length is None:
            source_indices.append(f"0 if {name}.shape[{axis}] == 1 else {index}")
        else:
            source_indices.append(index)
    return f"{name}[{create_tuple_string(source_indices)}]"


@register_funcify_and_cache_key(Subtensor)
@register_funcify_and_cache_key(IncSubtensor)
def numba_funcify_default_subtensor(op, node, scalar_out=False, **kwargs):
    """Create a Python function that assembles and uses an index on an array.

    With ``scalar_out``, return the indexed scalar directly instead of boxing it
    with ``np.asarray``.

    Examples
    --------
    Reading a matrix row ``x[i]``:

    .. code-block:: python

        def subtensor(x, idx_0):
            indices = (idx_0,)
            z = x[indices]
            return np.asarray(z)

    Incrementing ``x[:, i]``, where ``x`` has three dimensions and ``y`` has
    static shape ``(1, 5)``:

    .. code-block:: python

        def inc_subtensor(x, y, idx_0):
            z = np.copy(x)
            indices = (slice(None, None), idx_0)
            target = z[indices]
            # Check broadcast compatibility; we work directly on y.
            np.broadcast_to(y, target.shape)
            for j0 in range(target.shape[0]):
                for j1 in range(target.shape[1]):
                    target[(j0, j1)] += y[(0, j1)]
            return np.asarray(z)

    Setting a matrix row ``x[i]`` to a scalar in place:

    .. code-block:: python

        def set_subtensor(x, y, idx_0):
            z = x
            indices = (idx_0,)
            target = z[indices]
            y_scalar = y.item()
            for j0 in range(target.shape[0]):
                target[(j0,)] = y_scalar
            return np.asarray(z)
    """

    def convert_indices(indices_iterator, entry):
        if isinstance(entry, int):
            name, var = next(indices_iterator)
            if var.ndim == 0 and isinstance(var.type, TensorType):
                return f"{name}.item()"
            return name
        elif isinstance(entry, slice):
            start = convert_indices(indices_iterator, entry.start)
            stop = convert_indices(indices_iterator, entry.stop)
            # Omitting a missing step lets Numba preserve contiguous view types.
            if entry.step is None:
                return f"slice({start}, {stop})"
            step = convert_indices(indices_iterator, entry.step)
            return f"slice({start}, {stop}, {step})"
        elif isinstance(entry, type(None)):
            return "None"
        else:
            raise ValueError(f"Unknown index type: {entry}")

    set_or_inc = isinstance(op, IncSubtensor | AdvancedIncSubtensor)
    index_start_idx = 1 + int(set_or_inc)
    op_indices = list(node.inputs[index_start_idx:])
    idx_list = op.idx_list
    idx_names = [f"idx_{i}" for i in range(len(op_indices))]

    input_names = ["x", "y", *idx_names] if set_or_inc else ["x", *idx_names]

    indices_iterator = iter(zip(idx_names, op_indices))
    indices_creation_src = tuple(
        convert_indices(indices_iterator, idx) for idx in idx_list
    )

    if len(indices_creation_src) == 1:
        indices_creation_src = f"indices = ({indices_creation_src[0]},)"
    else:
        indices_creation_src = ", ".join(indices_creation_src)
        indices_creation_src = f"indices = ({indices_creation_src})"

    if set_or_inc:
        if op.inplace:
            index_prologue = [f"z = {input_names[0]}"]
        else:
            index_prologue = [f"z = np.copy({input_names[0]})"]

        if node.inputs[1].ndim == 0:
            # TODO FIXME: This is a hack to get around a weird Numba typing
            # issue.  See https://github.com/numba/numba/issues/6000
            y_name = f"{input_names[1]}.item()"
        else:
            y_name = input_names[1]

        function_name = "set_subtensor" if op.set_instead_of_inc else "inc_subtensor"
        update = "=" if op.set_instead_of_inc else "+="
        target_ndim = node.inputs[0].ndim - sum(
            isinstance(idx, int) for idx in idx_list
        )
        if isinstance(op, IncSubtensor) and target_ndim:
            loop_idxs = [f"j{k}" for k in range(target_ndim)]
            target_idx = create_tuple_string(loop_idxs)
            index_body = ["target = z[indices]"]
            y = node.inputs[1]
            if y.ndim == 0:
                index_body.append("y_scalar = y.item()")
                update_value = "y_scalar"
            else:
                index_body.extend(
                    [
                        "# Check broadcast compatibility; we work directly on y.",
                        "np.broadcast_to(y, target.shape)",
                    ]
                )
                update_value = _update_value("y", y.type.shape, loop_idxs)
            update_loop = _scalar_loop_nest(
                f"target[{target_idx}] {update} {update_value}",
                loop_idxs,
                "target.shape",
            )
            index_body.extend(update_loop.splitlines())
        else:
            index_body = [f"z[indices] {update} {y_name}"]
    else:
        function_name = "subtensor"
        index_prologue = []
        index_body = [f"z = {input_names[0]}[indices]"]

    return_src = "z" if scalar_out else "np.asarray(z)"
    subtensor_def_src = build_source_code(
        [
            f"def {function_name}({', '.join(input_names)}):",
            CODE_TOKEN.INDENT,
            *index_prologue,
            indices_creation_src,
            *index_body,
            f"return {return_src}",
            CODE_TOKEN.DEDENT,
        ]
    )

    func = compile_numba_function_src(
        subtensor_def_src,
        function_name=function_name,
        global_env=globals() | {"np": np},
    )
    cache_key = subtensor_op_cache_key(op, codegen=subtensor_def_src)
    return numba_basic.numba_njit(func, boundscheck=True), cache_key


@register_funcify_and_cache_key(ScalarSubtensor)
def numba_funcify_ScalarSubtensor(op, node, **kwargs):
    [subtensor_node] = [n for n in op.fgraph.apply_nodes if isinstance(n.op, Subtensor)]
    return numba_funcify_default_subtensor(
        subtensor_node.op, subtensor_node, scalar_out=True, **kwargs
    )


@register_funcify_and_cache_key(AdvancedSubtensor)
@register_funcify_and_cache_key(AdvancedIncSubtensor)
def numba_funcify_AdvancedSubtensor(op, node, **kwargs):
    if isinstance(op, AdvancedSubtensor):
        _x, *index_variables = node.inputs
    else:
        _x, _y, *index_variables = node.inputs

    reconstructed_indices = indices_from_subtensor(index_variables, op.idx_list)

    adv_idxs = []
    for i, idx in enumerate(reconstructed_indices):
        if isinstance(idx, TensorVariable):
            # This is an advanced tensor index
            adv_idxs.append(
                {
                    "axis": i,
                    "dtype": idx.type.dtype,
                    "bcast": idx.type.broadcastable,
                    "ndim": idx.type.ndim,
                }
            )

    must_ignore_duplicates = (
        isinstance(op, AdvancedIncSubtensor)
        and not op.set_instead_of_inc
        and op.ignore_duplicates
        # Only vector integer indices can have "duplicates", not scalars or boolean vectors
        and not all(
            adv_idx["ndim"] == 0 or adv_idx["dtype"] == "bool" for adv_idx in adv_idxs
        )
    )

    if (
        not must_ignore_duplicates
        and len(adv_idxs) >= 1
        and all(adv_idx["dtype"] != "bool" for adv_idx in adv_idxs)
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


def vector_integer_advanced_indexing(
    op: AdvancedSubtensor | AdvancedIncSubtensor, node, **kwargs
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

            # Create output buffer, shaped like the advanced group
            adv_idx_shape = idx0.shape
            basic_idx_shape = basic_indexed_x.shape[2:]
            out_buffer = np.empty((*adv_idx_shape, *basic_idx_shape), dtype=x.dtype)

            # Walk the broadcast advanced shape, reading each index view by position
            for a0 in range(adv_idx_shape[0]):
                for a1 in range(adv_idx_shape[1]):
                    s0 = idx0[(a0, a1)]
                    s1 = idx2[(a0, a1)]
                    for j0 in range(out_buffer.shape[2]):
                        out_buffer[(a0, a1, j0)] = basic_indexed_x[(s0, s1, j0)]

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

            # Validate broadcasting without using the resulting strided view
            adv_idx_shape = idx0.shape
            basic_idx_shape = basic_indexed_x.shape[2:]
            update_shape = (*adv_idx_shape, *basic_idx_shape)
            # Check broadcast compatibility; we work directly on y.
            np.broadcast_to(y, update_shape)

            # Index over tuples of advanced indices and update buffer
            for i, scalar_idxs in enumerate(zip(idx0, idx2)):
                for j0 in range(update_shape[1]):
                    basic_indexed_x[(*scalar_idxs, j0)] = y[(i, j0)]

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

            # Validate broadcasting in the original indexed dimension order
            adv_idx_shape = idx1.shape
            basic_idx_shape = basic_indexed_x.shape[2:]
            update_shape = (*adv_idx_shape, *basic_idx_shape)
            # Check broadcast compatibility; we work directly on y.
            np.broadcast_to(y, (update_shape[1], update_shape[0]))

            # Index over tuples of advanced indices and update buffer
            for i, scalar_idxs in enumerate(zip(idx1, idx2)):
                for j0 in range(update_shape[1]):
                    basic_indexed_x[(*scalar_idxs, j0)] += y[(j0, i)]

            # Return the original x, with the entries updated
            return x

    """

    if isinstance(op, AdvancedSubtensor):
        x, *index_variables = node.inputs
    else:
        x, y, *index_variables = node.inputs

    [out] = node.outputs

    reconstructed_indices = indices_from_subtensor(index_variables, op.idx_list)

    idx_args = [f"idx{i}" for i in range(len(index_variables))]
    var_to_arg = dict(zip(index_variables, idx_args))

    idxs = []

    def get_idx_str(val, is_slice_component=False):
        if val is None:
            return "None"
        if isinstance(val, Variable) and val in var_to_arg:
            arg = var_to_arg[val]
            if val.ndim == 0 and is_slice_component:
                return f"{arg}.item()"
            return arg
        raise ValueError(f"Unexpected index value: {val}")

    for idx in reconstructed_indices:
        if isinstance(idx, slice):
            start = get_idx_str(idx.start, is_slice_component=True)
            stop = get_idx_str(idx.stop, is_slice_component=True)
            if idx.step is None:
                idxs.append(f"slice({start}, {stop})")
            else:
                step = get_idx_str(idx.step, is_slice_component=True)
                idxs.append(f"slice({start}, {stop}, {step})")
        else:
            # It's a direct index variable
            idxs.append(get_idx_str(idx, is_slice_component=False))

    adv_indices_pos = tuple(
        i for i, idx in enumerate(reconstructed_indices) if not isinstance(idx, slice)
    )
    assert adv_indices_pos  # Otherwise it's just basic indexing
    basic_indices_pos = tuple(
        i for i, idx in enumerate(reconstructed_indices) if isinstance(idx, slice)
    )

    # Create index signature for generated function: "idx0, idx1, idx2, ..."
    idx_signature = ", ".join(idx_args)

    # String representations of advanced and basic indices for codegen
    adv_indices = [idxs[i] for i in adv_indices_pos]
    basic_indices = [idxs[i] for i in basic_indices_pos]

    to_tuple = create_tuple_string  # alias to make code more readable below

    # Number of trailing dimensions carried along by each scalar-index step.
    # Basic indices are always slices here, so `basic_indexed_x` keeps x's rank and the
    # advanced group consumes `len(adv_indices_pos)` of its dimensions.
    n_basic_dims = x.type.ndim - len(adv_indices_pos)
    trailing_idxs = [f"j{k}" for k in range(n_basic_dims)]
    buffer_idx = to_tuple(["i", *trailing_idxs])
    indexed_idx = to_tuple(["*scalar_idxs", *trailing_idxs])

    # Compute number of dimensions in advanced indices (after broadcasting)
    if len(adv_indices_pos) == 1:
        adv_idx = reconstructed_indices[adv_indices_pos[0]]
        adv_idx_ndim = adv_idx.ndim  # type: ignore[union-attr]
    else:
        # Multiple advanced indices - use max ndim (broadcast result ndim)
        adv_idx_ndim = max(reconstructed_indices[i].ndim for i in adv_indices_pos)  # type: ignore[union-attr]

    adv_loop_idxs = [f"a{k}" for k in range(adv_idx_ndim)]
    adv_pos = to_tuple(adv_loop_idxs)
    adv_scalar_names = [f"s{k}" for k in range(len(adv_indices))]
    nd_buffer_idx = to_tuple([*adv_loop_idxs, *trailing_idxs])
    nd_indexed_idx = to_tuple([*adv_scalar_names, *trailing_idxs])

    def advanced_loop_nest(assignment: str, shape: str) -> str:
        """Nest over the broadcast advanced-index shape, then the trailing dims.

        Each index view is read into a scalar at the advanced-position level, so the
        trailing loop has a loop-invariant base address and vectorizes. LLVM cannot hoist
        such a read on its own: Numba emits no TBAA, so it must assume the store aliases
        the index buffer.
        """
        lines = [
            f"{s} = {idx}[{adv_pos}]" for s, idx in zip(adv_scalar_names, adv_indices)
        ]
        lines.append(
            _scalar_loop_nest(assignment, trailing_idxs, shape, first_axis=adv_idx_ndim)
        )
        return _scalar_loop_nest("\n".join(lines), adv_loop_idxs, "adv_idx_shape")

    # Determine output position of advanced indexed dimensions
    # If advanced indices are consecutive, they go in the first advanced index position
    # Otherwise they go at the beginning
    if adv_indices_pos == tuple(range(adv_indices_pos[0], adv_indices_pos[-1] + 1)):
        # Consecutive - advanced dims will be at position of first advanced index
        out_adv_axis_pos = adv_indices_pos[0]
    else:
        # Non-consecutive - advanced dims go at the front
        out_adv_axis_pos = 0

    # Include trailing dimensions not covered by explicit indices
    explicit_basic_indices_pos = (
        *basic_indices_pos,
        *range(len(reconstructed_indices), x.type.ndim),
    )

    # Compute transpose to move advanced indexed dims to the front
    adv_axis_front_order = (*adv_indices_pos, *explicit_basic_indices_pos)
    adv_axis_front_transpose_needed = adv_axis_front_order != tuple(range(x.type.ndim))

    # Compute basic indices with "None" slices for dimensions that will be indexed by advanced indices
    basic_indices_with_none_slices = ", ".join(
        ":" for _ in range(len(adv_indices_pos))
    ) + (", " + ", ".join(basic_indices) if basic_indices else "")

    if isinstance(op, AdvancedSubtensor):
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
        if adv_idx_ndim == 1:
            copy_loop = _scalar_loop_nest(
                f"out_buffer[{buffer_idx}] = basic_indexed_x[{indexed_idx}]",
                trailing_idxs,
                "out_buffer.shape",
                first_axis=1,
            )
            gather_body = f"""
                # Create output buffer
                adv_idx_size = {adv_indices[0]}.size
                basic_idx_shape = basic_indexed_x.shape[{len(adv_indices)}:]
                out_buffer = np.empty((adv_idx_size, *basic_idx_shape), dtype=x.dtype)

                # Index over tuples of advanced indices and write to output buffer
                for i, scalar_idxs in enumerate(zip{to_tuple(adv_indices)}):
{indent(copy_loop, " " * 20)}
"""
        else:
            copy_loop = advanced_loop_nest(
                f"out_buffer[{nd_buffer_idx}] = basic_indexed_x[{nd_indexed_idx}]",
                "out_buffer.shape",
            )
            gather_body = f"""
                # Create output buffer, shaped like the advanced group
                adv_idx_shape = {adv_indices[0]}.shape
                basic_idx_shape = basic_indexed_x.shape[{len(adv_indices)}:]
                out_buffer = np.empty((*adv_idx_shape, *basic_idx_shape), dtype=x.dtype)

                # Walk the broadcast advanced shape, reading each index view by position
{indent(copy_loop, " " * 16)}
"""
        codegen += indent(
            dedent(
                gather_body
                + f"""
                # Move advanced output indexing group to its final position (if needed) and return
                return {f"out_buffer.transpose({final_axis_order})" if final_axis_transpose_needed else "out_buffer"}
                """
            ),
            " " * 4,
        )

    else:
        indexed_ndim = x[tuple(reconstructed_indices)].type.ndim

        # We transpose the advanced dimensions of x to the front for indexing
        # Apply the inverse permutation to the scalar indices when reading y.
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
            # Basic indexed dimensions follow the advanced dimensions.
            *(o for o in y_order if o not in y_adv_axis_front_order),
        )
        y_loop_idxs = (
            ["i", *trailing_idxs]
            if adv_idx_ndim == 1
            else [*adv_loop_idxs, *trailing_idxs]
        )
        if y.ndim == 0:
            y_prologue = "y_scalar = y.item()"
            update_value = "y_scalar"
        else:
            y_axis_order = tuple(y_adv_axis_front_order.index(axis) for axis in y_order)
            y_shape = to_tuple([f"update_shape[{axis}]" for axis in y_axis_order])
            y_prologue = ""
            update_value = _update_value(
                "y", y.type.shape, [y_loop_idxs[axis] for axis in y_axis_order]
            )

        func_name = f"{'set' if op.set_instead_of_inc else 'inc'}_advanced_integer_vector_indexing"
        codegen = dedent(
            f"""
            def {func_name}(x, y, {idx_signature}):
                {y_prologue}

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
        update = "=" if op.set_instead_of_inc else "+="
        if adv_idx_ndim == 1:
            update_loop = _scalar_loop_nest(
                f"basic_indexed_x[{indexed_idx}] {update} {update_value}",
                trailing_idxs,
                "update_shape",
                first_axis=1,
            )
            scatter_body = f"""
                # Index over tuples of advanced indices and update buffer
                for i, scalar_idxs in enumerate(zip{to_tuple(adv_indices)}):
{indent(update_loop, " " * 20)}
"""
        else:
            update_loop = advanced_loop_nest(
                f"basic_indexed_x[{nd_indexed_idx}] {update} {update_value}",
                "update_shape",
            )
            scatter_body = f"""
                # Walk the broadcast advanced shape, reading each index view by position
{indent(update_loop, " " * 16)}
"""
        update_code: list[str | CODE_TOKEN] = [
            CODE_TOKEN.INDENT,
            f"adv_idx_shape = {adv_indices[0]}.shape",
            f"basic_idx_shape = basic_indexed_x.shape[{len(adv_indices)}:]",
            "update_shape = (*adv_idx_shape, *basic_idx_shape)",
        ]
        if y.ndim:
            update_code.extend(
                [
                    "# Check broadcast compatibility; we work directly on y.",
                    f"np.broadcast_to(y, {y_shape})",
                ]
            )
        update_code.extend(dedent(scatter_body).strip("\n").splitlines())
        update_code.extend(
            [
                CODE_TOKEN.EMPTY_LINE,
                "# Return the original x, with the entries updated",
                "return x",
                CODE_TOKEN.DEDENT,
            ]
        )
        codegen += build_source_code(update_code)

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
