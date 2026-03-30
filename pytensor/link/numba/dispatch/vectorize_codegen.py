from __future__ import annotations

import base64
import operator
import pickle
from collections.abc import Callable, Sequence
from textwrap import indent
from typing import Any

import numba
import numpy as np
from llvmlite import ir
from numba import TypingError, types
from numba.core import cgutils
from numba.core.base import BaseContext
from numba.core.types.misc import NoneType
from numba.extending import overload
from numba.np import arrayobj

from pytensor.link.numba.cache import compile_numba_function_src
from pytensor.link.numba.dispatch import basic as numba_basic


# Numba is missing getitem(0d_array, Ellipsis), so o[...] += val fails.
# Register it so store_core_outputs can use o[...] += val naturally.
@overload(operator.getitem, inline="always")
def _getitem_0d_ellipsis(arr, idx):
    if (
        isinstance(arr, types.Array)
        and arr.ndim == 0
        and isinstance(idx, types.EllipsisType)
    ):

        def impl(arr, idx):
            return arr[()]

        return impl


def encode_literals(literals: Sequence) -> str:
    return base64.encodebytes(pickle.dumps(literals)).decode()


def store_core_outputs(
    core_op_fn: Callable, nin: int, nout: int, inc_outputs: frozenset = frozenset()
) -> Callable:
    """Create a Numba function that wraps a core function and stores its vectorized outputs.

    @njit
    def store_core_outputs(i0, i1, ..., in, o0, o1, ..., on):
        to0, to1, ..., ton = core_op_fn(i0, i1, ..., in)
        o0[...] = to0       # direct outputs
        o1[...] += to1      # inc outputs (indexed update)
        ...

    Parameters
    ----------
    inc_outputs : frozenset
        Output indices that use ``+=`` (increment) instead of ``=`` (assign).
        Used for indexed-update outputs in fused loops.
    """
    inputs = [f"i{i}" for i in range(nin)]
    outputs = [f"o{i}" for i in range(nout)]
    inner_outputs = [f"t{output}" for output in outputs]

    inp_signature = ", ".join(inputs)
    out_signature = ", ".join(outputs)
    inner_out_signature = ", ".join(inner_outputs)
    store_outputs = "\n".join(
        f"{output}[...] += {inner_output}"
        if i in inc_outputs
        else f"{output}[...] = {inner_output}"
        for i, (output, inner_output) in enumerate(
            zip(outputs, inner_outputs, strict=True)
        )
    )
    func_src = f"""
def store_core_outputs({inp_signature}, {out_signature}):
    {inner_out_signature} = core_op_fn({inp_signature})
{indent(store_outputs, " " * 4)}
"""
    global_env = {"core_op_fn": core_op_fn}

    func = compile_numba_function_src(
        func_src,
        "store_core_outputs",
        {**globals(), **global_env},
    )
    return numba_basic.numba_njit(func)


_jit_options = {
    "fastmath": {
        "arcp",  # Allow Reciprocal
        "contract",  # Allow floating-point contraction
        "afn",  # Approximate functions
        "reassoc",
        "nsz",  # TODO Do we want this one?
    },
    "no_cpython_wrapper": True,
    "no_cfunc_wrapper": True,
}


def _decode_literal(val, name):
    if not isinstance(val, types.Literal):
        raise TypingError(f"{name} must be literal.")
    return pickle.loads(base64.decodebytes(val.literal_value.encode()))


def _compute_vectorized_types(
    input_types,
    input_bc_patterns,
    output_bc_patterns,
    output_dtypes,
    inplace_pattern,
    allow_core_scalar,
    output_core_shape_types,
):
    """Compute core input/output types and return type for vectorized intrinsics."""
    batch_ndim = len(input_bc_patterns[0])

    if not all(isinstance(t, types.Array) for t in input_types):
        raise TypingError("Vectorized inputs must be arrays.")
    if not all(len(p) == batch_ndim for p in input_bc_patterns + output_bc_patterns):
        raise TypingError(
            "Vectorized broadcastable patterns must have the same length."
        )

    core_input_types = []
    for input_type, bc_pattern in zip(input_types, input_bc_patterns, strict=True):
        core_ndim = input_type.ndim - len(bc_pattern)
        if allow_core_scalar and core_ndim == 0:
            core_input_types.append(input_type.dtype)
        else:
            # FIXME: inheriting layout from the full array is wrong for F-order
            # inputs with batch dims — the core slice won't be F-contiguous.
            core_input_types.append(
                types.Array(
                    dtype=input_type.dtype, ndim=core_ndim, layout=input_type.layout
                )
            )

    core_out_types = [
        types.Array(numba.from_dtype(np.dtype(dtype)), len(output_core_shape), "C")
        for dtype, output_core_shape in zip(
            output_dtypes, output_core_shape_types, strict=True
        )
    ]

    out_types = [
        types.Array(
            numba.from_dtype(np.dtype(dtype)), batch_ndim + len(output_core_shape), "C"
        )
        for dtype, output_core_shape in zip(
            output_dtypes, output_core_shape_types, strict=True
        )
    ]

    for output_idx, inp_idx in inplace_pattern:
        output_type = input_types[inp_idx]
        core_out_types[output_idx] = types.Array(
            dtype=output_type.dtype,
            ndim=output_type.ndim - batch_ndim,
            layout=output_type.layout,
        )
        out_types[output_idx] = output_type

    ret_type = types.Tuple(out_types)
    if len(output_dtypes) == 1:
        ret_type = ret_type.types[0]

    return core_input_types, core_out_types, out_types, ret_type


def _codegen_return_outputs(
    ctx, builder, sig, outputs, inplace_pattern, extra_incref=frozenset()
):
    """Generate LLVM IR to return output arrays, handling incref for inplace.

    Parameters
    ----------
    extra_incref : frozenset
        Additional output indices (e.g. indexed-update outputs) that alias an input
        buffer and need an incref before returning, beyond inplace outputs.
    """
    incref_set = set(dict(inplace_pattern).keys()) | set(extra_incref)

    if len(outputs) == 1:
        if incref_set:
            ctx.nrt.incref(builder, sig.return_type, outputs[0]._getvalue())
        return outputs[0]._getvalue()

    for idx in sorted(incref_set):
        ctx.nrt.incref(
            builder,
            sig.return_type.types[idx],
            outputs[idx]._getvalue(),
        )
    return ctx.make_tuple(
        builder, sig.return_type, [out._getvalue() for out in outputs]
    )


NO_INDEXED_INPUTS = encode_literals(())
NO_INDEXED_OUTPUTS = encode_literals(())
NO_SIZE = None


def compute_itershape(
    ctx: BaseContext,
    builder: ir.IRBuilder,
    in_shapes: list[list[ir.Instruction]],
    broadcast_pattern: tuple[tuple[bool, ...], ...],
    size: list[ir.Instruction] | None,
):
    one = ir.IntType(64)(1)
    batch_ndim = len(broadcast_pattern[0])
    shape = [None] * batch_ndim
    if size is not None:
        shape = size
        for i in range(batch_ndim):
            for j, (bc, in_shape) in enumerate(
                zip(broadcast_pattern, in_shapes, strict=True)
            ):
                length = in_shape[i]
                if bc[i]:
                    with builder.if_then(
                        builder.icmp_unsigned("!=", length, one), likely=False
                    ):
                        msg = f"Vectorized input {j} is expected to have shape 1 in axis {i}"
                        ctx.call_conv.return_user_exc(builder, ValueError, (msg,))
                else:
                    with builder.if_then(
                        builder.icmp_unsigned("!=", length, shape[i]), likely=False
                    ):
                        with builder.if_else(
                            builder.icmp_unsigned("==", length, one)
                        ) as (
                            then,
                            otherwise,
                        ):
                            with then:
                                msg = (
                                    f"Incompatible vectorized shapes for input {j} and axis {i}. "
                                    f"Input {j} has shape 1, but is not statically "
                                    "known to have shape 1, and thus not broadcastable."
                                )
                                ctx.call_conv.return_user_exc(
                                    builder, ValueError, (msg,)
                                )
                            with otherwise:
                                msg = f"Vectorized input {j} has an incompatible shape in axis {i}."
                                ctx.call_conv.return_user_exc(
                                    builder, ValueError, (msg,)
                                )
    else:
        # Size is implied by the broadcast pattern
        for i in range(batch_ndim):
            for j, (bc, in_shape) in enumerate(
                zip(broadcast_pattern, in_shapes, strict=True)
            ):
                length = in_shape[i]
                if bc[i]:
                    with builder.if_then(
                        builder.icmp_unsigned("!=", length, one), likely=False
                    ):
                        msg = f"Vectorized input {j} is expected to have shape 1 in axis {i}"
                        ctx.call_conv.return_user_exc(builder, ValueError, (msg,))
                elif shape[i] is not None:
                    with builder.if_then(
                        builder.icmp_unsigned("!=", length, shape[i]), likely=False
                    ):
                        with builder.if_else(
                            builder.icmp_unsigned("==", length, one)
                        ) as (
                            then,
                            otherwise,
                        ):
                            with then:
                                msg = (
                                    f"Incompatible vectorized shapes for input {j} and axis {i}. "
                                    f"Input {j} has shape 1, but is not statically "
                                    "known to have shape 1, and thus not broadcastable."
                                )
                                ctx.call_conv.return_user_exc(
                                    builder, ValueError, (msg,)
                                )
                            with otherwise:
                                msg = f"Vectorized input {j} has an incompatible shape in axis {i}."
                                ctx.call_conv.return_user_exc(
                                    builder, ValueError, (msg,)
                                )
                else:
                    shape[i] = length
        for i in range(batch_ndim):
            if shape[i] is None:
                shape[i] = one
    return shape


def make_outputs(
    ctx: numba.core.base.BaseContext,
    builder: ir.IRBuilder,
    iter_shape: tuple[ir.Instruction, ...],
    out_bc: tuple[tuple[bool, ...], ...],
    dtypes: tuple[Any, ...],
    inplace: tuple[tuple[int, int], ...],
    inputs: tuple[Any, ...],
    input_types: tuple[Any, ...],
    output_core_shapes: tuple,
    update_outputs: dict | None = None,
) -> tuple[list[ir.Value], list[types.Array]]:
    """Allocate output arrays for vectorized loop.

    Parameters
    ----------
    update_outputs : dict, optional
        Mapping ``{output_idx: (array, array_type)}`` for outputs that reuse
        a scatter-target input buffer instead of being freshly allocated.
    """
    output_arrays = []
    output_arry_types = []
    one = ir.IntType(64)(1)
    inplace_dict = dict(inplace)
    for i, (core_shape, bc, dtype) in enumerate(
        zip(output_core_shapes, out_bc, dtypes, strict=True)
    ):
        if update_outputs is not None and i in update_outputs:
            output_arrays.append(update_outputs[i][0])
            output_arry_types.append(update_outputs[i][1])
            continue
        if i in inplace_dict:
            output_arrays.append(inputs[inplace_dict[i]])
            output_arry_types.append(input_types[inplace_dict[i]])
            # We need to incref once we return the inplace objects
            continue
        dtype = numba.from_dtype(np.dtype(dtype))
        output_ndim = len(iter_shape) + len(core_shape)
        arrtype = types.Array(dtype, output_ndim, "C")
        output_arry_types.append(arrtype)
        # This is actually an internal numba function, I guess we could
        # call `numba.nd.unsafe.ndarray` instead?
        batch_shape = [
            length if not bc_dim else one
            for length, bc_dim in zip(iter_shape, bc, strict=True)
        ]
        shape = batch_shape + core_shape
        array = arrayobj._empty_nd_impl(ctx, builder, arrtype, shape)
        output_arrays.append(array)

    # If there is no inplace or scatter operation, we know that all output
    # arrays don't alias. Informing llvm can make it easier to vectorize.
    if not inplace and not update_outputs:
        # The first argument is the output pointer
        arg = builder.function.args[0]
        arg.add_attribute("noalias")
    return output_arrays, output_arry_types


def make_loop_call(
    typingctx,
    context: numba.core.base.BaseContext,
    builder: ir.IRBuilder,
    core_func: Any,
    core_signature: types.FunctionType,
    iter_shape: tuple[ir.Instruction, ...],
    constant_inputs: tuple[ir.Instruction, ...],
    inputs: tuple[ir.Instruction, ...],
    outputs: tuple[ir.Instruction, ...],
    input_bc: tuple[tuple[bool, ...], ...],
    output_bc: tuple[tuple[bool, ...], ...],
    input_types: tuple[Any, ...],
    output_types: tuple[Any, ...],
    core_scalar: bool = True,
    input_read_spec: tuple[tuple[int, int] | None, ...] | None = None,
    idx_arrs: list | None = None,
    idx_types: tuple | None = None,
    idx_load_axis: tuple[int, ...] | None = None,
    idx_bc: tuple[tuple[bool, ...], ...] | None = None,
    output_update_spec: tuple[tuple[int, int] | None, ...] | None = None,
):
    safe = (False, False)

    n_outputs = len(outputs)

    # TODO I think this is better than the noalias attribute
    # for the input, but self_ref isn't supported in a released
    # llvmlite version yet
    # mod = builder.module
    # domain = mod.add_metadata([], self_ref=True)
    # input_scope = mod.add_metadata([domain], self_ref=True)
    # output_scope = mod.add_metadata([domain], self_ref=True)
    # input_scope_set = mod.add_metadata([input_scope, output_scope])
    # output_scope_set = mod.add_metadata([input_scope, output_scope])

    zero = ir.Constant(ir.IntType(64), 0)

    def _wrap_negative_index(idx_val, dim_size, signed):
        """Wrap a negative index by adding the dimension size: idx + size if idx < 0.

        Only emits the branch for signed index dtypes; unsigned indices are
        returned as-is since they cannot be negative.
        """
        if not signed:
            return idx_val
        is_neg = builder.icmp_signed("<", idx_val, zero)
        wrapped = builder.add(idx_val, dim_size)
        return builder.select(is_neg, wrapped, idx_val)

    # Setup loops and initialize accumulators for outputs
    # This part corresponds to opening the loops
    loop_stack = []
    loops = []
    output_accumulator: list[tuple[Any | None, int | None]] = [(None, None)] * n_outputs
    for dim, length in enumerate(iter_shape):
        # Find outputs that only have accumulations left
        for output in range(n_outputs):
            if output_accumulator[output][0] is not None:
                continue
            if all(output_bc[output][dim:]):
                value = outputs[output][0].type.pointee(0)
                accu = cgutils.alloca_once_value(builder, value)
                output_accumulator[output] = (accu, dim)

        loop = cgutils.for_range(builder, length)
        loop_stack.append(loop)
        loops.append(loop.__enter__())

    # Code in the inner most loop...
    idxs = [loopval.index for loopval in loops]

    # Load indirect indices for all index arrays.
    # Each index array is 1D and is accessed by the loop counter for its axis.
    indirect_idxs = []
    if idx_arrs is not None and idx_load_axis is not None:
        for gi_k, (gi_arr, gi_type, ax) in enumerate(
            zip(idx_arrs, idx_types, idx_load_axis)
        ):
            # Use zero if the index is statically broadcastable, loop counter otherwise
            idx_is_bc = idx_bc[gi_k][0] if idx_bc and idx_bc[gi_k] else False
            load_idx = zero if idx_is_bc else idxs[ax]
            gi_ptr = cgutils.get_item_pointer2(
                context,
                builder,
                gi_arr.data,
                cgutils.unpack_tuple(builder, gi_arr.shape),
                cgutils.unpack_tuple(builder, gi_arr.strides),
                gi_type.layout,
                [load_idx],
                False,
                False,
            )
            val = builder.load(gi_ptr)
            # Extend to i64 to match stride types in get_item_pointer2.
            i64 = ir.IntType(64)
            if val.type != i64:
                if gi_type.dtype.signed:
                    val = builder.sext(val, i64)
                else:
                    val = builder.zext(val, i64)
            # Negative indices are wrapped at point of use (see
            # _wrap_negative_index) since the dimension size depends on the
            # source/target array being indexed.
            indirect_idxs.append(val)

    # Load values from input arrays
    input_vals = []
    for input_i, (input, input_type, bc) in enumerate(
        zip(inputs, input_types, input_bc, strict=True)
    ):
        spec = input_read_spec[input_i] if input_read_spec is not None else None
        core_ndim = input_type.ndim - len(bc)

        if spec is not None:
            assert idx_types is not None
            # Single-index on one axis: replace that axis with the indirect index
            indexed_axes = {src_axis: idx_k for idx_k, src_axis in spec}
            input_shape = cgutils.unpack_tuple(builder, input.shape)
            idxs_bc = []
            result_dim = 0
            for src_dim in range(input_type.ndim):
                if src_dim in indexed_axes:
                    idx_k = indexed_axes[src_dim]
                    idx_val = _wrap_negative_index(
                        indirect_idxs[idx_k],
                        input_shape[src_dim],
                        signed=idx_types[idx_k].dtype.signed,
                    )
                    idxs_bc.append(idx_val)
                    result_dim += 1
                else:
                    if result_dim < len(bc):
                        idxs_bc.append(zero if bc[result_dim] else idxs[result_dim])
                    else:
                        idxs_bc.append(zero)
                    result_dim += 1
        else:
            idxs_bc = [
                zero if bc else idx for idx, bc in zip(idxs, bc, strict=True)
            ] + [zero] * core_ndim
        ptr = cgutils.get_item_pointer2(
            context,
            builder,
            input.data,
            cgutils.unpack_tuple(builder, input.shape),
            cgutils.unpack_tuple(builder, input.strides),
            input_type.layout,
            idxs_bc,
            *safe,
        )
        if core_scalar and core_ndim == 0:
            # Retrive scalar item at index
            val = builder.load(ptr)
            # val.set_metadata("alias.scope", input_scope_set)
            # val.set_metadata("noalias", output_scope_set)
        else:
            # Retrieve array item at index
            # This is a streamlined version of Numba's `GUArrayArg.load`
            # TODO check layout arg!
            core_arry_type = types.Array(
                dtype=input_type.dtype, ndim=core_ndim, layout=input_type.layout
            )
            core_array = context.make_array(core_arry_type)(context, builder)
            core_shape = cgutils.unpack_tuple(builder, input.shape)[
                input_type.ndim - core_ndim :
            ]
            core_strides = cgutils.unpack_tuple(builder, input.strides)[
                input_type.ndim - core_ndim :
            ]
            itemsize = context.get_abi_sizeof(context.get_data_type(input_type.dtype))
            context.populate_array(
                core_array,
                # TODO whey do we need to bitcast?
                data=builder.bitcast(ptr, core_array.data.type),
                shape=core_shape,
                strides=core_strides,
                itemsize=context.get_constant(types.intp, itemsize),
                # TODO what is meminfo about?
                meminfo=None,
            )
            val = core_array._getvalue()

        input_vals.append(val)

    # Create output slices to pass to inner func
    output_slices = []
    for output_i, (output, output_type, bc) in enumerate(
        zip(outputs, output_types, output_bc, strict=True)
    ):
        core_ndim = output_type.ndim - len(bc)
        size_type = output.shape.type.element  # pyright: ignore[reportAttributeAccessIssue]
        output_shape = cgutils.unpack_tuple(builder, output.shape)  # pyright: ignore[reportAttributeAccessIssue]
        output_strides = cgutils.unpack_tuple(builder, output.strides)  # pyright: ignore[reportAttributeAccessIssue]

        spec = output_update_spec[output_i] if output_update_spec is not None else None
        if spec is not None:
            assert idx_types is not None
            # Indexed-update output: same logic as indexed-read input
            indexed_axes = {src_axis: idx_k for idx_k, src_axis in spec}
            n_indexed = len(indexed_axes)
            source_batch_ndim = len(bc) + n_indexed - 1
            idxs_bc = []
            loop_dim = 0
            for src_dim in range(source_batch_ndim):
                if src_dim in indexed_axes:
                    idx_k = indexed_axes[src_dim]
                    idx_val = _wrap_negative_index(
                        indirect_idxs[idx_k],
                        output_shape[src_dim],
                        signed=idx_types[idx_k].dtype.signed,
                    )
                    idxs_bc.append(idx_val)
                else:
                    bc_dim = bc[loop_dim] if loop_dim < len(bc) else False
                    idxs_bc.append(zero if bc_dim else idxs[loop_dim])
                    loop_dim += 1
            idxs_bc += [zero] * core_ndim
        else:
            idxs_bc = [
                zero if bc else idx for idx, bc in zip(idxs, bc, strict=True)
            ] + [zero] * core_ndim
        ptr = cgutils.get_item_pointer2(
            context,
            builder,
            output.data,
            output_shape,
            output_strides,
            output_type.layout,
            idxs_bc,
            *safe,
        )
        # Retrieve array item at index
        # This is a streamlined version of Numba's `GUArrayArg.load`
        core_arry_type = types.Array(
            dtype=output_type.dtype, ndim=core_ndim, layout=output_type.layout
        )
        core_array = context.make_array(core_arry_type)(context, builder)
        core_shape = output_shape[-core_ndim:] if core_ndim > 0 else []
        core_strides = output_strides[-core_ndim:] if core_ndim > 0 else []
        itemsize = context.get_abi_sizeof(context.get_data_type(output_type.dtype))
        context.populate_array(
            core_array,
            # TODO whey do we need to bitcast?
            data=builder.bitcast(ptr, core_array.data.type),
            shape=cgutils.pack_array(builder, core_shape, ty=size_type),
            strides=cgutils.pack_array(builder, core_strides, ty=size_type),
            itemsize=context.get_constant(types.intp, itemsize),
            # TODO what is meminfo about?
            meminfo=None,
        )
        val = core_array._getvalue()
        output_slices.append(val)

    inner_codegen = context.get_function(core_func, core_signature)

    if isinstance(core_signature.args[0], types.StarArgTuple | types.StarArgUniTuple):
        input_vals = [context.make_tuple(builder, core_signature.args[0], input_vals)]

    inner_codegen(builder, [*constant_inputs, *input_vals, *output_slices])

    # Close the loops
    for depth, loop in enumerate(loop_stack[::-1]):
        loop.__exit__(None, None, None)

    return


@numba.extending.intrinsic(jit_options=_jit_options, prefer_literal=True)
def _vectorized(
    typingctx,
    core_func,
    input_bc_patterns,
    output_bc_patterns,
    output_dtypes,
    inplace_pattern,
    allow_core_scalar,
    constant_inputs_types,
    outer_input_types,
    output_core_shape_types,
    size_type,
    indexed_inputs,
    indexed_outputs,
):
    """Like _vectorized but with indirect indexing for reads and updates.

    Outer inputs are ordered as
    ``[elemwise_inputs..., idx_0, idx_1, ..., update_target_0, ...]``.

    ``indexed_inputs`` groups elemwise input positions by which index they
    read through: e.g. ``((0, 2), (1,))`` means idx_0 reads inputs 0 and 2,
    idx_1 reads input 1.  Entries may be empty ``()`` for update-only indices.

    ``indexed_outputs`` has one entry per index array (same length as
    ``indexed_inputs``).  ``None`` means that index is not used for updates.
    ``((out_0, out_1), mode)`` means that index updates outputs out_0 and
    out_1 with *mode* ``"set"`` or ``"inc"``.

    Parameters
    ----------
    outer_input_types : tuple of Array types
        ``(elemwise_input_0, ..., elemwise_input_N, idx_0, ..., update_target_0, ...)``
    indexed_inputs : literal str
        Encoded ``tuple[tuple[int, ...], ...]``.
    indexed_outputs : literal str
        Encoded ``tuple[tuple[tuple[int, ...], str] | None, ...]``.
    """
    arg_types = [
        core_func,
        input_bc_patterns,
        output_bc_patterns,
        output_dtypes,
        inplace_pattern,
        allow_core_scalar,
        constant_inputs_types,
        outer_input_types,
        output_core_shape_types,
        size_type,
        indexed_inputs,
        indexed_outputs,
    ]

    input_bc_patterns = _decode_literal(input_bc_patterns, "input_bc_patterns")
    output_bc_patterns = _decode_literal(output_bc_patterns, "output_bc_patterns")
    output_dtypes = _decode_literal(output_dtypes, "output_dtypes")
    inplace_pattern = _decode_literal(inplace_pattern, "inplace_pattern")
    indexed_inputs = _decode_literal(indexed_inputs, "indexed_inputs")
    indexed_outputs = _decode_literal(indexed_outputs, "indexed_outputs")

    if not isinstance(allow_core_scalar, types.Literal):
        raise TypingError("allow_core_scalar must be literal.")
    allow_core_scalar = allow_core_scalar.literal_value

    # Count scatter targets (one per scattered output)
    n_update_targets = sum(
        len(entry[0]) for entry in indexed_outputs if entry is not None
    )

    n_indices = len(indexed_inputs)
    n_elemwise = len(outer_input_types) - n_indices - n_update_targets
    input_types = tuple(outer_input_types[i] for i in range(n_elemwise))
    idx_types = tuple(outer_input_types[n_elemwise + k] for k in range(n_indices))
    update_target_types = tuple(
        outer_input_types[n_elemwise + n_indices + j] for j in range(n_update_targets)
    )

    # indexed_inputs entries are (positions, axis) — one per index array.
    # We aggregate per-input into a tuple of (idx_k, src_axis) pairs.
    #
    # idx_load_axis[k] = which loop dim loads index array k.
    _read_spec_dict: dict[int, list[tuple[int, int]]] = {}
    idx_load_axis = []
    idx_bc_list = []  # per index array: broadcastable tuple
    for k, entry in enumerate(indexed_inputs):
        positions, axis = entry[0], entry[1]
        idx_bc = entry[2] if len(entry) > 2 else (False,)
        idx_bc_list.append(idx_bc)
        for p in positions:
            _read_spec_dict.setdefault(p, []).append((k, axis))
    # idx_load_axis[k] = which loop dim to use when loading index array k.
    for k, entry in enumerate(indexed_inputs):
        _positions, axis = entry[0], entry[1]
        idx_load_axis.append(axis)
    input_read_spec = tuple(
        tuple(_read_spec_dict[p]) if p in _read_spec_dict else None
        for p in range(n_elemwise)
    )
    idx_load_axis = tuple(idx_load_axis)

    # Per-output: tuple of (idx_k, axis) pairs, or None.
    # Same format as input_read_spec.
    # indexed_outputs entries are (positions, mode, axis) or None.
    _update_spec_dict: dict[int, list[tuple[int, int]]] = {}
    update_out_to_target = {}
    update_out_indices = set()
    target_counter = 0
    for k, entry in enumerate(indexed_outputs):
        if entry is None:
            continue
        output_indices, _mode, axis = entry
        for out_idx in output_indices:
            _update_spec_dict.setdefault(out_idx, []).append((k, axis))
            update_out_to_target[out_idx] = target_counter
            update_out_indices.add(out_idx)
            target_counter += 1
    output_update_spec = tuple(
        tuple(_update_spec_dict[p]) if p in _update_spec_dict else None
        for p in range(len(output_bc_patterns))
    )
    update_out_indices = frozenset(update_out_indices)

    core_input_types, core_out_types, out_types, ret_type = _compute_vectorized_types(
        input_types,
        input_bc_patterns,
        output_bc_patterns,
        output_dtypes,
        inplace_pattern,
        allow_core_scalar,
        output_core_shape_types,
    )

    # Fix up output types for scattered outputs: they match the target buffer
    if update_out_to_target:
        core_out_types = list(core_out_types)
        out_types = list(out_types)
        batch_ndim = len(input_bc_patterns[0])
        for out_idx, target_idx in update_out_to_target.items():
            target_type = update_target_types[target_idx]
            out_types[out_idx] = target_type
            core_out_types[out_idx] = types.Array(
                dtype=target_type.dtype,
                ndim=target_type.ndim - batch_ndim,
                layout=target_type.layout,
            )
        out_types = tuple(out_types)
        core_out_types = tuple(core_out_types)

        if len(out_types) == 1:
            ret_type = out_types[0]
        else:
            ret_type = types.Tuple(out_types)

    sig = ret_type(*arg_types)

    size_is_none = isinstance(size_type, NoneType)

    # Save values for codegen closure
    input_bc_patterns_val = input_bc_patterns
    output_bc_patterns_val = output_bc_patterns
    output_dtypes_val = output_dtypes
    inplace_pattern_val = inplace_pattern
    input_read_spec_val = input_read_spec
    idx_types_val = idx_types
    idx_load_axis_val = idx_load_axis
    idx_bc_list_val = idx_bc_list
    output_update_spec_val = output_update_spec
    update_out_to_target_val = update_out_to_target
    update_target_types_val = update_target_types
    update_out_indices_val = update_out_indices

    def codegen(ctx, builder, sig, args):
        [
            _,
            _,
            _,
            _,
            _,
            _,
            constant_inputs,
            outer_inputs,
            output_core_shapes,
            size,
            _,
            _,
        ] = args

        constant_inputs = cgutils.unpack_tuple(builder, constant_inputs)
        all_outer = cgutils.unpack_tuple(builder, outer_inputs)
        output_core_shapes = [
            cgutils.unpack_tuple(builder, shape)
            for shape in cgutils.unpack_tuple(builder, output_core_shapes)
        ]
        size = None if size_is_none else cgutils.unpack_tuple(builder, size)

        # First n_elemwise outer inputs are elemwise inputs (source arrays)
        inputs = [
            arrayobj.make_array(input_types[i])(ctx, builder, all_outer[i])
            for i in range(n_elemwise)
        ]
        in_shapes = [cgutils.unpack_tuple(builder, obj.shape) for obj in inputs]

        # Next n_indices inputs are index arrays
        idx_arrs = [
            arrayobj.make_array(idx_types_val[k])(
                ctx, builder, all_outer[n_elemwise + k]
            )
            for k in range(n_indices)
        ]

        # Remaining inputs are scatter target buffers
        update_target_arrs = [
            arrayobj.make_array(update_target_types_val[j])(
                ctx, builder, all_outer[n_elemwise + n_indices + j]
            )
            for j in range(n_update_targets)
        ]

        # Build iter_shapes for compute_itershape.
        # For indexed inputs, substitute the index array length on the
        # indexed axis of the source shape.
        iter_shapes = list(in_shapes)
        iter_bc = list(input_bc_patterns_val)
        idx_shapes = [
            cgutils.unpack_tuple(builder, idx_arrs[k].shape) for k in range(n_indices)
        ]
        for p, spec in enumerate(input_read_spec_val):
            if spec is None:
                continue
            # Single-index: substitute on the indexed axis
            idx_k, axis = spec[0]
            iter_shapes[p] = list(iter_shapes[p])
            iter_shapes[p][axis] = idx_shapes[idx_k][0]

        # Each index array participates in iter_shape validation on its load
        # axis, just like any direct input.
        #
        # Read indices use their static broadcastable so shape-1 indices
        # broadcast correctly against other inputs.
        #
        # Write indices are forced to bc=False: unlike reads, numpy does
        # not allow the index to broadcast against the update value.  A
        # shape-1 write index at runtime must match a size-1 loop, not
        # silently repeat writes to the same target position.
        batch_ndim = len(input_bc_patterns_val[0]) if input_bc_patterns_val else 0
        one = ir.IntType(64)(1)
        for k in range(n_indices):
            ax = idx_load_axis_val[k]
            is_write = indexed_outputs[k] is not None
            idx_shape_entry = [one] * batch_ndim
            idx_shape_entry[ax] = idx_shapes[k][0]
            iter_shapes.append(idx_shape_entry)
            idx_bc_on_ax = idx_bc_list_val[k][0] if idx_bc_list_val[k] else False
            if is_write:
                idx_bc_on_ax = False
            iter_bc.append(
                tuple(True if d != ax else idx_bc_on_ax for d in range(batch_ndim))
            )

        iter_shape = compute_itershape(
            ctx,
            builder,
            iter_shapes,
            tuple(iter_bc),
            size,
        )

        # Build update_outputs dict for make_outputs: out_idx -> (array, type)
        update_outputs_dict = (
            {
                out_idx: (
                    update_target_arrs[target_idx],
                    update_target_types_val[target_idx],
                )
                for out_idx, target_idx in update_out_to_target_val.items()
            }
            if update_out_to_target_val
            else None
        )

        outputs, output_types = make_outputs(
            ctx,
            builder,
            iter_shape,
            output_bc_patterns_val,
            output_dtypes_val,
            inplace_pattern_val,
            inputs,
            input_types,
            output_core_shapes,
            update_outputs=update_outputs_dict,
        )

        core_signature = typingctx.resolve_function_type(
            core_func,
            [
                *constant_inputs_types,
                *core_input_types,
                *core_out_types,
            ],
            {},
        )

        make_loop_call(
            typingctx,
            ctx,
            builder,
            core_func,
            core_signature,
            iter_shape,
            constant_inputs,
            inputs,
            outputs,
            input_bc_patterns_val,
            output_bc_patterns_val,
            input_types,
            output_types,
            core_scalar=allow_core_scalar,
            input_read_spec=input_read_spec_val,
            idx_arrs=idx_arrs,
            idx_types=idx_types_val,
            idx_load_axis=idx_load_axis_val,
            idx_bc=idx_bc_list_val,
            output_update_spec=output_update_spec_val,
        )

        return _codegen_return_outputs(
            ctx,
            builder,
            sig,
            outputs,
            inplace_pattern,
            extra_incref=update_out_indices_val,
        )

    return sig, codegen
