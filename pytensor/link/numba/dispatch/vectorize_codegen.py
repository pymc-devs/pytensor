from __future__ import annotations

import base64
import pickle
from collections.abc import Sequence
from typing import Any

import numba
import numpy as np
from llvmlite import ir
from numba import TypingError, types
from numba.core import cgutils
from numba.core.base import BaseContext
from numba.np import arrayobj


def encode_literals(literals: Sequence) -> str:
    return base64.encodebytes(pickle.dumps(literals)).decode()


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


@numba.extending.intrinsic(jit_options=_jit_options, prefer_literal=True)
def _vectorized(
    typingctx,
    scalar_func,
    input_bc_patterns,
    output_bc_patterns,
    output_dtypes,
    inplace_pattern,
    inputs,
):
    arg_types = [
        scalar_func,
        input_bc_patterns,
        output_bc_patterns,
        output_dtypes,
        inplace_pattern,
        inputs,
    ]

    if not isinstance(input_bc_patterns, types.Literal):
        raise TypingError("input_bc_patterns must be literal.")
    input_bc_patterns = input_bc_patterns.literal_value
    input_bc_patterns = pickle.loads(base64.decodebytes(input_bc_patterns.encode()))

    if not isinstance(output_bc_patterns, types.Literal):
        raise TypeError("output_bc_patterns must be literal.")
    output_bc_patterns = output_bc_patterns.literal_value
    output_bc_patterns = pickle.loads(base64.decodebytes(output_bc_patterns.encode()))

    if not isinstance(output_dtypes, types.Literal):
        raise TypeError("output_dtypes must be literal.")
    output_dtypes = output_dtypes.literal_value
    output_dtypes = pickle.loads(base64.decodebytes(output_dtypes.encode()))

    if not isinstance(inplace_pattern, types.Literal):
        raise TypeError("inplace_pattern must be literal.")
    inplace_pattern = inplace_pattern.literal_value
    inplace_pattern = pickle.loads(base64.decodebytes(inplace_pattern.encode()))

    n_outputs = len(output_bc_patterns)

    if not len(inputs) > 0:
        raise TypingError("Empty argument list to elemwise op.")

    if not n_outputs > 0:
        raise TypingError("Empty list of outputs for elemwise op.")

    if not all(isinstance(input, types.Array) for input in inputs):
        raise TypingError("Inputs to elemwise must be arrays.")
    ndim = inputs[0].ndim

    if not all(input.ndim == ndim for input in inputs):
        raise TypingError("Inputs to elemwise must have the same rank.")

    if not all(len(pattern) == ndim for pattern in output_bc_patterns):
        raise TypingError("Invalid output broadcasting pattern.")

    scalar_signature = typingctx.resolve_function_type(
        scalar_func, [in_type.dtype for in_type in inputs], {}
    )

    # So we can access the constant values in codegen...
    input_bc_patterns_val = input_bc_patterns
    output_bc_patterns_val = output_bc_patterns
    output_dtypes_val = output_dtypes
    inplace_pattern_val = inplace_pattern
    input_types = inputs

    def codegen(
        ctx,
        builder,
        sig,
        args,
    ):
        [_, _, _, _, _, inputs] = args
        inputs = cgutils.unpack_tuple(builder, inputs)
        inputs = [
            arrayobj.make_array(ty)(ctx, builder, val)
            for ty, val in zip(input_types, inputs)
        ]
        in_shapes = [cgutils.unpack_tuple(builder, obj.shape) for obj in inputs]

        iter_shape = compute_itershape(
            ctx,
            builder,
            in_shapes,
            input_bc_patterns_val,
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
        )

        make_loop_call(
            typingctx,
            ctx,
            builder,
            scalar_func,
            scalar_signature,
            iter_shape,
            inputs,
            outputs,
            input_bc_patterns_val,
            output_bc_patterns_val,
            input_types,
            output_types,
        )

        if len(outputs) == 1:
            if inplace_pattern:
                assert inplace_pattern[0][0] == 0
                ctx.nrt.incref(builder, sig.return_type, outputs[0]._getvalue())
            return outputs[0]._getvalue()

        for inplace_idx in dict(inplace_pattern):
            ctx.nrt.incref(
                builder,
                sig.return_type.types[inplace_idx],
                outputs[inplace_idx]._get_value(),
            )
        return ctx.make_tuple(
            builder, sig.return_type, [out._getvalue() for out in outputs]
        )

    ret_types = [
        types.Array(numba.from_dtype(np.dtype(dtype)), ndim, "C")
        for dtype in output_dtypes
    ]

    for output_idx, input_idx in inplace_pattern:
        ret_types[output_idx] = input_types[input_idx]

    ret_type = types.Tuple(ret_types)

    if len(output_dtypes) == 1:
        ret_type = ret_type.types[0]
    sig = ret_type(*arg_types)

    return sig, codegen


def compute_itershape(
    ctx: BaseContext,
    builder: ir.IRBuilder,
    in_shapes: tuple[ir.Instruction, ...],
    broadcast_pattern: tuple[tuple[bool, ...], ...],
):
    one = ir.IntType(64)(1)
    ndim = len(in_shapes[0])
    shape = [None] * ndim
    for i in range(ndim):
        for j, (bc, in_shape) in enumerate(zip(broadcast_pattern, in_shapes)):
            length = in_shape[i]
            if bc[i]:
                with builder.if_then(
                    builder.icmp_unsigned("!=", length, one), likely=False
                ):
                    msg = (
                        f"Input {j} to elemwise is expected to have shape 1 in axis {i}"
                    )
                    ctx.call_conv.return_user_exc(builder, ValueError, (msg,))
            elif shape[i] is not None:
                with builder.if_then(
                    builder.icmp_unsigned("!=", length, shape[i]), likely=False
                ):
                    with builder.if_else(builder.icmp_unsigned("==", length, one)) as (
                        then,
                        otherwise,
                    ):
                        with then:
                            msg = (
                                f"Incompatible shapes for input {j} and axis {i} of "
                                f"elemwise. Input {j} has shape 1, but is not statically "
                                "known to have shape 1, and thus not broadcastable."
                            )
                            ctx.call_conv.return_user_exc(builder, ValueError, (msg,))
                        with otherwise:
                            msg = (
                                f"Input {j} to elemwise has an incompatible "
                                f"shape in axis {i}."
                            )
                            ctx.call_conv.return_user_exc(builder, ValueError, (msg,))
            else:
                shape[i] = length
    for i in range(ndim):
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
):
    arrays = []
    ar_types: list[types.Array] = []
    one = ir.IntType(64)(1)
    inplace_dict = dict(inplace)
    for i, (bc, dtype) in enumerate(zip(out_bc, dtypes)):
        if i in inplace_dict:
            arrays.append(inputs[inplace_dict[i]])
            ar_types.append(input_types[inplace_dict[i]])
            # We need to incref once we return the inplace objects
            continue
        dtype = numba.from_dtype(np.dtype(dtype))
        arrtype = types.Array(dtype, len(iter_shape), "C")
        ar_types.append(arrtype)
        # This is actually an internal numba function, I guess we could
        # call `numba.nd.unsafe.ndarray` instead?
        shape = [
            length if not bc_dim else one for length, bc_dim in zip(iter_shape, bc)
        ]
        array = arrayobj._empty_nd_impl(ctx, builder, arrtype, shape)
        arrays.append(array)

    # If there is no inplace operation, we know that all output arrays
    # don't alias. Informing llvm can make it easier to vectorize.
    if not inplace:
        # The first argument is the output pointer
        arg = builder.function.args[0]
        arg.add_attribute("noalias")
    return arrays, ar_types


def make_loop_call(
    typingctx,
    context: numba.core.base.BaseContext,
    builder: ir.IRBuilder,
    scalar_func: Any,
    scalar_signature: types.FunctionType,
    iter_shape: tuple[ir.Instruction, ...],
    inputs: tuple[ir.Instruction, ...],
    outputs: tuple[ir.Instruction, ...],
    input_bc: tuple[tuple[bool, ...], ...],
    output_bc: tuple[tuple[bool, ...], ...],
    input_types: tuple[Any, ...],
    output_types: tuple[Any, ...],
):
    safe = (False, False)
    n_outputs = len(outputs)

    # context.printf(builder, "iter shape: " + ', '.join(["%i"] * len(iter_shape)) + "\n", *iter_shape)

    # Extract shape and stride information from the array.
    # For later use in the loop body to do the indexing
    def extract_array(aryty, obj):
        shape = cgutils.unpack_tuple(builder, obj.shape)
        strides = cgutils.unpack_tuple(builder, obj.strides)
        data = obj.data
        layout = aryty.layout
        return (data, shape, strides, layout)

    # TODO I think this is better than the noalias attribute
    # for the input, but self_ref isn't supported in a released
    # llvmlite version yet
    # mod = builder.module
    # domain = mod.add_metadata([], self_ref=True)
    # input_scope = mod.add_metadata([domain], self_ref=True)
    # output_scope = mod.add_metadata([domain], self_ref=True)
    # input_scope_set = mod.add_metadata([input_scope, output_scope])
    # output_scope_set = mod.add_metadata([input_scope, output_scope])

    inputs = tuple(extract_array(aryty, ary) for aryty, ary in zip(input_types, inputs))

    outputs = tuple(
        extract_array(aryty, ary) for aryty, ary in zip(output_types, outputs)
    )

    zero = ir.Constant(ir.IntType(64), 0)

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

    # Load values from input arrays
    input_vals = []
    for array_info, bc in zip(inputs, input_bc):
        idxs_bc = [zero if bc else idx for idx, bc in zip(idxs, bc)]
        ptr = cgutils.get_item_pointer2(context, builder, *array_info, idxs_bc, *safe)
        val = builder.load(ptr)
        # val.set_metadata("alias.scope", input_scope_set)
        # val.set_metadata("noalias", output_scope_set)
        input_vals.append(val)

    inner_codegen = context.get_function(scalar_func, scalar_signature)

    if isinstance(scalar_signature.args[0], types.StarArgTuple | types.StarArgUniTuple):
        input_vals = [context.make_tuple(builder, scalar_signature.args[0], input_vals)]
    output_values = inner_codegen(builder, input_vals)

    if isinstance(scalar_signature.return_type, types.Tuple | types.UniTuple):
        output_values = cgutils.unpack_tuple(builder, output_values)
        func_output_types = scalar_signature.return_type.types
    else:
        output_values = [output_values]
        func_output_types = [scalar_signature.return_type]

    # Update output value or accumulators respectively
    for i, ((accu, _), value) in enumerate(zip(output_accumulator, output_values)):
        if accu is not None:
            load = builder.load(accu)
            # load.set_metadata("alias.scope", output_scope_set)
            # load.set_metadata("noalias", input_scope_set)
            new_value = builder.fadd(load, value)
            builder.store(new_value, accu)
            # TODO belongs to noalias scope
            # store.set_metadata("alias.scope", output_scope_set)
            # store.set_metadata("noalias", input_scope_set)
        else:
            idxs_bc = [zero if bc else idx for idx, bc in zip(idxs, output_bc[i])]
            ptr = cgutils.get_item_pointer2(context, builder, *outputs[i], idxs_bc)
            # store = builder.store(value, ptr)
            value = context.cast(
                builder, value, func_output_types[i], output_types[i].dtype
            )
            arrayobj.store_item(context, builder, output_types[i], value, ptr)
            # store.set_metadata("alias.scope", output_scope_set)
            # store.set_metadata("noalias", input_scope_set)

    # Close the loops and write accumulator values to the output arrays
    for depth, loop in enumerate(loop_stack[::-1]):
        for output, (accu, accu_depth) in enumerate(output_accumulator):
            if accu_depth == depth:
                idxs_bc = [
                    zero if bc else idx for idx, bc in zip(idxs, output_bc[output])
                ]
                ptr = cgutils.get_item_pointer2(
                    context, builder, *outputs[output], idxs_bc
                )
                load = builder.load(accu)
                # load.set_metadata("alias.scope", output_scope_set)
                # load.set_metadata("noalias", input_scope_set)
                # store = builder.store(load, ptr)
                load = context.cast(
                    builder, load, func_output_types[output], output_types[output].dtype
                )
                arrayobj.store_item(context, builder, output_types[output], load, ptr)
                # store.set_metadata("alias.scope", output_scope_set)
                # store.set_metadata("noalias", input_scope_set)
        loop.__exit__(None, None, None)

    return
