from __future__ import annotations

import base64
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
from numba.np import arrayobj

from pytensor.link.numba.cache import compile_numba_function_src
from pytensor.link.numba.dispatch import basic as numba_basic


def encode_literals(literals: Sequence) -> str:
    return base64.encodebytes(pickle.dumps(literals)).decode()


def store_core_outputs(core_op_fn: Callable, nin: int, nout: int) -> Callable:
    """Create a Numba function that wraps a core function and stores its vectorized outputs.

    @njit
    def store_core_outputs(i0, i1, ..., in, o0, o1, ..., on):
        to0, to1, ..., ton = core_op_fn(i0, i1, ..., in)
        o0[...] = to0
        o1[...] = to1
        ...
        on[...] = ton

    """
    inputs = [f"i{i}" for i in range(nin)]
    outputs = [f"o{i}" for i in range(nout)]
    inner_outputs = [f"t{output}" for output in outputs]

    inp_signature = ", ".join(inputs)
    out_signature = ", ".join(outputs)
    inner_out_signature = ", ".join(inner_outputs)
    store_outputs = "\n".join(
        f"{output}[...] = {inner_output}"
        for output, inner_output in zip(outputs, inner_outputs, strict=True)
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
    input_types,
    output_core_shape_types,
    size_type,
):
    arg_types = [
        core_func,
        input_bc_patterns,
        output_bc_patterns,
        output_dtypes,
        inplace_pattern,
        allow_core_scalar,
        constant_inputs_types,
        input_types,
        output_core_shape_types,
        size_type,
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

    if not isinstance(allow_core_scalar, types.Literal):
        raise TypeError("allow_core_scalar must be literal.")
    allow_core_scalar = allow_core_scalar.literal_value

    batch_ndim = len(input_bc_patterns[0])
    nin = len(constant_inputs_types) + len(input_types)
    nout = len(output_bc_patterns)

    if nin == 0:
        raise TypingError("Empty argument list to vectorized op.")

    if nout == 0:
        raise TypingError("Empty list of outputs for vectorized op.")

    if not all(isinstance(input, types.Array) for input in input_types):
        raise TypingError("Vectorized inputs must be arrays.")

    if not all(
        len(pattern) == batch_ndim for pattern in input_bc_patterns + output_bc_patterns
    ):
        raise TypingError(
            "Vectorized broadcastable patterns must have the same length."
        )

    core_input_types = []
    for input_type, bc_pattern in zip(input_types, input_bc_patterns, strict=True):
        core_ndim = input_type.ndim - len(bc_pattern)
        if allow_core_scalar and core_ndim == 0:
            core_input_type = input_type.dtype
        else:
            core_input_type = types.Array(
                dtype=input_type.dtype, ndim=core_ndim, layout=input_type.layout
            )
        core_input_types.append(core_input_type)

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

    for output_idx, input_idx in inplace_pattern:
        output_type = input_types[input_idx]
        core_out_types[output_idx] = types.Array(
            dtype=output_type.dtype,
            ndim=output_type.ndim - batch_ndim,
            layout=input_type.layout,
        )
        out_types[output_idx] = output_type

    ret_type = types.Tuple(out_types)

    if len(output_dtypes) == 1:
        ret_type = ret_type.types[0]
    sig = ret_type(*arg_types)

    # So we can access the constant values in codegen...
    input_bc_patterns_val = input_bc_patterns
    output_bc_patterns_val = output_bc_patterns
    output_dtypes_val = output_dtypes
    inplace_pattern_val = inplace_pattern
    input_types = input_types
    size_is_none = isinstance(size_type, NoneType)

    def codegen(
        ctx,
        builder,
        sig,
        args,
    ):
        [_, _, _, _, _, _, constant_inputs, inputs, output_core_shapes, size] = args

        constant_inputs = cgutils.unpack_tuple(builder, constant_inputs)
        inputs = cgutils.unpack_tuple(builder, inputs)
        output_core_shapes = [
            cgutils.unpack_tuple(builder, shape)
            for shape in cgutils.unpack_tuple(builder, output_core_shapes)
        ]
        size = None if size_is_none else cgutils.unpack_tuple(builder, size)

        inputs = [
            arrayobj.make_array(ty)(ctx, builder, val)
            for ty, val in zip(input_types, inputs, strict=True)
        ]
        in_shapes = [cgutils.unpack_tuple(builder, obj.shape) for obj in inputs]

        iter_shape = compute_itershape(
            ctx,
            builder,
            in_shapes,
            input_bc_patterns_val,
            size,
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
                outputs[inplace_idx]._getvalue(),
            )
        return ctx.make_tuple(
            builder, sig.return_type, [out._getvalue() for out in outputs]
        )

    return sig, codegen


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
) -> tuple[list[ir.Value], list[types.Array]]:
    output_arrays = []
    output_arry_types = []
    one = ir.IntType(64)(1)
    inplace_dict = dict(inplace)
    for i, (core_shape, bc, dtype) in enumerate(
        zip(output_core_shapes, out_bc, dtypes, strict=True)
    ):
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

    # If there is no inplace operation, we know that all output arrays
    # don't alias. Informing llvm can make it easier to vectorize.
    if not inplace:
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
    for input, input_type, bc in zip(inputs, input_types, input_bc, strict=True):
        core_ndim = input_type.ndim - len(bc)

        idxs_bc = [zero if bc else idx for idx, bc in zip(idxs, bc, strict=True)] + [
            zero
        ] * core_ndim
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
    for output, output_type, bc in zip(outputs, output_types, output_bc, strict=True):
        core_ndim = output_type.ndim - len(bc)
        size_type = output.shape.type.element  # pyright: ignore[reportAttributeAccessIssue]
        output_shape = cgutils.unpack_tuple(builder, output.shape)  # pyright: ignore[reportAttributeAccessIssue]
        output_strides = cgutils.unpack_tuple(builder, output.strides)  # pyright: ignore[reportAttributeAccessIssue]

        idxs_bc = [zero if bc else idx for idx, bc in zip(idxs, bc, strict=True)] + [
            zero
        ] * core_ndim
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
