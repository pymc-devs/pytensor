from __future__ import annotations

import base64
import pickle
from collections.abc import Callable, Sequence
from textwrap import indent
from typing import Any

import numba
import numpy as np
from llvmlite import ir
from llvmlite.ir.values import MDValue
from numba import TypingError, types
from numba.core import cgutils
from numba.core.base import BaseContext
from numba.core.types.misc import NoneType
from numba.np import arrayobj

from pytensor.link.numba.cache import compile_numba_function_src
from pytensor.link.numba.dispatch import basic as numba_basic
from pytensor.link.numba.dispatch._llvmlite_self_ref import (
    ensure_self_ref_metadata_support,
)


ensure_self_ref_metadata_support()


class _DistinctEmptyMetadata(MDValue):
    """A ``distinct !{}`` metadata node, usable as an LLVM access group.

    llvmlite's ``MDValue`` only emits *uniqued* ``!{}`` nodes, which LLVM rejects as
    access groups: an access group must be ``distinct`` so two function-local accesses
    are never considered identical (a plain uniqued ``!{}`` crashes the verifier). Each
    instance has its own identity, so every loop that asks for one gets a fresh group.
    """

    def __init__(self, parent):
        super().__init__(parent, [], name=str(len(parent.metadata)))

    def descr(self, buf):
        buf += ("distinct !{}", "\n")

    def __eq__(self, other):
        return self is other

    def __ne__(self, other):
        return self is not other

    def __hash__(self):
        return id(self)


def encode_literals(literals: Sequence) -> str:
    return base64.encodebytes(pickle.dumps(literals)).decode()


def store_core_outputs(
    core_op_fn: Callable, nin: int, nout: int, inc_outputs: frozenset = frozenset()
) -> Callable:
    """Create a Numba function that wraps a core function and stores its vectorized outputs.

    If ``core_op_fn`` has a ``handles_out=True`` attribute, it is assumed to
    already accept ``(inputs..., outputs...) -> None`` and is returned as-is.

    @njit
    def store_core_outputs(i0, i1, ..., in, o0, o1, ..., on):
        to0, to1, ..., ton = core_op_fn(i0, i1, ..., in)
        o0[...] = to0      # direct outputs
        o1 += to1          # inc outputs (in-place add works for 0d and Nd)
        ...

    ``inc_outputs`` lists output indices that use ``+=`` instead of ``=``.
    """
    if getattr(core_op_fn, "handles_out", False):
        return core_op_fn

    inputs = [f"i{i}" for i in range(nin)]
    outputs = [f"o{i}" for i in range(nout)]
    inner_outputs = [f"t{output}" for output in outputs]

    inp_signature = ", ".join(inputs)
    out_signature = ", ".join(outputs)
    inner_out_signature = ", ".join(inner_outputs)
    store_outputs = "\n".join(
        f"{output} += {inner_output}"
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


def _compute_idx_load_axes(indexed_inputs, indexed_outputs, idx_ndims):
    """Compute which loop dimensions load each index array.

    For a 1-D index on axis A this is ``(A,)``; for a 2-D index ``(A, A+1)``.
    Multi-index groups (``x[idx_a, idx_b]``) share the group's minimum axis.

    Parameters
    ----------
    indexed_inputs : tuple of ((tuple[int, ...], int) | None)
        Per-index: (source input positions, source axis) or None.
    indexed_outputs : tuple of ((tuple[int, ...], int, str, bool) | None)
        Per-index: (output positions, axis, mode, distinct) or None.
    idx_ndims : tuple of int
        Number of dimensions of each index array.
    """
    n_indices = len(indexed_inputs)
    if n_indices == 0:
        return ()

    src_to_indices: dict = {}
    for k, entry in enumerate(indexed_inputs):
        if entry is not None:
            for src in entry[0]:
                src_to_indices.setdefault(src, []).append(k)
    for k, entry in enumerate(indexed_outputs):
        if entry is not None:
            for out_idx in entry[0]:
                src_to_indices.setdefault(("w", out_idx), []).append(k)

    parent = list(range(n_indices))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for indices in src_to_indices.values():
        for i in range(1, len(indices)):
            a, b = find(indices[0]), find(indices[i])
            if a != b:
                parent[a] = b

    group_min_axis: dict[int, int] = {}
    for k, entry in enumerate(indexed_inputs):
        if entry is not None:
            _, axis = entry
            root = find(k)
            group_min_axis[root] = min(group_min_axis.get(root, axis), axis)
    for k, entry in enumerate(indexed_outputs):
        if entry is not None:
            root = find(k)
            _, out_axis, *_ = entry
            group_min_axis[root] = min(group_min_axis.get(root, out_axis), out_axis)

    return tuple(
        tuple(range(group_min_axis[find(k)], group_min_axis[find(k)] + idx_ndims[k]))
        for k in range(n_indices)
    )


def _core_slice_layout(full_layout, batch_bc_pattern):
    """Layout of a core slice (the trailing core dims at a fixed batch index).

    The core slice keeps the full array's layout when the array is C-contiguous (its
    trailing dims are contiguous) or when every batch dim is broadcastable: a size-1
    dim contributes a stride multiplier of 1, so it cannot make the core slice
    strided (``all(())`` is True, so the no-batch-dims case is covered too). A real
    (size > 1) batch dim in front of an F-/non-contiguous array's core dims makes the
    slice strided, so we fall back to the "any" layout — declaring it contiguous
    makes numba emit a contiguous load over a strided buffer and silently read the
    wrong elements.
    """
    if full_layout == "C" or all(batch_bc_pattern):
        return full_layout
    return "A"


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
            core_input_types.append(
                types.Array(
                    dtype=input_type.dtype,
                    ndim=core_ndim,
                    layout=_core_slice_layout(input_type.layout, bc_pattern),
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


NO_INDEXED_INPUTS = encode_literals(((), ()))
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

    ``update_outputs`` maps ``{output_idx: (array, array_type)}`` for outputs
    that reuse an indexed-write target buffer instead of being freshly allocated.
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

    # If there is no inplace operation, we know that all output arrays
    # don't alias. Informing llvm can make it easier to vectorize.
    if not inplace and not update_outputs:
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
    input_read_spec: tuple[tuple[tuple[int, int], ...] | None, ...] | None = None,
    idx_arrays: list | None = None,
    idx_arrays_type: tuple | None = None,
    idx_load_axes: tuple[tuple[int, ...], ...] | None = None,
    idx_bc: tuple[tuple[bool, ...], ...] | None = None,
    output_write_spec: tuple[tuple[tuple[int, int], ...] | None, ...] | None = None,
    inplace: tuple[tuple[int, int], ...] = (),
    distinct_outputs: frozenset = frozenset(),
):
    safe = (False, False)

    n_outputs = len(outputs)

    # Scoped noalias metadata: input loads and output stores are tagged with
    # alias scopes so LLVM can disambiguate them without runtime overlap checks
    # (and without loop versioning). Inputs share one scope (their loads never
    # conflict with each other) while each output gets its own, so that every
    # access can claim noalias against all buffers it is guaranteed not to
    # overlap: PyTensor guarantees distinct output buffers, and that inputs
    # don't alias outputs *except* for an input destroyed by an inplace output.
    # Loads of a destroyed input are tagged with its output's scope instead:
    # they stay MayAlias with that output's stores (LLVM resolves the exact
    # overlap through pointer identity, since the output reuses the input's
    # array struct) yet are still disambiguated from every other buffer.
    mod = builder.module
    domain = mod.add_metadata([], self_ref=True)
    input_scope = mod.add_metadata([domain], self_ref=True)
    output_scopes = [
        mod.add_metadata([domain], self_ref=True) for _ in range(n_outputs)
    ]
    input_scope_set = mod.add_metadata([input_scope])
    output_scope_set = mod.add_metadata(output_scopes)
    out_alias_sets = [mod.add_metadata([scope]) for scope in output_scopes]
    out_noalias_sets = [
        mod.add_metadata([input_scope, *(s for s in output_scopes if s is not scope)])
        for scope in output_scopes
    ]
    destroyed_inputs = {in_idx: out_idx for out_idx, in_idx in inplace}

    # When an indexed-update output writes through statically-distinct indices, its
    # read-modify-write carries no cross-iteration dependency, so the loop can vectorize
    # the value-compute (LLVM scalarizes only the indexed stores). We promise this with an
    # access group on those RMW load/stores plus `llvm.loop.parallel_accesses` on the
    # innermost latch. The group must be a *distinct* node so it is never uniqued with
    # another loop's.
    access_group = _DistinctEmptyMetadata(mod) if distinct_outputs else None

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
        for out in range(n_outputs):
            if output_accumulator[out][0] is not None:
                continue
            if all(output_bc[out][dim:]):
                value = outputs[out][0].type.pointee(0)
                accu = cgutils.alloca_once_value(builder, value)
                output_accumulator[out] = (accu, dim)

        loop = cgutils.for_range(builder, length)
        loop_stack.append(loop)
        loops.append(loop.__enter__())

    # Code in the inner most loop...
    loop_idxs = [loopval.index for loopval in loops]

    # Load indirect indices from all index arrays (1-D vectors).
    indirect_idxs = []
    if idx_arrays is not None:
        assert idx_arrays_type is not None
        assert idx_load_axes is not None
        for idx_counter, (idx_arr, idx_arr_type, load_axes) in enumerate(
            zip(idx_arrays, idx_arrays_type, idx_load_axes)
        ):
            read_idxs = []
            for d, ax in enumerate(load_axes):
                is_bc = (
                    idx_bc[idx_counter][d]
                    if idx_bc and len(idx_bc[idx_counter]) > d
                    else False
                )
                read_idxs.append(zero if is_bc else loop_idxs[ax])
            ptr = cgutils.get_item_pointer2(
                context,
                builder,
                idx_arr.data,
                cgutils.unpack_tuple(builder, idx_arr.shape),
                cgutils.unpack_tuple(builder, idx_arr.strides),
                idx_arr_type.layout,
                read_idxs,
                False,
                False,
            )
            val = builder.load(ptr)
            val.set_metadata("alias.scope", input_scope_set)
            val.set_metadata("noalias", output_scope_set)
            if access_group is not None:
                val.set_metadata("llvm.access.group", access_group)
            i64 = ir.IntType(64)
            if val.type != i64:
                if idx_arr_type.dtype.signed:
                    val = builder.sext(val, i64)
                else:
                    val = builder.zext(val, i64)
            indirect_idxs.append(val)

    # Load values from input arrays
    input_vals = []
    for input_i, (inp, inp_type, inp_bc) in enumerate(
        zip(inputs, input_types, input_bc, strict=True)
    ):
        spec = input_read_spec[input_i] if input_read_spec is not None else None
        n_indexed = len(spec) if spec else 0
        # n_indexed source axes are replaced by the index arrays' broadcast
        # loop dims.  n_index_loop_dims = max ndim of the index arrays in the
        # group (1 for 1D vectors, 2 for 2D matrices, etc.).
        if spec:
            assert idx_arrays_type is not None
            n_index_loop_dims = max(
                (idx_arrays_type[idx_k].ndim for idx_k, _ in spec), default=0
            )
        else:
            n_index_loop_dims = 0
        core_ndim = inp_type.ndim - len(inp_bc) - n_indexed + n_index_loop_dims

        if spec is not None:
            assert idx_arrays_type is not None
            indexed_axes = {src_axis: idx_k for idx_k, src_axis in spec}
            inp_shape = cgutils.unpack_tuple(builder, inp.shape)
            read_idx = []
            # result_dim is a cursor into the output loop dimensions.  It
            # advances by 1 for regular source dims, but jumps by the index
            # ndim for indexed dims (1-D idx → +1, 2-D mat → +2).
            # idxs[result_dim] selects which loop counter loads each dim.
            #
            # Ex 1: z = x[idx] + y — x(5,3), idx(4,), y(4,3) → z(4,3)
            #   output loops: i over 4, j over 3
            #   x (indexed on axis 0):        result_dim
            #     src 0 → indexed, use idx[i]     0 (+1) → 1
            #     src 1 → loop counter j           1 (+1) → 2
            #     loads as: x[idx[i], j]
            #   y (not indexed):
            #     loads as: y[i, j]
            #
            # Ex 2: z = x[mat] — x(5,3), mat(2,4) → z(2,4,3)
            #   output loops: i over 2, j over 4, k over 3
            #   x (indexed on axis 0, 2-D index): result_dim
            #     src 0 → indexed, use mat[i,j]    0 (+2) → 2
            #     src 1 → loop counter k            2 (+1) → 3
            #     loads as: x[mat[i,j], k]
            #
            # Ex 3: z = x[:, idx] — x(3,5,2), idx(4,) on axis 1 → z(3,4,2)
            #   output loops: i over 3, j over 4, k over 2
            #   x (indexed on axis 1):            result_dim
            #     src 0 → loop counter i            0 (+1) → 1
            #     src 1 → indexed, use idx[j]      1 (+1) → 2
            #     src 2 → loop counter k            2 (+1) → 3
            #     loads as: x[i, idx[j], k]
            result_dim = 0
            for src_dim in range(inp_type.ndim):
                if src_dim in indexed_axes:
                    idx_k = indexed_axes[src_dim]
                    idx_val = _wrap_negative_index(
                        indirect_idxs[idx_k],
                        inp_shape[src_dim],
                        signed=idx_arrays_type[idx_k].dtype.signed,
                    )
                    read_idx.append(idx_val)
                    if src_dim == max(indexed_axes):
                        result_dim += n_index_loop_dims
                elif result_dim < len(inp_bc):
                    read_idx.append(
                        zero if inp_bc[result_dim] else loop_idxs[result_dim]
                    )
                    result_dim += 1
                else:
                    read_idx.append(zero)
                    result_dim += 1
        else:
            read_idx = [
                zero if bc else idx for idx, bc in zip(loop_idxs, inp_bc, strict=True)
            ] + [zero] * core_ndim

        read_ptr = cgutils.get_item_pointer2(
            context,
            builder,
            inp.data,
            cgutils.unpack_tuple(builder, inp.shape),
            cgutils.unpack_tuple(builder, inp.strides),
            inp_type.layout,
            read_idx,
            *safe,
        )
        if core_scalar and core_ndim == 0:
            # Retrive scalar item at index
            read_val = builder.load(read_ptr)
            destination = destroyed_inputs.get(input_i)
            if destination is None:
                read_val.set_metadata("alias.scope", input_scope_set)
                read_val.set_metadata("noalias", output_scope_set)
            else:
                read_val.set_metadata("alias.scope", out_alias_sets[destination])
                read_val.set_metadata("noalias", out_noalias_sets[destination])
            # Every memory access in the loop must join the access group, or LLVM's
            # `isAnnotatedParallel` rejects the whole loop (one untagged load voids the
            # `llvm.loop.parallel_accesses` promise).
            if access_group is not None:
                read_val.set_metadata("llvm.access.group", access_group)
        else:
            # Retrieve array item at index
            # This is a streamlined version of Numba's `GUArrayArg.load`.
            # Layout must match the core type resolved in `_compute_vectorized_types`
            # (see `_core_slice_layout`): the core slice of an F/non-contiguous array
            # is strided, so it cannot claim a contiguous layout.
            read_array_type = types.Array(
                dtype=inp_type.dtype,
                ndim=core_ndim,
                layout=_core_slice_layout(inp_type.layout, inp_bc),
            )
            read_array = context.make_array(read_array_type)(context, builder)
            core_shape = cgutils.unpack_tuple(builder, inp.shape)[
                inp_type.ndim - core_ndim :
            ]
            core_strides = cgutils.unpack_tuple(builder, inp.strides)[
                inp_type.ndim - core_ndim :
            ]
            itemsize = context.get_abi_sizeof(context.get_data_type(inp_type.dtype))
            context.populate_array(
                read_array,
                # TODO whey do we need to bitcast?
                data=builder.bitcast(read_ptr, read_array.data.type),
                shape=core_shape,
                strides=core_strides,
                itemsize=context.get_constant(types.intp, itemsize),
                # TODO what is meminfo about?
                meminfo=None,
            )
            read_val = read_array._getvalue()

        input_vals.append(read_val)

    # Create output slices to pass to inner func
    output_slices = []
    scratch_outputs = []
    for output_i, (out, out_type, out_bc) in enumerate(
        zip(outputs, output_types, output_bc, strict=True)
    ):
        core_ndim = out_type.ndim - len(out_bc)
        size_type = out.shape.type.element  # pyright: ignore[reportAttributeAccessIssue]
        output_shape = cgutils.unpack_tuple(builder, out.shape)  # pyright: ignore[reportAttributeAccessIssue]
        output_strides = cgutils.unpack_tuple(builder, out.strides)  # pyright: ignore[reportAttributeAccessIssue]

        # Same cursor logic as indexed reads (see read_idx above):
        # iterate over target dims, use indirect index for indexed axes,
        # loop counter for the rest.
        spec = output_write_spec[output_i] if output_write_spec is not None else None
        if spec is not None:
            assert idx_arrays_type is not None
            indexed_axes = {src_axis: idx_k for idx_k, src_axis in spec}
            n_indexed = len(indexed_axes)
            n_index_loop_dims = max(idx_arrays_type[idx_k].ndim for idx_k, _ in spec)
            # Multi-axis indexing: N indexed axes contribute max(idx.ndim)
            # loop dims, so the target can have more batch dims than the loop.
            n_target_batch = n_indexed + len(out_bc) - n_index_loop_dims
            target_core_ndim = out_type.ndim - n_target_batch
            write_idx = []
            result_dim = 0
            for tgt_dim in range(n_target_batch):
                if tgt_dim in indexed_axes:
                    idx_k = indexed_axes[tgt_dim]
                    idx_val = _wrap_negative_index(
                        indirect_idxs[idx_k],
                        output_shape[tgt_dim],
                        signed=idx_arrays_type[idx_k].dtype.signed,
                    )
                    write_idx.append(idx_val)
                    if tgt_dim == max(indexed_axes):
                        result_dim += n_index_loop_dims
                else:
                    write_idx.append(
                        zero if out_bc[result_dim] else loop_idxs[result_dim]
                    )
                    result_dim += 1
            write_idx += [zero] * target_core_ndim
        else:
            write_idx = [
                zero if bc else idx for idx, bc in zip(loop_idxs, out_bc, strict=True)
            ] + [zero] * core_ndim
        effective_core_ndim = target_core_ndim if spec is not None else core_ndim
        write_ptr = cgutils.get_item_pointer2(
            context,
            builder,
            out.data,
            output_shape,
            output_strides,
            out_type.layout,
            write_idx,
            *safe,
        )
        write_array_type = types.Array(
            dtype=out_type.dtype, ndim=effective_core_ndim, layout=out_type.layout
        )
        write_array = context.make_array(write_array_type)(context, builder)
        if effective_core_ndim == 0:
            # Redirect the 0-d output slice through a stack slot so the store
            # into the real output buffer happens below, after the core call,
            # where it can carry the alias scope metadata. The slot is
            # initialized from the output buffer to preserve read-modify-write
            # semantics (`o += t` in `store_core_outputs`); SROA collapses the
            # slot after inlining.
            scratch = cgutils.alloca_once(builder, write_ptr.type.pointee)
            init_val = builder.load(write_ptr)
            init_val.set_metadata("alias.scope", out_alias_sets[output_i])
            init_val.set_metadata("noalias", out_noalias_sets[output_i])
            if output_i in distinct_outputs:
                init_val.set_metadata("llvm.access.group", access_group)
            builder.store(init_val, scratch)
            scratch_outputs.append((scratch, write_ptr, output_i))
            write_ptr = scratch
        core_shape = (
            output_shape[-effective_core_ndim:] if effective_core_ndim > 0 else []
        )
        core_strides = (
            output_strides[-effective_core_ndim:] if effective_core_ndim > 0 else []
        )
        itemsize = context.get_abi_sizeof(context.get_data_type(out_type.dtype))
        context.populate_array(
            write_array,
            # TODO whey do we need to bitcast?
            data=builder.bitcast(write_ptr, write_array.data.type),
            shape=cgutils.pack_array(builder, core_shape, ty=size_type),
            strides=cgutils.pack_array(builder, core_strides, ty=size_type),
            itemsize=context.get_constant(types.intp, itemsize),
            # TODO what is meminfo about?
            meminfo=None,
        )
        write_val = write_array._getvalue()
        output_slices.append(write_val)

    inner_codegen = context.get_function(core_func, core_signature)

    if isinstance(core_signature.args[0], types.StarArgTuple | types.StarArgUniTuple):
        input_vals = [context.make_tuple(builder, core_signature.args[0], input_vals)]

    inner_codegen(builder, [*constant_inputs, *input_vals, *output_slices])

    for scratch, write_ptr, output_i in scratch_outputs:
        out_val = builder.load(scratch)
        store = builder.store(out_val, write_ptr)
        store.set_metadata("alias.scope", out_alias_sets[output_i])
        store.set_metadata("noalias", out_noalias_sets[output_i])
        if output_i in distinct_outputs:
            store.set_metadata("llvm.access.group", access_group)

    # Close the loops.  Under a no-dup promise, tag the innermost loop's latch with
    # `llvm.loop.parallel_accesses` referencing the access group, so the vectorizer
    # treats the tagged RMW accesses as free of loop-carried dependencies. The latch is
    # the body block the builder sits in just before `for_range` emits its backedge.
    for depth, loop in enumerate(loop_stack[::-1]):
        if depth == 0 and access_group is not None:
            latch_block = builder.basic_block
            loop.__exit__(None, None, None)
            parallel_md = mod.add_metadata(
                [ir.MetaDataString(mod, "llvm.loop.parallel_accesses"), access_group]
            )
            loop_md = mod.add_metadata([parallel_md], self_ref=True)
            latch_block.terminator.set_metadata("llvm.loop", loop_md)
        else:
            loop.__exit__(None, None, None)


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
    """Vectorized intrinsic with optional indirect indexing for reads and writes.

    For indexed operations, outer inputs are ordered as
    ``[core_inputs..., idx_0, idx_1, ..., write_target_0, ...]``.

    ``indexed_inputs`` groups core input positions by which index they
    read through: e.g. ``((0, 2), (1,))`` means idx_0 reads inputs 0 and 2,
    idx_1 reads input 1.  ``None`` entries are update-only indices (no reads).

    ``indexed_outputs`` has one entry per index array (same length as
    ``indexed_inputs``).  ``None`` means that index is not used for updates.
    ``((out_0, out_1), mode)`` means that index updates outputs out_0 and
    out_1 with *mode* ``"set"`` or ``"inc"``.

    For non-indexed calls, both are ``()``.
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
    indexed_inputs, idx_broadcastable = _decode_literal(
        indexed_inputs, "indexed_inputs"
    )
    indexed_outputs = _decode_literal(indexed_outputs, "indexed_outputs")

    if not isinstance(allow_core_scalar, types.Literal):
        raise TypingError("allow_core_scalar must be literal.")
    allow_core_scalar = allow_core_scalar.literal_value

    # Count write targets (one per unique output index)
    write_out_idxs = set()
    for entry in indexed_outputs:
        if entry is not None:
            write_out_idxs.update(entry[0])
    n_write_targets = len(write_out_idxs)

    inplace_pattern = tuple(
        (out_idx, inp_idx)
        for out_idx, inp_idx in inplace_pattern
        if out_idx not in write_out_idxs
    )

    n_indices = len(indexed_inputs)
    n_core_inputs = len(outer_input_types) - n_indices - n_write_targets
    source_input_types = tuple(outer_input_types[i] for i in range(n_core_inputs))
    idx_types = tuple(outer_input_types[n_core_inputs + k] for k in range(n_indices))
    write_target_types = tuple(
        outer_input_types[n_core_inputs + n_indices + j] for j in range(n_write_targets)
    )

    idx_ndims = tuple(idx_types[k].ndim for k in range(n_indices))
    idx_load_axes = _compute_idx_load_axes(indexed_inputs, indexed_outputs, idx_ndims)

    # Aggregate per-input: which (idx_k, source_axis) pairs apply.
    read_spec_dict: dict[int, list[tuple[int, int]]] = {}
    for k, entry in enumerate(indexed_inputs):
        if entry is not None:
            sources, source_axis = entry
            for src in sources:
                read_spec_dict.setdefault(src, []).append((k, source_axis))
    input_read_spec = tuple(
        tuple(read_spec_dict[p]) if p in read_spec_dict else None
        for p in range(n_core_inputs)
    )

    # Build effective input types that match input_bc_patterns ndim.
    # For ND indices, the source ndim differs from the result ndim:
    # a 2-D index on 1 axis expands 1 source axis into 2 loop dims.
    input_types = []
    for p, src_type in enumerate(source_input_types):
        spec = input_read_spec[p]
        if spec is not None:
            n_indexed_axes = len(spec)
            n_index_loop_dims = max(idx_types[idx_k].ndim for idx_k, _ in spec)
            if n_indexed_axes != n_index_loop_dims:
                effective_ndim = src_type.ndim - n_indexed_axes + n_index_loop_dims
                input_types.append(
                    types.Array(src_type.dtype, effective_ndim, src_type.layout)
                )
            else:
                input_types.append(src_type)
        else:
            input_types.append(src_type)
    input_types = tuple(input_types)

    # Per-output: tuple of (idx_k, source_axis) pairs, or None.
    # Same format as input_read_spec.
    # indexed_outputs entries are (sources, source_axis, mode) or None.
    write_spec_dict: dict[int, list[tuple[int, int]]] = {}
    for k, entry in enumerate(indexed_outputs):
        if entry is None:
            continue
        sources, source_axis, _mode, *_ = entry
        for out_idx in sources:
            write_spec_dict.setdefault(out_idx, []).append((k, source_axis))
    # Write target buffers are appended to the outer inputs in ascending output
    # index order (see the rewriter's sorted(write_targets.items())), so the
    # target buffer index of an output is its rank among the write outputs.
    out_idx_to_write_target = {
        out_idx: target_idx
        for target_idx, out_idx in enumerate(sorted(write_spec_dict))
    }
    output_write_spec = tuple(
        tuple(write_spec_dict[p]) if p in write_spec_dict else None
        for p in range(len(output_bc_patterns))
    )

    core_input_types, core_out_types, out_types, ret_type = _compute_vectorized_types(
        input_types,
        input_bc_patterns,
        output_bc_patterns,
        output_dtypes,
        inplace_pattern,
        allow_core_scalar,
        output_core_shape_types,
    )

    if out_idx_to_write_target:
        core_out_types = list(core_out_types)
        out_types = list(out_types)
        batch_ndim = len(input_bc_patterns[0])
        for out_idx, target_idx in out_idx_to_write_target.items():
            target_type = write_target_types[target_idx]
            out_types[out_idx] = target_type
            # Multi-axis indexing: N indexed axes contribute max(idx.ndim)
            # loop dims, so the target can have more batch dims than the loop.
            spec = output_write_spec[out_idx]
            if spec is not None:
                n_indexed = len(spec)
                n_index_loop_dims = max(idx_types[idx_k].ndim for idx_k, _ in spec)
                n_target_batch = n_indexed + batch_ndim - n_index_loop_dims
            else:
                n_target_batch = batch_ndim
            core_out_types[out_idx] = types.Array(
                dtype=target_type.dtype,
                ndim=target_type.ndim - n_target_batch,
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
    write_idx_set = frozenset(
        k for k, entry in enumerate(indexed_outputs) if entry is not None
    )
    # Output positions whose indexed update was flagged distinct-index by the rewriter
    # (4th spec field) -> safe to emit the no-dup vectorization promise.
    distinct_output_idxs = frozenset(
        out_idx
        for entry in indexed_outputs
        if entry is not None and len(entry) > 3 and entry[3]
        for out_idx in entry[0]
    )

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

        # First n_core_inputs outer inputs are source arrays
        inputs = [
            arrayobj.make_array(source_input_types[i])(ctx, builder, all_outer[i])
            for i in range(n_core_inputs)
        ]
        in_shapes = [cgutils.unpack_tuple(builder, obj.shape) for obj in inputs]

        # Next n_indices inputs are index arrays
        idx_arrs = [
            arrayobj.make_array(idx_types[k])(
                ctx, builder, all_outer[n_core_inputs + k]
            )
            for k in range(n_indices)
        ]

        # Remaining inputs are write target buffers
        write_target_arrs = [
            arrayobj.make_array(write_target_types[j])(
                ctx, builder, all_outer[n_core_inputs + n_indices + j]
            )
            for j in range(len(write_target_types))
        ]

        idx_shapes = [
            cgutils.unpack_tuple(builder, idx_arrs[k].shape) for k in range(n_indices)
        ]

        one = ir.IntType(64)(1)
        iter_shapes = list(in_shapes)
        iter_bc = list(input_bc_patterns)

        for p, spec in enumerate(input_read_spec):
            if spec is None:
                continue
            indexed_axes = {src_axis: idx_k for idx_k, src_axis in spec}
            n_index_loop_dims = max(idx_types[idx_k].ndim for idx_k, _ in spec)
            # An indexed input imposes no constraint on the loop dims produced by
            # its indices: those are pinned by the index arrays' own iter_shape
            # entries below.  Mark each such loop dim broadcastable and copy the
            # source shape for the non-indexed dims.
            batch_ndim = len(input_bc_patterns[p])
            max_axis = max(a for _, a in spec)
            new_shape = []
            new_bc = []
            src_d = 0
            idx_d = 0
            for loop_d in range(batch_ndim):
                if src_d in indexed_axes and idx_d < n_index_loop_dims:
                    new_shape.append(one)
                    new_bc.append(True)
                    idx_d += 1
                    if idx_d >= n_index_loop_dims:
                        src_d = max_axis + 1
                else:
                    new_shape.append(in_shapes[p][src_d])
                    new_bc.append(iter_bc[p][loop_d])
                    src_d += 1
            iter_shapes[p] = new_shape
            iter_bc[p] = tuple(new_bc)

        # Each index array participates in iter_shape validation.
        # Write indices can broadcast against each other, but if ALL write
        # indices on a given loop dim are bc=True, none constrains the loop
        # size.  Force bc=False in that case so compute_itershape requires
        # the index length to match the loop.
        batch_ndim = len(input_bc_patterns[0]) if input_bc_patterns else 0

        write_all_bc = [True] * batch_ndim
        for k in range(n_indices):
            if k not in write_idx_set:
                continue
            for d, ax in enumerate(idx_load_axes[k]):
                if ax < batch_ndim:
                    idx_bc_on_d = (
                        idx_broadcastable[k][d]
                        if d < len(idx_broadcastable[k])
                        else False
                    )
                    if not idx_bc_on_d:
                        write_all_bc[ax] = False

        for k in range(n_indices):
            is_write = k in write_idx_set
            idx_shape_entry = [one] * batch_ndim
            bc_entry = [True] * batch_ndim
            for d, ax in enumerate(idx_load_axes[k]):
                if ax < batch_ndim and d < len(idx_shapes[k]):
                    idx_shape_entry[ax] = idx_shapes[k][d]
                idx_bc_on_d = (
                    idx_broadcastable[k][d] if d < len(idx_broadcastable[k]) else False
                )
                if is_write and idx_bc_on_d and write_all_bc[ax]:
                    idx_bc_on_d = False
                if ax < batch_ndim:
                    bc_entry[ax] = idx_bc_on_d
            iter_shapes.append(idx_shape_entry)
            iter_bc.append(tuple(bc_entry))

        iter_shape = compute_itershape(ctx, builder, iter_shapes, tuple(iter_bc), size)

        # Build update_outputs dict for make_outputs: out_idx -> (array, type)
        update_outputs_dict = (
            {
                out_idx: (
                    write_target_arrs[target_idx],
                    write_target_types[target_idx],
                )
                for out_idx, target_idx in out_idx_to_write_target.items()
            }
            if out_idx_to_write_target
            else None
        )

        outputs, output_types = make_outputs(
            ctx,
            builder,
            iter_shape,
            output_bc_patterns,
            output_dtypes,
            inplace_pattern,
            inputs,
            source_input_types,
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
            input_bc_patterns,
            output_bc_patterns,
            source_input_types,
            output_types,
            core_scalar=allow_core_scalar,
            input_read_spec=input_read_spec,
            idx_arrays=idx_arrs,
            idx_arrays_type=idx_types,
            idx_load_axes=idx_load_axes,
            idx_bc=idx_broadcastable,
            output_write_spec=output_write_spec,
            inplace=inplace_pattern,
            distinct_outputs=distinct_output_idxs,
        )

        return _codegen_return_outputs(
            ctx,
            builder,
            sig,
            outputs,
            inplace_pattern,
            extra_incref=frozenset(out_idx_to_write_target),
        )

    return sig, codegen
