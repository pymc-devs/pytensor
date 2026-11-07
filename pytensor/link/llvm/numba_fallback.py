"""Run Ops the LLVM backend doesn't implement through their numba implementation.

For such an Op we take PyTensor's numba implementation (``numba_funcify``) and wrap it in
a numba ``cfunc`` with the kernel ABI ``int32(PtCtx *ctx, PtArray *args, void *params)``:

* inputs become numba arrays that view the caller's memory (data, shape, strides; no copy),
  or plain scalars for inputs of ``ScalarType``,
* every output is *kernel-allocated*: the wrapper calls ``ctx->rt->alloc`` through the
  runtime table and copies numba's result into that buffer, so ownership stays with our
  runtime,
* exceptions raised by the numba code are caught and reported as a nonzero return code.

The wrapper needs no Python or GIL when the numba implementation is in nopython mode
(numba's object-mode fallbacks take the GIL themselves).
"""

from __future__ import annotations

import hashlib

import numba
import numpy as np
from llvmlite import ir
from numba import types
from numba.core import cgutils
from numba.extending import intrinsic
from numba.np.arrayobj import populate_array

from pytensor.scalar.basic import ScalarType
from pytensor.tensor.type import TensorType


PTARRAY_SIZE = 40  # sizeof(PtArray): data, shape, strides, owner, ndim
RUNTIME_ALLOC_OFFSET = 8  # offsetof(PtRuntime, alloc)

_i8p = ir.IntType(8).as_pointer()
_i64 = ir.IntType(64)
_i64p = _i64.as_pointer()


def _field(builder, base, arg_index, field, ty):
    off = arg_index * PTARRAY_SIZE + 8 * field
    p = builder.gep(builder.bitcast(base, _i8p), [ir.Constant(_i64, off)])
    return builder.load(builder.bitcast(p, ty.as_pointer()))


def _make_array(context, builder, arrty, data, shape, strides):
    ary = context.make_array(arrty)(context, builder)
    itemsize = context.get_abi_sizeof(context.get_data_type(arrty.dtype))
    intp = context.get_value_type(types.intp)
    populate_array(
        ary,
        data=builder.bitcast(data, ary.data.type),
        shape=cgutils.pack_array(builder, shape, ty=intp),
        strides=cgutils.pack_array(builder, strides, ty=intp),
        itemsize=context.get_constant(types.intp, itemsize),
        meminfo=None,  # borrowed memory
    )
    return ary._getvalue()


def array_input(index: int, ndim: int, dtype: np.dtype):
    """Intrinsic: numba array viewing args[index] (any strides, no copy)."""
    arrty = types.Array(numba.from_dtype(dtype), ndim, "A")

    @intrinsic
    def view(typingctx, args):
        def codegen(context, builder, sig, llargs):
            (base,) = llargs
            data = _field(builder, base, index, 0, _i8p)
            shp = _field(builder, base, index, 1, _i64)
            strd = _field(builder, base, index, 2, _i64)
            shp, strd = builder.inttoptr(shp, _i64p), builder.inttoptr(strd, _i64p)
            shape = [builder.load(builder.gep(shp, [ir.Constant(_i64, d)])) for d in range(ndim)]
            strides = [builder.load(builder.gep(strd, [ir.Constant(_i64, d)])) for d in range(ndim)]
            return _make_array(context, builder, arrty, data, shape, strides)

        return arrty(args), codegen

    return view


def scalar_input(index: int, dtype: np.dtype):
    """Intrinsic: load the scalar stored at args[index].data."""
    nbty = numba.from_dtype(dtype)

    @intrinsic
    def load(typingctx, args):
        def codegen(context, builder, sig, llargs):
            (base,) = llargs
            data = _field(builder, base, index, 0, _i8p)
            llty = context.get_data_type(nbty)
            v = builder.load(builder.bitcast(data, llty.as_pointer()))
            return context.data_model_manager[nbty].from_data(builder, v)

        return nbty(args), codegen

    return load


def allocated_output(index: int, ndim: int, dtype: np.dtype):
    """Intrinsic: allocate args[index] through ctx->rt->alloc with the given shape, fill
    the PtArray (data, owner, shape, C strides) and return a C-contiguous numba view."""
    nbdt = numba.from_dtype(dtype)
    arrty = types.Array(nbdt, ndim, "C")
    itemsize = np.dtype(dtype).itemsize

    @intrinsic
    def alloc(typingctx, ctx, args, shape):
        if not isinstance(shape, types.BaseTuple) or len(shape) != ndim:
            return None

        def codegen(context, builder, sig, llargs):
            ctx_p, base, shape_v = llargs
            dims = [context.cast(builder, builder.extract_value(shape_v, d), sig.args[2][d], types.int64) for d in range(ndim)]
            n = ir.Constant(_i64, itemsize)
            for d in dims:
                n = builder.mul(n, d)
            # rt = ctx->rt; data = rt->alloc(ctx, n, 64, &owner)
            rt = builder.load(builder.bitcast(ctx_p, _i8p.as_pointer()))
            fnty = ir.FunctionType(_i8p, [_i8p, _i64, _i64, _i8p.as_pointer()])
            fptr_p = builder.gep(rt, [ir.Constant(_i64, RUNTIME_ALLOC_OFFSET)])
            fptr = builder.load(builder.bitcast(fptr_p, fnty.as_pointer().as_pointer()))
            owner = cgutils.alloca_once(builder, _i8p)
            data = builder.call(fptr, [builder.bitcast(ctx_p, _i8p), n, ir.Constant(_i64, 64), owner])
            # fill the PtArray: data, owner, shape, strides
            entry = builder.gep(builder.bitcast(base, _i8p), [ir.Constant(_i64, index * PTARRAY_SIZE)])

            def slot(k, ty):
                return builder.bitcast(builder.gep(entry, [ir.Constant(_i64, 8 * k)]), ty.as_pointer())

            builder.store(data, slot(0, _i8p))
            builder.store(builder.load(owner), slot(3, _i8p))
            shp = builder.inttoptr(builder.load(slot(1, _i64)), _i64p)
            strd = builder.inttoptr(builder.load(slot(2, _i64)), _i64p)
            strides = [None] * ndim
            acc = ir.Constant(_i64, itemsize)
            for d in reversed(range(ndim)):
                strides[d] = acc
                acc = builder.mul(acc, dims[d])
            for d in range(ndim):
                builder.store(dims[d], builder.gep(shp, [ir.Constant(_i64, d)]))
                builder.store(strides[d], builder.gep(strd, [ir.Constant(_i64, d)]))
            return _make_array(context, builder, arrty, data, dims, strides)

        return arrty(ctx, args, shape), codegen

    return alloc


_SIG = types.int32(types.voidptr, types.voidptr, types.voidptr)

# bump when the generated wrapper changes
FALLBACK_VERSION = 2


def numba_fallback_kernel(node):
    """Compile a cfunc kernel for `node`; returns the numba cfunc object (keep it alive)."""
    from pytensor import config
    from pytensor.link.numba.cache import compile_numba_function_src
    from pytensor.link.numba.dispatch.basic import numba_funcify_ensure_cache

    op_fn, op_key = numba_funcify_ensure_cache(node.op, node=node)
    # module globals give numba a real module (`__name__`) to rebuild cached environments from
    env = globals() | {"op_fn": op_fn, "np": np}
    lines = ["def kernel(ctx, args, params):", "    try:"]
    names = []
    for i, inp in enumerate(node.inputs):
        t = inp.type
        if isinstance(t, ScalarType):
            env[f"in{i}"] = scalar_input(i, np.dtype(t.dtype))
        elif isinstance(t, TensorType):
            env[f"in{i}"] = array_input(i, t.ndim, np.dtype(t.dtype))
        else:
            raise NotImplementedError(f"numba fallback: input type {t}")
        lines.append(f"        x{i} = in{i}(args)")
        names.append(f"x{i}")
    lines.append(f"        r = op_fn({', '.join(names)})")
    n_in = len(node.inputs)
    for k, out in enumerate(node.outputs):
        t = out.type
        if not isinstance(t, ScalarType | TensorType):
            raise NotImplementedError(f"numba fallback: output type {t}")
        ndim = 0 if isinstance(t, ScalarType) else t.ndim
        env[f"out{k}"] = allocated_output(n_in + k, ndim, np.dtype(t.dtype))
        val = "r" if len(node.outputs) == 1 else f"r[{k}]"
        lines.append(f"        v{k} = np.asarray({val})")
        lines.append(f"        o{k} = out{k}(ctx, args, v{k}.shape)")
        lines.append(f"        o{k}[...] = v{k}")
    lines += ["        return 0", "    except Exception:", "        return 1"]
    src = "\n".join(lines)
    # Cache the wrapper on disk through PyTensor's numba cache, keyed by the Op's own
    # cache key and the variable types (the wrapper is fully determined by those).
    cache_key = None
    if op_key is not None and config.numba__cache:
        types_sig = ",".join(str(v.type) for v in (*node.inputs, *node.outputs))
        # hashed: the key ends up in a file name
        digest = hashlib.sha256(f"{op_key}|{types_sig}|fastmath{int(config.numba__fastmath)}".encode()).hexdigest()
        cache_key = f"llvm_fallback_v{FALLBACK_VERSION}_{digest}"
    kernel = compile_numba_function_src(src, "kernel", global_env=env, cache_key=cache_key)
    return numba.cfunc(_SIG, nopython=True, cache=cache_key is not None)(kernel)
