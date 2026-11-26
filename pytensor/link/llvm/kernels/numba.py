"""Numba kernels: Ops we don't implement, run through their numba implementation.

For such an Op we take PyTensor's numba implementation (``numba_funcify``) and wrap it in
a numba ``cfunc`` with the arguments of the kernel ABI, ``int32(PtCtx *ctx, PtArray *args,
int32 n_inputs, int32 n_outputs, void *config)``, run by the runtime's ``numba_entry``
kernel (its config is the cfunc's address):

* inputs become numba arrays that view the caller's memory (data, shape, strides; no copy),
  or plain scalars for inputs of ``ScalarType``,
* every output is *kernel-allocated*: the wrapper calls ``ctx->rt->alloc`` through the
  runtime table and copies numba's result into that buffer, so ownership stays with our
  runtime,
* it returns 1 on success. numba's cfunc wrapper prints an exception as "unraisable"
  and returns 0, which ``numba_entry`` reports as an error.

Random generators have no data in the backend (`is_absent`): a random variable draws
from the call context's random state. A kernel runs one core draw of it (no size, no
batch axes; the backend loops over the batch: dispatch/random.py), with numba's core
function for it (`numba_core_rv_funcify`). That gets a numba Generator whose bit
generator is numpy's ``bitgen_t`` interface over the call's state (`ctx_generator`);
the next generator isn't an output.

Any other op taking a generator (a random variable without a numba core function, or
without its core shape) runs its ``perform`` in object mode, given a numpy Generator
seeded from the call's state (`_perform_with_seeded_generator`).

The wrapper needs no Python or GIL when the numba implementation is in nopython mode
(numba's object-mode fallbacks take the GIL themselves).
"""

from __future__ import annotations

import hashlib

import numba
import numpy as np
from llvmlite import ir
from numba import types
from numba.core import cgutils, sigutils
from numba.core.ccallback import CFunc
from numba.core.compiler_lock import global_compiler_lock
from numba.extending import intrinsic
from numba.np.arrayobj import populate_array

from pytensor.scalar.basic import ScalarType
from pytensor.tensor.random.op import RandomVariable
from pytensor.tensor.random.type import RandomType
from pytensor.tensor.type import TensorType
from pytensor.tensor.type_other import NoneTypeT


PTARRAY_SIZE = 40  # sizeof(PtArray): data, shape, strides, owner, (ndim, dtype)

_i8p = ir.IntType(8).as_pointer()
_i64 = ir.IntType(64)
_i64p = _i64.as_pointer()


def _runtime_offsets() -> dict[str, int]:
    """The offset of each function in the runtime table (`ctx->rt`)."""
    from pytensor_llvm import runtime

    return runtime.RUNTIME_OFFSETS


def _runtime_function(b, ctx_p, name, ty):
    """Load the runtime function `name` from the table `ctx->rt`, as a `ty`."""
    rt = b.load(b.bitcast(ctx_p, _i8p.as_pointer()))
    p = b.gep(rt, [ir.Constant(_i64, _runtime_offsets()[name])])
    return b.load(b.bitcast(p, ty.as_pointer()))


def _field(b, base, arg_index, field, ty):
    off = arg_index * PTARRAY_SIZE + 8 * field
    p = b.gep(b.bitcast(base, _i8p), [ir.Constant(_i64, off)])
    return b.load(b.bitcast(p, ty.as_pointer()))


def _make_array(context, b, arrty, data, shape, strides):
    ary = context.make_array(arrty)(context, b)
    itemsize = context.get_abi_sizeof(context.get_data_type(arrty.dtype))
    intp = context.get_value_type(types.intp)
    populate_array(
        ary,
        data=b.bitcast(data, ary.data.type),
        shape=cgutils.pack_array(b, shape, ty=intp),
        strides=cgutils.pack_array(b, strides, ty=intp),
        itemsize=context.get_constant(types.intp, itemsize),
        meminfo=None,  # borrowed memory
    )
    return ary._getvalue()


def array_input(index: int, ndim: int, dtype: np.dtype):
    """Intrinsic: numba array viewing args[index] (any strides, no copy)."""
    arrty = types.Array(numba.from_dtype(dtype), ndim, "A")

    @intrinsic
    def view(typingctx, args):
        def codegen(context, b, sig, llargs):
            (base,) = llargs
            data = _field(b, base, index, 0, _i8p)
            shp = _field(b, base, index, 1, _i64)
            strd = _field(b, base, index, 2, _i64)
            shp, strd = b.inttoptr(shp, _i64p), b.inttoptr(strd, _i64p)
            shape = [b.load(b.gep(shp, [ir.Constant(_i64, d)])) for d in range(ndim)]
            strides = [b.load(b.gep(strd, [ir.Constant(_i64, d)])) for d in range(ndim)]
            return _make_array(context, b, arrty, data, shape, strides)

        return arrty(args), codegen

    return view


def scalar_input(index: int, dtype: np.dtype):
    """Intrinsic: load the scalar stored at args[index].data."""
    nbty = numba.from_dtype(dtype)

    @intrinsic
    def load(typingctx, args):
        def codegen(context, b, sig, llargs):
            (base,) = llargs
            data = _field(b, base, index, 0, _i8p)
            llty = context.get_data_type(nbty)
            v = b.load(b.bitcast(data, llty.as_pointer()))
            return context.data_model_manager[nbty].from_data(b, v)

        return nbty(args), codegen

    return load


def allocated_output(index: int, ndim: int, dtype: np.dtype, order=None):
    """Intrinsic: allocate args[index] through ctx->rt->alloc with the given shape,
    packed in axis `order` (outermost first; default C), fill the PtArray (data, owner,
    shape, strides) and return a numba view of it."""
    nbdt = numba.from_dtype(dtype)
    order = tuple(range(ndim)) if order is None else tuple(order)
    arrty = types.Array(nbdt, ndim, "C" if order == tuple(range(ndim)) else "A")
    itemsize = np.dtype(dtype).itemsize

    @intrinsic
    def alloc(typingctx, ctx, args, shape):
        if not isinstance(shape, types.BaseTuple) or len(shape) != ndim:
            return None

        def codegen(context, b, sig, llargs):
            ctx_p, base, shape_v = llargs
            lengths = [
                context.cast(
                    b,
                    b.extract_value(shape_v, d),
                    sig.args[2][d],
                    types.int64,
                )
                for d in range(ndim)
            ]
            n = ir.Constant(_i64, itemsize)
            for length in lengths:
                n = b.mul(n, length)
            # data = ctx->rt->alloc(ctx, n, 64, &owner)
            fnty = ir.FunctionType(_i8p, [_i8p, _i64, _i64, _i8p.as_pointer()])
            fptr = _runtime_function(b, ctx_p, "alloc", fnty.as_pointer())
            owner = cgutils.alloca_once(b, _i8p)
            data = b.call(
                fptr, [b.bitcast(ctx_p, _i8p), n, ir.Constant(_i64, 64), owner]
            )
            # fill the PtArray: data, owner, shape, strides
            entry = b.gep(
                b.bitcast(base, _i8p), [ir.Constant(_i64, index * PTARRAY_SIZE)]
            )

            def slot(k, ty):
                return b.bitcast(
                    b.gep(entry, [ir.Constant(_i64, 8 * k)]), ty.as_pointer()
                )

            b.store(data, slot(0, _i8p))
            b.store(b.load(owner), slot(3, _i8p))
            shp = b.inttoptr(b.load(slot(1, _i64)), _i64p)
            strd = b.inttoptr(b.load(slot(2, _i64)), _i64p)
            strides = [None] * ndim
            acc = ir.Constant(_i64, itemsize)
            for d in reversed(order):
                strides[d] = acc
                acc = b.mul(acc, lengths[d])
            for d in range(ndim):
                b.store(lengths[d], b.gep(shp, [ir.Constant(_i64, d)]))
                b.store(strides[d], b.gep(strd, [ir.Constant(_i64, d)]))
            return _make_array(context, b, arrty, data, lengths, strides)

        return arrty(ctx, args, shape), codegen

    return alloc


#: The bit generator members of numba's model of `np.random.Generator`
#: (`numba.np.random.generator_core`) that `ctx_generator` fills: numba's
#: distributions draw by calling `fnptr_next_*(state)`.
BIT_GENERATOR_MEMBERS = (
    "parent",
    "state_address",
    "state",
    "fnptr_next_uint64",
    "fnptr_next_uint32",
    "fnptr_next_double",
    "bit_generator",
)


@intrinsic
def ctx_generator(typingctx, ctx):
    """Intrinsic: a numba Generator drawing from the call context's random state.

    Its bit generator is numpy's `bitgen_t` interface over the context: the state is
    `ctx`, and `next_uint64`, `next_uint32` and `next_double` are the runtime's
    `rng_u64`, `rng_u32` and `rng_uniform`. It has no Python object (`parent`), so
    object mode can't use it, nor copy it."""
    gen_ty = types.npy_rng

    def codegen(context, b, sig, llargs):
        (ctx_p,) = llargs
        bitgen_ty = context.data_model_manager[gen_ty].get_member_fe_type(
            "bit_generator"
        )
        # (zero-filled: no Python object, no meminfo)
        bitgen = cgutils.create_struct_proxy(bitgen_ty)(context, b)
        state = b.ptrtoint(ctx_p, _i64)
        bitgen.state = state
        bitgen.state_address = state
        for member, name in (
            ("fnptr_next_uint64", "rng_u64"),
            ("fnptr_next_uint32", "rng_u32"),
            ("fnptr_next_double", "rng_uniform"),
        ):
            setattr(bitgen, member, _runtime_function(b, ctx_p, name, _i64))
        gen = cgutils.create_struct_proxy(gen_ty)(context, b)
        gen.bit_generator = bitgen._getvalue()
        return gen._getvalue()

    return gen_ty(ctx), codegen


def _is_core_draw(node) -> bool:
    """Is `node` a random variable's core draw (no size, no batch axes: the core of a
    `RandomVariableWithCoreShape`)?"""
    op = node.op
    return (
        isinstance(op, RandomVariable)
        and isinstance(op.size_param(node).type, NoneTypeT)
        and op.batch_ndim(node) == 0
    )


def _core_draw_function(node):
    """numba's core function for the random variable `node`, a core draw: `fn(rng,
    *params) -> draw`; and its cache key (None: not cacheable). NotImplementedError if
    numba has none."""
    from pytensor.link.numba.dispatch.random import numba_core_rv_funcify

    op = node.op
    match numba_core_rv_funcify(op, node):
        case (fn, (int() | None) as key):
            pass
        case fn:
            key = 0
    if key is None:
        return fn, None
    props = sorted((k, str(v)) for k, v in op._props_dict().items())
    return fn, f"core_draw|{type(op).__module__}.{type(op).__qualname__}|{props}|{key}"


def _perform_with_seeded_generator(node):
    """`fn(seed0, seed1, *values) -> (outputs with data)`: the op's `perform` in object
    mode, given a numpy Generator seeded with (seed0, seed1) for its generator inputs
    (`values`: the inputs with data, in order).

    For an op that takes a generator but has no implementation drawing from the call's
    state: the seed comes from that state, so seeding the session reproduces the draws
    and every call draws others. The draws come from numpy's stream, not the call's."""
    import warnings

    from pytensor.link.numba.dispatch.basic import get_numba_type, numba_njit

    warnings.warn(
        f"Numba will use object mode to run {node.op}'s perform method, with a numpy "
        "Generator seeded from the call's random state",
        UserWarning,
    )
    data_outputs = [
        k for k, out in enumerate(node.outputs) if not isinstance(out.type, RandomType)
    ]
    ret_sig = types.Tuple([get_numba_type(node.outputs[k].type) for k in data_outputs])

    def py_perform(seed0, seed1, values):
        rng = np.random.default_rng([seed0, seed1])
        values = iter(values)
        inputs = [
            rng
            if isinstance(inp.type, RandomType)
            else None
            if isinstance(inp.type, NoneTypeT)
            else next(values)
            for inp in node.inputs
        ]
        storage = [[None] for _ in node.outputs]
        node.op.perform(node, inputs, storage)
        return tuple(storage[k][0] for k in data_outputs)

    @numba_njit
    def perform(seed0, seed1, *values):
        with numba.objmode(ret=ret_sig):
            ret = py_perform(seed0, seed1, values)
        return ret

    return perform


_SIG = types.int32(
    types.voidptr, types.voidptr, types.int32, types.int32, types.voidptr
)


class _CFunc(CFunc):
    """numba's CFunc, but keeping its CompileResult.

    The code of a cfunc refers to the environments of the functions compiled into it
    (globals and constants: what object mode needs to box values). numba's
    `CFunc.compile` keeps only the library, and the environments rebuilt when loading
    from the cache are held by nothing else (`Environment._memo` is weak), so they are
    freed at once: object mode then fails ("'descr' is NULL"), or worse. A fresh compile
    happens to work, as the callees' dispatchers keep their environments alive.
    """

    @global_compiler_lock
    def compile(self):
        cres = self._cache.load_overload(self._sig, self._targetdescr.target_context)
        if cres is None:
            cres = self._compile_uncached()
            self._cache.save_overload(self._sig, cres)
        else:
            self._cache_hits += 1
        self._cres = cres
        self._library = cres.library
        self._wrapper_name = cres.fndesc.llvm_cfunc_wrapper_name
        self._wrapper_address = self._library.get_pointer_to_function(
            self._wrapper_name
        )


# bump when the generated wrapper changes
CACHE_VERSION = 9


def numba_kernel(node, orders=None):
    """Compile a cfunc kernel for `node`; returns the numba cfunc object (keep it
    alive). `orders`: per output, the axis order (outermost first) it is allocated
    packed in (None: C)."""
    from pytensor import config
    from pytensor.link.numba.cache import compile_numba_function_src
    from pytensor.link.numba.dispatch.basic import numba_funcify_ensure_cache

    # how the op runs: numba's core function of a core draw ("draw"), `perform` given
    # a seeded Generator ("seeded": other ops taking a generator), or numba's
    # implementation of the op ("op")
    kind = "op"
    if _is_core_draw(node):
        try:
            op_fn, op_key = _core_draw_function(node)
            kind = "draw"
        except NotImplementedError:
            kind = "seeded"
    elif any(isinstance(inp.type, RandomType) for inp in node.inputs):
        kind = "seeded"
    if kind == "draw":
        # (the core function takes the generator and the parameters)
        called = {node.op.rng_param(node), *node.op.dist_params(node)}
    elif kind == "seeded":
        op_fn, op_key = _perform_with_seeded_generator(node), None
        called = {
            inp
            for inp in node.inputs
            if not isinstance(inp.type, RandomType | NoneTypeT)
        }
    else:
        op_fn, op_key = numba_funcify_ensure_cache(node.op, node=node)
        called = set(node.inputs)
    # module globals give numba a real module (`__name__`) to rebuild cached environments from
    env = globals() | {"op_fn": op_fn, "np": np}
    lines = ["def kernel(ctx, args, n_inputs, n_outputs, params):"]
    names = []
    if kind == "seeded":
        lines.append("    seeds = ctx_generator(ctx)")
        for k in range(2):
            lines.append(f"    s{k} = seeds.integers(0, {np.iinfo(np.int64).max})")  # noqa: PERF401
        names += ["s0", "s1"]
    # (values without data aren't arguments: `position` counts those that are)
    position = 0
    for i, inp in enumerate(node.inputs):
        t = inp.type
        if inp in called:
            names.append(f"x{i}")
        if isinstance(t, RandomType):
            lines.append(f"    x{i} = ctx_generator(ctx)")
            continue
        if isinstance(t, NoneTypeT):
            lines.append(f"    x{i} = None")
            continue
        if isinstance(t, ScalarType):
            env[f"in{i}"] = scalar_input(position, np.dtype(t.dtype))
        elif isinstance(t, TensorType):
            env[f"in{i}"] = array_input(position, t.ndim, np.dtype(t.dtype))
        else:
            raise NotImplementedError(f"numba kernel: input type {t}")
        position += 1
        # (a core draw's parameters without axes are scalars, as numba's sampling
        # functions take them)
        scalar = kind == "draw" and isinstance(t, TensorType) and t.ndim == 0
        lines.append(f"    x{i} = in{i}(args){'[()]' if scalar else ''}")
    lines.append(f"    r = op_fn({', '.join(names)})")
    # (the result: the draw, the outputs with data (a tuple), or numba's result)
    j = 0
    for k, out in enumerate(node.outputs):
        t = out.type
        if isinstance(t, RandomType):
            continue  # (the next generator: the call's state has advanced)
        if not isinstance(t, ScalarType | TensorType):
            raise NotImplementedError(f"numba kernel: output type {t}")
        ndim = 0 if isinstance(t, ScalarType) else t.ndim
        order = None if orders is None else orders[k]
        env[f"out{k}"] = allocated_output(position, ndim, np.dtype(t.dtype), order)
        position += 1
        if kind == "draw" or (kind == "op" and len(node.outputs) == 1):
            val = "r"
        else:
            val = f"r[{j if kind == 'seeded' else k}]"
        j += 1
        lines.append(f"    v{k} = np.asarray({val})")
        lines.append(f"    o{k} = out{k}(ctx, args, v{k}.shape)")
        lines.append(f"    o{k}[...] = v{k}")
    # no try/except: numba lets execution go on after an object mode exception inside
    # one (with an empty result); uncaught, its cfunc wrapper reports it and returns 0
    lines.append("    return 1")
    src = "\n".join(lines)
    # Cache the wrapper on disk through PyTensor's numba cache, keyed by the Op's own
    # cache key and the variable types (the wrapper is fully determined by those), and
    # this module's name: a cached kernel imports it again when it is loaded.
    cache_key = None
    if op_key is not None and config.numba__cache:
        types_sig = ",".join(str(v.type) for v in (*node.inputs, *node.outputs))
        types_sig += f"|orders{orders}|runtime{sorted(_runtime_offsets().items())}"
        # hashed: the key ends up in a file name
        fastmath = int(config.numba__fastmath)
        key = f"{__name__}|{op_key}|{types_sig}|fastmath{fastmath}"
        digest = hashlib.sha256(key.encode()).hexdigest()
        cache_key = f"llvm_numba_kernel_v{CACHE_VERSION}_{digest}"
    kernel = compile_numba_function_src(
        src, "kernel", global_env=env, cache_key=cache_key
    )
    signature = sigutils.normalize_signature(_SIG)
    cfunc = _CFunc(kernel, signature, locals={}, options={"nopython": True})
    if cache_key is not None:
        cfunc.enable_caching()
    cfunc.compile()
    return cfunc
