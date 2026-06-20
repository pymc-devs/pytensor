import math
from hashlib import sha256

import numba
import numpy as np
from numba.core import types
from numba.core.extending import get_cython_function_address

from pytensor import config
from pytensor.graph.basic import Variable
from pytensor.link.numba.cache import _call_cached_ptr, compile_numba_function_src
from pytensor.link.numba.dispatch import basic as numba_basic
from pytensor.link.numba.dispatch.basic import (
    generate_fallback_impl,
    numba_funcify_and_cache_key,
    register_funcify_and_cache_key,
)
from pytensor.link.numba.dispatch.cython_support import wrap_cython_function
from pytensor.link.utils import (
    get_name_for_object,
)
from pytensor.scalar import ScalarLoop
from pytensor.scalar.basic import (
    Add,
    Cast,
    Clip,
    Composite,
    Expm1,
    Identity,
    Log1p,
    Mul,
    Pow,
    Reciprocal,
    ScalarOp,
    Second,
    Switch,
)
from pytensor.scalar.math import Erf, Erfc, GammaLn, Log1mexp, Sigmoid, Softplus


def scalar_op_cache_key(op, **extra_fields):
    # Scalar Ops don't have _props, because of their weird outputs_types_preference function
    # So we create hash differently
    return sha256(str((type(op), tuple(extra_fields.items()))).encode()).hexdigest()


@numba_basic.numba_njit(fastmath=False, inline="always")
def _log1p_via_log(x):
    # log1p(x) = log(1 + x) * x / ((1 + x) - 1): the factor recovers the bits lost to
    # cancellation in (1 + x) near 0 while lowering to the vectorizable `log` instead of the
    # scalar-only `log1p`. Shared by Log1p and Softplus. `inline="always"` keeps the caller's
    # loop call-free for the vectorizer, so the caller must also be `fastmath=False`: `reassoc`
    # would simplify `(1+x)-1` back to `x`, collapsing log1p to 0 in the underflow tail.
    # `type(x)(1)` keeps the literal in x's dtype (a bare `1` is int64; `int64 + float32` ->
    # float64, doubling the vectorized width on float32).
    one = type(x)(1)
    u = one + x
    um1 = u - one
    return x if um1 == 0 else np.log(u) * x / um1


@register_funcify_and_cache_key(ScalarOp)
def numba_funcify_ScalarOp(op, node, **kwargs):
    if not hasattr(op, "nfunc_spec"):
        return generate_fallback_impl(op, node=node, **kwargs), None

    scalar_func_path = op.nfunc_spec[0]
    scalar_func_numba = None

    *module_path, scalar_func_name = scalar_func_path.split(".")
    if not module_path:
        # Assume it is numpy, and numba has an implementation
        scalar_func_numba = getattr(np, scalar_func_name)

    input_dtypes = [np.dtype(input.type.dtype) for input in node.inputs]
    output_dtypes = [np.dtype(output.type.dtype) for output in node.outputs]

    if len(output_dtypes) != 1:
        raise ValueError("ScalarOps with more than one output are not supported")

    output_dtype = output_dtypes[0]

    input_inner_dtypes = None
    output_inner_dtype = None

    # Cython functions might have an additional argument
    cython_func = None
    has_pyx_skip_dispatch = False

    if scalar_func_path.startswith("scipy.special"):
        import scipy.special.cython_special

        cython_func = getattr(scipy.special.cython_special, scalar_func_name, None)
        if cython_func is not None:
            try:
                scalar_func_numba = wrap_cython_function(
                    cython_func, output_dtype, input_dtypes
                )
            except NotImplementedError:
                pass
            else:
                has_pyx_skip_dispatch = scalar_func_numba.has_pyx_skip_dispatch()
                input_inner_dtypes = scalar_func_numba.input_dtypes
                output_inner_dtype = scalar_func_numba.output_dtype

    if scalar_func_numba is None:
        return generate_fallback_impl(op, node, **kwargs), None

    scalar_op_fn_name = get_name_for_object(scalar_func_numba)
    prefix = "x" if scalar_func_name != "x" else "y"
    input_names = [f"{prefix}{i}" for i in range(len(node.inputs))]
    input_signature = ", ".join(input_names)

    if cython_func is not None:
        # Resolve the cython function pointer at call time, caching it in a module global keyed
        # by `unique_func_name` to make ops backed by `scipy.special.cython_special` cacheable.
        module_name = scalar_func_numba.module_name
        capi_name = scalar_func_numba.capi_name
        unique_func_name = f"{module_name}.{capi_name}"
        func_type_ref = types.FunctionType(scalar_func_numba.signature())

        @numba_basic.numba_njit
        def get_ptr_func():
            with numba.objmode(ptr=types.intp):
                ptr = get_cython_function_address(module_name, capi_name)
            return ptr

        global_env = {
            "_call_cached_ptr": _call_cached_ptr,
            "get_ptr_func": get_ptr_func,
            "func_type_ref": func_type_ref,
            "unique_func_name": unique_func_name,
        }
        scalar_func_setup = (
            "    scalar_func_numba = "
            "_call_cached_ptr(get_ptr_func, func_type_ref, unique_func_name)\n"
        )
    else:
        global_env = {"scalar_func_numba": scalar_func_numba}
        scalar_func_setup = ""

    if input_inner_dtypes is None and output_inner_dtype is None:
        if not has_pyx_skip_dispatch:
            scalar_op_src = f"""
def {scalar_op_fn_name}({input_signature}):
{scalar_func_setup}    return scalar_func_numba({input_signature})
            """
        else:
            scalar_op_src = f"""
def {scalar_op_fn_name}({input_signature}):
{scalar_func_setup}    return scalar_func_numba({input_signature}, np.intc(1))
            """

    else:
        global_env["direct_cast"] = numba_basic.direct_cast
        global_env["output_dtype"] = np.dtype(output_inner_dtype)
        input_tmp_dtype_names = {
            f"inp_tmp_dtype_{i}": i_dtype
            for i, i_dtype in enumerate(input_inner_dtypes)
        }
        global_env.update(input_tmp_dtype_names)
        converted_call_args = ", ".join(
            f"direct_cast({i_name}, {i_tmp_dtype_name})"
            for i_name, i_tmp_dtype_name in zip(
                input_names, input_tmp_dtype_names, strict=False
            )
        )
        if not has_pyx_skip_dispatch:
            scalar_op_src = f"""
def {scalar_op_fn_name}({input_signature}):
{scalar_func_setup}    return direct_cast(scalar_func_numba({converted_call_args}), output_dtype)
            """
        else:
            scalar_op_src = f"""
def {scalar_op_fn_name}({input_signature}):
{scalar_func_setup}    return direct_cast(scalar_func_numba({converted_call_args}, np.intc(1)), output_dtype)
            """

    scalar_op_fn = compile_numba_function_src(
        scalar_op_src,
        scalar_op_fn_name,
        globals() | global_env,
    )

    if cython_func is not None:
        cache_key = scalar_op_cache_key(
            op,
            cython_capi=unique_func_name,
            input_inner_dtypes=tuple(str(d) for d in input_inner_dtypes),
            output_inner_dtype=str(output_inner_dtype),
            has_pyx_skip_dispatch=has_pyx_skip_dispatch,
        )
    else:
        cache_key = scalar_op_cache_key(op)
    return numba_basic.numba_njit(scalar_op_fn), cache_key


@register_funcify_and_cache_key(Switch)
def numba_funcify_Switch(op, node, **kwargs):
    @numba_basic.numba_njit
    def switch(condition, x, y):
        if condition:
            return x
        else:
            return y

    return switch, scalar_op_cache_key(op)


def binary_to_nary_func(inputs: list[Variable], binary_op_name: str, binary_op: str):
    """Create a Numba-compatible N-ary function from a binary function."""
    var_prefix = "x" if binary_op_name != "x" else "y"
    input_names = [f"{var_prefix}{i}" for i in range(len(inputs))]
    input_signature = ", ".join(input_names)
    output_expr = binary_op.join(input_names)

    nary_src = f"""
def {binary_op_name}({input_signature}):
    return {output_expr}
    """
    nary_fn = compile_numba_function_src(nary_src, binary_op_name, globals())

    return nary_fn


@register_funcify_and_cache_key(Pow)
def numba_funcify_Pow(op, node, **kwargs):
    pow_dtype = node.inputs[1].type.dtype

    def pow(x, y):
        return x**y

    # Numba power fails when exponents are discrete integers and fasthmath=True
    # https://github.com/numba/numba/issues/9554
    fastmath = False if np.dtype(pow_dtype).kind in "ibu" else None

    return numba_basic.numba_njit(pow, fastmath=fastmath), scalar_op_cache_key(
        op, cache_version=1
    )


@register_funcify_and_cache_key(Log1p)
def numba_funcify_Log1p(op, node, **kwargs):
    out_dtype = node.outputs[0].dtype
    # `_log1p_via_log` (corrected `log`) vectorizes under a vector library and, on float32, beats
    # scalar `log1p` even without one, at ~1 ulp (glibc `log` near 1 is less accurate than
    # `log1p`'s small-arg path). On float64 with no library it is also ~0.6x SLOWER in the
    # small-arg regime log1p exists for, so there it only pays off vectorized. Use it under a
    # vector library (both dtypes) or float32 + `numba__fastmath` (the opt-in to trade the ulp
    # for a scalar win); else keep scalar `log1p`. The cache key's `corrected` flag keeps the two
    # from aliasing.
    corrected = bool(config.numba__veclib) or (
        out_dtype == "float32" and config.numba__fastmath
    )
    if not corrected:

        @numba_basic.numba_njit(fastmath=False)
        def log1p(x):
            return np.log1p(x)

        return log1p, scalar_op_cache_key(op, corrected=False, cache_version=5)

    @numba_basic.numba_njit(fastmath=False)
    def log1p(x):
        return _log1p_via_log(x)

    return log1p, scalar_op_cache_key(op, corrected=True, cache_version=5)


def _expm1_numba_src(output_dtype):
    # expm1(x) = x + x**2 * p(x), p(x) = sum_j x**j / (j + 2)!  for |x| < ln2, else exp(x) - 1
    # (no cancellation past ln2). The polynomial removes the near-0 cancellation without the
    # `log` the exact correction needs, leaving a single vectorizable `exp`; the caller must keep
    # `fastmath` off so it is evaluated as written. Term count is sized to eps: the tail at
    # |x| = ln2 rounds away, so 8 terms hold float32 to 1 ulp (7 give 3), float64 needs all 15
    # (14 give 2).
    n_terms = 8 if output_dtype == "float32" else 15
    cast = "np.float32" if output_dtype == "float32" else ""

    def lit(c):
        # Every literal must carry the output dtype or a float32 input promotes to float64 (a
        # bare `1` is int64; `int64 + float32` -> float64). Numba unifies both branch return
        # types, so the `np.exp(x) - 1` branch below must go through `lit` too.
        return f"{cast}({c!r})" if cast else repr(c)

    coeffs = [1.0 / math.factorial(j + 2) for j in range(n_terms)]
    poly = lit(coeffs[-1])
    for c in reversed(coeffs[:-1]):
        poly = f"({poly} * x + {lit(c)})"
    return (
        f"def expm1(x):\n"
        f"    if abs(x) < {math.log(2.0)!r}:\n"
        f"        return x + x * x * ({poly})\n"
        f"    return np.exp(x) - {lit(1.0)}\n"
    )


@register_funcify_and_cache_key(Expm1)
def numba_funcify_Expm1(op, node, **kwargs):
    out_dtype = node.outputs[0].dtype
    # Polynomial-near-0 + single `exp`: removes expm1's cancellation while lowering to a
    # vectorizable `exp`. On float32 the 8-term FMA poly plus one `exp` beat scalar `expm1` even
    # with no library; on float64 it is slower in the near-0 regime unless `exp` becomes a SIMD
    # call. Use the poly under a vector library (both dtypes) or float32 + `numba__fastmath` (the
    # opt-in to trade ~0.5 ulp for the scalar win); else fall back to scalar `expm1`. The cache
    # key's `poly` flag keeps the two from aliasing.
    poly = bool(config.numba__veclib) or (
        out_dtype == "float32" and config.numba__fastmath
    )
    if not poly:
        fn, _ = numba_funcify_ScalarOp(op, node, **kwargs)
        return fn, scalar_op_cache_key(op, poly=False, cache_version=4)

    src = _expm1_numba_src(out_dtype)
    expm1 = compile_numba_function_src(src, "expm1", {"np": np})
    return numba_basic.numba_njit(expm1, fastmath=False), scalar_op_cache_key(
        op, poly=True, cache_version=4
    )


@register_funcify_and_cache_key(Add)
def numba_funcify_Add(op, node, **kwargs):
    nary_add_fn = binary_to_nary_func(node.inputs, "add", "+")

    return numba_basic.numba_njit(nary_add_fn), scalar_op_cache_key(op)


@register_funcify_and_cache_key(Mul)
def numba_funcify_Mul(op, node, **kwargs):
    nary_mul_fn = binary_to_nary_func(node.inputs, "mul", "*")

    return numba_basic.numba_njit(nary_mul_fn), scalar_op_cache_key(op)


@register_funcify_and_cache_key(Cast)
def numba_funcify_Cast(op, node, **kwargs):
    dtype = np.dtype(op.o_type.dtype)

    @numba_basic.numba_njit
    def cast(x):
        return numba_basic.direct_cast(x, dtype)

    return cast, sha256(str((type(op), op.o_type.dtype)).encode()).hexdigest()


@register_funcify_and_cache_key(Identity)
def numba_funcify_type_casting(op, **kwargs):
    @numba_basic.numba_njit
    def identity(x):
        return x

    return identity, scalar_op_cache_key(op)


@register_funcify_and_cache_key(Clip)
def numba_funcify_Clip(op, **kwargs):
    @numba_basic.numba_njit
    def clip(x, min_val, max_val):
        if x < min_val:
            return min_val
        elif x > max_val:
            return max_val
        else:
            return x

    return clip, scalar_op_cache_key(op)


@register_funcify_and_cache_key(Composite)
def numba_funcify_Composite(op, node, **kwargs):
    _ = kwargs.pop("storage_map", None)

    composite_fn, fgraph_key = numba_funcify_and_cache_key(
        op.fgraph, squeeze_output=True, fgraph_name="numba_composite", **kwargs
    )
    if fgraph_key is None:
        composite_key = None
    else:
        composite_key = sha256(str((type(op), fgraph_key)).encode()).hexdigest()
    return composite_fn, composite_key


@register_funcify_and_cache_key(Second)
def numba_funcify_Second(op, node, **kwargs):
    @numba_basic.numba_njit
    def second(x, y):
        return y

    return second, scalar_op_cache_key(op)


@register_funcify_and_cache_key(Reciprocal)
def numba_funcify_Reciprocal(op, node, **kwargs):
    @numba_basic.numba_njit
    def reciprocal(x):
        # This is how the C-backend implementation works
        return np.divide(np.float32(1.0), x)

    return reciprocal, scalar_op_cache_key(op, cache_version=1)


@numba_basic.numba_njit(fastmath=False, inline="always")
def _sigmoid_via_exp(x):
    # sigmoid(x) = 1 / (1 + exp(-x)). The `d == 0` guard is DEAD (`1 + exp(-x)` >= 1) but the
    # division-guard select lets the loop vectorizer pull the division through and replace `exp`
    # with a vector call; a plain `1 / (1 + exp(-x))` stays scalar (bare division blocks it, like
    # `_log1p_via_log`'s `um1 == 0` select). `fastmath` off so the select is not proved dead. The
    # `1`/`0` carry x's dtype via `type(x)(...)` (a bare `1` is int64; `int64 + float32` ->
    # float64, halving the SIMD width on float32).
    one = type(x)(1)
    d = one + np.exp(-x)
    return type(x)(0) if d == type(x)(0) else one / d


@register_funcify_and_cache_key(Sigmoid)
def numba_funcify_Sigmoid(op, node, **kwargs):
    inp_dtype = node.inputs[0].type.dtype
    upcast_uint_dtype = {
        "uint8": np.float32,  # numpy uses float16, but not Numba
        "uint16": np.float32,
        "uint32": np.float64,
        "uint64": np.float64,
    }.get(inp_dtype)
    # `_sigmoid_via_exp` vectorizes under a vector library (2.7x float64 / 5.8x float32), but its
    # division-guard select is a no-op (the `d == 0` arm is dead), so it buys nothing scalar.
    # Unlike Log1p/Expm1/Softplus it is not a precision-for-speed trade, so `numba__fastmath` is
    # not the right gate -- only a wired library makes it pay. Gate solely on `numba__veclib`;
    # else emit plain `1 / (1 + exp(-x))`.
    vectorizable = bool(config.numba__veclib)

    if upcast_uint_dtype is not None:
        if vectorizable:

            @numba_basic.numba_njit(fastmath=False)
            def sigmoid(x):
                return _sigmoid_via_exp(numba_basic.direct_cast(x, upcast_uint_dtype))

        else:

            @numba_basic.numba_njit
            def sigmoid(x):
                # Can't negate uint
                float_x = numba_basic.direct_cast(x, upcast_uint_dtype)
                return 1 / (1 + np.exp(-float_x))

    elif vectorizable:

        @numba_basic.numba_njit(fastmath=False)
        def sigmoid(x):
            return _sigmoid_via_exp(x)

    else:

        @numba_basic.numba_njit
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

    return sigmoid, scalar_op_cache_key(op, veclib=vectorizable, cache_version=3)


@register_funcify_and_cache_key(GammaLn)
def numba_funcify_GammaLn(op, node, **kwargs):
    @numba_basic.numba_njit
    def gammaln(x):
        return math.lgamma(x)

    return gammaln, scalar_op_cache_key(op)


@register_funcify_and_cache_key(Log1mexp)
def numba_funcify_Log1mexp(op, node, **kwargs):
    # Mächler (2012) two-branch form with scalar `log1p`. `_log1p_via_log` (trades the `log1p`
    # libcall for `log` + a division) was reverted here: it is fast only when its argument is
    # away from 0 -- glibc `log(1+a)` is slow for `1+a` near 1, while `log1p` has a fast
    # small-argument path. This branch always feeds it small `a = -exp(x)`, the slow regime: a
    # ~30% float64 regression for no accuracy gain. (Verified to be the argument range, not the
    # `exp` or the branch: the corrected form alone flips from 1.8x faster on x in (-0.5, 5) to
    # 0.44x slower as x -> 0.)
    @numba_basic.numba_njit
    def logp1mexp(x):
        if x < np.log(0.5):
            return np.log1p(-np.exp(x))
        else:
            return np.log(-np.expm1(x))

    return logp1mexp, scalar_op_cache_key(op)


@register_funcify_and_cache_key(Erf)
def numba_funcify_Erf(op, node, **kwargs):
    if node.inputs[0].type.dtype.startswith("complex"):
        # Complex not supported by numba
        return numba_funcify_ScalarOp(op, node=node, **kwargs)

    @numba_basic.numba_njit
    def erf(x):
        return math.erf(x)

    return erf, scalar_op_cache_key(op)


@register_funcify_and_cache_key(Erfc)
def numba_funcify_Erfc(op, **kwargs):
    @numba_basic.numba_njit
    def erfc(x):
        return math.erfc(x)

    return erfc, scalar_op_cache_key(op)


@register_funcify_and_cache_key(Softplus)
def numba_funcify_Softplus(op, node, **kwargs):
    inp_dtype = node.inputs[0].type.dtype
    if inp_dtype.startswith("uint"):
        upcast_uint_dtype = {
            "uint8": np.float32,  # numpy uses float16, but not Numba
            "uint16": np.float32,
            "uint32": np.float64,
            "uint64": np.float64,
        }[inp_dtype]
    else:
        upcast_uint_dtype = None
    out_dtype = np.dtype(node.outputs[0].type.dtype)

    # Branchless `max(x, 0) + log1p(exp(-|x|))` is ~1.1-1.2x faster than the cascade for the
    # common mixed-sign case and vectorizes under a vector library, but routes log1p through the
    # corrected `log`, costing ~1 ulp. Use it under a vector library or `numba__fastmath` (the
    # opt-in to trade that ulp for speed); else use the accurate Mächler cascade. The cache key's
    # `branchless` flag keeps the two from aliasing.
    if bool(config.numba__veclib) or config.numba__fastmath:
        # Branch-free softplus: max(x, 0) + log1p(exp(-|x|)). Equivalent to the Mächler cascade
        # but feeds exp/log1p an argument in (0, 1] (their cheap regime) without a per-element
        # branch: ~1.5x faster on the common case, never overflows. log1p goes through
        # `_log1p_via_log` for the vectorizable `log`/`exp` (plain `np.log1p` has no vector form).
        # `type(x)(0)` keeps the literal in x's dtype (`max(float32, float64)` would return
        # float64, doubling the width). fastmath off so the inlined `(1+x)-1` is not reassociated
        # to `x` (collapsing log1p to 0 in the underflow tail); vectorization does not need it.
        @numba_basic.numba_njit(fastmath=False)
        def softplus(x):
            if upcast_uint_dtype is not None:
                # Can't negate uint; upcast once so the formula below is uniform.
                x = numba_basic.direct_cast(x, upcast_uint_dtype)
            value = max(x, type(x)(0)) + _log1p_via_log(np.exp(-abs(x)))
            return numba_basic.direct_cast(value, out_dtype)

        return softplus, scalar_op_cache_key(op, branchless=True, cache_version=5)

    @numba_basic.numba_njit
    def softplus(x):
        if x < -37.0:
            value = np.exp(x)
        elif x < 18.0:
            value = np.log1p(np.exp(x))
        elif x < 33.3:
            if upcast_uint_dtype is not None:
                # Can't negate uint
                x = numba_basic.direct_cast(x, upcast_uint_dtype)
            value = x + np.exp(-x)
        else:
            value = x
        return numba_basic.direct_cast(value, out_dtype)

    return softplus, scalar_op_cache_key(op, branchless=False, cache_version=5)


@register_funcify_and_cache_key(ScalarLoop)
def numba_funcify_ScalarLoop(op, node, **kwargs):
    inner_fn, inner_fn_cache_key = numba_funcify_and_cache_key(op.fgraph)
    if inner_fn_cache_key is None:
        loop_cache_key = None
    else:
        loop_cache_key = sha256(
            str((type(op), op.is_while, inner_fn_cache_key)).encode()
        ).hexdigest()

    if op.is_while:
        n_update = len(op.outputs) - 1

        @numba_basic.numba_njit
        def while_loop(n_steps, *inputs):
            carry, constant = inputs[:n_update], inputs[n_update:]

            until = False
            for i in range(n_steps):
                outputs = inner_fn(*carry, *constant)
                carry, until = outputs[:-1], outputs[-1]
                if until:
                    break

            return *carry, until

        return while_loop, loop_cache_key

    else:
        n_update = len(op.outputs)

        @numba_basic.numba_njit
        def for_loop(n_steps, *inputs):
            carry, constant = inputs[:n_update], inputs[n_update:]

            if n_steps < 0:
                raise ValueError("ScalarLoop does not have a termination condition.")

            for i in range(n_steps):
                carry = inner_fn(*carry, *constant)

            if n_update == 1:
                return carry[0]
            else:
                return carry

        return for_loop, loop_cache_key
