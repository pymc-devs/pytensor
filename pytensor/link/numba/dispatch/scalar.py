import math
from functools import cache
from hashlib import sha256

import numba
import numpy as np
from numba.core import types
from numba.core.extending import get_cython_function_address

from pytensor.graph.basic import Variable
from pytensor.link.numba.cache import _call_cached_ptr, compile_numba_function_src
from pytensor.link.numba.dispatch import _gamma_tables as gamma_tables
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
    Identity,
    Mul,
    Pow,
    Reciprocal,
    ScalarOp,
    Second,
    Sign,
    Switch,
)
from pytensor.scalar.math import (
    Erf,
    Erfc,
    GammaLn,
    Log1mexp,
    Psi,
    Sigmoid,
    Softplus,
)


def scalar_op_cache_key(op, **extra_fields):
    # Scalar Ops don't have _props, because of their weird outputs_types_preference function
    # So we create hash differently
    return sha256(str((type(op), tuple(extra_fields.items()))).encode()).hexdigest()


def register_scalar_funcify(op_type):
    """Register a scalar lowering and preserve its declared output dtypes.

    The lowering returns a Numba function and its cache key. Registration wraps
    that function to narrow promoted results, e.g. int8 * int8 to int64, back to
    each output's dtype. Elemwise normally enforces this when storing into its
    output array; native scalar graphs need the cast explicitly.

    The decorated name still refers to the original lowering, so lowerings can
    reuse each other without nesting these casting wrappers.
    """

    def register(funcify):
        @register_funcify_and_cache_key(op_type)
        def typed_funcify(op, node, **kwargs):
            func, cache_key = funcify(op, node=node, **kwargs)
            out_dtypes = tuple(out.type.dtype for out in node.outputs)
            if all(
                dtype in {"bool", "int64", "uint64", "float64", "complex128"}
                for dtype in out_dtypes
            ):
                return func, cache_key

            input_names = ", ".join(f"x{i}" for i in range(len(node.inputs)))
            call = f"func({input_names})"
            if len(out_dtypes) == 1:
                body = f"return direct_cast({call}, np.dtype({out_dtypes[0]!r}))"
            else:
                casts = ", ".join(
                    f"direct_cast(res[{i}], np.dtype({dtype!r}))"
                    for i, dtype in enumerate(out_dtypes)
                )
                body = f"res = {call}\n    return ({casts},)"
            wrapped = compile_numba_function_src(
                src=f"def cast_scalar_outputs({input_names}):\n    {body}\n",
                function_name="cast_scalar_outputs",
                global_env={
                    "np": np,
                    "func": func,
                    "direct_cast": numba_basic.direct_cast,
                },
            )
            if cache_key is not None:
                cache_version = 1
                cache_key = sha256(
                    str((cache_key, cache_version, out_dtypes)).encode()
                ).hexdigest()
            return numba_basic.numba_njit(wrapped), cache_key

        return funcify

    return register


@register_scalar_funcify(ScalarOp)
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
        scalar_op_src = f"""
def {scalar_op_fn_name}({input_signature}):
{scalar_func_setup}    return scalar_func_numba({input_signature})
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


@register_scalar_funcify(Sign)
def numba_funcify_Sign(op, node, **kwargs):
    dtype = node.inputs[0].type.dtype
    if dtype.startswith("uint"):
        scalar_dtype = np.dtype(dtype).type

        @numba_basic.numba_njit
        def unsigned_sign(x):
            return scalar_dtype(x != 0)

        return unsigned_sign, scalar_op_cache_key(op, unsigned_dtype=dtype)
    if not node.inputs[0].type.dtype.startswith("complex"):
        return numba_funcify_ScalarOp(op, node, **kwargs)

    complex_dtype = np.dtype(node.outputs[0].type.dtype).type

    @numba_basic.numba_njit(fastmath=False)
    def complex_sign(x):
        magnitude = abs(x)
        if np.isnan(magnitude):
            return complex_dtype(complex(np.nan, np.nan))
        if np.isinf(magnitude):
            if np.isinf(x.real):
                if np.isinf(x.imag):
                    return complex_dtype(complex(np.nan, np.nan))
                return complex_dtype(complex(1.0 if x.real > 0 else -1.0, 0.0))
            return complex_dtype(complex(0.0, 1.0 if x.imag > 0 else -1.0))
        if magnitude == 0:
            return complex_dtype(0j)
        return complex_dtype(complex(x.real / magnitude, x.imag / magnitude))

    return complex_sign, scalar_op_cache_key(
        op, complex_dtype=node.outputs[0].type.dtype
    )


@register_scalar_funcify(Switch)
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


@register_scalar_funcify(Pow)
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


@register_scalar_funcify(Add)
def numba_funcify_Add(op, node, **kwargs):
    nary_add_fn = binary_to_nary_func(node.inputs, "add", "+")

    return numba_basic.numba_njit(nary_add_fn), scalar_op_cache_key(op)


@register_scalar_funcify(Mul)
def numba_funcify_Mul(op, node, **kwargs):
    nary_mul_fn = binary_to_nary_func(node.inputs, "mul", "*")

    return numba_basic.numba_njit(nary_mul_fn), scalar_op_cache_key(op)


@register_scalar_funcify(Cast)
def numba_funcify_Cast(op, node, **kwargs):
    dtype = np.dtype(op.o_type.dtype)

    @numba_basic.numba_njit
    def cast(x):
        return numba_basic.direct_cast(x, dtype)

    return cast, sha256(str((type(op), op.o_type.dtype)).encode()).hexdigest()


@register_scalar_funcify(Identity)
def numba_funcify_type_casting(op, **kwargs):
    @numba_basic.numba_njit
    def identity(x):
        return x

    return identity, scalar_op_cache_key(op)


@register_scalar_funcify(Clip)
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


@register_scalar_funcify(Composite)
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


@register_scalar_funcify(Second)
def numba_funcify_Second(op, node, **kwargs):
    @numba_basic.numba_njit
    def second(x, y):
        return y

    return second, scalar_op_cache_key(op)


@register_scalar_funcify(Reciprocal)
def numba_funcify_Reciprocal(op, node, **kwargs):
    @numba_basic.numba_njit
    def reciprocal(x):
        # This is how the C-backend implementation works
        return np.divide(np.float32(1.0), x)

    return reciprocal, scalar_op_cache_key(op, cache_version=1)


@register_scalar_funcify(Sigmoid)
def numba_funcify_Sigmoid(op, node, **kwargs):
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

    @numba_basic.numba_njit
    def sigmoid(x):
        if upcast_uint_dtype is not None:
            # Can't negate uint
            x = numba_basic.direct_cast(x, upcast_uint_dtype)
        # exp(-|x|) is shared with Softplus, so LLVM can reuse it when both are fused
        e = np.exp(-np.abs(x))
        return (1 if x >= 0 else e) / (1 + e)

    return sigmoid, scalar_op_cache_key(op, cache_version=2)


_LGAMMA = gamma_tables.LGAMMA
_PSI = gamma_tables.PSI
_X0_HI = gamma_tables.X0_HI
_X0_LO = gamma_tables.X0_LO
_STIRLING_C = gamma_tables.STIRLING_C
_A0, _A1, _A2, _A3 = gamma_tables.STIRLING
# d/dx sum_k A_k x^-(2k + 1) = -x^-2 sum_k (2k + 1) A_k x^-2k
_B0, _B1, _B2, _B3 = gamma_tables.STIRLING * np.arange(1, 9, 2)
_INV_POW2 = np.array([1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125])


@cache
def _gamma_kernels(njit):
    """float64 lgamma and digamma (for x > 0) from the panels in _gamma_tables.

    [1, 64) is a table lookup and a degree-7 polynomial, with no libm call; below
    1, lgamma(x + 1) - log(x) and psi(x + 1) - 1 / x; from 64, Stirling's series.
    Within 2 ulp (lgamma) and 3 (digamma) of mpmath, next to the zeros at 1, 2
    and 1.4616 too. Only fma contraction is allowed, and LLVM inlines them, so the
    caller's fastmath flags do not reach them: reassociation would drop X0_LO.
    """

    def strict(fn):
        return njit(fn, fastmath={"contract"}, forceinline=True)

    @strict
    def panel_series(table, y):
        # Panel m of octave e of y in [1, 64), 32 panels an octave
        e = int(y >= 2.0) + int(y >= 4.0) + int(y >= 8.0) + int(y >= 16.0)
        e += int(y >= 32.0)
        u = (y * _INV_POW2[e] - 1.0) * 32.0
        m = int(u)
        s = 2.0 * (u - m) - 1.0
        row = 32 * e + m
        p = table[row, 7]
        for k in range(6, -1, -1):
            p = p * s + table[row, k]
        return p

    @strict
    def gammaln(x):
        if x >= 64.0:
            inv = 1.0 / x
            p = inv * inv
            series = inv * (_A0 + p * (_A1 + p * (_A2 + p * _A3)))
            return (x - 0.5) * (np.log(x) - 1.0) + _STIRLING_C + series
        if x > 0.0:
            # the table holds lgamma / ((x - 1) (x - 2)) on [1, 4)
            f = panel_series(_LGAMMA, x if x >= 1.0 else x + 1.0)
            if x >= 4.0:
                return f
            if x >= 1.0:
                return (x - 1.0) * (x - 2.0) * f
            return x * (x - 1.0) * f - np.log(x)
        return math.lgamma(x)

    @strict
    def psi_positive(x):
        if x >= 64.0:
            inv = 1.0 / x
            p = inv * inv
            series = p * (_B0 + p * (_B1 + p * (_B2 + p * _B3)))
            return np.log(x) - 0.5 * inv - series
        y = x if x >= 1.0 else x + 1.0
        r = panel_series(_PSI, y)
        if y < 2.0:
            # r (y - x0) with x0 in two parts; y - _X0_HI is exact
            r = r * (y - _X0_HI) - r * _X0_LO
        return r if x >= 1.0 else r - 1.0 / x

    return gammaln, psi_positive


@register_scalar_funcify(GammaLn)
def numba_funcify_GammaLn(op, node, **kwargs):
    if node.outputs[0].type.dtype != "float64":

        @numba_basic.numba_njit
        def gammaln(x):
            return math.lgamma(x)

        return gammaln, scalar_op_cache_key(op)

    # keyed on numba_njit too, which the pure-Python test path patches
    gammaln_f64, _ = _gamma_kernels(numba_basic.numba_njit)

    @numba_basic.numba_njit
    def gammaln(x):
        return gammaln_f64(np.float64(x))

    return gammaln, scalar_op_cache_key(op, cache_version=1)


@register_scalar_funcify(Psi)
def numba_funcify_Psi(op, node, **kwargs):
    scipy_psi, scipy_key = numba_funcify_ScalarOp(op, node=node, **kwargs)
    if node.outputs[0].type.dtype != "float64":
        return scipy_psi, scipy_key

    _, psi_positive = _gamma_kernels(numba_basic.numba_njit)

    @numba_basic.numba_njit
    def psi(x):
        x = np.float64(x)
        return psi_positive(x) if x > 0.0 else scipy_psi(x)

    return psi, scalar_op_cache_key(op, cache_version=1, scipy=scipy_key)


@register_scalar_funcify(Log1mexp)
def numba_funcify_Log1mexp(op, node, **kwargs):
    @numba_basic.numba_njit
    def logp1mexp(x):
        if x < np.log(0.5):
            return np.log1p(-np.exp(x))
        else:
            return np.log(-np.expm1(x))

    return logp1mexp, scalar_op_cache_key(op)


@register_scalar_funcify(Erf)
def numba_funcify_Erf(op, node, **kwargs):
    if node.inputs[0].type.dtype.startswith("complex"):
        # Complex not supported by numba
        return numba_funcify_ScalarOp(op, node=node, **kwargs)

    @numba_basic.numba_njit
    def erf(x):
        return math.erf(x)

    return erf, scalar_op_cache_key(op)


@register_scalar_funcify(Erfc)
def numba_funcify_Erfc(op, **kwargs):
    @numba_basic.numba_njit
    def erfc(x):
        return math.erfc(x)

    return erfc, scalar_op_cache_key(op)


@register_scalar_funcify(Softplus)
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

    @numba_basic.numba_njit
    def softplus(x):
        if upcast_uint_dtype is not None:
            # Can't negate uint
            x = numba_basic.direct_cast(x, upcast_uint_dtype)
        # Branch-free, and exp(-|x|) is shared with Sigmoid
        value = np.maximum(x, 0) + np.log1p(np.exp(-np.abs(x)))
        return numba_basic.direct_cast(value, out_dtype)

    return softplus, scalar_op_cache_key(op, cache_version=2)


@register_scalar_funcify(ScalarLoop)
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
