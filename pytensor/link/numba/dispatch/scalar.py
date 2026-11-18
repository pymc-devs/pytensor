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
from pytensor.link.numba.dispatch import _gammainc_tables as gammainc_tables
from pytensor.link.numba.dispatch import basic as numba_basic
from pytensor.link.numba.dispatch._gammainc import make_gammainc_kernel
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
    BaseComposite,
    Cast,
    Clip,
    Identity,
    Mul,
    Pow,
    Reciprocal,
    ScalarOp,
    Second,
    Sign,
    Switch,
    float64,
)
from pytensor.scalar.math import (
    Erf,
    Erfc,
    GammaIncAndGrad,
    GammaIncGrad,
    GammaLn,
    Log1mexp,
    Psi,
    Sigmoid,
    Softplus,
    gammainc,
    gammaincc,
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


@register_scalar_funcify(BaseComposite)
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


@cache
def _gamma_kernels(njit, dtype):
    """lgamma and digamma (for x > 0) in `dtype` from the panels in _gamma_tables.

    [1, 64) is a table lookup and a polynomial, with no libm call; below 1,
    lgamma(x + 1) - log(x) and psi(x + 1) - 1 / x; from 64, Stirling's series.
    float64 is within 2 ulp (lgamma) and 3 (digamma) of mpmath, float32 within 3,
    next to the zeros at 1, 2 and 1.4616 too. float32 has its own shorter tables
    and every operation stays in float32. Only fma contraction is allowed, and
    LLVM inlines them, so the caller's fastmath flags do not reach them:
    reassociation would drop x0's low part.
    """
    real = np.dtype(dtype).type
    if dtype == "float32":
        lgamma_table, psi_table = gamma_tables.LGAMMA32, gamma_tables.PSI32
        x0_hi, x0_lo = gamma_tables.X0_HI32, gamma_tables.X0_LO32
        stirling = gamma_tables.STIRLING32
    else:
        lgamma_table, psi_table = gamma_tables.LGAMMA, gamma_tables.PSI
        x0_hi, x0_lo = gamma_tables.X0_HI, gamma_tables.X0_LO
        stirling = gamma_tables.STIRLING
    # d/dx sum_k A_k x^-(2k + 1) = -x^-2 sum_k (2k + 1) A_k x^-2k
    dstirling = (stirling * np.arange(1, 2 * stirling.size, 2)).astype(dtype)
    stirling_c = real(gamma_tables.STIRLING_C)
    inv_pow2 = np.array([1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125], dtype=dtype)
    half, one, two, four, eight = map(real, (0.5, 1, 2, 4, 8))
    sixteen, thirty_two, sixty_four = map(real, (16, 32, 64))

    def strict(fn):
        return njit(fn, fastmath={"contract"}, forceinline=True)

    @strict
    def horner(coefs, row, s):
        top = coefs.shape[1] - 1
        p = coefs[row, top]
        for k in range(top - 1, -1, -1):
            p = p * s + coefs[row, k]
        return p

    @strict
    def panel_series(table, y):
        # Panel m of octave e of y in [1, 64), 32 panels an octave
        e = int(y >= two) + int(y >= four) + int(y >= eight) + int(y >= sixteen)
        e += int(y >= thirty_two)
        u = (y * inv_pow2[e] - one) * thirty_two
        m = int(u)
        return horner(table, 32 * e + m, two * (u - real(m)) - one)

    stirling_row = stirling.reshape(1, -1)
    dstirling_row = dstirling.reshape(1, -1)

    @strict
    def gammaln(x):
        if x >= sixty_four:
            inv = one / x
            series = inv * horner(stirling_row, 0, inv * inv)
            return (x - half) * (np.log(x) - one) + stirling_c + series
        if x > 0:
            # the table holds lgamma / ((x - 1) (x - 2)) on [1, 4)
            f = panel_series(lgamma_table, x if x >= one else x + one)
            if x >= four:
                return f
            if x >= one:
                return (x - one) * (x - two) * f
            return x * (x - one) * f - np.log(x)
        return real(math.lgamma(x))

    @strict
    def psi_positive(x):
        if x >= sixty_four:
            inv = one / x
            p = inv * inv
            return np.log(x) - half * inv - p * horner(dstirling_row, 0, p)
        y = x if x >= one else x + one
        r = panel_series(psi_table, y)
        if y < two:
            # r (y - x0) with x0 in two parts; y - x0_hi is exact
            r = r * (y - x0_hi) - r * x0_lo
        return r if x >= one else r - one / x

    return gammaln, psi_positive


@register_scalar_funcify(GammaLn)
def numba_funcify_GammaLn(op, node, **kwargs):
    dtype = node.outputs[0].type.dtype
    if dtype not in ("float32", "float64"):

        @numba_basic.numba_njit
        def gammaln(x):
            return math.lgamma(x)

        return gammaln, scalar_op_cache_key(op)

    # keyed on numba_njit too, which the pure-Python test path patches
    kernel, _ = _gamma_kernels(numba_basic.numba_njit, dtype)
    real = np.dtype(dtype).type

    @numba_basic.numba_njit
    def gammaln(x):
        return kernel(real(x))

    return gammaln, scalar_op_cache_key(op, cache_version=3)


@register_scalar_funcify(Psi)
def numba_funcify_Psi(op, node, **kwargs):
    scipy_psi, scipy_key = numba_funcify_ScalarOp(op, node=node, **kwargs)
    dtype = node.outputs[0].type.dtype
    if dtype not in ("float32", "float64"):
        return scipy_psi, scipy_key

    _, kernel = _gamma_kernels(numba_basic.numba_njit, dtype)
    real = np.dtype(dtype).type

    @numba_basic.numba_njit
    def psi(x):
        x = real(x)
        # scipy's psi, in double, below 0
        return kernel(x) if x > 0 else real(scipy_psi(x))

    return psi, scalar_op_cache_key(op, cache_version=3, scipy=scipy_key)


def _dyadic_cells(cuts, counts):
    """Put each zone's cuts on the coarsest dyadic grid that holds them all, so finding
    a patch is one floor: per zone the low end, the fine width, the last fine cell and
    its first row; per fine cell its patch interval and that interval's ends."""
    lo, width, last, first, cell, cell_lo, cell_hi = [], [], [], [], [], [], []
    for row, n in zip(cuts, counts, strict=True):
        row = [float(c) for c in row[: n + 1]]
        span = row[-1] - row[0]
        fine = 1
        while any(
            abs((c - row[0]) / span * fine - round((c - row[0]) / span * fine)) > 1e-6
            for c in row
        ):
            fine *= 2
        marks = [round((c - row[0]) / span * fine) for c in row]
        lo.append(row[0])
        width.append(span / fine)
        last.append(fine - 1)
        first.append(len(cell))
        for m in range(fine):
            i = max(q for q in range(n) if marks[q] <= m)
            cell.append(i)
            cell_lo.append(row[i])
            cell_hi.append(row[i + 1])
    return lo, width, last, first, cell, cell_lo, cell_hi


def _chebyshev_2d_src(n_arg, n_par):
    """Source of ``cheb(c, sy, sp)``: the sum of c[a, b] T_a(sy) T_b(sp) and its sy and
    sp derivatives, unrolled."""
    lines = ["def cheb(c, sy, sp):"]
    for v, s, n in (("p", "sp", n_par), ("y", "sy", n_arg)):
        lines += [
            f"    {v}0 = one",
            f"    {v}1 = {s}",
            f"    d{v}0 = zero",
            f"    d{v}1 = one",
        ]
        for i in range(2, n):
            lines.append(f"    {v}{i} = two * {s} * {v}{i - 1} - {v}{i - 2}")
            lines.append(
                f"    d{v}{i} = two * {v}{i - 1} + two * {s} * d{v}{i - 1} - d{v}{i - 2}"
            )
    for a in range(n_arg):
        lines.append(
            f"    r{a} = " + " + ".join(f"c[{a}, {b}] * p{b}" for b in range(n_par))
        )
        lines.append(
            f"    q{a} = " + " + ".join(f"c[{a}, {b}] * dp{b}" for b in range(n_par))
        )
    lines.append("    f = " + " + ".join(f"y{a} * r{a}" for a in range(n_arg)))
    lines.append("    fy = " + " + ".join(f"dy{a} * r{a}" for a in range(n_arg)))
    lines.append("    fp = " + " + ".join(f"y{a} * q{a}" for a in range(n_arg)))
    lines.append("    return f, fy, fp")
    return "\n".join(lines)


@cache
def _gammainc_patch_kernel(njit, dtype):
    """``(v, dv/dk, is_p)`` with v = ln P (is_p) or ln Q, for k >= ALO and finite x > 0.

    chebax's gammainc_array: the zone, set by k and x / k, fixes an analytic leading
    term of ln P or ln Q and the patch grid of its Chebyshev correction. A0 and B0 are
    k < 1 (P below x = k + 1, Q above), L1 and L2 the lower tail x / k <= LAM_LO, U1
    and U2 the upper tail x / k >= LAM_HI (L2 and U2 for k > ASPLIT), and T the rest,
    around x = k. float32 computes in float32 on the leading KY32 x KP32 coefficients
    of each patch.
    """
    t = gammainc_tables
    A0, B0, L1, U1, L2, U2, T = map(
        t.NAMES.index, ("A0", "B0", "L1", "U1", "L2", "U2", "T")
    )
    real = np.dtype(dtype).type
    n_arg, n_par = (t.KY32, t.KP32) if dtype == "float32" else t.TABLE.shape[1:]
    table = np.ascontiguousarray(t.TABLE[:, :n_arg, :n_par], dtype)
    narg, offset = np.array(t.NARG, np.int64), np.array(t.OFFSET, np.int64)
    luts = []
    for cuts, counts in ((t.PCUTS, t.NPAR), (t.YCUTS, t.NARG)):
        lo, width, last, first, cell, cell_lo, cell_hi = _dyadic_cells(cuts, counts)
        luts.append(
            (
                *(np.array(v, dtype) for v in (lo, width, cell_lo, cell_hi)),
                *(np.array(v, np.int64) for v in (last, first, cell)),
            )
        )
    (p_lo, p_width, p_cell_lo, p_cell_hi, p_last, p_first, p_cell) = luts[0]
    (y_lo, y_width, y_cell_lo, y_cell_hi, y_last, y_first, y_cell) = luts[1]
    erfcx_p = np.ascontiguousarray(
        t.ERFCX_P[:, : 5 if dtype == "float32" else 9], dtype
    )
    erfcx_k, erfcx_n = real(t.K), int(t.NP)
    asplit = real(t.ASPLIT)
    lam_lo, lam_hi = real(t.LAM_LO), real(t.LAM_HI)
    zero, half, one, two = real(0), real(0.5), real(1), real(2)
    tiny = real(np.finfo(dtype).tiny)
    phi_series = np.array(
        [1.0 if i % 2 == 0 else (i + 1) / (i + 2) for i in range(25)], dtype
    )
    z_series = real(0.2)
    sqrt_2pi, two_rsqrtpi = real(np.sqrt(2 * np.pi)), real(2 / np.sqrt(np.pi))
    half_ln_2pi = real(0.5 * np.log(2 * np.pi))

    def strict(fn):
        return njit(fn, fastmath={"contract"}, forceinline=True)

    cheb = strict(
        compile_numba_function_src(
            _chebyshev_2d_src(n_arg, n_par),
            "cheb",
            global_env={"one": one, "two": two, "zero": zero},
        )
    )

    @strict
    def locate(val, z, lo, width, cell_lo, cell_hi, last, first, cell):
        g = (val - lo[z]) / width[z]
        m = min(int(np.floor(g)) if g >= zero else 0, last[z]) + first[z]
        c0, c1 = cell_lo[m], cell_hi[m]
        h = c1 - c0
        return cell[m], two * (val - c0) / h - one, two / h

    @strict
    def erfcx_pos(x):
        # x >= 0: t h(t), t = K / (K + x), h on NP panels in t
        tt = erfcx_k / (erfcx_k + x)
        u = real(erfcx_n) * tt
        i = min(int(u), erfcx_n - 1)
        v = two * (u - real(i)) - one
        b1 = b2 = zero
        for j in range(erfcx_p.shape[1] - 1, 0, -1):
            b1, b2 = erfcx_p[i, j] - b2 + two * v * b1, b1
        return tt * (erfcx_p[i, 0] - b2 + v * b1)

    @strict
    def phi_eta(d):
        # phi = d - log1p(d), eta = sign(d) sqrt(2 phi) and d eta / d d
        z = d / (two + d)
        if abs(z) <= z_series:
            s = phi_series[24]
            ds = zero
            for i in range(23, -1, -1):
                ds = ds * z + s
                s = s * z + phi_series[i]
            rs = np.sqrt(s)
            deta_dz = two * rs + z * ds / rs
            return (
                two * z * z * s,
                two * z * rs,
                deta_dz * two / ((two + d) * (two + d)),
            )
        phi = d - np.log1p(d)
        eta = np.sqrt(two * phi) if d > zero else -np.sqrt(two * phi)
        return phi, eta, d / ((one + d) * eta)

    @strict
    def direct(k, x):
        lam = x / k
        big = k > asplit
        if k < one:
            z = A0 if x <= k + one else B0
        elif lam <= lam_lo:
            z = L2 if big else L1
        elif lam >= lam_hi:
            z = U2 if big else U1
        else:
            z = T
        if z == T:
            par, dpar = one / k, -one / (k * k)
        elif z == L2 or z == U2:
            par, dpar = asplit / k, -asplit / (k * k)
        else:
            par, dpar = np.log1p(k), one / (one + k)
        eta = phi = deta_dd = zero
        if z == A0:
            arg = x / (k + one)
            darg = -arg / (k + one)
        elif z == B0:
            arg = (k + one) / x
            darg = one / x
        elif z == L1 or z == L2:
            arg, darg = lam, -lam / k
        elif z == U1 or z == U2:
            arg, darg = k / x, one / x
        else:
            phi, eta, deta_dd = phi_eta((x - k) / k)
            arg, darg = eta, -deta_dd * lam / k
        i, sp, cp = locate(
            par, z, p_lo, p_width, p_cell_lo, p_cell_hi, p_last, p_first, p_cell
        )
        j, sy, cy = locate(
            arg, z, y_lo, y_width, y_cell_lo, y_cell_hi, y_last, y_first, y_cell
        )
        f, fsy, fsp = cheb(table[offset[z] + i * narg[z] + j], sy, sp)
        fk = fsp * cp * dpar + fsy * cy * darg
        if z == A0:
            lx = np.log(x)
            return k * lx - x + f, lx + fk, True
        if z == B0:
            lx = np.log(x)
            return (k - one) * lx - x + np.log(k) + f, lx + one / k + fk, False
        if z == T:
            hk = np.sqrt(half * k)
            y = eta * hk
            sg = -one if y < zero else one
            rt = sqrt_2pi * np.sqrt(k)
            e = erfcx_pos(abs(y))
            sm = half * e + sg * f / rt
            dy = hk * (darg + eta / (two * k))
            de = two * abs(y) * e - two_rsqrtpi
            dsm = half * de * sg * dy + sg * (fk / rt - f / (two * k * rt))
            return -k * phi + np.log(sm), -two * y * dy + dsm / sm, y < zero
        # x / k loses digits or is 0 below tiny (x = 5e-324, k = 10)
        lnlam = np.log(lam) if lam >= tiny else np.log(x) - np.log(k)
        pre = -(x - k) - half * np.log(k) - half_ln_2pi
        if z == L1 or z == L2:
            return k * lnlam + pre + f, lnlam - half / k + fk, True
        return (k - one) * lnlam + pre + f, lnlam + half / k + fk, False

    return direct


@cache
def _gammainc_kernel(njit, dtype):
    """``(P, Q, dP/dk)`` in `dtype`: chebax's patches where they apply, the series and
    continued fraction of `make_gammainc_kernel` (in float64) elsewhere. Also returns
    the cache keys of the scipy calls in the latter."""
    lgamma, digamma = _gamma_kernels(njit, "float64")
    scipy_fns, scipy_keys = zip(
        *(
            numba_funcify_ScalarOp(op, node=op.make_node(float64(), float64()))
            for op in (gammainc, gammaincc)
        ),
        strict=True,
    )
    classical = make_gammainc_kernel(
        lgamma, digamma, *scipy_fns, lambda fn: njit(fn, fastmath={"contract"})
    )
    direct = _gammainc_patch_kernel(njit, dtype)
    real = np.dtype(dtype).type
    alo, zero, one, inf = real(gammainc_tables.ALO), real(0), real(1), real(np.inf)

    def kernel(k, x):
        if k >= alo and k < inf and x > zero and x < inf:
            v, dv, is_p = direct(k, x)
            e = np.exp(v)
            if is_p:
                return e, one - e, e * dv
            return one - e, e, -(e * dv)
        p, q, dp = classical(np.float64(k), np.float64(x))
        return real(p), real(q), real(dp)

    return njit(kernel, fastmath={"contract"}), scipy_keys


@register_scalar_funcify(GammaIncGrad)
def numba_funcify_GammaIncGrad(op, node, **kwargs):
    dtype = node.outputs[0].type.dtype
    kernel, scipy_keys = _gammainc_kernel(numba_basic.numba_njit, dtype)
    real = np.dtype(dtype).type

    @numba_basic.numba_njit
    def gammainc_grad(k, x):
        return kernel(real(k), real(x))[2]

    return gammainc_grad, scalar_op_cache_key(op, cache_version=3, scipy=scipy_keys)


@register_scalar_funcify(GammaIncAndGrad)
def numba_funcify_GammaIncAndGrad(op, node, **kwargs):
    dtype = node.outputs[0].type.dtype
    kernel, scipy_keys = _gammainc_kernel(numba_basic.numba_njit, dtype)
    real = np.dtype(dtype).type

    @numba_basic.numba_njit
    def gammainc_and_grad(k, x):
        return kernel(real(k), real(x))

    return gammainc_and_grad, scalar_op_cache_key(op, cache_version=3, scipy=scipy_keys)


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
