import mlx.core as mx
import numpy as np

from pytensor.link.mlx.dispatch.basic import mlx_funcify
from pytensor.link.mlx.dispatch.scalar.helpers import (
    _metal_constants,
    _metal_unary_call,
    _working_precision,
)
from pytensor.scalar import _gamma_tables as gamma_tables
from pytensor.scalar.math import (
    GammaLn,
    Log1mexp,
    Psi,
    Sigmoid,
    Softplus,
)


_LANCZOS_G = 7.0
_LANCZOS_COEFFS = (
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7,
)

# Asymptotic expansion of psi, with terms B_2n / 2n, applied after the recurrence has
# carried the argument up by _PSI_SHIFTS. Single precision reaches its own limit with a
# shorter recurrence and fewer terms, and every one of them costs a pass over the array.
_PSI_COEFFS = (
    1 / 12,
    -1 / 120,
    1 / 252,
    -1 / 240,
    1 / 132,
    -691 / 32760,
    1 / 12,
)
_PSI_SHIFTS = 10
_PSI_COEFFS_SINGLE = _PSI_COEFFS[:3]
_PSI_SHIFTS_SINGLE = 6


def _flat(table):
    return tuple(float(v) for v in np.ravel(table))


# The panels cover [1, 64), where Stirling's series takes over
_TABLE_END = 64.0
_STIRLING32 = gamma_tables.STIRLING32
_DSTIRLING32 = _STIRLING32 * np.arange(1, 2 * _STIRLING32.size, 2)


# Both functions are a chain of elementwise ops on the vectorized path, long enough that
# mx.compile splits it into several kernel launches. A kernel is one launch, and it can
# branch, which lets positive arguments read the float32 Chebyshev panels of
# pytensor.scalar._gamma_tables: a row lookup and a short Horner sum on [1, 64), with no
# division, where the series and the recurrence above cost several. Below 1 the argument
# shifts up by one, and from 64 Stirling's series takes over. Negative arguments reflect
# onto the same positive kernels. Metal's sinpi and tanpi reduce the argument themselves,
# so the reflections need no half-angle detour.
_METAL_MATH_HEADER = (
    _metal_constants(
        LG_TABLE=_flat(gamma_tables.LGAMMA32),
        PSI_TABLE=_flat(gamma_tables.PSI32),
        STIRLING=_flat(_STIRLING32),
        DSTIRLING=_flat(_DSTIRLING32),
        INV_POW2=(1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125),
    )
    + f"""
#define LG_WIDTH          {gamma_tables.LGAMMA32.shape[1]}
#define PSI_WIDTH         {gamma_tables.PSI32.shape[1]}
#define STIRLING_C        {float(gamma_tables.STIRLING_C)!r}f
#define X0_HI             {float(gamma_tables.X0_HI32)!r}f
#define X0_LO             {float(gamma_tables.X0_LO32)!r}f
#define LG_LOG_PI         {float(np.log(np.pi))!r}f
#define TABLE_END         {_TABLE_END}f

// The series of panel m of octave e of y in [1, 64), 32 panels an octave. Scaling by
// a power of two and subtracting one are exact, so the panel index cannot leave [0, 32)
static inline float panel_series(constant float* table, int width, float y) {{
    int e = int(y >= 2.0f) + int(y >= 4.0f) + int(y >= 8.0f) + int(y >= 16.0f)
            + int(y >= 32.0f);
    float u = (y * INV_POW2[e] - 1.0f) * 32.0f;
    int m = int(u);
    float s = 2.0f * (u - float(m)) - 1.0f;
    constant float* row = table + (32 * e + m) * width;
    float p = row[width - 1];
    for (int k = width - 2; k >= 0; --k) p = p * s + row[k];
    return p;
}}

// lgamma(x) for finite x > 0
static inline float lgamma_positive(float x) {{
    if (x >= TABLE_END) {{
        float inv = 1.0f / x;
        float series = inv * (STIRLING[0] + STIRLING[1] * inv * inv);
        return (x - 0.5f) * (log(x) - 1.0f) + STIRLING_C + series;
    }}
    // the table holds lgamma(y) / ((y - 1) (y - 2)) on [1, 4)
    float f = panel_series(LG_TABLE, LG_WIDTH, x >= 1.0f ? x : x + 1.0f);
    if (x >= 4.0f) return f;
    if (x >= 1.0f) return (x - 1.0f) * (x - 2.0f) * f;
    return x * (x - 1.0f) * f - log(x);
}}

// digamma(x) for x >= 0 or +inf, -inf at 0
static inline float psi_positive(float x) {{
    if (x >= TABLE_END) {{
        float inv = 1.0f / x;
        float p = inv * inv;
        return log(x) - 0.5f * inv - p * (DSTIRLING[0] + DSTIRLING[1] * p);
    }}
    float y = x >= 1.0f ? x : x + 1.0f;
    float r = panel_series(PSI_TABLE, PSI_WIDTH, y);
    // the table holds digamma(y) / (y - x0) on [1, 2), and y - X0_HI is exact
    if (y < 2.0f) r = r * (y - X0_HI) - r * X0_LO;
    return x >= 1.0f ? r : r - 1.0f / x;
}}
"""
)


_METAL_GAMMALN_SOURCE = """
    uint i = thread_position_in_grid.x;
    float z = (float)x[i];
    float res;
    if (isnan(z)) {
        res = z;
    } else if (isinf(z)) {
        // both limits are +inf
        res = INFINITY;
    } else if (z > 0.0f) {
        res = lgamma_positive(z);
    } else {
        // log|sin(pi z)| is -inf at the poles, which makes the result +inf there
        float log_sin = log(fabs(sinpi(z)));
        res = LG_LOG_PI - log_sin - lgamma_positive(1.0f - z);
    }
    out[i] = (T)res;
"""


_METAL_PSI_SOURCE = """
    uint i = thread_position_in_grid.x;
    float z = (float)x[i];
    float res;
    if (isnan(z)) {
        res = z;
    } else if (z >= 0.0f) {
        res = psi_positive(z);
    } else {
        // the tangent lands on a true zero at every pole, so those come back infinite
        res = psi_positive(1.0f - z) - M_PI_F / tanpi(z - rint(z));
    }
    out[i] = (T)res;
"""


_METAL_SOURCES = {"gammaln": _METAL_GAMMALN_SOURCE, "psi": _METAL_PSI_SOURCE}


def _metal_math_call(name, x):
    """Evaluate ``name`` through its Metal kernel, or return None to fall back."""
    return _metal_unary_call(
        x, name=name, header=_METAL_MATH_HEADER, source=_METAL_SOURCES[name]
    )


def _lanczos_log_gamma(w, const):
    """``log Gamma(w + 1)`` by the Lanczos series, for ``w`` at or above ``-0.5``."""
    series = const(_LANCZOS_COEFFS[0])
    for i, coeff in enumerate(_LANCZOS_COEFFS[1:], start=1):
        series = series + const(coeff) / (w + const(float(i)))

    t = w + const(_LANCZOS_G + 0.5)
    return (
        const(0.5 * np.log(2.0 * np.pi))
        + (w + const(0.5)) * mx.log(t)
        - t
        + mx.log(series)
    )


def _log_abs_sin_pi(z, const):
    """``log|sin(pi z)|``, ``-inf`` at the integers.

    Reducing by the nearest integer is exact, where subtracting floor(z) rounds once z
    is small and negative, and it keeps the result accurate at large |z|. Taking the
    sine from the tangent by the half-angle identity keeps it off mx.sin, which is a
    float32 kernel at any dtype where mx.tan is not.
    """
    tan_half = mx.tan(const(0.5 * np.pi) * (z - mx.round(z)))
    # |tan_half| <= 1, so 1 + tan_half**2 is in [1, 2] and costs nothing to form
    return mx.log(const(2.0) * mx.abs(tan_half) / (const(1.0) + tan_half * tan_half))


def _pi_cot_pi(z, const):
    """``pi * cot(pi z)``, infinite at the integers.

    The cotangent is a reciprocal tangent: mx.tan is genuine at float64 where mx.cos
    and mx.sin are float32 kernels whatever they are handed. Reducing by the nearest
    integer is exact and puts the tangent on a true zero at every pole.
    """
    return const(np.pi) / mx.tan(const(np.pi) * (z - mx.round(z)))


@mlx_funcify.register(Sigmoid)
def mlx_funcify_Sigmoid(op, node, **kwargs):
    dtype = node.outputs[0].dtype

    def sigmoid(x):
        z, const, out_dtype = _working_precision(x, dtype)
        # exp(-|x|) is shared with Softplus, so MLX can reuse it when both are fused.
        # |x| as `where`: MLX's gradient of `abs` is 0 at 0, which would zero sigmoid'(0)
        e = mx.exp(-mx.where(z >= 0, z, -z))
        return (mx.where(z >= 0, const(1.0), e) / (const(1.0) + e)).astype(out_dtype)

    return sigmoid


@mlx_funcify.register(Softplus)
def mlx_funcify_Softplus(op, node, **kwargs):
    dtype = node.outputs[0].dtype

    def softplus(x):
        z, const, out_dtype = _working_precision(x, dtype)
        # Branch-free, and exp(-|x|) is shared with Sigmoid. `where` rather than `maximum`
        # and `abs`, whose MLX gradients are 0 at 0
        out = mx.where(z >= 0, z, const(0.0)) + mx.log1p(
            mx.exp(-mx.where(z >= 0, z, -z))
        )
        return out.astype(out_dtype)

    return softplus


@mlx_funcify.register(Log1mexp)
def mlx_funcify_Log1mexp(op, node, **kwargs):
    def log1mexp(x):
        return mx.where(x < mx.log(0.5), mx.log1p(-mx.exp(x)), mx.log(-mx.expm1(x)))

    return log1mexp


@mlx_funcify.register(GammaLn)
def mlx_funcify_GammaLn(op, node, **kwargs):
    dtype = node.outputs[0].dtype

    def gammaln(x):
        z, const, out_dtype = _working_precision(x, dtype)
        fast = _metal_math_call("gammaln", mx.asarray(x).astype(out_dtype))
        if fast is not None:
            return fast

        double = z.dtype == mx.float64
        half = const(0.5)
        one = const(1.0)

        # In double precision, small positive arguments go up by one via
        # lgamma(x) = lgamma(x + 1) - log(x) rather than through the reflection formula
        # below. Both are exact as mathematics, but reflection would route them through
        # mx.sin, which is only float32-accurate whatever the input dtype. Single
        # precision has nothing to protect, and the extra log costs a pass
        shift_up = (z > const(0.0)) & (z < half) if double else None
        y = mx.where(shift_up, z + one, z) if double else z
        reflect = y < half

        # Evaluating the series on the reflected argument means one polynomial
        # covers both branches, and neither can produce a NaN the other must mask
        w = mx.where(reflect, one - y, y) - one
        lanczos = _lanczos_log_gamma(w, const)
        out = mx.where(
            reflect,
            const(np.log(np.pi)) - _log_abs_sin_pi(y, const) - lanczos,
            lanczos,
        )
        if double:
            out = mx.where(shift_up, out - mx.log(z), out)

        # The series returns NaN at +/-inf, where both limits are +inf
        out = mx.where(mx.isinf(z), const(np.inf), out)

        return out.astype(out_dtype)

    return gammaln


@mlx_funcify.register(Psi)
def mlx_funcify_Psi(op, node, **kwargs):
    dtype = node.outputs[0].dtype

    def psi(x):
        z, const, out_dtype = _working_precision(x, dtype)
        fast = _metal_math_call("psi", mx.asarray(x).astype(out_dtype))
        if fast is not None:
            return fast

        double = z.dtype == mx.float64
        n_shifts = _PSI_SHIFTS if double else _PSI_SHIFTS_SINGLE
        coeffs = _PSI_COEFFS if double else _PSI_COEFFS_SINGLE

        # Only negative arguments need reflecting; the recurrence below walks a
        # small positive argument up to the asymptotic regime on its own, and it
        # keeps full precision where cot(pi x) would not
        reflect = z < const(0.0)
        y = mx.where(reflect, const(1.0) - z, z)

        # psi(y) = psi(y + n) - sum 1/(y + i), which holds for every y, so the shift
        # is applied unconditionally. Testing whether each element still needs it
        # would cost a comparison and a select per iteration and buy nothing
        one = const(1.0)
        shift = -one / y
        for i in range(1, n_shifts):
            shift = shift - one / (y + const(float(i)))
        y = y + const(float(n_shifts))

        r2 = one / (y * y)
        series = const(coeffs[-1])
        for coeff in reversed(coeffs[:-1]):
            series = const(coeff) + series * r2
        out = shift + mx.log(y) - const(0.5) / y - series * r2
        out = mx.where(reflect, out - _pi_cot_pi(z, const), out)

        return out.astype(out_dtype)

    return psi
