from collections.abc import Callable

import mlx.core as mx
import numpy as np

from pytensor.link.mlx.dispatch.basic import mlx_funcify
from pytensor.scalar.math import (
    Erfc,
    Erfcx,
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

# W. J. Cody, "Rational Chebyshev Approximation for the Error Function" (Math. Comp. 23,
# 1969), in the arrangement used by the netlib SPECFUN CALERF routine. Three intervals:
# a series for erf below 0.46875, and two rationals that yield erfcx directly above it.
#
# Deriving all three from one kernel, rather than composing them out of mx.erf, is what
# makes them accurate. mx.erf is a float32 kernel whatever dtype it is handed, as are
# mx.exp, mx.sin and mx.cos, while mx.log and mx.sqrt are genuine at float64. Both upper
# intervals here are free of exp, so erfcx holds the full working precision across the
# range where erfc has decayed far past anything 1 - erf could resolve. The subunit
# interval and the negative-argument reflections still need exp and inherit its ceiling
# of roughly 1e-7 relative, which is why erfcx keeps its rational branches rather than
# being derived from a single form.
_ERF_A = (
    3.16112374387056560e00,
    1.13864154151050156e02,
    3.77485237685302021e02,
    3.20937758913846947e03,
    1.85777706184603153e-1,
)
_ERF_B = (
    2.36012909523441209e01,
    2.44024637934444173e02,
    1.28261652607737228e03,
    2.84423683343917062e03,
)
_ERFCX_C = (
    5.64188496988670089e-1,
    8.88314979438837594e00,
    6.61191906371416295e01,
    2.98635138197400131e02,
    8.81952221241769090e02,
    1.71204761263407058e03,
    2.05107837782607147e03,
    1.23033935479799725e03,
    2.15311535474403846e-8,
)
_ERFCX_D = (
    1.57449261107098347e01,
    1.17693950891312499e02,
    5.37181101862009858e02,
    1.62138957456669019e03,
    3.29079923573345963e03,
    4.36261909014324716e03,
    3.43936767414372164e03,
    1.23033935480374942e03,
)
_ERFCX_P = (
    3.05326634961232344e-1,
    3.60344899949804439e-1,
    1.25781726111229246e-1,
    1.60837851487422766e-2,
    6.58749161529837803e-4,
    1.63153871373020978e-2,
)
_ERFCX_Q = (
    2.56852019228982242e00,
    1.87295284992346047e00,
    5.27905102951428412e-1,
    6.05183413124413191e-2,
    2.33520497626869185e-3,
)
_SQRT_PI_INV = 5.6418958354775628695e-1
_ERF_THRESH = 0.46875
_ERFCX_SPLIT = 4.0

_LN2 = 0.6931471805599453
_EXP_CLAMP = 1e4


def _metal_constants(**arrays):
    """Render Python float tuples as Metal ``constant`` arrays, one per keyword."""
    return "\n".join(
        f"constant float {name}[{len(values)}] = {{"
        + ", ".join(f"{v!r}f" for v in values)
        + "};"
        for name, values in arrays.items()
    )


# Generating the coefficients rather than transcribing them keeps the two implementations
# from drifting apart, and emitting them as named Metal arrays keeps everything below this
# point ordinary Metal: only the numbers are generated, not the surrounding code.
#
# The kernel is worth having because it can branch. The vectorized path has to evaluate
# every interval for every element and select afterwards, which costs it an order of
# magnitude at the sizes where memory bandwidth dominates; a kernel just takes the branch.


_METAL_HEADER = (
    _metal_constants(
        ERF_A=_ERF_A,
        ERF_B=_ERF_B,
        ERFCX_C=_ERFCX_C,
        ERFCX_D=_ERFCX_D,
        ERFCX_P=_ERFCX_P,
        ERFCX_Q=_ERFCX_Q,
        ERF_SPLITS=(_ERF_THRESH, _ERFCX_SPLIT, _SQRT_PI_INV),
    )
    + """
#define ERF_THRESH   ERF_SPLITS[0]
#define ERFCX_SPLIT  ERF_SPLITS[1]
#define SQRT_PI_INV  ERF_SPLITS[2]

static inline float erf_series(float t) {
    float z = t * t;
    float num = ERF_A[4] * z;
    float den = z;
    for (int i = 0; i < 3; ++i) {
        num = (num + ERF_A[i]) * z;
        den = (den + ERF_B[i]) * z;
    }
    return t * (num + ERF_A[3]) / (den + ERF_B[3]);
}

static inline float erfcx_upper(float y) {
    if (y <= ERFCX_SPLIT) {
        float num = ERFCX_C[8] * y;
        float den = y;
        for (int i = 0; i < 7; ++i) {
            num = (num + ERFCX_C[i]) * y;
            den = (den + ERFCX_D[i]) * y;
        }
        return (num + ERFCX_C[7]) / (den + ERFCX_D[7]);
    }
    float z = 1.0f / (y * y);
    float num = ERFCX_P[5] * z;
    float den = z;
    for (int i = 0; i < 4; ++i) {
        num = (num + ERFCX_P[i]) * z;
        den = (den + ERFCX_Q[i]) * z;
    }
    float r = z * (num + ERFCX_P[4]) / (den + ERFCX_Q[4]);
    return (SQRT_PI_INV - r) / y;
}
"""
)


_METAL_ERFC_SOURCE = """
    uint i = thread_position_in_grid.x;
    float xv = (float)x[i];
    float y = fabs(xv);
    float res;
    if (y <= ERF_THRESH) {
        res = 1.0f - erf_series(y);
    } else {
        res = exp(-y * y) * erfcx_upper(y);
    }
    if (xv < 0.0f) res = 2.0f - res;
    out[i] = (T)res;
"""


_METAL_ERFCX_SOURCE = """
    uint i = thread_position_in_grid.x;
    float xv = (float)x[i];
    float y = fabs(xv);
    float res;
    if (y <= ERF_THRESH) {
        res = exp(y * y) * (1.0f - erf_series(y));
    } else {
        res = erfcx_upper(y);
    }
    if (xv < 0.0f) res = 2.0f * exp(y * y) - res;
    out[i] = (T)res;
"""


_METAL_SOURCES = {"erfc": _METAL_ERFC_SOURCE, "erfcx": _METAL_ERFCX_SOURCE}


_METAL_KERNELS: dict[str, Callable | None] = {}


def _metal_erf_kernel(name):
    """Build and cache the Metal kernel for ``name``, or None if it cannot be built."""
    if name not in _METAL_KERNELS:
        try:
            _METAL_KERNELS[name] = mx.fast.metal_kernel(
                name=f"pytensor_{name}",
                input_names=["x"],
                output_names=["out"],
                header=_METAL_HEADER,
                source=_METAL_SOURCES[name],
            )
        except Exception:
            _METAL_KERNELS[name] = None
    return _METAL_KERNELS[name]


def _metal_erf_call(name, x):
    """Evaluate ``name`` through its Metal kernel, or return None to fall back.

    Metal has no float64 and the kernel needs the GPU stream, which between them leave
    exactly the case the vectorized path handles least well and float64 needs least: the
    default device at single precision.
    """
    if (
        x.dtype != mx.float32
        or not mx.metal.is_available()
        or mx.default_device() != mx.gpu
    ):
        return None
    kernel = _metal_erf_kernel(name)
    if kernel is None:
        return None
    # The kernel indexes a flat buffer, so a 0-d input has to be raised to 1-d before it
    # is passed: Metal binds a scalar as a value rather than a pointer, and subscripting
    # it fails to compile. Reshaping covers every rank at once
    flat = x.reshape(-1)
    (out,) = kernel(
        inputs=[flat],
        template=[("T", flat.dtype)],
        grid=(flat.size, 1, 1),
        threadgroup=(min(256, max(flat.size, 1)), 1, 1),
        output_shapes=[flat.shape],
        output_dtypes=[flat.dtype],
    )
    return out.reshape(x.shape)


def _exp_wide(t, const):
    r"""``exp(t)`` over the full exponent range of the working precision.

    ``mx.exp`` is a float32 kernel in range as well as in precision, flushing to zero
    below ``exp(-87)`` and overflowing above ``exp(88)`` whatever the dtype. This holds
    to the range the dtype itself supports, at ``mx.exp``'s float32 precision.
    """
    # Splitting e^t into 2^k e^r, with k an integer and |r| <= ln(2)/2, keeps the
    # exponential's argument small while mx.power supplies the scale over the full range.
    #
    # An infinite argument would leave r = -inf + inf = nan, so t is clamped first. The
    # bound is far outside the representable range of either precision, where 2**k has
    # already saturated to zero or infinity, so no finite result is altered
    t = mx.minimum(mx.maximum(t, const(-_EXP_CLAMP)), const(_EXP_CLAMP))
    k = mx.round(t / const(_LN2))
    r = t - k * const(_LN2)
    return mx.exp(r) * mx.power(const(2.0), k)


def _erf_series(x, const):
    """erf on |x| <= 0.46875, where 1 - erfc would cancel. Free of exp."""
    z = x * x
    num = const(_ERF_A[4]) * z
    den = z
    for a, b in zip(_ERF_A[:3], _ERF_B[:3]):
        num = (num + const(a)) * z
        den = (den + const(b)) * z
    return x * (num + const(_ERF_A[3])) / (den + const(_ERF_B[3]))


def _erfcx_upper(y, const):
    """erfcx on y >= 0.46875, where neither branch needs exp.

    Callers must clamp ``y`` to the threshold themselves. Keeping this separate from
    ``_erfcx_positive`` is what lets ``erf`` and ``erfc`` skip the subunit branch, which
    their own clamps guarantee is never selected -- every branch is evaluated for every
    element, so an unreachable one is not free.
    """
    split = const(_ERFCX_SPLIT)

    # 0.46875 <= y <= 4
    mid_y = mx.minimum(y, split)
    num = const(_ERFCX_C[8]) * mid_y
    den = mid_y
    for c, d in zip(_ERFCX_C[:7], _ERFCX_D[:7]):
        num = (num + const(c)) * mid_y
        den = (den + const(d)) * mid_y
    mid = (num + const(_ERFCX_C[7])) / (den + const(_ERFCX_D[7]))

    # y > 4, as an asymptotic rational in 1 / y^2
    high_y = mx.maximum(y, split)
    z = const(1.0) / (high_y * high_y)
    num = const(_ERFCX_P[5]) * z
    den = z
    for p, q in zip(_ERFCX_P[:4], _ERFCX_Q[:4]):
        num = (num + const(p)) * z
        den = (den + const(q)) * z
    r = z * (num + const(_ERFCX_P[4])) / (den + const(_ERFCX_Q[4]))
    high = (const(_SQRT_PI_INV) - r) / high_y

    return mx.where(y <= split, mid, high)


def _erfc_tail(y, const):
    """erfc on y >= 0.46875. Callers must clamp ``y`` to the threshold themselves.

    Multiplying by ``exp(-y**2)`` underflows to zero smoothly, where ``1 - erf(y)``
    cancels to it abruptly and loses every significant digit on the way.
    """
    return _exp_wide(-(y * y), const) * _erfcx_upper(y, const)


def _erfcx_positive(y, const):
    """erfcx on y >= 0, for non-negative ``y`` only."""
    thresh = const(_ERF_THRESH)
    # y < 0.46875 is the one branch that needs exp, and exp(y^2) <= 1.25 there
    small_y = mx.minimum(y, thresh)
    small = mx.exp(small_y * small_y) * (const(1.0) - _erf_series(small_y, const))
    return mx.where(y <= thresh, small, _erfcx_upper(mx.maximum(y, thresh), const))


def _erfc_positive(y, const):
    """erfc on y >= 0, for non-negative ``y`` only."""
    thresh = const(_ERF_THRESH)
    near = const(1.0) - _erf_series(mx.minimum(y, thresh), const)
    return mx.where(y <= thresh, near, _erfc_tail(mx.maximum(y, thresh), const))


def _working_precision(x):
    """Set up a series approximation over ``x``.

    Half precision is widened to float32, having too few mantissa bits for series
    coefficients to mean anything.

    Parameters
    ----------
    x : mx.array
        Input to the approximation.

    Returns
    -------
    z : mx.array
        ``x`` at the working precision.
    const : callable
        Materialize a Python float as an ``mx.array`` of the working precision. MLX
        weak-types Python floats to float32, which would otherwise silently pin a
        float64 approximation to float32 accuracy.
    out_dtype : MLX dtype
        The dtype the result should be cast back to.
    """
    x = mx.array(x)
    working_dtype = mx.float32 if x.dtype in (mx.float16, mx.bfloat16) else x.dtype

    def const(value):
        return mx.array(value, dtype=working_dtype)

    return x.astype(working_dtype), const, x.dtype


@mlx_funcify.register(Sigmoid)
def mlx_funcify_Sigmoid(op, **kwargs):
    return mx.sigmoid


@mlx_funcify.register(Erfc)
def mlx_funcify_Erfc(op, **kwargs):
    def erfc(x):
        x = mx.array(x)
        fast = _metal_erf_call("erfc", x)
        if fast is not None:
            return fast

        z, const, out_dtype = _working_precision(x)
        y = mx.abs(z)

        # Both branches evaluate erfc(|x|); erfc(-y) = 2 - erfc(y) restores the sign
        pos = _erfc_positive(y, const)
        out = mx.where(z >= const(0.0), pos, const(2.0) - pos)
        return out.astype(out_dtype)

    return erfc


@mlx_funcify.register(Erfcx)
def mlx_funcify_Erfcx(op, **kwargs):
    def erfcx(x):
        x = mx.array(x)
        fast = _metal_erf_call("erfcx", x)
        if fast is not None:
            return fast

        z, const, out_dtype = _working_precision(x)
        y = mx.abs(z)

        pos = _erfcx_positive(y, const)
        # erfcx(-y) = 2 exp(y^2) - erfcx(y). The exponential is unavoidable here and
        # genuinely overflows past y ~ 26.6, which is the function's own behavior
        reflected = const(2.0) * _exp_wide(y * y, const) - pos
        return mx.where(z >= const(0.0), pos, reflected).astype(out_dtype)

    return erfcx


@mlx_funcify.register(Softplus)
def mlx_funcify_Softplus(op, **kwargs):
    def softplus(x):
        return mx.where(
            x < -37.0,
            mx.exp(x),
            mx.where(
                x < 18.0,
                mx.log1p(mx.exp(x)),
                mx.where(
                    x < 33.3,
                    x + mx.exp(-x),
                    x,
                ),
            ),
        )

    return softplus


@mlx_funcify.register(Log1mexp)
def mlx_funcify_Log1mexp(op, node, **kwargs):
    def log1mexp(x):
        return mx.where(x < mx.log(0.5), mx.log1p(-mx.exp(x)), mx.log(-mx.expm1(x)))

    return log1mexp


@mlx_funcify.register(GammaLn)
def mlx_funcify_GammaLn(op, **kwargs):
    def gammaln(x):
        z, const, out_dtype = _working_precision(x)

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
        series = const(_LANCZOS_COEFFS[0])
        for i, coeff in enumerate(_LANCZOS_COEFFS[1:], start=1):
            series = series + const(coeff) / (w + const(float(i)))

        t = w + const(_LANCZOS_G + 0.5)
        lanczos = (
            const(0.5 * np.log(2.0 * np.pi))
            + (w + half) * mx.log(t)
            - t
            + mx.log(series)
        )

        # Reducing the argument before sin keeps log|sin(pi x)| accurate at large |x|
        log_sin = mx.log(mx.abs(mx.sin(const(np.pi) * (y - mx.floor(y)))))
        out = mx.where(reflect, const(np.log(np.pi)) - log_sin - lanczos, lanczos)
        if double:
            out = mx.where(shift_up, out - mx.log(z), out)

        # The series returns NaN at +/-inf, where both limits are +inf
        out = mx.where(mx.isinf(z), const(np.inf), out)

        return out.astype(out_dtype)

    return gammaln


@mlx_funcify.register(Psi)
def mlx_funcify_Psi(op, **kwargs):
    def psi(x):
        z, const, out_dtype = _working_precision(x)

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

        # psi(x) = psi(1 - x) - pi * cot(pi x)
        pi_z = const(np.pi) * z
        out = mx.where(reflect, out - const(np.pi) * mx.cos(pi_z) / mx.sin(pi_z), out)

        return out.astype(out_dtype)

    return psi
