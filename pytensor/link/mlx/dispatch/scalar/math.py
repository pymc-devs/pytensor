import mlx.core as mx
import numpy as np

from pytensor.link.mlx.dispatch.basic import mlx_funcify
from pytensor.link.mlx.dispatch.scalar.helpers import _working_precision
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


@mlx_funcify.register(Sigmoid)
def mlx_funcify_Sigmoid(op, **kwargs):
    return mx.sigmoid


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
        lanczos = _lanczos_log_gamma(w, const)

        # Reducing the argument keeps log|sin(pi x)| accurate at large |x|, and taking
        # the sine from the tangent by the half-angle identity keeps it off mx.sin,
        # which is a float32 kernel at any dtype where mx.tan is not
        tan_half = mx.tan(const(0.5 * np.pi) * (y - mx.floor(y)))
        log_sin = mx.log(const(2.0) * mx.abs(tan_half)) - mx.log1p(tan_half * tan_half)
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

        # psi(x) = psi(1 - x) - pi * cot(pi x), with the cotangent taken as a reciprocal
        # tangent rather than a ratio of cosine to sine: mx.tan is genuine at float64
        # where both of the others are float32 kernels whatever they are handed.
        # Reducing by the nearest integer is exact and costs the period nothing, and it
        # puts the tangent on a true zero at every pole rather than near one, so those
        # come back infinite instead of large and finite
        cot_arg = const(np.pi) * (z - mx.round(z))
        out = mx.where(reflect, out - const(np.pi) / mx.tan(cot_arg), out)

        return out.astype(out_dtype)

    return psi
