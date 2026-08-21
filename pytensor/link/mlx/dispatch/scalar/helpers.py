import mlx.core as mx
import numpy as np


def _exp(t, const):
    r"""``exp(t)`` at the working precision, over the full exponent range of the dtype.

    ``mx.exp`` is a float32 kernel in range as well as in precision: its float64 result
    is bitwise equal to its float32 one, it overflows past ``exp(88)`` whatever the
    dtype, and it flushes the float32 subnormals to zero from ``exp(-90)``. ``mx.power``
    carries none of those limits and is at least as accurate at both precisions, leaving
    only the :math:`x \epsilon` the exponent itself carries. It is free on the GPU at
    float32 and costs around a fifth more on the CPU float64 path -- which is the path
    that would otherwise be returning ``inf``.

    Parameters
    ----------
    t : mx.array
        Exponent, at the working precision.
    const : callable
        Materializes a Python float at the working precision.

    Returns
    -------
    mx.array
        ``exp(t)`` at the dtype of ``t``.
    """
    return mx.power(const(np.e), t)


_LN2 = 0.6931471805599453


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


# The floor a continued fraction puts under its own denominators. It has to be a normal
# at float32 as well as float64, or it rounds to zero there and takes the guard with it
_TINY = 1e-30

# 0.25**25 sits below float64 epsilon, so the series is exact to working precision
# everywhere it is selected
_DEFICIT_SERIES_TERMS = 25
_DEFICIT_MAX_Z = 0.25


def _metal_constants(**arrays):
    """Render Python float tuples as Metal ``constant`` arrays, one per keyword."""
    return "\n".join(
        f"constant float {name}[{len(values)}] = {{"
        + ", ".join(f"{v!r}f" for v in values)
        + "};"
        for name, values in arrays.items()
    )


def _log1p_deficit(z, one_plus_z, const):
    r"""``2 (z - \log(1 + z))``, without either cancellation it carries.

    The subtraction cancels as :math:`z \to 0`, which the Maclaurin series covers, and
    :math:`1 + z` cancels as :math:`z \to -1`, which the caller covers by passing the
    ratio it already holds exactly.

    Parameters
    ----------
    z : mx.array
        Argument, at the working precision.
    one_plus_z : mx.array
        ``1 + z``, formed without cancellation. Values at or below zero return ``inf``.
    const : callable
        Materializes a Python float at the working precision.

    Returns
    -------
    mx.array
        ``2 * (z - log(1 + z))``, non-negative over the whole domain.
    """
    series = const(0.0)
    for k in reversed(range(_DEFICIT_SERIES_TERMS)):
        series = series * (-z) + const(2.0 / (k + 2.0))
    return mx.where(
        mx.abs(z) <= const(_DEFICIT_MAX_Z),
        z * z * series,
        const(2.0) * (z - mx.log(one_plus_z)),
    )


# The Metal twin of _log1p_deficit, generated from the same constants so that the two
# cannot drift: a kernel that computes eta by a different formula from the vectorized
# path it is checked against is a kernel whose agreement test proves nothing.
_METAL_DEFICIT = (
    _metal_constants(
        DEFICIT_C=tuple(2.0 / (k + 2.0) for k in range(_DEFICIT_SERIES_TERMS))
    )
    + f"""
#define DEFICIT_TERMS {_DEFICIT_SERIES_TERMS}
#define DEFICIT_MAX_Z {_DEFICIT_MAX_Z}f

static inline float log1p_deficit(float z, float one_plus_z) {{
    if (fabs(z) <= DEFICIT_MAX_Z) {{
        float series = 0.0f;
        for (int k = DEFICIT_TERMS - 1; k >= 0; --k) series = series * (-z) + DEFICIT_C[k];
        return z * z * series;
    }}
    return 2.0f * (z - log(one_plus_z));
}}
"""
)
