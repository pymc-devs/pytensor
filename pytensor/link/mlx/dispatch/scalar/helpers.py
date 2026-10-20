from collections.abc import Callable

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


_METAL_UNARY_KERNELS: dict[tuple[str, str, str], Callable] = {}


def _metal_unary_kernel(name, header, source):
    """Build and cache the one-input Metal kernel ``name``.

    Metal compiles the source lazily, so a malformed source fails when the kernel is
    first evaluated rather than here.
    """
    key = (name, header, source)
    if key not in _METAL_UNARY_KERNELS:
        _METAL_UNARY_KERNELS[key] = mx.fast.metal_kernel(
            name=f"pytensor_{name}",
            input_names=["x"],
            output_names=["out"],
            header=header,
            source=source,
        )
    return _METAL_UNARY_KERNELS[key]


def _metal_unary_call(x, *, name, header, source):
    """Evaluate an elementwise Metal kernel over ``x``, or return None to fall back.

    Metal has no float64 and the kernel needs the GPU stream, which between them leave
    exactly the case the vectorized path handles least well and float64 needs least: the
    default device at single or half precision. The kernel reads half precision directly
    and computes in float32, so a half-precision input is rounded once, on the way out.

    Parameters
    ----------
    x : mx.array
        Input, of any rank.
    name : str
        Kernel name.
    header : str
        Metal source placed ahead of the kernel body.
    source : str
        Kernel body. It reads ``x[i]`` and writes ``out[i]`` for the thread index ``i``,
        computing in ``float`` and casting the result to ``T``, the dtype of ``x``.

    Returns
    -------
    out : mx.array or None
        The result at the shape and dtype of ``x``, or None where the kernel does not
        apply.
    """
    if (
        x.dtype not in (mx.float32, mx.float16, mx.bfloat16)
        or not mx.metal.is_available()
        or mx.default_device() != mx.gpu
    ):
        return None
    kernel = _metal_unary_kernel(name, header, source)
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
