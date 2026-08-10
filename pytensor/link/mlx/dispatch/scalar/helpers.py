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
