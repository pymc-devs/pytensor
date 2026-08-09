import mlx.core as mx


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
