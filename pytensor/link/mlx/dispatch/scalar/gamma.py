import mlx.core as mx
import numpy as np

from pytensor.link.mlx.dispatch.basic import mlx_funcify
from pytensor.link.mlx.dispatch.scalar.helpers import _exp, _working_precision
from pytensor.link.mlx.dispatch.scalar.math import _lanczos_log_gamma
from pytensor.scalar.math import Gamma


def _gamma_positive(z, const):
    """``Gamma(z)`` for ``z > 0``, as ``exp`` of the Lanczos series."""
    # Below 1/2 the series is evaluated one unit up and divided back down, which keeps
    # the argument inside the range the Lanczos coefficients were fitted for
    half = const(0.5)
    shift_up = z < half
    y = mx.where(shift_up, z + const(1.0), z)
    out = _exp(_lanczos_log_gamma(y - const(1.0), const), const)
    return mx.where(shift_up, out / z, out)


@mlx_funcify.register(Gamma)
def mlx_funcify_Gamma(op, **kwargs):
    def gamma(x):
        z, const, out_dtype = _working_precision(x)
        zero = const(0.0)

        # Both branches are evaluated for every element, so the one that will be
        # discarded is handed a safe argument rather than a clamped one: clamping the
        # positive branch would cap Gamma near the origin, where it grows like 1 / z
        positive = _gamma_positive(mx.where(z > zero, z, const(1.0)), const)

        # Gamma(z) = pi / (sin(pi z) Gamma(1 - z)) below zero. sin is a float32 kernel
        # whatever dtype it is handed and tan is not, so sin(t) is taken from the
        # half-angle identity 2 tan(t/2) / (1 + tan(t/2)**2), which holds float64
        negative = mx.where(z < zero, z, -const(1.0))
        tan_half = mx.tan(const(0.5 * np.pi) * negative)
        sin_pi_z = const(2.0) * tan_half / (const(1.0) + tan_half * tan_half)
        reflected = const(np.pi) / (
            sin_pi_z * _gamma_positive(const(1.0) - negative, const)
        )

        out = mx.where(z > zero, positive, reflected)

        # The poles: Gamma is +/-inf at zero, taking the sign of the zero, and undefined
        # at every negative integer, where the reflection divides by a sin that rounding
        # leaves merely tiny rather than exactly zero
        out = mx.where(z == zero, const(1.0) / z, out)
        # the series is inf / inf at the positive end, where the limit is plain inf
        out = mx.where(mx.isinf(z) & (z > zero), const(np.inf), out)
        # neither branch is handed a NaN -- both substitute a safe argument for the
        # elements the other will select -- so it has to be put back explicitly
        out = mx.where(mx.isnan(z), z, out)
        nan = zero / zero
        return mx.where((z < zero) & (z == mx.round(z)), nan, out).astype(out_dtype)

    return gamma
