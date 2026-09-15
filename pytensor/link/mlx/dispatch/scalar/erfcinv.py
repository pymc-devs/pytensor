import mlx.core as mx

from pytensor.link.mlx.dispatch.basic import mlx_funcify
from pytensor.link.mlx.dispatch.scalar.helpers import _LN2, _working_precision
from pytensor.scalar.math import Erfcinv


# Wichura, "Algorithm AS 241: The Percentage Points of the Normal Distribution" (Applied
# Statistics 37, 1988), the PPND16 variant. Three intervals again, but the two outer
# ones are rational in sqrt(-log(tail)) rather than in the probability itself, which is
# what holds them together where the probability underflows -- erfcinv(1e-300) is an
# ordinary 26.1. Nothing here needs exp, sin or cos, only log, sqrt and arithmetic, all
# of which MLX computes at the full working precision, so this carries no ceiling.
_NDTRI_A = (
    3.3871328727963666080e0,
    1.3314166789178437745e2,
    1.9715909503065514427e3,
    1.3731693765509461125e4,
    4.5921953931549871457e4,
    6.7265770927008700853e4,
    3.3430575583588128105e4,
    2.5090809287301226727e3,
)
_NDTRI_B = (
    4.2313330701600911252e1,
    6.8718700749205790830e2,
    5.3941960214247511077e3,
    2.1213794301586595867e4,
    3.9307895800092710610e4,
    2.8729085735721942674e4,
    5.2264952788528545610e3,
)
_NDTRI_C = (
    1.42343711074968357734e0,
    4.63033784615654529590e0,
    5.76949722146069140550e0,
    3.64784832476320460504e0,
    1.27045825245236838258e0,
    2.41780725177450611770e-1,
    2.27238449892691845833e-2,
    7.74545014278341407640e-4,
)
_NDTRI_D = (
    2.05319162663775882187e0,
    1.67638483018380384940e0,
    6.89767334985100004550e-1,
    1.48103976427480074590e-1,
    1.51986665636164571966e-2,
    5.47593808499534494600e-4,
    1.05075007164441684324e-9,
)
_NDTRI_E = (
    6.65790464350110377720e0,
    5.46378491116411436990e0,
    1.78482653991729133580e0,
    2.96560571828504891230e-1,
    2.65321895265761230930e-2,
    1.24266094738807843860e-3,
    2.71155556874348757815e-5,
    2.01033439929228813265e-7,
)
_NDTRI_F = (
    5.99832206555887937690e-1,
    1.36929880922735805310e-1,
    1.48753612908506148525e-2,
    7.86869131145613259100e-4,
    1.84631831751005468180e-5,
    1.42151175831644588870e-7,
    2.04426310338993978564e-15,
)
_NDTRI_CENTRAL_HALF_WIDTH = 0.425
# The central branch clamps q**2 at the point |q| is compared against, so the two must
# agree exactly or the unselected branch stops being finite
_NDTRI_CENTRAL_OFFSET = _NDTRI_CENTRAL_HALF_WIDTH**2
_NDTRI_MID_SHIFT = 1.6
_NDTRI_TAIL_SPLIT = 5.0
# The tail variable saturates rather than running to infinity at the poles, which keeps
# the unselected far branch from turning an inf into a nan. Far outside either
# precision: r = 30 corresponds to a tail probability of exp(-900)
_NDTRI_MAX_R = 30.0

_SQRT_2 = 1.4142135623730951


def _horner(coeffs, t, const):
    """Evaluate a polynomial listed lowest-order first, as AS241 tabulates them."""
    out = const(coeffs[-1])
    for c in reversed(coeffs[:-1]):
        out = out * t + const(c)
    return out


def _ndtri_ratio(num_coeffs, den_coeffs, t, const):
    # AS241's denominators are monic with the trailing 1 left implicit
    return _horner(num_coeffs, t, const) / (
        _horner(den_coeffs, t, const) * t + const(1.0)
    )


def _ndtri_of_half(x, const):
    """``ndtri(x / 2)`` for ``0 <= x <= 2``, taken on ``x`` rather than on ``x / 2``."""
    # Both reductions the algorithm needs are exact taken on x directly: for
    # 0.5 <= x <= 2 neither x - 1 nor 2 - x loses a bit, where the same quantities
    # reached as x / 2 - 0.5 and 1 - x / 2 cancel as x approaches 1
    half = const(0.5)
    q = (x - const(1.0)) * half
    half_width = const(_NDTRI_CENTRAL_HALF_WIDTH)
    offset = const(_NDTRI_CENTRAL_OFFSET)
    split = const(_NDTRI_TAIL_SPLIT)

    central_r = offset - mx.minimum(q * q, offset)
    central = q * _ndtri_ratio(_NDTRI_A, _NDTRI_B, central_r, const)

    # Outside the central band the branch is taken on whichever tail is the small one,
    # so the argument keeps its significant digits instead of cancelling against 1.
    # The halving is folded into the logarithm as -log(t / 2) == ln2 - log(t): forming
    # the product instead would flush the smallest float32 subnormals to zero, and this
    # way the sqrt argument is non-negative by construction, since t never exceeds 1
    tail_x = mx.where(q < const(0.0), x, const(2.0) - x)
    r = mx.minimum(mx.sqrt(const(_LN2) - mx.log(tail_x)), const(_NDTRI_MAX_R))

    mid = _ndtri_ratio(
        _NDTRI_C, _NDTRI_D, mx.minimum(r, split) - const(_NDTRI_MID_SHIFT), const
    )
    far = _ndtri_ratio(_NDTRI_E, _NDTRI_F, mx.maximum(r, split) - split, const)
    tail = mx.where(r <= split, mid, far)
    tail = mx.where(q < const(0.0), -tail, tail)

    return mx.where(mx.abs(q) <= half_width, central, tail)


@mlx_funcify.register(Erfcinv)
def mlx_funcify_Erfcinv(op, **kwargs):
    def erfcinv(x):
        z, const, out_dtype = _working_precision(x)

        # erfc(y) = 2 ndtr(-y sqrt(2)), so y = -ndtri(x / 2) / sqrt(2)
        out = -_ndtri_of_half(z, const) / const(_SQRT_2)

        # The poles are exact, and ndtri cannot reach them itself once r saturates. NaN
        # is computed rather than named: mx.compile inlines size-1 constants into its
        # generated source, where a bare `nan` is invalid in both backends -- Metal has
        # no such literal, and C++ resolves it to `nan(const char *)`. `inf` is fine.
        nan = const(0.0) / const(0.0)
        out = mx.where(z == const(0.0), const(float("inf")), out)
        out = mx.where(z == const(2.0), const(float("-inf")), out)
        out = mx.where((z < const(0.0)) | (z > const(2.0)), nan, out)
        return out.astype(out_dtype)

    return erfcinv
