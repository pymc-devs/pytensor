from collections.abc import Callable

import mlx.core as mx
import numpy as np
from scipy.special import gammaln as scipy_gammaln

from pytensor.graph.basic import Constant
from pytensor.graph.traversal import ancestors
from pytensor.link.mlx.dispatch.basic import mlx_funcify
from pytensor.link.mlx.dispatch.scalar.helpers import _exp, _working_precision
from pytensor.scalar.math import I0, I1, Ive


# Chebyshev expansions of the exponentially scaled modified Bessel functions, from the
# netlib Cephes ``i0.c`` and ``i1.c``, transcribed in their own arrangement: highest
# degree first, with ``chbevl`` halving the trailing term, and the same interval split
# and argument transformations the C source uses. Keeping Cephes' layout is what makes
# the coefficients checkable -- they are unreviewable by eye, and the only practical
# review is a diff against the source they came from.
#
# The scaled forms are the primitives because they are bounded on the whole line, where
# i0 itself overflows past x = 713 at float64 and x = 88 at float32 -- a von Mises
# concentration well inside what a user will write.
#
# A Chebyshev series is near-minimax at every truncation, so float32 takes a suffix of
# the same coefficients rather than a second fitted set: the leading entries are the
# high-order terms, each costs a pass over the array, and single precision cannot see
# the ones dropped.
_SPLIT = 8.0
_SINGLE_SMALL_TERMS = 18
_SINGLE_LARGE_TERMS = 10

_I0E_A = (
    -4.41534164647933937950e-18,
    3.33079451882223809783e-17,
    -2.43127984654795469359e-16,
    1.71539128555513303061e-15,
    -1.16853328779934516808e-14,
    7.67618549860493561688e-14,
    -4.85644678311192946090e-13,
    2.95505266312963983461e-12,
    -1.72682629144155570723e-11,
    9.67580903537323691224e-11,
    -5.18979560163526290666e-10,
    2.65982372468238665035e-9,
    -1.30002500998624804212e-8,
    6.04699502254191894932e-8,
    -2.67079385394061173391e-7,
    1.11738753912010371815e-6,
    -4.41673835845875056359e-6,
    1.64484480707288970893e-5,
    -5.75419501008210370398e-5,
    1.88502885095841655729e-4,
    -5.76375574538582365885e-4,
    1.63947561694133579842e-3,
    -4.32430999505057594430e-3,
    1.05464603945949983183e-2,
    -2.37374148058994688156e-2,
    4.93052842396707084878e-2,
    -9.49010970480476444210e-2,
    1.71620901522208775349e-1,
    -3.04682672343198398683e-1,
    6.76795274409476084995e-1,
)

_I0E_A_SINGLE = _I0E_A[-_SINGLE_SMALL_TERMS:]

_I0E_B = (
    -7.23318048787475395456e-18,
    -4.83050448594418207126e-18,
    4.46562142029675999901e-17,
    3.46122286769746109310e-17,
    -2.82762398051658348494e-16,
    -3.42548561967721913462e-16,
    1.77256013305652638360e-15,
    3.81168066935262242075e-15,
    -9.55484669882830764870e-15,
    -4.15056934728722208663e-14,
    1.54008621752140982691e-14,
    3.85277838274214270114e-13,
    7.18012445138366623367e-13,
    -1.79417853150680611778e-12,
    -1.32158118404477131188e-11,
    -3.14991652796324136454e-11,
    1.18891471078464383424e-11,
    4.94060238822496958910e-10,
    3.39623202570838634515e-9,
    2.26666899049817806459e-8,
    2.04891858946906374183e-7,
    2.89137052083475648297e-6,
    6.88975834691682398426e-5,
    3.36911647825569408990e-3,
    8.04490411014108831608e-1,
)

_I0E_B_SINGLE = _I0E_B[-_SINGLE_LARGE_TERMS:]

_I1E_A = (
    2.77791411276104639959e-18,
    -2.11142121435816608115e-17,
    1.55363195773620046921e-16,
    -1.10559694773538630805e-15,
    7.60068429473540693410e-15,
    -5.04218550472791168711e-14,
    3.22379336594557470981e-13,
    -1.98397439776494371520e-12,
    1.17361862988909016308e-11,
    -6.66348972350202774223e-11,
    3.62559028155211703701e-10,
    -1.88724975172282928790e-9,
    9.38153738649577178388e-9,
    -4.44505912879632808065e-8,
    2.00329475355213526229e-7,
    -8.56872026469545474066e-7,
    3.47025130813767847674e-6,
    -1.32731636560394358279e-5,
    4.78156510755005422638e-5,
    -1.61760815825896745588e-4,
    5.12285956168575772895e-4,
    -1.51357245063125314899e-3,
    4.15642294431288815669e-3,
    -1.05640848946261981558e-2,
    2.47264490306265168283e-2,
    -5.29459812080949914269e-2,
    1.02643658689847095384e-1,
    -1.76416518357834055153e-1,
    2.52587186443633654823e-1,
)

_I1E_A_SINGLE = _I1E_A[-_SINGLE_SMALL_TERMS:]

_I1E_B = (
    7.51729631084210481353e-18,
    4.41434832307170791151e-18,
    -4.65030536848935832153e-17,
    -3.20952592199342395980e-17,
    2.96262899764595013876e-16,
    3.30820231092092828324e-16,
    -1.88035477551078244854e-15,
    -3.81440307243700780478e-15,
    1.04202769841288027642e-14,
    4.27244001671195135429e-14,
    -2.10154184277266431302e-14,
    -4.08355111109219731823e-13,
    -7.19855177624590851209e-13,
    2.03562854414708950722e-12,
    1.41258074366137813316e-11,
    3.25260358301548823856e-11,
    -1.89749581235054123450e-11,
    -5.58974346219658380687e-10,
    -3.83538038596423702205e-9,
    -2.63146884688951950684e-8,
    -2.51223623787020892529e-7,
    -3.88256480887769039346e-6,
    -1.10588938762623716291e-4,
    -9.76109749136146840777e-3,
    7.78576235018280120474e-1,
)

_I1E_B_SINGLE = _I1E_B[-_SINGLE_LARGE_TERMS:]


def _chbevl(coeffs, x, const):
    """Evaluate a Chebyshev series at ``x`` in [-2, 2], in Cephes' arrangement.

    ``coeffs`` runs from the highest degree down, and the constant term is halved on the
    way out, so an array transcribed from Cephes is used exactly as it appears there.
    """
    b0, b1, b2 = const(coeffs[0]), const(0.0), const(0.0)
    for coeff in coeffs[1:]:
        b2, b1 = b1, b0
        b0 = x * b1 - b2 + const(coeff)
    return const(0.5) * (b0 - b2)


def _large_series(y, coeffs, const):
    """Evaluate the ``y > 8`` series, which carries an explicit ``sqrt(y)``.

    Callers pass unclamped ``y``: both intervals are evaluated for every element and
    selected afterwards, so this clamps its own argument rather than trusting one that
    would otherwise run the series far outside where it was fitted.
    """
    y = mx.maximum(y, const(_SPLIT))
    return _chbevl(coeffs, const(32.0) / y - const(2.0), const) / mx.sqrt(y)


def _i0e_positive(y, const):
    """``i0e`` for non-negative ``y``, as a series on whichever interval holds it."""
    double = y.dtype == mx.float64
    split = const(_SPLIT)

    small_t = mx.minimum(y, split) / const(2.0) - const(2.0)
    small = _chbevl(_I0E_A if double else _I0E_A_SINGLE, small_t, const)

    large = _large_series(y, _I0E_B if double else _I0E_B_SINGLE, const)
    return mx.where(y <= split, small, large)


def _i1e_positive(y, const):
    """``i1e`` for non-negative ``y``, as a series on whichever interval holds it."""
    double = y.dtype == mx.float64
    split = const(_SPLIT)

    # i1e vanishes like y / 2 at the origin, so the small-interval series is in
    # i1e(y) / y and is multiplied back. A series fitted to i1e itself holds absolute
    # error and lets the relative error run away as y -> 0
    small_y = mx.minimum(y, split)
    small_t = small_y / const(2.0) - const(2.0)
    small = _chbevl(_I1E_A if double else _I1E_A_SINGLE, small_t, const) * small_y

    large = _large_series(y, _I1E_B if double else _I1E_B_SINGLE, const)
    return mx.where(y <= split, small, large)


@mlx_funcify.register(I0)
def mlx_funcify_I0(op, **kwargs):
    def i0(x):
        z, const, out_dtype = _working_precision(x)

        # i0 is even, and the series computes the scaled form. Restoring the scale is
        # what caps the accuracy: exp is the one operation here that MLX cannot do
        # better than the exponent's own rounding.
        y = mx.abs(z)
        out = _i0e_positive(y, const) * _exp(y, const)
        return out.astype(out_dtype)

    return i0


@mlx_funcify.register(I1)
def mlx_funcify_I1(op, **kwargs):
    def i1(x):
        z, const, out_dtype = _working_precision(x)

        # i1 is odd, so the series runs on |x| and the sign is restored afterwards
        y = mx.abs(z)
        out = _i1e_positive(y, const) * _exp(y, const)
        return mx.where(z < const(0.0), -out, out).astype(out_dtype)

    return i1


# ive at real order is a different construction from i0 and i1: an ascending power series
# below x = 20 and a Hankel asymptotic expansion above it, rather than a Chebyshev fit.
# Both branches are polynomials once the order is known, which is why the order has to be
# a constant -- see mlx_funcify_Ive.
# The Hankel expansion needs x large against v**2, and the series needs a term for
# roughly every x / 2, so both the split and the trip count follow the order. Validated
# against scipy to 3e-14 over -20 <= v <= 20; past that the series grows faster than it
# is worth and the order is refused rather than approximated
_IVE_MAX_ORDER = 20.0
_IVE_ASYMPTOTIC_TERMS = 20


def _ive_split(order):
    """Argument at which the ascending series gives way to the Hankel expansion."""
    return np.maximum(20.0, 0.25 * order * order + 0.5 * np.abs(order) + 15.0)


def _ive_series_terms(order):
    """Trip count the ascending series needs to reach :func:`_ive_split`.

    One count covers the whole order array, since the loop is unrolled into the graph
    and cannot vary per element.
    """
    return int(0.75 * np.max(_ive_split(order)) + 30)


def _ive_series_step(order, k):
    """Ratio of consecutive ascending-series terms, ``t_k / t_{k-1}``, less ``x**2 / 4``."""
    return 1.0 / (k * (order + k))


def _ive_asymptotic_coeffs(order, n_terms):
    """Hankel coefficients ``a_k`` in ``ive(v, x) ~ sum (-1)**k a_k / x**k / sqrt(2 pi x)``.

    The expansion terminates for half-integer orders, where ``mu`` meets an odd square
    exactly and every later coefficient is zero.
    """
    mu = 4.0 * order * order
    term = np.ones_like(mu)
    coeffs = [term]
    for k in range(1, n_terms):
        term = term * -(mu - (2 * k - 1) ** 2) / (8.0 * k)
        coeffs.append(term)
    return tuple(coeffs)


# The vectorized ive evaluates a fixed trip count for every element and both branches on
# top of that -- 62x a single fused pass at n = 1e7, the worst ratio in this module. A
# kernel takes the branch and stops the series when it has converged, which for most
# arguments is long before the worst-case term count. The order is a constant, so its
# split, trip count and 1 / Gamma(v + 1) are baked into the source and the kernel is
# cached per order.
_METAL_IVE_SOURCE = """
    uint i = thread_position_in_grid.x;
    float y = fabs((float)x[i]);
    float res;
    if (y <= IVE_SPLIT) {
        float quarter = 0.25f * y * y;
        float term = 1.0f;
        float total = 1.0f;
        for (int k = 1; k < IVE_TERMS; ++k) {
            term *= quarter / (float)(k * (IVE_ORDER + k));
            total += term;
            // the terms alternate in sign for a negative non-integer order, so the
            // convergence test has to be on the magnitude or it fires on the first one
            if (fabs(term) <= 1e-9f * fabs(total)) break;
        }
        float scale = IVE_RGAMMA * exp(-y);
        if (IVE_ORDER != 0.0f) scale *= pow(0.5f * y, IVE_ORDER);
        res = total * scale;
    } else {
        float term = 1.0f;
        float total = 1.0f;
        for (int k = 1; k < IVE_ASYMPTOTIC_TERMS; ++k) {
            term *= -(IVE_MU - (float)((2 * k - 1) * (2 * k - 1))) / (8.0f * y * (float)k);
            total += term;
        }
        res = total * rsqrt(2.0f * 3.14159265358979323846f * y);
    }
    out[i] = (T)res;
"""


_METAL_IVE_KERNELS: dict[float, Callable | None] = {}


def _metal_ive_kernel(order):
    """Build and cache the Metal kernel for ``order``, or None if it cannot be built."""
    # .item() rather than float(): a scalar order arrives with shape (1,) once the op
    # has broadcast it, and NumPy is retiring the implicit conversion of those
    order = np.asarray(order).item()
    if order not in _METAL_IVE_KERNELS:
        # the trailing newline matters: MLX appends its own template declaration straight
        # after this header, and a #define running into it fails to compile
        header = (
            "\n".join(
                f"#define {name} {value}"
                for name, value in (
                    ("IVE_SPLIT", f"{_ive_split(order):.8f}f"),
                    ("IVE_TERMS", _ive_series_terms(order)),
                    ("IVE_ORDER", f"{order:.8f}f"),
                    ("IVE_MU", f"{4.0 * order * order:.8f}f"),
                    ("IVE_RGAMMA", f"{np.exp(-scipy_gammaln(order + 1.0)):.10e}f"),
                    ("IVE_ASYMPTOTIC_TERMS", _IVE_ASYMPTOTIC_TERMS),
                )
            )
            + "\n"
        )
        try:
            _METAL_IVE_KERNELS[order] = mx.fast.metal_kernel(
                name=f"pytensor_ive_{str(order).replace('.', '_').replace('-', 'neg')}",
                input_names=["x"],
                output_names=["out"],
                header=header,
                source=_METAL_IVE_SOURCE,
            )
        except Exception:
            _METAL_IVE_KERNELS[order] = None
    return _METAL_IVE_KERNELS[order]


def _metal_ive_call(order, y):
    """Evaluate ``ive(order, y)`` through its Metal kernel, or None to fall back.

    The kernel bakes the order into its source, so it serves a single order only; a
    vector of orders takes the vectorized path, where they cost one evaluation between
    them rather than one each. A scalar order reaches this with shape ``(1,)`` when the
    argument is an array, since the op broadcasts it, so size rather than rank decides.
    """
    if (
        np.size(order) != 1
        or y.dtype != mx.float32
        or not mx.metal.is_available()
        or mx.default_device() != mx.gpu
    ):
        return None
    kernel = _metal_ive_kernel(order)
    if kernel is None:
        return None
    flat = y.reshape(-1)
    (out,) = kernel(
        inputs=[flat],
        template=[("T", flat.dtype)],
        grid=(flat.size, 1, 1),
        threadgroup=(min(256, max(flat.size, 1)), 1, 1),
        output_shapes=[flat.shape],
        output_dtypes=[flat.dtype],
    )
    return out.reshape(y.shape)


def _ive_series(y, order, const):
    """``ive(order, y)`` by the ascending series, for ``y`` at or below the split."""
    quarter_sq = y * y * const(0.25)
    total = const(1.0)
    term = const(1.0)
    for k in range(1, _ive_series_terms(order)):
        term = term * quarter_sq * const(_ive_series_step(order, k))
        total = total + term

    # The (y/2)**v / Gamma(v+1) prefactor is taken in log space so that a large order
    # does not overflow on its way to a small answer. Only y = 0 has to be kept out of
    # the logarithm, and its limits are exact anyway: 1, 0 and a pole for a zero,
    # positive and negative order. Clamping the argument instead of substituting for it
    # would floor every y below the clamp along with it
    safe_y = mx.where(y > const(0.0), y, const(1.0))
    log_prefactor = (
        -y
        - const(scipy_gammaln(order + 1.0))
        + const(order) * mx.log(safe_y * const(0.5))
    )
    out = total * _exp(log_prefactor, const)
    return mx.where(y == const(0.0), const(np.where(order == 0.0, 1.0, 0.0)), out)


def _ive_asymptotic(y, order, const):
    """``ive(order, y)`` by the Hankel expansion, for ``y`` at or above the split.

    A plain polynomial in ``1 / y`` once the order is fixed.
    """
    inv_y = const(1.0) / y
    coeffs = _ive_asymptotic_coeffs(order, _IVE_ASYMPTOTIC_TERMS)
    acc = const(coeffs[-1])
    for c in reversed(coeffs[:-1]):
        acc = acc * inv_y + const(c)
    return acc * mx.rsqrt(const(2.0 * np.pi) * y)


def _ive_positive(y, order, const):
    """``ive(order, y)`` for non-negative ``y`` at a known constant ``order``.

    ``order`` is a NumPy array broadcasting against ``y``, so a graph asking for a whole
    vector of orders at once -- as the HSGP approximation of a periodic kernel does --
    is one evaluation rather than one per order.
    """
    fast = _metal_ive_call(order, y)
    if fast is not None:
        return fast

    # Both branches run for every element and are selected afterwards, so each is
    # clamped into its own interval first
    split = const(_ive_split(order))
    small = _ive_series(mx.minimum(y, split), order, const)
    large = _ive_asymptotic(mx.maximum(y, split), order, const)
    return mx.where(y <= split, small, large)


def _constant_order(var):
    """Return the order as a NumPy array, or None when it is not fixed by the graph.

    The order is used with the shape it has in the graph, so it broadcasts against the
    argument exactly as the op does -- a whole vector of orders is one evaluation.
    """
    if isinstance(var, Constant):
        return np.asarray(var.data, dtype="float64")
    if all(
        root.owner is not None or isinstance(root, Constant)
        for root in ancestors([var])
    ):
        return np.asarray(var.eval(), dtype="float64")
    return None


@mlx_funcify.register(Ive)
def mlx_funcify_Ive(op, node=None, **kwargs):
    order = None if node is None else _constant_order(node.inputs[0])
    if order is None:
        raise NotImplementedError(
            "MLX ive requires a constant order. The series is unrolled into the graph, "
            "so its trip count has to be known, and the coefficients of both branches "
            "along with the negative-integer and negative-argument rules are all fixed "
            "by the order. A symbolic order needs a different implementation rather "
            "than a different constant. An array of orders is fine: it is used with the "
            "shape it has in the graph, so a vector of them costs one evaluation."
        )
    order = np.asarray(order, dtype="float64")
    if np.max(np.abs(order)) > _IVE_MAX_ORDER:
        raise NotImplementedError(
            f"MLX ive supports orders up to {_IVE_MAX_ORDER:g}, not "
            f"{np.max(np.abs(order)):g}. The ascending series needs a term for roughly "
            "every x / 2 up to where the Hankel expansion takes over, and that "
            "crossover grows with the square of the order."
        )

    # I(-n, x) = I(n, x) for integer n. Reflecting here keeps the series ratio away from
    # the pole of Gamma at a non-positive integer, which is where 1 / (k + v) divides by
    # zero
    is_integer = order == np.round(order)
    reflected = np.where(is_integer & (order < 0.0), -order, order)

    # Only integer orders continue to negative argument, with the parity of the order;
    # everything else is outside the real domain there, as is zero for a negative
    # non-integer order
    negative_sign = np.where(is_integer & (np.round(reflected) % 2 == 1), -1.0, 1.0)
    negative_is_nan = ~is_integer
    zero_is_pole = negative_is_nan & (order < 0.0)

    def ive(v, x):
        z, const, out_dtype = _working_precision(x)
        out = _ive_positive(mx.abs(z), reflected, const)

        out = mx.where(z < const(0.0), out * const(negative_sign), out)

        # NaN is computed rather than named: mx.compile inlines a size-1 constant into
        # its generated source, where a bare `nan` is a literal neither backend has
        nan = const(0.0) / const(0.0)
        if negative_is_nan.any():
            off_domain = (z < const(0.0)) & (
                const(negative_is_nan.astype("float64")) != const(0.0)
            )
            out = mx.where(off_domain, nan, out)
        if zero_is_pole.any():
            at_pole = (z == const(0.0)) & (
                const(zero_is_pole.astype("float64")) != const(0.0)
            )
            out = mx.where(at_pole, nan, out)

        # The series tends to zero at either infinity, but scipy reports NaN there and
        # the op has to agree with it across backends
        out = mx.where(mx.isinf(z), nan, out)
        return out.astype(out_dtype)

    return ive
