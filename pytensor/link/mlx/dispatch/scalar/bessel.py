import mlx.core as mx

from pytensor.link.mlx.dispatch.basic import mlx_funcify
from pytensor.link.mlx.dispatch.scalar.helpers import _exp, _working_precision
from pytensor.scalar.math import I0, I1


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
