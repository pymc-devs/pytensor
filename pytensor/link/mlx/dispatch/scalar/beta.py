from functools import cache

import mlx.core as mx
import numpy as np

from pytensor.link.mlx.dispatch.basic import mlx_funcify
from pytensor.link.mlx.dispatch.scalar.erf import _METAL_HEADER
from pytensor.link.mlx.dispatch.scalar.helpers import (
    _METAL_DEFICIT,
    _TINY,
    _exp,
    _log1p_deficit,
    _metal_constants,
    _working_precision,
)
from pytensor.scalar.math import BetaInc, Erfc


# Two expansions cover the plane between them:
#
#   Lentz's continued fraction for the 2F1 the function is a multiple of, taken on
#   whichever side of the reflection converges,
#   and Temme's uniform asymptotic expansion through the transition at large min(a, b),
#   where the fraction needs a number of terms growing like sqrt(min(a, b)).
#
# Which one owns the center is the reverse of what the shape of the problem suggests.
# The fraction converges slowest exactly at the mode, so Temme takes a window around it
# and the fraction takes the wings, where it converges in a handful of terms however
# large the parameters are.
_CF_TERMS = 120
# Temme is selected only where the fraction is running out of terms, which is governed
# by min(a, b) near the mode. Below this the fraction is the more accurate of the two --
# 1e-13 against 1e-11 -- so handing it territory it covers well costs accuracy rather
# than buying it.
_TEMME_MIN_A = 5000.0
_TEMME_MAX_ETA = 0.375
# How far down in p the tabulated c_1 to c_3 stay good. They are polynomials in
# s = (q - p) / sqrt(p q), which is unbounded as p falls, but they enter divided by mu,
# mu**2 and mu**3, so the leading coefficient is what governs -- and that one is taken
# in closed form below and has no such limit.
_TEMME_MIN_P = 0.002
# DLMF 8.18.11 gives c_0 in closed form, which unlike the table has no lower bound in p
# -- 300x to 16000x better below _TEMME_MIN_P. It is a difference of two quantities that
# both diverge as eta -> 0, so it carries about eps / |eta| and the table takes over near
# the mode, where it is the cancellation-free form of the same thing. The thresholds are
# where each dtype's loss falls under what the table delivers.
_C0_MIN_ETA = {mx.float32: 6e-3, mx.float64: 1e-5}

# log Gamma(z) - (z - 1/2) log z + z - log(2 pi) / 2, the Stirling remainder, is
# reached by shifting the argument up to where its asymptotic series holds and walking
# back down. Eight shifts put every positive argument past z = 8, where seven terms of
# the series leave 8e-16.
_STIRLING_SHIFT = 8
_STIRLING_SERIES = (
    1.0 / 12.0,
    -1.0 / 360.0,
    1.0 / 1260.0,
    -1.0 / 1680.0,
    1.0 / 1188.0,
    -691.0 / 360360.0,
    1.0 / 156.0,
)

# c_k(eta) from DLMF 8.18.9, the uniform asymptotic expansion of I_x(a, b) for large
# a + b. The variable eta is 8.18.10, which is what _betainc_eta computes, and x0 there
# is p here. The expansion is Temme's: N. M. Temme, "Incomplete Laplace integrals:
# uniform asymptotic expansion with application to the incomplete beta function", SIAM
# J. Math. Anal. 18 (1987), 1638-1663; the coefficients beyond the leading one are in
# Temme, "Special Functions" (Wiley, 1996), section 11.3.3.2.
#
# Tabulated here as a polynomial in eta by one in s = (q - p) / sqrt(p q), four terms in
# 1 / mu, rather than generated. To regenerate: extract c_k at a grid of (eta, p) by
# solving the Vandermonde in 1 / mu across four mu, taking I_x itself by quadrature at
# working precision above the mu eta^2 / 2 nats the subtraction cancels; then fit over
# |eta| <= _TEMME_MAX_ETA and p >= _TEMME_MIN_P, degree ten in each direction, using
# only monomials eta^n s^j with n + j odd -- every c_k is odd under (eta, s) ->
# (-eta, -s), which is why half the entries below are exactly zero.
#
# Checked against DLMF 8.18.11, which gives c_0 in closed form: the two agree to 1e-11
# at p = 0.5 and 2e-8 at p = 0.05, and diverge below that, which is where _TEMME_MIN_P
# comes from. DiDonato and Morris, Algorithm 708 (ACM TOMS 18, 1992, 360-373), generate
# these by recursion at runtime instead, which is what Boost and therefore scipy do.
_TEMME_C = (
    (
        (
            0.0,
            0.3333333330673329,
            0.0,
            1.5539175185861069e-10,
            0.0,
            -2.920830305321465e-11,
            0.0,
            1.6856009284486797e-12,
            0.0,
            1.1262800587996669e-14,
            0.0,
        ),
        (
            -0.2500000002055691,
            0.0,
            -0.08333333219515646,
            0.0,
            -7.92090282795456e-10,
            0.0,
            2.0427080404776532e-10,
            0.0,
            -2.3057979733464434e-11,
            0.0,
            1.183636522128495e-12,
        ),
        (
            0.0,
            0.0666667306856028,
            0.0,
            0.01481477748588048,
            0.0,
            5.436935483370178e-09,
            0.0,
            -1.0757698746567679e-10,
            0.0,
            -2.0902198374916914e-11,
            0.0,
        ),
        (
            -0.01041664562797473,
            0.0,
            -0.00694457022779069,
            0.0,
            -0.0011573054120818673,
            0.0,
            -2.855699426235203e-08,
            0.0,
            3.400104523453018e-09,
            0.0,
            -1.8088417569339654e-10,
        ),
        (
            0.0,
            -0.004765365617265621,
            0.0,
            -0.002643804166297746,
            0.0,
            -0.0003528566795309683,
            0.0,
            -2.0006076637338333e-08,
            0.0,
            2.626095924052052e-09,
            0.0,
        ),
        (
            0.00260326227402341,
            0.0,
            0.004276232356120527,
            0.0,
            0.0016044223427206757,
            0.0,
            0.00017997698497740014,
            0.0,
            -1.4530335865157312e-07,
            0.0,
            7.665302846127965e-09,
        ),
        (
            0.0,
            -0.0015271456059031486,
            0.0,
            -0.0014289974735331784,
            0.0,
            -0.0004154096627402427,
            0.0,
            -3.790262298631556e-05,
            0.0,
            -1.0333469243522034e-07,
            0.0,
        ),
        (
            0.00011251526200489227,
            0.0,
            0.00010118467100329492,
            0.0,
            0.00018306117005977989,
            0.0,
            6.2808164762707085e-06,
            0.0,
            4.554410629981336e-06,
            0.0,
            -1.2315601315843194e-07,
        ),
        (
            0.0,
            -0.00011585326654026974,
            0.0,
            0.00010844476672111029,
            0.0,
            0.00026449342757691374,
            0.0,
            -6.202124842528533e-07,
            0.0,
            3.5550928222894893e-06,
            0.0,
        ),
        (
            -0.0001400954648476741,
            0.0,
            0.0003271387226920435,
            0.0,
            -0.0006073364957785407,
            0.0,
            4.68474546381375e-05,
            0.0,
            -2.639553705002459e-05,
            0.0,
            -1.294225510676747e-07,
        ),
        (
            0.0,
            0.0006161282557355886,
            0.0,
            0.000861660811817412,
            0.0,
            -0.0004527883293880741,
            0.0,
            9.511180848095947e-05,
            0.0,
            -1.226634698990852e-06,
            0.0,
        ),
    ),
    (
        (
            0.0,
            0.05000003955652955,
            0.0,
            0.0018518618859275206,
            0.0,
            -8.042970825923601e-10,
            0.0,
            7.536039787740963e-11,
            0.0,
            -2.0852307646432658e-12,
            0.0,
        ),
        (
            0.031249954851909907,
            0.0,
            0.020833305745250746,
            0.0,
            0.0034722054278990516,
            0.0,
            2.9446175954387158e-09,
            0.0,
            -3.703376105315025e-10,
            0.0,
            2.7437389166218296e-11,
        ),
        (
            0.0,
            -0.03570995415567371,
            0.0,
            -0.019844264177143568,
            0.0,
            -0.0026448807498114435,
            0.0,
            -4.8710905365930215e-08,
            0.0,
            1.244408836392763e-09,
            0.0,
        ),
        (
            0.01562527829404281,
            0.0,
            0.02395657106341011,
            0.0,
            0.008913563395356877,
            0.0,
            0.0009897442411995667,
            0.0,
            6.270554661820006e-08,
            0.0,
            -4.620137046540667e-09,
        ),
        (
            0.0,
            -0.008583198867916218,
            0.0,
            -0.00723074090885379,
            0.0,
            -0.0021971325798586707,
            0.0,
            -0.00020286370462155572,
            0.0,
            -7.263032350936574e-08,
            0.0,
        ),
        (
            1.7072014100630306e-05,
            0.0,
            0.00014052779571564408,
            0.0,
            -6.055723737888833e-05,
            0.0,
            3.0074209491329825e-05,
            0.0,
            -2.7974112499998954e-06,
            0.0,
            2.2234358722835607e-07,
        ),
        (
            0.0,
            0.007066098468119804,
            0.0,
            -0.001199578281445901,
            0.0,
            0.001975076692095429,
            0.0,
            0.0001834558460908995,
            0.0,
            1.952664963902548e-05,
            0.0,
        ),
        (
            -0.00011862218937191998,
            0.0,
            -0.004117631744372812,
            0.0,
            0.00020458376417462195,
            0.0,
            -0.0012113465479392,
            0.0,
            -4.575554247433864e-05,
            0.0,
            -1.2096714071097026e-05,
        ),
        (
            0.0,
            -0.029590216415618653,
            0.0,
            0.011703819038236682,
            0.0,
            0.0017967814377304841,
            0.0,
            -0.00048712882358397324,
            0.0,
            8.995665296059247e-05,
            0.0,
        ),
        (
            -0.004118165856081443,
            0.0,
            0.024908904804722406,
            0.0,
            -0.020257395027483895,
            0.0,
            0.005806913047524963,
            0.0,
            -0.0006907349612788764,
            0.0,
            4.041284457297848e-05,
        ),
        (
            0.0,
            0.026937359638382476,
            0.0,
            0.05171563610971884,
            0.0,
            -0.03358250944618249,
            0.0,
            0.005693965139667003,
            0.0,
            -0.00037974640972195106,
            0.0,
        ),
    ),
    (
        (
            0.0,
            -0.061030941938799606,
            0.0,
            -0.032744956480224095,
            0.0,
            -0.004134887488774737,
            0.0,
            1.5975171684253723e-08,
            0.0,
            -4.2689879235779937e-10,
            0.0,
        ),
        (
            0.039075863698630066,
            0.0,
            0.06408298246675594,
            0.0,
            0.024142098051133272,
            0.0,
            0.002681873404745388,
            0.0,
            4.1041700549123176e-08,
            0.0,
            -1.5893464981597827e-09,
        ),
        (
            0.0,
            -0.03134271900916538,
            0.0,
            -0.02774767713369759,
            0.0,
            -0.008110636934082432,
            0.0,
            -0.0007719457297316669,
            0.0,
            1.7769347143009595e-08,
            0.0,
        ),
        (
            -0.00020987424570287957,
            0.0,
            5.3752461904680096e-05,
            0.0,
            -0.00032025844195836695,
            0.0,
            3.111196370565988e-05,
            0.0,
            -7.997233596420458e-06,
            0.0,
            2.395834098091465e-07,
        ),
        (
            0.0,
            0.01484641545401981,
            0.0,
            0.01321816393119183,
            0.0,
            0.007189180426689744,
            0.0,
            0.0014804255731605542,
            0.0,
            0.00010512690723822604,
            0.0,
        ),
        (
            -0.0013333793893443302,
            0.0,
            -0.024407489670825233,
            0.0,
            -0.0025890439276683207,
            0.0,
            -0.006881434184703189,
            0.0,
            -0.0005384207081098949,
            0.0,
            -6.32277234045367e-05,
        ),
        (
            0.0,
            0.04949648560224619,
            0.0,
            -0.14057869171656395,
            0.0,
            0.07103829871092703,
            0.0,
            -0.009195181646137439,
            0.0,
            0.0008688854725665378,
            0.0,
        ),
        (
            -0.03295791372138786,
            0.0,
            0.19188519304758386,
            0.0,
            -0.14866007741366752,
            0.0,
            0.03892323119421598,
            0.0,
            -0.004186398016387263,
            0.0,
            0.00016590124948628968,
        ),
        (
            0.0,
            -0.5834230152099557,
            0.0,
            1.452950206035147,
            0.0,
            -0.615577955910731,
            0.0,
            0.0874961762584248,
            0.0,
            -0.00476708391146138,
            0.0,
        ),
        (
            -0.008493635011707211,
            0.0,
            0.07766116983839798,
            0.0,
            -0.08908598624909647,
            0.0,
            0.03853342339497103,
            0.0,
            -0.006139273466633765,
            0.0,
            0.0006381982107579688,
        ),
        (
            0.0,
            -0.958134847011747,
            0.0,
            -0.48356601430358737,
            0.0,
            0.3994247952781336,
            0.0,
            -0.06096724499291866,
            0.0,
            0.0022382809001534627,
            0.0,
        ),
    ),
    (
        (
            0.0,
            -0.04704475118158762,
            0.0,
            -0.03227729999062412,
            0.0,
            -0.0075613217271153065,
            0.0,
            -0.000595822795933122,
            0.0,
            2.736193849446622e-08,
            0.0,
        ),
        (
            -0.011679996814917304,
            0.0,
            -0.023810829703130715,
            0.0,
            -0.014131876636867213,
            0.0,
            -0.003340839366082326,
            0.0,
            -0.00027785099757536764,
            0.0,
            -2.0977195981999874e-08,
        ),
        (
            0.0,
            0.04189775928338757,
            0.0,
            0.07139286993467533,
            0.0,
            0.031352407864550916,
            0.0,
            0.0066485675022520015,
            0.0,
            0.0004851289223570393,
            0.0,
        ),
        (
            -0.01769728583396981,
            0.0,
            -0.06041904250321112,
            0.0,
            -0.06194746314544572,
            0.0,
            -0.021848002056118587,
            0.0,
            -0.0041055377955119265,
            0.0,
            -0.00026268078963992414,
        ),
        (
            0.0,
            0.9905529646491856,
            0.0,
            -1.1012276438404136,
            0.0,
            0.40900843849933455,
            0.0,
            -0.04620880359086965,
            0.0,
            0.004437700379648635,
            0.0,
        ),
        (
            0.02230924934697756,
            0.0,
            -0.12741644573829658,
            0.0,
            0.10345027903758505,
            0.0,
            -0.025377646364489692,
            0.0,
            0.003135837109783171,
            0.0,
            -4.16407312702205e-05,
        ),
        (
            0.0,
            -14.354540880794751,
            0.0,
            14.887694845622782,
            0.0,
            -4.5938785769076205,
            0.0,
            0.5582472490151686,
            0.0,
            -0.02945597310034065,
            0.0,
        ),
        (
            -2.664929072732071,
            0.0,
            15.824223098902436,
            0.0,
            -12.571109870829268,
            0.0,
            3.4504654206585212,
            0.0,
            -0.3918760244218524,
            0.0,
            0.01938099383730385,
        ),
        (
            0.0,
            35.522210722526815,
            0.0,
            -6.8037869279724115,
            0.0,
            -4.1260012205469,
            0.0,
            1.027257000170558,
            0.0,
            -0.06657375213467454,
            0.0,
        ),
        (
            9.20550536153251,
            0.0,
            -53.853093922074336,
            0.0,
            41.97517170162134,
            0.0,
            -11.098577102981372,
            0.0,
            1.2085077282540322,
            0.0,
            -0.04947997022015238,
        ),
        (
            0.0,
            -37.698787768273434,
            0.0,
            -68.01939471085392,
            0.0,
            41.339781525945355,
            0.0,
            -6.492041259120653,
            0.0,
            0.3235251177174521,
            0.0,
        ),
    ),
)


# The kernel flattens the table and indexes it by strides taken from the first row, so
# a ragged re-fit would leave the vectorized path correct and the kernel mis-indexed
assert {len(rows) for rows in _TEMME_C} == {len(_TEMME_C[0])}, "ragged eta degree"
assert {len(row) for rows in _TEMME_C for row in rows} == {len(_TEMME_C[0][0])}, (
    "ragged s degree"
)


def _betainc_eta(p, q, x, const):
    r"""Temme's variable, with

    .. math::
        -\eta^2 / 2 = p \log(x / p) + q \log((1 - x) / q)

    and the sign of :math:`x - p`, so :math:`\eta` vanishes at the mode.

    Written as two deficits rather than two logs, the sum of which cancels to nothing
    as :math:`x \to p` -- the middle of the window the expansion owns. Taking
    :math:`q` from the caller rather than as :math:`1 - p` is the same argument one
    level up: once :math:`b / a` falls below the working epsilon, :math:`p` rounds to
    one and the difference is zero where the ratio is not.

    Parameters
    ----------
    p : mx.array
        ``a / (a + b)``.
    q : mx.array
        ``b / (a + b)``, formed as its own ratio rather than as ``1 - p``.
    x : mx.array
        Argument, at the working precision.
    const : callable
        Materializes a Python float at the working precision.
    """
    offset = x - p
    # x / p and (1 - x) / q are the shifted arguments formed without cancellation,
    # which is what keeps eta accurate in the deep tail where x approaches an endpoint
    value = p * _log1p_deficit(offset / p, x / p, const) + q * _log1p_deficit(
        -offset / q, (const(1.0) - x) / q, const
    )
    return mx.sign(offset) * mx.sqrt(mx.maximum(value, const(0.0)))


def _temme(mu, s, eta, leading, use_leading, sign, const, erfc):
    r"""The uniform asymptotic expansion, for large :math:`\mu = a + b` near the mode.

    Parameters
    ----------
    s : mx.array
        :math:`(q - p) / \sqrt{p q}`, the asymmetry the coefficients are tabulated in.
    leading : mx.array
        :math:`c_0` from DLMF 8.18.11, used where ``use_leading`` is true and the
        tabulated coefficient used elsewhere.
    use_leading : mx.array
        Where the closed form is far enough from :math:`\eta = 0` to be worth taking.
    sign : mx.array
        ``+1`` where :math:`I_x(a, b)` is wanted and ``-1`` where its complement is.
        Both come out of the same expansion, by

        .. math::
            \operatorname{erfc}(z) + \operatorname{erfc}(-z) = 2

        so each is built from terms of its own magnitude rather than as a subtraction
        from one.
    """
    total = const(0.0)
    mu_power = const(1.0)
    for order, rows in enumerate(_TEMME_C):
        coefficient = const(0.0)
        for row in reversed(rows):
            inner = const(0.0)
            for value in reversed(row):
                inner = inner * s + const(value)
            coefficient = coefficient * eta + inner
        if order == 0:
            coefficient = mx.where(use_leading, leading, coefficient)
        total = total + coefficient * mu_power
        mu_power = mu_power / mu

    correction = (
        _exp(const(-0.5) * mu * eta * eta, const)
        * mx.rsqrt(const(2.0 * np.pi) * mu)
        * total
    )
    tail = const(0.5) * erfc(-sign * eta * mx.sqrt(mu * const(0.5)))
    return tail + sign * correction


def _log_gamma_deficit(z, const):
    r"""What :math:`\log \Gamma(z)` carries beyond its leading Stirling terms.

    The prefactor needs :math:`\log \Gamma` only in a combination whose leading terms
    cancel analytically, so this returns the part that survives and the cancelling
    terms are never formed.
    """
    shifted = z + const(float(_STIRLING_SHIFT))
    inverse = const(1.0) / shifted
    square = inverse * inverse

    series = const(0.0)
    for value in reversed(_STIRLING_SERIES):
        series = series * square + const(value)
    total = series * inverse

    # Delta(z) = Delta(z + 1) + (z + 1/2) log1p(1/z) - 1, walked back down to z. The
    # subtraction here cancels only to order one, not to order z log z
    for step in range(_STIRLING_SHIFT):
        rung = z + const(float(step))
        total = total + (rung + const(0.5)) * mx.log1p(const(1.0) / rung) - const(1.0)
    return total


def _betainc_cf(a, b, x, half_mu_eta_squared, const):
    """``I_x(a, b)`` by Lentz's algorithm, on the side the caller has already chosen.

    The reflection is not an optimization here. Applied to the wrong side the fraction
    converges arbitrarily slowly, so under a fixed trip count it would return a partial
    sum rather than merely take longer.
    """
    tiny = const(_TINY)
    one = const(1.0)
    mu = a + b
    a_plus = a + one
    a_minus = a - one

    c = one
    d = one - mu * x / a_plus
    d = one / mx.where(mx.abs(d) < tiny, tiny, d)
    h = d

    for m in range(1, _CF_TERMS + 1):
        step = const(float(m))
        even = const(float(2 * m))
        numerators = (
            step * (b - step) * x / ((a_minus + even) * (a + even)),
            -(a + step) * (mu + step) * x / ((a + even) * (a_plus + even)),
        )
        for numerator in numerators:
            d = one + numerator * d
            d = mx.where(mx.abs(d) < tiny, tiny, d)
            c = one + numerator / c
            c = mx.where(mx.abs(c) < tiny, tiny, c)
            d = one / d
            h = h * d * c

    # x**a (1 - x)**b Gamma(a + b) / (Gamma(a + 1) Gamma(b)) in log space, split so that
    # nothing of order mu log mu is ever formed. Against p = a / mu the argument part is
    # exactly -mu eta**2 / 2, which the caller already has cancellation-free, and what
    # is left of the beta function is 0.5 log(a b / (2 pi mu)) plus three Stirling
    # remainders -- all of them order log mu or smaller.
    log_prefactor = (
        -half_mu_eta_squared
        + const(0.5) * (mx.log(a) + mx.log(b) - mx.log(mu) - const(np.log(2.0 * np.pi)))
        - mx.log(a)
        + _log_gamma_deficit(mu, const)
        - _log_gamma_deficit(a, const)
        - _log_gamma_deficit(b, const)
    )
    return h * _exp(log_prefactor, const)


def _betainc(a, b, x, min_eta, const, erfc):
    r"""The regularized incomplete beta function over the whole domain.

    Both expansions run for every element and one is selected afterwards. Temme is
    handed substitute arguments where it will not be selected, since its window is
    where the fraction's are singular; the fraction survives Temme's territory on the
    real ones and needs no such guard.
    """
    one = const(1.0)

    # The coefficient table is a polynomial in s, which is fitted for p <= 1/2 only;
    # above that it extrapolates and loses four orders. The normalization is free,
    # because the expansion delivers the complement directly
    swapped = a > b
    lower = mx.where(swapped, b, a)
    upper = mx.where(swapped, a, b)
    mu = a + b

    # eta is computed in the orientation it was given and negated, rather than from
    # 1 - x under the normalized one: eta is odd under the swap, and forming 1 - x
    # would throw away a small x entirely -- at float32 it costs 3% of x = 8e-7, and
    # the answer there is proportional to x**a
    p_given, q_given = a / mu, b / mu
    eta = _betainc_eta(p_given, q_given, x, const)
    eta = mx.where(swapped, -eta, eta)
    p = lower / mu
    q = upper / mu

    use_temme = (
        (lower >= const(_TEMME_MIN_A))
        & (p >= const(_TEMME_MIN_P))
        & (mx.abs(eta) <= const(_TEMME_MAX_ETA))
    )
    # c_0 by DLMF 8.18.11. Both x - p and eta are odd under the normalization, so the
    # ratio is taken in the orientation it was given and negated, never from 1 - x
    offset = mx.where(swapped, p_given - x, x - p_given)
    use_leading = use_temme & (mx.abs(eta) >= const(min_eta))
    leading = one / mx.where(use_leading, eta, one) - mx.sqrt(p * q) / mx.where(
        use_leading, offset, one
    )

    safe_p = mx.where(use_temme, p, const(0.5))
    safe_q = mx.where(use_temme, q, const(0.5))
    temme = _temme(
        mu,
        (safe_q - safe_p) * mx.rsqrt(safe_p * safe_q),
        mx.where(use_temme, eta, const(0.0)),
        leading,
        use_leading,
        mx.where(swapped, -one, one),
        const,
        erfc,
    )

    # The fraction keeps the original orientation: its own reflection already takes
    # whichever side converges, and it complements only towards an answer near one,
    # where the subtraction costs nothing
    reflected = x > (a + one) / (mu + const(2.0))
    # eta is odd under the reflection and under the a <= b normalization alike, so the
    # exponent the prefactor needs is the same one the region test already computed
    half_mu_eta_squared = const(0.5) * mu * eta * eta
    fraction = _betainc_cf(
        mx.where(reflected, b, a),
        mx.where(reflected, a, b),
        mx.where(reflected, one - x, x),
        half_mu_eta_squared,
        const,
    )
    fraction = mx.where(reflected, one - fraction, fraction)

    return mx.where(use_temme, temme, fraction)


# The vectorized form above is memory-bound rather than compute-bound, and severely: it
# peaks at 2.9 KB per element. A loop whose per-iteration coefficients are arrays
# materializes one buffer per iteration, because MLX evaluates all of those
# sub-expressions before the chain that consumes them, and the fraction's numerators are
# functions of a, b and x at all 240 levels. The kernel holds the recurrence in four
# registers and touches the buffer twice, which removes the problem rather than easing
# it -- 79 s to 10 ms at n = 1e7. It also takes one branch per element instead of both,
# and stops the fraction on convergence, a median of 8 terms against the 120 the graph
# must always unroll.
_METAL_BETAINC_HEADER = (
    _METAL_HEADER
    + _METAL_DEFICIT
    + _metal_constants(
        BI_TEMME_C=tuple(value for term in _TEMME_C for row in term for value in row),
        BI_STIRLING=_STIRLING_SERIES,
    )
    + f"""
#define BI_CF_TERMS       {_CF_TERMS}
#define BI_TEMME_MIN_A    {_TEMME_MIN_A}f
#define BI_TEMME_MIN_P    {_TEMME_MIN_P}f
#define BI_TEMME_MAX_ETA  {_TEMME_MAX_ETA}f
#define BI_C0_MIN_ETA     {_C0_MIN_ETA[mx.float32]}f
#define BI_STIRLING_SHIFT {_STIRLING_SHIFT}
#define BI_STIRLING_N     {len(_STIRLING_SERIES)}
#define BI_TEMME_TERMS    {len(_TEMME_C)}
#define BI_TEMME_ETA_N    {len(_TEMME_C[0])}
#define BI_TEMME_S_N      {len(_TEMME_C[0][0])}
#define BI_LOG_2PI        {float(np.log(2.0 * np.pi))!r}f
#define BI_TINY           {_TINY}f
// just under one float32 epsilon: below this a term no longer moves the fraction
#define BI_CONVERGED      1.0e-7f

static inline float bi_eta(float p, float q, float x) {{
    float offset = x - p;
    float value = p * log1p_deficit(offset / p, x / p)
                + q * log1p_deficit(-offset / q, (1.0f - x) / q);
    return sign(offset) * sqrt(fmax(value, 0.0f));
}}

// log Gamma(z) beyond its leading Stirling terms, shifted up to where the series holds
// and walked back down
static inline float bi_stirling(float z) {{
    float shifted = z + (float)BI_STIRLING_SHIFT;
    float inverse = 1.0f / shifted;
    float square = inverse * inverse;
    float series = 0.0f;
    for (int k = BI_STIRLING_N - 1; k >= 0; --k) series = series * square + BI_STIRLING[k];
    float total = series * inverse;
    for (int k = 0; k < BI_STIRLING_SHIFT; ++k) {{
        float rung = z + (float)k;
        total += (rung + 0.5f) * log1p(1.0f / rung) - 1.0f;
    }}
    return total;
}}

static inline float bi_temme(float mu, float p, float q, float eta, float offset,
                             float sgn) {{
    float s = (q - p) * rsqrt(p * q);
    float total = 0.0f;
    float mu_power = 1.0f;
    float inverse_mu = 1.0f / mu;
    for (int k = 0; k < BI_TEMME_TERMS; ++k) {{
        int term = k * BI_TEMME_ETA_N * BI_TEMME_S_N;
        float coefficient = 0.0f;
        for (int n = BI_TEMME_ETA_N - 1; n >= 0; --n) {{
            int row = term + n * BI_TEMME_S_N;
            float inner = 0.0f;
            for (int j = BI_TEMME_S_N - 1; j >= 0; --j) inner = inner * s + BI_TEMME_C[row + j];
            coefficient = coefficient * eta + inner;
        }}
        // DLMF 8.18.11, which the table only replaces near eta = 0 where it cancels
        if (k == 0 && fabs(eta) >= BI_C0_MIN_ETA) {{
            coefficient = 1.0f / eta - sqrt(p * q) / offset;
        }}
        total += coefficient * mu_power;
        mu_power *= inverse_mu;
    }}
    float correction = exp(-0.5f * mu * eta * eta) * rsqrt(2.0f * M_PI_F * mu) * total;
    return 0.5f * erfc_full(-sgn * eta * sqrt(mu * 0.5f)) + sgn * correction;
}}

static inline float bi_cf(float a, float b, float x, float half_mu_eta_squared) {{
    float total = a + b;
    float a_plus = a + 1.0f;
    float a_minus = a - 1.0f;
    float c = 1.0f;
    float d = 1.0f - total * x / a_plus;
    if (fabs(d) < BI_TINY) d = BI_TINY;
    d = 1.0f / d;
    float h = d;

    for (int m = 1; m <= BI_CF_TERMS; ++m) {{
        float step = (float)m;
        float even = (float)(2 * m);
        float first = step * (b - step) * x / ((a_minus + even) * (a + even));
        float second = -(a + step) * (total + step) * x / ((a + even) * (a_plus + even));
        float delta = 1.0f;
        for (int part = 0; part < 2; ++part) {{
            float numerator = (part == 0) ? first : second;
            d = 1.0f + numerator * d;
            if (fabs(d) < BI_TINY) d = BI_TINY;
            c = 1.0f + numerator / c;
            if (fabs(c) < BI_TINY) c = BI_TINY;
            d = 1.0f / d;
            delta = d * c;
            h *= delta;
        }}
        if (fabs(delta - 1.0f) <= BI_CONVERGED) break;
    }}

    // split so that nothing of order mu log mu is ever formed
    float log_prefactor = -half_mu_eta_squared
        + 0.5f * (log(a) + log(b) - log(total) - BI_LOG_2PI)
        - log(a)
        + bi_stirling(total) - bi_stirling(a) - bi_stirling(b);
    return h * exp(log_prefactor);
}}
"""
)


_METAL_BETAINC_SOURCE = """
    uint i = thread_position_in_grid.x;
    float a = (float)a_buf[i];
    float b = (float)b_buf[i];
    float x = (float)x_buf[i];
    float mu = a + b;

    // eta is taken in the orientation it was given and negated, never from 1 - x, which
    // would throw away a small x entirely
    float p_given = a / mu;
    float eta = bi_eta(p_given, b / mu, x);
    bool swapped = a > b;
    float lower = swapped ? b : a;
    float p = lower / mu;
    float q = (swapped ? a : b) / mu;
    // x - p and eta are both odd under the normalization, so neither is taken from 1 - x
    float offset = swapped ? (p_given - x) : (x - p_given);
    if (swapped) eta = -eta;

    float res;
    if (lower >= BI_TEMME_MIN_A && p >= BI_TEMME_MIN_P && fabs(eta) <= BI_TEMME_MAX_ETA) {
        res = bi_temme(mu, p, q, eta, offset, swapped ? -1.0f : 1.0f);
    } else {
        bool reflected = x > (a + 1.0f) / (mu + 2.0f);
        float first = reflected ? b : a;
        float second = reflected ? a : b;
        float value = reflected ? (1.0f - x) : x;
        float tail = bi_cf(first, second, value, 0.5f * mu * eta * eta);
        res = reflected ? 1.0f - tail : tail;
    }
    out[i] = (T)res;
"""


@cache
def _metal_betainc_kernel():
    """Build and cache the Metal kernel, or None if it cannot be built.

    The guard catches a failure to construct the kernel object. It does not catch a
    malformed source: Metal compiles lazily on first call and aborts the process rather
    than raising, so the source has no runtime safety net and the fallback cannot stand
    in for one.
    """
    try:
        return mx.fast.metal_kernel(
            name="pytensor_betainc",
            input_names=["a_buf", "b_buf", "x_buf"],
            output_names=["out"],
            header=_METAL_BETAINC_HEADER,
            source=_METAL_BETAINC_SOURCE,
        )
    except Exception:
        return None


def _metal_betainc_call(a, b, x):
    """Evaluate through the Metal kernel, or return None to fall back.

    Metal has no float64 and the kernel needs the GPU stream, which between them leave
    exactly the case the vectorized path handles least well and float64 needs least: the
    default device at single precision.
    """
    if (
        a.dtype != mx.float32
        or b.dtype != mx.float32
        or x.dtype != mx.float32
        or not mx.metal.is_available()
        or mx.default_device() != mx.gpu
    ):
        return None
    kernel = _metal_betainc_kernel()
    if kernel is None:
        return None
    # The kernel indexes three flat buffers in lockstep, so the operands are broadcast
    # against each other first: the op admits scalar shape parameters against an array of
    # values, and a 0-d input binds as a value rather than a pointer and cannot be
    # subscripted
    first, second, value = mx.broadcast_arrays(a, b, x)
    flat = [operand.reshape(-1) for operand in (first, second, value)]
    (out,) = kernel(
        inputs=flat,
        template=[("T", flat[-1].dtype)],
        grid=(flat[-1].size, 1, 1),
        threadgroup=(min(256, max(flat[-1].size, 1)), 1, 1),
        output_shapes=[flat[-1].shape],
        output_dtypes=[flat[-1].dtype],
    )
    return out.reshape(value.shape)


@mlx_funcify.register(BetaInc)
def mlx_funcify_BetaInc(op, **kwargs):
    erfc = mlx_funcify(Erfc())

    def betainc(a, b, x):
        z, const, out_dtype = _working_precision(x)
        first = mx.array(a).astype(z.dtype)
        second = mx.array(b).astype(z.dtype)
        out = _metal_betainc_call(first, second, z)
        if out is None:
            out = _betainc(first, second, z, _C0_MIN_ETA[z.dtype], const, erfc)
        return out.astype(out_dtype)

    return betainc
