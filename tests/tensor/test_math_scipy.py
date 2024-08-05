import warnings

import numpy as np
import pytest

from pytensor.gradient import verify_grad
from pytensor.scalar import ScalarLoop
from pytensor.tensor.elemwise import Elemwise


scipy = pytest.importorskip("scipy")

from functools import partial

from scipy import special, stats

from pytensor import function, grad
from pytensor import tensor as pt
from pytensor.compile.mode import get_default_mode
from pytensor.configdefaults import config
from pytensor.tensor import gammaincc, inplace, vector
from tests import unittest_tools as utt
from tests.tensor.utils import (
    _good_broadcast_unary_chi2sf,
    _good_broadcast_unary_normal,
    _good_broadcast_unary_normal_float,
    _good_broadcast_unary_normal_float_no_complex,
    _good_broadcast_unary_normal_float_no_complex_small_neg_range,
    _good_broadcast_unary_normal_no_complex,
    _grad_broadcast_unary_0_2_no_complex,
    _grad_broadcast_unary_abs1_no_complex,
    _grad_broadcast_unary_normal,
    _grad_broadcast_unary_normal_small_neg_range,
    check_floatX,
    copymod,
    integers_ranged,
    makeBroadcastTester,
    random_ranged,
    upcast_int8_nfunc,
)


mode_no_scipy = get_default_mode()


def scipy_special_gammau(k, x):
    return special.gammaincc(k, x) * special.gamma(k)


def scipy_special_gammal(k, x):
    return special.gammainc(k, x) * special.gamma(k)


# Precomputing the result is brittle(it have been broken!)
# As if we do any modification to random number here,
# The input random number will change and the output!
expected_erf = special.erf
expected_erfc = special.erfc
expected_erfinv = special.erfinv
expected_erfcinv = special.erfcinv
expected_owenst = special.owens_t
expected_gamma = special.gamma
expected_gammaln = special.gammaln
expected_psi = special.psi
expected_tri_gamma = partial(special.polygamma, 1)
expected_chi2sf = stats.chi2.sf
expected_gammainc = special.gammainc
expected_gammaincc = special.gammaincc
expected_gammau = scipy_special_gammau
expected_gammal = scipy_special_gammal
expected_gammaincinv = special.gammaincinv
expected_gammainccinv = special.gammainccinv
expected_j0 = special.j0
expected_j1 = special.j1
expected_jv = special.jv
expected_i0 = special.i0
expected_i1 = special.i1
expected_iv = special.iv
expected_ive = special.ive
expected_erfcx = special.erfcx
expected_sigmoid = special.expit
expected_hyp2f1 = special.hyp2f1
expected_betaincinv = special.betaincinv

TestErfBroadcast = makeBroadcastTester(
    op=pt.erf,
    expected=expected_erf,
    good=_good_broadcast_unary_normal,
    grad=_grad_broadcast_unary_normal,
    eps=2e-10,
    mode=mode_no_scipy,
)
TestErfInplaceBroadcast = makeBroadcastTester(
    op=inplace.erf_inplace,
    expected=expected_erf,
    good=_good_broadcast_unary_normal_float,
    mode=mode_no_scipy,
    eps=2e-10,
    inplace=True,
)

TestErfcBroadcast = makeBroadcastTester(
    op=pt.erfc,
    expected=expected_erfc,
    good=_good_broadcast_unary_normal_float_no_complex,
    grad=_grad_broadcast_unary_normal,
    eps=2e-10,
    mode=mode_no_scipy,
)
TestErfcInplaceBroadcast = makeBroadcastTester(
    op=inplace.erfc_inplace,
    expected=expected_erfc,
    good=_good_broadcast_unary_normal_float_no_complex,
    eps=2e-10,
    mode=mode_no_scipy,
    inplace=True,
)

TestErfcxBroadcast = makeBroadcastTester(
    op=pt.erfcx,
    expected=expected_erfcx,
    good=_good_broadcast_unary_normal_float_no_complex_small_neg_range,
    grad=_grad_broadcast_unary_normal_small_neg_range,
    eps=2e-10,
    mode=mode_no_scipy,
)
TestErfcxInplaceBroadcast = makeBroadcastTester(
    op=inplace.erfcx_inplace,
    expected=expected_erfcx,
    good=_good_broadcast_unary_normal_float_no_complex_small_neg_range,
    eps=2e-10,
    mode=mode_no_scipy,
    inplace=True,
)

TestErfinvBroadcast = makeBroadcastTester(
    op=pt.erfinv,
    expected=expected_erfinv,
    good={
        "normal": [random_ranged(-0.9, 0.9, (2, 3))],
        "empty": [np.asarray([], dtype=config.floatX)],
    },
    grad=_grad_broadcast_unary_abs1_no_complex,
    eps=2e-10,
    mode=mode_no_scipy,
)

TestErfcinvBroadcast = makeBroadcastTester(
    op=pt.erfcinv,
    expected=expected_erfcinv,
    good={
        "normal": [random_ranged(0.001, 1.9, (2, 3))],
        "empty": [np.asarray([], dtype=config.floatX)],
    },
    grad=_grad_broadcast_unary_0_2_no_complex,
    eps=2e-10,
    mode=mode_no_scipy,
)

rng = np.random.default_rng(seed=utt.fetch_seed())
_good_broadcast_binary_owenst = dict(
    normal=(
        random_ranged(-5, 5, (2, 3), rng=rng),
        random_ranged(-5, 5, (2, 3), rng=rng),
    ),
    empty=(np.asarray([], dtype=config.floatX), np.asarray([], dtype=config.floatX)),
    int=(
        integers_ranged(-5, 5, (2, 3), rng=rng),
        integers_ranged(-5, 5, (2, 3), rng=rng),
    ),
    uint8=(
        integers_ranged(1, 6, (2, 3), rng=rng).astype("uint8"),
        integers_ranged(1, 6, (2, 3), rng=rng).astype("uint8"),
    ),
    uint16=(
        integers_ranged(1, 10, (2, 3), rng=rng).astype("uint16"),
        integers_ranged(1, 10, (2, 3), rng=rng).astype("uint16"),
    ),
    uint64=(
        integers_ranged(1, 10, (2, 3), rng=rng).astype("uint64"),
        integers_ranged(1, 10, (2, 3), rng=rng).astype("uint64"),
    ),
)

_grad_broadcast_binary_owenst = dict(
    normal=(
        random_ranged(-5, 5, (2, 3), rng=rng),
        random_ranged(-5, 5, (2, 3), rng=rng),
    )
)

TestOwensTBroadcast = makeBroadcastTester(
    op=pt.owens_t,
    expected=expected_owenst,
    good=_good_broadcast_binary_owenst,
    grad=_grad_broadcast_binary_owenst,
    eps=2e-10,
    mode=mode_no_scipy,
)
TestOwensTInplaceBroadcast = makeBroadcastTester(
    op=inplace.owens_t_inplace,
    expected=expected_owenst,
    good=_good_broadcast_binary_owenst,
    eps=2e-10,
    mode=mode_no_scipy,
    inplace=True,
)

rng = np.random.default_rng(seed=utt.fetch_seed())
_good_broadcast_unary_gammaln = dict(
    normal=(random_ranged(-1 + 1e-2, 10, (2, 3), rng=rng),),
    empty=(np.asarray([], dtype=config.floatX),),
    int=(integers_ranged(1, 10, (2, 3), rng=rng),),
    uint8=(integers_ranged(1, 6, (2, 3), rng=rng).astype("uint8"),),
    uint16=(integers_ranged(1, 10, (2, 3), rng=rng).astype("uint16"),),
    uint64=(integers_ranged(1, 10, (2, 3), rng=rng).astype("uint64"),),
)
_grad_broadcast_unary_gammaln = dict(
    # smaller range as our grad method does not estimate it well enough.
    normal=(random_ranged(1e-1, 8, (2, 3), rng=rng),),
)

TestGammaBroadcast = makeBroadcastTester(
    op=pt.gamma,
    expected=expected_gamma,
    good=_good_broadcast_unary_gammaln,
    grad=_grad_broadcast_unary_gammaln,
    mode=mode_no_scipy,
    eps=1e-5,
)
TestGammaInplaceBroadcast = makeBroadcastTester(
    op=inplace.gamma_inplace,
    expected=expected_gamma,
    good=_good_broadcast_unary_gammaln,
    mode=mode_no_scipy,
    eps=1e-5,
    inplace=True,
)

TestGammalnBroadcast = makeBroadcastTester(
    op=pt.gammaln,
    expected=expected_gammaln,
    good=_good_broadcast_unary_gammaln,
    grad=_grad_broadcast_unary_gammaln,
    eps=2e-10,
    mode=mode_no_scipy,
)
TestGammalnInplaceBroadcast = makeBroadcastTester(
    op=inplace.gammaln_inplace,
    expected=expected_gammaln,
    good=_good_broadcast_unary_gammaln,
    eps=2e-10,
    mode=mode_no_scipy,
    inplace=True,
)

rng = np.random.default_rng(seed=utt.fetch_seed())
_good_broadcast_unary_psi = dict(
    normal=(random_ranged(1, 10, (2, 3), rng=rng),),
    empty=(np.asarray([], dtype=config.floatX),),
    int=(integers_ranged(1, 10, (2, 3), rng=rng),),
    uint8=(integers_ranged(1, 10, (2, 3), rng=rng).astype("uint8"),),
    uint16=(integers_ranged(1, 10, (2, 3), rng=rng).astype("uint16"),),
)

TestPsiBroadcast = makeBroadcastTester(
    op=pt.psi,
    expected=expected_psi,
    good=_good_broadcast_unary_psi,
    eps=2e-10,
    mode=mode_no_scipy,
)
TestPsiInplaceBroadcast = makeBroadcastTester(
    op=inplace.psi_inplace,
    expected=expected_psi,
    good=_good_broadcast_unary_psi,
    eps=2e-10,
    mode=mode_no_scipy,
    inplace=True,
)

_good_broadcast_unary_tri_gamma = _good_broadcast_unary_psi

TestTriGammaBroadcast = makeBroadcastTester(
    op=pt.tri_gamma,
    expected=expected_tri_gamma,
    good=_good_broadcast_unary_psi,
    eps=2e-8,
    mode=mode_no_scipy,
)
TestTriGammaInplaceBroadcast = makeBroadcastTester(
    op=inplace.tri_gamma_inplace,
    expected=expected_tri_gamma,
    good=_good_broadcast_unary_tri_gamma,
    eps=2e-8,
    mode=mode_no_scipy,
    inplace=True,
)

TestChi2SFBroadcast = makeBroadcastTester(
    op=pt.chi2sf,
    expected=expected_chi2sf,
    good=_good_broadcast_unary_chi2sf,
    eps=2e-10,
    mode=mode_no_scipy,
    name="Chi2SF",
)

TestChi2SFInplaceBroadcast = makeBroadcastTester(
    op=inplace.chi2sf_inplace,
    expected=expected_chi2sf,
    good=_good_broadcast_unary_chi2sf,
    eps=2e-10,
    mode=mode_no_scipy,
    inplace=True,
    name="Chi2SF",
)

rng = np.random.default_rng(seed=utt.fetch_seed())
_good_broadcast_binary_gamma = dict(
    normal=(
        random_ranged(1e-2, 10, (2, 3), rng=rng),
        random_ranged(1e-2, 10, (2, 3), rng=rng),
    ),
    empty=(np.asarray([], dtype=config.floatX), np.asarray([], dtype=config.floatX)),
    int=(
        integers_ranged(1, 10, (2, 3), rng=rng),
        integers_ranged(1, 10, (2, 3), rng=rng),
    ),
    uint8=(
        integers_ranged(1, 6, (2, 3), rng=rng).astype("uint8"),
        integers_ranged(1, 6, (2, 3), rng=rng).astype("uint8"),
    ),
    uint16=(
        integers_ranged(1, 10, (2, 3), rng=rng).astype("uint16"),
        integers_ranged(1, 10, (2, 3), rng=rng).astype("uint16"),
    ),
    uint64=(
        integers_ranged(1, 10, (2, 3), rng=rng).astype("uint64"),
        integers_ranged(1, 10, (2, 3), rng=rng).astype("uint64"),
    ),
)

_good_broadcast_binary_gamma_grad = dict(
    normal=_good_broadcast_binary_gamma["normal"],
    specific_branches=(
        np.array([0.7, 11.0, 19.0]),
        np.array([16.0, 31.0, 3.0]),
    ),
)

TestGammaIncBroadcast = makeBroadcastTester(
    op=pt.gammainc,
    expected=expected_gammainc,
    good=_good_broadcast_binary_gamma,
    grad=_good_broadcast_binary_gamma_grad,
    eps=2e-8,
    mode=mode_no_scipy,
)

TestGammaIncInplaceBroadcast = makeBroadcastTester(
    op=inplace.gammainc_inplace,
    expected=expected_gammainc,
    good=_good_broadcast_binary_gamma,
    eps=2e-8,
    mode=mode_no_scipy,
    inplace=True,
)

TestGammaInccBroadcast = makeBroadcastTester(
    op=pt.gammaincc,
    expected=expected_gammaincc,
    good=_good_broadcast_binary_gamma,
    grad=_good_broadcast_binary_gamma_grad,
    eps=2e-8,
    mode=mode_no_scipy,
)

TestGammaInccInplaceBroadcast = makeBroadcastTester(
    op=inplace.gammaincc_inplace,
    expected=expected_gammaincc,
    good=_good_broadcast_binary_gamma,
    eps=2e-8,
    mode=mode_no_scipy,
    inplace=True,
)


def test_gammainc_ddk_tabulated_values():
    # This test replicates part of the old STAN test:
    # https://github.com/stan-dev/math/blob/21333bb70b669a1bd54d444ecbe1258078d33153/test/unit/math/prim/scal/fun/grad_reg_lower_inc_gamma_test.cpp
    k, x = pt.scalars("k", "x")
    gammainc_out = pt.gammainc(k, x)
    gammaincc_ddk = pt.grad(gammainc_out, k)
    f_grad = function([k, x], gammaincc_ddk)

    rtol = 1e-5 if config.floatX == "float64" else 1e-2
    atol = 1e-10 if config.floatX == "float64" else 1e-6

    for test_k, test_x, expected_ddk in (
        (0.0001, 0, 0),  # Limit condition
        (0.0001, 0.0001, -8.62594024578651),
        (0.0001, 6.2501, -0.0002705821702813008),
        (0.0001, 12.5001, -2.775406821933887e-7),
        (0.0001, 18.7501, -3.653379783274905e-10),
        (0.0001, 25.0001, -5.352425240798134e-13),
        (0.0001, 29.7501, -3.912723010174313e-15),
        (4.7501, 0.0001, 0),
        (4.7501, 6.2501, -0.1330287013623819),
        (4.7501, 12.5001, -0.004712176128251421),
        (4.7501, 18.7501, -0.00004898939126595217),
        (4.7501, 25.0001, -3.098781566343336e-7),
        (4.7501, 29.7501, -5.478399030091586e-9),
        (9.5001, 0.0001, -5.869126325643798e-15),
        (9.5001, 6.2501, -0.07717967485372858),
        (9.5001, 12.5001, -0.07661095137424883),
        (9.5001, 18.7501, -0.005594043337407605),
        (9.5001, 25.0001, -0.0001410123206233104),
        (9.5001, 29.7501, -5.75023943432906e-6),
        (14.2501, 0.0001, -7.24495484418588e-15),
        (14.2501, 6.2501, -0.003689474744087815),
        (14.2501, 12.5001, -0.1008796179460247),
        (14.2501, 18.7501, -0.05124664255610913),
        (14.2501, 25.0001, -0.005115177188580634),
        (14.2501, 29.7501, -0.0004793406401524598),
        (19.0001, 0.0001, -8.26027539153394e-15),
        (19.0001, 6.2501, -0.00003509660448733015),
        (19.0001, 12.5001, -0.02624562607393565),
        (19.0001, 18.7501, -0.0923829735092193),
        (19.0001, 25.0001, -0.03641281853907181),
        (19.0001, 29.7501, -0.007828749832965796),
    ):
        np.testing.assert_allclose(
            f_grad(test_k, test_x), expected_ddk, rtol=rtol, atol=atol
        )


def test_gammaincc_ddk_performance(benchmark):
    rng = np.random.default_rng(1)
    k = vector("k")
    x = vector("x")

    out = gammaincc(k, x)
    grad_fn = function([k, x], grad(out.sum(), wrt=[k]), mode="FAST_RUN")
    vals = [
        # Values that hit the second branch of the gradient
        np.full((1000,), 3.2),
        np.full((1000,), 0.01),
    ]

    verify_grad(gammaincc, vals, rng=rng)
    benchmark(grad_fn, *vals)


TestGammaUBroadcast = makeBroadcastTester(
    op=pt.gammau,
    expected=expected_gammau,
    good=_good_broadcast_binary_gamma,
    eps=2e-8,
    mode=mode_no_scipy,
)

TestGammaUInplaceBroadcast = makeBroadcastTester(
    op=inplace.gammau_inplace,
    expected=expected_gammau,
    good=_good_broadcast_binary_gamma,
    eps=2e-8,
    mode=mode_no_scipy,
    inplace=True,
)

TestGammaLBroadcast = makeBroadcastTester(
    op=pt.gammal,
    expected=expected_gammal,
    good=_good_broadcast_binary_gamma,
    eps=2e-8,
    mode=mode_no_scipy,
)

TestGammaLInplaceBroadcast = makeBroadcastTester(
    op=inplace.gammal_inplace,
    expected=expected_gammal,
    good=_good_broadcast_binary_gamma,
    eps=2e-8,
    mode=mode_no_scipy,
    inplace=True,
)

rng = np.random.default_rng(seed=utt.fetch_seed())
_good_broadcast_binary_gamma = dict(
    normal=(
        random_ranged(0, 100, (2, 3), rng=rng),
        random_ranged(0, 1, (2, 3), rng=rng),
    ),
    empty=(np.asarray([], dtype=config.floatX), np.asarray([], dtype=config.floatX)),
)

TestGammaIncInvBroadcast = makeBroadcastTester(
    op=pt.gammaincinv,
    expected=expected_gammaincinv,
    good=_good_broadcast_binary_gamma,
    eps=2e-8,
    mode=mode_no_scipy,
)

TestGammaIncInvInplaceBroadcast = makeBroadcastTester(
    op=inplace.gammaincinv_inplace,
    expected=expected_gammaincinv,
    good=_good_broadcast_binary_gamma,
    eps=2e-8,
    mode=mode_no_scipy,
    inplace=True,
)

TestGammaInccInvBroadcast = makeBroadcastTester(
    op=pt.gammainccinv,
    expected=expected_gammainccinv,
    good=_good_broadcast_binary_gamma,
    eps=2e-8,
    mode=mode_no_scipy,
)

TestGammaInccInvInplaceBroadcast = makeBroadcastTester(
    op=inplace.gammainccinv_inplace,
    expected=expected_gammainccinv,
    good=_good_broadcast_binary_gamma,
    eps=2e-8,
    mode=mode_no_scipy,
    inplace=True,
)

rng = np.random.default_rng(seed=utt.fetch_seed())
_good_broadcast_unary_bessel = dict(
    normal=(random_ranged(-10, 10, (2, 3), rng=rng),),
    empty=(np.asarray([], dtype=config.floatX),),
    int=(integers_ranged(-10, 10, (2, 3), rng=rng),),
    uint8=(integers_ranged(0, 10, (2, 3), rng=rng).astype("uint8"),),
    uint16=(integers_ranged(0, 10, (2, 3), rng=rng).astype("uint16"),),
)

_grad_broadcast_unary_bessel = dict(
    normal=(random_ranged(-10.0, 10.0, (2, 3)),),
)

_good_broadcast_binary_bessel = dict(
    normal=(
        random_ranged(-5, 5, (2, 3), rng=rng),
        random_ranged(0, 10, (2, 3), rng=rng),
    ),
    empty=(np.asarray([], dtype=config.floatX), np.asarray([], dtype=config.floatX)),
    integers=(
        integers_ranged(-5, 5, (2, 3), rng=rng),
        integers_ranged(-10, 10, (2, 3), rng=rng),
    ),
    uint8=(
        integers_ranged(0, 5, (2, 3), rng=rng).astype("uint8"),
        integers_ranged(0, 10, (2, 3), rng=rng).astype("uint8"),
    ),
    uint16=(
        integers_ranged(0, 5, (2, 3), rng=rng).astype("uint16"),
        integers_ranged(0, 10, (2, 3), rng=rng).astype("uint16"),
    ),
)

_grad_broadcast_binary_bessel = dict(
    normal=(random_ranged(1, 5, (2, 3), rng=rng), random_ranged(0, 10, (2, 3), rng=rng))
)

TestJ0Broadcast = makeBroadcastTester(
    op=pt.j0,
    expected=expected_j0,
    good=_good_broadcast_unary_bessel,
    grad=_grad_broadcast_unary_bessel,
    eps=2e-10,
    mode=mode_no_scipy,
)

TestJ0InplaceBroadcast = makeBroadcastTester(
    op=inplace.j0_inplace,
    expected=expected_j0,
    good=_good_broadcast_unary_bessel,
    eps=2e-10,
    mode=mode_no_scipy,
    inplace=True,
)

TestJ1Broadcast = makeBroadcastTester(
    op=pt.j1,
    expected=expected_j1,
    good=_good_broadcast_unary_bessel,
    grad=_grad_broadcast_unary_bessel,
    eps=2e-10,
    mode=mode_no_scipy,
)

TestJ1InplaceBroadcast = makeBroadcastTester(
    op=inplace.j1_inplace,
    expected=expected_j1,
    good=_good_broadcast_unary_bessel,
    eps=2e-10,
    mode=mode_no_scipy,
    inplace=True,
)

TestJvBroadcast = makeBroadcastTester(
    op=pt.jv,
    expected=expected_jv,
    good=_good_broadcast_binary_bessel,
    eps=2e-10,
    mode=mode_no_scipy,
)

TestJvInplaceBroadcast = makeBroadcastTester(
    op=inplace.jv_inplace,
    expected=expected_jv,
    good=_good_broadcast_binary_bessel,
    eps=2e-10,
    mode=mode_no_scipy,
    inplace=True,
)


def test_verify_jv_grad():
    # Verify Jv gradient.
    # Implemented separately due to need to fix first input for which grad is
    # not defined.
    v_val, x_val = _grad_broadcast_binary_bessel["normal"]

    def fixed_first_input_jv(x):
        return pt.jv(v_val, x)

    utt.verify_grad(fixed_first_input_jv, [x_val])


TestI0Broadcast = makeBroadcastTester(
    op=pt.i0,
    expected=expected_i0,
    good=_good_broadcast_unary_bessel,
    grad=_grad_broadcast_unary_bessel,
    eps=2e-10,
    mode=mode_no_scipy,
)

TestI0InplaceBroadcast = makeBroadcastTester(
    op=inplace.i0_inplace,
    expected=expected_i0,
    good=_good_broadcast_unary_bessel,
    eps=2e-10,
    mode=mode_no_scipy,
    inplace=True,
)

TestI1Broadcast = makeBroadcastTester(
    op=pt.i1,
    expected=expected_i1,
    good=_good_broadcast_unary_bessel,
    grad=_grad_broadcast_unary_bessel,
    eps=2e-10,
    mode=mode_no_scipy,
)

TestI1InplaceBroadcast = makeBroadcastTester(
    op=inplace.i1_inplace,
    expected=expected_i1,
    good=_good_broadcast_unary_bessel,
    eps=2e-10,
    mode=mode_no_scipy,
    inplace=True,
)

TestIvBroadcast = makeBroadcastTester(
    op=pt.iv,
    expected=expected_iv,
    good=_good_broadcast_binary_bessel,
    eps=2e-10,
    mode=mode_no_scipy,
)

TestIvInplaceBroadcast = makeBroadcastTester(
    op=inplace.iv_inplace,
    expected=expected_iv,
    good=_good_broadcast_binary_bessel,
    eps=2e-10,
    mode=mode_no_scipy,
    inplace=True,
)

TestIveBroadcast = makeBroadcastTester(
    op=pt.ive,
    expected=expected_ive,
    good=_good_broadcast_binary_bessel,
    eps=2e-10,
    mode=mode_no_scipy,
)

TestIveInplaceBroadcast = makeBroadcastTester(
    op=inplace.ive_inplace,
    expected=expected_ive,
    good=_good_broadcast_binary_bessel,
    eps=2e-10,
    mode=mode_no_scipy,
    inplace=True,
)


def test_verify_iv_grad():
    # Verify Iv gradient.
    # Implemented separately due to need to fix first input for which grad is
    # not defined.
    v_val, x_val = _grad_broadcast_binary_bessel["normal"]

    def fixed_first_input_iv(x):
        return pt.iv(v_val, x)

    utt.verify_grad(fixed_first_input_iv, [x_val])


def test_verify_ive_grad():
    # Verify Ive gradient.
    # Implemented separately due to need to fix first input for which grad is
    # not defined.
    v_val, x_val = _grad_broadcast_binary_bessel["normal"]

    def fixed_first_input_ive(x):
        return pt.ive(v_val, x)

    utt.verify_grad(fixed_first_input_ive, [x_val])


TestSigmoidBroadcast = makeBroadcastTester(
    op=pt.sigmoid,
    expected=expected_sigmoid,
    good=_good_broadcast_unary_normal_no_complex,
    eps=1e-8,
)

TestSigmoidInplaceBroadcast = makeBroadcastTester(
    op=inplace.sigmoid_inplace,
    expected=expected_sigmoid,
    good=_good_broadcast_unary_normal_no_complex,
    grad=_grad_broadcast_unary_normal,
    eps=1e-8,
    inplace=True,
)


class TestSigmoid:
    def test_elemwise(self):
        utt.verify_grad(pt.sigmoid, [np.random.random((3, 4))])


_good_broadcast_unary_softplus = dict(
    copymod(
        _good_broadcast_unary_normal_no_complex,
        without=["uint8", "uint16", "big_scalar"],
    ),  # numpy function overflows with uint16.
    uint8=[
        np.arange(0, 89, dtype="uint8")
    ],  # the range is different in new added uint8.
    int8=[np.arange(-127, 89, dtype="int8")],
)

expected_sofplus = upcast_int8_nfunc(
    lambda inputs: check_floatX(inputs, np.log1p(np.exp(inputs)))
)

TestSoftplusBroadcast = makeBroadcastTester(
    op=pt.softplus,
    expected=expected_sofplus,
    good=_good_broadcast_unary_softplus,
    eps=1e-8,
)

TestSoftplusInplaceBroadcast = makeBroadcastTester(
    op=inplace.softplus_inplace,
    expected=expected_sofplus,
    good=_good_broadcast_unary_softplus,
    grad=_grad_broadcast_unary_normal,
    eps=1e-8,
    inplace=True,
)


class TestSoftplus:
    def test_elemwise(self):
        utt.verify_grad(pt.softplus, [np.random.random((3, 4))])

    def test_accuracy(self):
        # Test all approximations are working (cutoff points are -37, 18, 33.3)
        x_test = np.array([-40.0, -17.5, 17.5, 18.5, 40.0])
        y_th = pt.softplus(x_test).eval()
        y_np = np.log1p(np.exp(x_test))
        np.testing.assert_allclose(y_th, y_np, rtol=10e-10)


rng = np.random.default_rng(seed=utt.fetch_seed())
_good_broadcast_unary_log1mexp = dict(
    normal=(random_ranged(-10.0, 0, (2, 3), rng=rng),),
    float32=(random_ranged(-10.0, 0, (2, 3), rng=rng).astype("float32"),),
    empty=(np.asarray([], dtype=config.floatX),),
    int=(integers_ranged(-10, -1, (2, 3), rng=rng),),
)

_grad_broadcast_unary_log1mexp = dict(
    normal=(random_ranged(-10.0, 0.0, (2, 3)),),
)


def expected_log1mexp(x):
    return check_floatX(x, np.log(-np.expm1(x)))


TestLog1mexpBroadcast = makeBroadcastTester(
    op=pt.log1mexp,
    expected=expected_log1mexp,
    good=_good_broadcast_unary_log1mexp,
    grad=_grad_broadcast_unary_log1mexp,
    eps=1e-8,
)

TestLog1mexpInplaceBroadcast = makeBroadcastTester(
    op=inplace.log1mexp_inplace,
    expected=expected_log1mexp,
    good=_good_broadcast_unary_log1mexp,
    eps=1e-8,
    inplace=True,
)

_good_broadcast_ternary_betainc = dict(
    normal=(
        random_ranged(0, 1000, (2, 3)),
        random_ranged(0, 1000, (2, 3)),
        random_ranged(0, 1, (2, 3)),
    ),
)

TestBetaincBroadcast = makeBroadcastTester(
    op=pt.betainc,
    expected=special.betainc,
    good=_good_broadcast_ternary_betainc,
    grad=_good_broadcast_ternary_betainc,
)

TestBetaincInplaceBroadcast = makeBroadcastTester(
    op=inplace.betainc_inplace,
    expected=special.betainc,
    good=_good_broadcast_ternary_betainc,
    grad=_good_broadcast_ternary_betainc,
    inplace=True,
)


class TestBetaIncGrad:
    def test_stan_grad_partial(self):
        # This test combines the following STAN tests:
        # https://github.com/stan-dev/math/blob/master/test/unit/math/prim/fun/inc_beta_dda_test.cpp
        # https://github.com/stan-dev/math/blob/master/test/unit/math/prim/fun/inc_beta_ddb_test.cpp
        # https://github.com/stan-dev/math/blob/master/test/unit/math/prim/fun/inc_beta_ddz_test.cpp
        a, b, z = pt.scalars("a", "b", "z")
        betainc_out = pt.betainc(a, b, z)
        betainc_grad = pt.grad(betainc_out, [a, b, z])
        f_grad = function([a, b, z], betainc_grad)

        decimal_precision = 7 if config.floatX == "float64" else 3

        for test_a, test_b, test_z, expected_dda, expected_ddb, expected_ddz in (
            (1.5, 1.25, 0.001, -0.00028665637, 4.41357328e-05, 0.063300692),
            (1.5, 1.25, 0.5, -0.26038693947, 0.29301795, 1.1905416),
            (1.5, 1.25, 0.6, -0.23806757, 0.32279575, 1.23341068),
            (1.5, 1.25, 0.999, -0.00022264493, 0.0018969609, 0.35587692),
            (15000, 1.25, 0.001, 0, 0, 0),
            (15000, 1.25, 0.5, 0, 0, 0),
            (15000, 1.25, 0.6, 0, 0, 0),
            (15000, 1.25, 0.999, -6.59543226e-10, 2.00849793e-06, 0.009898182),
            (1.5, 12500, 0.001, -3.93756641e-05, 1.47821755e-09, 0.1848717),
            (1.5, 12500, 0.5, 0, 0, 0),
            (1.5, 12500, 0.6, 0, 0, 0),
            (1.5, 12500, 0.999, 0, 0, 0),
            (15000, 12500, 0.001, 0, 0, 0),
            (15000, 12500, 0.5, -8.72102443e-53, 9.55282792e-53, 5.01131256e-48),
            (15000, 12500, 0.6, -4.085621e-14, -5.5067062e-14, 1.15135267e-71),
            (15000, 12500, 0.999, 0, 0, 0),
        ):
            np.testing.assert_almost_equal(
                f_grad(test_a, test_b, test_z),
                [expected_dda, expected_ddb, expected_ddz],
                decimal=decimal_precision,
            )

    def test_boik_robison_cox(self):
        # This test compares against the tabulated values in:
        # Boik, R. J., & Robison-Cox, J. F. (1998). Derivatives of the incomplete beta function.
        # Journal of Statistical Software, 3(1), 1-20.
        a, b, z = pt.scalars("a", "b", "z")
        betainc_out = pt.betainc(a, b, z)
        betainc_grad = pt.grad(betainc_out, [a, b])
        f_grad = function([a, b, z], betainc_grad)
        decimal = 7 if config.floatX == "float64" else 5
        for test_a, test_b, test_z, expected_dda, expected_ddb in (
            (1.5, 11.0, 0.001, -4.5720356e-03, 1.1845673e-04),
            (1.5, 11.0, 0.5, -2.5501997e-03, 9.0824388e-04),
            (1000.0, 1000.0, 0.5, -8.9224793e-03, 8.9224793e-03),
            (1000.0, 1000.0, 0.55, -3.6713108e-07, 4.0584118e-07),
        ):
            np.testing.assert_almost_equal(
                f_grad(test_a, test_b, test_z),
                [expected_dda, expected_ddb],
                decimal=decimal,
            )

    def test_beta_inc_stan_grad_combined(self):
        # This test replicates the following STAN test:
        # https://github.com/stan-dev/math/blob/master/test/unit/math/prim/fun/grad_reg_inc_beta_test.cpp
        a, b, z = pt.scalars("a", "b", "z")
        betainc_out = pt.betainc(a, b, z)
        betainc_grad = pt.grad(betainc_out, [a, b])
        f_grad = function([a, b, z], betainc_grad)

        for test_a, test_b, test_z, expected_dda, expected_ddb in (
            (1.0, 1.0, 1.0, 0, np.nan),
            (1.0, 1.0, 0.4, -0.36651629, 0.30649537),
        ):
            np.testing.assert_allclose(
                f_grad(test_a, test_b, test_z), [expected_dda, expected_ddb]
            )


_good_broadcast_ternary_betaincinv = dict(
    normal=(
        random_ranged(0, 1000, (2, 3)),
        random_ranged(0, 1000, (2, 3)),
        random_ranged(0, 1, (2, 3)),
    ),
)

TestBetaincinvBroadcast = makeBroadcastTester(
    op=pt.betaincinv,
    expected=special.betaincinv,
    good=_good_broadcast_ternary_betaincinv,
)

TestBetaincinvInplaceBroadcast = makeBroadcastTester(
    op=inplace.betaincinv_inplace,
    expected=special.betaincinv,
    good=_good_broadcast_ternary_betaincinv,
    inplace=True,
)

_good_broadcast_quaternary_hyp2f1 = dict(
    normal=(
        random_ranged(0, 20, (2, 3)),
        random_ranged(0, 20, (2, 3)),
        random_ranged(0, 20, (2, 3)),
        random_ranged(-0.9, 0.9, (2, 3)),
    ),
)

TestHyp2F1Broadcast = makeBroadcastTester(
    op=pt.hyp2f1,
    expected=expected_hyp2f1,
    good=_good_broadcast_quaternary_hyp2f1,
    grad=_good_broadcast_quaternary_hyp2f1,
)

TestHyp2F1InplaceBroadcast = makeBroadcastTester(
    op=inplace.hyp2f1_inplace,
    expected=expected_hyp2f1,
    good=_good_broadcast_quaternary_hyp2f1,
    inplace=True,
)


class TestHyp2F1Grad:
    few_iters_case = (
        2.0,
        1.0,
        2.0,
        0.4,
        0.4617734323582945,
        0.851376039609984,
        -0.4617734323582945,
        2.777777777777778,
    )

    many_iters_case = (
        3.70975,
        1.0,
        2.70975,
        0.999696,
        29369830.002773938200417693317785,
        36347869.41885337,
        -30843032.10697079073015067426929807,
        26278034019.28811,
    )

    def test_hyp2f1_grad_stan_cases(self):
        """This test reuses the same test cases as in:
        https://github.com/stan-dev/math/blob/master/test/unit/math/prim/fun/grad_2F1_test.cpp
        https://github.com/andrjohns/math/blob/develop/test/unit/math/prim/fun/hypergeometric_2F1_test.cpp

        Note: The expected_ddz was computed from the perform method, as it is not part of all Stan tests
        """
        a1, a2, b1, z = pt.scalars("a1", "a2", "b1", "z")
        hyp2f1_out = pt.hyp2f1(a1, a2, b1, z)
        hyp2f1_grad = pt.grad(hyp2f1_out, [a1, a2, b1, z])
        f_grad = function([a1, a2, b1, z], hyp2f1_grad)

        rtol = 1e-9 if config.floatX == "float64" else 2e-3
        for (
            test_a1,
            test_a2,
            test_b1,
            test_z,
            expected_dda1,
            expected_dda2,
            expected_ddb1,
            expected_ddz,
        ) in (
            (
                3.70975,
                1.0,
                2.70975,
                -0.2,
                -0.0488658806159776,
                -0.193844936204681,
                0.0677809985598383,
                0.8652952472723672,
            ),
            (3.70975, 1.0, 2.70975, 0, 0, 0, 0, 1.369037734108313),
            (
                1.0,
                1.0,
                1.0,
                0.6,
                2.290726829685388,
                2.290726829685388,
                -2.290726829685388,
                6.25,
            ),
            (
                1.0,
                31.0,
                41.0,
                1.0,
                6.825270649241036,
                0.4938271604938271,
                -0.382716049382716,
                17.22222222222223,
            ),
            (
                1.0,
                -2.1,
                41.0,
                1.0,
                -0.04921317604093563,
                0.02256814168279349,
                0.00118482743834665,
                -0.04854621426218426,
            ),
            (
                1.0,
                -0.5,
                10.6,
                0.3,
                -0.01443822031245647,
                0.02829710651967078,
                0.00136986255602642,
                -0.04846036062115473,
            ),
            (
                1.0,
                -0.5,
                10.0,
                0.3,
                -0.0153218866216130,
                0.02999436412836072,
                0.0015413242328729,
                -0.05144686244336445,
            ),
            (
                -0.5,
                -4.5,
                11.0,
                0.3,
                -0.1227022810085707,
                -0.01298849638043795,
                -0.0053540982315572,
                0.1959735211840362,
            ),
            (
                -0.5,
                -4.5,
                -3.2,
                0.9,
                0.85880025358111,
                0.4677704416159314,
                -4.19010422485256,
                -2.959196647856408,
            ),
            (
                3.70975,
                1.0,
                2.70975,
                -0.2,
                -0.0488658806159776,
                -0.193844936204681,
                0.0677809985598383,
                0.865295247272367,
            ),
            self.few_iters_case,
            self.many_iters_case,
            # Cases where series does not converge
            (1.0, 12.0, 10.0, 1.0, np.nan, np.nan, np.nan, np.inf),
            (1.0, 12.0, 20.0, 1.2, np.nan, np.nan, np.nan, np.inf),
            # Case where series converges under Euler transform (not implemented!)
            # (1.0, 1.0, 2.0, -5.0, -0.321040199556840, -0.321040199556840, 0.129536268190289, 0.0383370454357889),
            (1.0, 1.0, 2.0, -5.0, np.nan, np.nan, np.nan, 0.0383370454357889),
        ):
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                warnings.filterwarnings(
                    "ignore",
                    category=RuntimeWarning,
                    message="divide by zero encountered in log",
                )
                result = np.array(f_grad(test_a1, test_a2, test_b1, test_z))

            np.testing.assert_allclose(
                result,
                np.array([expected_dda1, expected_dda2, expected_ddb1, expected_ddz]),
                rtol=rtol,
            )

    @pytest.mark.parametrize("case", (few_iters_case, many_iters_case))
    @pytest.mark.parametrize("wrt", ("a", "all"))
    def test_benchmark(self, case, wrt, benchmark):
        a1, a2, b1, z = pt.scalars("a1", "a2", "b1", "z")
        hyp2f1_out = pt.hyp2f1(a1, a2, b1, z)
        hyp2f1_grad = pt.grad(hyp2f1_out, wrt=a1 if wrt == "a" else [a1, a2, b1, z])
        f_grad = function([a1, a2, b1, z], hyp2f1_grad)

        (test_a1, test_a2, test_b1, test_z, *expected_dds) = case

        result = benchmark(f_grad, test_a1, test_a2, test_b1, test_z)

        rtol = 1e-9 if config.floatX == "float64" else 2e-3
        expected_result = expected_dds[0] if wrt == "a" else np.array(expected_dds)
        np.testing.assert_allclose(
            result,
            expected_result,
            rtol=rtol,
        )

    @pytest.mark.parametrize("wrt", ([0], [1], [2], [0, 1], [1, 2], [0, 2], [0, 1, 2]))
    def test_unused_grad_loop_opt(self, wrt):
        """Test that we don't compute unnecessary outputs in the grad scalar loop"""
        (
            test_a1,
            test_a2,
            test_b1,
            test_z,
            *expected_dds,
            expected_ddz,
        ) = self.few_iters_case

        a1, a2, b1, z = pt.scalars("a1", "a2", "b1", "z")
        hyp2f1_out = pt.hyp2f1(a1, a2, b1, z)
        wrt_vars = [v for i, v in enumerate((a1, a2, b1, z)) if i in wrt]
        hyp2f1_grad = pt.grad(hyp2f1_out, wrt=wrt_vars)

        mode = get_default_mode().including("local_useless_2f1grad_loop")
        f_grad = function([a1, a2, b1, z], hyp2f1_grad, mode=mode)

        if len(wrt) == 3 and (config.mode == "FAST_COMPILE" or not config.cxx):
            # In this case we actually get two scalar_loops, because the merged one can't be executed in Python
            [scalar_loop_op1, scalar_loop_op2] = [
                node.op.scalar_op
                for node in f_grad.maker.fgraph.toposort()
                if isinstance(node.op, Elemwise)
                and isinstance(node.op.scalar_op, ScalarLoop)
            ]
            assert scalar_loop_op1.nin == 10 + 3 * 2  # wrt=[0, 1]
            assert scalar_loop_op2.nin == 10 + 3 * 1  # wrt=[2]
        else:
            [scalar_loop_op] = [
                node.op.scalar_op
                for node in f_grad.maker.fgraph.apply_nodes
                if isinstance(node.op, Elemwise)
                and isinstance(node.op.scalar_op, ScalarLoop)
            ]
            assert scalar_loop_op.nin == 10 + 3 * len(wrt)

        rtol = 1e-9 if config.floatX == "float64" else 2e-3
        np.testing.assert_allclose(
            f_grad(test_a1, test_a2, test_b1, test_z),
            [dd for i, dd in enumerate(expected_dds) if i in wrt],
            rtol=rtol,
        )
