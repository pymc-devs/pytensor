import numpy as np
import pytest
import scipy

import pytensor.scalar as ps
import pytensor.scalar.basic as psb
import pytensor.scalar.math as psm
import pytensor.tensor as pt
from pytensor import config, function
from pytensor.graph import Apply
from pytensor.scalar import ScalarLoop, UnaryScalarOp
from pytensor.scalar.basic import Composite
from pytensor.tensor import tensor
from pytensor.tensor.elemwise import Elemwise
from tests.link.numba.test_basic import compare_numba_and_py, numba_mode, py_mode


rng = np.random.default_rng(42849)


@pytest.mark.parametrize(
    "x, y",
    [
        (
            (pt.lvector(), np.arange(4, dtype="int64")),
            (pt.dvector(), np.arange(4, dtype="float64")),
        ),
        (
            (pt.dmatrix(), np.arange(4, dtype="float64").reshape((2, 2))),
            (pt.lscalar(), np.array(4, dtype="int64")),
        ),
    ],
)
def test_Second(x, y):
    x, x_test = x
    y, y_test = y
    # We use the `Elemwise`-wrapped version of `Second`
    g = pt.second(x, y)
    compare_numba_and_py(
        [x, y],
        g,
        [x_test, y_test],
    )


@pytest.mark.parametrize(
    "v, min, max",
    [
        ((pt.scalar(), np.array(10, dtype=config.floatX)), 3.0, 7.0),
        ((pt.scalar(), np.array(1, dtype=config.floatX)), 3.0, 7.0),
        ((pt.scalar(), np.array(10, dtype=config.floatX)), 7.0, 3.0),
    ],
)
def test_Clip(v, min, max):
    v, v_test = v
    g = ps.clip(v, min, max)

    compare_numba_and_py(
        [v],
        [g],
        [v_test],
    )


@pytest.mark.parametrize(
    "inputs, input_values, scalar_fn",
    [
        (
            [pt.scalar("x"), pt.scalar("y"), pt.scalar("z")],
            [
                np.array(10, dtype=config.floatX),
                np.array(20, dtype=config.floatX),
                np.array(30, dtype=config.floatX),
            ],
            lambda x, y, z: ps.add(x, y, z),
        ),
        (
            [pt.scalar("x"), pt.scalar("y"), pt.scalar("z")],
            [
                np.array(10, dtype=config.floatX),
                np.array(20, dtype=config.floatX),
                np.array(30, dtype=config.floatX),
            ],
            lambda x, y, z: ps.mul(x, y, z),
        ),
        (
            [pt.scalar("x"), pt.scalar("y")],
            [
                np.array(10, dtype=config.floatX),
                np.array(20, dtype=config.floatX),
            ],
            lambda x, y: x + y * 2 + ps.exp(x - y),
        ),
    ],
)
def test_Composite(inputs, input_values, scalar_fn):
    composite_inputs = [ps.ScalarType(config.floatX)(name=i.name) for i in inputs]
    comp_op = Elemwise(Composite(composite_inputs, [scalar_fn(*composite_inputs)]))
    compare_numba_and_py(inputs, [comp_op(*inputs)], input_values)


@pytest.mark.parametrize(
    "v, dtype",
    [
        ((pt.fscalar(), np.array(1.0, dtype="float32")), psb.float64),
        pytest.param(
            (pt.dscalar(), np.array(1.0, dtype="float64")),
            psb.float32,
            marks=pytest.mark.xfail(reason="Scalar downcasting not supported in numba"),
        ),
    ],
)
def test_Cast(v, dtype):
    v, v_test = v
    g = psb.Cast(dtype)(v)
    compare_numba_and_py(
        [v],
        [g],
        [v_test],
    )


@pytest.mark.parametrize(
    "v, dtype",
    [
        ((pt.iscalar(), np.array(10, dtype="int32")), psb.float64),
    ],
)
def test_reciprocal(v, dtype):
    v, v_test = v
    g = psb.reciprocal(v)
    compare_numba_and_py(
        [v],
        [g],
        [v_test],
    )


@pytest.mark.parametrize("composite", (False, True))
def test_isnan(composite):
    # Testing with tensor just to make sure Elemwise does not revert the scalar behavior of fastmath
    x = tensor(shape=(2,), dtype="float64")

    if composite:
        x_scalar = psb.float64()
        scalar_out = ~psb.isnan(x_scalar)
        out = Elemwise(Composite([x_scalar], [scalar_out]))(x)
    else:
        out = pt.isnan(x)

    compare_numba_and_py(
        [x],
        [out],
        [np.array([1, 0], dtype="float64")],
    )


@pytest.mark.parametrize(
    "dtype",
    [
        "float32",
        "float64",
        "int16",
        "int64",
        "uint32",
    ],
)
def test_Softplus(dtype):
    x = ps.get_scalar_type(dtype)("x")
    g = psm.softplus(x)

    py_fn = function([x], g, mode=py_mode)
    numba_fn = function([x], g, mode=numba_mode)
    for value in (-40, -32, 0, 32, 40):
        if value < 0 and dtype.startswith("u"):
            continue
        test_x = np.dtype(dtype).type(value)
        np.testing.assert_allclose(
            py_fn(test_x),
            getattr(np, g.dtype)(numba_fn(test_x)),
            strict=True,
            err_msg=f"Failed for value {value}",
        )


@pytest.mark.parametrize(
    "test_base",
    [np.bool(True), np.int16(3), np.uint16(3), np.float32(0.5), np.float64(0.5)],
)
@pytest.mark.parametrize(
    "test_exponent",
    [np.bool(True), np.int16(2), np.uint16(2), np.float32(2.0), np.float64(2.0)],
)
def test_power_fastmath_bug(test_base, test_exponent):
    # Test we don't fail to compile power with discrete exponents due to https://github.com/numba/numba/issues/9554
    base = pt.scalar("base", dtype=test_base.dtype)
    exponent = pt.scalar("exponent", dtype=test_exponent.dtype)
    out = pt.power(base, exponent)
    compare_numba_and_py(
        [base, exponent],
        [out],
        [test_base, test_exponent],
    )


def test_cython_obj_mode_fallback():
    """Test that unsupported cython signatures fallback to obj-mode"""

    # Create a ScalarOp with a non-standard dtype
    class IntegerGamma(UnaryScalarOp):
        # We'll try to check for scipy cython impl
        nfunc_spec = ("scipy.special.gamma", 1, 1)

        def make_node(self, x):
            x = psb.as_scalar(x)
            assert x.dtype == "int64"
            out = x.type()
            return Apply(self, [x], [out])

        def impl(self, x):
            return scipy.special.gamma(x).astype("int64")

    x = pt.scalar("x", dtype="int64")
    g = Elemwise(IntegerGamma())(x)
    assert g.type.dtype == "int64"

    with pytest.warns(UserWarning, match="Numba will use object mode"):
        compare_numba_and_py(
            [x],
            [g],
            [np.array(5, dtype="int64")],
        )


def test_erf_complex():
    x = pt.scalar("x", dtype="complex128")
    g = pt.erf(x)

    compare_numba_and_py(
        [x],
        [g],
        [np.array(0.5 + 1j, dtype="complex128")],
    )


CYTHON_SPECIAL_CASES = [
    (pt.erfcx, [np.array([-1.0, 0.0, 1.0, 3.0])]),
    (pt.erfinv, [np.array([-0.5, 0.0, 0.3, 0.9])]),
    (pt.erfcinv, [np.array([0.2, 0.7, 1.0, 1.5])]),
    (pt.psi, [np.array([0.5, 1.0, 3.7, 12.3])]),
    (pt.gamma, [np.array([0.5, 1.0, 3.7, 5.2])]),
    (pt.j0, [np.array([0.1, 1.0, 5.0, 9.0])]),
    (pt.j1, [np.array([0.1, 1.0, 5.0, 9.0])]),
    (pt.i0, [np.array([0.1, 1.0, 3.0, 5.0])]),
    (pt.i1, [np.array([0.1, 1.0, 3.0, 5.0])]),
    (pt.owens_t, [np.array([0.5, 1.0, 2.0]), np.array([0.3, 0.7, 1.2])]),
    (pt.gammainc, [np.array([1.0, 2.0, 3.0]), np.array([0.5, 2.0, 4.0])]),
    (pt.gammaincc, [np.array([1.0, 2.0, 3.0]), np.array([0.5, 2.0, 4.0])]),
    (pt.gammaincinv, [np.array([1.0, 2.0, 3.0]), np.array([0.2, 0.5, 0.8])]),
    (pt.gammainccinv, [np.array([1.0, 2.0, 3.0]), np.array([0.2, 0.5, 0.8])]),
    (pt.jv, [np.array([0.0, 1.0, 2.0]), np.array([1.0, 3.0, 5.0])]),
    (pt.ive, [np.array([0.0, 1.0, 2.0]), np.array([1.0, 3.0, 5.0])]),
    (pt.kve, [np.array([0.0, 1.0, 2.0]), np.array([1.0, 3.0, 5.0])]),
    (pt.betainc, [np.array([1.0, 2.0]), np.array([2.0, 3.0]), np.array([0.3, 0.6])]),
    (pt.betaincinv, [np.array([1.0, 2.0]), np.array([2.0, 3.0]), np.array([0.3, 0.6])]),
    (
        pt.hyp2f1,
        [
            np.array([0.5, 1.0]),
            np.array([0.5, 1.0]),
            np.array([1.5, 2.0]),
            np.array([0.2, 0.4]),
        ],
    ),
]


@pytest.mark.parametrize(
    "op_fn, test_values",
    CYTHON_SPECIAL_CASES,
    ids=[
        "erfcx",
        "erfinv",
        "erfcinv",
        "psi",
        "gamma",
        "j0",
        "j1",
        "i0",
        "i1",
        "owens_t",
        "gammainc",
        "gammaincc",
        "gammaincinv",
        "gammainccinv",
        "jv",
        "ive",
        "kve",
        "betainc",
        "betaincinv",
        "hyp2f1",
    ],
)
def test_cython_special_funcs(op_fn, test_values):
    """Scalar ops backed by ``scipy.special.cython_special`` resolve their C function pointer at
    runtime (see ``numba_funcify_ScalarOp``), so the funcified kernel is njit-only. Skip the
    object-mode eval path, as is done for the LAPACK wrappers."""
    inputs = [pt.vector(f"x{i}", dtype="float64") for i in range(len(test_values))]
    out = op_fn(*inputs)
    compare_numba_and_py(inputs, [out], test_values, eval_obj_mode=False)


class TestScalarLoop:
    def test_scalar_for_loop_single_out(self):
        n_steps = ps.int64("n_steps")
        x0 = ps.float64("x0")
        const = ps.float64("const")
        x = x0 + const

        op = ScalarLoop(init=[x0], constant=[const], update=[x])
        x = op(n_steps, x0, const)

        fn = function([n_steps, x0, const], [x], mode=numba_mode)

        res_x = fn(n_steps=5, x0=0, const=1)
        np.testing.assert_allclose(res_x, 5)

        res_x = fn(n_steps=5, x0=0, const=2)
        np.testing.assert_allclose(res_x, 10)

        res_x = fn(n_steps=4, x0=3, const=-1)
        np.testing.assert_allclose(res_x, -1)

    def test_scalar_for_loop_multiple_outs(self):
        n_steps = ps.int64("n_steps")
        x0 = ps.float64("x0")
        y0 = ps.int64("y0")
        const = ps.float64("const")
        x = x0 + const
        y = y0 + 1

        op = ScalarLoop(init=[x0, y0], constant=[const], update=[x, y])
        x, y = op(n_steps, x0, y0, const)

        fn = function([n_steps, x0, y0, const], [x, y], mode=numba_mode)

        res_x, res_y = fn(n_steps=5, x0=0, y0=0, const=1)
        np.testing.assert_allclose(res_x, 5)
        np.testing.assert_allclose(res_y, 5)

        res_x, res_y = fn(n_steps=5, x0=0, y0=0, const=2)
        np.testing.assert_allclose(res_x, 10)
        np.testing.assert_allclose(res_y, 5)

        res_x, res_y = fn(n_steps=4, x0=3, y0=2, const=-1)
        np.testing.assert_allclose(res_x, -1)
        np.testing.assert_allclose(res_y, 6)

    def test_scalar_while_loop(self):
        n_steps = ps.int64("n_steps")
        x0 = ps.float64("x0")
        x = x0 + 1
        until = x >= 10

        op = ScalarLoop(init=[x0], update=[x], until=until)
        fn = function([n_steps, x0], op(n_steps, x0), mode=numba_mode)
        np.testing.assert_allclose(fn(n_steps=20, x0=0), [10, True])
        np.testing.assert_allclose(fn(n_steps=20, x0=1), [10, True])
        np.testing.assert_allclose(fn(n_steps=5, x0=1), [6, False])
        np.testing.assert_allclose(fn(n_steps=0, x0=1), [1, False])

    def test_loop_with_cython_wrapped_op(self):
        x = ps.float64("x")
        op = ScalarLoop(init=[x], update=[ps.psi(x)])
        out = op(1, x)

        fn = function([x], out, mode=numba_mode)
        x_test = np.float64(0.5)
        res = fn(x_test)
        expected_res = ps.psi(x).eval({x: x_test})
        np.testing.assert_allclose(res, expected_res)


def _max_ulp_err(got, ref64, dtype):
    """Max error in ulps of `dtype`, over finite points, vs a float64 reference.

    Uses the true spacing at each value, not a relative-error proxy. `np.spacing` is
    signed (negative for negative args), so we take abs -- otherwise a `> 0` mask would
    silently drop every point on the negative branch, which is the whole reason log1p /
    expm1 exist. A float64 reference resolves float32 accuracy exactly; for float64 it
    measures agreement with numpy's own (well-tested) log1p / expm1.
    """
    got = np.asarray(got).astype("float64")
    target = np.asarray(ref64).astype(
        dtype
    )  # reference correctly rounded to target dtype
    sp = np.abs(np.spacing(target)).astype("float64")
    target = target.astype("float64")
    finite = np.isfinite(got) & np.isfinite(target) & (sp > 0)
    return (np.abs(got[finite] - target[finite]) / sp[finite]).max()


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_vectorizable_log1p(dtype):
    """log1p lowered via a corrected log(1 + x) stays accurate over a wide range.

    The naive log(1 + x) collapses near 0 (the regime log1p exists for), so we sweep
    densely and deep into the near-zero cancellation region in BOTH signs -- a narrow or
    positive-only grid never stresses the correction and a broken one slips through. We
    also check the domain edge: log1p(-1) = -inf, log1p(x < -1) = nan. The corrected form
    is emitted under a vector library (both dtypes) or on float32 under numba__fastmath, so
    we wire numba__veclib to exercise the corrected lowering for both dtypes; we do not wire
    an actual library, since this checks the corrected form's accuracy, not its SIMD lowering.
    """
    with config.change_flags(numba__veclib="libmvec"):
        x = pt.vector("x", dtype=dtype)
        fn = function([x], pt.log1p(x), mode=numba_mode)

    neg = -np.logspace(-20, np.log10(0.999), 4000)  # (-0.999, 0), deep near-zero
    pos = np.logspace(-20, 2.5, 4000)  # (0, ~316)
    edge = np.array([-1.0, -1.5, -10.0])  # -inf, nan, nan
    x_test = np.concatenate([neg, pos, edge]).astype(dtype)

    got = fn(x_test)
    with np.errstate(invalid="ignore", divide="ignore"):  # x <= -1 is intentional
        ref = np.log1p(x_test.astype("float64"))

    assert got.dtype == np.dtype(
        dtype
    )  # output dtype; does NOT reveal an internal upcast
    np.testing.assert_array_equal(np.isnan(got), np.isnan(ref))  # nan for x < -1
    np.testing.assert_array_equal(np.isinf(got), np.isinf(ref))  # -inf at x == -1
    assert _max_ulp_err(got, ref, dtype) < 4  # ~2 ulp worst case observed


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_vectorizable_expm1(dtype):
    """expm1 lowered via a polynomial near 0 / exp(x) - 1 elsewhere stays accurate.

    The naive exp(x) - 1 collapses near 0, so we sweep densely deep near-zero in both
    signs. The polynomial branch (|x| < ln2) carries no exp, so its accuracy is
    independent of whatever vector exp is wired in -- we bound it tightly there, the
    branch the rewrite exists for. Across the full domain (incl. the exp(x) - 1 branch,
    which only inherits exp's own accuracy) we keep a looser bound, plus the overflow edge
    where it must go to +inf exactly where numpy does. The polynomial is emitted under a
    vector library (both dtypes) or on float32 under numba__fastmath, so we wire numba__veclib
    to exercise the polynomial for both dtypes; we do not actually wire a library in here, since
    this checks the polynomial's accuracy, not its SIMD lowering.
    """
    with config.change_flags(numba__veclib="libmvec"):
        x = pt.vector("x", dtype=dtype)
        fn = function([x], pt.expm1(x), mode=numba_mode)

    ln2 = np.log(2.0)
    small = np.logspace(-20, np.log10(ln2), 4000)  # |x| in (0, ln2): polynomial branch
    mid = np.logspace(np.log10(ln2), 1, 2000)  # |x| in (ln2, 10): exp(x) - 1 branch
    overflow = np.array([1e3, 1e5])  # exp overflows -> +inf in both float32 and float64
    x_test = np.concatenate([-small, small, -mid, mid, overflow]).astype(dtype)

    got = fn(x_test)
    with np.errstate(over="ignore"):  # the overflow points are intentional
        ref = np.expm1(x_test.astype("float64"))
        ref_same_dtype = np.expm1(x_test)

    assert got.dtype == np.dtype(
        dtype
    )  # output dtype; does NOT reveal an internal upcast
    np.testing.assert_array_equal(np.isinf(got), np.isinf(ref_same_dtype))  # +inf
    poly = np.abs(x_test) < ln2
    assert _max_ulp_err(got[poly], ref[poly], dtype) < 4  # cancellation region: ~1 ulp
    assert (
        _max_ulp_err(got, ref, dtype) < 16
    )  # full domain incl. vector exp's own error


@pytest.mark.parametrize("dtype", ["float64", "float32"])
@pytest.mark.parametrize(
    "op, lo, hi",
    [
        (pt.log1p, -0.5, 5.0),
        (pt.expm1, -5.0, 5.0),
        (pt.log1mexp, -5.0, -0.01),
        (pt.softplus, -30.0, 30.0),
    ],
    ids=["log1p", "expm1", "log1mexp", "softplus"],
)
def test_vectorizable_op_benchmark(op, lo, hi, dtype, benchmark):
    """Throughput of the SIMD/cache-friendly log1p / expm1 / log1mexp / softplus lowerings.

    Runs under the default config (numba__fastmath on, no vector library), so it is
    reproducible anywhere; comparing this branch against main (pytest-benchmark tracks results
    across runs) shows the PR's speedup. It is largest for float32. The precision-for-speed
    rewrites active here are the float32 log1p/expm1 polynomials and softplus (both dtypes),
    gated on fastmath; float64 log1p/expm1 and sigmoid (both dtypes) keep main's form unless a
    vector library is wired, so they show ~parity here. A wired library widens every win
    further -- not exercised here.
    """
    x = pt.vector("x", dtype=dtype)
    fn = function([x], op(x), mode=numba_mode)
    fn.trust_input = True
    x_test = rng.uniform(lo, hi, 100_000).astype(dtype)
    fn(x_test)  # compile before timing
    benchmark(fn, x_test)
