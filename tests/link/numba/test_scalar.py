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


def test_scalar_dispatch_output_dtype():
    # Widen after each Op so its rounding and overflow remain observable.
    x, y = ps.int8(), ps.int8()
    compare_numba_and_py(
        [x, y],
        (x * y).astype("float64"),
        [np.int8(100), np.int8(2)],
        assert_fn=np.testing.assert_equal,
    )

    x = ps.float32()
    compare_numba_and_py(
        [x],
        psm.sigmoid(x).astype("float64"),
        [np.float32(0.5)],
        assert_fn=np.testing.assert_equal,
    )


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


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_float_floor_div(dtype):
    # `floor(x / y)` is not the floor division of the exact operands, which NumPy does
    x = pt.vector("x", dtype=dtype)
    y = pt.vector("y", dtype=dtype)
    x_test = np.array(
        [3.0, 7.5, -7.5, -3.0, -7.5, 0.5, 7.0, 1.0, -1.0, 0.0], dtype=dtype
    )
    y_test = np.array(
        [-0.3, -0.3, 0.3, 0.3, np.inf, -np.inf, 2.0, 0.0, 0.0, 0.0], dtype=dtype
    )
    compare_numba_and_py([x, y], [pt.floor_div(x, y)], [x_test, y_test])


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


def test_gammaln_psi():
    # mpmath at 40 digits; next to the zeros of both too
    x = pt.vector("x", dtype="float64")
    fn = function([x], [pt.gammaln(x), pt.psi(x)], mode=numba_mode)
    xs = [1e-300, 1e-10, 0.3, 1 - 2**-53, 1.0, 1 + 2**-52, 1.4616321449683622]
    xs += [1.75, 2.0, 2.5, 3.999, 17.5, 63.99, 64.0, 1e5, 1e300]
    gammaln_ref = [
        690.7755278982137,
        23.025850929882736,
        1.0957979948180756,
        6.408381213480008e-17,
        0.0,
        -1.2816762426960008e-16,
        -0.12148629053584961,
        -0.08440112102048555,
        0.0,
        0.2846828704729192,
        1.790503493484443,
        32.08111489594735,
        200.96780668432012,
        201.00931639928152,
        1051287.7089736569,
        6.897755278982137e302,
    ]
    psi_ref = [
        -9.999999999999999e299,
        -10000000000.577215,
        -3.502524222200133,
        -0.5772156649015331,
        -0.5772156649015329,
        -0.5772156649015325,
        -9.241265521729427e-17,
        0.24747245354686118,
        0.42278433509846713,
        0.7031566406452432,
        1.2558338054487181,
        2.8333574322286843,
        4.1508927493430345,
        4.1510502388042365,
        11.512920464961896,
        690.7755278982137,
    ]
    gammaln, psi = fn(np.array(xs))
    np.testing.assert_array_max_ulp(gammaln, gammaln_ref, maxulp=2)
    np.testing.assert_array_max_ulp(psi, psi_ref, maxulp=3)
    # libm's lgamma and scipy's psi below 0, as before
    edges = np.array([-2.5, -1.0, -0.0, 0.0, np.inf, np.nan])
    gammaln, psi = fn(edges)
    np.testing.assert_allclose(gammaln, [-0.056243716497674054, *[np.inf] * 4, np.nan])
    np.testing.assert_equal(psi, scipy.special.psi(edges))


def test_gammaln_psi_float32():
    x = pt.vector("x", dtype="float32")
    fn = function([x], [pt.gammaln(x), pt.psi(x)], mode=numba_mode)
    xs = [1e-30, 0.3, 1 - 2**-24, 1.0, 1 + 2**-23, 1.4616321, 1.75, 2.0, 2.5]
    xs = np.array([*xs, 3.999, 17.5, 63.99, 64.0, 1e5, 1e30], dtype="float32")
    gammaln, psi = fn(xs)
    assert gammaln.dtype == psi.dtype == "float32"
    # against the float64 kernels, which are within 2 and 3 ulp of mpmath
    x64 = pt.vector("x64", dtype="float64")
    gammaln_ref, psi_ref = function(
        [x64], [pt.gammaln(x64), pt.psi(x64)], mode=numba_mode
    )(xs.astype("float64"))
    np.testing.assert_array_max_ulp(gammaln, gammaln_ref.astype("float32"), maxulp=3)
    np.testing.assert_array_max_ulp(psi, psi_ref.astype("float32"), maxulp=3)
    edges = np.array([-2.5, -1.0, -0.0, 0.0, np.inf, np.nan], dtype="float32")
    gammaln, psi = fn(edges)
    np.testing.assert_allclose(
        gammaln, [-0.056243716497674054, *[np.inf] * 4, np.nan], rtol=1e-6
    )
    np.testing.assert_allclose(psi, scipy.special.psi(edges), rtol=1e-6)


CYTHON_SPECIAL_CASES = [
    (pt.erfcx, [np.array([-1.0, 0.0, 1.0, 3.0])]),
    (pt.erfinv, [np.array([-0.5, 0.0, 0.3, 0.9])]),
    (pt.erfcinv, [np.array([0.2, 0.7, 1.0, 1.5])]),
    (pt.ndtri_exp, [np.array([-745.0, -100.0, -5.0, -0.5])]),
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
        "ndtri_exp",
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
