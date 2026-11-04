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
    (pt.ndtri_exp, [np.array([-745.0, -100.0, -5.0, -0.5])]),
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
        "ndtri_exp",
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


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("count_dtype", ["int32", "int64", "uint64"])
@pytest.mark.parametrize("cls_name", ["LogGammaIncrement", "PsiIncrement"])
def test_gamma_increment_numba(dtype, count_dtype, cls_name):
    cls = getattr(psm, cls_name)
    x = pt.vector("x", dtype=dtype)
    n = pt.vector("n", dtype=count_dtype)
    out = Elemwise(cls(dtype, count_dtype))(x, n)
    values = np.array(
        [0.1, 1.0, 10.0, 100.0, 0.5, 2.3, -0.5, 0.0, np.inf, np.nan], dtype=dtype
    )
    counts = np.array([0, 1, 2, 3, 4, 9, 1, 1, 0, 0], dtype=count_dtype)
    fn = function([x, n], out, mode=numba_mode)
    reference = function([x, n], out, mode=py_mode)
    np.testing.assert_allclose(
        fn(values, counts),
        reference(values, counts),
        rtol=2e-5 if dtype == "float32" else 1e-10,
        atol=2e-5 if dtype == "float32" else 1e-10,
        equal_nan=True,
    )


def test_gamma_increment_large_base_numba():
    x, n = pt.dvector("x"), pt.lvector("n")
    out = Elemwise(psm.LogGammaIncrement(x.dtype, n.dtype))(x, n)
    fn = function([x, n], [out, pt.grad(out.sum(), x)], mode=numba_mode)
    actual, derivative = fn(np.array([1e12, 1e12]), np.array([0, 1], dtype="int64"))
    np.testing.assert_allclose(actual, [0.0, np.log(1e12)], rtol=1e-14)
    np.testing.assert_allclose(derivative, [0.0, 1e-12], rtol=1e-14)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("unary", [pt.gammaln, pt.psi])
def test_positive_gamma_psi(dtype, unary):
    x = pt.vector("x", dtype=dtype)
    fn = function([x], unary(x), mode=numba_mode)
    reference = scipy.special.gammaln if unary is pt.gammaln else scipy.special.psi
    low, high = (-35, 35) if dtype == "float32" else (-300, 300)
    values = list(np.logspace(low, high, 50))
    for boundary in (1.0, 2.0, 10.0, 13.0, 1000.0, 1e8, 1e17):
        point = np.asarray(boundary, dtype=dtype)
        values.extend(
            [
                np.nextafter(point, np.asarray(0, dtype=dtype)),
                point,
                np.nextafter(point, np.asarray(np.inf, dtype=dtype)),
            ]
        )
    values.extend(
        [
            -0.5,
            -1.0,
            0.0,
            -0.0,
            np.inf,
            -np.inf,
            np.nan,
            np.nextafter(np.asarray(0, dtype=dtype), np.asarray(1, dtype=dtype)),
        ]
    )
    values = np.asarray(values, dtype=dtype)
    expected = reference(values)
    if unary is pt.gammaln:
        # Preserve math.lgamma's convention outside the native positive kernel.
        expected[np.isneginf(values)] = np.inf
        subnormal = (values != 0) & (np.abs(values) < np.finfo(dtype).tiny)
        expected[subnormal] = -np.log(np.abs(values[subnormal]))
    np.testing.assert_allclose(
        fn(values),
        expected,
        rtol=2e-6 if dtype == "float32" else 3e-14,
        atol=2e-6 if dtype == "float32" else 3e-14,
        equal_nan=True,
    )


@pytest.mark.parametrize("fastmath", [False, True])
def test_float64_gamma_psi_accuracy(fastmath):
    x = pt.dvector("x")
    with config.change_flags(numba__fastmath=fastmath):
        fn = function([x], [pt.gammaln(x), pt.psi(x)], mode=numba_mode)

    # References computed with 100-digit mpmath precision.
    gamma_values = np.array(
        [
            np.nextafter(0.0, 1.0),
            np.nextafter(0.0, -1.0),
            np.nextafter(np.finfo("float64").tiny, 0.0),
            np.finfo("float64").tiny,
            np.nextafter(1.0, 0.0),
            np.nextafter(1.0, 2.0),
            np.nextafter(2.0, 1.0),
            np.nextafter(2.0, 3.0),
            np.nextafter(2.556348e305, np.inf),
        ]
    )
    expected_gamma = np.array(
        [
            744.4400719213812,
            744.4400719213812,
            708.3964185322641,
            708.3964185322641,
            6.408381213480008e-17,
            -1.2816762426960008e-16,
            -9.387698065543117e-17,
            1.8775396131086244e-16,
            1.7951366714594416e308,
        ]
    )
    actual_gamma, _ = fn(gamma_values)
    np.testing.assert_allclose(actual_gamma, expected_gamma, rtol=2e-15, atol=4e-16)

    psi_values = np.array(
        [
            1.4616321449683614,
            1.4616321449683618,
            1.461632144968362,
            1.4616321449683622,
            1.4616321449683625,
            1.4616321449683627,
            1.4616321449683631,
        ]
    )
    expected_psi = np.array(
        [
            -9.518782609666346e-16,
            -5.221454580919643e-16,
            -3.072790566546293e-16,
            -9.241265521729427e-17,
            1.2245374622004068e-16,
            3.3732014765737563e-16,
            7.670529505320453e-16,
        ]
    )
    _, actual_psi = fn(psi_values)
    np.testing.assert_allclose(actual_psi, expected_psi, rtol=2e-15, atol=1e-32)


def test_native_gamma_derivative_numba():
    x = pt.dvector("x")
    y = pt.gammaln(x)
    fn = function([x], [y, pt.grad(y.sum(), x)], mode=numba_mode)
    values = np.array([0.01, 0.5, 1.0, 1.4616321449683623, 2.0, 10.0, 13.0, 1000.0])
    _, gradient = fn(values)
    np.testing.assert_allclose(
        gradient, scipy.special.psi(values), rtol=1e-13, atol=1e-14
    )
