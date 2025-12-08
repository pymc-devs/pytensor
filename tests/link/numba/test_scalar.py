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
