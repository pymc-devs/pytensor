import numpy as np
import pytest

import pytensor
import pytensor.tensor as pt
from pytensor.compile.mode import Mode
from pytensor.graph.fg import FunctionGraph
from pytensor.link.c.basic import DualLinker
from pytensor.link.numba import NumbaLinker
from pytensor.scalar.basic import (
    EQ,
    ComplexError,
    Composite,
    IntDiv,
    ScalarType,
    TrueDiv,
    add,
    and_,
    arccos,
    arccosh,
    arcsin,
    arcsinh,
    arctan,
    arctan2,
    arctanh,
    complex64,
    constant,
    cos,
    cosh,
    deg2rad,
    eq,
    exp,
    exp2,
    expm1,
    float32,
    floats,
    int8,
    int32,
    int64,
    ints,
    invert,
    log,
    log1p,
    log2,
    log10,
    mul,
    neg,
    neq,
    rad2deg,
    reciprocal,
    sin,
    sinh,
    sqrt,
    tan,
    tanh,
    true_div,
)
from pytensor.tensor.type import fscalar, imatrix, matrix
from tests.link.test_link import make_function


def test_mul_add_true():
    x, y, _z = floats("xyz")
    e = mul(add(x, y), true_div(x, y))
    g = FunctionGraph([x, y], [e])
    fn = make_function(DualLinker().accept(g))
    assert fn(1.0, 2.0) == 1.5


class TestComposite:
    def test_straightforward(self):
        x, y, _z = floats("xyz")
        e = mul(add(x, y), true_div(x, y))
        C = Composite([x, y], [e])
        c = C.make_node(x, y)
        # print c.c_code(['x', 'y'], ['z'], dict(id = 0))
        g = FunctionGraph([x, y], [c.out])
        fn = make_function(DualLinker().accept(g))
        assert fn(1.0, 2.0) == 1.5

    def test_flatten(self):
        # Test that we flatten multiple Composite.
        x, y, z = floats("xyz")
        C = Composite([x, y], [x + y])
        CC = Composite([x, y], [C(x * y, y)])
        assert not isinstance(CC.outputs[0].owner.op, Composite)

        # Test with multiple outputs
        CC = Composite([x, y, z], [C(x * y, y), C(x * z, y)])
        # We don't flatten that case.
        assert isinstance(CC.outputs[0].owner.op, Composite)

    @pytest.mark.parametrize("literal_value", (70.0, -np.inf, np.float32("nan")))
    def test_with_constants(self, literal_value):
        x, y, _z = floats("xyz")
        e = mul(add(literal_value, y), true_div(x, y))
        comp_op = Composite([x, y], [e])
        comp_node = comp_op.make_node(x, y)

        c_code = comp_node.op.c_code(comp_node, "dummy", ["x", "y"], ["z"], dict(id=0))
        assert constant(literal_value).type.c_literal(literal_value) in c_code

        # Make sure caching of the c_code template works
        assert hasattr(comp_node.op, "_c_code")

        g = FunctionGraph([x, y], [comp_node.out])

        # Default checker does not allow `nan`
        def checker(x, y):
            np.testing.assert_equal(x, y)

        fn = make_function(DualLinker(checker=checker).accept(g))

        test_x = 1.0
        test_y = 2.0
        np.testing.assert_equal(
            fn(test_x, test_y),
            (literal_value + test_y) * (test_x / test_y),
        )

    def test_negative_constant(self):
        # Test that a negative constant is wrapped in parentheses to avoid confusing - (unary minus) and -- (decrement)
        x = int64("x")
        e = neg(constant(-1.5)) % x
        comp_op = Composite([x], [e])
        comp_node = comp_op.make_node(x)

        c_code = comp_node.op.c_code(comp_node, "dummy", ["x", "y"], ["z"], dict(id=0))
        assert "-1.5" in c_code

        g = FunctionGraph([x], [comp_node.out])
        fn = make_function(DualLinker().accept(g))
        assert fn(2) == 1.5
        assert fn(1) == 0.5

    def test_many_outputs(self):
        x, y, z = floats("xyz")
        e0 = x + y + z
        e1 = x + y * z
        e2 = x / y
        C = Composite([x, y, z], [e0, e1, e2])
        c = C.make_node(x, y, z)
        # print c.c_code(['x', 'y', 'z'], ['out0', 'out1', 'out2'], dict(id = 0))
        g = FunctionGraph([x, y, z], c.outputs)
        fn = make_function(DualLinker().accept(g))
        assert fn(1.0, 2.0, 3.0) == [6.0, 7.0, 0.5]

    def test_identical_outputs(self):
        x, y, z = floats("xyz")
        e0 = x + y + z
        e1 = x + y + z
        e2 = x / y
        C = Composite([x, y, z], [e0, e1, e2])
        c = C.make_node(x, y, z)
        g = FunctionGraph([x, y, z], c.outputs)
        fn = make_function(DualLinker().accept(g))
        assert fn(1.0, 2.0, 3.0) == [6.0, 6.0, 0.5]

    def test_composite_printing(self):
        x, y, z = floats("xyz")
        e0 = x + y + z
        e1 = x + y * z
        e2 = x / y
        e3 = x // 5
        e4 = -x
        e5 = x - y
        e6 = x**y + (-z)
        e7 = x % 3
        C = Composite([x, y, z], [e0, e1, e2, e3, e4, e5, e6, e7])
        c = C.make_node(x, y, z)
        g = FunctionGraph([x, y, z], c.outputs)
        make_function(DualLinker().accept(g))

        assert str(g) == (
            "FunctionGraph(*1 -> Composite{...}(x, y, z), *1::1, *1::2, *1::3, *1::4, *1::5, *1::6, *1::7)"
        )

    def test_non_scalar_error(self):
        x = float32("x")
        with pytest.raises(
            TypeError,
            match="The fgraph of Composite must be exclusively composed of scalar operations",
        ):
            Composite([x], [(pt.zeros((2,)) + x).sum()])

    def test_multi_out_perform(self):
        from pytensor.graph.basic import Apply
        from pytensor.scalar.basic import ScalarOp

        class MultiOutOp(ScalarOp):
            def make_node(self, x):
                return Apply(self, [x], [x.type(), x.type()])

            def perform(self, node, inputs, outputs):
                outputs[1][0] = outputs[0][0] = inputs[0]

            def c_code(self, *args):
                return "dummy"

        x = float32("x")
        comp_op = Composite([x], MultiOutOp()(x))

        y, z = comp_op(x)

        fn = pytensor.function([x], [y, z], mode=Mode("py", None))

        assert fn(1.0) == [1.0, 1.0]


class TestLogical:
    def test_gt(self):
        x, y, _z = floats("xyz")
        fn = make_function(DualLinker().accept(FunctionGraph([x, y], [x > y])))
        for a, b in ((3.0, 9), (3, 0.9), (3, 3)):
            assert fn(a, b) == (a > b)

    def test_lt(self):
        x, y, _z = floats("xyz")
        fn = make_function(DualLinker().accept(FunctionGraph([x, y], [x < y])))
        for a, b in ((3.0, 9), (3, 0.9), (3, 3)):
            assert fn(a, b) == (a < b)

    def test_le(self):
        x, y, _z = floats("xyz")
        fn = make_function(DualLinker().accept(FunctionGraph([x, y], [x <= y])))
        for a, b in ((3.0, 9), (3, 0.9), (3, 3)):
            assert fn(a, b) == (a <= b)

    def test_ge(self):
        x, y, _z = floats("xyz")
        fn = make_function(DualLinker().accept(FunctionGraph([x, y], [x >= y])))
        for a, b in ((3.0, 9), (3, 0.9), (3, 3)):
            assert fn(a, b) == (a >= b)

    def test_eq(self):
        x, y, _z = floats("xyz")
        fn = make_function(DualLinker().accept(FunctionGraph([x, y], [eq(x, y)])))
        for a, b in ((3.0, 9), (3, 0.9), (3, 3)):
            assert fn(a, b) == (a == b)

    def test_neq(self):
        x, y, _z = floats("xyz")
        fn = make_function(DualLinker().accept(FunctionGraph([x, y], [neq(x, y)])))
        for a, b in ((3.0, 9), (3, 0.9), (3, 3)):
            assert fn(a, b) == (a != b)

    def test_or(self):
        x, y, _z = ints("xyz")
        fn = make_function(DualLinker().accept(FunctionGraph([x, y], [x | y])))
        for a, b in ((0, 1), (0, 0), (1, 0), (1, 1)):
            assert fn(a, b) == (a | b), (a, b)

    def test_xor(self):
        x, y, _z = ints("xyz")
        fn = make_function(DualLinker().accept(FunctionGraph([x, y], [x ^ y])))
        for a, b in ((0, 1), (0, 0), (1, 0), (1, 1)):
            assert fn(a, b) == (a ^ b), (a, b)

    def test_and(self):
        x, y, _z = ints("xyz")
        fn = make_function(DualLinker().accept(FunctionGraph([x, y], [and_(x, y)])))
        for a, b in ((0, 1), (0, 0), (1, 0), (1, 1)):
            assert fn(a, b) == (a & b), (a, b)

        x, y, _z = ints("xyz")
        fn = make_function(DualLinker().accept(FunctionGraph([x, y], [x & y])))
        for a, b in ((0, 1), (0, 0), (1, 0), (1, 1)):
            assert fn(a, b) == (a & b), (a, b)

    def test_not(self):
        x, y, _z = ints("xyz")
        fn = make_function(DualLinker().accept(FunctionGraph([x, y], [invert(x)])))
        for a, b in ((0, 1), (0, 0), (1, 0), (1, 1)):
            assert fn(a, b) == ~a, (a,)

        x, y, _z = ints("xyz")
        fn = make_function(DualLinker().accept(FunctionGraph([x, y], [~x])))
        for a, b in ((0, 1), (0, 0), (1, 0), (1, 1)):
            assert fn(a, b) == ~a, (a,)


class TestUpgradeToFloat:
    # Test for Ops whose output has to be floating point, even when all
    # inputs are ints.
    # In particular, when the inputs are int8, the output should be
    # at least float32, not float16.

    unary_ops_vals = [
        (reciprocal, list(range(-127, 0)) + list(range(1, 127))),
        (sqrt, list(range(0, 128))),
        (log, list(range(1, 128))),
        (log2, list(range(1, 128))),
        (log10, list(range(1, 128))),
        (log1p, list(range(0, 128))),
        (exp, list(range(-127, 89))),
        (exp2, list(range(-127, 89))),
        (expm1, list(range(-127, 89))),
        (deg2rad, list(range(-127, 128))),
        (rad2deg, list(range(-127, 128))),
        (cos, list(range(-127, 128))),
        (arccos, list(range(-1, 2))),
        (cosh, list(range(-89, 90))),
        (arccosh, list(range(1, 128))),
        (sin, list(range(-127, 128))),
        (arcsin, list(range(-1, 2))),
        (sinh, list(range(-89, 90))),
        (arcsinh, list(range(-127, 128))),
        (tan, list(range(-3, 4))),
        (arctan, list(range(-127, 128))),
        (tanh, list(range(-127, 128))),
        (arctanh, [0]),
    ]

    binary_ops_vals = [(arctan2, list(range(-127, 128)), list(range(-127, 128)))]

    @staticmethod
    def _test_unary(unary_op, x_range):
        xi = int8("xi")
        xf = float32("xf")

        ei = unary_op(xi)
        fi = pytensor.function([xi], ei)

        ef = unary_op(xf)
        ff = pytensor.function([xf], ef)

        for x_val in x_range:
            outi = fi(x_val)
            outf = ff(x_val)

            if not isinstance(ff.maker.linker, NumbaLinker):
                # Numba doesn't return numpy scalars
                assert outi.dtype == outf.dtype, "incorrect dtype"
            assert np.allclose(outi, outf), "insufficient precision"

    @staticmethod
    def _test_binary(binary_op, x_range, y_range):
        xi = int8("xi")
        yi = int8("yi")
        xf = float32("xf")
        yf = float32("yf")

        ei = binary_op(xi, yi)
        fi = pytensor.function([xi, yi], ei)

        ef = binary_op(xf, yf)
        ff = pytensor.function([xf, yf], ef)

        for x_val in x_range:
            for y_val in y_range:
                outi = fi(x_val, y_val)
                outf = ff(x_val, y_val)

                if not isinstance(ff.maker.linker, NumbaLinker):
                    # Numba doesn't return numpy scalars
                    assert outi.dtype == outf.dtype, "incorrect dtype"
                assert np.allclose(outi, outf), "insufficient precision"

    def test_true_div(self):
        # true_div's upcast policy is not exactly "upgrade_to_float",
        # so the test is a little bit different
        x_range = list(range(-127, 128))
        y_range = list(range(-127, 0)) + list(range(1, 127))

        xi = int8("xi")
        yi = int8("yi")
        xf = ScalarType(pytensor.config.floatX)("xf")
        yf = ScalarType(pytensor.config.floatX)("yf")

        ei = true_div(xi, yi)
        fi = pytensor.function([xi, yi], ei)

        ef = true_div(xf, yf)
        ff = pytensor.function([xf, yf], ef)

        for x_val in x_range:
            for y_val in y_range:
                outi = fi(x_val, y_val)
                outf = ff(x_val, y_val)

                if not isinstance(ff.maker.linker, NumbaLinker):
                    # Numba doesn't return numpy scalars
                    assert outi.dtype == outf.dtype, "incorrect dtype"
                assert np.allclose(outi, outf), "insufficient precision"

    def test_unary(self):
        # Automatically define all individual unary tests
        for unary_op, x_range in self.unary_ops_vals:
            self._test_unary(unary_op, x_range)

    def test_binary(self):
        # Automatically define all individual binary tests
        for binary_op, x_range, y_range in self.binary_ops_vals:
            self._test_binary(binary_op, x_range, y_range)


def test_mod_complex_fail():
    # Make sure % fails on complex numbers.
    x = complex64()
    y = int32()
    with pytest.raises(ComplexError):
        x % y


def test_grad_gt():
    x = float32(name="x")
    y = float32(name="y")
    z = x > y
    g = pytensor.gradient.grad(z, y)
    assert g.eval({y: 1.0}) == 0.0


def test_grad_switch():
    # This is a code snippet from the mailing list
    # It caused an assert to be raised due to the
    # switch op's grad method not handling integer
    # inputs correctly

    x = matrix()
    c = matrix()

    s = pytensor.tensor.switch(c, x, 0)
    l = s.sum()

    pytensor.gradient.grad(l, x)

    # Bug reported in https://github.com/pymc-devs/pytensor/issues/331
    x = matrix(dtype=int)
    s = pytensor.tensor.switch(0, x, -x)
    l = s.sum()

    pytensor.gradient.grad(l, x)


def test_grad_identity():
    # Check that the grad method of Identity correctly handles int dytpes
    x = imatrix("x")
    # tensor_copy is Elemwise{Identity}
    y = pytensor.tensor.tensor_copy(x)
    l = y.sum(dtype=pytensor.config.floatX)
    pytensor.gradient.grad(l, x)


def test_grad_abs():
    a = fscalar("a")
    b = 0.5 * (a + pytensor.tensor.abs(a))
    c = pytensor.grad(b, a)
    f = pytensor.function([a], c, mode=Mode(optimizer=None))
    # Currently PyTensor return 0.5, but it isn't sure it won't change
    # in the futur.
    ret = f(0.0)
    assert ret == 0.5, ret


def test_constant():
    c = constant(2, name="a")
    assert c.name == "a"
    assert c.dtype == "int8"
    c = constant(2, dtype="float32")
    assert c.name is None
    assert c.dtype == "float32"


def test_shape():
    a = float32("a")
    assert isinstance(a.type, ScalarType)
    assert a.shape.type.ndim == 1
    assert a.shape.type.shape == (0,)
    assert a.shape.type.dtype == "int64"

    b = constant(2, name="b")
    assert isinstance(b.type, ScalarType)
    assert b.shape.type.ndim == 1
    assert b.shape.type.shape == (0,)
    assert b.shape.type.dtype == "int64"


def test_grad_log10():
    # Ensure that log10 does not upcast gradient
    # This is a regression test for
    # https://github.com/pymc-devs/pytensor/issues/667
    a = float32("log10_a")
    b = log10(a)
    b_grad = pytensor.gradient.grad(b, a)
    assert b.dtype == "float32"
    assert b_grad.dtype == "float32"


def test_scalar_hash_default_output_type_preference():
    # Old hash used `getattr(self, "output_type_preference", 0)`
    # whereas equality used `getattr(self, "output_type_preference", None)`.
    # Since 27d797076668fbf0617654fd9b91f92ddb6737e6,
    # output_type_preference is always present (None if not specified),
    # which led to C-caching errors when comparing old cached Ops and fresh Ops,
    # as they evaluated equal but hashed differently

    new_eq = EQ()
    old_eq = EQ()
    del old_eq.output_types_preference  # mimic old Op
    assert new_eq == old_eq
    assert hash(new_eq) == hash(old_eq)


def test_rtruediv():
    x = ScalarType(dtype="float64")()
    y = 1.0 / x
    assert isinstance(y.owner.op, TrueDiv)
    assert isinstance(y.type, ScalarType)
    assert y.eval({x: 2.0}) == 0.5


def test_rfloordiv():
    x = ScalarType(dtype="float64")()
    y = 5.0 // x
    assert isinstance(y.owner.op, IntDiv)
    assert isinstance(y.type, ScalarType)
    assert y.eval({x: 2.0}) == 2.0
