import re

import numpy as np
import pytest

import pytensor
from pytensor import Mode, function, grad
from pytensor.compile.ops import DeepCopyOp
from pytensor.configdefaults import config
from pytensor.graph.basic import Variable, equal_computations
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.replace import clone_replace, vectorize_node
from pytensor.graph.type import Type
from pytensor.misc.safe_asarray import _asarray
from pytensor.scalar.basic import ScalarConstant
from pytensor.tensor import as_tensor_variable, broadcast_to, get_vector_length, row
from pytensor.tensor.basic import MakeVector, constant, stack
from pytensor.tensor.elemwise import DimShuffle, Elemwise
from pytensor.tensor.rewriting.shape import ShapeFeature
from pytensor.tensor.shape import (
    Reshape,
    Shape,
    Shape_i,
    SpecifyShape,
    Unbroadcast,
    _specify_shape,
    reshape,
    shape,
    shape_i,
    shape_tuple,
    specify_broadcastable,
    specify_shape,
    unbroadcast,
)
from pytensor.tensor.subtensor import Subtensor
from pytensor.tensor.type import (
    TensorType,
    dmatrix,
    dtensor4,
    dvector,
    fvector,
    iscalar,
    ivector,
    lscalar,
    matrix,
    scalar,
    tensor,
    tensor3,
    vector,
)
from pytensor.tensor.type_other import NoneConst
from pytensor.tensor.variable import TensorVariable
from pytensor.typed_list import make_list
from tests import unittest_tools as utt
from tests.graph.utils import MyType2
from tests.tensor.utils import eval_outputs, random
from tests.test_rop import RopLopChecker


def test_shape_basic():
    s = shape([])
    assert s.type.shape == (1,)

    s = shape([10])
    assert s.type.shape == (1,)

    s = shape(lscalar())
    assert s.type.shape == (0,)

    class MyType(Type):
        def filter(self, *args, **kwargs):
            raise NotImplementedError()

        def __eq__(self, other):
            return isinstance(other, MyType) and other.thingy == self.thingy

    s = shape(Variable(MyType(), None))
    assert s.type.shape == (None,)

    s = shape(np.array(1))
    assert np.array_equal(eval_outputs([s]), [])

    s = shape(np.ones((5, 3)))
    assert np.array_equal(eval_outputs([s]), [5, 3])

    s = shape(np.ones(2))
    assert np.array_equal(eval_outputs([s]), [2])

    s = shape(np.ones((5, 3, 10)))
    assert np.array_equal(eval_outputs([s]), [5, 3, 10])


class TestReshape(utt.InferShapeTester, utt.OptimizationTestMixin):
    def setup_method(self):
        self.shared = pytensor.shared
        self.op = Reshape
        # The tag canonicalize is needed for the shape test in FAST_COMPILE
        self.mode = None
        self.ignore_topo = (
            DeepCopyOp,
            MakeVector,
            Shape_i,
            DimShuffle,
            Elemwise,
        )
        super().setup_method()

    def function(self, inputs, outputs, ignore_empty=False):
        f = function(inputs, outputs, mode=self.mode)
        if self.mode is not None or config.mode != "FAST_COMPILE":
            topo = f.maker.fgraph.toposort()
            topo_ = [node for node in topo if not isinstance(node.op, self.ignore_topo)]
            if ignore_empty:
                assert len(topo_) <= 1, topo_
            else:
                assert len(topo_) == 1, topo_
            if len(topo_) > 0:
                assert type(topo_[0].op) is self.op
        return f

    def test_basics(self):
        a = dvector()
        b = dmatrix()
        d = dmatrix()

        b_val1 = np.asarray([[0, 1, 2], [3, 4, 5]])
        c_val1 = np.asarray([0, 1, 2, 3, 4, 5])
        b_val2 = b_val1.T
        c_val2 = np.asarray([0, 3, 1, 4, 2, 5])

        # basic to 1 dim(without list)
        c = reshape(b, as_tensor_variable(6), ndim=1)
        f = self.function([b], c)
        f_out1 = f(b_val1)
        f_out2 = f(b_val2)
        assert np.array_equal(f_out1, c_val1), (f_out1, c_val1)
        assert np.array_equal(f_out2, c_val2), (f_out2, c_val2)

        # basic to 1 dim(with list)
        c = reshape(b, (as_tensor_variable(6),), ndim=1)
        f = self.function([b], c)
        assert np.array_equal(
            f(np.asarray([[0, 1, 2], [3, 4, 5]])), np.asarray([0, 1, 2, 3, 4, 5])
        )

        # basic to shape object of same ndim
        c = reshape(b, d.shape)
        f = self.function([b, d], c)
        assert np.array_equal(
            f(np.asarray([[0, 1, 2], [3, 4, 5]]), [[0, 1], [2, 3], [4, 5]]),
            np.asarray([[0, 1], [2, 3], [4, 5]]),
        )

        # basic to 2 dims
        c = reshape(a, [2, 3])
        f = self.function([a], c)
        assert np.array_equal(
            f(np.asarray([0, 1, 2, 3, 4, 5])), np.asarray([[0, 1, 2], [3, 4, 5]])
        )

        # test that it works without inplace operations
        a_val = np.asarray([0, 1, 2, 3, 4, 5])
        a_val_copy = np.asarray([0, 1, 2, 3, 4, 5])
        b_val = np.asarray([[0, 1, 2], [3, 4, 5]])

        f_sub = self.function([a, b], c - b)
        assert np.array_equal(f_sub(a_val, b_val), np.zeros_like(b_val))
        assert np.array_equal(a_val, a_val_copy)

        # test that it works with inplace operations
        a_val = _asarray([0, 1, 2, 3, 4, 5], dtype="float64")
        a_val_copy = _asarray([0, 1, 2, 3, 4, 5], dtype="float64")
        b_val = _asarray([[0, 1, 2], [3, 4, 5]], dtype="float64")

        f_sub = self.function([a, b], c - b)
        assert np.array_equal(f_sub(a_val, b_val), np.zeros_like(b_val))
        assert np.array_equal(a_val, a_val_copy)

        # verify gradient
        def just_vals(v):
            return Reshape(2)(v, _asarray([2, 3], dtype="int32"))

        utt.verify_grad(just_vals, [a_val], mode=self.mode)

        # test infer_shape
        self._compile_and_check([a], [c], (a_val,), self.op)

        # test broadcast flag for constant value of 1
        c = reshape(b, (b.shape[0], b.shape[1], 1))
        # That reshape may get replaced with a dimshuffle, with is ignored,
        # so we pass "ignore_empty=True"
        f = self.function([b], c, ignore_empty=True)
        assert np.array_equal(
            f(np.asarray([[0, 1, 2], [3, 4, 5]])),
            np.asarray([[[0], [1], [2]], [[3], [4], [5]]]),
        )
        assert f.maker.fgraph.toposort()[-1].outputs[0].type.shape == (
            None,
            None,
            1,
        )

        # test broadcast flag for constant value of 1 if it cannot be
        # replaced with dimshuffle
        c = reshape(b, (b.shape[1], b.shape[0], 1))
        f = self.function([b], c, ignore_empty=True)
        assert np.array_equal(
            f(np.asarray([[0, 1, 2], [3, 4, 5]])),
            np.asarray([[[0], [1]], [[2], [3]], [[4], [5]]]),
        )
        assert f.maker.fgraph.toposort()[-1].outputs[0].type.shape == (
            None,
            None,
            1,
        )

    def test_m1(self):
        t = tensor3()
        rng = np.random.default_rng(seed=utt.fetch_seed())
        val = rng.uniform(size=(3, 4, 5)).astype(config.floatX)
        for out in [
            t.reshape([-1]),
            t.reshape([-1, 5]),
            t.reshape([5, -1]),
            t.reshape([5, -1, 3]),
        ]:
            self._compile_and_check([t], [out], [val], self.op)

    def test_reshape_long_in_shape(self):
        v = dvector("v")
        r = v.reshape((v.shape[0], 1))
        assert np.allclose(r.eval({v: np.arange(5.0)}).T, np.arange(5.0))

    def test_bad_shape(self):
        a = matrix("a")
        shapes = ivector("shapes")
        rng = np.random.default_rng(seed=utt.fetch_seed())
        a_val = rng.uniform(size=(3, 4)).astype(config.floatX)

        # Test reshape to 1 dim
        r = a.reshape(shapes, ndim=1)

        f = self.function([a, shapes], r)
        with pytest.raises(ValueError):
            f(a_val, [13])

        # Test reshape to 2 dim
        r = a.reshape(shapes, ndim=2)

        f = self.function([a, shapes], r)

        with pytest.raises(ValueError):
            f(a_val, [-1, 5])
        with pytest.raises(ValueError):
            f(a_val, [7, -1])
        with pytest.raises(ValueError):
            f(a_val, [7, 5])
        with pytest.raises(ValueError):
            f(a_val, [-1, -1])
        with pytest.raises(
            ValueError, match=".*Shape argument to Reshape has incorrect length.*"
        ):
            f(a_val, [3, 4, 1])

    def test_0(self):
        x = fvector("x")
        f = self.function([x], x.reshape((0, 100)))
        assert f(np.ndarray((0,), dtype="float32")).shape == (0, 100)

    def test_empty_shp(self):
        const = constant([1]).reshape(())
        f = function([], const)
        assert f().shape == ()

    def test_more_shapes(self):
        # TODO: generalize infer_shape to account for tensor variable
        # (non-constant) input shape
        admat = dmatrix()
        ndim = 1
        admat_val = random(3, 4)
        self._compile_and_check(
            [admat], [Reshape(ndim)(admat, [12])], [admat_val], Reshape
        )

        self._compile_and_check(
            [admat], [Reshape(ndim)(admat, [-1])], [admat_val], Reshape
        )

        ndim = 2
        self._compile_and_check(
            [admat], [Reshape(ndim)(admat, [4, 3])], [admat_val], Reshape
        )

        self._compile_and_check(
            [admat], [Reshape(ndim)(admat, [4, -1])], [admat_val], Reshape
        )

        self._compile_and_check(
            [admat], [Reshape(ndim)(admat, [3, -1])], [admat_val], Reshape
        )

        self._compile_and_check(
            [admat], [Reshape(ndim)(admat, [-1, 3])], [admat_val], Reshape
        )
        self._compile_and_check(
            [admat], [Reshape(ndim)(admat, [-1, 4])], [admat_val], Reshape
        )

        aivec = ivector()
        self._compile_and_check(
            [admat, aivec], [Reshape(ndim)(admat, aivec)], [admat_val, [4, 3]], Reshape
        )

        self._compile_and_check(
            [admat, aivec], [Reshape(ndim)(admat, aivec)], [admat_val, [4, -1]], Reshape
        )

        adtens4 = dtensor4()
        ndim = 4
        adtens4_val = random(2, 4, 3, 5)
        self._compile_and_check(
            [adtens4], [Reshape(ndim)(adtens4, [1, -1, 10, 4])], [adtens4_val], Reshape
        )

        self._compile_and_check(
            [adtens4], [Reshape(ndim)(adtens4, [1, 3, 10, 4])], [adtens4_val], Reshape
        )

        self._compile_and_check(
            [adtens4, aivec],
            [Reshape(ndim)(adtens4, aivec)],
            [adtens4_val, [1, -1, 10, 4]],
            Reshape,
        )

        self._compile_and_check(
            [adtens4, aivec],
            [Reshape(ndim)(adtens4, aivec)],
            [adtens4_val, [1, 3, 10, 4]],
            Reshape,
        )

    def test_rebuild(self):
        x = as_tensor_variable(50)
        i = vector("i")
        i_test = np.zeros((100,), dtype=config.floatX)
        y = reshape(i, (100 // x, x))
        assert y.type.shape == (2, 50)
        assert tuple(y.shape.eval({i: i_test})) == (2, 50)
        assert y.eval({i: i_test}).shape == (2, 50)

        x_new = as_tensor_variable(25)
        y_new = clone_replace(y, {x: x_new}, rebuild_strict=False)
        assert y_new.type.shape == (4, 25)
        assert tuple(y_new.shape.eval({i: i_test})) == (4, 25)
        assert y_new.eval({i: i_test}).shape == (4, 25)

    def test_static_shape(self):
        dim = lscalar("dim")
        x1 = tensor(shape=(2, 2, None))
        x2 = specify_shape(x1, (2, 2, 6))

        assert reshape(x1, (6, 2)).type.shape == (6, 2)
        assert reshape(x1, (6, -1)).type.shape == (6, None)
        assert reshape(x1, (6, dim)).type.shape == (6, None)
        assert reshape(x1, (6, dim, 2)).type.shape == (6, None, 2)
        assert reshape(x1, (6, 3, 99)).type.shape == (6, 3, 99)

        assert reshape(x2, (6, 4)).type.shape == (6, 4)
        assert reshape(x2, (6, -1)).type.shape == (6, 4)
        assert reshape(x2, (6, dim)).type.shape == (6, 4)
        assert reshape(x2, (6, dim, 2)).type.shape == (6, 2, 2)
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Reshape: Input shape (2, 2, 6) is incompatible with new shape (6, 3, 99)"
            ),
        ):
            reshape(x2, (6, 3, 99))


def test_shape_i_hash():
    assert isinstance(Shape_i(np.int64(1)).__hash__(), int)


class TestSpecifyShape(utt.InferShapeTester):
    mode = None
    input_type = TensorType

    def test_check_inputs(self):
        with pytest.raises(TypeError, match="must be integer types"):
            specify_shape([[1, 2, 3], [4, 5, 6]], (2.2, 3))

        with pytest.raises(TypeError, match="must be integer types"):
            _specify_shape([[1, 2, 3], [4, 5, 6]], *(2.2, 3))

        with pytest.raises(ValueError, match="will never match"):
            specify_shape(matrix(), [4])

        with pytest.raises(ValueError, match="will never match"):
            _specify_shape(matrix(), *[4])

        with pytest.raises(ValueError, match="must have fixed dimensions"):
            specify_shape(matrix(), vector(dtype="int32"))

    def test_scalar_shapes(self):
        with pytest.raises(ValueError, match="will never match"):
            specify_shape(vector(), shape=())
        with pytest.raises(ValueError, match="will never match"):
            specify_shape(matrix(), shape=[])

        x = scalar()
        y = specify_shape(x, shape=())
        f = pytensor.function([x], y, mode=self.mode)
        assert f(15) == 15

        x = vector()
        s = lscalar()
        y = specify_shape(x, shape=s)
        f = pytensor.function([x, s], y, mode=self.mode)
        assert f([15], 1) == [15]

        x = vector()
        s = as_tensor_variable(1, dtype=np.int64)
        y = specify_shape(x, shape=s)
        f = pytensor.function([x], y, mode=self.mode)
        assert f([15]) == [15]

    def test_partial_shapes(self):
        x = matrix()
        s1 = lscalar()
        y = specify_shape(x, (s1, None))
        f = pytensor.function([x, s1], y, mode=self.mode)
        assert f(np.zeros((2, 5), dtype=config.floatX), 2).shape == (2, 5)
        assert f(np.zeros((3, 5), dtype=config.floatX), 3).shape == (3, 5)

    def test_fixed_shapes(self):
        x = vector()
        shape = as_tensor_variable([2])
        y = specify_shape(x, shape)
        assert y.type.shape == (2,)
        assert isinstance(y.shape.owner.op, Shape)

    def test_fixed_partial_shapes(self):
        x = TensorType("floatX", (None, None))("x")
        y = specify_shape(x, (None, 5))
        assert y.type.shape == (None, 5)

        x = TensorType("floatX", (3, None))("x")
        y = specify_shape(x, (None, 5))
        assert y.type.shape == (3, 5)

    def test_python_perform(self):
        """Test the Python `Op.perform` implementation."""
        x = scalar()
        s = as_tensor_variable([], dtype=np.int32)
        y = specify_shape(x, s)
        f = pytensor.function([x], y, mode=Mode("py"))
        assert f(12) == 12

        x = vector()
        s1 = iscalar()
        shape = as_tensor_variable([s1])
        y = specify_shape(x, shape)
        f = pytensor.function([x, shape], y, mode=Mode("py"))
        assert f([1], (1,)) == [1]

        with pytest.raises(AssertionError, match="SpecifyShape:.*"):
            assert f([1], (2,)) == [1]

        x = matrix()
        y = specify_shape(x, (None, 2))
        f = pytensor.function([x], y, mode=Mode("py"))
        assert f(np.zeros((3, 2), dtype=config.floatX)).shape == (3, 2)
        with pytest.raises(AssertionError, match="SpecifyShape:.*"):
            assert f(np.zeros((3, 3), dtype=config.floatX))

    def test_bad_shape(self):
        """Test that at run-time we raise an exception when the shape is not the one specified."""
        specify_shape = SpecifyShape()

        x = vector()
        xval = np.random.random(2).astype(config.floatX)
        f = pytensor.function([x], specify_shape(x, 2), mode=self.mode)

        assert np.array_equal(f(xval), xval)

        xval = np.random.random(3).astype(config.floatX)
        with pytest.raises(AssertionError, match="SpecifyShape:.*"):
            f(xval)

        assert isinstance(
            next(n for n in f.maker.fgraph.toposort() if isinstance(n.op, SpecifyShape))
            .inputs[0]
            .type,
            self.input_type,
        )

        x = matrix()
        xval = np.random.random((2, 3)).astype(config.floatX)
        f = pytensor.function([x], specify_shape(x, 2, 3), mode=self.mode)
        assert isinstance(
            next(n for n in f.maker.fgraph.toposort() if isinstance(n.op, SpecifyShape))
            .inputs[0]
            .type,
            self.input_type,
        )

        assert np.array_equal(f(xval), xval)

        for shape_ in [(4, 3), (2, 8)]:
            xval = np.random.random(shape_).astype(config.floatX)
            with pytest.raises(AssertionError, match="SpecifyShape:.*"):
                f(xval)

        s = iscalar("s")
        f = pytensor.function([x, s], specify_shape(x, None, s), mode=self.mode)
        x_val = np.zeros((3, 2), dtype=config.floatX)
        assert f(x_val, 2).shape == (3, 2)
        with pytest.raises(AssertionError, match="SpecifyShape:.*"):
            f(xval, 3)

    def test_infer_shape(self):
        rng = np.random.default_rng(3453)
        adtens4 = dtensor4()
        aivec = TensorVariable(TensorType("int64", (4,)), None)
        aivec_val = [3, 4, 2, 5]
        adtens4_val = rng.random(aivec_val)
        self._compile_and_check(
            [adtens4, aivec],
            [specify_shape(adtens4, aivec)],
            [adtens4_val, aivec_val],
            SpecifyShape,
        )

    def test_infer_shape_partial(self):
        rng = np.random.default_rng(3453)
        adtens4 = dtensor4()
        aivec = [iscalar(), iscalar(), None, iscalar()]
        aivec_val = [3, 4, 5]
        adtens4_val = rng.random((3, 4, 2, 5))
        self._compile_and_check(
            [adtens4, *(ivec for ivec in aivec if ivec is not None)],
            [specify_shape(adtens4, aivec)],
            [adtens4_val, *aivec_val],
            SpecifyShape,
        )

    def test_direct_return(self):
        """Test that when specified shape does not provide new information, input is
        returned directly."""
        x = TensorType("float64", shape=(1, 2, None))("x")

        assert specify_shape(x, (1, 2, None)) is x
        assert specify_shape(x, (None, None, None)) is x

        assert specify_shape(x, (1, 2, 3)) is not x
        assert specify_shape(x, (None, None, 3)) is not x
        assert specify_shape(x, (1, 3, None)) is not x

    def test_specify_shape_in_grad(self):
        x = matrix()
        y = specify_shape(x, (2, 3))
        z = y + 1
        z_grad = grad(z.sum(), wrt=x)
        assert isinstance(z_grad.owner.op, SpecifyShape)

    def test_rebuild(self):
        x = as_tensor_variable(50)
        i = matrix("i")
        i_test = np.zeros((4, 50), dtype=config.floatX)
        y = specify_shape(i, (None, x))
        assert y.type.shape == (None, 50)
        assert tuple(y.shape.eval({i: i_test})) == (4, 50)
        assert y.eval({i: i_test}).shape == (4, 50)

        x_new = as_tensor_variable(100)
        i_test = np.zeros((4, 100), dtype=config.floatX)
        y_new = clone_replace(y, {x: x_new}, rebuild_strict=False)
        assert y_new.type.shape == (None, 100)
        assert tuple(y_new.shape.eval({i: i_test})) == (4, 100)
        assert y_new.eval({i: i_test}).shape == (4, 100)


class TestSpecifyBroadcastable:
    def test_basic(self):
        x = matrix()
        assert specify_broadcastable(x, 0).type.shape == (1, None)
        assert specify_broadcastable(x, 1).type.shape == (None, 1)
        assert specify_broadcastable(x, -1).type.shape == (None, 1)
        assert specify_broadcastable(x, 0, 1).type.shape == (1, 1)

        x = row()
        assert specify_broadcastable(x, 0) is x
        assert specify_broadcastable(x, 1) is not x
        assert specify_broadcastable(x, -2) is x

    def test_validation(self):
        x = matrix()
        axis = 2
        with pytest.raises(
            ValueError,
            match=f"axis {axis} is out of bounds for array of dimension {axis}",
        ):
            specify_broadcastable(x, axis)


class TestRopLop(RopLopChecker):
    def test_shape(self):
        self.check_nondiff_rop(self.x.shape[0])

    def test_specifyshape(self):
        self.check_rop_lop(specify_shape(self.x, self.in_shape), self.in_shape)

    def test_reshape(self):
        new_shape = constant(
            np.asarray([self.mat_in_shape[0] * self.mat_in_shape[1]], dtype="int64")
        )

        self.check_mat_rop_lop(
            self.mx.reshape(new_shape), (self.mat_in_shape[0] * self.mat_in_shape[1],)
        )


@config.change_flags(compute_test_value="raise")
def test_nonstandard_shapes():
    a = tensor3(config.floatX)
    a.tag.test_value = np.random.random((2, 3, 4)).astype(config.floatX)
    b = tensor3(config.floatX)
    b.tag.test_value = np.random.random((2, 3, 4)).astype(config.floatX)

    tl = make_list([a, b])
    tl_shape = shape(tl)
    assert np.array_equal(tl_shape.get_test_value(), (2, 2, 3, 4))

    # There's no `FunctionGraph`, so it should return a `Subtensor`
    tl_shape_i = shape_i(tl, 0)
    assert isinstance(tl_shape_i.owner.op, Subtensor)
    assert tl_shape_i.get_test_value() == 2

    tl_fg = FunctionGraph([a, b], [tl], features=[ShapeFeature()])
    tl_shape_i = shape_i(tl, 0, fgraph=tl_fg)
    assert not isinstance(tl_shape_i.owner.op, Subtensor)
    assert tl_shape_i.get_test_value() == 2

    none_shape = shape(NoneConst)
    assert np.array_equal(none_shape.get_test_value(), [])


def test_shape_i_basics():
    with pytest.raises(TypeError):
        Shape_i(0)([1, 2])

    with pytest.raises(TypeError):
        Shape_i(0)(scalar())


def test_get_vector_length():
    # Test `Shape`s
    x = pytensor.shared(np.zeros((2, 3, 4, 5)))
    assert get_vector_length(x.shape) == 4

    # Test `SpecifyShape`
    x = specify_shape(ivector(), (10,))
    assert get_vector_length(x) == 10


class TestUnbroadcast:
    def test_basic(self):
        x = matrix()
        assert unbroadcast(x, 0) is x
        assert unbroadcast(x, 1) is x
        assert unbroadcast(x, 1, 0) is x
        assert unbroadcast(x, 0, 1) is x

        x = row()
        assert unbroadcast(x, 0) is not x
        assert unbroadcast(x, 1) is x
        assert unbroadcast(x, 1, 0) is not x
        assert unbroadcast(x, 0, 1) is not x

        assert unbroadcast(unbroadcast(x, 0), 0).owner.inputs[0] is x

    def test_infer_shape(self):
        x = matrix()
        y = unbroadcast(x, 0)
        f = pytensor.function([x], y.shape)
        assert (f(np.zeros((2, 5), dtype=config.floatX)) == [2, 5]).all()
        topo = f.maker.fgraph.toposort()
        if config.mode != "FAST_COMPILE":
            assert len(topo) == 3
            assert isinstance(topo[0].op, Shape_i)
            assert isinstance(topo[1].op, Shape_i)
            assert isinstance(topo[2].op, MakeVector)

        x = row()
        y = unbroadcast(x, 0)
        f = pytensor.function([x], y.shape)
        assert (f(np.zeros((1, 5), dtype=config.floatX)) == [1, 5]).all()
        topo = f.maker.fgraph.toposort()
        if config.mode != "FAST_COMPILE":
            assert len(topo) == 2
            assert isinstance(topo[0].op, Shape_i)
            assert isinstance(topo[1].op, MakeVector)

    def test_error_checks(self):
        with pytest.raises(TypeError, match="needs integer axes"):
            Unbroadcast(0.0)

        with pytest.raises(ValueError, match="^Trying to unbroadcast"):
            Unbroadcast(1)(vector())


class TestUnbroadcastInferShape(utt.InferShapeTester):
    def test_basic(self):
        rng = np.random.default_rng(3453)
        adtens4 = tensor(dtype="float64", shape=(1, 1, 1, None))
        adtens4_val = rng.random((1, 1, 1, 3)).astype(config.floatX)
        self._compile_and_check(
            [adtens4],
            [Unbroadcast(0, 2)(adtens4)],
            [adtens4_val],
            Unbroadcast,
            warn=False,
        )


def test_shape_tuple():
    x = Variable(MyType2(), None, None)
    assert shape_tuple(x) == ()

    x = tensor(dtype=np.float64, shape=(1, 2, None))
    res = shape_tuple(x)
    assert isinstance(res, tuple)
    assert isinstance(res[0], ScalarConstant)
    assert res[0].data == 1
    assert isinstance(res[1], ScalarConstant)
    assert res[1].data == 2
    assert not isinstance(res[2], ScalarConstant)


class TestVectorize:
    @pytensor.config.change_flags(cxx="")  # For faster eval
    def test_shape(self):
        vec = tensor(shape=(None,), dtype="float64")
        mat = tensor(shape=(None, None), dtype="float64")
        node = shape(vec).owner

        [vect_out] = vectorize_node(node, mat).outputs
        assert equal_computations(
            [vect_out], [broadcast_to(mat.shape[1:], (*mat.shape[:1], 1))]
        )

        mat_test_value = np.ones((5, 3))
        ref_fn = np.vectorize(lambda vec: np.asarray(vec.shape), signature="(vec)->(1)")
        np.testing.assert_array_equal(
            vect_out.eval({mat: mat_test_value}),
            ref_fn(mat_test_value),
        )

        mat = tensor(shape=(None, None), dtype="float64")
        tns = tensor(shape=(None, None, None, None), dtype="float64")
        node = shape(mat).owner
        [vect_out] = vectorize_node(node, tns).outputs
        assert equal_computations(
            [vect_out], [broadcast_to(tns.shape[2:], (*tns.shape[:2], 2))]
        )

        tns_test_value = np.ones((4, 6, 5, 3))
        ref_fn = np.vectorize(
            lambda vec: np.asarray(vec.shape), signature="(m1,m2)->(2)"
        )
        np.testing.assert_array_equal(
            vect_out.eval({tns: tns_test_value}),
            ref_fn(tns_test_value),
        )

    @pytensor.config.change_flags(cxx="")  # For faster eval
    def test_reshape(self):
        x = scalar("x", dtype=int)
        vec = tensor(shape=(None,), dtype="float64")
        mat = tensor(shape=(None, None), dtype="float64")

        shape = (-1, x)
        node = reshape(vec, shape).owner

        [vect_out] = vectorize_node(node, mat, shape).outputs
        assert equal_computations([vect_out], [reshape(mat, (*mat.shape[:1], -1, x))])

        x_test_value = 2
        mat_test_value = np.ones((5, 6))
        ref_fn = np.vectorize(
            lambda x, vec: vec.reshape(-1, x), signature="(),(vec1)->(mat1,mat2)"
        )
        np.testing.assert_array_equal(
            vect_out.eval({x: x_test_value, mat: mat_test_value}),
            ref_fn(x_test_value, mat_test_value),
        )

        new_shape = (5, -1, x)
        [vect_out] = vectorize_node(node, mat, new_shape).outputs
        assert equal_computations([vect_out], [reshape(mat, new_shape)])

        new_shape = stack([[-1, x], [x - 1, -1]], axis=0)
        print(new_shape.type)
        [vect_out] = vectorize_node(node, vec, new_shape).outputs
        vec_test_value = np.arange(6)
        np.testing.assert_allclose(
            vect_out.eval({x: 3, vec: vec_test_value}),
            np.broadcast_to(vec_test_value.reshape(2, 3), (2, 2, 3)),
        )

        with pytest.raises(
            ValueError,
            match="Invalid shape length passed into vectorize node of Reshape",
        ):
            vectorize_node(node, vec, (5, 2, x))

        with pytest.raises(
            ValueError,
            match="Invalid shape length passed into vectorize node of Reshape",
        ):
            vectorize_node(node, mat, (5, 3, 2, x))

    def test_specify_shape(self):
        x = scalar("x", dtype=int)
        mat = tensor(shape=(None, None))
        tns = tensor(shape=(None, None, None))

        shape = (x, None)
        node = specify_shape(mat, shape).owner
        vect_node = vectorize_node(node, tns, *shape)
        assert equal_computations(
            vect_node.outputs, [specify_shape(tns, (None, x, None))]
        )

        new_shape = (5, 2, x)
        vect_node = vectorize_node(node, tns, *new_shape)
        assert equal_computations(vect_node.outputs, [specify_shape(tns, (5, 2, x))])

        with pytest.raises(NotImplementedError):
            vectorize_node(node, mat, *([x, x], None))

        with pytest.raises(
            ValueError,
            match="Invalid number of shape arguments passed into vectorize node of SpecifyShape",
        ):
            vectorize_node(node, mat, *(5, 2, x))

        with pytest.raises(
            ValueError,
            match="Invalid number of shape arguments passed into vectorize node of SpecifyShape",
        ):
            vectorize_node(node, tns, *(5, 3, 2, x))

    def test_unbroadcast(self):
        mat = tensor(
            shape=(
                1,
                1,
            )
        )
        tns = tensor(shape=(4, 1, 1, 1))

        node = unbroadcast(mat, 0).owner
        vect_node = vectorize_node(node, tns)
        assert equal_computations(vect_node.outputs, [unbroadcast(tns, 2)])
