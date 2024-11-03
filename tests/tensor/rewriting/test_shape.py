import copy

import numpy as np
import pytest

import pytensor.tensor as pt
from pytensor import shared
from pytensor.compile.function import function
from pytensor.compile.mode import get_default_mode, get_mode
from pytensor.compile.ops import deep_copy_op
from pytensor.configdefaults import config
from pytensor.graph.basic import Apply, Variable, equal_computations
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.op import Op
from pytensor.graph.rewriting.basic import check_stack_trace, node_rewriter, out2in
from pytensor.graph.rewriting.utils import rewrite_graph
from pytensor.graph.type import Type
from pytensor.tensor.basic import alloc, as_tensor_variable
from pytensor.tensor.elemwise import DimShuffle, Elemwise
from pytensor.tensor.math import add, exp, maximum
from pytensor.tensor.rewriting.basic import register_specialize
from pytensor.tensor.rewriting.shape import (
    ShapeFeature,
    local_reshape_to_dimshuffle,
    local_useless_reshape,
    local_useless_specify_shape,
)
from pytensor.tensor.shape import (
    Reshape,
    Shape_i,
    SpecifyShape,
    reshape,
    shape,
    specify_shape,
)
from pytensor.tensor.subtensor import set_subtensor
from pytensor.tensor.type import (
    fmatrix,
    iscalar,
    lscalar,
    matrix,
    scalar,
    tensor,
    tensor3,
    tensor4,
    vector,
)
from tests import unittest_tools as utt


rewrite_mode = config.mode

if rewrite_mode == "FAST_COMPILE":
    rewrite_mode = "FAST_RUN"

rewrite_mode = get_mode(rewrite_mode)


class TestShapeRewriter:
    def test_basic(self):
        mode = config.mode
        if mode == "FAST_COMPILE":
            mode = "FAST_RUN"
        v = vector()
        m = matrix()
        f = function([v, m], (v + m).shape, mode=mode)
        for node in f.maker.fgraph.toposort():
            assert node.op != add

    def test_constant(self):
        mode = config.mode
        if mode == "FAST_COMPILE":
            mode = "FAST_RUN"

        v = vector()
        f = function([v], v.dimshuffle("x", "x", 0).shape[1], mode=mode)
        topo = f.maker.fgraph.toposort()
        assert len(topo) == 1
        assert topo[0].op == deep_copy_op

    @staticmethod
    def max_pool_c01b(c01b, pool_shp, pool_stride, img_shp):
        """
        Like max_pool but with input using axes ('c', 0, 1, 'b')
          (Alex Krizhevsky format)

        pool_shp, pool_stride and img_shp are int that represent
        the same shp in x and y.
        """
        mx = None

        # Compute index in pooled space of last needed pool
        # (needed = each input pixel must appear in at least one pool)
        def last_pool(im_shp, p_shp, p_strd):
            rval = int(np.ceil(float(im_shp - p_shp) / p_strd))
            assert p_strd * rval + p_shp >= im_shp
            assert p_strd * (rval - 1) + p_shp < im_shp
            return rval

        # Compute starting row of the last pool
        last_pool_r = last_pool(img_shp, pool_shp, pool_stride) * pool_stride
        # Compute number of rows needed in img for all indexes to work out
        required_r = last_pool_r + pool_shp

        last_pool_c = last_pool(img_shp, pool_shp, pool_stride) * pool_stride
        required_c = last_pool_c + pool_shp

        wide_infinity = pt.alloc(
            -np.inf, c01b.shape[0], required_r, required_c, c01b.shape[3]
        )

        c01b = set_subtensor(wide_infinity[:, 0:img_shp, 0:img_shp, :], c01b)

        for row_within_pool in range(pool_shp):
            row_stop = last_pool_r + row_within_pool + 1
            for col_within_pool in range(pool_shp):
                col_stop = last_pool_c + col_within_pool + 1
                cur = c01b[
                    :,
                    row_within_pool:row_stop:pool_stride,
                    col_within_pool:col_stop:pool_stride,
                    :,
                ]
                if mx is None:
                    mx = cur
                else:
                    mx = maximum(mx, cur)
        return mx

    def test_broadcasted_dims(self):
        # This test a case that caused a crash during rewriting
        shp = (1, 1, 1, 1)
        rng = np.random.default_rng(utt.fetch_seed())
        a = shared(rng.random(shp).astype(config.floatX))
        out = self.max_pool_c01b(a, 1, 1, 1)

        # max_pool_c01b use -inf and this will trigger DebugMode error.
        mode = copy.copy(get_default_mode())
        mode.check_isfinite = False
        f = function([], out, mode=mode)
        f()

    def test_constant_merge(self):
        # This test the error in gh-1122 that is a caused by the
        # combination of merge rewriter and ShapeFeature.

        x = pt.constant([0, 0])
        y = x[1:]
        x1 = x - pt.join(0, y, y)
        x1.eval()

    def test_local_track_shape_i(self):
        class IdentityNoShape(Op):
            """Op that does not infer the output shape from the input one"""

            def make_node(self, x):
                x = as_tensor_variable(x)
                return Apply(self, [x], [x.type()])

            def perform(self, node, inp, out_):
                (x,) = inp
                (out,) = out_
                out[0] = x.copy()

            # def infer_shape(self, fgraph, node, (xshp,)):
            # return [tuple([self.shape_i(i)(r) for i in range(r.ndim)])]

        identity_noshape = IdentityNoShape()

        class IdentityShape(Op):
            """Op that does infer the output shape from the input one"""

            def make_node(self, x):
                x = as_tensor_variable(x)
                return Apply(self, [x], [x.type()])

            def perform(self, node, inp, out_):
                (x,) = inp
                (out,) = out_
                out[0] = x.copy()

            def infer_shape(self, fgraph, node, xshp_):
                # Could also just return.
                (xshp,) = xshp_
                return (xshp,)

        identity_shape = IdentityShape()

        @node_rewriter([IdentityNoShape])
        def local_identity_noshape_to_identity_shape(fgraph, node):
            """Transform the first `Op` into the second."""
            if isinstance(node.op, IdentityNoShape):
                return [identity_shape(node.inputs[0])]

        mode = get_default_mode().including("ShapeOpt", "specialize")
        rng = np.random.default_rng(utt.fetch_seed())
        x = tensor3("x")
        ins_x = identity_noshape(x)

        # Without the rewrite
        f = function([x], ins_x.shape, mode=mode)
        xval = rng.standard_normal((3, 4, 7)).astype(config.floatX)
        assert np.all(f(xval) == [3, 4, 7])
        f_ops = [node.op for node in f.maker.fgraph.toposort()]
        assert len(f_ops) == 5
        assert identity_noshape in f_ops
        assert identity_shape not in f_ops

        # Register the rewrite
        register_specialize(local_identity_noshape_to_identity_shape)

        mode = get_default_mode().including("ShapeOpt", "specialize")
        # The `identity_shape` hOph should not be needed anymore to compute
        # the shape
        g = function([x], ins_x.shape, mode=mode)
        xval = rng.standard_normal((6, 1, 2)).astype(config.floatX)
        assert np.all(g(xval) == [6, 1, 2])
        g_ops = [node.op for node in g.maker.fgraph.toposort()]
        assert len(g_ops) == 4
        assert identity_noshape not in g_ops
        assert identity_shape not in g_ops

        # Test multiple applications of an `Op` without an `Op.infer_shape`
        ins_x3 = identity_noshape(identity_noshape(identity_noshape(x)))
        h = function([x], ins_x3.shape, mode=mode)
        xval = rng.standard_normal((6, 1, 2)).astype(config.floatX)
        assert np.all(h(xval) == [6, 1, 2])
        h_ops = [node.op for node in h.maker.fgraph.toposort()]
        assert len(h_ops) == 4
        assert identity_noshape not in h_ops
        assert identity_shape not in h_ops

    def test_no_shapeopt(self):
        """Test that a basic example works even when `ShapeOpt` is excluded."""
        X = matrix()
        expr = X.shape[0]

        mode = get_default_mode().excluding("ShapeOpt")
        f = function([X], expr, mode=mode)
        # FIXME: This is not a good test.
        f([[1, 2], [2, 3]])

    def test_shape_of_useless_alloc(self):
        """Test that local_shape_to_shape_i does not create circular graph.

        Regression test for #565
        """
        alpha = vector(shape=(None,), dtype="float64")
        channel = vector(shape=(None,), dtype="float64")

        broadcast_channel = alloc(
            channel,
            maximum(
                shape(alpha)[0],
                shape(channel)[0],
            ),
        )
        out = shape(broadcast_channel)
        fn = function([alpha, channel], out)
        assert fn([1.0, 2, 3], [1.0, 2, 3]) == (3,)


class TestReshape:
    def setup_method(self):
        self.mode = rewrite_mode
        self.op = Reshape

    def test_local_reshape(self):
        a = fmatrix()
        b = self.op(3)(a, [2, 3, 4])
        c = self.op(1)(b, [24])
        f = function([a], c, mode=self.mode)
        topo = f.maker.fgraph.toposort()
        assert sum(isinstance(node.op, self.op) for node in topo) == 1

        # Check stack trace
        assert check_stack_trace(f, ops_to_check=[self.op])


class TestLocalUselessReshape:
    def setup_method(self):
        self.rng = np.random.default_rng(utt.fetch_seed())

    def test_0(self):
        mode = get_default_mode().including("local_useless_reshape")
        i = iscalar("i")
        m = pt.mgrid[0:i,]
        f = function([i], m, mode=mode)
        topo = f.maker.fgraph.toposort()
        assert not any(isinstance(n.op, Reshape) for n in topo)

    def test_1(self):
        x = matrix("x")
        r = x.reshape(x.shape)

        m0 = get_default_mode()
        m1 = m0.including("local_useless_reshape")
        f1 = function([x], r, mode=m1)
        topo = f1.maker.fgraph.toposort()
        assert not any(isinstance(n.op, Reshape) for n in topo)

        m2 = m1.excluding("ShapeOpt")
        f2 = function([x], r, mode=m2)
        topo = f2.maker.fgraph.toposort()
        assert not any(isinstance(n.op, Reshape) for n in topo)

        # We do not need tests checking that stack traces are copied over,
        # because local_useless_reshape only removes nodes from the graph

    def test_2(self):
        x = matrix("x")
        r = x.reshape([Shape_i(i)(x) for i in range(x.ndim)])

        m0 = get_default_mode()
        m1 = m0.including("local_useless_reshape")
        f1 = function([x], r, mode=m1)
        topo = f1.maker.fgraph.toposort()
        assert not any(isinstance(n.op, Reshape) for n in topo)

        m2 = m1.excluding("ShapeOpt")
        f2 = function([x], r, mode=m2)
        topo = f2.maker.fgraph.toposort()
        assert not any(isinstance(n.op, Reshape) for n in topo)

    def test_m1(self):
        x = matrix("x")
        r = x.reshape((x.shape[0], -1))

        m0 = get_default_mode()
        m1 = m0.including("local_useless_reshape")
        f1 = function([x], r, mode=m1)
        topo = f1.maker.fgraph.toposort()
        assert not any(isinstance(n.op, Reshape) for n in topo)

        m2 = m1.excluding("ShapeOpt")
        f2 = function([x], r, mode=m2)
        topo = f2.maker.fgraph.toposort()
        assert not any(isinstance(n.op, Reshape) for n in topo)

    def test_constant_shape(self):
        # Where reshape is a constant that matches the shape
        x = matrix(shape=(2, 3))
        shape = pt.as_tensor(np.array([2, 3]))
        out = reshape(x, shape)
        new_out = rewrite_graph(out)
        assert new_out is x

        x = matrix(shape=(2, 3))
        shape = pt.as_tensor(np.array([-1, 3]))
        out = reshape(x, shape)
        new_out = rewrite_graph(out)
        assert new_out is x

        x = matrix(shape=(None, 3))
        shape = pt.as_tensor(np.array([-1, 3]))
        out = reshape(x, shape)
        new_out = rewrite_graph(out)
        assert new_out is x

        x = matrix(shape=(None, 3))
        shape = pt.as_tensor(np.array([2, 3]))
        out = reshape(x, shape)
        new_out = rewrite_graph(out)
        # This could be rewritten as a specify_shape(x, (2, 3))
        assert new_out is not x

        x = matrix(shape=(2, 3))
        shape = pt.as_tensor(np.array([3, 2]))
        out = reshape(x, shape)
        new_out = rewrite_graph(out)
        assert new_out is not x

    def test_all_but_one_match(self):
        x = matrix(shape=(None, None))
        shape = [x.shape[0], 3]
        out = reshape(x, shape)
        new_out = rewrite_graph(out)
        assert equal_computations([new_out], [specify_shape(x, (None, 3))])

        # Rewrite does not apply if there's also a -1
        shape = [-1, 3]
        out = reshape(x, shape)
        new_out = rewrite_graph(out)
        assert new_out is out


class TestLocalReshapeToDimshuffle:
    def setup_method(self):
        self.rng = np.random.default_rng(utt.fetch_seed())

    def test_basic(self):
        reshape_lift = out2in(local_reshape_to_dimshuffle)
        useless_reshape = out2in(local_useless_reshape)
        x = shared(self.rng.standard_normal((4,)))
        y = shared(self.rng.standard_normal((5, 6)))
        reshape_x = reshape(x, (1, 4))
        reshape_y = reshape(y, (1, 5, 1, 6, 1, 1))

        g = FunctionGraph([x, y], [reshape_x, reshape_y], clone=False)

        assert equal_computations(
            g.outputs,
            [
                Reshape(2)(x, as_tensor_variable((1, 4), ndim=1)),
                Reshape(6)(y, as_tensor_variable((1, 5, 1, 6, 1, 1), ndim=1)),
            ],
        )

        reshape_lift.rewrite(g)
        useless_reshape.rewrite(g)

        exp_x = SpecifyShape()(x, 4).dimshuffle("x", 0)
        assert equal_computations([g.outputs[0]], [exp_x])

        exp_y = Reshape(2)(y, as_tensor_variable((5, 6), ndim=1)).dimshuffle(
            "x", 0, "x", 1, "x", "x"
        )
        assert equal_computations([g.outputs[1]], [exp_y])

        assert check_stack_trace(g, ops_to_check=(DimShuffle, Reshape))


def test_local_reshape_lift():
    x = tensor4()
    out = exp(x).reshape([x.size])
    assert out.ndim == 1
    mode = get_default_mode()
    mode = mode.including("local_reshape_lift")
    f = function([x], out, mode=mode)
    f(np.random.random((5, 4, 3, 2)).astype(config.floatX))
    topo = f.maker.fgraph.toposort()
    assert isinstance(topo[-2].op, Reshape)
    assert isinstance(topo[-1].op, Elemwise)
    assert check_stack_trace(f, ops_to_check="last")


class TestShapeI(utt.InferShapeTester):
    def setup_method(self):
        super().setup_method()

    def test_perform(self):
        rng = np.random.default_rng(utt.fetch_seed())

        advec = vector()
        advec_val = rng.random(3).astype(config.floatX)
        f = function([advec], Shape_i(0)(advec))
        out = f(advec_val)
        np.testing.assert_allclose(out, advec_val.shape[0])

        admat = matrix()
        admat_val = rng.random((4, 3)).astype(config.floatX)
        for i in range(2):
            f = function([admat], Shape_i(i)(admat))
            out = f(admat_val)
            np.testing.assert_allclose(out, admat_val.shape[i])

    def test_infer_shape(self):
        admat = matrix()
        admat_val = np.random.random((3, 4)).astype(config.floatX)
        self._compile_and_check([admat], [Shape_i(0)(admat)], [admat_val], Shape_i)

        self._compile_and_check([admat], [Shape_i(1)(admat)], [admat_val], Shape_i)


class TestSameShape:
    def test_scalar(self):
        x = scalar()
        cst = pt.constant(1)
        o = x + cst
        fgraph = FunctionGraph([x], [o], clone=False)
        shape_feature = ShapeFeature()
        fgraph.attach_feature(shape_feature)
        assert shape_feature.same_shape(x, o)

    def test_vector(self):
        x = vector()
        cst = pt.constant(1)
        o = x + cst
        fgraph = FunctionGraph([x], [o], clone=False)
        shape_feature = ShapeFeature()
        fgraph.attach_feature(shape_feature)
        assert shape_feature.same_shape(x, o)

    def test_no_static_shapes(self):
        x = vector()
        y = vector()
        o = x + y
        fgraph = FunctionGraph([x, y], [o], clone=False)
        shape_feature = ShapeFeature()
        fgraph.attach_feature(shape_feature)
        # We no longer assume that `x` has the same shape as `y` simply because
        # neither has static shape information.  Instead, when there is no
        # static shape information is available, we assume that `x` and/or `y`
        # could have shapes `(1,)` and/or `(n,)`, where `n != 1`, or any
        # combination of the two.
        assert not shape_feature.same_shape(x, o)
        assert not shape_feature.same_shape(y, o)

    @pytest.mark.parametrize(
        "y_dim_0",
        [2, None],
    )
    def test_vector_dim(self, y_dim_0):
        x = pt.tensor(dtype="floatX", shape=(2, None))
        y = pt.tensor(dtype="floatX", shape=(y_dim_0, None))
        o = x + y
        fgraph = FunctionGraph([x, y], [o], clone=False)
        shape_feature = ShapeFeature()
        fgraph.attach_feature(shape_feature)
        assert shape_feature.same_shape(x, o, 0, 0)
        assert not shape_feature.same_shape(x, o, 1, 1)

    def test_vector_dim_err(self):
        x = vector()
        y = vector()
        o = x + y
        fgraph = FunctionGraph([x, y], [o], clone=False)
        shape_feature = ShapeFeature()
        fgraph.attach_feature(shape_feature)
        with pytest.raises(IndexError):
            shape_feature.same_shape(x, o, 1, 0)
        with pytest.raises(IndexError):
            shape_feature.same_shape(x, o, 0, 1)


def test_useless_specify_shape():
    x = tensor("x", shape=(None, 5, 3))

    # We avoid the helper specify_shape that optimizes some (but not all) cases eagerly
    ss = SpecifyShape()

    out = ss(x, None, 5, None)
    assert isinstance(out.owner.op, SpecifyShape)
    ret = local_useless_specify_shape.transform(None, out.owner)
    assert ret == [x]

    # SpecifyShape is needed to enfore unknown dim is 3
    out = ss(x, 3, 5, None)
    assert isinstance(out.owner.op, SpecifyShape)
    ret = local_useless_specify_shape.transform(None, out.owner)
    assert ret is None

    # SpecifyShape is needed to raise mismatch between static and specified dim
    out = ss(x, None, 5, 4)
    assert isinstance(out.owner.op, SpecifyShape)
    ret = local_useless_specify_shape.transform(None, out.owner)
    assert ret is None


@pytest.mark.parametrize(
    "shape",
    [lscalar(), iscalar()],
)
def test_local_Shape_of_SpecifyShape(shape):
    x = vector()
    s = specify_shape(x, shape).shape

    fgraph = FunctionGraph(outputs=[s], clone=False)
    _ = rewrite_graph(fgraph, clone=False)

    assert x not in fgraph.variables
    assert shape in fgraph.variables


@pytest.mark.parametrize(
    "s1",
    [lscalar(), iscalar()],
)
def test_local_Shape_of_SpecifyShape_partial(s1):
    x = matrix()
    s = specify_shape(x, (s1, None)).shape

    fgraph = FunctionGraph(outputs=[s], clone=False)
    assert any(isinstance(apply.op, SpecifyShape) for apply in fgraph.apply_nodes)

    _ = rewrite_graph(fgraph, clone=False)

    assert x in fgraph.variables
    assert s1 in fgraph.variables
    assert not any(isinstance(apply.op, SpecifyShape) for apply in fgraph.apply_nodes)


def test_local_specify_shape_lift():
    x = vector("x")
    out = specify_shape([1.0] + x, shape=(5,))  # noqa: RUF005

    new_out = rewrite_graph(out)
    assert equal_computations([new_out], [[1.0] + specify_shape(x, shape=(5,))])  # noqa: RUF005


def test_local_Shape_i_ground():
    x = tensor(dtype=np.float64, shape=(None, 2))
    s = Shape_i(1)(x)

    fgraph = FunctionGraph(outputs=[s], clone=False)
    _ = rewrite_graph(fgraph, clone=False)

    assert x not in fgraph.variables
    assert fgraph.outputs[0].data == 2

    # A test for a non-`TensorType`
    class MyType(Type):
        ndim = 1

        def filter(self, *args, **kwargs):
            raise NotImplementedError()

        def __eq__(self, other):
            return isinstance(other, MyType) and other.thingy == self.thingy

    class MyVariable(Variable):
        pass

    x = MyVariable(MyType(), None, None)
    s = Shape_i(0)(x)
    fgraph = FunctionGraph(outputs=[s], clone=False)
    _ = rewrite_graph(fgraph, clone=False)

    assert fgraph.outputs[0] == s


def test_Shape_i_canonicalize():
    """Make sure the canonicalizations work together to produce the correct graphs for shapes in a single dimension.

    In other words, ``shape(x)[i]`` should result in a simple ``Shape_i(0)(x)``
    and nothing else.  The rewrites `local_shape_to_shape_i`,
    `local_subtensor_remove_broadcastable_index`, and
    `local_useless_dimshuffle_makevector` need to work together to accomplish
    this, and we confirm that here.
    """
    x = vector()
    y = shape(x)[0]

    y_fg = FunctionGraph(outputs=[y], copy_inputs=False, features=[ShapeFeature()])

    y_rewritten_fg = rewrite_graph(
        y_fg,
        clone=False,
        include=[
            "canonicalize",
        ],
    )

    y_rewritten = y_rewritten_fg.outputs[0]

    assert isinstance(y_rewritten.owner.op, Shape_i)
    assert y_rewritten.owner.op.i == 0
    assert y_rewritten.owner.inputs[0] == x
