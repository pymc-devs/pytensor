import copy

import numpy as np
import pytest

import pytensor.scalar as ps
import pytensor.tensor as pt
from pytensor import shared
from pytensor.compile.maker import function
from pytensor.compile.mode import Mode, get_default_mode, get_mode
from pytensor.compile.ops import deep_copy_op
from pytensor.configdefaults import config
from pytensor.graph.basic import Apply, Variable, equal_computations
from pytensor.graph.destroyhandler import DestroyHandler, _contains_cycle
from pytensor.graph.fg import FrozenFunctionGraph, FunctionGraph
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
from pytensor.tensor.signal import convolve1d
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

            # def infer_shape(self, node, (xshp,)):
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

            def infer_shape(self, node, xshp_):
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

        # Or if more than one dimension cannot be matched
        x = tensor(shape=(None, None, None))
        shape = [x.shape[0], 3, 3]
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

    def test_expand_dims(self):
        x = pt.scalar()
        # This reshape does an implicit expand_dims
        out = x.reshape((1, -1))
        assert isinstance(out.owner.op, Reshape)
        new_out = rewrite_graph(out, include=("canonicalize",))
        assert equal_computations([new_out], [pt.expand_dims(x, (0, 1))])

    def test_squeeze_of_alloc(self):
        # This shows up in the graph of repeat
        x = pt.vector("x", shape=(9,))
        bcast_x = pt.alloc(x, 1, 12, x.shape[0])

        # This reshape does an implicit squeeze
        out = bcast_x.reshape((12, x.shape[0]))

        new_out = rewrite_graph(out, include=("canonicalize", "ShapeOpt"))
        assert equal_computations([new_out], [pt.alloc(x, 12, 9)], strict_dtype=False)

    def test_reshape_implies_size_1_input(self):
        x = pt.matrix("x", shape=(None, None))
        out = pt.reshape(x, (1, 1, 1))

        new_out = rewrite_graph(out, include=("canonicalize",))
        assert equal_computations(
            [new_out], [x.dimshuffle("x", "x", "x")], strict_dtype=False
        )


def test_expand_dims_squeeze_reshape_fusion():
    x = pt.tensor("x", shape=(1, 9))
    reshape_x = x.squeeze(0).reshape((3, 3))[..., None]

    assert isinstance(reshape_x.owner.op, DimShuffle)
    assert isinstance(reshape_x.owner.inputs[0].owner.op, Reshape)
    assert isinstance(reshape_x.owner.inputs[0].owner.inputs[0].owner.op, DimShuffle)

    out = rewrite_graph(reshape_x, include=("specialize",))

    # In this case we cannot get rid of the reshape, squeeze or expand_dims,
    # so we fuse them all in one reshape
    assert equal_computations([out], [x.reshape((3, 3, 1))])


def test_implicit_broadcasting_via_repeat():
    x = pt.vector("x", shape=(3,), dtype=int)
    y = pt.vector("y", shape=(9,), dtype=int)
    out = x[None, :].repeat(9, axis=0) <= y[:, None].repeat(3, axis=1)
    # There are two Reshapes in the graph
    assert isinstance(out.owner.inputs[0].owner.op, Reshape)
    assert isinstance(out.owner.inputs[1].owner.op, Reshape)

    new_out = rewrite_graph(out, include=("canonicalize", "specialize"))
    assert equal_computations([new_out], [x[None] <= y[:, None]])

    no_rewrite_mode = Mode(linker="py", optimizer=None)
    x_test = np.arange(3) + 1
    y_test = np.arange(9)
    np.testing.assert_allclose(
        new_out.eval({x: x_test, y: y_test}, mode=no_rewrite_mode),
        out.eval({x: x_test, y: y_test}, mode=no_rewrite_mode),
    )


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
        utt.assert_allclose(out, advec_val.shape[0])

        admat = matrix()
        admat_val = rng.random((4, 3)).astype(config.floatX)
        for i in range(2):
            f = function([admat], Shape_i(i)(admat))
            out = f(admat_val)
            utt.assert_allclose(out, admat_val.shape[i])

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

    def test_distinct_passthrough_ops(self):
        # Different unary Elemwises (exp vs cos) over the same input have
        # passthrough kernels that bottom out at the same input shape.
        x = vector()
        a = pt.exp(x)
        b = pt.cos(x)
        fgraph = FunctionGraph([x], [a, b], clone=False)
        shape_feature = ShapeFeature()
        fgraph.attach_feature(shape_feature)
        assert shape_feature.same_shape(a, b)

    def test_chained_passthrough(self):
        # ``exp(x)`` and ``exp(x + 1)`` should be same_shape: the inner Add
        # passthrough cascades through the outer Elemwise's passthrough
        # back to ``shape_key(x, 0)``.
        x = vector()
        a = pt.exp(x)
        b = pt.exp(x + 1)
        fgraph = FunctionGraph([x], [a, b], clone=False)
        shape_feature = ShapeFeature()
        fgraph.attach_feature(shape_feature)
        assert shape_feature.same_shape(a, b)

    def test_distinct_sources_shared_shape_arg(self):
        # ``alloc(0., n)`` and ``alloc(1., n)`` have different sources but
        # share the same shape input ``n``. The dim_kernel for Alloc has
        # ``input_slot`` bindings to the shape args; both Allocs bind to
        # the same live ``n``, so same_shape must hold.
        n = iscalar("n")
        a = alloc(np.float64(0.0), n)
        b = alloc(np.float64(1.0), n)
        fgraph = FunctionGraph([n], [a, b], clone=False)
        shape_feature = ShapeFeature()
        fgraph.attach_feature(shape_feature)
        assert shape_feature.same_shape(a, b)

        # Same shape vars at swapped positions: ``alloc(0., n, n+1)``
        # vs ``alloc(0., n+1, n)``. Per-dim queries should detect the
        # cross-dim equivalences; the overall ``same_shape`` (no dims)
        # compares dim-by-dim positionally and should fail.
        n_plus_1 = n + 1
        c = alloc(np.float64(0.0), n, n_plus_1)
        d = alloc(np.float64(0.0), n_plus_1, n)
        fgraph = FunctionGraph([n], [c, d], clone=False)
        shape_feature = ShapeFeature()
        fgraph.attach_feature(shape_feature)
        # Overall fails (dim 0 of c is n, dim 0 of d is n+1, etc.).
        assert not shape_feature.same_shape(c, d)
        # Cross-dim works: same shape var on each side.
        assert shape_feature.same_shape(c, d, 0, 1)  # n == n
        assert shape_feature.same_shape(c, d, 1, 0)  # n+1 == n+1
        # Same-dim comparisons should not match.
        assert not shape_feature.same_shape(c, d, 0, 0)
        assert not shape_feature.same_shape(c, d, 1, 1)

    def test_baked_in_shape_subexpr_limitation(self):
        # KNOWN LIMITATION (documented on ``shape_key``): kernel input
        # bindings compare the live ``node.inputs[k]`` by ``id``, not by
        # structural shape-equivalence. Two structurally-equivalent
        # live shape inputs that happen to be distinct ``Variable``
        # objects yield different ``same_shape`` results.
        #
        # ``reshape(x, exp(s).shape)`` and ``reshape(x, cos(s).shape)``
        # both have output shape equal to ``s.shape`` at runtime, but
        # the live shape inputs (``exp(s).shape`` vs ``cos(s).shape``)
        # are distinct Variables. Reshape's kernel binds the shape
        # input via ``input_slot``, which compares by ``id`` only, so
        # ``same_shape`` returns ``False``.
        #
        # Closing this gap requires inlining sub-kernels at build time
        # (so the parent kernel resolves through ``exp/cos`` into ``s``
        # directly, content-addressing the whole chain). If that lands,
        # flip the assert.
        x = vector()
        s = vector()
        a = reshape(x, pt.exp(s).shape)
        b = reshape(x, pt.cos(s).shape)
        fgraph = FunctionGraph([x, s], [a, b], clone=False)
        shape_feature = ShapeFeature()
        fgraph.attach_feature(shape_feature)
        assert not shape_feature.same_shape(a, b)


def test_unaliased_shape_tuple_blockwise_convolve():
    """Recreate the ``Blockwise(Convolve1d)`` situation from the convolve1d
    gradient that originally triggered the destroy-handler cycle.

    The setup mirrors what late inplace fusion produces: an inplace
    ``Composite{(i + j) - 1}`` over two ``Shape_i`` scalars feeding an
    ``Alloc`` that's the first input of a ``Blockwise(Convolve1d)``. Lazy
    shape materialization traces through the Alloc and pulls the live
    destroyer output into the shape arithmetic. With the second convolve
    input also derived from ``larger`` (same source as the destroyed
    scalar), the shape ends up with an Apply reading *both* the destroyed
    ``Shape_i`` and the destroyer's output — the dual-reference pattern
    that breaks scheduling.

    Asserts:

    1. the naive ``shape_tuple``, once imported into the fgraph alongside
       the inplace destroyer, is flagged as cyclic by the destroy handler
       (this is the bug);
    2. ``unaliased_shape_tuple`` produces the same shape with the cycle-
       pattern Applys rerouted through ``deep_copy_op``, so importing it
       is cycle-free.
    """
    larger = pt.matrix("larger", shape=(8, None))
    smaller = pt.matrix("smaller", shape=(8, None))

    # Pre-warm the ShapeFeature ``Shape_i`` cache so the destroyer's
    # destroyed inputs are the *same* Apply nodes the lazy shape
    # materialization will return later.
    warm_fg = FunctionGraph([larger, smaller], [larger], clone=False)
    warm_sf = ShapeFeature()
    warm_fg.attach_feature(warm_sf)
    larger_s1 = warm_sf.get_shape(larger, 1)
    smaller_s1 = warm_sf.get_shape(smaller, 1)

    # Inplace Composite{(i + j) - 1}: destroys input 0 (= ``larger.shape[1]``).
    sx, sy = ps.int64(), ps.int64()
    inplace_comp = Elemwise(
        ps.Composite([sx, sy], [ps.sub(ps.add(sx, sy), ps.constant(1, dtype="int64"))]),
        inplace_pattern={0: 0},
    )
    new_dim = inplace_comp(larger_s1, smaller_s1)
    a = alloc(pt.zeros((1, 1)), 1, new_dim)
    # Slice of ``larger`` as the second convolve input — its shape depends
    # on ``larger.shape[1]`` (= the destroyed scalar) too. That's what
    # makes the convolve shape arithmetic combine the destroyer's output
    # with the destroyed scalar in a single Apply.
    out = convolve1d(a, larger[:, ::-1], mode="full")

    fg = FunctionGraph([larger, smaller], [out], clone=False)
    sf = ShapeFeature()
    fg.attach_feature(sf)
    fg.attach_feature(DestroyHandler())
    sf._shape_i_cache[(id(larger), 1)] = larger_s1
    sf._shape_i_cache[(id(smaller), 1)] = smaller_s1

    naive_shape = sf.shape_tuple(out)
    safe_shape = sf.unaliased_shape_tuple(out)

    # Importing each shape into a fresh fgraph (with the destroyer present)
    # tells us whether the destroy handler accepts it. A new fgraph per
    # check keeps the cycle from the naive case from poisoning the safe one.
    def imports_with_cycle(shape_vars):
        check_fg = FunctionGraph([larger, smaller], [out, *shape_vars], clone=False)
        check_fg.attach_feature(DestroyHandler())
        dh = check_fg.destroy_handler
        return _contains_cycle(check_fg, dh.orderings(check_fg, ordered=False))

    # Naive lazy shape: destroy handler rejects it.
    assert imports_with_cycle(naive_shape)
    # Cycle-broken version: imports cleanly.
    assert not imports_with_cycle(safe_shape)


class _NoShapeOp(Op):
    """Op without ``infer_shape``, used to drive the kernel-borrow
    override path in ``ShapeFeature.on_change_input``."""

    __props__ = ()

    def make_node(self, x):
        return Apply(self, [x], [x.type()])

    def perform(self, node, inputs, outputs):
        outputs[0][0] = inputs[0]


_no_shape = _NoShapeOp()


class TestKernelReroute:
    """When ``r`` is replaced by ``new_r`` whose Op has no
    ``infer_shape``, ``ShapeFeature.on_change_input`` rederives r's
    shape kernel against ``new_r.owner.inputs`` by matching kernel-
    input bindings via ``shape_key``. The override stores
    ``(dim_kernel, role_bindings)`` per dim — no live ``Variable``
    is pinned. None of these paths are exercised by in-tree rewriters
    today (every common Op has an ``infer_shape``), so each test
    wires up the replacement explicitly.
    """

    def test_passthrough_reroute(self):
        """Passthrough kernel (``exp(x)``: shape = ``[x.shape]``)
        reroutes against ``new_r.owner.inputs = [x]``. The override
        stores only ``(dim_kernel, ((0, 0),))``; ``shape_key`` matches
        ``x.shape[0]`` exactly.
        """
        x = vector("x")
        r = exp(x)
        new_r = _no_shape(x)

        fg = FunctionGraph([x], [r], clone=False)
        sf = ShapeFeature()
        fg.attach_feature(sf)
        fg.replace(r, new_r, reason="reroute_test")

        assert new_r in sf._overrides
        ov = sf._overrides[new_r]
        assert len(ov) == 1
        dim_kernel, role_bindings = ov[0]

        # Pure-kernel structure: no live Variables.
        assert isinstance(dim_kernel, FrozenFunctionGraph)
        assert role_bindings == ((0, 0),)
        for binding in role_bindings:
            assert all(isinstance(b, int) for b in binding)

        live = sf.get_shape(new_r, 0)
        assert live.owner is not None and isinstance(live.owner.op, Shape_i)
        assert live.owner.op.i == 0
        assert live.owner.inputs[0] is x

        assert sf.shape_key(new_r, 0) == sf.shape_key(x, 0)
        assert sf.same_shape(new_r, x, 0, 0)

    def test_no_shape_key_match(self):
        """When the only role's binding has no ``shape_key`` match in
        ``new_r.owner.inputs``, reroute gives up for that dim. Here r
        depends on ``a.shape[0]`` but ``new_r.owner.inputs = [b]`` —
        ``id``-keyed leaves of two distinct vectors don't match, so
        no override is installed.
        """
        a = vector("a")
        b = vector("b")
        r = exp(a)
        new_r = _no_shape(b)

        fg = FunctionGraph([a, b], [r], clone=False)
        sf = ShapeFeature()
        fg.attach_feature(sf)
        fg.replace(r, new_r, reason="reroute_test")

        # Reroute returned None for every dim (here, just dim 0), so
        # no override entry is recorded.
        assert new_r not in sf._overrides
        # Shape lookups fall back to Shape_i on new_r itself.
        live = sf.get_shape(new_r, 0)
        assert isinstance(live.owner.op, Shape_i) and live.owner.inputs[0] is new_r

    def test_input_slot_kernel_skipped(self):
        """A kernel with an ``input_slot`` role (Alloc — its shape
        elements are input *values*, not input *shapes*) can't be
        rerouted by shape-key matching. Reroute bails before searching
        new_r's inputs.
        """
        n = iscalar("n")
        r = alloc(np.float64(0.0), n)
        other = pt.tensor("other", dtype="float64", shape=(None,))
        new_r = _no_shape(other)

        fg = FunctionGraph([n, other], [r], clone=False)
        sf = ShapeFeature()
        fg.attach_feature(sf)
        fg.replace(r, new_r, reason="reroute_test")

        # input_slot role → reroute returns None for every dim.
        assert new_r not in sf._overrides

    def test_partial_reroute(self):
        """Per-dim independence: with ``r = dot(A, B)`` and
        ``new_r = no_shape(A)``, dim 0 (``A.shape[0]``) reroutes
        cleanly while dim 1 (``B.shape[1]``) has no counterpart in
        ``[A]``. The override is installed but with ``None`` at dim 1
        — falling back to ``Shape_i`` for that dim only.
        """
        A = matrix("A")
        B = matrix("B")
        r = pt.dot(A, B)
        new_r = _no_shape(A)

        fg = FunctionGraph([A, B], [r], clone=False)
        sf = ShapeFeature()
        fg.attach_feature(sf)
        fg.replace(r, new_r, reason="reroute_test")

        assert new_r in sf._overrides
        ov = sf._overrides[new_r]
        assert len(ov) == 2

        # Dim 0 rerouted to A.shape[0].
        assert ov[0] is not None
        dim_kernel0, role_bindings0 = ov[0]
        assert isinstance(dim_kernel0, FrozenFunctionGraph)
        assert role_bindings0 == ((0, 0),)

        # Dim 1's binding (B.shape[1]) has no counterpart in [A].
        assert ov[1] is None

        s0 = sf.get_shape(new_r, 0)
        assert isinstance(s0.owner.op, Shape_i) and s0.owner.op.i == 0
        assert s0.owner.inputs[0] is A
        s1 = sf.get_shape(new_r, 1)
        assert isinstance(s1.owner.op, Shape_i) and s1.owner.op.i == 1
        assert s1.owner.inputs[0] is new_r

        assert sf.shape_key(new_r, 0) == sf.shape_key(A, 0)
        assert sf.shape_key(new_r, 1) == ("leaf", id(new_r), 1)


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


def test_local_lift_specify_shape_elemwise():
    x = vector("x")
    out = specify_shape([1.0] + x, shape=(5,))  # noqa: RUF005

    new_out = rewrite_graph(out)
    assert equal_computations([new_out], [[1.0] + specify_shape(x, shape=(5,))])  # noqa: RUF005


def test_local_lift_specify_shape_inc_subtensor():
    x = matrix("x")
    y = vector("y")
    out = specify_shape(set_subtensor(x[1:4], y), shape=(5, None))

    new_out = rewrite_graph(out)
    assert equal_computations(
        [new_out], [set_subtensor(specify_shape(x, shape=(5, None))[1:4], y)]
    )


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
