from functools import partial

import numpy as np
import pytest

import pytensor.tensor as pt
from pytensor import Mode
from pytensor.compile import shared
from pytensor.compile.builders import OpFromGraph
from pytensor.compile.maker import function
from pytensor.configdefaults import config
from pytensor.gradient import (
    DisconnectedType,
    disconnected_type,
    grad,
    pushforward,
    verify_grad,
)
from pytensor.graph.basic import equal_computations
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.rewriting.basic import MergeOptimizer
from pytensor.graph.rewriting.utils import rewrite_graph
from pytensor.graph.utils import MissingInputError
from pytensor.printing import debugprint
from pytensor.tensor.basic import constant
from pytensor.tensor.math import dot, exp, sigmoid
from pytensor.tensor.math import round as pt_round
from pytensor.tensor.math import sum as pt_sum
from pytensor.tensor.random.utils import RandomStream
from pytensor.tensor.rewriting.shape import ShapeOptimizer
from pytensor.tensor.shape import specify_shape
from pytensor.tensor.type import (
    dscalars,
    matrices,
    matrix,
    scalar,
    vector,
    vectors,
)
from tests import unittest_tools
from tests.graph.utils import MyVariable


class TestOpFromGraph(unittest_tools.InferShapeTester):
    def test_valid_input(self):
        x, _y, _z = matrices("xyz")

        with pytest.raises(ValueError, match=r"Expected at least.*"):
            OpFromGraph([x], [x])()

        with pytest.raises(ValueError, match=r"Expected 1 input\(s\)"):
            OpFromGraph([x], [x]).make_node()

        with pytest.raises(TypeError):
            OpFromGraph((x,), (x,))

        with pytest.raises(TypeError):
            OpFromGraph([1], [1])

        with pytest.raises(NotImplementedError):
            OpFromGraph([x], [x], updates={})

    def test_clone(self):
        x, _y, _z = matrices("xyz")

        ofg = OpFromGraph([x], [2 * x])

        ofg_clone = ofg.clone()

        assert ofg_clone.fgraph is not ofg.fgraph
        assert ofg_clone.fgraph.outputs != ofg.fgraph.outputs
        assert equal_computations(ofg_clone.fgraph.outputs, ofg.fgraph.outputs)

    @pytest.mark.parametrize(
        "cls_ofg", [OpFromGraph, partial(OpFromGraph, inline=True)]
    )
    def test_straightforward(self, cls_ofg):
        x, y, z = matrices("xyz")
        e = x + y * z
        op = cls_ofg([x, y, z], [e])
        # (1+3*5=array of 16) - (3+1*5=array of 8)
        f = op(x, y, z) - op(y, z, x)

        fn = function([x, y, z], f)
        xv = np.ones((2, 2), dtype=config.floatX)
        yv = np.ones((2, 2), dtype=config.floatX) * 3
        zv = np.ones((2, 2), dtype=config.floatX) * 5
        assert np.all(8.0 == fn(xv, yv, zv))
        assert np.all(8.0 == fn(xv, yv, zv))

    @pytest.mark.parametrize(
        "cls_ofg", [OpFromGraph, partial(OpFromGraph, inline=True)]
    )
    def test_size_changes(self, cls_ofg):
        x, y, z = matrices("xyz")
        e = dot(x, y)
        op = cls_ofg([x, y], [e])
        f = op(x, op(y, z))
        fn = function([x, y, z], f)
        xv = np.ones((2, 3), dtype=config.floatX)
        yv = np.ones((3, 4), dtype=config.floatX) * 3
        zv = np.ones((4, 5), dtype=config.floatX) * 5
        res = fn(xv, yv, zv)
        assert res.shape == (2, 5)
        assert np.all(180.0 == res)
        res = fn(xv, yv, zv)
        assert res.shape == (2, 5)
        assert np.all(180.0 == res)

    def test_overrides_deprecated_api(self):
        inp = scalar("x")
        out = inp + 1
        for kwarg in ("lop_overrides", "rop_overrides"):
            with pytest.warns(FutureWarning):
                OpFromGraph([inp], [out], **{kwarg: lambda *args: [inp + 1]})

    @pytest.mark.parametrize(
        "cls_ofg", [OpFromGraph, partial(OpFromGraph, inline=True)]
    )
    def test_grad(self, cls_ofg):
        x, y, z = matrices("xyz")
        e = x + y * z
        op = cls_ofg([x, y, z], [e])
        f = op(x, y, z)
        f = f - grad(pt_sum(f), y)
        fn = function([x, y, z], f)
        xv = np.ones((2, 2), dtype=config.floatX)
        yv = np.ones((2, 2), dtype=config.floatX) * 3
        zv = np.ones((2, 2), dtype=config.floatX) * 5
        assert np.all(11.0 == fn(xv, yv, zv))

    @pytest.mark.parametrize(
        "cls_ofg", [OpFromGraph, partial(OpFromGraph, inline=True)]
    )
    def test_grad_grad(self, cls_ofg):
        x, y, z = matrices("xyz")
        e = x + y * z
        op = cls_ofg([x, y, z], [e])
        f = op(x, y, z)
        f = f - grad(pt_sum(f), y)
        f = f - grad(pt_sum(f), y)

        fn = function([x, y, z], f)
        xv = np.ones((2, 2), dtype=config.floatX)
        yv = np.ones((2, 2), dtype=config.floatX) * 3
        zv = np.ones((2, 2), dtype=config.floatX) * 5
        np.testing.assert_array_almost_equal(6.0, fn(xv, yv, zv), 4)

    @pytest.mark.parametrize(
        "cls_ofg", [OpFromGraph, partial(OpFromGraph, inline=True)]
    )
    def test_shared(self, cls_ofg):
        x, y, z = matrices("xyz")
        s = shared(np.random.random((2, 2)).astype(config.floatX))
        e = x + y * z + s
        with pytest.warns(
            DeprecationWarning,
            match="Implicit capture of shared variables is deprecated",
        ):
            op = cls_ofg([x, y, z], [e])
        # (1+3*5=array of 16) - (3+1*5=array of 8)
        f = op(x, y, z) - op(y, z, x)

        fn = function([x, y, z], f)
        xv = np.ones((2, 2), dtype=config.floatX)
        yv = np.ones((2, 2), dtype=config.floatX) * 3
        zv = np.ones((2, 2), dtype=config.floatX) * 5
        # print function, function.__module__
        # print fn.maker.fgraph.toposort()
        np.testing.assert_array_almost_equal(8.0, fn(xv, yv, zv), 4)
        np.testing.assert_array_almost_equal(8.0, fn(xv, yv, zv), 4)

    @pytest.mark.parametrize(
        "cls_ofg", [OpFromGraph, partial(OpFromGraph, inline=True)]
    )
    def test_shared_grad(self, cls_ofg):
        x, y, z = matrices("xyz")
        s = shared(np.random.random((2, 2)).astype(config.floatX))
        e = x + y * z + s
        with pytest.warns(
            DeprecationWarning,
            match="Implicit capture of shared variables is deprecated",
        ):
            op = cls_ofg([x, y, z], [e])
        f = op(x, y, z)
        f = f - grad(pt_sum(f), y)
        fn = function([x, y, z], f)
        xv = np.ones((2, 2), dtype=config.floatX)
        yv = np.ones((2, 2), dtype=config.floatX) * 3
        zv = np.ones((2, 2), dtype=config.floatX) * 5
        np.testing.assert_array_almost_equal(11.0 + s.get_value(), fn(xv, yv, zv), 4)

        # grad again the shared variable
        f = op(x, y, z)
        f = f - grad(pt_sum(f), s)
        fn = function([x, y, z], f)
        np.testing.assert_array_almost_equal(15.0 + s.get_value(), fn(xv, yv, zv), 4)

    @pytest.mark.parametrize(
        "cls_ofg", [OpFromGraph, partial(OpFromGraph, inline=True)]
    )
    @pytest.mark.parametrize("use_deprecated_name", [False, True])
    def test_pullback_override(self, cls_ofg, use_deprecated_name):
        x = vector()
        y = 1.0 / (1.0 + exp(-x))

        def pullback_ov(inps, outs, grads):
            (y_,) = outs
            (dedy_,) = grads
            return [2.0 * y_ * (1.0 - y_) * dedy_]

        y_, dedy = vector(), vector()
        op_lop_ov = cls_ofg([x, y_, dedy], [2.0 * y_ * (1.0 - y_) * dedy])

        xx = vector()
        yy1 = pt_sum(sigmoid(xx))
        gyy1 = 2.0 * grad(yy1, xx)

        for ov in [pullback_ov, op_lop_ov]:
            if use_deprecated_name:
                with pytest.warns(FutureWarning, match="lop_overrides is deprecated"):
                    op = cls_ofg([x], [y], lop_overrides=ov)
            else:
                op = cls_ofg([x], [y], pullback=ov)
            yy2 = pt_sum(op(xx))
            gyy2 = grad(yy2, xx)
            fn = function([xx], [gyy1, gyy2])

            xval = np.random.random((32,)).astype(config.floatX)
            y1val, y2val = fn(xval)
            np.testing.assert_array_almost_equal(y1val, y2val, 4)

    @pytest.mark.parametrize(
        "cls_ofg", [OpFromGraph, partial(OpFromGraph, inline=True)]
    )
    @pytest.mark.parametrize("use_op_pushforward", [True, False])
    def test_pushforward(self, cls_ofg, use_op_pushforward):
        a = vector()
        M = matrix()
        b = dot(a, M)
        op_matmul = cls_ofg([a, M], [b])
        x = vector()
        W = matrix()
        y = op_matmul(x, W)
        du = vector()
        dv = pushforward(y, x, du, use_op_pushforward=use_op_pushforward)
        fn = function([x, W, du], dv)
        xval = np.random.random((16,)).astype(config.floatX)
        Wval = np.random.random((16, 16)).astype(config.floatX)
        duval = np.random.random((16,)).astype(config.floatX)
        dvval = np.dot(duval, Wval)
        dvval2 = fn(xval, Wval, duval)
        np.testing.assert_array_almost_equal(dvval2, dvval, 4)

    @pytest.mark.parametrize("use_op_pushforward", [True, False])
    def test_pushforward_multiple_outputs(self, use_op_pushforward):
        a = vector()
        M = matrix()
        b = dot(a, M)
        op_matmul = OpFromGraph([a, M], [b, -b])

        x = vector()
        W = matrix()
        du = vector()

        xval = np.random.random((16,)).astype(config.floatX)
        Wval = np.random.random((16, 16)).astype(config.floatX)
        duval = np.random.random((16,)).astype(config.floatX)

        y = op_matmul(x, W)[0]
        dv = pushforward(y, x, du, use_op_pushforward=use_op_pushforward)
        fn = function([x, W, du], dv)
        result_dvval = fn(xval, Wval, duval)
        expected_dvval = np.dot(duval, Wval)
        np.testing.assert_array_almost_equal(result_dvval, expected_dvval, 4)

        y = op_matmul(x, W)[1]
        dv = pushforward(y, x, du, use_op_pushforward=use_op_pushforward)
        fn = function([x, W, du], dv)
        result_dvval = fn(xval, Wval, duval)
        expected_dvval = -np.dot(duval, Wval)
        np.testing.assert_array_almost_equal(result_dvval, expected_dvval, 4)

        y = pt.add(*op_matmul(x, W))
        dv = pushforward(y, x, du, use_op_pushforward=use_op_pushforward)
        fn = function([x, W, du], dv)
        result_dvval = fn(xval, Wval, duval)
        expected_dvval = np.zeros_like(np.dot(duval, Wval))
        np.testing.assert_array_almost_equal(result_dvval, expected_dvval, 4)

    @pytest.mark.parametrize(
        "cls_ofg", [OpFromGraph, partial(OpFromGraph, inline=True)]
    )
    @pytest.mark.parametrize(
        "use_op_pushforward",
        [
            True,
            pytest.param(
                False, marks=pytest.mark.xfail(reason="Custom ROp is ignored")
            ),
        ],
    )
    @pytest.mark.parametrize("use_deprecated_name", [False, True])
    def test_pushforward_override(
        self, cls_ofg, use_op_pushforward, use_deprecated_name
    ):
        x, y = vectors("xy")

        def ro(inps, epts):
            x, y = inps
            u, v = epts
            return [u * y * 2.0 + x * v * 1.5]

        u, v = vectors("uv")
        op_mul_rop = cls_ofg([x, y, u, v], ro([x, y], [u, v]))
        if use_deprecated_name:
            with pytest.warns(FutureWarning, match="rop_overrides is deprecated"):
                op_mul = cls_ofg([x, y], [x * y], rop_overrides=ro)
                op_mul2 = cls_ofg([x, y], [x * y], rop_overrides=op_mul_rop)
        else:
            op_mul = cls_ofg([x, y], [x * y], pushforward=ro)
            op_mul2 = cls_ofg([x, y], [x * y], pushforward=op_mul_rop)

        # single override case
        xx, yy = vector("xx"), vector("yy")
        du, dv = vector("du"), vector("dv")
        for op in [op_mul, op_mul2]:
            zz = op_mul(xx, yy)
            dw = pushforward(
                zz,
                [xx, yy],
                [du, dv],
                use_op_pushforward=use_op_pushforward,
            )
            fn = function([xx, yy, du, dv], dw)
            vals = np.random.random((4, 32)).astype(config.floatX)
            dwval = fn(*vals)
            np.testing.assert_array_almost_equal(
                dwval, vals[0] * vals[3] * 1.5 + vals[1] * vals[2] * 2.0, 4
            )

        # TODO list override case

    @pytest.mark.parametrize(
        "cls_ofg", [OpFromGraph, partial(OpFromGraph, inline=True)]
    )
    def test_connection_pattern_override(self, cls_ofg):
        x, y = vectors("xy")

        def f1(x, y):
            del x
            # but we know how to backpropagate for x for some reasons
            # and we don't care about the gradient wrt y.
            return y + pt_round(y)

        def f1_back(inputs, outputs, output_gradients):
            return [output_gradients[0], disconnected_type()]

        op = cls_ofg(
            inputs=[x, y],
            outputs=[f1(x, y)],
            pullback=f1_back,
            connection_pattern=[[True], [False]],
            on_unused_input="ignore",
        )

        c = op(x, y)

        g1 = grad(c.sum(), x)

        out = g1.eval(
            {x: np.ones((5,), dtype=np.float32), y: np.ones((5,), dtype=np.float32)}
        )
        np.testing.assert_array_almost_equal(out, [1.0] * 5, 4)

    @pytest.mark.parametrize(
        "cls_ofg", [OpFromGraph, partial(OpFromGraph, inline=True)]
    )
    def test_nested(self, cls_ofg):
        x, y = vectors("xy")
        u, v = x + y, x - y
        op_ft = cls_ofg([x, y], [u, v])
        op_ift = cls_ofg([x, y], [u / 2, v / 2])

        xx, yy = vector("xx"), vector("yy")
        xx2, yy2 = op_ift(*op_ft(xx, yy))
        fn = function([xx, yy], [xx2, yy2])

        xv = np.random.random((16,)).astype(config.floatX)
        yv = np.random.random((16,)).astype(config.floatX)
        xv2, yv2 = fn(xv, yv)
        np.testing.assert_array_almost_equal(xv, xv2, 4)
        np.testing.assert_array_almost_equal(yv, yv2, 4)

    @pytest.mark.parametrize(
        "cls_ofg", [OpFromGraph, partial(OpFromGraph, inline=True)]
    )
    def test_connection_pattern(self, cls_ofg):
        # Basic case
        x, y, z = matrices("xyz")
        out1 = x * y
        out2 = y * z

        op1 = cls_ofg([x, y, z], [out1, out2])
        results = op1.connection_pattern(None)
        expect_result = [[True, False], [True, True], [False, True]]
        assert results == expect_result

        # Graph with ops that don't have a 'full' connection pattern
        # and with ops that have multiple outputs
        m, n, p, q = matrices("mnpq")
        o1, o2 = op1(m, n, p)
        out1, out2 = op1(o1, q, o2)
        op2 = cls_ofg([m, n, p, q], [out1, out2])

        results = op2.connection_pattern(None)
        expect_result = [[True, False], [True, True], [False, True], [True, True]]
        assert results == expect_result

        # Inner graph where some computation doesn't rely on explicit inputs
        srng = RandomStream(seed=234)
        rv_u = srng.uniform((2, 2))
        x, y = matrices("xy")
        out1 = x + rv_u
        out2 = y + 3
        out3 = 3 + rv_u
        with pytest.warns(
            DeprecationWarning,
            match="Implicit capture of shared variables is deprecated",
        ):
            op3 = cls_ofg([x, y], [out1, out2, out3])

        results = op3.connection_pattern(None)
        expect_result = [
            [True, False, False],
            [False, True, False],
            [True, False, True],
        ]
        assert results == expect_result

    def test_infer_shape(self):
        # test infer shape does not need to against inline case
        # since the Op is remove during optimization phase
        x = matrix("x")
        y = matrix("y")
        o1 = x + y
        o2 = x * y
        op_graph = OpFromGraph([x, y], [o1, o2])

        q = matrix("q")
        p = matrix("p")
        self._compile_and_check(
            [q, p],
            op_graph(q, p),
            [
                np.ones([3, 4], dtype=config.floatX),
                np.ones([3, 4], dtype=config.floatX),
            ],
            OpFromGraph,
        )

        # Make sure `OpFromGraph.infer_shape` can handle objects without a
        # shape
        x = MyVariable("x")
        y = matrix("y")
        z = specify_shape(vector("z"), (2,))

        op_graph = OpFromGraph([x, y, z], [x, y])

        op_var = op_graph(x, y, z)

        fg = FunctionGraph(outputs=[op_var[1]], clone=False)
        opt_res = rewrite_graph(fg, custom_rewrite=ShapeOptimizer())

        assert opt_res.shape_feature.shape_of[x] is None
        assert opt_res.shape_feature.shape_of[z][0].data == 2

    def test_make_node_shared(self):
        """Make sure we can provide `OpFromGraph.make_node` new shared inputs and get a valid `OpFromGraph`."""

        x = pt.scalar("x")
        y = shared(1.0, name="y")

        with pytest.warns(
            DeprecationWarning,
            match="Implicit capture of shared variables is deprecated",
        ):
            test_ofg = OpFromGraph([x], [x + y], on_unused_input="ignore")
        assert test_ofg.shared_inputs == [y]

        out = test_ofg(x)

        y_clone = y.clone()
        assert y_clone != y
        y_clone.name = "y_clone"

        with pytest.warns(
            DeprecationWarning,
            match="Implicit capture of shared variables is deprecated",
        ):
            out_new = test_ofg.make_node(*([*out.owner.inputs[:1], y_clone])).outputs[0]

        assert "on_unused_input" in out_new.owner.op.kwargs
        assert out_new.owner.op.shared_inputs == [y_clone]

        out_fn = function([x], out_new)
        assert np.array_equal(out_fn(1.0), 2.0)

        y_clone.set_value(2.0)
        assert np.array_equal(out_fn(1.0), 3.0)

        # This should also work, because the containers are the same:
        # y.set_value(1.0)
        # assert np.array_equal(out_fn(1.0), 2.0)

    def test_shared_with_constant_input(self):
        """Make sure that a constant input can be given to an `OpFromGraph` instance."""
        x = pt.scalar("x")
        y = shared(1.0, name="y")

        with pytest.warns(
            DeprecationWarning,
            match="Implicit capture of shared variables is deprecated",
        ):
            test_ofg = OpFromGraph([x], [x + y])
        assert test_ofg.shared_inputs == [y]

        out = test_ofg(pt.as_tensor(1.0, dtype=config.floatX))

        out_fn = function([], out)
        assert np.array_equal(out_fn(), 2.0)

    def test_missing_input(self):
        x = pt.lscalar("x")

        with pytest.raises(MissingInputError):
            OpFromGraph([], [x])

    def test_shared_to_nonshared_input(self):
        """Make sure that shared variables can be replaced with non-shared variables."""
        x = pt.scalar("x")
        y = shared(1.0, name="y")

        with pytest.warns(
            DeprecationWarning,
            match="Implicit capture of shared variables is deprecated",
        ):
            test_ofg = OpFromGraph([], [y])
        assert test_ofg.shared_inputs == [y]

        out_1_fn = function([], test_ofg())
        res_1 = out_1_fn()

        assert np.array_equal(res_1, 1.0)

        test_ofg_new = test_ofg.make_node(x)
        assert test_ofg_new.op.shared_inputs == []

        out_2_fn = function([x], test_ofg_new.outputs[0])
        res_2 = out_2_fn(np.array(1.0, dtype=config.floatX))

        assert np.array_equal(res_2, 1.0)

    def test_outputs_consistency(self):
        """Make sure that `OpFromGraph.fn` doesn't change the value of `OpFromGraph.inner_outputs`."""

        x = scalar("x")
        op = OpFromGraph([x], [x**2 / x], mode="FAST_RUN")

        # Confirm that the inner-graph is as expected
        assert equal_computations(op.inner_outputs, [x**2 / x], op.inner_inputs, [x])

        # These outputs of the compiled `op.fgraph` should differ from the
        # original, uncompiled `op.fgraph` outputs
        fn = op.fn
        new_inputs = fn.maker.fgraph.inputs
        new_outputs = fn.maker.fgraph.outputs
        assert not equal_computations(new_outputs, [x**2 / x], new_inputs, [x])

        # The original `op.fgraph` outputs should stay the same, though
        assert equal_computations(op.inner_outputs, [x**2 / x], op.inner_inputs, [x])

    def test_explicit_input_from_constant(self):
        x = pt.dscalar("x")
        y = constant(1.0, dtype=x.type.dtype, name="y")
        test_ofg = OpFromGraph([x, y], [x + y])

        out = test_ofg(x, y)
        assert out.eval({x: 5}) == 6

        out = test_ofg(x, x)
        assert out.eval({x: 5}) == 10

    def test_explicit_input_from_shared(self):
        x = pt.dscalar("x")
        y = shared(1.0, name="y")

        with pytest.raises(
            ValueError,
            match=r"The inner-graph implicitly depends on the following shared variables \[y\]",
        ):
            OpFromGraph([x], [x + y], strict=True)

        test_ofg = OpFromGraph([x, y], [x + y], strict=True)

        out = test_ofg(x, y)
        assert out.eval({x: 5}) == 6
        y.set_value(2.0)
        assert out.eval({x: 6}) == 8

        out = test_ofg(y, y)
        assert out.eval() == 4

    def test_implicit_shared_inputs_deprecated(self):
        x = pt.dscalar("x")
        y = shared(1.0, name="y")

        with pytest.warns(
            DeprecationWarning,
            match="Implicit capture of shared variables is deprecated",
        ):
            OpFromGraph([x], [x + y])

    @pytest.mark.parametrize("use_custom_pullback", [False, True])
    def test_pullback_disconnected_output_grad(self, use_custom_pullback):
        x, y = dscalars("x", "y")
        rng = np.random.default_rng(594)
        point = list(rng.normal(size=(2,)))

        out1 = x + y
        out2 = x * y
        out3 = out1 * out2  # Create dependency between outputs

        if use_custom_pullback:
            # Regression test for https://github.com/pymc-devs/pytensor/issues/2064
            def pullback_3out(inputs, outputs, output_grads):
                x_, y_ = inputs
                dout1, dout2, dout3 = output_grads
                dx = pt.zeros_like(x_)
                dy = pt.zeros_like(y_)
                if not isinstance(dout1.type, DisconnectedType):
                    dx = dx + dout1
                    dy = dy + dout1
                if not isinstance(dout2.type, DisconnectedType):
                    dx = dx + dout2 * y_
                    dy = dy + dout2 * x_
                if not isinstance(dout3.type, DisconnectedType):
                    dx = dx + dout3 * (2 * x_ * y_ + y_**2)
                    dy = dy + dout3 * (x_**2 + 2 * x_ * y_)
                return [dx, dy]

            op = OpFromGraph([x, y], [out1, out2, out3], pullback=pullback_3out)
        else:
            op = OpFromGraph([x, y], [out1, out2, out3])

        verify_grad(lambda x, y: pt.add(*op(x, y)), point, rng=rng)
        verify_grad(lambda x, y: pt.add(*op(x, y)[:-1]), point, rng=rng)
        verify_grad(lambda x, y: pt.add(*op(x, y)[1:]), point, rng=rng)
        verify_grad(lambda x, y: pt.add(*op(x, y)[::2]), point, rng=rng)
        verify_grad(lambda x, y: op(x, y)[0], point, rng=rng)
        verify_grad(lambda x, y: op(x, y)[1], point, rng=rng)
        verify_grad(lambda x, y: op(x, y)[2], point, rng=rng)

        # Two fully-independent outputs: x**2 only depends on x, y**3 only on y.
        # Each (output, wrt) combination exercises a distinct disconnection pattern.
        if use_custom_pullback:

            def pullback_2out(inputs, outputs, output_grads):
                x_, y_ = inputs
                dout0, dout1 = output_grads
                dx = (
                    disconnected_type()
                    if isinstance(dout0.type, DisconnectedType)
                    else dout0 * 2 * x_
                )
                dy = (
                    disconnected_type()
                    if isinstance(dout1.type, DisconnectedType)
                    else dout1 * 3 * y_**2
                )
                return [dx, dy]

            op = OpFromGraph([x, y], [x**2, y**3], pullback=pullback_2out)
        else:
            op = OpFromGraph([x, y], [x**2, y**3])

        # Cross terms are disconnected and must resolve to DisconnectedType with a warning.
        with pytest.warns(UserWarning):
            grad_out0_wrt_y = grad(
                op(x, y)[0],
                wrt=y,
                return_disconnected="disconnected",
                disconnected_inputs="warn",
            )
            assert isinstance(grad_out0_wrt_y.type, DisconnectedType)

        with pytest.warns(UserWarning):
            grad_out1_wrt_x = grad(
                op(x, y)[1],
                wrt=x,
                return_disconnected="disconnected",
                disconnected_inputs="warn",
            )
            assert isinstance(grad_out1_wrt_x.type, DisconnectedType)

        # Related terms stay connected and must return the correct numerical gradient
        # even though the sibling output is entirely disconnected from the cost.
        fn_dx = function([x, y], grad(op(x, y)[0], wrt=x))
        np.testing.assert_allclose(fn_dx(3.0, 2.0), 6.0)  # d/dx x**2 = 2x

        fn_dy = function([x, y], grad(op(x, y)[1], wrt=y))
        np.testing.assert_allclose(fn_dy(3.0, 2.0), 12.0)  # d/dy y**3 = 3y**2

    def test_repeated_inputs(self):
        x = pt.dscalar("x")
        y = pt.dscalar("y")

        with pytest.raises(
            ValueError,
            match="The following variables were provided more than once as inputs to the "
            "OpFromGraph",
        ):
            OpFromGraph([x, x, y], [x + y])

        # Test that repeated inputs will be allowed if unused inputs are ignored
        g = OpFromGraph([x, x, y], [x + y], on_unused_input="ignore")
        f = g(x, x, y)
        assert f.eval({x: 5, y: 5}) == 10

    def test_equality_and_hashing(self):
        x, y = dscalars("x", "y")
        e = x + y * x

        op1 = OpFromGraph([x, y], [e])
        op2 = OpFromGraph([x, y], [e])

        # Same output with same inputs are equal with consistent hash
        assert op1 == op2
        assert hash(op1) == hash(op2)
        assert {op1: "v"}[op2] == "v"

        # Distinct variables with the same graph structure are equal
        a, b = dscalars("a", "b")
        op3 = OpFromGraph([a, b], [a + b * a])
        assert op1 == op3
        assert hash(op1) == hash(op3)

        # Different graphs are not equal
        op_different = OpFromGraph([x, y], [x * y + x])
        assert op1 != op_different

        # inline flag participates in equality
        op_inline = OpFromGraph([x, y], [e], inline=True)
        assert op1 != op_inline

        # destroy_map participates in equality
        op_destroy = OpFromGraph([x, y], [e], destroy_map={0: (0,)})
        assert op1 != op_destroy

        # Multi-output OFGs are also hashed and compared based on their inner graph structure
        op_multi1 = OpFromGraph([x, y], [x + y, x * y])
        op_multi2 = OpFromGraph([a, b], [a + b, a * b])
        assert op_multi1 == op_multi2

        # OFG is hashable, and different OFGs have different hashes
        assert hash(op1) != hash(op_inline)

    def test_equality_shared_variables(self):
        x = scalar("x")
        s = shared(np.array(1.0, dtype=config.floatX))

        with pytest.warns(
            DeprecationWarning,
            match="Implicit capture of shared variables is deprecated",
        ):
            op1 = OpFromGraph([x], [x + s])
        with pytest.warns(
            DeprecationWarning,
            match="Implicit capture of shared variables is deprecated",
        ):
            op2 = OpFromGraph([x], [x + s])
        assert op1 == op2

        # Same value, different shared object -> not equal
        s2 = shared(np.array(1.0, dtype=config.floatX))
        with pytest.warns(
            DeprecationWarning,
            match="Implicit capture of shared variables is deprecated",
        ):
            op3 = OpFromGraph([x], [x + s2])
        assert op1 != op3

    def test_equality_callable_overrides(self):
        x, y = dscalars("x", "y")
        e = x + y

        op_plain = OpFromGraph([x, y], [e])

        # lop override present vs absent
        op_with_lop = OpFromGraph(
            [x, y],
            [e],
            pullback=lambda inps, outs, grads: [grads[0], grads[0]],
        )
        assert op_plain != op_with_lop

        # Structurally identical callable overrides are equal
        op_with_lop2 = OpFromGraph(
            [x, y],
            [e],
            pullback=lambda inps, outs, grads: [grads[0], grads[0]],
        )
        assert op_with_lop == op_with_lop2

        # Structurally different callable override are not equal
        op_with_lop3 = OpFromGraph(
            [x, y],
            [e],
            pullback=lambda inps, outs, grads: [grads[0] * 2, grads[0]],
        )
        assert op_with_lop != op_with_lop3

        # Overrides returning disconnected_type for different inputs are not equal
        op_disc_y = OpFromGraph(
            [x, y],
            [e],
            pullback=lambda inps, outs, grads: [grads[0], disconnected_type()],
        )
        op_disc_x = OpFromGraph(
            [x, y],
            [e],
            pullback=lambda inps, outs, grads: [disconnected_type(), grads[0]],
        )
        assert op_disc_y != op_disc_x

        # Same disconnected pattern is equal
        op_disc_y2 = OpFromGraph(
            [x, y],
            [e],
            pullback=lambda inps, outs, grads: [grads[0], disconnected_type()],
        )
        assert op_disc_y == op_disc_y2

        # All disconnected is still an override — not equal to no override
        op_all_disc = OpFromGraph(
            [x, y],
            [e],
            pullback=lambda inps, outs, grads: [
                disconnected_type(),
                disconnected_type(),
            ],
        )
        assert op_all_disc != op_plain
        assert op_all_disc != op_disc_y

        # rop override follows the same logic
        op_with_rop = OpFromGraph(
            [x, y],
            [e],
            pushforward=lambda inps, epts: [epts[0] + epts[1]],
        )
        op_with_rop2 = OpFromGraph(
            [x, y],
            [e],
            pushforward=lambda inps, epts: [epts[0] + epts[1]],
        )
        assert op_with_rop == op_with_rop2
        assert op_with_rop != op_plain

    def test_equality_list_overrides(self):
        x, y = dscalars("x", "y")
        e = x + y

        def scale_grad(inps, outs, grads):
            return grads[0] * 2

        op1 = OpFromGraph([x, y], [e], pullback=[scale_grad, None])
        op2 = OpFromGraph([x, y], [e], pullback=[scale_grad, None])
        assert op1 == op2

        def scale_grad_3x(inps, outs, grads):
            return grads[0] * 3

        op3 = OpFromGraph([x, y], [e], pullback=[scale_grad_3x, None])
        assert op1 != op3

        # Position of None vs callable matters
        op4 = OpFromGraph([x, y], [e], pullback=[None, scale_grad])
        assert op1 != op4

    def test_merge_identical_ofgs(self):
        x, y = dscalars("x", "y")
        e = x + y * x

        op1 = OpFromGraph([x, y], [e])
        op2 = OpFromGraph([x, y], [e])

        a, b = dscalars("a", "b")

        # Two OFG with the same inputs are collapsed to one node by MergeOptimizer
        fg = FunctionGraph([a, b], [op1(a, b), op2(a, b)])
        MergeOptimizer().rewrite(fg)
        ofg_nodes = [n for n in fg.toposort() if isinstance(n.op, OpFromGraph)]
        assert len(ofg_nodes) == 1

        # Different inputs are different graphs, so both nodes survive
        c, d = dscalars("c", "d")
        fg = FunctionGraph([a, b, c, d], [op1(a, b), op2(c, d)])
        MergeOptimizer().rewrite(fg)
        ofg_nodes = [n for n in fg.toposort() if isinstance(n.op, OpFromGraph)]
        assert len(ofg_nodes) == 2

        # Check numerics to make sure the merged OFG is correct
        fn = function(
            [a, b, c, d],
            [op1(a, b), op2(c, d)],
            mode=Mode(optimizer="merge", linker="py"),
        )
        r1, r2 = fn(2.0, 3.0, 4.0, 5.0)
        np.testing.assert_allclose(r1, 2.0 + 3.0 * 2.0)
        np.testing.assert_allclose(r2, 4.0 + 5.0 * 4.0)


@config.change_flags(floatX="float64")
def test_debugprint():
    x, y, z = matrices("xyz")
    e = x + y * z
    op = OpFromGraph([x, y, z], [e])
    out = op(x, y, z)

    output_str = debugprint(out, file="str")
    lines = output_str.split("\n")

    exp_res = """OpFromGraph{inline=False} [id A]
 ├─ x [id B]
 ├─ y [id C]
 └─ z [id D]

Inner graphs:

OpFromGraph{inline=False} [id A]
 ← Add [id E]
    ├─ i0 [id F]
    └─ Mul [id G]
       ├─ i1 [id H]
       └─ i2 [id I]
"""

    for truth, out in zip(exp_res.split("\n"), lines, strict=True):
        assert truth.strip() == out.strip()
