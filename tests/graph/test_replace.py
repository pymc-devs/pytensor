import numpy as np
import pytest
import scipy.special

import pytensor.tensor as pt
from pytensor import config, function, shared
from pytensor.graph.basic import equal_computations
from pytensor.graph.replace import (
    _vectorize_node,
    clone_replace,
    graph_replace,
    vectorize_graph,
)
from pytensor.graph.traversal import graph_inputs
from pytensor.tensor import dvector, fvector, vector
from tests import unittest_tools as utt
from tests.graph.utils import MyOp, MyVariable, op_multiple_outputs


class TestCloneReplace:
    def test_cloning_no_replace_strict_copy_inputs(self):
        # This has nothing to do with scan, but it refers to the clone
        # function that scan uses internally and that pfunc uses now and
        # that users might want to use
        x = vector("x")
        y = vector("y")
        z = shared(0.25)

        f1 = z * (x + y) ** 2 + 5
        f2 = clone_replace(f1, replace=None, rebuild_strict=True, copy_inputs_over=True)
        f2_inp = tuple(graph_inputs([f2]))

        assert z in f2_inp
        assert x in f2_inp
        assert y in f2_inp

    def test_cloning_no_replace_strict_not_copy_inputs(self):
        # This has nothing to do with scan, but it refers to the clone
        # function that scan uses internally and that pfunc uses now and
        # that users might want to use
        x = vector("x")
        y = vector("y")
        z = shared(0.25)

        f1 = z * (x + y) ** 2 + 5
        f2 = clone_replace(
            f1, replace=None, rebuild_strict=True, copy_inputs_over=False
        )
        f2_inp = graph_inputs([f2])

        assert z not in f2_inp
        assert x not in f2_inp
        assert y not in f2_inp

    def test_cloning_replace_strict_copy_inputs(self):
        # This has nothing to do with scan, but it refers to the clone
        # function that scan uses internally and that pfunc uses now and
        # that users might want to use
        x = vector("x")
        y = vector("y")
        y2 = vector("y2")
        z = shared(0.25)

        f1 = z * (x + y) ** 2 + 5
        f2 = clone_replace(
            f1, replace={y: y2}, rebuild_strict=True, copy_inputs_over=True
        )
        f2_inp = tuple(graph_inputs([f2]))
        assert z in f2_inp
        assert x in f2_inp
        assert y2 in f2_inp

    def test_cloning_replace_not_strict_copy_inputs(self):
        # This has nothing to do with scan, but it refers to the clone
        # function that scan uses internally and that pfunc uses now and
        # that users might want to use
        x = vector("x")
        y = fvector("y")
        y2 = dvector("y2")
        z = shared(0.25)

        f1 = z * (x + y) ** 2 + 5
        f2 = clone_replace(
            f1, replace={y: y2}, rebuild_strict=False, copy_inputs_over=True
        )
        f2_inp = tuple(graph_inputs([f2]))
        assert z in f2_inp
        assert x in f2_inp
        assert y2 in f2_inp

    def test_cloning_replace_strict_not_copy_inputs(self):
        # This has nothing to do with scan, but it refers to the clone
        # function that scan uses internally and that pfunc uses now and
        # that users might want to use
        x = vector("x")
        y = vector("y")
        y2 = vector("y2")
        z = shared(0.25)

        f1 = z * (x + y) ** 2 + 5
        f2 = clone_replace(
            f1, replace=[(y, y2)], rebuild_strict=True, copy_inputs_over=False
        )
        f2_inp = graph_inputs([f2])
        assert z not in f2_inp
        assert x not in f2_inp
        assert y2 not in f2_inp

    def test_cloning_replace_not_strict_not_copy_inputs(self):
        # This has nothing to do with scan, but it refers to the clone
        # function that scan uses internally and that pfunc uses now and
        # that users might want to use
        x = vector("x")
        y = fvector("y")
        y2 = dvector("y2")
        z = shared(0.25)

        f1 = z * (x + y) ** 2 + 5
        f2 = clone_replace(
            f1, replace=[(y, y2)], rebuild_strict=False, copy_inputs_over=False
        )
        f2_inp = graph_inputs([f2])
        assert z not in f2_inp
        assert x not in f2_inp
        assert y2 not in f2_inp

    def test_clone(self):
        def test(x, y, mention_y):
            if mention_y:
                d = 0.1 + 0 * y
            else:
                d = 0.1
            out = clone_replace(y, replace={x: x + d})
            return function([], out)()

        x = shared(np.asarray(0.0, dtype=config.floatX))
        utt.assert_allclose(
            test(x, pt.sum((x + 1) ** 2), mention_y=False), 1.21000003815
        )
        utt.assert_allclose(
            test(x, pt.sum((x + 1) ** 2), mention_y=True), 1.21000003815
        )


class TestGraphReplace:
    def test_graph_replace(self):
        op = MyOp("op")
        x = MyVariable("x")
        y = MyVariable("y")
        z = MyVariable("w")
        out = op(x, z)

        new_x = op(y)
        new_out = graph_replace([out], {x: new_x})[0]
        assert new_out.owner.inputs[0] is new_x
        # the old reference is kept
        assert new_out.owner.inputs[1] is z

        # test replace itself
        new_out = graph_replace([out], {out: new_x})[0]
        assert new_out is new_x
        assert new_out.owner.inputs[0] is y
        assert len(new_out.owner.inputs) == 1

        # the case where inputs have to be replaced in reverse topological order
        out2 = op(out, new_x)

        new_x2 = x.clone(name="new_x")
        new_x22 = new_x.clone(name="new_x2")
        new_out2 = graph_replace([out2], {x: new_x2, new_x: new_x22})[0]
        assert new_out2.owner.inputs[1] is new_x22
        assert new_out2.owner.inputs[0].owner.inputs[0] is new_x2
        # the old reference is still kept
        assert new_out2.owner.inputs[0].owner.inputs[1] is z

    def test_non_list_input(self):
        op = MyOp("op")
        x = MyVariable("x")
        y = MyVariable("y")
        out = op(x, y)

        new_x = x.clone(name="new_x")
        new_y = y.clone(name="new_y")
        # test non list inputs as well
        oc = graph_replace(out, {x: new_x, y: new_y})
        assert oc.owner.inputs[1] is new_y
        assert oc.owner.inputs[0] is new_x

    def test_graph_replace_advanced(self):
        op = MyOp("op")
        x = MyVariable("x")
        y = MyVariable("y")
        z = MyVariable("z")
        w = MyVariable("w")

        z_op = op(z)
        xw_op = op(x, w)
        y_op = op(y)
        out = op(xw_op, y_op)

        new_x = x.clone(name="new_x")
        new_yop = y_op.clone(name="new_yop")

        # now yet another replacement that could only appear after new_y2: z
        # show we can do that after the prev clone
        # the case where new variable is referenced during the replacements
        new_yop_op = op(new_yop)
        # the reference new_yop: z_op is not a part of the original graph so the replacement is unsafe
        new_out = graph_replace([out], {x: new_x, y_op: new_yop_op})
        new_out = graph_replace(new_out, {new_yop: z_op})[0]
        assert new_out.owner.inputs[1].owner.inputs[0] is z_op
        assert new_out.owner.inputs[0].owner.inputs[0] is new_x
        # the old reference is still kept
        assert new_out.owner.inputs[0].owner.inputs[1] is w

        new_z = z.clone(name="new_z")
        new_out = graph_replace([new_out], {z: new_z})[0]
        # new reference appear
        assert new_out.owner.inputs[1].owner.inputs[0] is not z_op
        assert new_out.owner.inputs[1].owner.inputs[0].owner.inputs[0] is new_z
        # the old reference is still kept
        assert new_out.owner.inputs[0].owner.inputs[0] is new_x
        assert new_out.owner.inputs[0].owner.inputs[1] is w

    def test_graph_replace_disconnected(self):
        op = MyOp("op")
        fake_op = MyOp("fake_op")
        x = MyVariable("x")
        fake = fake_op(x)
        out = op(x)
        [new_out] = graph_replace([out], {fake: x.clone()}, strict=False)
        assert new_out is out
        with pytest.raises(ValueError, match="Some replacements were not used"):
            graph_replace([out], {fake: x.clone()}, strict=True)


class TestVectorizeGraph:
    def test_basic(self):
        x = pt.vector("x")
        y = pt.exp(x) / pt.sum(pt.exp(x))

        new_x = pt.matrix("new_x")
        [new_y] = vectorize_graph([y], {x: new_x})

        # Check we can pass both a sequence or a single variable
        alt_new_y = vectorize_graph(y, {x: new_x})
        assert equal_computations([new_y], [alt_new_y])

        fn = function([new_x], new_y)
        test_new_y = np.array([[0, 1, 2], [2, 1, 0]]).astype(config.floatX)
        np.testing.assert_allclose(
            fn(test_new_y),
            scipy.special.softmax(test_new_y, axis=-1),
        )

    def test_multiple_outputs(self):
        x = pt.vector("x")
        y1 = x[0]
        y2 = x[-1]

        new_x = pt.matrix("new_x")
        [new_y1, new_y2] = vectorize_graph([y1, y2], {x: new_x})

        fn = function([new_x], [new_y1, new_y2])
        new_x_test = np.arange(9).reshape(3, 3).astype(config.floatX)
        new_y1_res, new_y2_res = fn(new_x_test)
        np.testing.assert_allclose(new_y1_res, [0, 3, 6])
        np.testing.assert_allclose(new_y2_res, [2, 5, 8])

    def test_multi_output_node(self):
        x = pt.scalar("x")
        node = op_multiple_outputs.make_node(x)
        y1, y2 = node.outputs
        out = pt.add(y1, y2)

        new_x = pt.vector("new_x")
        new_y1 = pt.vector("new_y1")
        new_y2 = pt.vector("new_y2")

        # Cases where either x or both of y1 and y2 are given replacements
        new_out = vectorize_graph(out, {x: new_x})
        expected_new_out = pt.add(*_vectorize_node(node.op, node, new_x).outputs)
        assert equal_computations([new_out], [expected_new_out])

        new_out = vectorize_graph(out, {y1: new_y1, y2: new_y2})
        expected_new_out = pt.add(new_y1, new_y2)
        assert equal_computations([new_out], [expected_new_out])

        new_out = vectorize_graph(out, {x: new_x, y1: new_y1, y2: new_y2})
        expected_new_out = pt.add(new_y1, new_y2)
        assert equal_computations([new_out], [expected_new_out])

        # Special case where x is given a replacement as well as only one of y1 and y2
        # The graph combines the replaced variable with the other vectorized output
        new_out = vectorize_graph(out, {x: new_x, y1: new_y1})
        expected_new_out = pt.add(
            new_y1, _vectorize_node(node.op, node, new_x).outputs[1]
        )
        assert equal_computations([new_out], [expected_new_out])

    def test_multi_output_node_random_variable(self):
        """This is a regression test for #569.

        Functionally, it covers the same case as `test_multiple_output_node`
        """

        # RandomVariables have two outputs, a hidden RNG and the visible draws
        beta0 = pt.random.normal(name="beta0")
        beta1 = pt.random.normal(name="beta1")

        out1 = beta0 + 1
        out2 = beta1 * pt.exp(out1)

        # We replace the second output of each RandomVariable
        new_beta0 = pt.tensor("new_beta0", shape=(3,))
        new_beta1 = pt.tensor("new_beta1", shape=(3,))

        new_outs = vectorize_graph(
            [out1, out2],
            replace={
                beta0: new_beta0,
                beta1: new_beta1,
            },
        )

        expected_new_outs = [
            new_beta0 + 1,
            new_beta1 * pt.exp(new_beta0 + 1),
        ]
        assert equal_computations(new_outs, expected_new_outs)

    def test_non_variable_raises(self):
        x = pt.scalar("x", dtype=int)
        y = pt.scalar("y", dtype=int)
        non_variable_shape = (x, y)
        variable_shape = pt.as_tensor(non_variable_shape)

        non_variable_shape_out = pt.zeros(non_variable_shape)
        variable_shape_out = pt.zeros(variable_shape)

        non_variable_batch_shape = (non_variable_shape, non_variable_shape)
        variable_batch_shape = pt.stacklists(non_variable_batch_shape)

        msg = r"Some of the replaced items are not Variables"
        with pytest.raises(ValueError, match=msg):
            vectorize_graph(
                non_variable_shape_out, {non_variable_shape: non_variable_batch_shape}
            )

        with pytest.raises(ValueError, match=msg):
            vectorize_graph(
                variable_shape_out, {variable_shape: non_variable_batch_shape}
            )

        batch_out = vectorize_graph(
            variable_shape_out, {variable_shape: variable_batch_shape}
        )
        assert batch_out.type.shape == (2, None, None)
        np.testing.assert_array_equal(
            batch_out.eval({x: 3, y: 4}),
            np.zeros((2, 3, 4)),
        )
