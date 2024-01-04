import numpy as np
import pytest
import scipy.special

import pytensor.tensor as pt
from pytensor import config, function, shared
from pytensor.graph.basic import equal_computations, graph_inputs
from pytensor.graph.replace import (
    clone_replace,
    graph_replace,
    vectorize_graph,
    vectorize_node,
)
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
        f2_inp = graph_inputs([f2])

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
        f2_inp = graph_inputs([f2])
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
        f2_inp = graph_inputs([f2])
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
        x = MyVariable("x")
        y = MyVariable("y")
        z = MyVariable("z")
        w = MyVariable("w")
        MyOp("zop")(z)
        x2 = MyOp("xop")(x, w)
        x2.name = "x2"
        y2 = MyOp("yop")(y)
        y2.name = "y2"

        yc = graph_replace([x2], {x: y2})[0]
        assert yc.owner.inputs[0] is y2
        # the old reference is kept
        assert yc.owner.inputs[1] is w

        # test replace itself
        yc = graph_replace([x2], {x2: y2})[0]
        assert yc is y2
        assert yc.owner.inputs[0] is y
        assert len(yc.owner.inputs) == 1

        # the case where inputs have to be replaced in reverse topological order
        o = MyOp("xyop")(x2, y2)
        new_x = x.clone(name="x_new")
        new_y2 = y2.clone(name="y2_new")

        oc = graph_replace([o], {x: new_x, y2: new_y2})[0]
        assert oc.owner.inputs[1] is new_y2
        assert oc.owner.inputs[0].owner.inputs[0] is new_x
        # the old reference is still kept
        assert oc.owner.inputs[0].owner.inputs[1] is w

    def test_non_list_input(self):
        x = MyVariable("x")
        y = MyVariable("y")
        o = MyOp("xyop")(x, y)
        new_x = x.clone(name="x_new")
        new_y = y.clone(name="y2_new")
        # test non list inputs as well
        oc = graph_replace(o, {x: new_x, y: new_y})
        assert oc.owner.inputs[1] is new_y
        assert oc.owner.inputs[0] is new_x

    def test_graph_replace_advanced(self):
        x = MyVariable("x")
        y = MyVariable("y")
        z = MyVariable("z")
        w = MyVariable("w")
        z2 = MyOp("zop")(z)
        x2 = MyOp("xop")(x, w)
        x2.name = "x2"
        y2 = MyOp("yop")(y)
        y2.name = "y2"
        o = MyOp("xyop")(x2, y2)
        new_x = x.clone(name="x_new")
        new_y2 = y2.clone(name="y2_new")
        new_y21 = MyOp("ny2op")(new_y2)
        # now yet another replacement that could only appear after new_y2: z
        # show we can do that after the prev clone
        # the case where new variable is referenced during the replacements
        new_y21 = MyOp("ny2op")(new_y2)
        # the reference new_y2: z2 is not a part of the original graph so the replacement is unsafe
        oc = graph_replace([o], {x: new_x, y2: new_y21})
        oc = graph_replace(oc, {new_y2: z2})[0]
        assert oc.owner.inputs[1].owner.inputs[0] is z2
        assert oc.owner.inputs[0].owner.inputs[0] is new_x
        # the old reference is still kept
        assert oc.owner.inputs[0].owner.inputs[1] is w

        new_z = z.clone(name="z_new")
        oc = graph_replace([oc], {z: new_z})[0]
        # new reference appear
        assert oc.owner.inputs[1].owner.inputs[0] is not z2
        assert oc.owner.inputs[1].owner.inputs[0].owner.inputs[0] is new_z
        # the old reference is still kept
        assert oc.owner.inputs[0].owner.inputs[0] is new_x
        assert oc.owner.inputs[0].owner.inputs[1] is w

    def test_graph_replace_disconnected(self):
        x = MyVariable("x")
        fake = MyOp("fake")(x)
        o = MyOp("o")(x)
        oc = graph_replace([o], {fake: x.clone()}, strict=False)
        assert oc[0] is o
        with pytest.raises(ValueError, match="Some replacements were not used"):
            oc = graph_replace([o], {fake: x.clone()}, strict=True)


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
        expected_new_out = pt.add(*vectorize_node(node, new_x).outputs)
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
        expected_new_out = pt.add(new_y1, vectorize_node(node, new_x).outputs[1])
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
