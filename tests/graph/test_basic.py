import pickle
from itertools import count

import numpy as np
import pytest

from pytensor import shared
from pytensor import tensor as pt
from pytensor.compile import UnusedInputError
from pytensor.graph.basic import (
    Apply,
    NominalVariable,
    Variable,
    ancestors,
    apply_depends_on,
    applys_between,
    as_string,
    clone,
    clone_get_equiv,
    equal_computations,
    explicit_graph_inputs,
    general_toposort,
    get_var_by_name,
    graph_inputs,
    io_toposort,
    orphans_between,
    truncated_graph_inputs,
    variable_depends_on,
    vars_between,
    walk,
)
from pytensor.graph.op import Op
from pytensor.graph.type import Type
from pytensor.printing import debugprint
from pytensor.tensor import constant
from pytensor.tensor.math import max_and_argmax
from pytensor.tensor.type import TensorType, iscalars, matrix, scalars, vector
from pytensor.tensor.type_other import NoneConst
from pytensor.tensor.variable import TensorVariable
from tests.graph.utils import MyInnerGraphOp, op_multiple_outputs


class MyType(Type):
    def __init__(self, thingy):
        self.thingy = thingy

    def filter(self, *args, **kwargs):
        raise NotImplementedError()

    def __eq__(self, other):
        return isinstance(other, MyType) and other.thingy == self.thingy

    def __hash__(self):
        return hash((type(self), self.thingy))

    def __str__(self):
        return f"R{self.thingy}"

    def __repr__(self):
        return f"R{self.thingy}"


def MyVariable(thingy):
    return Variable(MyType(thingy), None, None)


class MyOp(Op):
    __props__ = ()

    def make_node(self, *inputs):
        for input in inputs:
            assert isinstance(input, Variable)
            assert isinstance(input.type, MyType)
        outputs = [MyVariable(sum(input.type.thingy for input in inputs))]
        return Apply(self, list(inputs), outputs)

    def perform(self, *args, **kwargs):
        raise NotImplementedError("No Python implementation available.")


MyOp = MyOp()


class X:
    def leaf_formatter(self, leaf):
        return str(leaf.type)

    def node_formatter(self, node, argstrings):
        return f"{node.op}({', '.join(argstrings)})"

    def str(self, inputs, outputs):
        return as_string(
            inputs,
            outputs,
            leaf_formatter=self.leaf_formatter,
            node_formatter=self.node_formatter,
        )


class TestStr(X):
    def test_as_string(self):
        r1, r2 = MyVariable(1), MyVariable(2)
        node = MyOp.make_node(r1, r2)
        s = self.str([r1, r2], node.outputs)
        assert s == ["MyOp(R1, R2)"]

    def test_as_string_deep(self):
        r1, r2, r5 = MyVariable(1), MyVariable(2), MyVariable(5)
        node = MyOp.make_node(r1, r2)
        node2 = MyOp.make_node(node.outputs[0], r5)
        s = self.str([r1, r2, r5], node2.outputs)
        assert s == ["MyOp(MyOp(R1, R2), R5)"]

    def test_multiple_references(self):
        r1, r2, r5 = MyVariable(1), MyVariable(2), MyVariable(5)
        node = MyOp.make_node(r1, r2)
        node2 = MyOp.make_node(node.outputs[0], node.outputs[0])
        assert self.str([r1, r2, r5], node2.outputs) == ["MyOp(*1 -> MyOp(R1, R2), *1)"]

    def test_cutoff(self):
        r1, r2 = MyVariable(1), MyVariable(2)
        node = MyOp.make_node(r1, r2)
        node2 = MyOp.make_node(node.outputs[0], node.outputs[0])
        assert self.str(node.outputs, node2.outputs) == ["MyOp(R3, R3)"]
        assert self.str(node2.inputs, node2.outputs) == ["MyOp(R3, R3)"]


class TestClone(X):
    def test_accurate(self):
        r1, r2 = MyVariable(1), MyVariable(2)
        node = MyOp.make_node(r1, r2)
        _, new = clone([r1, r2], node.outputs, False)
        assert self.str([r1, r2], new) == ["MyOp(R1, R2)"]

    def test_copy(self):
        r1, r2, r5 = MyVariable(1), MyVariable(2), MyVariable(5)
        node = MyOp.make_node(r1, r2)
        node2 = MyOp.make_node(node.outputs[0], r5)
        _, new = clone([r1, r2, r5], node2.outputs, False)
        assert (
            node2.outputs[0].type == new[0].type and node2.outputs[0] is not new[0]
        )  # the new output is like the old one but not the same object
        assert node2 is not new[0].owner  # the new output has a new owner
        assert new[0].owner.inputs[1] is r5  # the inputs are not copied
        assert (
            new[0].owner.inputs[0].type == node.outputs[0].type
            and new[0].owner.inputs[0] is not node.outputs[0]
        )  # check that we copied deeper too

    def test_not_destructive(self):
        # Checks that manipulating a cloned graph leaves the original unchanged.
        r1, r2, r5 = MyVariable(1), MyVariable(2), MyVariable(5)
        node = MyOp.make_node(MyOp.make_node(r1, r2).outputs[0], r5)
        _, new = clone([r1, r2, r5], node.outputs, False)
        new_node = new[0].owner
        new_node.inputs = [MyVariable(7), MyVariable(8)]
        assert self.str(graph_inputs(new_node.outputs), new_node.outputs) == [
            "MyOp(R7, R8)"
        ]
        assert self.str(graph_inputs(node.outputs), node.outputs) == [
            "MyOp(MyOp(R1, R2), R5)"
        ]

    def test_constant(self):
        r1, r2, r5 = MyVariable(1), MyVariable(2), MyVariable(5)
        node = MyOp.make_node(MyOp.make_node(r1, r2).outputs[0], r5)
        _, new = clone([r1, r2, r5], node.outputs, False)
        new_node = new[0].owner
        new_node.inputs = [MyVariable(7), MyVariable(8)]
        c1 = pt.constant(1.5)

        i, o = clone([c1], [c1])
        assert i[0] is c1 and o[0] is c1

        i, o = clone([c1], [c1], False)
        assert i[0] is c1 and o[0] is c1

        i, o = clone([c1], [c1], True, False)
        assert i[0] is c1 and o[0] is c1

        i, o = clone([c1], [c1], False, True)
        assert i[0] is c1 and o[0] is c1

    def test_clone_inner_graph(self):
        r1, r2, r3 = MyVariable(1), MyVariable(2), MyVariable(3)
        o1 = MyOp(r1, r2)
        o1.name = "o1"

        # Inner graph
        igo_in_1 = MyVariable(4)
        igo_in_2 = MyVariable(5)
        igo_out_1 = MyOp(igo_in_1, igo_in_2)
        igo_out_1.name = "igo1"

        igo = MyInnerGraphOp([igo_in_1, igo_in_2], [igo_out_1])

        o2 = igo(r3, o1)
        o2.name = "o1"

        o2_node = o2.owner
        o2_node_clone = o2_node.clone(clone_inner_graph=True)

        assert o2_node_clone is not o2_node
        assert o2_node_clone.op.fgraph is not o2_node.op.fgraph
        assert equal_computations(
            o2_node_clone.op.fgraph.outputs, o2_node.op.fgraph.outputs
        )


def prenode(obj):
    if isinstance(obj, Variable):
        if obj.owner:
            return [obj.owner]
    if isinstance(obj, Apply):
        return obj.inputs


class TestToposort:
    def test_simple(self):
        # Test a simple graph
        r1, r2, r5 = MyVariable(1), MyVariable(2), MyVariable(5)
        o = MyOp(r1, r2)
        o.name = "o1"
        o2 = MyOp(o, r5)
        o2.name = "o2"

        clients = {}
        res = general_toposort([o2], prenode, clients=clients)

        assert clients == {
            o2.owner: [o2],
            o: [o2.owner],
            r5: [o2.owner],
            o.owner: [o],
            r1: [o.owner],
            r2: [o.owner],
        }
        assert res == [r5, r2, r1, o.owner, o, o2.owner, o2]

        with pytest.raises(ValueError):
            general_toposort(
                [o2], prenode, compute_deps_cache=lambda x: None, deps_cache=None
            )

        res = io_toposort([r5], [o2])
        assert res == [o.owner, o2.owner]

    def test_double_dependencies(self):
        # Test a graph with double dependencies
        r1, r5 = MyVariable(1), MyVariable(5)
        o = MyOp.make_node(r1, r1)
        o2 = MyOp.make_node(o.outputs[0], r5)
        all = general_toposort(o2.outputs, prenode)
        assert all == [r5, r1, o, o.outputs[0], o2, o2.outputs[0]]

    def test_inputs_owners(self):
        # Test a graph where the inputs have owners
        r1, r5 = MyVariable(1), MyVariable(5)
        o = MyOp.make_node(r1, r1)
        r2b = o.outputs[0]
        o2 = MyOp.make_node(r2b, r2b)
        all = io_toposort([r2b], o2.outputs)
        assert all == [o2]

        o2 = MyOp.make_node(r2b, r5)
        all = io_toposort([r2b], o2.outputs)
        assert all == [o2]

    def test_not_connected(self):
        # Test a graph which is not connected
        r1, r2, r3, r4 = MyVariable(1), MyVariable(2), MyVariable(3), MyVariable(4)
        o0 = MyOp.make_node(r1, r2)
        o1 = MyOp.make_node(r3, r4)
        all = io_toposort([r1, r2, r3, r4], o0.outputs + o1.outputs)
        assert all == [o1, o0] or all == [o0, o1]

    def test_io_chain(self):
        # Test inputs and outputs mixed together in a chain graph
        r1, r2 = MyVariable(1), MyVariable(2)
        o0 = MyOp.make_node(r1, r2)
        o1 = MyOp.make_node(o0.outputs[0], r1)
        all = io_toposort([r1, o0.outputs[0]], [o0.outputs[0], o1.outputs[0]])
        assert all == [o1]

    def test_outputs_clients(self):
        # Test when outputs have clients
        r1, r2, r4 = MyVariable(1), MyVariable(2), MyVariable(4)
        o0 = MyOp.make_node(r1, r2)
        MyOp.make_node(o0.outputs[0], r4)
        all = io_toposort([], o0.outputs)
        assert all == [o0]

    def test_multi_output_nodes(self):
        l0, r0 = op_multiple_outputs(shared(0.0))
        l1, r1 = op_multiple_outputs(shared(0.0))

        v0 = r0 + 1
        v1 = pt.exp(v0)
        out = r1 * v1

        # When either r0 or r1 is provided as an input, the respective node shouldn't be part of the toposort
        assert set(io_toposort([], [out])) == {
            r0.owner,
            r1.owner,
            v0.owner,
            v1.owner,
            out.owner,
        }
        assert set(io_toposort([r0], [out])) == {
            r1.owner,
            v0.owner,
            v1.owner,
            out.owner,
        }
        assert set(io_toposort([r1], [out])) == {
            r0.owner,
            v0.owner,
            v1.owner,
            out.owner,
        }
        assert set(io_toposort([r0, r1], [out])) == {v0.owner, v1.owner, out.owner}

        # When l0 and/or l1 are provided, we still need to compute the respective nodes
        assert set(io_toposort([l0, l1], [out])) == {
            r0.owner,
            r1.owner,
            v0.owner,
            v1.owner,
            out.owner,
        }


class TestEval:
    def setup_method(self):
        self.x, self.y = scalars("x", "y")
        self.z = self.x + self.y
        self.w = 2 * self.z

    def test_eval(self):
        assert self.w.eval({self.x: 1.0, self.y: 2.0}) == 6.0
        assert self.w.eval({self.z: 3}) == 6.0
        assert hasattr(self.w, "_fn_cache"), "variable must have cache after eval"
        assert not hasattr(
            pickle.loads(pickle.dumps(self.w)), "_fn_cache"
        ), "temporary functions must not be serialized"

    def test_eval_with_strings(self):
        assert self.w.eval({"x": 1.0, self.y: 2.0}) == 6.0
        assert self.w.eval({self.z: 3}) == 6.0

    def test_eval_with_strings_multiple_matches(self):
        e = scalars("e")
        t = e + 1
        t.name = "e"
        with pytest.raises(Exception, match="Found multiple variables with name e"):
            t.eval({"e": 1})

    def test_eval_with_strings_no_match(self):
        e = scalars("e")
        t = e + 1
        t.name = "p"
        with pytest.raises(Exception, match="o not found in graph"):
            t.eval({"o": 1})

    def test_eval_kwargs(self):
        with pytest.raises(UnusedInputError):
            self.w.eval({self.z: 3, self.x: 2.5})
        assert self.w.eval({self.z: 3, self.x: 2.5}, on_unused_input="ignore") == 6.0

    @pytest.mark.filterwarnings("error")
    def test_eval_unashable_kwargs(self):
        y_repl = constant(2.0, dtype="floatX")

        assert self.w.eval({self.x: 1.0}, givens=((self.y, y_repl),)) == 6.0

        with pytest.warns(
            UserWarning,
            match="Keyword arguments could not be used to create a cache key",
        ):
            # givens dict is not hashable
            assert self.w.eval({self.x: 1.0}, givens={self.y: y_repl}) == 6.0


class TestAutoName:
    def test_auto_name(self):
        # Get counter value
        autoname_id = next(Variable.__count__)
        Variable.__count__ = count(autoname_id)
        r1, r2 = MyVariable(1), MyVariable(2)
        assert r1.auto_name == "auto_" + str(autoname_id)
        assert r2.auto_name == "auto_" + str(autoname_id + 1)

    def test_constant(self):
        # Get counter value
        autoname_id = next(Variable.__count__)
        Variable.__count__ = count(autoname_id)
        r1 = pt.constant(1.5)
        assert r1.auto_name == "auto_" + str(autoname_id), (
            r1.auto_name,
            "auto_" + str(autoname_id),
        )

        r3 = pt.constant(1.6)
        assert r3.auto_name == "auto_" + str(autoname_id + 1)

    def test_tensorvariable(self):
        # Get counter value
        autoname_id = next(Variable.__count__)
        Variable.__count__ = count(autoname_id)
        r1 = TensorType(dtype="int32", shape=())("myvar")
        r2 = TensorVariable(TensorType(dtype="int32", shape=()), None)
        r3 = shared(np.random.standard_normal((3, 4)))
        assert r1.auto_name == "auto_" + str(autoname_id)
        assert r2.auto_name == "auto_" + str(autoname_id + 1)
        assert r3.auto_name == "auto_" + str(autoname_id + 2)

    def test_clone(self):
        # Get counter value
        autoname_id = next(Variable.__count__)
        Variable.__count__ = count(autoname_id)
        r1 = MyVariable(1)
        r2 = r1.clone()
        assert r1.auto_name == "auto_" + str(autoname_id)
        assert r2.auto_name == "auto_" + str(autoname_id + 1)

        assert r1.name is None and r1.name is r2.name

        r3_name = "r3"
        r3 = r1.clone(name=r3_name)
        assert r3.name == r3_name


def test_equal_computations():
    a, b = iscalars(2)

    with pytest.raises(ValueError):
        equal_computations([a], [a, b])

    assert equal_computations([a], [a])
    assert equal_computations([pt.as_tensor(1)], [pt.as_tensor(1)])
    assert not equal_computations([b], [a])
    assert not equal_computations([pt.as_tensor(1)], [pt.as_tensor(2)])

    assert equal_computations([2], [2])
    assert equal_computations([np.r_[2, 1]], [np.r_[2, 1]])
    assert equal_computations([np.r_[2, 1]], [pt.as_tensor(np.r_[2, 1])])
    assert equal_computations([pt.as_tensor(np.r_[2, 1])], [np.r_[2, 1]])

    assert not equal_computations([2], [a])
    assert not equal_computations([np.r_[2, 1]], [a])
    assert not equal_computations([a], [2])
    assert not equal_computations([a], [np.r_[2, 1]])

    assert equal_computations([NoneConst], [NoneConst])

    m = matrix()
    max_argmax1 = max_and_argmax(m)
    max_argmax2 = max_and_argmax(m)
    assert equal_computations(max_argmax1, max_argmax2)


def test_walk():
    r1, r2, r3 = MyVariable(1), MyVariable(2), MyVariable(3)
    o1 = MyOp(r1, r2)
    o1.name = "o1"
    o2 = MyOp(r3, o1)
    o2.name = "o2"

    def expand(r):
        if r.owner:
            return r.owner.inputs

    res = walk([o2], expand, bfs=True, return_children=False)
    res_list = list(res)
    assert res_list == [o2, r3, o1, r1, r2]

    res = walk([o2], expand, bfs=False, return_children=False)
    res_list = list(res)
    assert res_list == [o2, o1, r2, r1, r3]

    res = walk([o2], expand, bfs=True, return_children=True)
    res_list = list(res)
    assert res_list == [
        (o2, [r3, o1]),
        (r3, None),
        (o1, [r1, r2]),
        (r1, None),
        (r2, None),
    ]


def test_ancestors():
    r1, r2, r3 = MyVariable(1), MyVariable(2), MyVariable(3)
    o1 = MyOp(r1, r2)
    o1.name = "o1"
    o2 = MyOp(r3, o1)
    o2.name = "o2"

    res = ancestors([o2], blockers=None)
    res_list = list(res)
    assert res_list == [o2, r3, o1, r1, r2]

    res = ancestors([o2], blockers=None)
    assert r3 in res
    res_list = list(res)
    assert res_list == [o1, r1, r2]

    res = ancestors([o2], blockers=[o1])
    res_list = list(res)
    assert res_list == [o2, r3, o1]


def test_graph_inputs():
    r1, r2, r3 = MyVariable(1), MyVariable(2), MyVariable(3)
    o1 = MyOp(r1, r2)
    o1.name = "o1"
    o2 = MyOp(r3, o1)
    o2.name = "o2"

    res = graph_inputs([o2], blockers=None)
    res_list = list(res)
    assert res_list == [r3, r1, r2]


def test_explicit_graph_inputs():
    x = pt.fscalar()
    y = pt.constant(2)
    z = shared(1)
    a = pt.sum(x + y + z)
    b = pt.true_div(x, y)

    res = list(explicit_graph_inputs([a]))
    res1 = list(explicit_graph_inputs(b))

    assert res == [x]
    assert res1 == [x]


def test_variables_and_orphans():
    r1, r2, r3 = MyVariable(1), MyVariable(2), MyVariable(3)
    o1 = MyOp(r1, r2)
    o1.name = "o1"
    o2 = MyOp(r3, o1)
    o2.name = "o2"

    vars_res = vars_between([r1, r2], [o2])
    orphans_res = orphans_between([r1, r2], [o2])

    vars_res_list = list(vars_res)
    orphans_res_list = list(orphans_res)
    assert vars_res_list == [o2, o1, r3, r2, r1]
    assert orphans_res_list == [r3]


def test_ops():
    r1, r2, r3, r4 = MyVariable(1), MyVariable(2), MyVariable(3), MyVariable(4)
    o1 = MyOp(r1, r2)
    o1.name = "o1"
    o2 = MyOp(r3, r4)
    o2.name = "o2"
    o3 = MyOp(r3, o1, o2)
    o3.name = "o3"

    res = applys_between([r1, r2], [o3])
    res_list = list(res)
    assert res_list == [o3.owner, o2.owner, o1.owner]


def test_apply_depends_on():
    r1, r2, r3 = MyVariable(1), MyVariable(2), MyVariable(3)
    o1 = MyOp(r1, r2)
    o1.name = "o1"
    o2 = MyOp(r1, o1)
    o2.name = "o2"
    o3 = MyOp(r3, o1, o2)
    o3.name = "o3"

    assert apply_depends_on(o2.owner, o1.owner)
    assert apply_depends_on(o2.owner, o2.owner)
    assert apply_depends_on(o3.owner, [o1.owner, o2.owner])


@pytest.mark.xfail(reason="Not implemented")
def test_io_connection_pattern():
    raise AssertionError()


@pytest.mark.xfail(reason="Not implemented")
def test_view_roots():
    raise AssertionError()


def test_get_var_by_name():
    r1, r2, r3 = MyVariable(1), MyVariable(2), MyVariable(3)
    o1 = MyOp(r1, r2)
    o1.name = "o1"

    # Inner graph
    igo_in_1 = MyVariable(4)
    igo_in_2 = MyVariable(5)
    igo_out_1 = MyOp(igo_in_1, igo_in_2)
    igo_out_1.name = "igo1"

    igo = MyInnerGraphOp([igo_in_1, igo_in_2], [igo_out_1])

    o2 = igo(r3, o1)
    o2.name = "o1"

    res = get_var_by_name([o1, o2], "blah")

    assert res == ()

    res = get_var_by_name([o1, o2], "o1")

    assert set(res) == {o1, o2}

    (res,) = get_var_by_name([o1, o2], o1.auto_name)

    assert res == o1

    (res,) = get_var_by_name([o1, o2], "igo1")

    exp_res = igo.fgraph.outputs[0]
    assert res == exp_res


def test_clone_new_inputs():
    """Make sure that `Apply.clone_with_new_inputs` properly handles `Type` changes."""

    x = pt.tensor(dtype=np.float64, shape=(None,))
    y = pt.tensor(dtype=np.float64, shape=(1,))

    z = pt.add(x, y)
    assert z.type.shape == (None,)

    x_new = pt.tensor(dtype=np.float64, shape=(1,))

    # The output nodes should be reconstructed, because the input types' static
    # shape information increased in specificity
    z_node_new = z.owner.clone_with_new_inputs([x_new, y])

    assert z_node_new.outputs[0].type.shape == (1,)
    assert z_node_new.inputs[0].type.shape == (1,)
    assert z_node_new.inputs[1].type.shape == (1,)

    # Now, attempt to decrease the specificity of the first input's static
    # shape information, but, because we're using strict conversion, we
    # shouldn't lose any information
    z = pt.add(x_new, y)
    assert z.type.shape == (1,)

    z_node_new = z.owner.clone_with_new_inputs([x, y], strict=True)

    assert z_node_new.outputs[0].type.shape == (1,)
    assert z_node_new.inputs[0].type.shape == (1,)
    assert z_node_new.inputs[1].type.shape == (1,)


def test_clone_get_equiv():
    x = vector("x")
    y = vector("y")
    z = vector("z")
    a = x * y
    a_node = a.owner
    b = a + 1.0

    memo = {a: z}
    _ = clone_get_equiv([x, y], [b], copy_inputs=False, copy_orphans=False, memo=memo)

    assert x in memo
    assert y in memo
    assert memo[a] is z
    # All the outputs of `a` already had replacements/clones in the map, so
    # there is no need to re-clone it (unless another replacement/clone
    # re-introduces `a.owner` somehow).
    assert a_node not in memo
    assert equal_computations([memo[b]], [z + 1.0])


def test_NominalVariable():
    type1 = MyType(1)

    nv1 = NominalVariable(1, type1)
    nv2 = NominalVariable(1, type1)

    assert nv1 is nv2
    assert nv1.equals(nv2)
    assert hash(nv1) == hash(nv2)

    type2 = MyType(2)
    nv3 = NominalVariable(1, type2)

    assert not nv1.equals(nv3)
    assert hash(nv1) != hash(nv3)

    type3 = MyType(1)

    assert type3 == type1

    nv4 = NominalVariable(1, type3)

    assert nv1 is nv4
    assert nv1.equals(nv4)
    assert hash(nv1) == hash(nv4)

    nv5 = NominalVariable(2, type3)
    assert not nv4.equals(nv5)
    assert hash(nv4) != hash(nv5)

    assert repr(nv5) == f"NominalVariable(2, {type3!r})"

    assert nv5.signature() == (type3, 2)

    nv5_pkld = pickle.dumps(nv5)
    nv5_unpkld = pickle.loads(nv5_pkld)

    assert type(nv5_unpkld) is type(nv5)
    assert nv5_unpkld.equals(nv5)
    assert nv5_unpkld is nv5

    nv5_clone = nv5.clone()
    assert type(nv5_clone) is type(nv5)
    assert nv5_clone.equals(nv5)
    assert nv5_clone is nv5


def test_NominalVariable_create_variable_type():
    ttype = TensorType("float64", (None, None))
    ntv = NominalVariable(0, ttype)

    assert isinstance(ntv, TensorVariable)
    assert isinstance(ntv, NominalVariable)
    assert ntv.ndim == 2
    assert ntv.broadcastable == (False, False)
    assert ntv.dtype == "float64"

    ntv2 = NominalVariable(0, ttype)

    assert type(ntv2) is type(ntv)
    assert ntv2.equals(ntv)
    assert ntv2 is ntv

    ntv_pkld = pickle.dumps(ntv)
    ntv_unpkld = pickle.loads(ntv_pkld)

    assert type(ntv_unpkld) is type(ntv)
    assert ntv_unpkld.equals(ntv)
    assert ntv_unpkld is ntv


def test_variable_depends_on():
    x = MyVariable(1)
    x.name = "x"
    y = MyVariable(1)
    y.name = "y"
    x2 = MyOp(x)
    x2.name = "x2"
    y2 = MyOp(y)
    y2.name = "y2"
    o = MyOp(x2, y)
    assert variable_depends_on(o, x)
    assert variable_depends_on(o, [x])
    assert not variable_depends_on(o, [y2])
    assert variable_depends_on(o, [y2, x])
    assert not variable_depends_on(y, [y2])
    assert variable_depends_on(y, [y])


class TestTruncatedGraphInputs:
    def test_basic(self):
        """
        * No conditions
            n - n - (o)

        * One condition
            n - (c) - o

        * Two conditions where on depends on another, both returned
            (c) - (c) - o

        * Additional nodes are present
               (c) - n - o
            n - (n) -'

        * Disconnected condition not returned
            (c) - n - o
             c

        * Disconnected output is present and returned
            (c) - (c) - o
            (o)

        * Condition on itself adds itself
            n - (c) - (o/c)
        """
        x = MyVariable(1)
        x.name = "x"
        y = MyVariable(1)
        y.name = "y"
        z = MyVariable(1)
        z.name = "z"
        x2 = MyOp(x)
        x2.name = "x2"
        y2 = MyOp(y, x2)
        y2.name = "y2"
        o = MyOp(y2)
        o2 = MyOp(o)
        # No conditions
        assert truncated_graph_inputs([o]) == [o]
        # One condition
        assert truncated_graph_inputs([o2], [y2]) == [y2]
        # Condition on itself adds itself
        assert truncated_graph_inputs([o], [y2, o]) == [o, y2]
        # Two conditions where on depends on another, both returned
        assert truncated_graph_inputs([o2], [y2, o]) == [o, y2]
        # Additional nodes are present
        assert truncated_graph_inputs([o], [y]) == [x2, y]
        # Disconnected condition
        assert truncated_graph_inputs([o2], [y2, z]) == [y2]
        # Disconnected output is present
        assert truncated_graph_inputs([o2, z], [y2]) == [z, y2]

    def test_repeated_input(self):
        """Test that truncated_graph_inputs does not return repeated inputs."""
        x = MyVariable(1)
        x.name = "x"
        y = MyVariable(1)
        y.name = "y"

        trunc_inp1 = MyOp(x, y)
        trunc_inp1.name = "trunc_inp1"

        trunc_inp2 = MyOp(x, y)
        trunc_inp2.name = "trunc_inp2"

        o = MyOp(trunc_inp1, trunc_inp1, trunc_inp2, trunc_inp2)
        o.name = "o"

        assert truncated_graph_inputs([o], [trunc_inp1]) == [trunc_inp2, trunc_inp1]

    def test_repeated_nested_input(self):
        """Test that truncated_graph_inputs does not return repeated inputs."""
        x = MyVariable(1)
        x.name = "x"
        y = MyVariable(1)
        y.name = "y"

        trunc_inp = MyOp(x, y)
        trunc_inp.name = "trunc_inp"

        o1 = MyOp(trunc_inp, trunc_inp, x, x)
        o1.name = "o1"

        assert truncated_graph_inputs([o1], [trunc_inp]) == [x, trunc_inp]

        # Reverse order of inputs
        o2 = MyOp(x, x, trunc_inp, trunc_inp)
        o2.name = "o2"

        assert truncated_graph_inputs([o2], [trunc_inp]) == [trunc_inp, x]

    def test_single_pass_per_node(self, mocker):
        import pytensor.graph.basic

        inspect = mocker.spy(pytensor.graph.basic, "variable_depends_on")
        x = pt.dmatrix("x")
        m = x.shape[0][None, None]

        f = x / m
        w = x / m - f
        truncated_graph_inputs([w], [x])
        # make sure there were exactly the same calls as unique variables seen by the function
        assert len(inspect.call_args_list) == len(
            {a for ((a, b), kw) in inspect.call_args_list}
        )


def test_dprint():
    r1, r2 = MyVariable(1), MyVariable(2)
    o1 = MyOp(r1, r2)
    assert o1.dprint(file="str") == debugprint(o1, file="str")
    assert o1.owner.dprint(file="str") == debugprint(o1.owner, file="str")
