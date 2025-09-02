import pytest

from pytensor import Variable, shared
from pytensor import tensor as pt
from pytensor.graph import Apply, ancestors, graph_inputs
from pytensor.graph.traversal import (
    apply_ancestors,
    apply_depends_on,
    explicit_graph_inputs,
    general_toposort,
    get_var_by_name,
    io_toposort,
    orphans_between,
    toposort,
    toposort_with_orderings,
    truncated_graph_inputs,
    variable_ancestors,
    variable_depends_on,
    vars_between,
    walk,
)
from tests.graph.test_basic import MyOp, MyVariable
from tests.graph.utils import MyInnerGraphOp, op_multiple_outputs


class TestToposort:
    @staticmethod
    def prenode(obj):
        if isinstance(obj, Variable):
            if obj.owner:
                return [obj.owner]
        if isinstance(obj, Apply):
            return obj.inputs

    def test_simple(self):
        # Test a simple graph
        r1, r2, r5 = MyVariable(1), MyVariable(2), MyVariable(5)
        o = MyOp(r1, r2)
        o.name = "o1"
        o2 = MyOp(o, r5)
        o2.name = "o2"

        res = general_toposort([o2], self.prenode)
        assert res == [r5, r2, r1, o.owner, o, o2.owner, o2]

        def circular_dependency(obj):
            if obj is o:
                # o2 depends on o, so o cannot depend on o2
                return [o2, *self.prenode(obj)]
            return self.prenode(obj)

        with pytest.raises(ValueError, match="graph contains cycles"):
            general_toposort([o2], circular_dependency)

        res = io_toposort([r5], [o2])
        assert res == [o.owner, o2.owner]

    def test_double_dependencies(self):
        # Test a graph with double dependencies
        r1, r5 = MyVariable(1), MyVariable(5)
        o = MyOp.make_node(r1, r1)
        o2 = MyOp.make_node(o.outputs[0], r5)
        all = general_toposort(o2.outputs, self.prenode)
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
    assert res_list == [o2, o1, r2, r1, r3]

    res = ancestors([o2], blockers=None)
    assert o1 in res
    res_list = list(res)
    assert res_list == [r2, r1, r3]

    res = ancestors([o2], blockers=[o1])
    res_list = list(res)
    assert res_list == [o2, o1, r3]


def test_graph_inputs():
    r1, r2, r3 = MyVariable(1), MyVariable(2), MyVariable(3)
    o1 = MyOp(r1, r2)
    o1.name = "o1"
    o2 = MyOp(r3, o1)
    o2.name = "o2"

    res = graph_inputs([o2], blockers=None)
    res_list = list(res)
    assert res_list == [r2, r1, r3]


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
    assert vars_res_list == [o2, o1, r2, r1, r3]
    assert orphans_res_list == [r3]


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
        import pytensor.graph.traversal

        inspect = mocker.spy(pytensor.graph.traversal, "variable_depends_on")
        x = pt.dmatrix("x")
        m = x.shape[0][None, None]

        f = x / m
        w = x / m - f
        truncated_graph_inputs([w], [x])
        # make sure there were exactly the same calls as unique variables seen by the function
        assert len(inspect.call_args_list) == len(
            {a for ((a, b), kw) in inspect.call_args_list}
        )


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


@pytest.mark.parametrize(
    "func",
    [
        lambda x: all(variable_ancestors([x])),
        lambda x: all(variable_ancestors([x], blockers=[x.clone()])),
        lambda x: all(apply_ancestors([x])),
        lambda x: all(apply_ancestors([x], blockers=[x.clone()])),
        lambda x: all(toposort([x])),
        lambda x: all(toposort([x], blockers=[x.clone()])),
        lambda x: all(toposort_with_orderings([x], orderings={x: []})),
        lambda x: all(
            toposort_with_orderings([x], blockers=[x.clone()], orderings={x: []})
        ),
    ],
    ids=[
        "variable_ancestors",
        "variable_ancestors_with_blockers",
        "apply_ancestors",
        "apply_ancestors_with_blockers)",
        "toposort",
        "toposort_with_blockers",
        "toposort_with_orderings",
        "toposort_with_orderings_and_blockers",
    ],
)
def test_traversal_benchmark(func, benchmark):
    r1 = MyVariable(1)
    out = r1
    for i in range(50):
        out = MyOp(out, out)

    benchmark(func, out)
