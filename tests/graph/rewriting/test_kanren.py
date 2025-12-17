from copy import copy

import numpy as np
import pytest
from etuples import etuple
from kanren import eq, fact, run
from kanren.assoccomm import associative, commutative, eq_assoccomm
from kanren.core import lall
from unification import var, vars

import pytensor.tensor as pt
from pytensor.graph.basic import Apply
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.op import Op
from pytensor.graph.rewriting.basic import EquilibriumGraphRewriter
from pytensor.graph.rewriting.kanren import KanrenRelationSub
from pytensor.graph.rewriting.unify import eval_if_etuple
from pytensor.graph.rewriting.utils import rewrite_graph
from pytensor.tensor.math import Dot, _dot
from tests.graph.utils import MyType, MyVariable


@pytest.fixture(autouse=True)
def clear_assoccomm():
    old_commutative_index = copy(commutative.index)
    old_commutative_facts = copy(commutative.facts)
    old_associative_index = copy(associative.index)
    old_associative_facts = copy(associative.facts)
    try:
        yield
    finally:
        commutative.index = old_commutative_index
        commutative.facts = old_commutative_facts
        associative.index = old_associative_index
        associative.facts = old_associative_facts


def test_kanren_basic():
    A_pt = pt.matrix("A")
    B_pt = pt.matrix("B")

    y_pt = pt.dot(A_pt, B_pt)

    q = var()
    res = list(run(None, q, eq(y_pt, etuple(_dot, q, B_pt))))

    assert res == [A_pt]


def test_KanrenRelationSub_filters():
    A_pt = pt.matrix("A")
    B_pt = pt.matrix("B")
    C_pt = pt.matrix("C")
    D_pt = pt.matrix("D")

    fact(commutative, _dot)
    fact(commutative, pt.add)
    fact(associative, pt.add)

    Z_pt = A_pt.dot((B_pt + C_pt) + D_pt)

    fgraph = FunctionGraph(outputs=[Z_pt], clone=False)

    def distributes(in_lv, out_lv):
        A_lv, B_lv, C_lv, D_lv = vars(4)
        return lall(
            # lhs == A * (x + y + z)
            eq_assoccomm(
                etuple(_dot, A_lv, etuple(pt.add, B_lv, etuple(pt.add, C_lv, D_lv))),
                in_lv,
            ),
            # This relation does nothing but provide us with a means of
            # generating associative-commutative matches in the `kanren`
            # output.
            eq((A_lv, B_lv, C_lv, D_lv), out_lv),
        )

    def results_filter(results):
        _results = [eval_if_etuple(v) for v in results]

        # Make sure that at least a couple permutations are present
        assert (A_pt, B_pt, C_pt, D_pt) in _results
        assert (A_pt, C_pt, B_pt, D_pt) in _results
        assert (A_pt, D_pt, B_pt, C_pt) in _results

        return None

    _ = KanrenRelationSub(distributes, results_filter=results_filter).transform(
        fgraph, fgraph.outputs[0].owner
    )

    res = KanrenRelationSub(distributes, node_filter=lambda x: False).transform(
        fgraph, fgraph.outputs[0].owner
    )
    assert res is False


def test_KanrenRelationSub_multiout():
    class MyMultiOutOp(Op):
        def make_node(self, *inputs):
            outputs = [MyType()(), MyType()()]
            return Apply(self, list(inputs), outputs)

        def perform(self, node, inputs, output_storage):
            output_storage[0] = np.array(inputs[0])
            output_storage[1] = np.array(inputs[0])

    x = MyVariable("x")
    y = MyVariable("y")
    multi_op = MyMultiOutOp()
    o1, o2 = multi_op(x, y)
    fgraph = FunctionGraph([x, y], [o1], clone=False)

    def relation(in_lv, out_lv):
        return eq(in_lv, out_lv)

    res = KanrenRelationSub(relation).transform(fgraph, fgraph.outputs[0].owner)

    assert res == [o1, o2]


def test_KanrenRelationSub_dot():
    """Make sure we can run miniKanren "optimizations" over a graph until a fixed-point/normal-form is reached."""
    A_pt = pt.matrix("A")
    B_pt = pt.matrix("B")
    C_pt = pt.matrix("C")
    D_pt = pt.matrix("D")
    E_pt = pt.matrix("E")

    Z_pt = A_pt.dot(E_pt + B_pt.dot(C_pt + D_pt))

    fgraph = FunctionGraph(outputs=[Z_pt], clone=False)

    assert isinstance(fgraph.outputs[0].owner.op, Dot)

    def distributes(in_lv, out_lv):
        return lall(
            # lhs == A * (x + b)
            eq(
                etuple(_dot, var("A"), etuple(pt.add, var("E"), var("B"))),
                in_lv,
            ),
            # rhs == A * x + A * b
            eq(
                etuple(
                    pt.add,
                    etuple(_dot, var("A"), var("E")),
                    etuple(_dot, var("A"), var("B")),
                ),
                out_lv,
            ),
        )

    distribute_opt = EquilibriumGraphRewriter(
        [KanrenRelationSub(distributes)], max_use_ratio=10
    )

    fgraph_opt = rewrite_graph(fgraph, custom_rewrite=distribute_opt)
    (expr_opt,) = fgraph_opt.outputs

    assert expr_opt.owner.op == pt.add
    assert isinstance(expr_opt.owner.inputs[0].owner.op, Dot)
    assert fgraph_opt.inputs[-1] is A_pt
    assert expr_opt.owner.inputs[0].owner.inputs[0].name == "A"
    assert expr_opt.owner.inputs[1].owner.op == pt.add
    assert isinstance(expr_opt.owner.inputs[1].owner.inputs[0].owner.op, Dot)
    assert isinstance(expr_opt.owner.inputs[1].owner.inputs[1].owner.op, Dot)
