import pytest

import pytensor.tensor as pt
from pytensor.assumptions import (
    DIAGONAL,
    LOWER_TRIANGULAR,
    POSITIVE_DEFINITE,
    SYMMETRIC,
    UPPER_TRIANGULAR,
    AssumptionFeature,
    AssumptionKey,
    ConflictingAssumptionsError,
    FactState,
    register_assumption,
)
from pytensor.assumptions.specify import assume
from pytensor.graph.basic import Apply
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.op import Op
from pytensor.tensor.basic import Eye
from pytensor.tensor.rewriting.assumptions import DrainSpecifyAssumptions
from pytensor.tensor.type import TensorType
from tests.assumptions.conftest import make_fgraph


@pytest.mark.parametrize(
    "left, right, expected",
    [
        (FactState.UNKNOWN, FactState.UNKNOWN, FactState.UNKNOWN),
        (FactState.TRUE, FactState.UNKNOWN, FactState.TRUE),
        (FactState.FALSE, FactState.UNKNOWN, FactState.FALSE),
        (FactState.TRUE, FactState.FALSE, FactState.CONFLICT),
    ],
    ids=["unknown+unknown", "true+unknown", "false+unknown", "true+false"],
)
def test_fact_state_join(left, right, expected):
    assert FactState.join(left, right) == expected


@pytest.mark.parametrize(
    "state",
    [FactState.TRUE, FactState.UNKNOWN, FactState.FALSE, FactState.CONFLICT],
)
def test_fact_state_has_no_bool(state):
    with pytest.raises(TypeError, match="FactState has no boolean value"):
        bool(state)


def test_double_attach_keeps_single_feature():
    x = pt.matrix("x")
    fg = FunctionGraph([x], [x], clone=False)
    fg.attach_feature(AssumptionFeature())
    fg.attach_feature(AssumptionFeature())
    assert sum(isinstance(f, AssumptionFeature) for f in fg._features) == 1


def test_bare_input_is_unknown():
    x = pt.matrix("x")
    _, af = make_fgraph(x)
    assert af.get(x, DIAGONAL) == FactState.UNKNOWN


def test_custom_op_via_register_assumption():
    class AlwaysSymmetricOp(Op):
        __props__ = ()

        def make_node(self, x):
            return Apply(self, [x], [TensorType(dtype=x.dtype, shape=(None, None))()])

        def perform(self, node, inputs, outputs):
            outputs[0][0] = inputs[0]

    @register_assumption(SYMMETRIC, AlwaysSymmetricOp)
    def _always_symmetric(key, op, feature, fgraph, node, input_states):
        return [FactState.TRUE]

    x = pt.matrix("x")
    y = AlwaysSymmetricOp()(x)
    _, af = make_fgraph(y)
    assert af.check(y, SYMMETRIC)
    assert af.get(y, DIAGONAL) == FactState.UNKNOWN


def test_register_custom_assumption_key():
    INVERTIBLE = AssumptionKey("invertible")

    @register_assumption(INVERTIBLE, Eye)
    def _eye_invertible(key, op, feature, fgraph, node, input_states):
        return [FactState.TRUE]

    e = pt.eye(5)
    _, af = make_fgraph(e)
    assert af.check(e, INVERTIBLE)


@pytest.mark.parametrize(
    "stronger, weaker",
    [
        (DIAGONAL, SYMMETRIC),
        (DIAGONAL, LOWER_TRIANGULAR),
        (DIAGONAL, UPPER_TRIANGULAR),
        (POSITIVE_DEFINITE, SYMMETRIC),
    ],
)
def test_implication(stronger, weaker):
    x = pt.matrix("x", shape=(3, 3))
    x_asserted = assume(x, **{stronger.name: True})
    _, af = make_fgraph(x_asserted)
    assert af.check(x_asserted, weaker)


@pytest.mark.parametrize(
    "stronger, weaker",
    [
        (DIAGONAL, SYMMETRIC),
        (DIAGONAL, LOWER_TRIANGULAR),
        (DIAGONAL, UPPER_TRIANGULAR),
        (POSITIVE_DEFINITE, SYMMETRIC),
    ],
)
def test_contrapositive_implication(stronger, weaker):
    """``weaker = FALSE`` forces ``stronger = FALSE`` (contrapositive of the implication)."""
    x = pt.matrix("x", shape=(3, 3))
    x_asserted = assume(x, **{weaker.name: False})
    _, af = make_fgraph(x_asserted)
    assert af.get(x_asserted, weaker) is FactState.FALSE
    assert af.get(x_asserted, stronger) is FactState.FALSE


def test_symmetric_does_not_imply_diagonal():
    x = pt.matrix("x", shape=(3, 3))
    x_sym = assume(x, symmetric=True)
    _, af = make_fgraph(x_sym)
    assert not af.check(x_sym, DIAGONAL)


def test_deep_graph_no_recursion_error():
    x = pt.eye(5)
    for _ in range(2000):
        x = x * 1.0
    _, af = make_fgraph(x)
    assert af.check(x, DIAGONAL)


def test_replace_propagates_true_fact_to_new_var():
    """``fg.replace`` carries a TRUE fact from the old var onto the new one, and
    the downstream TRUE survives."""
    v = pt.vector("v", shape=(5,))
    d_old = pt.diag(v)
    y = pt.sin(d_old)
    fg, af = make_fgraph(y)
    assert af.check(d_old, DIAGONAL)
    assert af.check(y, DIAGONAL)

    d_new = pt.diag(v)
    fg.replace(d_old, d_new, reason="replace")
    assert af.get(d_new, DIAGONAL) is FactState.TRUE
    assert af.get(y, DIAGONAL) is FactState.TRUE


def test_drain_preserves_implied_facts():
    x = pt.matrix("x", shape=(4, 4))
    fg, af = make_fgraph(assume(x, diagonal=True) + 1, inputs=[x])
    DrainSpecifyAssumptions().apply(fg)
    assert af.check(x, SYMMETRIC)


def test_replace_reprobes_downstream_unknown():
    """A downstream UNKNOWN is dropped after ``replace`` so it re-probes and picks
    up the richer ancestor."""
    v = pt.vector("v", shape=(5,))
    x_opaque = pt.matrix("x_opaque", shape=(5, 5))
    y = pt.sin(x_opaque)
    fg, af = make_fgraph(y, inputs=[x_opaque, v])
    assert af.get(y, DIAGONAL) is FactState.UNKNOWN

    fg.replace(x_opaque, pt.diag(v), reason="replace")
    assert af.check(y, DIAGONAL)


def test_replace_unknown_does_not_override_new_inference():
    """An UNKNOWN cached on the old var must not mask the new var's own inference."""
    v = pt.vector("v", shape=(5,))
    x = pt.matrix("x", shape=(5, 5))
    fg, af = make_fgraph(pt.sin(x), inputs=[x, v])
    assert af.get(x, DIAGONAL) is FactState.UNKNOWN

    d = pt.diag(v)
    fg.replace(x, d, reason="replace")
    assert af.check(d, DIAGONAL)


def test_conflict_during_substitution_does_not_corrupt_graph():
    """A conflicting substitution must not raise from ``on_change_input``.

    ``fg.change_node_input`` mutates ``node.inputs[i]`` and ``clients`` *before*
    firing feature callbacks, so raising from ``on_change_input`` leaves the graph
    half-updated while the caller's ``except`` reads the substitution as rejected.
    The substitution should land cleanly; the conflict is cached and surfaces on
    the next ``feature.get(new_var, key)``.
    """
    v = pt.vector("v", shape=(5,))
    x = pt.matrix("x", shape=(5, 5))
    old_true = pt.diag(v)
    new_false = assume(x, diagonal=False)
    sink = pt.sin(old_true)

    fg, af = make_fgraph(sink, inputs=[v, x])
    assert af.get(old_true, DIAGONAL) is FactState.TRUE
    assert af.get(new_false, DIAGONAL) is FactState.FALSE

    fg.replace(old_true, new_false, reason="conflict")

    assert sink.owner.inputs[0] is new_false
    assert fg.clients[old_true] == []
    assert (sink.owner, 0) in fg.clients[new_false]

    with pytest.raises(ConflictingAssumptionsError):
        af.get(new_false, DIAGONAL)
