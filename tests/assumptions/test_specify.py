import pytest

import pytensor.tensor as pt
from pytensor.assumptions import (
    DIAGONAL,
    LOWER_TRIANGULAR,
    ORTHOGONAL,
    POSITIVE_DEFINITE,
    SYMMETRIC,
    UNIQUE_INDICES,
    UPPER_TRIANGULAR,
    ConflictingAssumptionsError,
    FactState,
)
from pytensor.assumptions.specify import SpecifyAssumptions, assume
from pytensor.tensor.rewriting.assumptions import (
    DrainSpecifyAssumptions,
    drain_specify_assumptions_node,
)
from tests.assumptions.conftest import make_fgraph


@pytest.mark.parametrize(
    "key",
    [
        DIAGONAL,
        LOWER_TRIANGULAR,
        UPPER_TRIANGULAR,
        SYMMETRIC,
        POSITIVE_DEFINITE,
        ORTHOGONAL,
    ],
    ids=lambda k: k.name,
)
def test_assume_records_false_assertions(key):
    x = pt.matrix("x", shape=(3, 3))
    x_not = assume(x, **{key.name: False})
    _, af = make_fgraph(x_not)
    assert af.get(x_not, key) is FactState.FALSE


def test_assume_unique_indices_on_vector():
    idx = pt.vector("idx", dtype="int64")
    idx_uniq = assume(idx, unique_indices=True)
    _, af = make_fgraph(idx_uniq)
    assert af.check(idx_uniq, UNIQUE_INDICES)


def test_assume_multiple_true_kwargs():
    x = pt.matrix("x", shape=(3, 3))
    x_both = assume(x, diagonal=True, positive_definite=True)
    _, af = make_fgraph(x_both)
    assert af.check(x_both, DIAGONAL)
    assert af.check(x_both, POSITIVE_DEFINITE)
    assert af.check(x_both, SYMMETRIC)


def test_assume_mixed_true_and_false_kwargs():
    x = pt.matrix("x", shape=(3, 3))
    x_mix = assume(x, diagonal=True, orthogonal=False)
    _, af = make_fgraph(x_mix)
    assert af.check(x_mix, DIAGONAL)
    assert af.get(x_mix, ORTHOGONAL) is FactState.FALSE


def test_assume_chained_combines_facts():
    x = pt.matrix("x", shape=(3, 3))
    chained = assume(assume(x, diagonal=True), positive_definite=True)
    _, af = make_fgraph(chained)
    assert af.check(chained, DIAGONAL)
    assert af.check(chained, POSITIVE_DEFINITE)


def test_specify_assumptions_op_equal_for_same_facts():
    a = SpecifyAssumptions({"diagonal": FactState.TRUE, "symmetric": FactState.FALSE})
    b = SpecifyAssumptions({"symmetric": FactState.FALSE, "diagonal": FactState.TRUE})
    assert a == b
    assert hash(a) == hash(b)


def test_assume_none_kwarg_passes_through():
    x = pt.matrix("x", shape=(3, 3))
    x_diag = assume(x, diagonal=True, symmetric=None)
    _, af = make_fgraph(x_diag)
    assert af.check(x_diag, DIAGONAL)
    # SYMMETRIC is recovered via the DIAGONAL -> SYMMETRIC implication, not asserted directly.
    assert af.check(x_diag, SYMMETRIC)


def test_assume_no_kwargs_returns_input_unchanged():
    x = pt.matrix("x")
    assert assume(x) is x


def test_assume_conflict_with_inferred_fact_raises():
    e = pt.eye(5)
    e_not_diag = assume(e, diagonal=False)
    _, af = make_fgraph(e_not_diag)
    with pytest.raises(ConflictingAssumptionsError):
        af.get(e_not_diag, DIAGONAL)


def test_whole_graph_drain_moves_the_fact_onto_the_input():
    """Draining is not just node removal; the declaration has to land on ``x`` itself.

    Consumers query ``x``, never the marker, so a drain that drops the node without
    transferring the fact discards the assumption without any visible failure.
    """
    x = pt.matrix("x")
    out = pt.linalg.det(assume(x, positive_definite=True))
    fg, feature = make_fgraph(out)

    DrainSpecifyAssumptions().apply(fg)

    assert not any(isinstance(node.op, SpecifyAssumptions) for node in fg.apply_nodes)
    assert feature.check(x, POSITIVE_DEFINITE)


def test_local_drain_moves_the_fact_onto_the_input():
    """The per-node drain owes the same guarantee as the whole-graph pass."""
    x = pt.matrix("x")
    marker = assume(x, positive_definite=True)
    fg, feature = make_fgraph(pt.linalg.det(marker))

    [replacement] = drain_specify_assumptions_node.transform(fg, marker.owner)
    fg.replace(marker, replacement, reason="test")

    assert replacement is x
    assert feature.check(x, POSITIVE_DEFINITE)
