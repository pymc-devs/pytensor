import math

import pytest

import pytensor.tensor as pt
from pytensor.assumptions import (
    DIAGONAL,
    LOWER_TRIANGULAR,
    ORTHOGONAL,
    PERMUTATION,
    POSITIVE_DEFINITE,
    SELECTION,
    SYMMETRIC,
    UNIQUE_INDICES,
    UPPER_TRIANGULAR,
    ConflictingAssumptionsError,
    FactState,
)
from pytensor.assumptions.specify import SpecifyAssumptions, assume
from tests.assumptions.conftest import RankBoundedOp, make_fgraph


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
    a = SpecifyAssumptions({DIAGONAL: FactState.TRUE, SYMMETRIC: FactState.FALSE})
    b = SpecifyAssumptions({SYMMETRIC: FactState.FALSE, DIAGONAL: FactState.TRUE})
    assert a == b
    assert hash(a) == hash(b)


def test_assume_none_kwarg_passes_through():
    x = pt.matrix("x", shape=(3, 3))
    x_diag = assume(x, diagonal=True, symmetric=None)
    _, af = make_fgraph(x_diag)
    assert af.check(x_diag, DIAGONAL)
    # SYMMETRIC is recovered via the DIAGONAL -> SYMMETRIC implication, not asserted directly.
    assert af.check(x_diag, SYMMETRIC)


@pytest.mark.parametrize(
    "position, key",
    enumerate(
        [
            DIAGONAL,
            LOWER_TRIANGULAR,
            UPPER_TRIANGULAR,
            SYMMETRIC,
            POSITIVE_DEFINITE,
            ORTHOGONAL,
            SELECTION,
            PERMUTATION,
            UNIQUE_INDICES,
        ]
    ),
    ids=lambda p: getattr(p, "name", p),
)
def test_assume_keeps_released_positional_order(position, key):
    flags = [None] * position + [True]
    x = assume(pt.matrix("x", shape=(3, 3)), *flags)
    _, af = make_fgraph(x)
    assert af.get(x, key) is FactState.TRUE


def test_assume_no_kwargs_returns_input_unchanged():
    x = pt.matrix("x")
    assert assume(x) is x


def test_assume_conflict_with_inferred_fact_raises():
    e = pt.eye(5)
    e_not_diag = assume(e, diagonal=False)
    _, af = make_fgraph(e_not_diag)
    with pytest.raises(ConflictingAssumptionsError):
        af.get(e_not_diag, DIAGONAL)


def test_assume_valued_key_joins_declared_and_inferred(rank):
    x = pt.matrix("x")
    declared = assume(x, rank_bound=5)
    tightened = assume(RankBoundedOp(3)(x), rank_bound=5)
    _, af = make_fgraph(declared, tightened)
    assert af.get(x, rank) == math.inf
    assert af.get(declared, rank) == 5
    assert af.get(tightened, rank) == 3


def test_specify_assumptions_valued_round_trip(rank):
    op = SpecifyAssumptions({rank: 5, DIAGONAL: FactState.TRUE})
    assert SpecifyAssumptions(dict(op.assumptions)) == op
    assert str(op) == "SpecifyAssumptions{diagonal, rank_bound=5.0}"


def test_valued_key_rejects_bool(rank):
    with pytest.raises(TypeError, match="takes a float value"):
        rank.assume(pt.matrix("x"))
    with pytest.raises(TypeError, match="takes a float value"):
        SpecifyAssumptions({rank: True})
