import pytest

import pytensor.tensor as pt
from pytensor.assumptions import (
    DIAGONAL,
    POSITIVE_DEFINITE,
    ConflictingAssumptionsError,
    FactState,
)
from pytensor.assumptions.specify import assume
from pytensor.graph.fg import FunctionGraph
from pytensor.tensor.rewriting.assumptions import DrainSpecifyAssumptions
from tests.unittest_tools import assert_equal_computations


def _drain(*outputs):
    """Build a ``FunctionGraph`` in place and apply only the drain rewriter."""
    fgraph = FunctionGraph(outputs=list(outputs), clone=False)
    DrainSpecifyAssumptions().rewrite(fgraph)
    return fgraph


def test_unshared_marker_is_drained():
    x = pt.matrix("x", shape=(3, 3))
    fgraph = _drain(pt.linalg.inv(assume(x, diagonal=True)))
    assert_equal_computations(fgraph.outputs, [pt.linalg.inv(x)])
    assert fgraph.assumption_feature.check(x, DIAGONAL)


def test_marker_with_multiple_clients_is_drained():
    x = pt.matrix("x", shape=(3, 3))
    x_diag = assume(x, diagonal=True)
    fgraph = _drain(pt.linalg.inv(x_diag), pt.linalg.inv(x_diag.T))
    assert_equal_computations(fgraph.outputs, [pt.linalg.inv(x), pt.linalg.inv(x.T)])


@pytest.mark.parametrize("declared", [True, False], ids=["true", "false"])
def test_assertion_state_survives_drain(declared):
    x = pt.matrix("x", shape=(3, 3))
    fgraph = _drain(assume(x, diagonal=declared))
    assert_equal_computations(fgraph.outputs, [x])
    expected = FactState.TRUE if declared else FactState.FALSE
    assert fgraph.assumption_feature.get(x, DIAGONAL) is expected


def test_nested_assume_fully_drained():
    z = pt.matrix("z", shape=(3, 3))
    out = assume(assume(z, diagonal=True), positive_definite=True)
    fgraph = _drain(out)
    assert_equal_computations(fgraph.outputs, [z])
    feature = fgraph.assumption_feature
    assert feature.check(z, DIAGONAL)
    assert feature.check(z, POSITIVE_DEFINITE)


def test_aliased_input_drains_marker():
    """Assumptions are per-variable: the marker is drained even when the input
    has other clients."""
    y = pt.matrix("y", shape=(3, 3))
    fgraph = _drain(
        pt.linalg.inv(assume(y, diagonal=True)),
        pt.linalg.inv(y),
    )
    assert_equal_computations(
        fgraph.outputs,
        [pt.linalg.inv(y), pt.linalg.inv(y)],
    )
    assert fgraph.assumption_feature.check(y, DIAGONAL)


def test_conflicting_assertion_raises_during_drain():
    eye_not_diag = assume(pt.eye(3), diagonal=False)
    with pytest.raises(ConflictingAssumptionsError):
        _drain(eye_not_diag)
