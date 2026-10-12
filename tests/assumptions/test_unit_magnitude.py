import numpy as np
import pytest

import pytensor.tensor as pt
from pytensor.assumptions import (
    UNIT_MAGNITUDE,
    ConflictingAssumptionsError,
    FactState,
    assume,
)
from pytensor.graph.rewriting.basic import dfs_rewriter
from pytensor.tensor.rewriting.assumptions import local_drain_specify_assumptions
from tests.assumptions.conftest import make_fgraph


@pytest.mark.parametrize(
    "value, expected",
    [
        ([1, -1, 1j], FactState.TRUE),
        ([1, 0], FactState.FALSE),
        ([1, np.nan], FactState.FALSE),
    ],
)
def test_constant_unit_magnitude(value, expected):
    x = pt.as_tensor(value)
    _, feature = make_fgraph(x)
    assert feature.get(x, UNIT_MAGNITUDE) is expected


@pytest.mark.parametrize("dtype", ["float64", "complex128"])
def test_unit_magnitude_propagation(dtype):
    x = assume(pt.vector(dtype=dtype), unit_magnitude=True)
    outputs = [x, -x, abs(x), x.conj(), pt.sqr(x), x[:, None], x.reshape((1, -1))]
    outputs.append(x.astype("complex128"))
    _, feature = make_fgraph(*outputs)
    assert all(feature.check(out, UNIT_MAGNITUDE) for out in outputs)


def test_unit_magnitude_not_preserved_by_lossy_casts():
    x = assume(pt.zvector(), unit_magnitude=True)
    y = assume(pt.dvector(), unit_magnitude=True)
    outputs = [x.real, y.astype("uint8")]
    _, feature = make_fgraph(*outputs)
    assert all(feature.get(out, UNIT_MAGNITUDE) is FactState.UNKNOWN for out in outputs)


def test_drain_refreshes_unknown():
    x = pt.dvector()
    expanded = x[:, None]
    y = expanded.astype("complex128")
    fgraph, feature = make_fgraph(y)
    assert feature.get(y, UNIT_MAGNITUDE) is FactState.UNKNOWN
    fgraph.change_node_input(expanded.owner, 0, assume(x, unit_magnitude=True))
    drain = dfs_rewriter(local_drain_specify_assumptions)
    drain.rewrite(fgraph)
    assert feature.check(y, UNIT_MAGNITUDE)
    with pytest.raises(ConflictingAssumptionsError):
        local_drain_specify_assumptions.transform(
            fgraph, assume(x, unit_magnitude=False).owner
        )
