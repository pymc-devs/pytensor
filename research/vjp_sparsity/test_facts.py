import pytest

import pytensor.tensor as pt
from pytensor.graph.fg import FunctionGraph
from research.vjp_sparsity.facts import BOTH, COLUMNS, NEITHER, ROWS, orthonormal_sides


tall = pt.matrix("tall", shape=(5, 3))
square = pt.matrix("square", shape=(4, 4))
other_tall = pt.matrix("other_tall", shape=(3, 2))
tall_q = pt.linalg.qr(tall, mode="economic")[0]


@pytest.mark.parametrize(
    "variable, expected",
    [
        (tall_q, frozenset({COLUMNS})),
        (tall_q.mT, frozenset({ROWS})),
        (pt.linalg.qr(tall, mode="full")[0], BOTH),
        (pt.linalg.qr(square, mode="economic")[0], BOTH),
        (tall_q @ pt.linalg.qr(other_tall, mode="economic")[0], frozenset({COLUMNS})),
        (tall_q @ other_tall, NEITHER),
    ],
    ids=[
        "economic tall Q",
        "transpose",
        "full Q",
        "economic square Q",
        "product of column-orthonormal factors",
        "product with an unknown factor",
    ],
)
def test_orthonormal_sides(variable, expected):
    fgraph = FunctionGraph(outputs=[variable], clone=False)
    assert orthonormal_sides(fgraph, variable) == expected
