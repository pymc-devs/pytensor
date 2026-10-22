import numpy as np
import pytest

import pytensor.tensor as pt
from tests.sparsity.conftest import detected_and_numeric_patterns


matrix = pt.matrix("matrix")
row = pt.matrix("row", shape=(1, None))
mask = pt.constant(np.array([[0.0, 0.0, 1.0]]))
condition = pt.constant(np.array([[True, False, True]]))

CASES = {
    "matrix * row": ([matrix * row], [matrix, row]),
    "matrix * mask": ([matrix * mask], [matrix]),
    "mask / row": ([mask / row], [row]),
    "switch(condition, matrix, row)": (
        [pt.switch(condition, matrix, row)],
        [matrix, row],
    ),
    "floor(matrix) + row": ([pt.floor(matrix) + row], [matrix, row]),
    "sign(matrix) * row": ([pt.sign(matrix) * row], [matrix, row]),
    "matrix // row": ([matrix // row], [matrix, row]),
    "sin(matrix * mask) * row": ([pt.sin(matrix * mask) * row], [matrix, row]),
    "(matrix * mask + matrix * mask) * row": (
        [(matrix * mask + matrix * mask) * row],
        [matrix, row],
    ),
    "(matrix * mask - 2 * matrix * mask) * row": (
        [(matrix * mask - 2 * matrix * mask) * row],
        [matrix, row],
    ),
    "switch(condition, matrix * mask, 0) * row": (
        [pt.switch(condition, matrix * mask, 0.0) * row],
        [matrix, row],
    ),
    "(mask / row) * matrix": ([(mask / row) * matrix], [matrix, row]),
    "second(matrix, mask) * row": ([pt.second(matrix, mask) * row], [matrix, row]),
    # A float condition selects a branch but carries no derivative.
    "switch(matrix, 2 * row, row)": ([pt.switch(matrix, 2 * row, row)], [matrix, row]),
    # The condition depends on the data, so both branches stay in the pattern.
    "switch(matrix > 0, matrix, row)": (
        [pt.switch(matrix > 0, matrix, row)],
        [matrix, row],
    ),
}
CONSERVATIVE_NNZ = {"switch(matrix > 0, matrix, row)": 24}


@pytest.fixture(scope="module")
def patterns():
    rng = np.random.default_rng(0)
    values = [rng.uniform(0.5, 1.5, size=(4, 3)), rng.uniform(0.5, 1.5, size=(1, 3))]
    return detected_and_numeric_patterns(CASES, inputs=[matrix, row], values=values)


@pytest.mark.parametrize("title", CASES)
def test_elemwise_pattern(patterns, title):
    detected, numeric = patterns[title]
    assert not (numeric & ~detected).any()
    if title in CONSERVATIVE_NNZ:
        assert detected.sum() == CONSERVATIVE_NNZ[title]
    else:
        np.testing.assert_array_equal(detected, numeric)
