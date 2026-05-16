import numpy as np
import pytest

import pytensor.tensor as pt
from pytensor.assumptions import DIAGONAL, LOWER_TRIANGULAR, UPPER_TRIANGULAR, FactState
from tests.assumptions.conftest import make_fgraph


@pytest.mark.parametrize("key", [DIAGONAL, LOWER_TRIANGULAR, UPPER_TRIANGULAR])
def test_add_broadcast_nonzero_scalar_breaks_zero_pattern(key):
    diag_like = pt.alloc(np.float64(0), 5, 5)
    add_with_nonzero = pt.constant(np.array([[1.0]])) + diag_like
    add_with_zero = pt.constant(np.array([[0.0]])) + diag_like

    _, af = make_fgraph(add_with_nonzero, add_with_zero)
    assert af.get(add_with_nonzero, key) == FactState.UNKNOWN
    assert af.check(add_with_zero, key)
