import pytest

import pytensor.tensor as pt
from pytensor.assumptions import (
    LOWER_TRIANGULAR,
    ORTHOGONAL,
    POSITIVE_DEFINITE,
    SYMMETRIC,
    UPPER_TRIANGULAR,
    FactState,
)
from pytensor.assumptions.specify import assume
from tests.assumptions.conftest import make_fgraph


class TestLeftExpandDimsPropagation:
    """Adding new broadcast dims on the left of a tagged matrix preserves the assumption."""

    @pytest.mark.parametrize(
        "key",
        [SYMMETRIC, POSITIVE_DEFINITE, ORTHOGONAL, LOWER_TRIANGULAR, UPPER_TRIANGULAR],
    )
    def test_single_left_axis(self, key):
        x = pt.matrix("x", shape=(4, 4))
        x_tagged = assume(x, **{key.name: True})
        y = x_tagged[None, :, :]
        _, af = make_fgraph(y)
        assert af.check(y, key)

    @pytest.mark.parametrize(
        "key",
        [SYMMETRIC, POSITIVE_DEFINITE, ORTHOGONAL, LOWER_TRIANGULAR, UPPER_TRIANGULAR],
    )
    def test_multiple_left_axes(self, key):
        x = pt.matrix("x", shape=(4, 4))
        x_tagged = assume(x, **{key.name: True})
        y = x_tagged[None, None, :, :]
        _, af = make_fgraph(y)
        assert af.check(y, key)

    def test_triangular_transpose_does_not_propagate(self):
        """Transpose swaps the triangle, so a lower-triangular matrix is no longer
        known to be lower-triangular."""
        x = pt.matrix("x", shape=(4, 4))
        x_lower = assume(x, lower_triangular=True)
        y = x_lower.T
        _, af = make_fgraph(y)
        assert af.get(y, LOWER_TRIANGULAR) == FactState.UNKNOWN

    def test_right_expand_dims_does_not_propagate(self):
        """Adding a broadcast dim on the right shifts the matrix axes; the property is lost."""
        x = pt.matrix("x", shape=(4, 4))
        x_sym = assume(x, symmetric=True)
        y = x_sym[:, :, None]
        _, af = make_fgraph(y)
        assert af.get(y, SYMMETRIC) == FactState.UNKNOWN


def test_permute_batch_axes_forwards_property():
    x = pt.tensor("x", shape=(2, 3, 4, 4))
    x_orth = assume(x, orthogonal=True)
    y = x_orth.dimshuffle(1, 0, 2, 3)
    _, af = make_fgraph(y)
    assert af.check(y, ORTHOGONAL)


def test_moving_core_axis_does_not_propagate():
    x = pt.tensor("x", shape=(2, 3, 4, 4))
    x_sym = assume(x, symmetric=True)
    y = x_sym.dimshuffle(0, 2, 1, 3)
    _, af = make_fgraph(y)
    assert af.get(y, SYMMETRIC) == FactState.UNKNOWN
