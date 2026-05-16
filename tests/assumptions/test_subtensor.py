import pytest

import pytensor.tensor as pt
from pytensor.assumptions import (
    ALL_KEYS,
    POSITIVE_DEFINITE,
    SYMMETRIC,
    FactState,
)
from pytensor.assumptions.specify import assume
from tests.assumptions.conftest import make_fgraph


class TestSubtensorMatrixPropertyPropagation:
    """A ``Subtensor`` that leaves the trailing two axes alone preserves matrix properties."""

    @pytest.mark.parametrize("key", ALL_KEYS)
    def test_scalar_index_strips_leading_axis(self, key):
        x = pt.tensor3("x", shape=(5, 4, 4))
        x_tagged = assume(x, **{key.name: True})
        y = x_tagged[2]
        _, af = make_fgraph(y, inputs=[x])
        assert af.check(y, key)

    @pytest.mark.parametrize("key", ALL_KEYS)
    def test_explicit_full_slices(self, key):
        x = pt.tensor3("x", shape=(5, 4, 4))
        x_tagged = assume(x, **{key.name: True})
        y = x_tagged[2, :, :]
        _, af = make_fgraph(y, inputs=[x])
        assert af.check(y, key)

    @pytest.mark.parametrize("key", ALL_KEYS)
    def test_partial_slice_on_batch_axis(self, key):
        x = pt.tensor3("x", shape=(5, 4, 4))
        x_tagged = assume(x, **{key.name: True})
        y = x_tagged[1:3]
        _, af = make_fgraph(y, inputs=[x])
        assert af.check(y, key)

    def test_indexing_a_matrix_axis_breaks_property(self):
        x = pt.tensor3("x", shape=(5, 4, 4))
        x_sym = assume(x, symmetric=True)
        y = x_sym[2, :, 0]
        _, af = make_fgraph(y, inputs=[x])
        assert af.get(y, SYMMETRIC) == FactState.UNKNOWN

    def test_subtensor_after_expand_dims(self):
        """The user-reported pattern: ``Subtensor[i, :, :](ExpandDims[0](X))``.

        After expand_dims promotes the matrix to ``(1, n, n)``, indexing the
        leading axis recovers the original matrix; the property must survive
        the round trip.
        """
        x = pt.matrix("x", shape=(4, 4))
        x_psd = assume(x, positive_definite=True)
        x_3d = x_psd[None, :, :]
        y = x_3d[0]
        _, af = make_fgraph(y, inputs=[x])
        assert af.check(y, POSITIVE_DEFINITE)
        assert af.check(y, SYMMETRIC)
