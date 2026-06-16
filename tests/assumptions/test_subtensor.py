import pytest

import pytensor.tensor as pt
from pytensor.assumptions import (
    DIAGONAL,
    LOWER_TRIANGULAR,
    MATRIX_KEYS,
    ORTHOGONAL,
    POSITIVE_DEFINITE,
    SYMMETRIC,
    FactState,
)
from pytensor.assumptions.specify import assume
from tests.assumptions.conftest import make_fgraph


class TestSubtensorMatrixPropertyPropagation:
    """A ``Subtensor`` that leaves the trailing two axes alone preserves matrix properties."""

    @pytest.mark.parametrize("key", MATRIX_KEYS)
    def test_scalar_index_strips_leading_axis(self, key):
        x = pt.tensor3("x", shape=(5, 4, 4))
        x_tagged = assume(x, **{key.name: True})
        y = x_tagged[2]
        _, af = make_fgraph(y)
        assert af.check(y, key)

    @pytest.mark.parametrize("key", MATRIX_KEYS)
    def test_explicit_full_slices(self, key):
        x = pt.tensor3("x", shape=(5, 4, 4))
        x_tagged = assume(x, **{key.name: True})
        y = x_tagged[2, :, :]
        _, af = make_fgraph(y)
        assert af.check(y, key)

    @pytest.mark.parametrize("key", MATRIX_KEYS)
    def test_partial_slice_on_batch_axis(self, key):
        x = pt.tensor3("x", shape=(5, 4, 4))
        x_tagged = assume(x, **{key.name: True})
        y = x_tagged[1:3]
        _, af = make_fgraph(y)
        assert af.check(y, key)

    def test_indexing_a_matrix_axis_breaks_property(self):
        x = pt.tensor3("x", shape=(5, 4, 4))
        x_sym = assume(x, symmetric=True)
        y = x_sym[2, :, 0]
        _, af = make_fgraph(y)
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
        _, af = make_fgraph(y)
        assert af.check(y, POSITIVE_DEFINITE)
        assert af.check(y, SYMMETRIC)


def test_incsubtensor_full_slice_set_forwards_value():
    base = pt.tensor3("base", shape=(5, 4, 4))
    value = pt.tensor3("value", shape=(5, 4, 4))
    value_diag = assume(value, diagonal=True)
    y = pt.set_subtensor(base[:], value_diag)
    _, af = make_fgraph(y)
    assert af.check(y, DIAGONAL)


def test_incsubtensor_partial_batch_set_with_both_tagged_forwards():
    base = pt.tensor3("base", shape=(5, 4, 4))
    value = pt.tensor3("value", shape=(2, 4, 4))
    base_sym = assume(base, symmetric=True)
    value_sym = assume(value, symmetric=True)
    y = pt.set_subtensor(base_sym[1:3], value_sym)
    _, af = make_fgraph(y)
    assert af.check(y, SYMMETRIC)


def test_incsubtensor_partial_batch_set_with_untagged_base_is_unknown():
    base = pt.tensor3("base", shape=(5, 4, 4))
    value = pt.tensor3("value", shape=(2, 4, 4))
    value_orth = assume(value, orthogonal=True)
    y = pt.set_subtensor(base[1:3], value_orth)
    _, af = make_fgraph(y)
    assert af.get(y, ORTHOGONAL) == FactState.UNKNOWN


def test_incsubtensor_partial_batch_inc_with_both_tagged_forwards():
    base = pt.tensor3("base", shape=(5, 4, 4))
    value = pt.tensor3("value", shape=(2, 4, 4))
    base_pd = assume(base, positive_definite=True)
    value_pd = assume(value, positive_definite=True)
    y = pt.inc_subtensor(base_pd[1:3], value_pd)
    _, af = make_fgraph(y)
    assert af.check(y, POSITIVE_DEFINITE)


def test_incsubtensor_partial_batch_inc_with_untagged_base_is_unknown():
    base = pt.tensor3("base", shape=(5, 4, 4))
    value = pt.tensor3("value", shape=(2, 4, 4))
    value_lower = assume(value, lower_triangular=True)
    y = pt.inc_subtensor(base[1:3], value_lower)
    _, af = make_fgraph(y)
    assert af.get(y, LOWER_TRIANGULAR) == FactState.UNKNOWN


def test_incsubtensor_batch_inc_orthogonal_is_unknown():
    """Orthogonality is not closed under addition, so ``inc`` never forwards it"""
    base = pt.tensor3("base", shape=(5, 4, 4))
    value = pt.tensor3("value", shape=(2, 4, 4))
    base_orth = assume(base, orthogonal=True)
    value_orth = assume(value, orthogonal=True)
    y = pt.inc_subtensor(base_orth[1:3], value_orth)
    _, af = make_fgraph(y)
    assert af.get(y, ORTHOGONAL) == FactState.UNKNOWN


def test_incsubtensor_index_reaching_core_axis_is_unknown():
    base = pt.tensor3("base", shape=(5, 4, 4))
    value = pt.tensor3("value", shape=(2, 4, 2))
    base_sym = assume(base, symmetric=True)
    value_sym = assume(value, symmetric=True)
    y = pt.set_subtensor(base_sym[1:3, :, 0:2], value_sym)
    _, af = make_fgraph(y)
    assert af.get(y, SYMMETRIC) == FactState.UNKNOWN
