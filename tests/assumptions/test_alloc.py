import numpy as np
import pytest

import pytensor.tensor as pt
from pytensor.assumptions import (
    ALL_KEYS,
    DIAGONAL,
    LOWER_TRIANGULAR,
    ORTHOGONAL,
    PERMUTATION,
    POSITIVE_DEFINITE,
    SYMMETRIC,
    UPPER_TRIANGULAR,
    FactState,
)
from pytensor.assumptions.specify import assume
from tests.assumptions.conftest import make_fgraph


def test_eye_identity_has_all_properties():
    e = pt.eye(5)
    _, af = make_fgraph(e)
    for key in ALL_KEYS:
        assert af.check(e, key), f"Eye should be {key}"


@pytest.mark.parametrize(
    "eye_args",
    [
        pytest.param(dict(n=5, k=1), id="offset"),
        pytest.param(dict(n=5, m=3), id="rectangular"),
    ],
)
def test_eye_non_identity_is_false(eye_args):
    e = pt.eye(**eye_args)
    _, af = make_fgraph(e)
    assert af.get(e, DIAGONAL) == FactState.FALSE


@pytest.mark.parametrize(
    "key, expected",
    [
        (SYMMETRIC, FactState.TRUE),
        (DIAGONAL, FactState.TRUE),
        (POSITIVE_DEFINITE, FactState.FALSE),
        (PERMUTATION, FactState.FALSE),
        (ORTHOGONAL, FactState.FALSE),
    ],
)
def test_square_eye_with_empty_band_is_zero_matrix(key, expected):
    """``eye(1, 1, k=1)`` has an empty band, so it is the all-zero matrix ``[[0.]]``."""
    e = pt.eye(1, 1, 1)
    _, af = make_fgraph(e)
    assert af.get(e, key) == expected


def test_empty_band_eye_merges_with_zero_constant_without_conflict():
    """An all-zero ``Eye``, constant-folded and merged onto an equal zero
    ``Constant``, must not produce conflicting SYMMETRIC evidence."""
    e = pt.eye(1, 1, 1)
    z = pt.constant(np.zeros((1, 1)))
    out = pt.add(e, z)
    fg, af = make_fgraph(out)
    af.get(e, SYMMETRIC)
    af.get(z, SYMMETRIC)
    fg.replace(e, z, reason="constant_fold+merge")
    # A conflict would raise ConflictingAssumptionsError here.
    assert af.get(z, SYMMETRIC) == FactState.TRUE


def test_eye_symbolic_same_shape_is_identity():
    n = pt.iscalar("n")
    e = pt.eye(n, n, 0)
    _, af = make_fgraph(e)
    assert af.check(e, DIAGONAL)


def test_eye_symbolic_different_shapes_is_unknown():
    n = pt.iscalar("n")
    m = pt.iscalar("m")
    e = pt.eye(n, m, 0)
    _, af = make_fgraph(e)
    assert af.get(e, DIAGONAL) == FactState.UNKNOWN


@pytest.mark.parametrize(
    "key", [DIAGONAL, SYMMETRIC, LOWER_TRIANGULAR, UPPER_TRIANGULAR]
)
def test_alloc_diag_properties(key):
    v = pt.vector("v", shape=(5,))
    d = pt.diag(v)
    _, af = make_fgraph(d)
    assert af.check(d, key)


def test_zeros_matrix_is_diagonal():
    z = pt.zeros((5, 5))
    _, af = make_fgraph(z)
    assert af.check(z, DIAGONAL)


def test_ones_matrix_is_not_diagonal():
    o = pt.ones((5, 5))
    _, af = make_fgraph(o)
    assert af.get(o, DIAGONAL) == FactState.FALSE


@pytest.mark.parametrize(
    "key", [DIAGONAL, SYMMETRIC, LOWER_TRIANGULAR, UPPER_TRIANGULAR]
)
def test_zeros_vector_is_not_matrix_property(key):
    z = pt.zeros((5,))
    _, af = make_fgraph(z)
    assert af.get(z, key) == FactState.UNKNOWN


@pytest.mark.parametrize(
    "key", [DIAGONAL, SYMMETRIC, LOWER_TRIANGULAR, UPPER_TRIANGULAR]
)
def test_nonsquare_zeros_matrix_is_not_matrix_property(key):
    z = pt.zeros((3, 4))
    _, af = make_fgraph(z)
    assert af.get(z, key) == FactState.UNKNOWN


def test_alloc_broadcast_matrix_value_forwards_property():
    m = pt.matrix("m", shape=(4, 4))
    m_pd = assume(m, positive_definite=True)
    y = pt.alloc(m_pd, 3, 4, 4)
    _, af = make_fgraph(y)
    assert af.check(y, POSITIVE_DEFINITE)


def test_alloc_broadcast_scalar_value_is_symmetric():
    """A square Alloc of a scalar fill is symmetric -- every entry equals that
    one value -- whether or not the value is statically known."""
    s = pt.scalar("s")
    y = pt.alloc(s, 4, 4)
    _, af = make_fgraph(y)
    assert af.check(y, SYMMETRIC)


def test_alloc_broadcast_vector_value_is_unknown():
    v = pt.vector("v", shape=(4,))
    y = pt.alloc(v, 4, 4)
    _, af = make_fgraph(y)
    assert af.get(y, SYMMETRIC) == FactState.UNKNOWN
