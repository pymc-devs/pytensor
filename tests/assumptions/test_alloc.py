import pytest

import pytensor.tensor as pt
from pytensor.assumptions import (
    ALL_KEYS,
    DIAGONAL,
    LOWER_TRIANGULAR,
    SYMMETRIC,
    UPPER_TRIANGULAR,
    FactState,
)
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


def test_eye_symbolic_same_shape_is_identity():
    n = pt.iscalar("n")
    e = pt.eye(n, n, 0)
    _, af = make_fgraph(e, inputs=[n])
    assert af.check(e, DIAGONAL)


def test_eye_symbolic_different_shapes_is_unknown():
    n = pt.iscalar("n")
    m = pt.iscalar("m")
    e = pt.eye(n, m, 0)
    _, af = make_fgraph(e, inputs=[n, m])
    assert af.get(e, DIAGONAL) == FactState.UNKNOWN


@pytest.mark.parametrize(
    "key", [DIAGONAL, SYMMETRIC, LOWER_TRIANGULAR, UPPER_TRIANGULAR]
)
def test_alloc_diag_properties(key):
    v = pt.vector("v", shape=(5,))
    d = pt.diag(v)
    _, af = make_fgraph(d, inputs=[v])
    assert af.check(d, key)


def test_zeros_matrix_is_diagonal():
    z = pt.zeros((5, 5))
    _, af = make_fgraph(z)
    assert af.check(z, DIAGONAL)


def test_ones_matrix_is_not_diagonal():
    o = pt.ones((5, 5))
    _, af = make_fgraph(o)
    assert af.get(o, DIAGONAL) == FactState.UNKNOWN


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
