import numpy as np
import pytest

import pytensor.tensor as pt
from pytensor.assumptions import TRIDIAGONAL, FactState
from pytensor.assumptions.specify import assume
from pytensor.tensor.basic import alloc_diag
from tests.assumptions.conftest import make_fgraph


@pytest.mark.parametrize(
    "data, expected",
    [
        (np.diag(np.arange(1.0, 5.0)) + np.diag(np.ones(3), k=1), FactState.TRUE),
        (np.eye(4, k=-1), FactState.TRUE),
        (np.eye(4, k=2), FactState.FALSE),
        (np.ones((4, 5)), FactState.FALSE),
        (np.stack([np.eye(4), np.eye(4, k=1)]), FactState.TRUE),
        (np.stack([np.eye(4), np.eye(4, k=2)]), FactState.FALSE),
    ],
    ids=[
        "banded",
        "subdiagonal",
        "superdiagonal_2",
        "nonsquare",
        "batch_banded",
        "batch_one_outside_band",
    ],
)
def test_constant_tridiagonality(data, expected):
    c = pt.constant(data)
    _, af = make_fgraph(c)
    assert af.get(c, TRIDIAGONAL) is expected


@pytest.mark.parametrize(
    "n, offset, expected",
    [
        (5, 1, FactState.TRUE),
        (5, -1, FactState.TRUE),
        (5, 2, FactState.FALSE),
        (2, -3, FactState.TRUE),
    ],
    ids=["superdiagonal", "subdiagonal", "outside_band", "band_misses_shape"],
)
def test_eye_follows_offset(n, offset, expected):
    eye = pt.eye(n, k=offset)
    _, af = make_fgraph(eye)
    assert af.get(eye, TRIDIAGONAL) is expected


@pytest.mark.parametrize(
    "diag, offset, expected",
    [
        (pt.dvector("d"), 1, FactState.TRUE),
        (pt.dvector("d"), 2, FactState.UNKNOWN),
        (pt.constant(np.zeros(3)), 2, FactState.TRUE),
        (pt.constant(np.ones(3)), 2, FactState.FALSE),
    ],
    ids=["superdiagonal", "symbolic_outside_band", "zeros", "ones_outside_band"],
)
def test_alloc_diag_follows_offset(diag, offset, expected):
    banded = alloc_diag(diag, offset=offset)
    _, af = make_fgraph(banded)
    assert af.get(banded, TRIDIAGONAL) is expected


def test_op_rules_agree_with_constant_inference():
    """Structural rules must never contradict the constant checker, or constant
    folding joins the states into a compilation-killing CONFLICT."""
    rect = pt.eye(4, 5)
    small_fill = pt.alloc(3.0, 2, 2)
    big_fill = pt.alloc(3.0, 5, 5)
    _, af = make_fgraph(rect, small_fill, big_fill)
    assert af.get(rect, TRIDIAGONAL) is FactState.FALSE
    assert af.get(small_fill, TRIDIAGONAL) is FactState.TRUE
    assert af.get(big_fill, TRIDIAGONAL) is FactState.FALSE


def test_diagonal_implies_tridiagonal():
    x = assume(pt.dmatrix("x"), diagonal=True)
    _, af = make_fgraph(x)
    assert af.check(x, TRIDIAGONAL)


def test_transpose_expand_dims_and_batch_index_propagate():
    x = assume(pt.dmatrix("x"), tridiagonal=True)
    stack = assume(pt.dtensor3("xs"), tridiagonal=True)
    transposed = x.mT
    expanded = pt.expand_dims(x, 0)
    indexed = stack[0]
    _, af = make_fgraph(transposed, expanded, indexed)
    assert af.check(transposed, TRIDIAGONAL)
    assert af.check(expanded, TRIDIAGONAL)
    assert af.check(indexed, TRIDIAGONAL)


def test_cholesky_of_tridiagonal_is_tridiagonal():
    x = assume(pt.dmatrix("x"), tridiagonal=True, positive_definite=True)
    L = pt.linalg.cholesky(x)
    _, af = make_fgraph(L)
    assert af.check(L, TRIDIAGONAL)


def test_sum_and_block_diag_preserve_the_band():
    x = assume(pt.dmatrix("x"), tridiagonal=True)
    y = assume(pt.dmatrix("y"), tridiagonal=True)
    added = x + y
    scaled = 2.0 * x
    blocked = pt.linalg.block_diag(x, y)
    dense = x + pt.dmatrix("z")
    _, af = make_fgraph(added, scaled, blocked, dense)
    assert af.check(added, TRIDIAGONAL)
    assert af.check(scaled, TRIDIAGONAL)
    assert af.check(blocked, TRIDIAGONAL)
    assert not af.check(dense, TRIDIAGONAL)


def test_batch_inc_subtensor_preserves_the_band():
    base = assume(pt.tensor3("base", shape=(5, 4, 4)), tridiagonal=True)
    value = assume(pt.tensor3("value", shape=(2, 4, 4)), tridiagonal=True)
    y = pt.inc_subtensor(base[1:3], value)
    _, af = make_fgraph(y)
    assert af.check(y, TRIDIAGONAL)
