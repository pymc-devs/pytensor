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
    ],
    ids=["banded", "subdiagonal", "superdiagonal_2", "nonsquare"],
)
def test_constant_tridiagonality(data, expected):
    c = pt.constant(data)
    _, af = make_fgraph(c)
    assert af.get(c, TRIDIAGONAL) is expected


@pytest.mark.parametrize(
    "offset, expected",
    [(1, FactState.TRUE), (-1, FactState.TRUE), (2, FactState.UNKNOWN)],
)
def test_eye_and_alloc_diag_follow_offset(offset, expected):
    # A large offset is UNKNOWN, not FALSE: a small matrix's offset band is empty
    eye = pt.eye(5, k=offset)
    banded = alloc_diag(pt.dvector("d"), offset=offset)
    _, af = make_fgraph(eye, banded)
    assert af.get(eye, TRIDIAGONAL) is expected
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
