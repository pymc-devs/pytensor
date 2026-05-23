import numpy as np
import pytest

import pytensor.tensor as pt
from pytensor.assumptions import ORTHOGONAL, PERMUTATION, SELECTION, FactState
from pytensor.assumptions.specify import assume
from tests.assumptions.conftest import make_fgraph


@pytest.mark.parametrize(
    "n, m, expected",
    [
        (5, 5, FactState.TRUE),
        (5, 3, FactState.FALSE),
        (3, 5, FactState.FALSE),
    ],
    ids=["square", "tall", "wide"],
)
def test_eye_is_permutation_only_when_square(n, m, expected):
    e = pt.eye(n, m)
    _, af = make_fgraph(e)
    assert af.get(e, PERMUTATION) == expected


def test_offset_eye_is_not_permutation():
    e = pt.eye(5, 5, 1)
    _, af = make_fgraph(e)
    assert af.get(e, PERMUTATION) == FactState.FALSE


def test_constant_permutation():
    c = pt.constant(np.eye(4)[:, [2, 0, 3, 1]])
    _, af = make_fgraph(c)
    assert af.check(c, PERMUTATION)


def test_constant_batched_permutation():
    data = np.broadcast_to(np.eye(4)[:, [1, 0, 3, 2]], (3, 4, 4)).copy()
    c = pt.constant(data)
    _, af = make_fgraph(c)
    assert af.check(c, PERMUTATION)


@pytest.mark.parametrize(
    "data",
    [
        np.eye(3)[:, [0, 2]],  # non-square
        np.array([[2.0, 0.0], [0.0, 1.0]]),  # entry not in {0, 1}
        np.arange(4),  # not a matrix
    ],
    ids=["nonsquare", "non_binary", "vector"],
)
def test_non_permutation_constants_are_false(data):
    c = pt.constant(data)
    _, af = make_fgraph(c)
    assert af.get(c, PERMUTATION) == FactState.FALSE


def test_selection_with_duplicate_column_is_not_permutation():
    # Every column one-hot (a selection) but a repeated column leaves a row with two 1s
    # and a row with none, so it is not a permutation.
    c = pt.constant(np.array([[1.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]))
    _, af = make_fgraph(c)
    assert af.check(c, SELECTION)
    assert af.get(c, PERMUTATION) == FactState.FALSE


def test_transpose_is_permutation():
    # Unlike a general selection, a permutation is closed under transpose.
    p = assume(pt.matrix("x", shape=(4, 4)), permutation=True)
    _, af = make_fgraph(p.T)
    assert af.check(p.T, PERMUTATION)


@pytest.mark.parametrize(
    "new_order, expected",
    [
        (("x", 0, 1), FactState.TRUE),  # left: trailing two axes stay the matrix
        ((0, 1, "x"), FactState.UNKNOWN),  # right: matrix shifts off the trailing axes
    ],
    ids=["left", "right"],
)
def test_expand_dims_permutation(new_order, expected):
    p = assume(pt.matrix("p", shape=(4, 4)), permutation=True)
    expanded = p.dimshuffle(*new_order)
    _, af = make_fgraph(expanded)
    assert af.get(expanded, PERMUTATION) == expected


def test_product_of_permutations_is_permutation():
    a = assume(pt.matrix("a", shape=(4, 4)), permutation=True)
    b = assume(pt.matrix("b", shape=(4, 4)), permutation=True)
    _, af = make_fgraph(a @ b)
    assert af.check(a @ b, PERMUTATION)


def test_inverse_of_permutation_is_permutation():
    p = assume(pt.matrix("p", shape=(4, 4)), permutation=True)
    inv = pt.linalg.inv(p)
    _, af = make_fgraph(inv)
    assert af.check(inv, PERMUTATION)


def test_block_diag_of_permutations_is_permutation():
    bd = pt.linalg.block_diag(pt.eye(3), pt.constant(np.eye(2)[:, [1, 0]]))
    _, af = make_fgraph(bd)
    assert af.check(bd, PERMUTATION)


def test_kron_of_permutations_is_permutation():
    k = pt.linalg.kron(pt.eye(2), pt.constant(np.eye(3)[:, [2, 0, 1]]))
    _, af = make_fgraph(k)
    assert af.check(k, PERMUTATION)


def test_permutation_implies_selection_and_orthogonal():
    p = assume(pt.matrix("p", shape=(4, 4)), permutation=True)
    _, af = make_fgraph(p)
    assert af.check(p, SELECTION)
    assert af.check(p, ORTHOGONAL)


def test_not_selection_implies_not_permutation():
    # Contrapositive of PERMUTATION -> SELECTION.
    x = assume(pt.matrix("x", shape=(4, 4)), selection=False)
    _, af = make_fgraph(x)
    assert af.get(x, PERMUTATION) == FactState.FALSE


def test_batch_index_preserves_permutation():
    x = pt.tensor("x", shape=(3, 4, 4))
    xs = assume(x, permutation=True)
    _, af = make_fgraph(xs[0])
    assert af.check(xs[0], PERMUTATION)
