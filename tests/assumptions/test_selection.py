import numpy as np
import pytest

import pytensor.tensor as pt
from pytensor.assumptions import SELECTION, FactState
from pytensor.assumptions.specify import assume
from tests.assumptions.conftest import make_fgraph


def test_identity_eye_is_selection():
    e = pt.eye(5)
    _, af = make_fgraph(e)
    assert af.check(e, SELECTION)


@pytest.mark.parametrize(
    "n, m, expected",
    [
        (5, 3, FactState.TRUE),  # tall: columns are distinct unit vectors
        (3, 5, FactState.FALSE),  # wide: trailing columns are all zero
    ],
    ids=["tall", "wide"],
)
def test_nonsquare_eye_selection(n, m, expected):
    e = pt.eye(n, m)
    _, af = make_fgraph(e)
    assert af.get(e, SELECTION) == expected


def test_offset_eye_is_not_selection():
    e = pt.eye(5, 5, 1)
    _, af = make_fgraph(e)
    assert af.get(e, SELECTION) == FactState.FALSE


def test_eye_column_index_is_selection():
    idx = pt.lvector("idx")
    S = pt.eye(5)[:, idx]
    _, af = make_fgraph(S)
    assert af.check(S, SELECTION)


def test_eye_column_constant_index_is_selection():
    S = pt.eye(5)[:, np.array([0, 2, 4])]
    _, af = make_fgraph(S)
    assert af.check(S, SELECTION)


def test_eye_column_slice_is_selection():
    S = pt.eye(5)[:, 1:4]
    _, af = make_fgraph(S)
    assert af.check(S, SELECTION)


def test_eye_row_index_is_not_claimed_selection():
    # Selecting rows of the identity leaves all-zero columns, so it is not a
    # selection matrix; the inference declines to claim it (UNKNOWN, not FALSE).
    idx = pt.lvector("idx")
    S = pt.eye(5)[idx]
    _, af = make_fgraph(S)
    assert af.get(S, SELECTION) == FactState.UNKNOWN


def test_constant_selection_matrix():
    c = pt.constant(np.eye(6)[:, [0, 2, 5, 1]])
    _, af = make_fgraph(c)
    assert af.check(c, SELECTION)


def test_constant_batched_selection_matrix():
    data = np.broadcast_to(np.eye(4)[:, [0, 2, 3]], (3, 4, 3)).copy()
    c = pt.constant(data)
    _, af = make_fgraph(c)
    assert af.check(c, SELECTION)


@pytest.mark.parametrize(
    "data",
    [
        np.full((3, 3), 1.0),  # not one-hot per column
        np.array([[1.0, 0.0], [1.0, 0.0]]),  # a zero column
        np.array([[2.0, 0.0], [0.0, 1.0]]),  # entry not in {0, 1}
        np.array([[0.5, 0.0], [0.5, 1.0]]),  # columns sum to 1 but are not binary
        np.arange(5),  # not a matrix
    ],
    ids=["all_ones", "zero_column", "non_binary", "fractional_column", "vector"],
)
def test_non_selection_constants_are_false(data):
    c = pt.constant(data)
    _, af = make_fgraph(c)
    assert af.get(c, SELECTION) == FactState.FALSE


def test_block_diag_of_selections_is_selection():
    bd = pt.linalg.block_diag(pt.eye(3), pt.eye(2)[:, [0]])
    _, af = make_fgraph(bd)
    assert af.check(bd, SELECTION)


def test_block_diag_with_generic_block_is_unknown():
    x = pt.matrix("x", shape=(3, 3))
    bd = pt.linalg.block_diag(pt.eye(3), x)
    _, af = make_fgraph(bd)
    assert af.get(bd, SELECTION) == FactState.UNKNOWN


def test_kron_of_selections_is_selection():
    k = pt.linalg.kron(pt.eye(3), pt.eye(2))
    _, af = make_fgraph(k)
    assert af.check(k, SELECTION)


def test_diagonal_is_not_claimed_selection():
    # A general diagonal matrix (e.g. diag([2, 3])) is not a selection matrix.
    d = pt.diag(pt.as_tensor([2.0, 3.0]))
    _, af = make_fgraph(d)
    assert not af.check(d, SELECTION)


def test_assume_selection():
    x = pt.matrix("x", shape=(5, 3))
    xs = assume(x, selection=True)
    _, af = make_fgraph(xs)
    assert af.check(xs, SELECTION)


def test_transpose_is_not_propagated():
    # S.T is not a selection matrix in general (its columns may be all-zero), so
    # transposition does not carry the property.
    idx = pt.lvector("idx")
    S = pt.eye(5)[:, idx]
    _, af = make_fgraph(S.T)
    assert af.get(S.T, SELECTION) == FactState.UNKNOWN


@pytest.mark.parametrize(
    "new_order, expected",
    [
        (("x", 0, 1), FactState.TRUE),  # left: trailing two axes stay the matrix
        (
            (0, 1, "x"),
            FactState.UNKNOWN,
        ),  # right: matrix shifts to (k, 1), not a selection
    ],
    ids=["left", "right"],
)
def test_expand_dims_selection(new_order, expected):
    idx = pt.lvector("idx")
    expanded = (pt.eye(5)[:, idx]).dimshuffle(*new_order)
    _, af = make_fgraph(expanded)
    assert af.get(expanded, SELECTION) == expected


class TestSubtensorPropagation:
    def test_batch_index_preserves_selection(self):
        x = pt.tensor("x", shape=(3, 5, 2))
        xs = assume(x, selection=True)
        _, af = make_fgraph(xs[0])
        assert af.check(xs[0], SELECTION)

    def test_set_batch_slice_preserves_selection(self):
        x = pt.tensor("x", shape=(3, 5, 2))
        xs = assume(x, selection=True)
        v = pt.tensor("v", shape=(5, 2))
        vs = assume(v, selection=True)
        y = pt.set_subtensor(xs[0], vs)
        _, af = make_fgraph(y)
        assert af.check(y, SELECTION)

    def test_inc_is_not_selection(self):
        # Selection is not closed under addition, so an inc write breaks it.
        x = pt.tensor("x", shape=(3, 5, 2))
        xs = assume(x, selection=True)
        v = pt.tensor("v", shape=(5, 2))
        vs = assume(v, selection=True)
        y = pt.inc_subtensor(xs[0], vs)
        _, af = make_fgraph(y)
        assert af.get(y, SELECTION) == FactState.UNKNOWN
