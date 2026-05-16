import numpy as np
import pytest

import pytensor.tensor as pt
from pytensor.assumptions import DIAGONAL, FactState
from pytensor.assumptions.specify import assume
from pytensor.graph import rewrite_graph
from tests.assumptions.conftest import make_fgraph


def test_constant_identity_matrix_is_diagonal():
    """A literal Constant identity matrix is recognized as DIAGONAL via the
    Constant-data inference path. Reaches this case after constant-folding
    or ``scan_inline_invariant_constants`` collapses an ``Eye`` into a bare
    ``Constant``."""
    c = pt.constant(np.eye(5))
    _, af = make_fgraph(c)
    assert af.check(c, DIAGONAL)


def test_constant_non_diagonal_matrix_is_false():
    c = pt.constant(np.ones((5, 5)))
    _, af = make_fgraph(c)
    assert af.get(c, DIAGONAL) == FactState.FALSE


def test_constant_batched_diagonal_matrix_is_diagonal():
    data = np.broadcast_to(np.eye(4), (3, 4, 4)).copy()
    c = pt.constant(data)
    _, af = make_fgraph(c)
    assert af.check(c, DIAGONAL)


def test_constant_batched_partial_diagonal_is_false():
    """If any batch slice has off-diagonal entries, the whole tensor is NOT DIAGONAL."""
    data = np.broadcast_to(np.eye(4), (3, 4, 4)).copy()
    data[1, 0, 1] = 1.0
    c = pt.constant(data)
    _, af = make_fgraph(c)
    assert af.get(c, DIAGONAL) == FactState.FALSE


def test_constant_nonsquare_matrix_is_false():
    c = pt.constant(np.zeros((4, 5)))
    _, af = make_fgraph(c)
    assert af.get(c, DIAGONAL) == FactState.FALSE


@pytest.mark.parametrize(
    "data", [np.arange(5), np.array(3.0)], ids=["1d_vector", "0d_scalar"]
)
def test_constant_sub_matrix_ndim_is_false(data):
    """A scalar or vector is not a matrix, so it is not a diagonal matrix."""
    c = pt.constant(data)
    _, af = make_fgraph(c)
    assert af.get(c, DIAGONAL) == FactState.FALSE


def test_cholesky_of_diagonal_is_diagonal():
    v = pt.vector("v", shape=(3,))
    L = pt.linalg.cholesky(pt.diag(v), lower=True)
    _, af = make_fgraph(L)
    assert af.check(L, DIAGONAL)


def test_inv_propagates_diagonal():
    x = pt.matrix("x", shape=(3, 3))
    x_diag = assume(x, diagonal=True)
    inv_x = pt.linalg.inv(x_diag)
    _, af = make_fgraph(inv_x)
    assert af.check(inv_x, DIAGONAL)


def test_pinv_propagates_diagonal():
    x = pt.matrix("x", shape=(3, 3))
    x_diag = assume(x, diagonal=True)
    px = pt.linalg.pinv(x_diag)
    _, af = make_fgraph(px)
    assert af.check(px, DIAGONAL)


def test_block_diag_of_diagonal_blocks_is_diagonal():
    bd = pt.linalg.block_diag(pt.eye(3), pt.eye(4))
    _, af = make_fgraph(bd)
    assert af.check(bd, DIAGONAL)


def test_block_diag_of_generic_blocks_is_unknown():
    x = pt.matrix("x", shape=(3, 3))
    y = pt.matrix("y", shape=(4, 4))
    bd = pt.linalg.block_diag(x, y)
    _, af = make_fgraph(bd)
    assert af.get(bd, DIAGONAL) == FactState.UNKNOWN


def test_kron_of_eyes_is_diagonal():
    k = pt.linalg.kron(pt.eye(3), pt.eye(4))
    _, af = make_fgraph(k)
    assert af.check(k, DIAGONAL)


def test_kron_with_generic_input_not_diagonal():
    x = pt.matrix("x", shape=(3, 3))
    k = pt.linalg.kron(x, pt.eye(4))
    _, af = make_fgraph(k)
    assert af.get(k, DIAGONAL) == FactState.UNKNOWN


def test_matmul_diagonal_diagonal_is_diagonal():
    y = pt.eye(5) @ pt.diag(pt.ones(5))
    _, af = make_fgraph(y)
    assert af.check(y, DIAGONAL)


def test_matmul_diagonal_generic_is_unknown():
    x = pt.matrix("x", shape=(5, 5))
    y = pt.eye(5) @ x
    _, af = make_fgraph(y)
    assert af.get(y, DIAGONAL) == FactState.UNKNOWN


def test_dot_orthogonal_xxt_is_diagonal():
    x = pt.dmatrix("x")
    x_orth = assume(x, orthogonal=True)
    y = pt.dot(x_orth, x_orth.T)
    _, af = make_fgraph(y)
    assert af.check(y, DIAGONAL)


class TestSubtensorDiagonalPreservation:
    def test_set_diagonal_entries(self):
        d = pt.eye(5)
        idx = pt.arange(5)
        v = pt.vector("v", shape=(5,))
        y = pt.set_subtensor(d[idx, idx], v)
        _, af = make_fgraph(y)
        assert af.check(y, DIAGONAL)

    def test_inc_diagonal_entries(self):
        d = pt.eye(5)
        idx = pt.arange(5)
        v = pt.vector("v", shape=(5,))
        y = pt.inc_subtensor(d[idx, idx], v)
        _, af = make_fgraph(y)
        assert af.check(y, DIAGONAL)

    def test_set_scalar_diagonal_entry(self):
        d = pt.eye(5)
        i = pt.iscalar("i")
        y = rewrite_graph(pt.set_subtensor(d[i, i], 1.0), include=["merge"])
        _, af = make_fgraph(y)
        assert af.check(y, DIAGONAL)

    def test_set_batched_diagonal_entries(self):
        x = pt.tensor("x", shape=(3, 5, 5))
        idx = pt.arange(5)
        v = pt.matrix("v", shape=(3, 5))
        y = pt.set_subtensor(assume(x, diagonal=True)[:, idx, idx], v)
        _, af = make_fgraph(y)
        assert af.check(y, DIAGONAL)

    def test_set_off_diagonal_entries_is_unknown(self):
        d = pt.eye(5)
        v = pt.vector("v", shape=(5,))
        y = pt.set_subtensor(d[pt.arange(5), pt.arange(1, 6)], v)
        _, af = make_fgraph(y)
        assert af.get(y, DIAGONAL) == FactState.UNKNOWN

    def test_non_diagonal_base_is_unknown(self):
        x = pt.matrix("x", shape=(5, 5))
        v = pt.vector("v", shape=(5,))
        y = pt.set_subtensor(x[pt.arange(5), pt.arange(5)], v)
        _, af = make_fgraph(y)
        assert af.get(y, DIAGONAL) == FactState.UNKNOWN


def test_transpose_preserves_diagonal():
    e = pt.eye(5)
    _, af = make_fgraph(e.T)
    assert af.check(e.T, DIAGONAL)


def test_expand_dims_preserves_diagonal():
    e = pt.eye(5)
    e3d = e.dimshuffle("x", 0, 1)
    _, af = make_fgraph(e3d)
    assert af.check(e3d, DIAGONAL)


def test_transpose_of_generic_matrix_is_unknown():
    x = pt.matrix("x", shape=(3, 3))
    xT = x.T
    _, af = make_fgraph(xT)
    assert af.get(xT, DIAGONAL) == FactState.UNKNOWN


class TestElemwiseAssumptions:
    def test_mul_diagonal_by_anything_is_diagonal(self):
        x = pt.matrix("x", shape=(5, 5))
        y = pt.eye(5) * x
        _, af = make_fgraph(y)
        assert af.check(y, DIAGONAL)

    def test_add_diagonal_plus_diagonal(self):
        y = pt.eye(5) + pt.diag(pt.ones(5))
        _, af = make_fgraph(y)
        assert af.check(y, DIAGONAL)

    def test_add_diagonal_plus_generic_is_unknown(self):
        x = pt.matrix("x", shape=(5, 5))
        y = pt.eye(5) + x
        _, af = make_fgraph(y)
        assert not af.check(y, DIAGONAL)

    def test_sub_diagonal_minus_diagonal(self):
        y = pt.eye(5) - pt.diag(pt.ones(5))
        _, af = make_fgraph(y)
        assert af.check(y, DIAGONAL)

    def test_truediv_diagonal_over_anything_is_diagonal(self):
        x = pt.matrix("x", shape=(5, 5))
        y = pt.eye(5) / x
        _, af = make_fgraph(y)
        assert af.check(y, DIAGONAL)

    def test_truediv_anything_over_diagonal_is_unknown(self):
        x = pt.matrix("x", shape=(5, 5))
        y = x / pt.eye(5)
        _, af = make_fgraph(y)
        assert not af.check(y, DIAGONAL)

    @pytest.mark.parametrize(
        "transform",
        [
            pytest.param(lambda d: -d, id="neg"),
            pytest.param(lambda d: abs(d), id="abs"),
            pytest.param(lambda d: d**2, id="pow2"),
        ],
    )
    def test_zero_preserving_unary_preserves_diagonal(self, transform):
        y = transform(pt.eye(5))
        _, af = make_fgraph(y)
        assert af.check(y, DIAGONAL)
