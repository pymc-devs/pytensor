import pytest

import pytensor.tensor as pt
from pytensor.assumptions import ORTHOGONAL, FactState
from pytensor.assumptions.specify import assume
from tests.assumptions.conftest import make_fgraph


def test_block_diag_of_orthogonal_blocks_is_orthogonal():
    # Mirrors a structural-time-series transition matrix: identity for the level
    # component, a 2x2 rotation for a cycle component (orthogonal by construction).
    theta = pt.scalar("theta")
    cycle = pt.stack(
        [
            pt.stack([pt.cos(theta), pt.sin(theta)]),
            pt.stack([-pt.sin(theta), pt.cos(theta)]),
        ]
    )
    cycle = assume(cycle, orthogonal=True)
    T = pt.linalg.block_diag(pt.eye(3), cycle)
    _, af = make_fgraph(T)
    assert af.check(T, ORTHOGONAL)


def test_block_diag_with_one_non_orthogonal_block_is_unknown():
    x = pt.matrix("x", shape=(3, 3))
    bd = pt.linalg.block_diag(pt.eye(3), x)
    _, af = make_fgraph(bd)
    assert af.get(bd, ORTHOGONAL) == FactState.UNKNOWN


def test_qr_full_mode_q_is_orthogonal():
    x = pt.matrix("x", shape=(3, 3))
    Q, R = pt.linalg.qr(x, mode="full")
    _, af = make_fgraph(Q, R)
    assert af.check(Q, ORTHOGONAL)
    assert af.get(R, ORTHOGONAL) == FactState.UNKNOWN


@pytest.mark.parametrize("mode", ["economic", "reduced"])
def test_qr_reduced_mode_q_is_not_claimed_orthogonal(mode):
    # In economic/reduced mode Q has orthonormal columns but is not square,
    # so orthogonality is not claimed.
    x = pt.matrix("x", shape=(5, 3))
    Q, R = pt.linalg.qr(x, mode=mode)
    _, af = make_fgraph(Q, R)
    assert af.get(Q, ORTHOGONAL) == FactState.UNKNOWN
    assert af.get(R, ORTHOGONAL) == FactState.UNKNOWN


def test_svd_full_matrices_u_and_v_are_orthogonal():
    x = pt.matrix("x", shape=(3, 3))
    U, S, V = pt.linalg.svd(x, full_matrices=True, compute_uv=True)
    _, af = make_fgraph(U, S, V)
    assert af.check(U, ORTHOGONAL)
    assert af.check(V, ORTHOGONAL)
    assert af.get(S, ORTHOGONAL) == FactState.UNKNOWN


def test_svd_reduced_u_and_v_are_not_claimed_orthogonal():
    # With full_matrices=False, U and V have orthonormal columns/rows but are
    # not square, so orthogonality is not claimed.
    x = pt.matrix("x", shape=(5, 3))
    U, S, V = pt.linalg.svd(x, full_matrices=False, compute_uv=True)
    _, af = make_fgraph(U, S, V)
    assert af.get(U, ORTHOGONAL) == FactState.UNKNOWN
    assert af.get(V, ORTHOGONAL) == FactState.UNKNOWN


def test_eigh_eigenvectors_are_orthogonal():
    x = pt.matrix("x", shape=(3, 3))
    w, v = pt.linalg.eigh(x)
    _, af = make_fgraph(w, v)
    assert af.check(v, ORTHOGONAL)
    assert af.get(w, ORTHOGONAL) == FactState.UNKNOWN


def test_schur_z_is_orthogonal():
    x = pt.matrix("x", shape=(3, 3))
    T, Z = pt.linalg.schur(x)
    _, af = make_fgraph(T, Z)
    assert af.check(Z, ORTHOGONAL)
    assert af.get(T, ORTHOGONAL) == FactState.UNKNOWN


def test_qz_q_and_z_are_orthogonal():
    a = pt.matrix("a", shape=(3, 3))
    b = pt.matrix("b", shape=(3, 3))
    AA, BB, Q, Z = pt.linalg.qz(a, b)
    _, af = make_fgraph(AA, BB, Q, Z)
    assert af.check(Q, ORTHOGONAL)
    assert af.check(Z, ORTHOGONAL)
    assert af.get(AA, ORTHOGONAL) == FactState.UNKNOWN
    assert af.get(BB, ORTHOGONAL) == FactState.UNKNOWN
