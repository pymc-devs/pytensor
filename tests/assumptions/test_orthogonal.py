import pytest

import pytensor.tensor as pt
from pytensor.assumptions import ORTHOGONAL, ConflictingAssumptionsError
from pytensor.assumptions.specify import SpecifyAssumptions, assume
from pytensor.printing import debugprint
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
    assert af.get(bd, ORTHOGONAL) == ORTHOGONAL.unknown


def test_qr_full_mode_q_is_orthogonal():
    x = pt.matrix("x", shape=(3, 3))
    Q, R = pt.linalg.qr(x, mode="full")
    _, af = make_fgraph(Q, R)
    assert af.check(Q, ORTHOGONAL)
    assert af.get(R, ORTHOGONAL) == ORTHOGONAL.unknown


@pytest.mark.parametrize(
    "shape, expected",
    [
        ((5, 3), frozenset({"columns"})),
        ((3, 3), ORTHOGONAL.true_value),
        ((3, 5), ORTHOGONAL.true_value),
        ((None, None), frozenset({"columns"})),
    ],
    ids=["tall", "square", "wide", "dynamic"],
)
def test_qr_economic_q_has_orthonormal_columns(shape, expected):
    x = pt.matrix("x", shape=shape)
    Q, R = pt.linalg.qr(x, mode="economic")
    _, af = make_fgraph(Q, R)
    assert af.get(Q, ORTHOGONAL) == expected
    assert af.get(R, ORTHOGONAL) == ORTHOGONAL.unknown


def test_transpose_swaps_orthonormal_sides():
    Q, _ = pt.linalg.qr(pt.matrix("x", shape=(5, 3)), mode="economic")
    refuted = assume(
        pt.matrix("y", shape=(5, 3)), orthogonal={"columns", "not_orthogonal"}
    )
    q_t, refuted_t, batched_q_t = Q.mT, refuted.mT, Q[None].mT
    _, af = make_fgraph(q_t, refuted_t, batched_q_t)
    assert af.get(q_t, ORTHOGONAL) == {"rows"}
    assert af.get(refuted_t, ORTHOGONAL) == {"rows", "not_orthogonal"}
    assert af.get(batched_q_t, ORTHOGONAL) == {"rows"}


def test_product_keeps_shared_orthonormal_sides():
    tall_q, _ = pt.linalg.qr(pt.matrix("x", shape=(5, 3)), mode="economic")
    other_q, _ = pt.linalg.qr(pt.matrix("y", shape=(3, 2)), mode="economic")
    refuted = assume(pt.matrix("z", shape=(3, 2)), orthogonal=False)
    orthogonal_u = assume(pt.matrix("u"), orthogonal=True)
    orthogonal_v = assume(pt.matrix("v"), orthogonal=True)
    full = orthogonal_u @ orthogonal_v
    columns = tall_q @ other_q
    rows = other_q.mT @ tall_q.mT
    mixed = tall_q.mT @ tall_q
    unknown = tall_q @ pt.matrix("w", shape=(3, 2))
    with_refuted = tall_q @ refuted
    _, af = make_fgraph(full, columns, rows, mixed, unknown, with_refuted)
    assert af.get(full, ORTHOGONAL) == ORTHOGONAL.true_value
    assert af.get(columns, ORTHOGONAL) == {"columns"}
    assert af.get(rows, ORTHOGONAL) == {"rows"}
    assert af.get(mixed, ORTHOGONAL) == ORTHOGONAL.unknown
    assert af.get(unknown, ORTHOGONAL) == ORTHOGONAL.unknown
    assert af.get(with_refuted, ORTHOGONAL) == ORTHOGONAL.unknown


def test_svd_full_matrices_u_and_v_are_orthogonal():
    x = pt.matrix("x", shape=(3, 3))
    U, S, V = pt.linalg.svd(x, full_matrices=True, compute_uv=True)
    _, af = make_fgraph(U, S, V)
    assert af.check(U, ORTHOGONAL)
    assert af.check(V, ORTHOGONAL)
    assert af.get(S, ORTHOGONAL) == ORTHOGONAL.unknown


def test_svd_reduced_u_and_v_are_not_claimed_orthogonal():
    # With full_matrices=False, U and V have orthonormal columns/rows but are
    # not square, so orthogonality is not claimed.
    x = pt.matrix("x", shape=(5, 3))
    U, S, V = pt.linalg.svd(x, full_matrices=False, compute_uv=True)
    _, af = make_fgraph(U, S, V)
    assert af.get(U, ORTHOGONAL) == ORTHOGONAL.unknown
    assert af.get(V, ORTHOGONAL) == ORTHOGONAL.unknown


def test_eigh_eigenvectors_are_orthogonal():
    x = pt.matrix("x", shape=(3, 3))
    w, v = pt.linalg.eigh(x)
    _, af = make_fgraph(w, v)
    assert af.check(v, ORTHOGONAL)
    assert af.get(w, ORTHOGONAL) == ORTHOGONAL.unknown


def test_schur_z_is_orthogonal():
    x = pt.matrix("x", shape=(3, 3))
    T, Z = pt.linalg.schur(x)
    _, af = make_fgraph(T, Z)
    assert af.check(Z, ORTHOGONAL)
    assert af.get(T, ORTHOGONAL) == ORTHOGONAL.unknown


def test_qz_q_and_z_are_orthogonal():
    a = pt.matrix("a", shape=(3, 3))
    b = pt.matrix("b", shape=(3, 3))
    AA, BB, Q, Z = pt.linalg.qz(a, b)
    _, af = make_fgraph(AA, BB, Q, Z)
    assert af.check(Q, ORTHOGONAL)
    assert af.check(Z, ORTHOGONAL)
    assert af.get(AA, ORTHOGONAL) == ORTHOGONAL.unknown
    assert af.get(BB, ORTHOGONAL) == ORTHOGONAL.unknown


@pytest.mark.parametrize(
    "shape, expected",
    [
        ((3, 3), ORTHOGONAL.true_value),
        ((5, 3), frozenset({"columns"})),
        ((None, None), frozenset({"columns"})),
    ],
    ids=["square", "tall", "dynamic"],
)
def test_one_orthonormal_side_upgrades_only_when_square(shape, expected):
    x = assume(pt.matrix("x", shape=shape), orthogonal="columns")
    _, af = make_fgraph(x)
    assert af.get(x, ORTHOGONAL) == expected


def test_square_with_one_side_and_not_orthogonal_conflicts():
    x = assume(pt.matrix("x", shape=(3, 3)), orthogonal={"rows", "not_orthogonal"})
    _, af = make_fgraph(x)
    with pytest.raises(ConflictingAssumptionsError):
        af.get(x, ORTHOGONAL)


def test_debugprint_marks_conflicted_orthogonal_beside_refuted_permutation():
    x = assume(
        pt.matrix("x", shape=(3, 3)),
        permutation=False,
        orthogonal={"rows", "not_orthogonal"},
    )
    printed = debugprint(x, print_assumptions=True, file="str")
    assert "orth=CONFLICT" in printed
    assert "!perm" in printed


def test_orthogonal_facts_print_by_side():
    op = SpecifyAssumptions({ORTHOGONAL: {"columns", "not_orthogonal"}})
    assert str(op) == "SpecifyAssumptions{orthogonal=columns, !orthogonal}"


def test_unknown_orthogonal_fact_raises():
    with pytest.raises(ValueError, match="Unknown orthogonality facts"):
        assume(pt.matrix("x"), orthogonal="diagonal")
