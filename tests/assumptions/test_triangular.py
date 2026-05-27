import pytest

import pytensor.tensor as pt
from pytensor.assumptions import (
    DIAGONAL,
    LOWER_TRIANGULAR,
    UPPER_TRIANGULAR,
    FactState,
)
from pytensor.assumptions.specify import assume
from pytensor.tensor.basic import alloc_diag
from tests.assumptions.conftest import make_fgraph


@pytest.mark.parametrize(
    "offset, lower, upper, diagonal",
    [
        (0, FactState.TRUE, FactState.TRUE, FactState.TRUE),
        (1, FactState.FALSE, FactState.TRUE, FactState.FALSE),
        (-1, FactState.TRUE, FactState.FALSE, FactState.FALSE),
        (2, FactState.FALSE, FactState.TRUE, FactState.FALSE),
        (-2, FactState.TRUE, FactState.FALSE, FactState.FALSE),
    ],
)
def test_alloc_diag_offset_triangular(offset, lower, upper, diagonal):
    v = pt.vector("v", shape=(5,))
    d = alloc_diag(v, offset=offset, axis1=0, axis2=1)
    _, af = make_fgraph(d)
    assert af.get(d, LOWER_TRIANGULAR) == lower
    assert af.get(d, UPPER_TRIANGULAR) == upper
    assert af.get(d, DIAGONAL) == diagonal


@pytest.mark.parametrize(
    "lower, expected_true, expected_false",
    [
        (True, LOWER_TRIANGULAR, UPPER_TRIANGULAR),
        (False, UPPER_TRIANGULAR, LOWER_TRIANGULAR),
    ],
)
def test_cholesky_triangularity(lower, expected_true, expected_false):
    x = pt.matrix("x", shape=(3, 3))
    L = pt.linalg.cholesky(x, lower=lower)
    _, af = make_fgraph(L)
    assert af.check(L, expected_true)
    assert not af.check(L, expected_false)


def test_inv_propagates_lower_triangular():
    x = pt.matrix("x", shape=(3, 3))
    x_lower = assume(x, lower_triangular=True)
    inv_x = pt.linalg.inv(x_lower)
    _, af = make_fgraph(inv_x)
    assert af.check(inv_x, LOWER_TRIANGULAR)


@pytest.mark.parametrize("key", [LOWER_TRIANGULAR, UPPER_TRIANGULAR])
def test_kron_of_eyes_is_triangular(key):
    k = pt.linalg.kron(pt.eye(3), pt.eye(4))
    _, af = make_fgraph(k)
    assert af.check(k, key)


@pytest.mark.parametrize(
    "eye_args, lower, upper",
    [
        pytest.param((5, 5, 0), FactState.TRUE, FactState.TRUE, id="identity"),
        pytest.param((5, 5, -1), FactState.TRUE, FactState.FALSE, id="subdiagonal"),
        pytest.param((5, 5, 2), FactState.FALSE, FactState.TRUE, id="superdiagonal"),
        pytest.param((5, 3, 0), FactState.TRUE, FactState.TRUE, id="nonsquare"),
    ],
)
def test_eye_triangularity_follows_offset(eye_args, lower, upper):
    """Eye is lower-triangular iff its diagonal sits on or below the main one,
    upper-triangular iff on or above -- regardless of whether it is square."""
    e = pt.eye(*eye_args)
    _, af = make_fgraph(e)
    assert af.get(e, LOWER_TRIANGULAR) == lower
    assert af.get(e, UPPER_TRIANGULAR) == upper


def test_eye_symbolic_offset_is_unknown():
    k = pt.iscalar("k")
    e = pt.eye(5, 5, k)
    _, af = make_fgraph(e)
    assert af.get(e, LOWER_TRIANGULAR) == FactState.UNKNOWN
    assert af.get(e, UPPER_TRIANGULAR) == FactState.UNKNOWN


class TestTriangularElemwiseAssumptions:
    @pytest.mark.parametrize("key", [LOWER_TRIANGULAR, UPPER_TRIANGULAR])
    @pytest.mark.parametrize(
        "build, expected",
        [
            pytest.param(lambda a, b: a + b, True, id="add"),
            pytest.param(lambda a, b: a - b, True, id="sub"),
            pytest.param(lambda a, b: a * b, True, id="mul"),
            pytest.param(lambda a, b: 2.0 * a, True, id="scalar_mul"),
            pytest.param(lambda a, b: -a, True, id="neg"),
            pytest.param(lambda a, b: a**3, True, id="pow_const"),
            pytest.param(lambda a, b: pt.exp(a), False, id="non_zero_preserving_unary"),
        ],
    )
    def test_elemwise_propagation(self, key, build, expected):
        kwarg = {key.name: True}
        x = pt.matrix("x", shape=(5, 5))
        y = pt.matrix("y", shape=(5, 5))
        a = assume(x, **kwarg)
        b = assume(y, **kwarg)
        z = build(a, b)
        _, af = make_fgraph(z)
        assert af.check(z, key) is expected


@pytest.mark.parametrize("mode", ["full", "economic"])
def test_qr_r_is_upper_triangular(mode):
    x = pt.matrix("x", shape=(3, 3))
    Q, R = pt.linalg.qr(x, mode=mode)
    _, af = make_fgraph(Q, R)
    assert af.check(R, UPPER_TRIANGULAR)
    assert af.get(R, LOWER_TRIANGULAR) == FactState.UNKNOWN
    assert af.get(Q, UPPER_TRIANGULAR) == FactState.UNKNOWN


@pytest.mark.parametrize(
    "output, expected",
    [("complex", True), ("real", False)],
)
def test_schur_t_is_upper_triangular_only_for_complex(output, expected):
    # Complex Schur form is upper-triangular; real Schur form is only
    # quasi-upper-triangular (2x2 blocks), so it is not claimed.
    x = pt.matrix("x", shape=(3, 3))
    T, Z = pt.linalg.schur(x, output=output)
    _, af = make_fgraph(T, Z)
    assert af.check(T, UPPER_TRIANGULAR) is expected
    assert af.get(Z, UPPER_TRIANGULAR) == FactState.UNKNOWN


def test_qz_aa_and_bb_are_upper_triangular():
    a = pt.matrix("a", shape=(3, 3))
    b = pt.matrix("b", shape=(3, 3))
    AA, BB, Q, Z = pt.linalg.qz(a, b)
    _, af = make_fgraph(AA, BB, Q, Z)
    assert af.check(AA, UPPER_TRIANGULAR)
    assert af.check(BB, UPPER_TRIANGULAR)
    assert af.get(Q, UPPER_TRIANGULAR) == FactState.UNKNOWN
    assert af.get(Z, UPPER_TRIANGULAR) == FactState.UNKNOWN


def test_lu_l_is_lower_and_u_is_upper_triangular():
    x = pt.matrix("x", shape=(3, 3))
    P, L, U = pt.linalg.lu(x)
    _, af = make_fgraph(P, L, U)
    assert af.check(L, LOWER_TRIANGULAR)
    assert af.check(U, UPPER_TRIANGULAR)
    assert af.get(L, UPPER_TRIANGULAR) == FactState.UNKNOWN
    assert af.get(U, LOWER_TRIANGULAR) == FactState.UNKNOWN
    assert af.get(P, LOWER_TRIANGULAR) == FactState.UNKNOWN


def test_lu_permute_l_pl_is_not_lower_triangular():
    # With permute_l=True the outputs are (PL, U); PL is the product of a
    # permutation and a unit-lower-triangular matrix, so it is not claimed.
    x = pt.matrix("x", shape=(3, 3))
    PL, U = pt.linalg.lu(x, permute_l=True)
    _, af = make_fgraph(PL, U)
    assert af.get(PL, LOWER_TRIANGULAR) == FactState.UNKNOWN
    assert af.check(U, UPPER_TRIANGULAR)


class TestMatrixTransposeFlipsTriangle:
    """Matrix transpose maps lower-triangular to upper-triangular and vice versa."""

    @pytest.mark.parametrize(
        "asserted, flipped",
        [
            (LOWER_TRIANGULAR, UPPER_TRIANGULAR),
            (UPPER_TRIANGULAR, LOWER_TRIANGULAR),
        ],
    )
    def test_transpose_flips_triangle(self, asserted, flipped):
        x = pt.matrix("x", shape=(4, 4))
        x_tagged = assume(x, **{asserted.name: True})
        y = x_tagged.T
        _, af = make_fgraph(y)
        assert af.check(y, flipped)
        assert af.get(y, asserted) == FactState.UNKNOWN

    @pytest.mark.parametrize(
        "asserted, flipped",
        [
            (LOWER_TRIANGULAR, UPPER_TRIANGULAR),
            (UPPER_TRIANGULAR, LOWER_TRIANGULAR),
        ],
    )
    def test_batched_transpose_flips_triangle(self, asserted, flipped):
        x = pt.tensor("x", shape=(2, 3, 4, 4))
        x_tagged = assume(x, **{asserted.name: True})
        y = x_tagged.mT
        _, af = make_fgraph(y)
        assert af.check(y, flipped)
        assert af.get(y, asserted) == FactState.UNKNOWN

    @pytest.mark.parametrize("key", [LOWER_TRIANGULAR, UPPER_TRIANGULAR])
    def test_double_transpose_recovers_triangle(self, key):
        # Exercises the cross-key inference chain: the first transpose flips
        # via a feature.check on the input, and the second transpose flips
        # back via a feature.check on the freshly-inferred intermediate.
        x = pt.matrix("x", shape=(4, 4))
        x_tagged = assume(x, **{key.name: True})
        y = x_tagged.T.T
        _, af = make_fgraph(y)
        assert af.check(y, key)
