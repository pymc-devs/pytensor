import pytest

import pytensor.tensor as pt
from pytensor.assumptions import POSITIVE_DEFINITE, SYMMETRIC, FactState
from pytensor.assumptions.specify import assume
from tests.assumptions.conftest import make_fgraph


def test_inv_propagates_positive_definite():
    x = pt.matrix("x", shape=(3, 3))
    x_psd = assume(x, positive_definite=True)
    inv_x = pt.linalg.inv(x_psd)
    _, af = make_fgraph(inv_x)
    assert af.check(inv_x, POSITIVE_DEFINITE)


def test_kron_of_eyes_is_positive_definite():
    k = pt.linalg.kron(pt.eye(3), pt.eye(4))
    _, af = make_fgraph(k)
    assert af.check(k, POSITIVE_DEFINITE)


def test_dot_xxt_is_psd_and_symmetric():
    x = pt.matrix("x")
    gram = pt.dot(x, x.T)
    _, af = make_fgraph(gram)
    assert af.check(gram, POSITIVE_DEFINITE)
    assert af.check(gram, SYMMETRIC)


def test_dot_xtx_is_psd_and_symmetric():
    x = pt.matrix("x")
    gram = pt.dot(x.T, x)
    _, af = make_fgraph(gram)
    assert af.check(gram, POSITIVE_DEFINITE)
    assert af.check(gram, SYMMETRIC)


@pytest.mark.parametrize(
    "hermitian_transpose",
    [
        pytest.param(lambda x: x.conj().T, id="conj_then_T"),
        pytest.param(lambda x: x.T.conj(), id="T_then_conj"),
    ],
)
def test_dot_xxH_complex_is_psd(hermitian_transpose):
    x = pt.cmatrix("x")
    gram = pt.dot(x, hermitian_transpose(x))
    _, af = make_fgraph(gram)
    assert af.check(gram, POSITIVE_DEFINITE)


class TestKalmanPSDPropagation:
    """End-to-end PSD propagation through patterns Kalman filters generate.

    Each piece relies on a rule already in the assumptions system; this class
    pins them down together so the ``cholesky``-friendly chain stays intact.
    """

    def test_psd_plus_psd(self):
        x = pt.matrix("x", shape=(4, 4))
        y = pt.matrix("y", shape=(4, 4))
        x_psd = assume(x, positive_definite=True)
        y_psd = assume(y, positive_definite=True)
        s = x_psd + y_psd
        _, af = make_fgraph(s)
        assert af.check(s, POSITIVE_DEFINITE)

    def test_inv_psd(self):
        x = pt.matrix("x", shape=(4, 4))
        x_psd = assume(x, positive_definite=True)
        s = pt.linalg.inv(x_psd)
        _, af = make_fgraph(s)
        assert af.check(s, POSITIVE_DEFINITE)

    def test_kalman_predict(self):
        """``T @ P @ T.T + Q`` is PSD when ``P`` and ``Q`` are PSD."""
        T = pt.matrix("T", shape=(4, 4))
        P = pt.matrix("P", shape=(4, 4))
        Q = pt.matrix("Q", shape=(4, 4))
        P_psd = assume(P, positive_definite=True)
        Q_psd = assume(Q, positive_definite=True)
        P_next = T @ P_psd @ T.T + Q_psd
        _, af = make_fgraph(P_next)
        assert af.check(P_next, POSITIVE_DEFINITE)

    def test_joseph_form(self):
        """``(I - K @ H) @ P @ (I - K @ H).T + K @ R @ K.T`` is PSD when ``P``, ``R`` are PSD.

        Each summand is a congruence on a PSD matrix, then ``+`` of two PSDs.
        """
        I = pt.eye(4)
        K = pt.matrix("K", shape=(4, 2))
        H = pt.matrix("H", shape=(2, 4))
        P = pt.matrix("P", shape=(4, 4))
        R = pt.matrix("R", shape=(2, 2))
        P_psd = assume(P, positive_definite=True)
        R_psd = assume(R, positive_definite=True)
        IKH = I - K @ H
        P_update = IKH @ P_psd @ IKH.T + K @ R_psd @ K.T
        _, af = make_fgraph(P_update)
        assert af.check(P_update, POSITIVE_DEFINITE)


def test_discrete_lyapunov_propagates_positive_definite():
    a = pt.matrix("a", shape=(3, 3))
    q = pt.matrix("q", shape=(3, 3))
    q_psd = assume(q, positive_definite=True)
    x = pt.linalg.solve_discrete_lyapunov(a, q_psd, method="bilinear")
    _, af = make_fgraph(x)
    assert af.check(x, POSITIVE_DEFINITE)


def test_discrete_lyapunov_untagged_is_unknown_positive_definite():
    a = pt.matrix("a", shape=(3, 3))
    q = pt.matrix("q", shape=(3, 3))
    x = pt.linalg.solve_discrete_lyapunov(a, q, method="bilinear")
    _, af = make_fgraph(x)
    assert af.get(x, POSITIVE_DEFINITE) == FactState.UNKNOWN


class TestPSDElemwiseAssumptions:
    @staticmethod
    def _psd(name, shape=(5, 5)):
        x = pt.tensor(name, shape=shape)
        return assume(x, positive_definite=True), x

    def test_add_pd_plus_pd(self):
        a, _ = self._psd("a")
        b, _ = self._psd("b")
        z = a + b
        _, af = make_fgraph(z)
        assert af.check(z, POSITIVE_DEFINITE)

    @pytest.mark.parametrize(
        "scale, expected",
        [
            pytest.param(2.5, True, id="positive"),
            pytest.param(-1.0, False, id="negative"),
            pytest.param(0.0, False, id="zero"),
        ],
    )
    def test_scalar_times_pd(self, scale, expected):
        a, _ = self._psd("a")
        z = scale * a
        _, af = make_fgraph(z)
        assert af.check(z, POSITIVE_DEFINITE) is expected

    @pytest.mark.parametrize(
        "shape",
        [
            pytest.param((), id="0d_scalar"),
            pytest.param((1, 1), id="1x1_matrix"),
            pytest.param((5, 1), id="col_vector"),
            pytest.param((5, 5), id="generic_matrix"),
        ],
    )
    def test_add_pd_plus_other_is_unknown(self, shape):
        a, _ = self._psd("a")
        y = pt.tensor("y", shape=shape)
        z = a + y
        _, af = make_fgraph(z)
        assert not af.check(z, POSITIVE_DEFINITE)

    def test_batched_pd_plus_pd(self):
        a, _ = self._psd("a", shape=(4, 5, 5))
        b, _ = self._psd("b", shape=(4, 5, 5))
        z = a + b
        _, af = make_fgraph(z)
        assert af.check(z, POSITIVE_DEFINITE)

    def test_unrelated_elemwise_is_unknown(self):
        a, _ = self._psd("a")
        z = -a
        _, af = make_fgraph(z)
        assert not af.check(z, POSITIVE_DEFINITE)

    def test_mul_of_only_positive_scalars_is_not_pd(self):
        # Mul of a single positive scalar (no matrix factor) should not be claimed PD —
        # the result is a 0d scalar, not a matrix.
        z = pt.constant(2.0) * pt.constant(3.0)
        _, af = make_fgraph(z)
        assert not af.check(z, POSITIVE_DEFINITE)
