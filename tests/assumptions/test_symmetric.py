import pytest

import pytensor.tensor as pt
from pytensor.assumptions import SYMMETRIC, FactState
from pytensor.assumptions.specify import assume
from tests.assumptions.conftest import make_fgraph


def test_inv_propagates_symmetric():
    x = pt.matrix("x", shape=(3, 3))
    x_sym = assume(x, symmetric=True)
    inv_x = pt.linalg.inv(x_sym)
    _, af = make_fgraph(inv_x)
    assert af.check(inv_x, SYMMETRIC)


def test_pinv_propagates_symmetric():
    x = pt.matrix("x", shape=(3, 3))
    x_sym = assume(x, symmetric=True)
    px = pt.linalg.pinv(x_sym)
    _, af = make_fgraph(px)
    assert af.check(px, SYMMETRIC)


def test_kron_of_eyes_is_symmetric():
    k = pt.linalg.kron(pt.eye(3), pt.eye(4))
    _, af = make_fgraph(k)
    assert af.check(k, SYMMETRIC)


def test_discrete_lyapunov_propagates_symmetric():
    a = pt.matrix("a", shape=(3, 3))
    q = pt.matrix("q", shape=(3, 3))
    q_sym = assume(q, symmetric=True)
    x = pt.linalg.solve_discrete_lyapunov(a, q_sym, method="bilinear")
    _, af = make_fgraph(x)
    assert af.check(x, SYMMETRIC)


def test_discrete_lyapunov_untagged_is_unknown_symmetric():
    a = pt.matrix("a", shape=(3, 3))
    q = pt.matrix("q", shape=(3, 3))
    x = pt.linalg.solve_discrete_lyapunov(a, q, method="bilinear")
    _, af = make_fgraph(x)
    assert af.get(x, SYMMETRIC) == FactState.UNKNOWN


class TestSymmetricElemwiseAssumptions:
    @staticmethod
    def _sym(name, shape=(5, 5)):
        x = pt.tensor(name, shape=shape)
        return assume(x, symmetric=True), x

    @pytest.mark.parametrize(
        "binop",
        [
            pytest.param(lambda a, b: a + b, id="add"),
            pytest.param(lambda a, b: a - b, id="sub"),
            pytest.param(lambda a, b: a * b, id="mul"),
            pytest.param(lambda a, b: a / b, id="truediv"),
            pytest.param(lambda a, b: a**b, id="pow"),
        ],
    )
    def test_binop_of_symmetric_is_symmetric(self, binop):
        a, _ = self._sym("a")
        b, _ = self._sym("b")
        z = binop(a, b)
        _, af = make_fgraph(z)
        assert af.check(z, SYMMETRIC)

    @pytest.mark.parametrize(
        "unop",
        [
            pytest.param(lambda a: -a, id="neg"),
            pytest.param(lambda a: pt.exp(a), id="exp"),
            pytest.param(abs, id="abs"),
        ],
    )
    def test_unop_of_symmetric_is_symmetric(self, unop):
        a, _ = self._sym("a")
        z = unop(a)
        _, af = make_fgraph(z)
        assert af.check(z, SYMMETRIC)

    @pytest.mark.parametrize(
        "shape, expected",
        [
            pytest.param((), True, id="0d_scalar"),
            pytest.param((1, 1), True, id="1x1_matrix"),
            pytest.param((5,), False, id="1d_vector"),
            pytest.param((5, 1), False, id="col_vector"),
            pytest.param((1, 5), False, id="row_vector"),
            pytest.param((5, 5), False, id="generic_matrix"),
        ],
    )
    def test_add_symmetric_plus_other(self, shape, expected):
        a, _ = self._sym("a")
        y = pt.tensor("y", shape=shape)
        z = a + y
        _, af = make_fgraph(z)
        assert af.check(z, SYMMETRIC) is expected

    def test_batched_symmetric_add(self):
        a, _ = self._sym("a", shape=(4, 5, 5))
        b, _ = self._sym("b", shape=(4, 5, 5))
        z = a + b
        _, af = make_fgraph(z)
        assert af.check(z, SYMMETRIC)

    def test_scalar_only_elemwise_is_unknown(self):
        s = pt.scalar("s")
        z = s + s
        _, af = make_fgraph(z)
        assert not af.check(z, SYMMETRIC)
