import pytest

import pytensor.tensor as pt
from pytensor.assumptions import POSITIVE_DEFINITE, SYMMETRIC, FactState
from pytensor.assumptions.specify import assume
from tests.assumptions.conftest import make_fgraph


class TestCongruenceMatmul:
    """Tests for ``M @ S @ M.T`` -> symmetric/PSD when ``S`` is symmetric/PSD."""

    @pytest.mark.parametrize("key", [SYMMETRIC, POSITIVE_DEFINITE])
    def test_left_associative(self, key):
        m = pt.matrix("m", shape=(4, 3))
        s = pt.matrix("s", shape=(3, 3))
        s_tagged = assume(s, **{key.name: True})
        y = (m @ s_tagged) @ m.T
        _, af = make_fgraph(y)
        assert af.check(y, key)

    @pytest.mark.parametrize("key", [SYMMETRIC, POSITIVE_DEFINITE])
    def test_right_associative(self, key):
        m = pt.matrix("m", shape=(4, 3))
        s = pt.matrix("s", shape=(3, 3))
        s_tagged = assume(s, **{key.name: True})
        y = m @ (s_tagged @ m.T)
        _, af = make_fgraph(y)
        assert af.check(y, key)

    @pytest.mark.parametrize("key", [SYMMETRIC, POSITIVE_DEFINITE])
    def test_mirror_mtsm(self, key):
        """``M.T @ S @ M`` is also a congruence."""
        m = pt.matrix("m", shape=(3, 4))
        s = pt.matrix("s", shape=(3, 3))
        s_tagged = assume(s, **{key.name: True})
        y = m.T @ s_tagged @ m
        _, af = make_fgraph(y)
        assert af.check(y, key)

    @pytest.mark.parametrize("key", [SYMMETRIC, POSITIVE_DEFINITE])
    def test_inner_not_tagged_is_unknown(self, key):
        m = pt.matrix("m", shape=(4, 3))
        s = pt.matrix("s", shape=(3, 3))
        y = m @ s @ m.T
        _, af = make_fgraph(y)
        assert af.get(y, key) == FactState.UNKNOWN

    def test_psd_implies_symmetric_via_congruence(self):
        m = pt.matrix("m", shape=(4, 3))
        s = pt.matrix("s", shape=(3, 3))
        s_psd = assume(s, positive_definite=True)
        y = m @ s_psd @ m.T
        _, af = make_fgraph(y)
        assert af.check(y, POSITIVE_DEFINITE)
        assert af.check(y, SYMMETRIC)
