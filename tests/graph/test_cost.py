import pytest

import pytensor.tensor as pt
from pytensor.graph.cost import (
    Bound,
    Cost,
    CostExpr,
    _provably_cheaper,
    is_zero,
    provably_cheaper,
    provably_le,
)


class TestCostExpr:
    def test_zero_coefficients_dropped(self):
        n = pt.scalar("n")
        assert CostExpr.from_dim_product([0, n]).monomials == {}

    def test_negative_coefficient_rejected(self):
        n = pt.scalar("n")
        with pytest.raises(ValueError, match="non-negative"):
            CostExpr({((n, 1),): -1})

    def test_equal_polynomials_compare_equal(self):
        m, n = pt.scalar("m"), pt.scalar("n")
        left = CostExpr.from_dim_product([m, n]) + CostExpr.from_dim_product([2, m])
        right = CostExpr.from_dim_product([2, m]) + CostExpr.from_dim_product([n, m])
        assert left == right
        assert hash(left) == hash(right)
        assert left != CostExpr.from_dim_product([m, n])

    def test_multiplication(self):
        m, n, p = (pt.scalar(name) for name in "mnp")
        mn = CostExpr.from_dim_product([m, n])
        assert 3 * mn == CostExpr.from_dim_product([3, m, n])
        assert mn * 0 == CostExpr.zero()
        assert mn * CostExpr.from_dim_product([2, n]) == CostExpr.from_dim_product(
            [2, m, n, n]
        )
        total = CostExpr.from_dim_product([m]) + CostExpr.from_dim_product([n])
        assert CostExpr.from_dim_product([p]) * total == CostExpr.from_dim_product(
            [p, m]
        ) + CostExpr.from_dim_product([p, n])

    def test_negative_scale_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            CostExpr.from_dim_product([pt.scalar("n")]) * -1


class TestProvablyCheaper:
    def test_int_dominance(self):
        a = CostExpr.from_dim_product([10, 20, 5])  # 1000
        b = CostExpr.from_dim_product([10, 20, 5]) + CostExpr.from_dim_product(
            [1, 2, 3]
        )  # 1006
        assert _provably_cheaper(a, b)
        assert not _provably_cheaper(b, a)
        assert not _provably_cheaper(a, a)

    def test_zero_vs_positive(self):
        a = CostExpr.zero()
        b = CostExpr.from_dim_product([7])
        assert _provably_cheaper(a, b)
        assert not _provably_cheaper(b, a)
        assert not _provably_cheaper(a, a)

    def test_symbolic_no_proof(self):
        # m*k*n vs k*n*p: different monomials, no dominance either way.
        m, k, n, p = (pt.scalar(name) for name in "mknp")
        a = CostExpr.from_dim_product([m, k, n])
        b = CostExpr.from_dim_product([k, n, p])
        assert not _provably_cheaper(a, b)
        assert not _provably_cheaper(b, a)

    def test_extra_unmatched_b_monomial(self):
        # a = m*k, b = m*k + p: b strictly larger under p >= 1.
        m, k, p = (pt.scalar(name) for name in "mkp")
        a = CostExpr.from_dim_product([m, k])
        b = CostExpr.from_dim_product([m, k]) + CostExpr.from_dim_product([p])
        assert _provably_cheaper(a, b)
        assert not _provably_cheaper(b, a)

    def test_tie_at_ones_is_cheaper(self):
        # a = m*k, b = m*k*p: equal at p == 1, so a is never worse and better for p > 1
        m, k, p = (pt.scalar(name) for name in "mkp")
        a = CostExpr.from_dim_product([m, k])
        b = CostExpr.from_dim_product([m, k, p])
        assert _provably_cheaper(a, b)
        assert not _provably_cheaper(b, a)

    def test_zero_able_symbol_breaks_dominance(self):
        # m*k <= m*k*p fails at p = 0
        m, k, p = (pt.scalar(name) for name in "mkp")
        a = CostExpr.from_dim_product([m, k])
        b = CostExpr.from_dim_product([m, k, p])
        assert not _provably_cheaper(a, b, zero_able=frozenset({p}))

    def test_zero_able_symbol_on_both_sides(self):
        # m*p <= m*k*p: both sides vanish at p = 0
        m, k, p = (pt.scalar(name) for name in "mkp")
        a = CostExpr.from_dim_product([m, p])
        b = CostExpr.from_dim_product([m, k, p])
        assert _provably_cheaper(a, b, zero_able=frozenset({p}))

    def test_split_coefficient(self):
        # 2*n^2 <= n^3 + n^2 needs one unit of a's coefficient on each b monomial
        n = pt.scalar("n")
        a = CostExpr.from_dim_product([2, n, n])
        b = CostExpr.from_dim_product([n, n, n]) + CostExpr.from_dim_product([n, n])
        assert _provably_cheaper(a, b)
        assert not _provably_cheaper(b, a)

    @pytest.mark.parametrize(
        "build",
        [
            # 2*n*k > k*n^2 + n^2 at n = 1, k = 10
            lambda n, k: (
                CostExpr.from_dim_product([2, n, k]),
                CostExpr.from_dim_product([k, n, n])
                + CostExpr.from_dim_product([n, n]),
            ),
            # 3*n^2 > n^3 + n^2 at n = 1
            lambda n, k: (
                CostExpr.from_dim_product([3, n, n]),
                CostExpr.from_dim_product([n, n, n])
                + CostExpr.from_dim_product([n, n]),
            ),
        ],
        ids=["low_rank", "excess_coefficient"],
    )
    def test_false_for_some_sizes(self, build):
        a, b = build(pt.scalar("n"), pt.scalar("k"))
        assert not _provably_cheaper(a, b)

    def test_assignment_needs_rerouting(self):
        # Assigning a's x to b's first dominator (x*y) leaves a's x*y uncovered. The
        # valid assignment is x <-> x*z and x*y <-> x*y.
        x, y, z, w = (pt.scalar(name) for name in "xyzw")
        a = CostExpr.from_dim_product([x]) + CostExpr.from_dim_product([x, y])
        b = (
            CostExpr.from_dim_product([x, y])
            + CostExpr.from_dim_product([x, z])
            + CostExpr.from_dim_product([w])
        )
        assert _provably_cheaper(a, b)


class TestBound:
    def test_sum_of_exact_bounds(self):
        m, n = pt.scalar("m"), pt.scalar("n")
        total = Bound.exact(CostExpr.from_dim_product([m])) + Bound.exact(
            CostExpr.from_dim_product([n])
        )
        expected = CostExpr.from_dim_product([m]) + CostExpr.from_dim_product([n])
        assert total == Bound.exact(expected)

    def test_unknown_keeps_lower_bound(self):
        m = pt.scalar("m")
        total = Bound.exact(CostExpr.from_dim_product([m])) + Bound.unknown()
        assert total == Bound(CostExpr.from_dim_product([m]), None)

    def test_scaling(self):
        m, n = pt.scalar("m"), pt.scalar("n")
        steps = CostExpr.from_dim_product([n])
        body = Bound(CostExpr.zero(), CostExpr.from_dim_product([m]))
        assert body * steps == Bound(CostExpr.zero(), CostExpr.from_dim_product([m, n]))
        assert Bound.unknown() * steps == Bound.unknown()
        assert Bound.unknown() * 0 == Bound.exact(CostExpr.zero())


class TestCost:
    def test_componentwise_sum_and_scaling(self):
        m, n = pt.scalar("m"), pt.scalar("n")
        node = Cost(
            Bound.exact(CostExpr.from_dim_product([2, m])),
            Bound.exact(CostExpr.from_dim_product([8, m])),
        )
        steps = CostExpr.from_dim_product([n])
        assert (node + node) * steps == Cost(
            Bound.exact(CostExpr.from_dim_product([4, m, n])),
            Bound.exact(CostExpr.from_dim_product([16, m, n])),
        )

    def test_zero_and_unknown(self):
        node = Cost(
            Bound.exact(CostExpr.from_dim_product([3])),
            Bound.exact(CostExpr.from_dim_product([5])),
        )
        assert node + Cost.zero() == node
        assert (node + Cost.unknown()).flops == Bound(
            CostExpr.from_dim_product([3]), None
        )


def _exact(flops, bytes_):
    return Cost(Bound.exact(flops), Bound.exact(bytes_))


class TestQueries:
    def test_is_zero(self):
        m = pt.scalar("m")
        assert is_zero(Cost.zero())
        assert not is_zero(Cost.unknown())
        assert not is_zero(_exact(CostExpr.zero(), CostExpr.from_dim_product([m])))

    def test_le_and_cheaper(self):
        m, k = pt.scalar("m"), pt.scalar("k")
        small = _exact(CostExpr.from_dim_product([m]), CostExpr.from_dim_product([m]))
        large = _exact(
            CostExpr.from_dim_product([m, k]), CostExpr.from_dim_product([m, k])
        )
        assert provably_le(small, large)
        assert provably_cheaper(small, large)
        assert provably_le(small, small)
        assert not provably_cheaper(small, small)
        assert not provably_le(large, small)

    def test_pareto_order(self):
        # Cheaper in FLOPs but dearer in bytes: neither side is provably cheaper.
        m, k = pt.scalar("m"), pt.scalar("k")
        a = _exact(CostExpr.from_dim_product([m]), CostExpr.from_dim_product([m, k]))
        b = _exact(CostExpr.from_dim_product([m, k]), CostExpr.from_dim_product([m]))
        assert not provably_le(a, b)
        assert not provably_le(b, a)

    def test_unknown_upper_end_never_provable(self):
        m = pt.scalar("m")
        known = _exact(CostExpr.from_dim_product([m]), CostExpr.from_dim_product([m]))
        assert not provably_le(known + Cost.unknown(), known)
        # An unknown on the larger side only lowers its lower end.
        assert provably_le(known, known + Cost.unknown())

    def test_zero_able_passes_through(self):
        m, k, p = (pt.scalar(name) for name in "mkp")
        a = _exact(CostExpr.from_dim_product([m, k]), CostExpr.zero())
        b = _exact(CostExpr.from_dim_product([m, k, p]), CostExpr.zero())
        assert provably_le(a, b)
        assert not provably_le(a, b, zero_able=frozenset({p}))
