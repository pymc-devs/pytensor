import math
import operator
from collections import defaultdict, deque
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from itertools import pairwise

from pytensor.graph.basic import Variable


# A "dim entry" is an int (statically known size) or a Variable, an opaque symbol for an
# integer >= 1 unless a caller declares it zero-able (>= 0). A CostExpr is a sum of
# monomials in those symbols, with literal-int factors folded into each monomial's
# coefficient.

DimEntry = int | Variable


def _symbol_order(pair: tuple[Variable, int]) -> tuple[str, int]:
    """Sort key putting a monomial's ``(symbol, exponent)`` pairs in canonical order."""
    return (pair[0].name or "", id(pair[0]))


class CostExpr:
    """Polynomial in dimension symbols with non-negative integer coefficients.

    Built from ``{((symbol, exponent), ...): coefficient}``. The constructor merges
    repeated symbols, orders each monomial's pairs, and drops zero exponents and zero
    coefficients, so equal polynomials have equal ``monomials``. A negative coefficient
    raises ``ValueError``: the dominance prover is sound only for non-negative
    coefficients.
    """

    __slots__ = ("monomials",)

    def __init__(self, monomials: dict[tuple, int] | None = None):
        self.monomials: dict[tuple, int] = {}
        for key, coef in (monomials or {}).items():
            coef = operator.index(coef)
            if coef < 0:
                raise ValueError(
                    f"CostExpr coefficients must be non-negative, got {coef} for {key}"
                )
            if not coef:
                continue
            sym_exps: dict[Variable, int] = defaultdict(int)
            for sym, exp in key:
                sym_exps[sym] += exp
            canonical = tuple(
                sorted(
                    ((sym, exp) for sym, exp in sym_exps.items() if exp),
                    key=_symbol_order,
                )
            )
            self.monomials[canonical] = self.monomials.get(canonical, 0) + coef

    @classmethod
    def zero(cls) -> "CostExpr":
        return cls()

    @classmethod
    def from_dim_product(cls, dims: Sequence[DimEntry]) -> "CostExpr":
        """Build a single monomial from the product of `dims`."""
        coef = 1
        exps: dict[Variable, int] = {}
        for d in dims:
            if isinstance(d, Variable):
                exps[d] = exps.get(d, 0) + 1
            else:
                coef *= operator.index(d)
        if coef < 0:
            raise ValueError(f"CostExpr coefficients must be non-negative, got {coef}")
        expr = cls()
        if coef:
            expr.monomials[tuple(sorted(exps.items(), key=_symbol_order))] = coef
        return expr

    @classmethod
    def sum(cls, exprs: Iterable["CostExpr"]) -> "CostExpr":
        """The sum of `exprs`, in time linear in their total number of monomials."""
        # Every operand is canonical, so merge without re-canonicalizing each key
        exprs = iter(exprs)
        result = cls()
        first = next(exprs, None)
        if first is not None:
            result.monomials = dict(first.monomials)
        for expr in exprs:
            for key, coef in expr.monomials.items():
                result.monomials[key] = result.monomials.get(key, 0) + coef
        return result

    def evaluate(self, values: dict[Variable, int]) -> int:
        """Value of the polynomial with each symbol replaced by ``values[symbol]``."""
        return sum(
            coef * math.prod(values[sym] ** exp for sym, exp in key)
            for key, coef in self.monomials.items()
        )

    def __eq__(self, other: object) -> bool:
        return isinstance(other, CostExpr) and self.monomials == other.monomials

    def __hash__(self) -> int:
        return hash(frozenset(self.monomials.items()))

    def __add__(self, other: "CostExpr") -> "CostExpr":
        return CostExpr.sum((self, other))

    def __mul__(self, other: "CostExpr | int") -> "CostExpr":
        if not isinstance(other, CostExpr):
            scale = operator.index(other)
            if scale < 0:
                raise ValueError(
                    f"CostExpr coefficients must be non-negative, got scale {scale}"
                )
            # Scaling by a positive integer keeps every key canonical
            scaled = CostExpr()
            if scale:
                scaled.monomials = {k: v * scale for k, v in self.monomials.items()}
            return scaled
        # Both operands are canonical, so only merged keys need sorting
        product = CostExpr()
        for left_key, left_coef in self.monomials.items():
            for right_key, right_coef in other.monomials.items():
                if not left_key or not right_key:
                    key = left_key or right_key
                else:
                    exps = dict(left_key)
                    for sym, exp in right_key:
                        exps[sym] = exps.get(sym, 0) + exp
                    key = tuple(sorted(exps.items(), key=_symbol_order))
                product.monomials[key] = (
                    product.monomials.get(key, 0) + left_coef * right_coef
                )
        return product

    __rmul__ = __mul__

    def format(
        self, label: Callable[[Variable], str] = lambda sym: sym.name or "<unnamed>"
    ) -> str:
        """`self` as a sum of products, with each symbol shown as ``label(symbol)``.

        Factors and terms are sorted, so the string depends only on the labels. The
        default label is the symbol's name. ``label=pytensor.printing.pprint`` shows an
        unnamed shape symbol as the expression it stands for, such as ``Shape_i{0}(x)``.
        """
        terms = []
        for key, coef in self.monomials.items():
            factors = sorted(
                label(sym) if exp == 1 else f"{label(sym)}^{exp}" for sym, exp in key
            )
            coefficient = [str(coef)] if coef != 1 or not key else []
            terms.append("*".join(coefficient + factors))
        return " + ".join(sorted(terms)) or "0"

    def __repr__(self) -> str:
        return f"CostExpr({self.format()})"


@dataclass(frozen=True)
class Bound:
    """An interval ``[lo, hi]`` of costs, with ``hi`` None when it is unbounded."""

    lo: CostExpr
    hi: CostExpr | None

    @classmethod
    def exact(cls, cost: CostExpr) -> "Bound":
        return cls(cost, cost)

    @classmethod
    def unknown(cls) -> "Bound":
        return cls(CostExpr.zero(), None)

    @classmethod
    def sum(cls, bounds: Iterable["Bound"]) -> "Bound":
        bounds = list(bounds)
        lo = CostExpr.sum(bound.lo for bound in bounds)
        if all(bound.lo is bound.hi for bound in bounds):
            return cls.exact(lo)
        his = [bound.hi for bound in bounds if bound.hi is not None]
        hi = CostExpr.sum(his) if len(his) == len(bounds) else None
        return cls(lo, hi)

    def __add__(self, other: "Bound") -> "Bound":
        return Bound.sum((self, other))

    def __mul__(self, factor: CostExpr | int) -> "Bound":
        if factor == 0 or (isinstance(factor, CostExpr) and not factor.monomials):
            return Bound.exact(CostExpr.zero())
        lo = self.lo * factor
        if self.hi is self.lo:
            return Bound.exact(lo)
        return Bound(lo, None if self.hi is None else self.hi * factor)

    __rmul__ = __mul__


@dataclass(frozen=True)
class Cost:
    """The cost of a computation as independent FLOP and byte bounds.

    The two components are never combined into one number, so comparisons between
    costs hold on every backend.
    """

    flops: Bound
    bytes: Bound

    @classmethod
    def zero(cls) -> "Cost":
        return cls(Bound.exact(CostExpr.zero()), Bound.exact(CostExpr.zero()))

    @classmethod
    def unknown(cls) -> "Cost":
        return cls(Bound.unknown(), Bound.unknown())

    @classmethod
    def sum(cls, costs: Iterable["Cost"]) -> "Cost":
        costs = list(costs)
        return cls(
            Bound.sum(cost.flops for cost in costs),
            Bound.sum(cost.bytes for cost in costs),
        )

    def __add__(self, other: "Cost") -> "Cost":
        return Cost.sum((self, other))

    def __mul__(self, factor: CostExpr | int) -> "Cost":
        return Cost(self.flops * factor, self.bytes * factor)

    __rmul__ = __mul__


def _polynomial_le(
    a: CostExpr, b: CostExpr, zero_able: frozenset[Variable] = frozenset()
) -> bool:
    """True iff a <= b for every assignment of the dim symbols.

    Symbols range over integers >= 1, except those in `zero_able`, which range over
    integers >= 0. A monomial of `b` dominates a monomial of `a` when its exponents are
    at least `a`'s and every zero-able symbol in it also appears in `a`'s, so that
    zeroing that symbol zeroes both. A dominating monomial is at least as large
    everywhere, so ``a <= b`` holds when every unit of coefficient in `a` can be paid
    for by a unit of coefficient on a dominating monomial of `b`, each unit of `b` used
    once. That assignment is a transportation problem, solved as a max flow from the
    `a` monomials to the `b` monomials. Returns False for "not provable"; never claims
    ``b <= a``.
    """
    if a.monomials == b.monomials:
        return True
    if not a.monomials:
        return True

    a_terms = list(a.monomials.items())
    b_terms = list(b.monomials.items())
    covers = []
    for a_key, _ in a_terms:
        a_exp = dict(a_key)
        row = [
            j
            for j, (b_key, _) in enumerate(b_terms)
            if all(dict(b_key).get(sym, 0) >= exp for sym, exp in a_exp.items())
            and all(sym in a_exp for sym, _ in b_key if sym in zero_able)
        ]
        if not row:
            return False
        covers.append(row)

    # Edmonds-Karp over source -> a_i -> b_j -> sink. Edges a_i -> b_j are uncapped,
    # so only flow already sent along them can be pushed back.
    remaining_a = [coef for _, coef in a_terms]
    remaining_b = [coef for _, coef in b_terms]
    sent: dict[tuple[int, int], int] = defaultdict(int)
    received_by_b: list[list[int]] = [[] for _ in b_terms]

    def augmenting_path() -> list[tuple[str, int]] | None:
        parent: dict[tuple[str, int], tuple[str, int] | None] = {
            ("a", i): None for i, left in enumerate(remaining_a) if left
        }
        queue = deque(parent)
        while queue:
            side, idx = queue.popleft()
            if side == "a":
                for j in covers[idx]:
                    if ("b", j) not in parent:
                        parent[("b", j)] = (side, idx)
                        if remaining_b[j]:
                            path = [("b", j)]
                            while (step := parent[path[-1]]) is not None:
                                path.append(step)
                            return path[::-1]
                        queue.append(("b", j))
            else:
                for i in received_by_b[idx]:
                    if sent[(i, idx)] and ("a", i) not in parent:
                        parent[("a", i)] = (side, idx)
                        queue.append(("a", i))
        return None

    while (path := augmenting_path()) is not None:
        start, end = path[0][1], path[-1][1]
        amount = min(remaining_a[start], remaining_b[end])
        for (from_side, from_idx), (_, to_idx) in pairwise(path):
            if from_side == "b":
                amount = min(amount, sent[(to_idx, from_idx)])
        for (from_side, from_idx), (_, to_idx) in pairwise(path):
            if from_side == "a":
                if not sent[(from_idx, to_idx)]:
                    received_by_b[to_idx].append(from_idx)
                sent[(from_idx, to_idx)] += amount
            else:
                sent[(to_idx, from_idx)] -= amount
        remaining_a[start] -= amount
        remaining_b[end] -= amount

    return not any(remaining_a)


def is_zero(cost: Cost) -> bool:
    """Whether `cost` is exactly zero in every component, whatever the symbols."""
    return all(bound.hi == CostExpr.zero() for bound in (cost.flops, cost.bytes))


def _bound_le(smaller: Bound, larger: Bound, zero_able: frozenset[Variable]) -> bool:
    return smaller.hi is not None and _polynomial_le(smaller.hi, larger.lo, zero_able)


def provably_le(a: Cost, b: Cost, zero_able: frozenset[Variable] = frozenset()) -> bool:
    """Whether `a` is no more than `b` in every component, for every symbol assignment.

    Compares the upper bound of each component of `a` with the lower bound of the same
    component of `b`. Symbols are integers >= 1, except those in `zero_able`, which are
    integers >= 0. False means "not provable", never that `b` is smaller.
    """
    return _bound_le(a.flops, b.flops, zero_able) and _bound_le(
        a.bytes, b.bytes, zero_able
    )


def provably_cheaper(
    a: Cost, b: Cost, zero_able: frozenset[Variable] = frozenset()
) -> bool:
    """Whether `a` is provably no more than `b` in every component, and they differ."""
    return a != b and provably_le(a, b, zero_able)
