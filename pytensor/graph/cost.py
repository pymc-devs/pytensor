import math
from collections import defaultdict, deque
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

from pytensor.graph.basic import Variable


# A "dim entry" is an int (statically known size) or a Variable, an opaque symbol for an
# integer >= 1 unless a caller declares it zero-able (>= 0). A CostExpr is a sum of
# monomials in those symbols, with literal-int factors folded into each monomial's
# coefficient.

DimEntry = int | Variable


def _monomial_key(sym_exps: dict[Variable, int]) -> tuple:
    """Canonical key of a monomial: its ``(symbol, exponent)`` pairs, ordered."""
    return tuple(
        sorted(
            sym_exps.items(),
            key=lambda kv: (getattr(kv[0], "name", None) or "", id(kv[0])),
        )
    )


class CostExpr:
    """Polynomial in dimension symbols with non-negative integer coefficients.

    Stores its monomials as ``{sorted_(symbol,exp)_tuple: int_coef}``. Monomials with a
    zero coefficient are dropped, so equal polynomials have equal ``monomials``. A
    negative coefficient raises ``ValueError``: the dominance prover is sound only for
    non-negative coefficients.
    """

    __slots__ = ("monomials",)

    def __init__(self, monomials=None):
        self.monomials: dict[tuple, int] = {}
        for key, coef in (monomials or {}).items():
            if coef < 0:
                raise ValueError(
                    f"CostExpr coefficients must be non-negative, got {coef} for {key}"
                )
            if coef:
                self.monomials[key] = coef

    @classmethod
    def zero(cls) -> "CostExpr":
        return cls()

    @classmethod
    def from_dim_product(cls, dims: Sequence[DimEntry]) -> "CostExpr":
        """Build a single monomial from the product of `dims`.

        Symbols are sorted so that equal products map to the same monomial key.
        """
        coef = 1
        sym_exps: dict[Variable, int] = defaultdict(int)
        for d in dims:
            if isinstance(d, int):
                coef *= d
            else:
                sym_exps[d] += 1
        return cls({_monomial_key(sym_exps): coef})

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
        result = dict(self.monomials)
        for k, v in other.monomials.items():
            result[k] = result.get(k, 0) + v
        return CostExpr(result)

    def __mul__(self, other: "CostExpr | int") -> "CostExpr":
        if isinstance(other, int):
            return CostExpr({k: v * other for k, v in self.monomials.items()})
        result: dict[tuple, int] = defaultdict(int)
        for left_key, left_coef in self.monomials.items():
            for right_key, right_coef in other.monomials.items():
                sym_exps: dict[Variable, int] = defaultdict(int)
                for sym, exp in (*left_key, *right_key):
                    sym_exps[sym] += exp
                result[_monomial_key(sym_exps)] += left_coef * right_coef
        return CostExpr(result)

    __rmul__ = __mul__

    def __repr__(self) -> str:
        if not self.monomials:
            return "CostExpr(0)"
        terms = []
        for sym_exps, coef in self.monomials.items():
            parts = [str(coef)] if coef != 1 or not sym_exps else []
            for sym, exp in sym_exps:
                name = getattr(sym, "name", None) or "<unnamed>"
                parts.append(f"{name}^{exp}" if exp != 1 else name)
            terms.append("*".join(parts) if parts else "1")
        return "CostExpr(" + " + ".join(terms) + ")"


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

    def __add__(self, other: "Bound") -> "Bound":
        hi = None if self.hi is None or other.hi is None else self.hi + other.hi
        return Bound(self.lo + other.lo, hi)

    def __mul__(self, factor: CostExpr | int) -> "Bound":
        if factor == 0 or factor == CostExpr.zero():
            return Bound.exact(CostExpr.zero())
        return Bound(self.lo * factor, None if self.hi is None else self.hi * factor)

    __rmul__ = __mul__


def _provably_cheaper(
    a: CostExpr, b: CostExpr, zero_able: frozenset[Variable] = frozenset()
) -> bool:
    """True iff a <= b for every assignment of the dim symbols, and the two differ.

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
        return False
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
