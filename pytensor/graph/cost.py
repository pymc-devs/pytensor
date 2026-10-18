import math
from collections import defaultdict, deque
from collections.abc import Sequence
from itertools import pairwise

from pytensor.graph.basic import Variable


# A "dim entry" is an int (statically known size) or a Variable (a scalar shape var
# from ShapeFeature, treated as an opaque positive symbol >= 1). A CostExpr is a sum
# of monomials in those symbols, with literal-int factors folded into each monomial's
# coefficient.

DimEntry = int | Variable


class CostExpr:
    """Polynomial in positive dim symbols.

    Stores its monomials as ``{sorted_(symbol,exp)_tuple: int_coef}``. The FLOP cost
    of a matmul is (approximately) the product of all unique dim lengths. Given
    inputs A of shape ``(a, b)`` and B of shape ``(b, c)``, computing ``A @ B``
    costs ``a * b * c``, a single monomial. The cost of a chain of matmuls is the sum
    of its monomials.
    """

    __slots__ = ("monomials",)

    def __init__(self, monomials=None):
        self.monomials: dict[tuple, int] = (
            dict(monomials) if monomials is not None else {}
        )

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
        key = tuple(
            sorted(
                sym_exps.items(),
                key=lambda kv: (getattr(kv[0], "name", None) or "", id(kv[0])),
            )
        )
        return cls({key: coef})

    def evaluate(self, values: dict[Variable, int]) -> int:
        """Value of the polynomial with each symbol replaced by ``values[symbol]``."""
        return sum(
            coef * math.prod(values[sym] ** exp for sym, exp in key)
            for key, coef in self.monomials.items()
        )

    def __add__(self, other: "CostExpr") -> "CostExpr":
        result = dict(self.monomials)
        for k, v in other.monomials.items():
            result[k] = result.get(k, 0) + v
            if result[k] == 0:
                del result[k]
        return CostExpr(result)

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


def _provably_cheaper(a: CostExpr, b: CostExpr) -> bool:
    """True iff a <= b for every assignment of the dim symbols to integers >= 1, and
    the two polynomials differ.

    A monomial of `b` whose exponents dominate a monomial of `a` is at least as large
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
