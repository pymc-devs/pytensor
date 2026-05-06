from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import cast

from pytensor.compile.mode import optdb
from pytensor.graph.basic import Apply, Variable
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.rewriting.basic import GraphRewriter, copy_stack_trace
from pytensor.tensor.blas import BatchedDot, Dot22
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.elemwise import DimShuffle
from pytensor.tensor.math import Dot, matmul


# A "dim entry" is an int (statically known size) or a Variable (a scalar shape var
# from ShapeFeature, treated as an opaque positive symbol >= 1). A CostExpr is a sum
# of monomials in those symbols, with literal-int factors folded into each monomial's
# coefficient.

DimEntry = int | Variable
Shape = tuple[DimEntry, ...]


class CostExpr:
    """Polynomial in positive dim symbols.

    Stores its monomials as ``{sorted_(symbol,exp)_tuple: int_coef}``. The FLOP cost
    of a matmul is (approximately) the product of all unique dim lengths. Given
    inputs A of shape ``(a, b)`` and B of shape ``(b, c)``, computing ``A @ B``
    costs ``a * b * c``. We call such a product a "monomial". A chain of matmuls
    sums these monomials to give the total cost.
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

        Symbol ordering within the monomial sorts on ``name`` first so the same dim
        across runs sorts deterministically, falling back to ``id`` only as a tiebreak
        between unnamed symbols within one process.
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


def _provably_less(a: CostExpr, b: CostExpr) -> bool:
    """True iff a < b is provable assuming every dim symbol is >= 1.

    Match each a-monomial to a distinct b-monomial that dominates it (b's coef >= a's,
    b's exponents >= a's). A complete matching is sound (a <= b); strict (a < b) iff
    sum(b.coefs) > sum(a.coefs). Matching uses Kuhn's algorithm. Returns False for
    "not provable"; never claims b <= a.
    """
    if not a.monomials:
        return any(c > 0 for c in b.monomials.values())
    if not b.monomials:
        return False

    a_terms = list(a.monomials.items())
    b_terms = list(b.monomials.items())
    n_a = len(a_terms)
    n_b = len(b_terms)
    if n_b < n_a:
        return False

    # For each a-index, list the b-indices that can cover it.
    candidates: list[list[int]] = []
    for a_key, a_coef in a_terms:
        a_exp = dict(a_key)
        row = []
        for b_idx, (b_key, b_coef) in enumerate(b_terms):
            if b_coef < a_coef:
                continue
            b_exp = dict(b_key)
            if all(b_exp.get(s, 0) >= e for s, e in a_exp.items()):
                row.append(b_idx)
        if not row:
            return False
        candidates.append(row)

    # Kuhn's algorithm: for each a-index, try to find an augmenting path.
    matched_b_to_a = [-1] * n_b

    def try_assign(a_idx: int, seen: list[bool]) -> bool:
        for b_idx in candidates[a_idx]:
            if seen[b_idx]:
                continue
            seen[b_idx] = True
            if matched_b_to_a[b_idx] == -1 or try_assign(matched_b_to_a[b_idx], seen):
                matched_b_to_a[b_idx] = a_idx
                return True
        return False

    for a_idx in range(n_a):
        if not try_assign(a_idx, [False] * n_b):
            return False

    return sum(b.monomials.values()) > sum(a.monomials.values())


def _operand_shape_raw(var: Variable, fgraph: FunctionGraph) -> Shape:
    """Raw shape of `var` as a tuple of dim entries.

    Each entry is an int when statically known and a scalar Variable from
    ShapeFeature otherwise. Falls back to ``var.shape[i]`` when ShapeFeature is absent.
    """
    static = var.type.shape
    shape_feature = getattr(fgraph, "shape_feature", None)
    if shape_feature is not None and var in shape_feature.shape_of:
        symbolic = shape_feature.shape_of[var]
    else:
        symbolic = tuple(var.shape[i] for i in range(var.type.ndim))  # type: ignore[attr-defined]
    return tuple(int(s) if s is not None else symbolic[i] for i, s in enumerate(static))


def _matmul_result_shape(left: Shape, right: Shape) -> Shape:
    """Shape of ``left @ right``: right-align the leading batch tuples, broadcast
    them (preferring the non-literal-1 side), then append ``(m, n)`` from
    ``left[-2]`` and ``right[-1]``."""
    left_batch, right_batch = left[:-2], right[:-2]
    n = max(len(left_batch), len(right_batch))
    pad_l = (1,) * (n - len(left_batch)) + tuple(left_batch)
    pad_r = (1,) * (n - len(right_batch)) + tuple(right_batch)
    batch = tuple(
        b if (isinstance(a, int) and a == 1) else a for a, b in zip(pad_l, pad_r)
    )
    return (*batch, left[-2], right[-1])


def _contract_cost(left: Shape, right: Shape) -> CostExpr:
    """FLOPs of ``left @ right``: ``prod(broadcast_batch) * m * k * n``."""
    result = _matmul_result_shape(left, right)
    return CostExpr.from_dim_product([*result[:-2], left[-2], left[-1], right[-1]])


def _classify_dimshuffle_lift(op: DimShuffle, input_ndim: int) -> tuple[bool, bool]:
    """Classify a DimShuffle for matmul-lift purposes.

    A DimShuffle commutes with matmul iff it touches only batch dimensions (the
    leading ``input_ndim - 2``), or it performs a matrix-transpose (swap of the last
    two dims) -- possibly combined with batch-only operations. Returns
    ``(is_liftable, swaps_order)``, where ``swaps_order`` is True iff the lift swaps
    operand order in the matmul: ``(L @ R).T = R.T @ L.T``.

    DimShuffle disallows duplicate input indices in `new_order`, so once the core dim
    indices appear in the last two positions of the output they cannot appear
    elsewhere -- earlier positions are batch-only.
    """
    if input_ndim < 2:
        return False, False
    new_order = op.new_order
    if len(new_order) < 2:
        return False, False
    last_two = (new_order[-2], new_order[-1])
    if last_two == (input_ndim - 2, input_ndim - 1):
        return True, False
    if last_two == (input_ndim - 1, input_ndim - 2):
        return True, True
    return False, False


def _is_chain_link(node: Apply) -> bool:
    """True iff `node` is a 2-operand matmul we can rebuild via ``matmul()``."""
    op = node.op
    if isinstance(op, Dot | Dot22 | BatchedDot):
        return True
    if isinstance(op, Blockwise) and isinstance(op.core_op, Dot):
        return True
    return False


def _find_chain_top(start: Apply, fgraph: FunctionGraph) -> Apply:
    """Walk up to the topmost chain-link consumer along single-client edges.

    Follows single-client edges where the current output feeds the consumer's *left*
    input, and walks through a single-client liftable DimShuffle when its output then
    feeds a chain-link's left input.
    """
    current = start
    while True:
        clients = fgraph.clients[current.outputs[0]]
        if len(clients) != 1:
            break
        client_node, client_idx = clients[0]
        # `client_node` may be the literal "output" sentinel for fgraph outputs;
        # the type stub is too narrow to express this.
        if not isinstance(client_node, Apply):
            break  # type: ignore[unreachable]

        if _is_chain_link(client_node) and client_idx == 0:
            current = client_node
            continue

        if (
            isinstance(client_node.op, DimShuffle)
            and client_idx == 0
            and len(fgraph.clients[client_node.outputs[0]]) == 1
        ):
            ds_consumer, ds_consumer_idx = fgraph.clients[client_node.outputs[0]][0]
            if (
                isinstance(ds_consumer, Apply)
                and _is_chain_link(ds_consumer)
                and ds_consumer_idx == 0
            ):
                liftable, _ = _classify_dimshuffle_lift(
                    client_node.op, current.outputs[0].type.ndim
                )
                if liftable:
                    current = ds_consumer
                    continue

        break
    return current


def _decompose_operand(
    root: Variable,
    fgraph: FunctionGraph,
    visited: set[Apply],
    consumed: list[Apply],
) -> list[tuple[Variable, tuple[DimShuffle, ...]]]:
    """Iteratively decompose `root` into chain leaves.

    Each leaf is ``(base_var, lifts)`` where `lifts` is a tuple of DimShuffle Ops to
    apply at materialization time (outermost first). Replace recursion with an
    explicit work stack to keep the Python stack bounded for deep matmul chains
    (autodiff-generated graphs can produce hundreds of links).

    Two descent paths grow the chain:

    - Single-client chain-link matmul: descend into both inputs. When the parent
      carries inherited lifts, both children must share the parent's output ndim --
      otherwise an inherited lift would reference indices missing on a narrower
      operand (``Blockwise(Dot)`` broadcasts heterogeneous-ndim operands).
    - Single-client liftable DimShuffle wrapping a single-client chain-link matmul:
      descend into the inner matmul's two inputs with the DimShuffle prepended to the
      inherited-lift list. For matrix-transpose lifts, swapping operand order:
      (``(L @ R).T = R.T @ L.T``). Both inner-matmul operands must have ndim equal
      to the DimShuffle's ``input_ndim`` for the same reason.

    Append each chain-link Apply we descend into to `consumed` and add to `visited`.
    """
    out: list[tuple[Variable, tuple[DimShuffle, ...]]] = []
    # Push children right-then-left so the leftmost leaf comes out first (preserving
    # the natural left-to-right operand order).
    stack: list[tuple[Variable, tuple[DimShuffle, ...]]] = [(root, ())]

    while stack:
        var, lifts = stack.pop()
        owner = var.owner
        if owner is None:
            out.append((var, lifts))
            continue

        if isinstance(owner.op, DimShuffle) and len(fgraph.clients[var]) == 1:
            ds_input = owner.inputs[0]
            inner = ds_input.owner
            if (
                inner is not None
                and _is_chain_link(inner)
                and inner not in visited
                and len(fgraph.clients[ds_input]) == 1
            ):
                ds_op = owner.op
                liftable, swaps = _classify_dimshuffle_lift(ds_op, ds_input.type.ndim)
                if liftable and all(
                    inp.type.ndim == ds_op.input_ndim for inp in inner.inputs
                ):
                    visited.add(inner)
                    consumed.append(inner)
                    new_lifts = (ds_op, *lifts)
                    left, right = inner.inputs
                    if swaps:
                        left, right = right, left
                    stack.append((right, new_lifts))
                    stack.append((left, new_lifts))
                    continue

        if (
            _is_chain_link(owner)
            and owner not in visited
            and len(fgraph.clients[var]) == 1
            and (
                not lifts or all(inp.type.ndim == var.type.ndim for inp in owner.inputs)
            )
        ):
            visited.add(owner)
            consumed.append(owner)
            stack.append((owner.inputs[1], lifts))
            stack.append((owner.inputs[0], lifts))
            continue

        out.append((var, lifts))

    return out


def _build_unification(
    chain_shapes: list[Shape], extra_shapes: Sequence[Shape] = ()
) -> tuple[list[Shape], Callable[[Shape], Shape]]:
    """Canonicalize dim entries across chain operand shapes via union-find.

    Returns ``(unified_chain_shapes, canonicalize)`` where ``canonicalize(shape)`` maps
    any Shape (over the same dim entries) to its canonical representatives. Two
    equality sources drive the union-find:

    - Adjacent contracting dims of the chain.
    - For each right-aligned batch position, all non-literal-1 entries across *every*
      operand must agree at runtime (broadcasting requires it); unioning them as one
      class catches transitive equalities a 1 in the middle would otherwise mask.

    The unification also seeds `parent` with `extra_shapes` so the caller can
    canonicalize shapes outside the chain (e.g., raw inputs of consumed inner
    matmuls). Two ints unifying to different values would mean the input graph is
    ill-formed (matmul construction rejects mismatched contract dims and
    non-broadcastable batch dims), so the union prefers the int representative
    without checking for conflict.
    """
    parent: dict[DimEntry, DimEntry] = {}
    for shape in (*chain_shapes, *extra_shapes):
        for d in shape:
            parent.setdefault(d, d)

    def find(k: DimEntry) -> DimEntry:
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    def union(a: DimEntry, b: DimEntry) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if isinstance(ra, int):
            parent[rb] = ra
        elif isinstance(rb, int):
            parent[ra] = rb
        else:
            parent[rb] = ra

    for i in range(len(chain_shapes) - 1):
        union(chain_shapes[i][-1], chain_shapes[i + 1][-2])

    # Batch broadcast: at each right-aligned position across the whole chain, all
    # non-literal-1 entries must be equal. Unioning them as one class catches
    # transitive equalities a literal-1 in the middle would otherwise mask.
    max_batch = max((len(s) - 2 for s in chain_shapes), default=0)
    for pos in range(max_batch):
        anchor: DimEntry | None = None
        for s in chain_shapes:
            n_batch = len(s) - 2
            if pos >= n_batch:
                continue
            d = s[n_batch - 1 - pos]
            if isinstance(d, int) and d == 1:
                continue
            if anchor is None:
                anchor = d
            else:
                union(anchor, d)

    rep_entry: dict[DimEntry, DimEntry] = {}
    for shape in (*chain_shapes, *extra_shapes):
        for d in shape:
            r = find(d)
            if r not in rep_entry:
                rep_entry[r] = d
            elif isinstance(d, int) and not isinstance(rep_entry[r], int):
                rep_entry[r] = d

    def canonicalize(shape: Shape) -> Shape:
        return tuple(rep_entry[find(d)] if d in parent else d for d in shape)

    unified = [canonicalize(s) for s in chain_shapes]
    return unified, canonicalize


def _solve_chain(shapes: list[Shape]) -> tuple[CostExpr, dict]:
    """Standard matrix-chain DP over the symbolic operand shapes.

    Returns ``(best_total_cost, dp_table)`` where ``dp_table[(i, j)]`` is
    ``(cost, split_k, result_shape)`` for the subchain ``shapes[i..j]`` inclusive.
    Comparison uses ``_provably_less``; when neither candidate is provably less the
    first-found candidate (lower split index) wins as a deterministic tie-break.
    """
    n = len(shapes)
    dp: dict[tuple[int, int], tuple[CostExpr, int | None, Shape]] = {}
    for i in range(n):
        dp[(i, i)] = (CostExpr.zero(), None, shapes[i])

    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            best: tuple[CostExpr, int, Shape] | None = None
            for k in range(i, j):
                lc, _, ls = dp[(i, k)]
                rc, _, rs = dp[(k + 1, j)]
                step = _contract_cost(ls, rs)
                total = lc + rc + step
                result = _matmul_result_shape(ls, rs)
                if best is None or _provably_less(total, best[0]):
                    best = (total, k, result)

            # Length >= 2 always has at least one split -- invariant violated.
            assert best is not None, f"DP found no candidate for subchain ({i},{j})"

            dp[(i, j)] = best

    return dp[(0, n - 1)][0], dp


def _existing_cost(
    consumed: list[Apply],
    fgraph: FunctionGraph,
    canonicalize: Callable[[Shape], Shape],
) -> CostExpr:
    """Total FLOPs of the user's existing chain.

    Walks consumed matmul nodes in topological order (reversed insertion order). Each step looks up its input shapes
    in the running ``var_shape`` table. The chain leaves take shapes from ``_operand_shape_raw + canonicalize`` while
    intermediate matmul outputs come from ``_matmul_result_shape``.

    Lifted DimShuffles preserve FLOPs (they only touch size-1 batch dims or swap core dims), so the canonicalized
    raw-shape sum compares directly to ``_solve_chain``'s symbolic cost.
    """
    var_shape: dict[Variable, Shape] = {}
    total = CostExpr.zero()
    for node in reversed(consumed):
        l_input, r_input = node.inputs
        if l_input not in var_shape:
            var_shape[l_input] = canonicalize(_operand_shape_raw(l_input, fgraph))
        if r_input not in var_shape:
            var_shape[r_input] = canonicalize(_operand_shape_raw(r_input, fgraph))
        l_shape = var_shape[l_input]
        r_shape = var_shape[r_input]
        total = total + _contract_cost(l_shape, r_shape)
        var_shape[node.outputs[0]] = _matmul_result_shape(l_shape, r_shape)
    return total


# Mirrors the dtype gate inside Dot22.make_node (pytensor/tensor/blas.py). If that
# list grows (e.g., bfloat16) update both -- there's no canonical export to import.
_BLAS_DTYPES = ("float16", "float32", "float64", "complex64", "complex128")


def _select_emit_op(left: Variable, right: Variable) -> Variable:
    """Emit ``left @ right`` via the cheapest semantically-equivalent op.

    Routing:

    - 2-D float/complex pair: ``Dot22``.
    - 3-D float/complex pair whose batch dims share static broadcastability (both statically ``1``, or neither
      statically ``1``): ``BatchedDot``.
    - Anything else: ``matmul()``, which lowers to ``Blockwise(Dot)``.
    """
    l_dt, r_dt = left.type.dtype, right.type.dtype
    if l_dt != r_dt or l_dt not in _BLAS_DTYPES:
        return matmul(left, right)  # type: ignore[arg-type,no-any-return]
    if left.type.ndim == right.type.ndim == 2:
        return cast(Variable, Dot22()(left, right))
    if left.type.ndim == right.type.ndim == 3:
        if (left.type.shape[0] == 1) == (right.type.shape[0] == 1):
            return cast(Variable, BatchedDot()(left, right))
    return matmul(left, right)  # type: ignore[arg-type,no-any-return]


def _build_tree(
    operands: list[tuple[Variable, tuple[DimShuffle, ...]]],
    dp: dict,
    i_top: int,
    j_top: int,
) -> Variable:
    """Materialize the optimal matmul tree from the DP table.

    Walks the DP split tree in post-order using an explicit work stack. Like ``_decompose_operand``, this function
    avoids recursion because deep chains can blow the Python stack.
    """
    materialized: dict[tuple[int, int], Variable] = {}
    work: list[tuple[int, int, bool]] = [(i_top, j_top, False)]

    while work:
        i, j, ready = work.pop()
        if i == j:
            var, lifts = operands[i]
            for lift in reversed(lifts):
                var = cast(Variable, lift(var))
            materialized[(i, j)] = var
            continue
        _, split, _ = dp[(i, j)]
        if not ready:
            work.append((i, j, True))
            work.append((split + 1, j, False))
            work.append((i, split, False))
        else:
            left = materialized.pop((i, split))
            right = materialized.pop((split + 1, j))
            materialized[(i, j)] = _select_emit_op(left, right)

    return materialized[(i_top, j_top)]


class ReassociateMatmulChain(GraphRewriter):
    """Re-associate matmul chains when a strictly cheaper order can be proven.

    Runs after BLAS (1.7) and specialize (2.0). For each maximal chain of matmul
    links in the fgraph, runs a DP over symbolic operand shapes. Replaces the chain
    only when the new total cost provably beats the user's existing parenthesization
    (under "every dim symbol is positive").

    Decomposes liftable single-client ``DimShuffle(matmul(L, R))`` patterns into
    ``DimShuffle(L) @ DimShuffle(R)`` (swapping operand order for matrix-transpose)
    to expose longer chains. The lift commits atomically with the reassociation:
    ``_build_tree`` constructs the lifted vars and runs only when ``_provably_less``
    fires, so unsuccessful attempts add no graph nodes.

    The pass walks a snapshot toposort once: replacements inside the loop create new
    chain-link nodes that this pass does not re-examine. For matrix chain
    reassociation a single global optimum doesn't get better by re-running; the
    trade-off is that the pass skips lift-exposed extensions of an already-rewritten
    chain.
    """

    def apply(self, fgraph: FunctionGraph) -> None:
        # `visited` only tracks nodes consumed by *committed* rewrites. Toposort
        # visits leaves before consumers, so a leaf later absorbed into a longer
        # chain (via the consumer) must remain decomposable when the consumer is
        # reached. Without this, processing `C @ D` first in `(A @ B) @ (C @ D)`
        # would mark it visited and block the outer matmul from seeing the full
        # 4-element chain.
        visited: set[Apply] = set()

        for node in list(fgraph.toposort()):
            if node in visited or not _is_chain_link(node):
                continue

            top = _find_chain_top(node, fgraph)
            if top in visited:
                continue

            # Local snapshot so the recursive decomposition can avoid re-entering
            # the same chain link within this attempt without polluting `visited`
            # until we commit.
            local_visited = set(visited)
            local_visited.add(top)
            consumed: list[Apply] = [top]
            left_ops = _decompose_operand(
                top.inputs[0], fgraph, local_visited, consumed
            )
            right_ops = _decompose_operand(
                top.inputs[1], fgraph, local_visited, consumed
            )
            operands = [*left_ops, *right_ops]

            if len(operands) < 3:
                continue

            # Compute each operand's shape after applying its pending lifts. `lifts`
            # is outermost-first; apply in reverse so the innermost transformation
            # touches the raw shape first. ``_decompose_operand`` only propagates a
            # lift through a chain-link whose operand ndim matches the lift's input
            # ndim, so indexing into ``shape`` by ``lift.new_order`` is in-bounds.
            op_shapes: list[Shape] = []
            for base, lifts in operands:
                shape: Shape = _operand_shape_raw(base, fgraph)
                for lift in reversed(lifts):
                    shape = tuple(1 if x == "x" else shape[x] for x in lift.new_order)
                op_shapes.append(shape)

            if any(len(s) < 2 for s in op_shapes):
                continue

            # Pre-collect raw input shapes of every consumed matmul so unification
            # canonicalizes those symbols too. `_existing_cost` then uses the same
            # canonical reps as the DP, so the comparison can see equalities through
            # ShapeFeature symbols on either side of the chain.
            raw_extras = [
                _operand_shape_raw(inp, fgraph) for c in consumed for inp in c.inputs
            ]

            unified, canonicalize = _build_unification(op_shapes, raw_extras)
            new_cost, dp = _solve_chain(unified)
            old_cost = _existing_cost(consumed, fgraph, canonicalize)

            if not _provably_less(new_cost, old_cost):
                continue

            visited.update(consumed)

            new_out = _build_tree(operands, dp, 0, len(operands) - 1)
            old_out = top.outputs[0]
            copy_stack_trace(old_out, new_out)
            fgraph.replace(old_out, new_out, reason="reassociate_matmul")


reassociate_matmul_chain = ReassociateMatmulChain()
reassociate_matmul_chain.name = "reassociate_matmul_chain"

optdb.register(
    "reassociate_matmul_chain",
    reassociate_matmul_chain,
    "fast_run",
    position=2.5,
)
