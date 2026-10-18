import math
from collections.abc import Callable, Sequence
from typing import cast

from pytensor.compile.mode import optdb
from pytensor.graph.basic import Apply, Variable
from pytensor.graph.cost import (
    FLOPS_PER_MULTIPLY_ADD,
    CostExpr,
    DimEntry,
    _polynomial_le,
)
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.rewriting.basic import GraphRewriter, copy_stack_trace
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.elemwise import DimShuffle
from pytensor.tensor.math import Dot, _dot, matmul
from pytensor.tensor.rewriting.shape import ShapeFeature


Shape = tuple[DimEntry, ...]


def _operand_shape_raw(var: Variable, fgraph: FunctionGraph) -> Shape:
    """Shape of ``var``: static lengths as ints, the rest as ShapeFeature symbols."""
    symbolic = fgraph.shape_feature.shape_tuple(var)
    return tuple(
        symbolic[i] if length is None else int(length)
        for i, length in enumerate(var.type.shape)
    )


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
    """FLOPs of ``left @ right``."""
    return CostExpr.from_dim_product(
        [FLOPS_PER_MULTIPLY_ADD, *_matmul_result_shape(left, right), left[-1]]
    )


def _dimshuffle_lift(op: DimShuffle) -> tuple[bool, bool]:
    """Classify lifting the DimShuffle `op` through the matmul it wraps.

    A DimShuffle commutes with matmul iff it touches only batch dimensions (the
    leading ``input_ndim - 2``), or it performs a matrix-transpose (swap of the last
    two dims) -- possibly combined with batch-only operations. Both matmul operands
    have the DimShuffle's input ndim, because ``Blockwise`` pads its inputs to a common
    batch ndim and a core ``Dot`` is 2-D. Returns ``(is_liftable, swaps_order)``, where
    ``swaps_order`` is True iff the lift swaps operand order in the matmul:
    ``(L @ R).T = R.T @ L.T``.

    DimShuffle disallows duplicate input indices in `new_order`, so once the core dim
    indices appear in the last two positions of the output they cannot appear
    elsewhere -- earlier positions are batch-only.
    """
    input_ndim = op.input_ndim
    new_order = op.new_order
    if input_ndim < 2 or len(new_order) < 2:
        return False, False
    last_two = (new_order[-2], new_order[-1])
    if last_two == (input_ndim - 2, input_ndim - 1):
        return True, False
    if last_two == (input_ndim - 1, input_ndim - 2):
        return True, True
    return False, False


def _is_chain_link(node: Apply) -> bool:
    """True iff `node` is a 2-operand matmul that ``matmul()`` can rebuild."""
    op = node.op
    return isinstance(op, Dot) or (
        isinstance(op, Blockwise) and isinstance(op.core_op, Dot)
    )


def _next_link(node: Apply, fgraph: FunctionGraph) -> Apply | None:
    """The chain link that consumes `node`'s output, if any.

    The output must have a single client that is a chain link, or a single-client
    liftable DimShuffle whose only client is a chain link.
    """
    clients = fgraph.clients[node.outputs[0]]
    if len(clients) != 1:
        return None
    client_node, _ = clients[0]
    if _is_chain_link(client_node):
        return client_node

    if (
        isinstance(client_node.op, DimShuffle)
        and len(fgraph.clients[client_node.outputs[0]]) == 1
    ):
        ds_consumer, _ = fgraph.clients[client_node.outputs[0]][0]
        if _is_chain_link(ds_consumer):
            liftable, _ = _dimshuffle_lift(client_node.op)
            if liftable:
                return ds_consumer
    return None


def _find_chain_top(start: Apply, fgraph: FunctionGraph) -> Apply:
    """Walk up to the topmost chain link along single-client edges."""
    current = start
    while (consumer := _next_link(current, fgraph)) is not None:
        current = consumer
    return current


def _decompose_operand(
    root: Variable,
    fgraph: FunctionGraph,
    visited: set[Apply],
    consumed: list[Apply],
) -> list[tuple[Variable, tuple[DimShuffle, ...]]]:
    """Iteratively decompose `root` into chain leaves.

    Each leaf is ``(base_var, lifts)`` where `lifts` is a tuple of DimShuffle Ops to
    apply at materialization time (outermost first).

    Two descent paths grow the chain:

    - Single-client chain-link matmul: descend into both inputs.
    - Single-client liftable DimShuffle wrapping a single-client chain-link matmul:
      descend into the inner matmul's two inputs with the DimShuffle prepended to the
      inherited-lift list. For matrix-transpose lifts, swapping operand order:
      (``(L @ R).T = R.T @ L.T``).

    Matmul operands share the matmul's output ndim (``Blockwise`` pads them), so an
    inherited lift always indexes in bounds.

    Append each chain-link Apply descended into to `consumed` and add it to `visited`.
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
                liftable, swaps = _dimshuffle_lift(ds_op)
                if liftable:
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


def _solve_chain(shapes: list[tuple[int, ...]]) -> tuple[int, dict]:
    """Matrix-chain DP over integer operand shapes.

    Returns ``(best_total_cost, dp_table)`` where ``dp_table[(i, j)]`` is
    ``(cost, split_k, result_shape)`` for the subchain ``shapes[i..j]`` inclusive.
    Ties keep the lower split index.
    """
    n = len(shapes)
    dp: dict[tuple[int, int], tuple[int, int | None, tuple[int, ...]]] = {
        (i, i): (0, None, shapes[i]) for i in range(n)
    }

    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            best = None
            for k in range(i, j):
                lc, _, ls = dp[(i, k)]
                rc, _, rs = dp[(k + 1, j)]
                result = cast(tuple[int, ...], _matmul_result_shape(ls, rs))
                flops = FLOPS_PER_MULTIPLY_ADD * math.prod(result) * ls[-1]
                total = lc + rc + flops
                if best is None or total < best[0]:
                    best = (total, k, result)

            # Length >= 2 always has at least one split -- invariant violated.
            assert best is not None, f"DP found no candidate for subchain ({i},{j})"
            dp[(i, j)] = best

    return dp[(0, n - 1)][0], dp


def _tree_cost(shapes: list[Shape], dp: dict) -> CostExpr:
    """Symbolic cost of the parenthesization recorded in `dp`, over `shapes`."""
    total = CostExpr.zero()
    result_shape: dict[tuple[int, int], Shape] = {}
    work = [(0, len(shapes) - 1, False)]
    while work:
        i, j, ready = work.pop()
        if i == j:
            result_shape[(i, j)] = shapes[i]
            continue
        split = dp[(i, j)][1]
        if not ready:
            work.extend([(i, j, True), (split + 1, j, False), (i, split, False)])
            continue
        left, right = result_shape[(i, split)], result_shape[(split + 1, j)]
        total = total + _contract_cost(left, right)
        result_shape[(i, j)] = _matmul_result_shape(left, right)
    return total


def _existing_cost(
    consumed: list[Apply],
    fgraph: FunctionGraph,
    canonicalize: Callable[[Shape], Shape],
) -> CostExpr:
    """Total FLOPs of the user's existing chain, comparable to ``_tree_cost``.

    Visits `consumed` in reverse, which is a topological order. Chain leaves take their
    canonicalized raw shapes, and intermediate matmul outputs take
    ``_matmul_result_shape`` of their inputs.
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


def _build_tree(
    operands: list[tuple[Variable, tuple[DimShuffle, ...]]],
    dp: dict,
) -> Variable:
    """Materialize the matmul tree recorded in the DP table over `operands`."""
    top = (0, len(operands) - 1)
    materialized: dict[tuple[int, int], Variable] = {}
    work: list[tuple[int, int, bool]] = [(*top, False)]

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
            # Canonicalize lowers unbatched matmuls to the core Dot before this pass
            # runs, and BlasOpt only specializes that form
            if left.type.ndim == right.type.ndim == 2:
                materialized[(i, j)] = cast(Variable, _dot(left, right))
            else:
                materialized[(i, j)] = matmul(left, right)  # type: ignore[arg-type]

    return materialized[top]


class ReassociateMatmulChain(GraphRewriter):
    """Re-associate matmul chains when a strictly cheaper order can be proven.

    Runs before BlasOpt (1.7) and emits the same core ``Dot`` / ``matmul`` forms
    canonicalize produces, so the usual BLAS and backend specializations apply to the
    rebuilt chain. For each maximal chain of matmul links in the fgraph, runs a
    matrix-chain DP over the operand shapes and replaces the chain only when the new
    total cost provably beats the user's existing parenthesization (under "every dim
    symbol is positive").

    Decomposes liftable single-client ``DimShuffle(matmul(L, R))`` patterns into
    ``DimShuffle(L) @ DimShuffle(R)`` (swapping operand order for matrix-transpose)
    to expose longer chains. The lift commits atomically with the reassociation:
    ``_build_tree`` constructs the lifted vars and runs only when the new cost is
    provably no more than the old one and differs from it, so unsuccessful attempts
    add no graph nodes.

    The pass walks a snapshot toposort once: replacements inside the loop create new
    chain-link nodes that this pass does not re-examine. For matrix chain
    reassociation a single global optimum doesn't get better by re-running; the
    trade-off is that the pass skips lift-exposed extensions of an already-rewritten
    chain.
    """

    def add_requirements(self, fgraph: FunctionGraph) -> None:
        if not hasattr(fgraph, "shape_feature"):
            fgraph.attach_feature(ShapeFeature())

    def apply(self, fgraph: FunctionGraph) -> None:
        # A chain needs one matmul feeding another. Most graphs have none, so check
        # that with a linear scan before paying for a toposort.
        if not any(
            _is_chain_link(node) and _next_link(node, fgraph) is not None
            for node in fgraph.apply_nodes
        ):
            return

        # Only nodes consumed by committed rewrites. Toposort reaches `C @ D` before
        # the outer matmul of `(A @ B) @ (C @ D)`, and marking it on a failed attempt
        # would hide it from the full four-operand chain.
        visited: set[Apply] = set()
        # Every link of a chain climbs to the same top, so a top whose attempt failed
        # would otherwise be decomposed and solved again once per link
        attempted: set[Apply] = set()

        for node in list(fgraph.toposort()):
            if node in visited or not _is_chain_link(node):
                continue

            top = _find_chain_top(node, fgraph)
            if top in visited or top in attempted:
                continue
            attempted.add(top)

            # Snapshot so this attempt does not re-enter its own links, while
            # `visited` stays untouched until the rewrite commits
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

            # `lifts` is outermost-first, so the innermost lift applies to the raw shape
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
            old_cost = _existing_cost(consumed, fgraph, canonicalize)
            # Pick the candidate order with the integer DP, which is what numpy's
            # multi_dot runs, after giving each shape symbol a large distinct value.
            # A provable win is strictly cheaper at such a generic point, so a
            # candidate that is not cheaper there ends the attempt before any
            # polynomial work. Otherwise its symbolic cost is proven once.
            symbols = dict.fromkeys(
                d for shape in unified for d in shape if not isinstance(d, int)
            )
            point = {sym: 1009 + 7919 * i for i, sym in enumerate(symbols)}
            numeric = [
                tuple(d if isinstance(d, int) else point[d] for d in shape)
                for shape in unified
            ]
            numeric_cost, dp = _solve_chain(numeric)
            if numeric_cost >= old_cost.evaluate(point):
                continue
            new_cost = _tree_cost(unified, dp)

            if new_cost == old_cost or not _polynomial_le(new_cost, old_cost):
                continue

            visited.update(consumed)

            new_out = _build_tree(operands, dp)
            old_out = top.outputs[0]
            copy_stack_trace(old_out, new_out)
            fgraph.replace(old_out, new_out, reason="reassociate_matmul")


reassociate_matmul_chain = ReassociateMatmulChain()
reassociate_matmul_chain.name = "reassociate_matmul_chain"

optdb.register(
    "reassociate_matmul_chain",
    reassociate_matmul_chain,
    "fast_run",
    position=1.65,
)
