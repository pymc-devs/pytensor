"""Ancestor-reachability bitsets for rewrites that contract nodes.

Some rewrites contract a group of nodes into a single node (fusing chains of
operations, merging sibling loops). Doing that without forming a cycle needs
cheap "does A depend on B?" queries, which these helpers answer with one integer
bitset per node.

The bitsets reflect data-dependency edges (``Apply.inputs``) only, so they must
be built and used before in-placing, whose destroy/view orderings they do not
capture.
"""

from collections.abc import Sequence
from functools import reduce
from operator import or_

from pytensor.graph.basic import Apply
from pytensor.graph.fg import FunctionGraph, Output


def ancestor_bitsets(
    fgraph: FunctionGraph,
    toposorted_nodes: Sequence[Apply] | None = None,
) -> tuple[dict[Apply | None, int], dict[Apply | None, int]]:
    """Build ``(ancestors, bitflags)`` reachability bitsets for ``fgraph``.

    Each node gets a one-hot ``bitflag`` (``1 << toposort_index``) and an
    ``ancestors`` bitset: the union of its inputs' ancestor bitsets plus its own
    flag. ``A`` is then an ancestor of ``C`` iff ``ancestors[C] & bitflags[A]``.

    For nodes ``A -> B -> C``, ``bitflags == {A: 0b001, B: 0b010, C: 0b100}`` and
    ``ancestors == {A: 0b001, B: 0b011, C: 0b111}``. Root variables
    (``owner is None``) map to ``0``, and every ``Output`` dummy Op shares a
    single high bit (nothing depends on them).

    Assigning the bitflags needs a topological ordering of the nodes; pass
    ``toposorted_nodes`` to reuse one already computed, else ``fgraph.toposort()``
    is called.
    """
    # Bit flag per node, in topological order so that a node's flag is a higher
    # bit than all of its ancestors'.
    if toposorted_nodes is None:
        toposorted_nodes = fgraph.toposort()
    bitflags: dict[Apply | None, int] = {
        node: 1 << i for i, node in enumerate(toposorted_nodes)
    }
    # The ancestor bitset of each node is the union of its inputs' bitsets, plus
    # its own flag. Root variables have `None` as owner, handled with a bitset 0.
    ancestors: dict[Apply | None, int] = {None: 0}
    for node, node_bitflag in bitflags.items():
        ancestors[node] = reduce(
            or_,
            (ancestors[inp.owner] for inp in node.inputs),  # type: ignore[union-attr]
            node_bitflag,
        )
    # Done after building `ancestors` to keep the loop above simple.
    bitflags[None] = 0
    # Nothing ever depends on the special Output nodes, so they all share a
    # single new bit.
    out_bitflag = 1 << len(bitflags)
    bitflags |= (
        (client, out_bitflag)
        for out in fgraph.outputs
        for client, _ in fgraph.clients[out]
        if isinstance(client.op, Output)
    )
    return ancestors, bitflags


def greedy_independent_subset(
    ancestors: dict[Apply | None, int],
    bitflags: dict[Apply | None, int],
    candidates: Sequence[Apply],
) -> list[Apply]:
    """Greedy maximal subset of ``candidates`` that is pairwise independent.

    Two nodes are independent when neither is a (transitive) ancestor of the
    other. This walks ``candidates`` in order, keeping each one that is
    independent of all already kept, so the first candidate is always kept. The
    result is *maximal* (none of the dropped candidates can be added back), not
    *maximum* (the largest possible, which is NP-hard to find).

    A pairwise-independent set can be contracted into a single node without
    forming a cycle, since no member reaches another.
    """
    kept: list[Apply] = []
    kept_bits = 0
    kept_ancestors = 0
    for node in candidates:
        anc = ancestors[node]
        bf = bitflags[node]
        # `anc & kept_bits`: a kept node is an ancestor of `node`.
        # `bf & kept_ancestors`: `node` is an ancestor of a kept node.
        if (anc & kept_bits) or (bf & kept_ancestors):
            continue
        kept.append(node)
        kept_bits |= bf
        kept_ancestors |= anc
    return kept


def update_ancestors_after_contraction_bits(
    ancestors: dict[Apply | None, int],
    member_bits: int,
    combined_ancestors: int,
) -> None:
    """Update ``ancestors`` in place after contracting the nodes in ``member_bits``.

    After contraction every node that depended on *any* member depends on *all*
    of them, so each such node's ancestor bitset gains ``combined_ancestors``
    (the union of the members' ancestor closures). Only existing values change
    (no keys added or removed), so callers' aliases of ``ancestors`` stay valid.

    This is the low-level form, taking the contracted set's bitset and ancestor
    closure directly; :func:`update_ancestors_after_contraction` derives them
    from the contracted nodes.
    """
    for node, anc in ancestors.items():
        if anc & member_bits:
            ancestors[node] = anc | combined_ancestors


def update_ancestors_after_contraction(
    ancestors: dict[Apply | None, int],
    bitflags: dict[Apply | None, int],
    nodes: Sequence[Apply],
) -> None:
    """Update ``ancestors`` in place after contracting ``nodes`` into one node.

    Derives the contracted set's bitset and combined ancestor closure from
    ``nodes`` and defers to :func:`update_ancestors_after_contraction_bits`.
    """
    member_bits = reduce(or_, (bitflags[node] for node in nodes), 0)
    combined_ancestors = reduce(or_, (ancestors[node] for node in nodes), 0)
    update_ancestors_after_contraction_bits(ancestors, member_bits, combined_ancestors)
