from collections.abc import Generator, Iterable, Sequence
from typing import TYPE_CHECKING, Optional, cast

from pytensor.graph.basic import (
    Apply,
    Variable,
)
from pytensor.graph.fg import FunctionGraph, Output
from pytensor.graph.rewriting.db import RewriteDatabaseQuery


if TYPE_CHECKING:
    from pytensor.graph.rewriting.basic import GraphRewriter


def rewrite_graph(
    graph: Variable | Sequence[Variable] | FunctionGraph,
    include: Sequence[str] = ("canonicalize",),
    custom_rewrite: Optional["GraphRewriter"] = None,
    clone: bool = False,
    **kwargs,
) -> Variable | Sequence[Variable] | FunctionGraph:
    """Easily apply rewrites to a graph.

    Parameters
    ----------
    graph
        A `FunctionGraph` or `Variable` to be rewritten.
    include
        String names of the rewrites to be queried, via a
        `RewriteDatabaseQuery` instance, and applied.  The default rewrite
        query string is ``"canonicalization"``.
    custom_rewrite
        A custom `Rewriter` to also be applied.
    clone
        Whether or not to clone the input graph before rewriting.
    **kwargs
        Keyword arguments passed to a `RewriteDatabaseQuery` object.
    """
    from pytensor.compile import optdb

    if isinstance(graph, FunctionGraph):
        fgraph = graph
    else:
        outputs = [graph] if isinstance(graph, Variable) else graph
        fgraph = FunctionGraph(outputs=outputs, clone=clone, copy_inputs=False)

    query_rewrites = optdb.query(RewriteDatabaseQuery(include=include, **kwargs))
    query_rewrites.rewrite(fgraph)

    if custom_rewrite is not None:
        custom_rewrite.rewrite(fgraph)

    if isinstance(graph, FunctionGraph):
        return fgraph
    if isinstance(graph, Variable):
        return fgraph.outputs[0]
    return fgraph.outputs


def rewrite_subgraph(
    outputs: Sequence[Variable],
    frontier: Iterable[Variable],
    include: Sequence[str] = ("canonicalize",),
    **kwargs,
) -> list[Variable]:
    """Rewrite the subgraph between ``frontier`` and ``outputs`` in isolation.

    The ``frontier`` variables are temporarily detached from their owners, so
    they act as inputs of the subgraph: rewrites can neither reach past them
    nor modify the graph they belong to. This allows simplifying fresh
    expressions that hang off the variables of an existing `FunctionGraph`
    without mutating it behind its (and its features') back.

    The rewrite is in place: ``outputs`` must not belong to a `FunctionGraph`.

    Parameters
    ----------
    outputs
        The outputs of the subgraph to rewrite.
    frontier
        Variables at which the subgraph stops; every path from ``outputs``
        into the surrounding graph must go through one of them.
    include
        Rewrite query names, as in `rewrite_graph`.
    **kwargs
        Keyword arguments passed to `rewrite_graph`.
    """
    saved_owners = [(v, v.owner, v.index) for v in frontier]
    for v, _, _ in saved_owners:
        v.owner = None
    try:
        rewritten = cast(
            Sequence[Variable],
            rewrite_graph(list(outputs), include=include, clone=False, **kwargs),
        )
        return list(rewritten)
    finally:
        for v, owner, idx in saved_owners:
            v.owner = owner
            v.index = idx


def get_clients_at_depth(
    fgraph: FunctionGraph, node: Apply, depth: int
) -> Generator[Apply, None, None]:
    """Yields node clients at given depth."""
    for var in node.outputs:
        if depth > 0:
            for out_node, _ in fgraph.clients[var]:
                if isinstance(out_node.op, Output):
                    continue
                yield from get_clients_at_depth(fgraph, out_node, depth - 1)
        else:
            assert var.owner is not None
            yield var.owner
