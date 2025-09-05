import copy
from collections.abc import Generator, Sequence
from typing import TYPE_CHECKING, Optional

import pytensor
from pytensor.graph.basic import (
    Apply,
    Variable,
    equal_computations,
)
from pytensor.graph.fg import FunctionGraph, Output
from pytensor.graph.rewriting.db import RewriteDatabaseQuery
from pytensor.graph.traversal import graph_inputs, vars_between


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
        fgraph = FunctionGraph(outputs=outputs, clone=clone)

    query_rewrites = optdb.query(RewriteDatabaseQuery(include=include, **kwargs))
    query_rewrites.rewrite(fgraph)

    if custom_rewrite is not None:
        custom_rewrite.rewrite(fgraph)

    if isinstance(graph, FunctionGraph):
        return fgraph
    if isinstance(graph, Variable):
        return fgraph.outputs[0]
    return fgraph.outputs


def is_same_graph_with_merge(
    var1: Variable,
    var2: Variable,
    givens: (
        list[tuple[Variable, Variable]]
        | tuple[tuple[Variable, Variable], ...]
        | dict[Variable, Variable]
        | None
    ) = None,
) -> bool:
    """
    Merge-based implementation of `pytensor.graph.basic.is_same_graph`.

    See help on `pytensor.graph.basic.is_same_graph` for additional documentation.

    """
    from pytensor.graph.rewriting.basic import MergeOptimizer

    givens = {} if givens is None else dict(givens)

    # Copy variables since the MergeOptimizer will modify them.
    *vars, givens = copy.deepcopy((var1, var2, givens))
    # Create FunctionGraph.
    inputs = list(graph_inputs(vars))
    # The clone isn't needed as we did a deepcopy and we cloning will
    # break the mapping in givens.
    fgraph = pytensor.graph.fg.FunctionGraph(inputs, vars, clone=False)
    # Perform Variable substitution.
    for to_replace, replace_by in givens.items():
        fgraph.replace(to_replace, replace_by)
    # Perform merge optimization.
    MergeOptimizer().rewrite(fgraph)
    # When two variables perform the same computations, they will have the same
    # owner in the rewritten graph.
    # We need to be careful with the special case where the owner is None,
    # which happens when the graph is made of a single Variable.
    # We also need to make sure we replace a Variable if it is present in
    # `givens`.
    vars_replaced = [givens.get(v, v) for v in fgraph.outputs]
    o1, o2 = (v.owner for v in vars_replaced)
    if o1 is None and o2 is None:
        # Comparing two single-Variable graphs: they are equal if they are
        # the same Variable.
        return vars_replaced[0] == vars_replaced[1]
    return o1 is o2


def is_same_graph(
    var1: Variable,
    var2: Variable,
    givens: (
        list[tuple[Variable, Variable]]
        | tuple[tuple[Variable, Variable], ...]
        | dict[Variable, Variable]
        | None
    ) = None,
) -> bool:
    """
    Return True iff Variables `var1` and `var2` perform the same computation.

    By 'performing the same computation', we mean that they must share the same
    graph, so that for instance this function will return False when comparing
    (x * (y * z)) with ((x * y) * z).

    The current implementation is not efficient since, when possible, it
    verifies equality by calling two different functions that are expected to
    return the same output. The goal is to verify this assumption, to
    eventually get rid of one of them in the future.

    Parameters
    ----------
    var1
        The first Variable to compare.
    var2
        The second Variable to compare.
    givens
        Similar to the `givens` argument of `pytensor.function`, it can be used
        to perform substitutions in the computational graph of `var1` and
        `var2`. This argument is associated to neither `var1` nor `var2`:
        substitutions may affect both graphs if the substituted variable
        is present in both.

    Examples
    --------

        ======  ======  ======  ======
        var1    var2    givens  output
        ======  ======  ======  ======
        x + 1   x + 1   {}      True
        x + 1   y + 1   {}      False
        x + 1   y + 1   {x: y}  True
        ======  ======  ======  ======

    """
    givens = {} if givens is None else dict(givens)

    # Get result from the merge-based function.
    rval1 = is_same_graph_with_merge(var1=var1, var2=var2, givens=givens)

    if not givens:
        rval2 = equal_computations(xs=[var1], ys=[var2])
        assert rval1 == rval2
        return rval1

    # We need to build the `in_xs` and `in_ys` lists. To do this, we need
    # to be able to tell whether a variable belongs to the computational
    # graph of `var1` or `var2`.
    # The typical case we want to handle is when `to_replace` belongs to
    # one of these graphs, and `replace_by` belongs to the other one. In
    # other situations, the current implementation of `equal_computations`
    # is probably not appropriate, so we do not call it.
    use_equal_computations = True
    in_xs = []
    in_ys = []
    # Compute the sets of all variables found in each computational graph.
    inputs_var1 = graph_inputs([var1])
    inputs_var2 = graph_inputs([var2])
    all_vars1 = set(vars_between(inputs_var1, [var1]))
    all_vars2 = set(vars_between(inputs_var2, [var2]))

    for to_replace, replace_by in givens.items():
        # Map a substitution variable to the computational graphs it
        # belongs to.
        inside = {v: [v in all_vars1, v in all_vars2] for v in (to_replace, replace_by)}
        if (
            inside[to_replace][0]
            and not inside[to_replace][1]
            and inside[replace_by][1]
            and not inside[replace_by][0]
        ):
            # Substitute variable in `var1` by one from `var2`.
            in_xs.append(to_replace)
            in_ys.append(replace_by)
        elif (
            inside[to_replace][1]
            and not inside[to_replace][0]
            and inside[replace_by][0]
            and not inside[replace_by][1]
        ):
            # Substitute variable in `var2` by one from `var1`.
            in_xs.append(replace_by)
            in_ys.append(to_replace)
        else:
            use_equal_computations = False
            break

    if use_equal_computations:
        rval2 = equal_computations(xs=[var1], ys=[var2], in_xs=in_xs, in_ys=in_ys)
        assert rval2 == rval1
    return rval1


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
