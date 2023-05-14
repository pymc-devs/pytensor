from functools import partial
from typing import (
    Collection,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

from pytensor.graph.basic import Constant, Variable, truncated_graph_inputs
from pytensor.graph.fg import FunctionGraph


def clone_replace(
    output: Collection[Variable],
    replace: Optional[
        Union[Iterable[Tuple[Variable, Variable]], Dict[Variable, Variable]]
    ] = None,
    **rebuild_kwds,
) -> List[Variable]:
    """Clone a graph and replace subgraphs within it.

    It returns a copy of the initial subgraph with the corresponding
    substitutions.

    Parameters
    ----------
    output
        PyTensor expression that represents the computational graph.
    replace
        Dictionary describing which subgraphs should be replaced by what.
    rebuild_kwds
        Keywords to `rebuild_collect_shared`.

    """
    from pytensor.compile.function.pfunc import rebuild_collect_shared

    items: Union[List[Tuple[Variable, Variable]], Tuple[Tuple[Variable, Variable], ...]]
    if isinstance(replace, dict):
        items = list(replace.items())
    elif isinstance(replace, (list, tuple)):
        items = replace
    elif replace is None:
        items = []
    else:
        raise ValueError(
            "replace is neither a dictionary, list, "
            f"tuple or None ! The value provided is {replace},"
            f"of type {type(replace)}"
        )
    tmp_replace = [(x, x.type()) for x, y in items]
    new_replace = [(x, y) for ((_, x), (_, y)) in zip(tmp_replace, items)]
    _, _outs, _ = rebuild_collect_shared(output, [], tmp_replace, [], **rebuild_kwds)

    # TODO Explain why we call it twice ?!
    _, outs, _ = rebuild_collect_shared(_outs, [], new_replace, [], **rebuild_kwds)

    return cast(List[Variable], outs)


def graph_replace(
    outputs: Sequence[Variable],
    replace: Dict[Variable, Variable],
    *,
    strict=True,
) -> List[Variable]:
    """Replace variables in ``outputs`` by ``replace``.

    Parameters
    ----------
    outputs: Sequence[Variable]
        Output graph
    replace: Dict[Variable, Variable]
        Replace mapping
    strict: bool
        Raise an error if some replacements were not used
    return_unused: bool
        Return replacements that were not used

    Returns
    -------
    List[Variable]
        Output graph with subgraphs replaced

    Raises
    ------
    ValueError
        If some replacemens could not be applied and strict is True
    """
    # collect minimum graph inputs which is required to compute outputs
    # and depend on replacements
    # additionally remove constants, they do not matter in clone get equiv
    conditions = [
        c
        for c in truncated_graph_inputs(outputs, replace)
        if not isinstance(c, Constant)
    ]
    # for the function graph we need the clean graph where
    # inputs do not have owners
    # this is exactly the reason to clone conditions
    equiv = {c: c.clone(name=f"i-{i}") for i, c in enumerate(conditions)}
    # some replace keys may disappear
    # the reason is they are outside the graph
    # clone the graph but preserve the equiv mapping
    fg = FunctionGraph(
        conditions,
        outputs,
        # clone_get_equiv kwargs
        copy_orphans=False,
        copy_inputs=False,
        memo=equiv,
    )
    # replace the conditions back
    fg_replace = {equiv[c]: c for c in conditions}
    # add the replacements on top of input mappings
    fg_replace.update({equiv[r]: v for r, v in replace.items() if r in equiv})
    # replacements have to be done in reverse topological order so that nested
    # expressions get recursively replaced correctly

    # some replacements may be initially outside the graph
    # but later introduced by a replacement
    # So far FunctionGraph does these replacements inplace it is thus unsafe
    # apply them using fg.replace, it may change the original graph
    if strict:
        non_fg_replace = {r: v for r, v in replace.items() if r not in equiv}
        if non_fg_replace:
            raise ValueError(f"Some replacements were not used: {non_fg_replace}")
    toposort = fg.toposort()

    def toposort_key(fg: FunctionGraph, ts, pair):
        key, _ = pair
        if key.owner is not None:
            return ts.index(key.owner)
        else:
            if key in fg.variables:
                return -1
            else:
                raise ValueError(f"{key} is not a part of graph")

    sorted_replacements = sorted(
        fg_replace.items(),
        # sort based on the fg toposort, if a variable has no owner, it goes first
        key=partial(toposort_key, fg, toposort),
        reverse=True,
    )
    fg.replace_all(sorted_replacements, import_missing=True)
    return list(fg.outputs)
