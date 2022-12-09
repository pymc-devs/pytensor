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

from pytensor.graph.basic import Constant, Variable


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
