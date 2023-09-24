from functools import partial, singledispatch
from typing import Iterable, Mapping, Optional, Sequence, Union, cast, overload

from pytensor.graph.basic import Apply, Constant, Variable, truncated_graph_inputs
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.op import Op


ReplaceTypes = Union[Iterable[tuple[Variable, Variable]], dict[Variable, Variable]]


def _format_replace(replace: Optional[ReplaceTypes] = None) -> dict[Variable, Variable]:
    items: dict[Variable, Variable]
    if isinstance(replace, dict):
        # PyLance has issues with type resolution
        items = cast(dict[Variable, Variable], replace)
    elif isinstance(replace, Iterable):
        items = dict(replace)
    elif replace is None:
        items = {}
    else:
        raise ValueError(
            "replace is neither a dictionary, list, "
            f"tuple or None ! The value provided is {replace},"
            f"of type {type(replace)}"
        )
    return items


@overload
def clone_replace(
    output: Sequence[Variable],
    replace: Optional[ReplaceTypes] = None,
    **rebuild_kwds,
) -> list[Variable]:
    ...


@overload
def clone_replace(
    output: Variable,
    replace: Optional[
        Union[Iterable[tuple[Variable, Variable]], dict[Variable, Variable]]
    ] = None,
    **rebuild_kwds,
) -> Variable:
    ...


def clone_replace(
    output: Union[Sequence[Variable], Variable],
    replace: Optional[ReplaceTypes] = None,
    **rebuild_kwds,
) -> Union[list[Variable], Variable]:
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

    items = list(_format_replace(replace).items())

    tmp_replace = [(x, x.type()) for x, y in items]
    new_replace = [(x, y) for ((_, x), (_, y)) in zip(tmp_replace, items)]
    _, _outs, _ = rebuild_collect_shared(output, [], tmp_replace, [], **rebuild_kwds)

    # TODO Explain why we call it twice ?!
    _, outs, _ = rebuild_collect_shared(_outs, [], new_replace, [], **rebuild_kwds)

    return outs


@overload
def graph_replace(
    outputs: Variable,
    replace: Optional[ReplaceTypes] = None,
    *,
    strict=True,
) -> Variable:
    ...


@overload
def graph_replace(
    outputs: Sequence[Variable],
    replace: Optional[ReplaceTypes] = None,
    *,
    strict=True,
) -> list[Variable]:
    ...


def graph_replace(
    outputs: Union[Sequence[Variable], Variable],
    replace: Optional[ReplaceTypes] = None,
    *,
    strict=True,
) -> Union[list[Variable], Variable]:
    """Replace variables in ``outputs`` by ``replace``.

    Parameters
    ----------
    outputs: Union[Sequence[Variable], Variable]
        Output graph
    replace: Dict[Variable, Variable]
        Replace mapping
    strict: bool
        Raise an error if some replacements were not used
    return_unused: bool
        Return replacements that were not used

    Returns
    -------
    Union[Variable, List[Variable]]
        Output graph with subgraphs replaced, see function overload for the exact type

    Raises
    ------
    ValueError
        If some replacements could not be applied and strict is True
    """
    as_list = False
    if not isinstance(outputs, Sequence):
        outputs = [outputs]
    else:
        as_list = True
    replace_dict = _format_replace(replace)
    # collect minimum graph inputs which is required to compute outputs
    # and depend on replacements
    # additionally remove constants, they do not matter in clone get equiv
    conditions = [
        c
        for c in truncated_graph_inputs(outputs, replace_dict)
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
    fg_replace.update({equiv[r]: v for r, v in replace_dict.items() if r in equiv})
    # replacements have to be done in reverse topological order so that nested
    # expressions get recursively replaced correctly

    # some replacements may be initially outside the graph
    # but later introduced by a replacement
    # So far FunctionGraph does these replacements inplace it is thus unsafe
    # apply them using fg.replace, it may change the original graph
    if strict:
        non_fg_replace = {r: v for r, v in replace_dict.items() if r not in equiv}
        if non_fg_replace:
            raise ValueError(f"Some replacements were not used: {non_fg_replace}")
    toposort = fg.toposort()

    def toposort_key(
        fg: FunctionGraph, ts: list[Apply], pair: tuple[Variable, Variable]
    ) -> int:
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
    if as_list:
        return list(fg.outputs)
    else:
        return fg.outputs[0]


@singledispatch
def _vectorize_node(op: Op, node: Apply, *bached_inputs) -> Apply:
    # Default implementation is provided in pytensor.tensor.blockwise
    raise NotImplementedError


def vectorize_node(node: Apply, *batched_inputs) -> Apply:
    """Returns vectorized version of node with new batched inputs."""
    op = node.op
    return _vectorize_node(op, node, *batched_inputs)


@overload
def vectorize(
    outputs: Variable,
    replace: Mapping[Variable, Variable],
) -> Variable:
    ...


@overload
def vectorize(
    outputs: Sequence[Variable],
    replace: Mapping[Variable, Variable],
) -> Sequence[Variable]:
    ...


def vectorize(
    outputs: Union[Variable, Sequence[Variable]],
    replace: Mapping[Variable, Variable],
) -> Union[Variable, Sequence[Variable]]:
    """Vectorize outputs graph given mapping from old variables to expanded counterparts version.

    Expanded dimensions must be on the left. Behavior is similar to the functional `numpy.vectorize`.

    Examples
    --------
    .. code-block:: python

        import pytensor
        import pytensor.tensor as pt

        from pytensor.graph import vectorize

        # Original graph
        x = pt.vector("x")
        y = pt.exp(x) / pt.sum(pt.exp(x))

        # Vectorized graph
        new_x = pt.matrix("new_x")
        new_y = vectorize(y, replace={x: new_x})

        fn = pytensor.function([new_x], new_y)
        fn([[0, 1, 2], [2, 1, 0]])
        # array([[0.09003057, 0.24472847, 0.66524096],
        #        [0.66524096, 0.24472847, 0.09003057]])


    .. code-block:: python

        import pytensor
        import pytensor.tensor as pt

        from pytensor.graph import vectorize

        # Original graph
        x = pt.vector("x")
        y1 = x[0]
        y2 = x[-1]

        # Vectorized graph
        new_x = pt.matrix("new_x")
        [new_y1, new_y2] = vectorize([y1, y2], replace={x: new_x})

        fn = pytensor.function([new_x], [new_y1, new_y2])
        fn([[-10, 0, 10], [-11, 0, 11]])
        # [array([-10., -11.]), array([10., 11.])]

    """
    if isinstance(outputs, Sequence):
        seq_outputs = outputs
    else:
        seq_outputs = [outputs]

    inputs = truncated_graph_inputs(seq_outputs, ancestors_to_include=replace.keys())
    new_inputs = [replace.get(inp, inp) for inp in inputs]

    def transform(var: Variable) -> Variable:
        if var in inputs:
            return new_inputs[inputs.index(var)]

        node = var.owner
        batched_inputs = [transform(inp) for inp in node.inputs]
        batched_node = vectorize_node(node, *batched_inputs)
        batched_var = batched_node.outputs[var.owner.outputs.index(var)]

        return cast(Variable, batched_var)

    # TODO: MergeOptimization or node caching?
    seq_vect_outputs = [transform(out) for out in seq_outputs]

    if isinstance(outputs, Sequence):
        return seq_vect_outputs
    else:
        [vect_output] = seq_vect_outputs
        return vect_output
