from collections import deque
from collections.abc import (
    Callable,
    Generator,
    Iterable,
    Reversible,
    Sequence,
)
from typing import (
    Literal,
    TypeVar,
    overload,
)

from pytensor.graph.basic import Apply, Constant, Node, Variable


T = TypeVar("T", bound=Node)
NodeAndChildren = tuple[T, Iterable[T] | None]


@overload
def walk(
    nodes: Iterable[T],
    expand: Callable[[T], Iterable[T] | None],
    bfs: bool = True,
    return_children: Literal[False] = False,
) -> Generator[T, None, None]: ...


@overload
def walk(
    nodes: Iterable[T],
    expand: Callable[[T], Iterable[T] | None],
    bfs: bool,
    return_children: Literal[True],
) -> Generator[NodeAndChildren, None, None]: ...


def walk(
    nodes: Iterable[T],
    expand: Callable[[T], Iterable[T] | None],
    bfs: bool = True,
    return_children: bool = False,
) -> Generator[T | NodeAndChildren, None, None]:
    r"""Walk through a graph, either breadth- or depth-first.

    Parameters
    ----------
    nodes
        The nodes from which to start walking.
    expand
        A callable that is applied to each node in `nodes`, the results of
        which are either new nodes to visit or ``None``.
    bfs
        If ``True``, breath first search is used; otherwise, depth first
        search.
    return_children
        If ``True``, each output node will be accompanied by the output of
        `expand` (i.e. the corresponding child nodes).

    Notes
    -----
    A node will appear at most once in the return value, even if it
    appears multiple times in the `nodes` parameter.

    """

    rval_set: set[T] = set()
    nodes = deque(nodes)
    nodes_pop: Callable[[], T] = nodes.popleft if bfs else nodes.pop
    node: T
    new_nodes: Iterable[T] | None
    try:
        if return_children:
            while True:
                node = nodes_pop()
                if node not in rval_set:
                    new_nodes = expand(node)
                    yield node, new_nodes
                    rval_set.add(node)
                    if new_nodes:
                        nodes.extend(new_nodes)
        else:
            while True:
                node = nodes_pop()
                if node not in rval_set:
                    yield node
                    rval_set.add(node)
                    new_nodes = expand(node)
                    if new_nodes:
                        nodes.extend(new_nodes)
    except IndexError:
        return None


def ancestors(
    graphs: Iterable[Variable],
    blockers: Iterable[Variable] | None = None,
) -> Generator[Variable, None, None]:
    r"""Return the variables that contribute to those in given graphs (inclusive), stopping at blockers.

    Parameters
    ----------
    graphs : list of `Variable` instances
        Output `Variable` instances from which to search backward through
        owners.
    blockers : list of `Variable` instances
        A collection of `Variable`\s that, when found, prevent the graph search
        from preceding from that point.

    Yields
    ------
    `Variable`\s
        All ancestor variables, in the order found by a right-recursive depth-first search
        started at the variables in `graphs`.
    """

    seen = set()
    queue = list(graphs)
    try:
        if blockers:
            blockers = frozenset(blockers)
            while True:
                if (var := queue.pop()) not in seen:
                    yield var
                    seen.add(var)
                    if var not in blockers and (apply := var.owner) is not None:
                        queue.extend(apply.inputs)
        else:
            while True:
                if (var := queue.pop()) not in seen:
                    yield var
                    seen.add(var)
                    if (apply := var.owner) is not None:
                        queue.extend(apply.inputs)
    except IndexError:
        return


variable_ancestors = ancestors


def apply_ancestors(
    graphs: Iterable[Variable],
    blockers: Iterable[Variable] | None = None,
) -> Generator[Apply, None, None]:
    """Return the Apply nodes that contribute to those in given graphs (inclusive)."""
    seen = {None}  # This filters out Variables without an owner
    for var in ancestors(graphs, blockers):
        # For multi-output nodes, we'll see multiple variables
        # but we should only yield the Apply once
        if (apply := var.owner) not in seen:
            yield apply
            seen.add(apply)
    return


def graph_inputs(
    graphs: Iterable[Variable], blockers: Iterable[Variable] | None = None
) -> Generator[Variable, None, None]:
    r"""Return the inputs required to compute the given Variables.

    Parameters
    ----------
    graphs : list of `Variable` instances
        Output `Variable` instances from which to search backward through
        owners.
    blockers : list of `Variable` instances
        A collection of `Variable`\s that, when found, prevent the graph search
        from preceding from that point.

    Yields
    ------
        Input nodes with no owner, in the order found by a breath first search started at the nodes in `graphs`.

    """
    yield from (var for var in ancestors(graphs, blockers) if var.owner is None)


def explicit_graph_inputs(
    graph: Variable | Iterable[Variable],
) -> Generator[Variable, None, None]:
    """
    Get the root variables needed as inputs to a function that computes `graph`

    Parameters
    ----------
    graph : TensorVariable
        Output `Variable` instances for which to search backward through
        owners.

    Returns
    -------
    iterable
        Generator of root Variables (without owner) needed to compile a function that evaluates `graphs`.

    Examples
    --------

    .. code-block:: python

        import pytensor
        import pytensor.tensor as pt
        from pytensor.graph.traversal import explicit_graph_inputs

        x = pt.vector("x")
        y = pt.constant(2)
        z = pt.mul(x * y)

        inputs = list(explicit_graph_inputs(z))
        f = pytensor.function(inputs, z)
        eval = f([1, 2, 3])

        print(eval)
        # [2. 4. 6.]
    """
    from pytensor.compile.sharedvalue import SharedVariable

    if isinstance(graph, Variable):
        graph = (graph,)

    return (
        var
        for var in ancestors(graph)
        if var.owner is None and not isinstance(var, Constant | SharedVariable)
    )


def vars_between(
    ins: Iterable[Variable], outs: Iterable[Variable]
) -> Generator[Variable, None, None]:
    r"""Extract the `Variable`\s within the sub-graph between input and output nodes.

    Notes
    -----
    This function is like ancestors(outs, blockers=ins),
    except it can also yield disconnected output variables from multi-output apply nodes.

    Parameters
    ----------
    ins
        Input `Variable`\s.
    outs
        Output `Variable`\s.

    Yields
    ------
    The `Variable`\s that are involved in the subgraph that lies
    between `ins` and `outs`. This includes `ins`, `outs`,
    ``orphans_between(ins, outs)`` and all values of all intermediary steps from
    `ins` to `outs`.

    """

    def expand(var: Variable, ins=frozenset(ins)) -> Iterable[Variable] | None:
        if var.owner is not None and var not in ins:
            return (*var.owner.inputs, *var.owner.outputs)
        return None

    # With bfs = False, it iterates similarly to ancestors
    yield from walk(outs, expand, bfs=False)


def orphans_between(
    ins: Iterable[Variable], outs: Iterable[Variable]
) -> Generator[Variable, None, None]:
    r"""Extract the root `Variable`\s not within the sub-graph between input and output nodes.

    Parameters
    ----------
    ins : list
        Input `Variable`\s.
    outs : list
        Output `Variable`\s.

    Yields
    -------
    Variable
        The `Variable`\s upon which one or more `Variable`\s in `outs`
        depend, but are neither in `ins` nor in the sub-graph that lies between
        them.

    Examples
    --------
    >>> from pytensor.graph.traversal import orphans_between
    >>> from pytensor.tensor import scalars
    >>> x, y = scalars("xy")
    >>> list(orphans_between([x], [(x + y)]))
    [y]

    """
    ins = frozenset(ins)
    yield from (
        var
        for var in vars_between(ins, outs)
        if ((var.owner is None) and (var not in ins))
    )


def applys_between(
    ins: Iterable[Variable], outs: Iterable[Variable]
) -> Generator[Apply, None, None]:
    r"""Extract the `Apply`\s contained within the sub-graph between given input and output variables.

    Notes
    -----
    This is identical to apply_ancestors(outs, blockers=ins)

    Parameters
    ----------
    ins : list
        Input `Variable`\s.
    outs : list
        Output `Variable`\s.

    Yields
    ------
    The `Apply`\s that are contained within the sub-graph that lies
    between `ins` and `outs`, including the owners of the `Variable`\s in
    `outs` and intermediary `Apply`\s between `ins` and `outs`, but not the
    owners of the `Variable`\s in `ins`.

    """
    return apply_ancestors(outs, blockers=ins)


def apply_depends_on(apply: Apply, depends_on: Apply | Iterable[Apply]) -> bool:
    """Determine if any `depends_on` is in the graph given by ``apply``.

    Parameters
    ----------
    apply : Apply
        The Apply node to check.
    depends_on : Union[Apply, Collection[Apply]]
        Apply nodes to check dependency on

    Returns
    -------
    bool

    """
    if isinstance(depends_on, Apply):
        depends_on = frozenset((depends_on,))
    else:
        depends_on = frozenset(depends_on)
    return (apply in depends_on) or any(
        apply in depends_on for apply in apply_ancestors(apply.inputs)
    )


def variable_depends_on(
    variable: Variable, depends_on: Variable | Iterable[Variable]
) -> bool:
    """Determine if any `depends_on` is in the graph given by ``variable``.

    Notes
    -----
    The interpretation of dependency is done at a variable level.
    A variable may depend on some output variables from a multi-output apply node but not others.


    Parameters
    ----------
    variable: Variable
        T to check
    depends_on: Iterable[Variable]
        Nodes to check dependency on

    Returns
    -------
    bool
    """
    if isinstance(depends_on, Variable):
        depends_on_set = frozenset((depends_on,))
    else:
        depends_on_set = frozenset(depends_on)
    return any(var in depends_on_set for var in variable_ancestors([variable]))


def truncated_graph_inputs(
    outputs: Sequence[Variable],
    ancestors_to_include: Iterable[Variable] | None = None,
) -> list[Variable]:
    """Get the truncate graph inputs.

    Unlike :func:`graph_inputs` this function will return
    the closest variables to outputs that do not depend on
    ``ancestors_to_include``. So given all the returned
    variables provided there is no missing variable to
    compute the output and all variables are independent
    from each other.

    Parameters
    ----------
    outputs : Iterable[Variable]
        Variable to get conditions for
    ancestors_to_include : Optional[Iterable[Variable]]
        Additional ancestors to assume, by default None

    Returns
    -------
    List[Variable]
        Variables required to compute ``outputs``

    Examples
    --------
    The returned variables marked in (parenthesis), ancestors variables are ``c``, output variables are ``o``

    * No ancestors to include

    .. code-block::

        n - n - (o)

    * One ancestors to include

    .. code-block::

        n - (c) - o

    * Two ancestors to include where on depends on another, both returned

    .. code-block::

        (c) - (c) - o

    * Additional variables are present

    .. code-block::

           (c) - n - o
        n - (n) -'

    * Disconnected ancestors to include not returned

    .. code-block::

        (c) - n - o
         c

    * Disconnected output is present and returned

    .. code-block::

        (c) - (c) - o
        (o)

    * ancestors to include that include itself adds itself

    .. code-block::

        n - (c) - (o/c)

    """
    truncated_inputs: list[Variable] = list()
    seen: set[Variable] = set()

    # simple case, no additional ancestors to include
    if not ancestors_to_include:
        # just filter out unique variables
        for variable in outputs:
            if variable not in seen:
                seen.add(variable)
                truncated_inputs.append(variable)
        return truncated_inputs

    # blockers have known independent variables and ancestors to include
    blockers: set[Variable] = set(ancestors_to_include)
    # enforce O(1) check for variable in ancestors to include
    ancestors_to_include = blockers.copy()
    candidates = list(outputs)
    try:
        while True:
            if (variable := candidates.pop()) not in seen:
                seen.add(variable)
                # check if the variable is independent, never go above blockers;
                # blockers are independent variables and ancestors to include
                if variable in ancestors_to_include:
                    # ancestors to include that are present in the graph (not disconnected)
                    # should be added to truncated_inputs
                    truncated_inputs.append(variable)
                    # if the ancestors to include is still dependent on other ancestors we need to go above,
                    # FIXME: This seems wrong? The other ancestors above are either redundant given this variable,
                    #  or another path leads to them and the special casing isn't needed
                    #  It seems the only reason we are expanding on these inputs is to find other ancestors_to_include
                    #  (instead of treating them as disconnected), but this may yet cause other unrelated variables
                    #  to become "independent" in the process
                    if variable_depends_on(variable, ancestors_to_include - {variable}):
                        # owner can never be None for a dependent variable
                        candidates.extend(
                            n for n in variable.owner.inputs if n not in seen
                        )
                else:
                    # A regular variable to check
                    # if we've found an independent variable and it is not in blockers so far
                    # it is a new independent variable not present in ancestors to include
                    if variable_depends_on(variable, blockers):
                        # If it's not an independent variable, inputs become candidates
                        candidates.extend(variable.owner.inputs)
                    else:
                        # otherwise it's a truncated input itself
                        truncated_inputs.append(variable)
                    # all regular variables fall to blockers
                    # 1. it is dependent - we already expanded on the inputs, nothing to do if we find it again
                    # 2. it is independent - this is a truncated input, search for other nodes can stop here
                    blockers.add(variable)
    except IndexError:  # pop from an empty list
        pass

    return truncated_inputs


def walk_toposort(
    graphs: Iterable[T],
    deps: Callable[[T], Iterable[T] | None],
) -> Generator[T, None, None]:
    """Perform a topological sort of all nodes starting from a given node.

    Parameters
    ----------
    graphs:
        An iterable of nodes from which to start the topological sort.
    deps : callable
        A Python function that takes a node as input and returns its dependence.

    Notes
    -----

    ``deps(i)`` should behave like a pure function (no funny business with internal state).

    The order of the return value list is determined by the order of nodes
    returned by the `deps` function.
    """

    # Cache the dependencies (ancestors) as we iterate over the nodes with the deps function
    deps_cache: dict[T, list[T]] = {}

    def compute_deps_cache(obj, deps_cache=deps_cache):
        if obj in deps_cache:
            return deps_cache[obj]
        d = deps_cache[obj] = deps(obj) or []
        return d

    clients: dict[T, list[T]] = {}
    sources: deque[T] = deque()
    total_nodes = 0
    for node, children in walk(
        graphs, compute_deps_cache, bfs=False, return_children=True
    ):
        total_nodes += 1
        # Mypy doesn't know that toposort will not return `None` because of our `or []` in the `compute_deps_cache`
        for child in children:  # type: ignore
            clients.setdefault(child, []).append(node)
        if not deps_cache[node]:
            # Add nodes without dependencies to the stack
            sources.append(node)

    rset: set[T] = set()
    try:
        while True:
            if (node := sources.popleft()) not in rset:
                yield node
                total_nodes -= 1
                rset.add(node)
                # Iterate over each client node (that is, it depends on the current node)
                for client in clients.get(node, []):
                    # Remove itself from the dependent (ancestor) list of each client
                    d = deps_cache[client] = [
                        a for a in deps_cache[client] if a is not node
                    ]
                    if not d:
                        # If there are no dependencies left to visit for this node, add it to the stack
                        sources.append(client)
    except IndexError:
        pass

    if total_nodes != 0:
        raise ValueError("graph contains cycles")


def general_toposort(
    outputs: Iterable[T],
    deps: Callable[[T], Iterable[T] | None],
    compute_deps_cache: Callable[[T], Iterable[T] | None] | None = None,
    deps_cache: dict[T, list[T]] | None = None,
    clients: dict[T, list[T]] | None = None,
) -> list[T]:
    """Perform a topological sort of all nodes starting from a given node.

    Parameters
    ----------
    deps : callable
        A Python function that takes a node as input and returns its dependence.
    compute_deps_cache : optional
        If provided, `deps_cache` should also be provided. This is a function like
        `deps`, but that also caches its results in a ``dict`` passed as `deps_cache`.

    Notes
    -----
    This is a simple wrapper around `walk_toposort` for backwards compatibility

    ``deps(i)`` should behave like a pure function (no funny business with
    internal state).

    The order of the return value list is determined by the order of nodes
    returned by the `deps` function.
    """
    # TODO: Deprecate me later
    if compute_deps_cache is not None:
        raise ValueError("compute_deps_cache is no longer supported")
    if deps_cache is not None:
        raise ValueError("deps_cache is no longer supported")
    if clients is not None:
        raise ValueError("clients is no longer supported")
    return list(walk_toposort(outputs, deps))


def toposort(
    graphs: Iterable[Variable],
    blockers: Iterable[Variable] | None = None,
) -> Generator[Apply, None, None]:
    """Topologically sort of Apply nodes between graphs (outputs) and blockers (inputs).

    This is a streamlined version of `io_toposort_generator` when no additional ordering
    constraints are needed.
    """

    # We can put blocker variables in computed, as we only return apply nodes
    computed = set(blockers or ())
    todo = list(graphs)
    try:
        while True:
            if (cur := todo.pop()) not in computed and (apply := cur.owner) is not None:
                uncomputed_inputs = tuple(
                    i
                    for i in apply.inputs
                    if (i not in computed and i.owner is not None)
                )
                if not uncomputed_inputs:
                    yield apply
                    computed.update(apply.outputs)
                else:
                    todo.append(cur)
                    todo.extend(uncomputed_inputs)
    except IndexError:  # queue is empty
        return


def toposort_with_orderings(
    graphs: Iterable[Variable],
    *,
    blockers: Iterable[Variable] | None = None,
    orderings: dict[Apply, list[Apply]] | None = None,
) -> Generator[Apply, None, None]:
    """Perform topological of nodes between blocker (input) and graphs (output) variables with arbitrary extra orderings

    Extra orderings can be used to force sorting of variables that are not naturally related in the graph.
    This can be used by inplace optimizations to ensure a variable is only destroyed after all other uses.
    Those other uses show up as dependencies of the destroying node, in the orderings dictionary.


    Parameters
    ----------
    graphs : list or tuple of Variable instances
        Graph inputs.
    outputs : list or tuple of Apply instances
        Graph outputs.
    orderings : dict
        Keys are `Apply` or `Variable` instances, values are lists of `Apply` or `Variable` instances.

    """
    if not orderings:
        # Faster branch
        yield from toposort(graphs, blockers=blockers)

    else:
        # the inputs are used to decide where to stop expanding
        if blockers:

            def compute_deps(obj, blocker_set=frozenset(blockers), orderings=orderings):
                if obj in blocker_set:
                    return None
                if isinstance(obj, Apply):
                    return [*obj.inputs, *orderings.get(obj, [])]
                else:
                    if (apply := obj.owner) is not None:
                        return [apply, *orderings.get(apply, [])]
                    else:
                        return orderings.get(obj, [])
        else:
            # mypy doesn't like conditional functions with different signatures,
            # but passing the globals as optional is faster
            def compute_deps(obj, orderings=orderings):  # type: ignore[misc]
                if isinstance(obj, Apply):
                    return [*obj.inputs, *orderings.get(obj, [])]
                else:
                    if (apply := obj.owner) is not None:
                        return [apply, *orderings.get(apply, [])]
                    else:
                        return orderings.get(obj, [])

        yield from (
            apply
            for apply in walk_toposort(graphs, deps=compute_deps)
            # mypy doesn't understand that our generator will return both Apply and Variables
            if isinstance(apply, Apply)  # type: ignore
        )


def io_toposort(
    inputs: Iterable[Variable],
    outputs: Reversible[Variable],
    orderings: dict[Apply, list[Apply]] | None = None,
    clients: dict[Variable, list[Variable]] | None = None,
) -> list[Apply]:
    """Perform topological of nodes between input and output variables.

    Notes
    -----
    This is just a wrapper around `toposort_with_extra_orderings` for backwards compatibility

    Parameters
    ----------
    inputs : list or tuple of Variable instances
        Graph inputs.
    outputs : list or tuple of Apply instances
        Graph outputs.
    orderings : dict
        Keys are `Apply` instances, values are lists of `Apply` instances.
    """
    # TODO: Deprecate me later
    if clients is not None:
        raise ValueError("clients is no longer supported")

    return list(toposort_with_orderings(outputs, blockers=inputs, orderings=orderings))


def get_var_by_name(
    graphs: Iterable[Variable], target_var_id: str
) -> tuple[Variable, ...]:
    r"""Get variables in a graph using their names.

    Parameters
    ----------
    graphs:
        The graph, or graphs, to search.
    target_var_id:
        The name to match against either ``Variable.name`` or
        ``Variable.auto_name``.

    Returns
    -------
    A ``tuple`` containing all the `Variable`\s that match `target_var_id`.

    """
    from pytensor.graph.op import HasInnerGraph

    def expand(r: Variable) -> list[Variable] | None:
        if (apply := r.owner) is not None:
            if isinstance(apply.op, HasInnerGraph):
                return [*apply.inputs, *apply.op.inner_outputs]
            else:
                # Mypy doesn't know these will never be None
                return apply.inputs  # type: ignore
        else:
            return None

    return tuple(
        var
        for var in walk(graphs, expand)
        if (target_var_id == var.name or target_var_id == var.auto_name)
    )
