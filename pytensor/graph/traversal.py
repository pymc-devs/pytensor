from collections import deque
from collections.abc import (
    Callable,
    Collection,
    Generator,
    Iterable,
    Reversible,
    Sequence,
)
from typing import (
    TypeVar,
    cast,
)

from pytensor.graph.basic import Apply, Constant, Node, Variable
from pytensor.misc.ordered_set import OrderedSet


T = TypeVar("T", bound=Node)
NodeAndChildren = tuple[T, Iterable[T] | None]


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
    try:
        if return_children:
            while True:
                node: T = nodes_pop()
                if node not in rval_set:
                    new_nodes: Iterable[T] | None = expand(node)
                    yield node, new_nodes
                    rval_set.add(node)
                    if new_nodes:
                        nodes.extend(new_nodes)
        else:
            while True:
                node: T = nodes_pop()
                if node not in rval_set:
                    yield node
                    rval_set.add(node)
                    new_nodes: Iterable[T] | None = expand(node)
                    if new_nodes:
                        nodes.extend(new_nodes)
    except IndexError:
        return None


def ancestors(
    graphs: Iterable[Variable], blockers: Collection[Variable] | None = None
) -> Generator[Variable, None, None]:
    r"""Return the variables that contribute to those in given graphs (inclusive).

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
        All input variables, in the order found by a breath-first search started at the variables in `graphs`.

    """

    blockers = set() if blockers is None else set(blockers)
    seen = set()
    queue = list(graphs)
    try:
        while True:
            if (var := queue.pop()) not in seen:
                yield var
                seen.add(var)
                if var not in blockers and (node := var.owner) is not None:
                    queue.extend(reversed(node.inputs))
    except IndexError:
        return


variable_ancestors = ancestors


def apply_ancestors(graphs: Iterable[Apply], blockers: Collection[Apply] = ()):
    seen = {None, *blockers}
    queue = list(graphs)
    try:
        while True:
            if (node := queue.pop()) not in seen:
                yield node
                seen.add(node)
                queue.extend(i.owner for i in reversed(node.inputs))
    except IndexError:
        return


def graph_inputs(
    graphs: Iterable[Variable], blockers: Collection[Variable] | None = None
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
    yield from (r for r in ancestors(graphs, blockers) if r.owner is None)


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
        graph = [graph]

    return (
        v
        for v in ancestors(graph)
        if v.owner is None and not isinstance(v, Constant | SharedVariable)
    )


def vars_between(
    ins: Iterable[Variable], outs: Iterable[Variable]
) -> Generator[Variable, None, None]:
    r"""Extract the `Variable`\s within the sub-graph between input and output nodes.

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

    ins = set(ins)

    def expand(r: Variable) -> Iterable[Variable] | None:
        if r.owner and r not in ins:
            return reversed(r.owner.inputs + r.owner.outputs)
        return None

    yield from cast(Generator[Variable, None, None], walk(outs, expand))


def orphans_between(
    ins: Collection[Variable], outs: Iterable[Variable]
) -> Generator[Variable, None, None]:
    r"""Extract the `Variable`\s not within the sub-graph between input and output nodes.

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
    yield from (r for r in vars_between(ins, outs) if r.owner is None and r not in ins)


def applys_between(
    ins: Collection[Variable], outs: Iterable[Variable]
) -> Generator[Apply, None, None]:
    r"""Extract the `Apply`\s contained within the sub-graph between given input and output variables.

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
    yield from (
        r.owner for r in vars_between(ins, outs) if r not in ins and r.owner is not None
    )


def apply_depends_on(apply: Apply, depends_on: Apply | Collection[Apply]) -> bool:
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
    if not isinstance(depends_on, Collection):
        depends_on = {depends_on}
    else:
        depends_on = set(depends_on)
    return any(ancestor in depends_on for ancestor in apply_ancestors([apply]))


def variable_depends_on(
    variable: Variable, depends_on: Variable | Collection[Variable]
) -> bool:
    """Determine if any `depends_on` is in the graph given by ``variable``.
    Parameters
    ----------
    variable: Variable
        Node to check
    depends_on: Collection[Variable]
        Nodes to check dependency on

    Returns
    -------
    bool
    """
    if not isinstance(depends_on, Collection):
        depends_on = {depends_on}
    else:
        depends_on = set(depends_on)
    return any(interim in depends_on for interim in ancestors([variable]))


def truncated_graph_inputs(
    outputs: Sequence[Variable],
    ancestors_to_include: Collection[Variable] | None = None,
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
    outputs : Collection[Variable]
        Variable to get conditions for
    ancestors_to_include : Optional[Collection[Variable]]
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
    # simple case, no additional ancestors to include
    truncated_inputs: list[Variable] = list()
    # blockers have known independent variables and ancestors to include
    candidates = list(outputs)
    if not ancestors_to_include:  # None or empty
        # just filter out unique variables
        for variable in candidates:
            if variable not in truncated_inputs:
                truncated_inputs.append(variable)
        # no more actions are needed
        return truncated_inputs

    blockers: set[Variable] = set(ancestors_to_include)
    # variables that go here are under check already, do not repeat the loop for them
    seen: set[Variable] = set()
    # enforce O(1) check for variable in ancestors to include
    ancestors_to_include = blockers.copy()

    while candidates:
        # on any new candidate
        variable = candidates.pop()
        # we've looked into this variable already
        if variable in seen:
            continue
        # check if the variable is independent, never go above blockers;
        # blockers are independent variables and ancestors to include
        elif variable in ancestors_to_include:
            # The case where variable is in ancestors to include so we check if it depends on others
            # it should be removed from the blockers to check against the rest
            dependent = variable_depends_on(variable, ancestors_to_include - {variable})
            # ancestors to include that are present in the graph (not disconnected)
            # should be added to truncated_inputs
            truncated_inputs.append(variable)
            if dependent:
                # if the ancestors to include is still dependent we need to go above, the search is not yet finished
                # owner can never be None for a dependent variable
                candidates.extend(n for n in variable.owner.inputs if n not in seen)
        else:
            # A regular variable to check
            dependent = variable_depends_on(variable, blockers)
            # all regular variables fall to blockers
            # 1. it is dependent - further search irrelevant
            # 2. it is independent - the search variable is inside the closure
            blockers.add(variable)
            # if we've found an independent variable and it is not in blockers so far
            # it is a new independent variable not present in ancestors to include
            if dependent:
                # populate search if it's not an independent variable
                # owner can never be None for a dependent variable
                candidates.extend(n for n in variable.owner.inputs if n not in seen)
            else:
                # otherwise, do not search beyond
                truncated_inputs.append(variable)
        # add variable to seen, no point in checking it once more
        seen.add(variable)
    return truncated_inputs


def general_toposort_generator(
    graphs: Iterable[Variable | Apply],
    deps: Callable[[T], OrderedSet | list[T]] | None,
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
    deps_cache = {}

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
        for child in children:
            clients.setdefault(child, []).append(node)
        if not deps_cache[node]:
            # Add nodes without dependencies to the stack
            sources.append(node)

    rset: set[T] = set()
    try:
        while True:
            node: T = sources.popleft()
            if node not in rset:
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
    deps: Callable[[T], OrderedSet | list[T]] | None,
    compute_deps_cache: Callable[[T], OrderedSet | list[T] | None] | None = None,
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

    ``deps(i)`` should behave like a pure function (no funny business with
    internal state).

    The order of the return value list is determined by the order of nodes
    returned by the `deps` function.
    """
    if compute_deps_cache is not None:
        raise ValueError("compute_deps_cache is no longer supported")
    if deps_cache is not None:
        raise ValueError("deps_cache is no longer supported")
    if clients is not None:
        raise ValueError("clients is no longer supported")
    return list(general_toposort_generator(outputs, deps))


def apply_toposort(
    graphs: Iterable[Apply],
    blockers: Iterable[Apply] = (),
) -> Generator[Apply]:
    """Topologically sort a sequence of Apply nodes.

    Faster than variable_toposort because we don't need to worry about specific variable edges from blockers
    """
    computed = {None, *blockers}
    todo = list(graphs)
    while todo:
        cur = todo[-1]
        if cur in computed:
            todo.pop()
            continue
        # Since computed includes None we don't need to filter it in this check
        if all(i.owner in computed for i in cur.inputs):
            computed.add(cur)
            yield todo.pop()
        else:
            todo.extend(i.owner for i in cur.inputs if i.owner is not None)


def variable_toposort(
    graphs: Iterable[Variable],
    blockers: Iterable[Variable] = (),
    orderings: dict[Apply, list[Apply]] | None = None,
) -> Generator[Apply, None, None]:
    """Topologically sort a sequence of Apply nodes, given a set of grapph and blocker variables.

    Allows to specify additional ordering constraints between Apply or Variable nodes using the `orderings` parameter.

    When there are no 'orderings' specified, this function is similar to `apply_toposort` except it can handle
    the case where multi-output nodes are partially blocked.
    """

    if not orderings:
        computed = set(blockers)
        todo = [o.owner for o in graphs if o.owner is not None]
        while todo:
            cur = todo[-1]
            # It's faster to short circuit on the first output, as most nodes will have all edges non-computed
            # Starting the `all` iterator has a non-negligeable cost
            if cur.outputs[0] in computed and all(
                out in computed for out in cur.outputs[1:]
            ):
                todo.pop()
                continue
            if all(i in computed or i.owner is None for i in cur.inputs):
                computed.update(cur.outputs)
                yield todo.pop()
            else:
                todo.extend(
                    i.owner
                    for i in cur.inputs
                    if ((i.owner is not None) and (i not in computed))
                )
    else:
        # the inputs are used to decide where to stop expanding
        def compute_deps(obj, input_set=set(blockers), orderings=orderings):
            if obj in input_set:
                return []

            if isinstance(obj, Apply):
                return [*obj.inputs, *orderings.get(obj, [])]
            else:
                node = obj.owner
                if node is None:
                    return orderings.get(obj, [])
                else:
                    return [node, *orderings.get(node, [])]

        yield from (
            node
            for node in general_toposort_generator(graphs, deps=compute_deps)
            if isinstance(node, Apply)
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
    If sorting from root or single-output node variables, without orderings or clients,
    it's slightly faster to use `list(apply_toposort((o.owner for o in outputs)))` instead,
    as the individual variables can be ignored

    Parameters
    ----------
    inputs : list or tuple of Variable instances
        Graph inputs.
    outputs : list or tuple of Apply instances
        Graph outputs.
    orderings : dict
        Keys are `Apply` instances, values are lists of `Apply` instances.
    """
    if clients is not None:
        raise ValueError("clients is no longer supported")
    return list(
        variable_toposort(
            outputs,
            blockers=inputs if inputs is not None else (),
            orderings=orderings,
        )
    )


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

    def expand(r) -> list[Variable] | None:
        if not r.owner:
            return None

        res = list(r.owner.inputs)

        if isinstance(r.owner.op, HasInnerGraph):
            res.extend(r.owner.op.inner_outputs)

        return res

    results: tuple[Variable, ...] = ()
    for var in walk(graphs, expand, False):
        var = cast(Variable, var)
        if target_var_id == var.name or target_var_id == var.auto_name:
            results += (var,)

    return results
