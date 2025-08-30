from collections import deque
from collections.abc import (
    Callable,
    Collection,
    Generator,
    Iterable,
    Iterator,
    Reversible,
    Sequence,
)
from typing import (
    TypeVar,
    cast,
    overload,
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
    hash_fn: Callable[[T], int] = id,
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
    hash_fn
        The function used to produce hashes of the elements in `nodes`.
        The default is ``id``.

    Notes
    -----
    A node will appear at most once in the return value, even if it
    appears multiple times in the `nodes` parameter.

    """

    nodes = deque(nodes)

    rval_set: set[int] = set()

    nodes_pop: Callable[[], T]
    if bfs:
        nodes_pop = nodes.popleft
    else:
        nodes_pop = nodes.pop

    while nodes:
        node: T = nodes_pop()

        node_hash: int = hash_fn(node)

        if node_hash not in rval_set:
            rval_set.add(node_hash)

            new_nodes: Iterable[T] | None = expand(node)

            if return_children:
                yield node, new_nodes
            else:
                yield node

            if new_nodes:
                nodes.extend(new_nodes)


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
        All input nodes, in the order found by a left-recursive depth-first
        search started at the nodes in `graphs`.

    """

    def expand(r: Variable) -> Iterator[Variable] | None:
        if r.owner and (not blockers or r not in blockers):
            return reversed(r.owner.inputs)
        return None

    yield from cast(Generator[Variable, None, None], walk(graphs, expand, False))


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
        Input nodes with no owner, in the order found by a left-recursive
        depth-first search started at the nodes in `graphs`.

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
        for v in graph_inputs(graph)
        if isinstance(v, Variable) and not isinstance(v, Constant | SharedVariable)
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
    computed = set()
    todo = [apply]
    if not isinstance(depends_on, Collection):
        depends_on = {depends_on}
    else:
        depends_on = set(depends_on)
    while todo:
        cur = todo.pop()
        if cur.outputs[0] in computed:
            continue
        if all(i in computed or i.owner is None for i in cur.inputs):
            computed.update(cur.outputs)
            if cur in depends_on:
                return True
        else:
            todo.append(cur)
            todo.extend(i.owner for i in cur.inputs if i.owner)
    return False


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


@overload
def general_toposort(
    outputs: Iterable[T],
    deps: Callable[[T], OrderedSet | list[T]],
    compute_deps_cache: None,
    deps_cache: None,
    clients: dict[T, list[T]] | None,
) -> list[T]: ...


@overload
def general_toposort(
    outputs: Iterable[T],
    deps: None,
    compute_deps_cache: Callable[[T], OrderedSet | list[T] | None],
    deps_cache: dict[T, list[T]] | None,
    clients: dict[T, list[T]] | None,
) -> list[T]: ...


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
    deps_cache : dict
        A ``dict`` mapping nodes to their children.  This is populated by
        `compute_deps_cache`.
    clients : dict
        If a ``dict`` is passed, it will be filled with a mapping of
        nodes-to-clients for each node in the subgraph.

    Notes
    -----

    ``deps(i)`` should behave like a pure function (no funny business with
    internal state).

    ``deps(i)`` will be cached by this function (to be fast).

    The order of the return value list is determined by the order of nodes
    returned by the `deps` function.

    The second option removes a Python function call, and allows for more
    specialized code, so it can be faster.

    """
    if compute_deps_cache is None:
        if deps_cache is None:
            deps_cache = {}

        def _compute_deps_cache_(io):
            if io not in deps_cache:
                d = deps(io)

                if d:
                    if not isinstance(d, list | OrderedSet):
                        raise TypeError(
                            "Non-deterministic collections found; make"
                            " toposort non-deterministic."
                        )
                    deps_cache[io] = list(d)
                else:
                    deps_cache[io] = None

                return d
            else:
                return deps_cache[io]

        _compute_deps_cache = _compute_deps_cache_

    else:
        _compute_deps_cache = compute_deps_cache

    if deps_cache is None:
        raise ValueError("deps_cache cannot be None")

    search_res: list[NodeAndChildren] = cast(
        list[NodeAndChildren],
        list(walk(outputs, _compute_deps_cache, bfs=False, return_children=True)),
    )

    _clients: dict[T, list[T]] = {}
    sources: deque[T] = deque()
    search_res_len = len(search_res)
    for snode, children in search_res:
        if children:
            for child in children:
                _clients.setdefault(child, []).append(snode)
        if not deps_cache.get(snode):
            sources.append(snode)

    if clients is not None:
        clients.update(_clients)

    rset: set[T] = set()
    rlist: list[T] = []
    while sources:
        node: T = sources.popleft()
        if node not in rset:
            rlist.append(node)
            rset.add(node)
            for client in _clients.get(node, []):
                d = [a for a in deps_cache[client] if a is not node]
                deps_cache[client] = d
                if not d:
                    sources.append(client)

    if len(rlist) != search_res_len:
        raise ValueError("graph contains cycles")

    return rlist


def io_toposort(
    inputs: Iterable[Variable],
    outputs: Reversible[Variable],
    orderings: dict[Apply, list[Apply]] | None = None,
    clients: dict[Variable, list[Variable]] | None = None,
) -> list[Apply]:
    """Perform topological sort from input and output nodes.

    Parameters
    ----------
    inputs : list or tuple of Variable instances
        Graph inputs.
    outputs : list or tuple of Apply instances
        Graph outputs.
    orderings : dict
        Keys are `Apply` instances, values are lists of `Apply` instances.
    clients : dict
        If provided, it will be filled with mappings of nodes-to-clients for
        each node in the subgraph that is sorted.

    """
    if not orderings and clients is None:  # ordering can be None or empty dict
        # Specialized function that is faster when more then ~10 nodes
        # when no ordering.

        # Do a new stack implementation with the vm algo.
        # This will change the order returned.
        computed = set(inputs)
        todo = [o.owner for o in reversed(outputs) if o.owner]
        order = []
        while todo:
            cur = todo.pop()
            if all(out in computed for out in cur.outputs):
                continue
            if all(i in computed or i.owner is None for i in cur.inputs):
                computed.update(cur.outputs)
                order.append(cur)
            else:
                todo.append(cur)
                todo.extend(
                    i.owner for i in cur.inputs if (i.owner and i not in computed)
                )
        return order

    iset = set(inputs)

    if not orderings:  # ordering can be None or empty dict
        # Specialized function that is faster when no ordering.
        # Also include the cache in the function itself for speed up.

        deps_cache: dict = {}

        def compute_deps_cache(obj):
            if obj in deps_cache:
                return deps_cache[obj]
            rval = []
            if obj not in iset:
                if isinstance(obj, Variable):
                    if obj.owner:
                        rval = [obj.owner]
                elif isinstance(obj, Apply):
                    rval = list(obj.inputs)
                if rval:
                    deps_cache[obj] = list(rval)
                else:
                    deps_cache[obj] = rval
            else:
                deps_cache[obj] = rval
            return rval

        topo = general_toposort(
            outputs,
            deps=None,
            compute_deps_cache=compute_deps_cache,
            deps_cache=deps_cache,
            clients=clients,
        )

    else:
        # the inputs are used only here in the function that decides what
        # 'predecessors' to explore
        def compute_deps(obj):
            rval = []
            if obj not in iset:
                if isinstance(obj, Variable):
                    if obj.owner:
                        rval = [obj.owner]
                elif isinstance(obj, Apply):
                    rval = list(obj.inputs)
                rval.extend(orderings.get(obj, []))
            else:
                assert not orderings.get(obj, None)
            return rval

        topo = general_toposort(
            outputs,
            deps=compute_deps,
            compute_deps_cache=None,
            deps_cache=None,
            clients=clients,
        )
    return [o for o in topo if isinstance(o, Apply)]


def get_var_by_name(
    graphs: Iterable[Variable], target_var_id: str, ids: str = "CHAR"
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
