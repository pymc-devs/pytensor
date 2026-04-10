"""Functions for printing PyTensor graphs."""

import hashlib
import logging
import sys
from abc import ABC, abstractmethod
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from copy import copy
from dataclasses import dataclass
from functools import reduce, singledispatch
from io import StringIO
from pathlib import Path
from typing import Any, Literal, TextIO

import numpy as np
import rich.tree

from pytensor.compile import SharedVariable
from pytensor.compile.debug.profiling import ProfileStats
from pytensor.compile.executor import Function
from pytensor.compile.io import In, Out
from pytensor.configdefaults import config
from pytensor.graph.basic import Apply, Constant, Variable
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.op import HasInnerGraph, Op, StorageMapType
from pytensor.graph.traversal import graph_inputs, toposort
from pytensor.graph.utils import Scratchpad


IDTypesType = Literal["id", "int", "CHAR", "auto", ""]

_logger = logging.getLogger("pytensor.printing")
VALID_ASSOC = {"left", "right", "either"}


def char_from_number(number: int) -> str:
    """Convert a number to a string.

    It renders it in base 26 using capital letters as digits.
    For example: 3·26² + 2·26¹ + 0·26⁰ → "DCA"

    Parameters
    ----------
    number : int
        The number to be converted.

    Returns
    -------
    str
        The converted string.
    """

    base = 26

    remainders = []

    while number != 0:
        number, remainder = number // base, number % base
        remainders.append(remainder)

    if not remainders:
        remainders = [0]

    return "".join(chr(ord("A") + r) for r in remainders[::-1])


@dataclass
class GraphNode:
    """A record of a single node in a `debugprint` / `print_graph` traversal.

    Carries the graph variable, its structural position in the traversal, and
    any render-time context needed to build a label.  Renderers consume a
    stream of these objects produced by `_iter_graph_nodes`.

    Parameters
    ----------
    var
        The `Variable` this record represents.  The single source of truth —
        all graph structure is accessed through it via ``var.owner`` etc.
    is_repeat
        ``True`` if this node's owner `Apply` was already expanded earlier in
        the traversal.  Renderers should show a placeholder (e.g. ``"···"``)
        rather than re-expanding children.
    is_last_child
        ``True`` if this node is the last input among its siblings.  Used by
        the text renderer to choose ``"└─"`` vs ``"├─"``.
    ancestor_is_last
        One ``bool`` per ancestor level (outermost first), recording whether
        that ancestor was itself a last child.  ``False`` → draw a ``" │ "``
        continuation column; ``True`` → draw ``"   "`` (no bar).  Used
        exclusively by the text renderer — the rich renderer ignores it.
    parent_node
        The `Apply` node whose input list contains ``var``.  Needed for
        ``op_information`` annotation lookup.
    inner_to_outer
        Mapping from inner-graph `Variable` s to their corresponding
        outer-graph counterparts, used to print ``"-> [id X]"`` suffixes.
    inner_graph_node
        The `Apply` node that owns the inner graph in which ``var`` lives.
    is_inner_graph_header
        ``True`` for the single header line of an inner-graph op (prints
        only ``op + id``, without type/name/annotation details).
    topo_order
        Toposort of the enclosing `FunctionGraph`, used to print topo index.
    profile
        Profiling stats for the enclosing compiled function.
    storage_map
        VM storage map (only available after a function has been executed
        with ``allow_gc=False``).
    """

    var: Variable
    is_repeat: bool
    is_last_child: bool
    ancestor_is_last: list[bool]
    parent_node: Apply | None
    inner_to_outer: dict[Variable, Variable] | None
    inner_graph_node: Apply | None
    is_inner_graph_header: bool
    topo_order: Sequence[Apply] | None
    profile: "ProfileStats | None"
    storage_map: "StorageMapType | None"

    @property
    def node(self) -> Apply | None:
        """The `Apply` node that produced ``var``, or ``None`` for leaves."""
        return self.var.owner

    @property
    def is_leaf(self) -> bool:
        """``True`` if ``var`` has no owner (graph input or constant)."""
        return self.var.owner is None

    @property
    def output_idx(self) -> str:
        """``".N"`` suffix for multi-output ops, empty string otherwise."""
        node = self.var.owner
        if node is None or len(node.outputs) == 1:
            return ""
        return f".{node.outputs.index(self.var)}"


def _assign_id(
    obj: "Literal['output'] | Apply | Variable",
    used_ids: dict,
    done: dict,
    id_type: IDTypesType,
    var: "Variable",
) -> str:
    """Return (and register) the ``[id X]`` string for *obj*.

    The ``var`` argument is the `Variable` currently being printed; it is
    used for ``id_type="id"`` and ``id_type="auto"``.
    """
    if obj in used_ids:
        return used_ids[obj]
    if obj == "output":
        id_str = "output"
    elif id_type == "id":
        id_str = f"[id {id(var)}]"
    elif id_type == "int":
        id_str = f"[id {len(used_ids)}]"
    elif id_type == "CHAR":
        id_str = f"[id {char_from_number(len(used_ids))}]"
    elif id_type == "auto":
        id_str = f"[id {var.auto_name}]"
    else:
        id_str = ""
    done[obj] = id_str
    used_ids[obj] = id_str
    return id_str


def _build_label(
    gnode: GraphNode,
    done: dict,
    used_ids: dict,
    id_type: IDTypesType,
    print_type: bool,
    print_shape: bool,
    print_destroy_map: bool,
    print_view_map: bool,
    print_op_info: bool,
    op_information: dict,
) -> str:
    """Return the formatted label string for a single `GraphNode`.

    This is a pure function with respect to the graph; its only side-effects
    are updating the shared ``done`` and ``used_ids`` registries.

    Parameters
    ----------
    gnode
        The node to label.
    done
        Shared registry mapping `Apply`/`Variable` objects to their id
        strings, used for repeat-detection.
    used_ids
        Shared registry mapping objects to their ``[id X]`` strings, ensuring
        stable IDs across an entire print call.
    id_type, print_type, print_shape, print_destroy_map, print_view_map,
    print_op_info, op_information
        Same semantics as the identically-named parameters of `debugprint`.

    Returns
    -------
    str
        The formatted label, ready to be written or passed to ``rich.Tree``.
    """
    var = gnode.var

    if print_type:
        type_str = f" <{var.type}>"
    else:
        type_str = ""

    if print_shape and hasattr(var.type, "shape"):
        shape_str = f" shape={str(var.type.shape).replace('None', '?')}"
    else:
        shape_str = ""

    if gnode.is_leaf:
        id_str = _assign_id(var, used_ids, done, id_type, var)
        if id_str:
            id_str = f" {id_str}"

        if gnode.storage_map and var in gnode.storage_map:
            data = f" {gnode.storage_map[var]}"
        else:
            data = ""

        label = f"{var}{id_str}{type_str}{shape_str}{data}"

        if print_op_info and var.owner and var.owner not in op_information:
            op_information.update(op_debug_information(var.owner.op, var.owner))

        if gnode.inner_to_outer is not None and var in gnode.inner_to_outer:
            outer_var = gnode.inner_to_outer[var]
            outer_id_str = _assign_id(
                outer_var.owner if outer_var.owner else outer_var,
                used_ids,
                done,
                id_type,
                outer_var,
            )
            label = f"{label} -> {outer_id_str}"

        # TODO: This entire approach will only print `Op` info for two levels
        # of nesting.
        for node in dict.fromkeys(
            [gnode.inner_graph_node, gnode.parent_node, var.owner]
        ):
            node_info = op_information.get(node)
            if node_info and var in node_info:
                label = f"{label} ({node_info[var]})"

        return label

    # Owned variable — label is built from the Apply node
    node = gnode.node
    assert node is not None

    id_str = _assign_id(node, used_ids, done, id_type, var)
    if id_str:
        id_str = f" {id_str}"

    var_name = getattr(var, "name", "") or ""
    if var_name:
        var_name = f" '{var_name}'"

    destroy_map_str = (
        f" d={node.op.destroy_map}" if print_destroy_map and node.op.destroy_map else ""
    )
    view_map_str = (
        f" v={node.op.view_map}" if print_view_map and node.op.view_map else ""
    )

    topo_str = f" {gnode.topo_order.index(node)}" if gnode.topo_order else ""

    if gnode.storage_map and node.outputs[0] in gnode.storage_map:
        data = f" {gnode.storage_map[node.outputs[0]]}"
    else:
        data = ""

    if gnode.is_inner_graph_header:
        return f"{node.op}{id_str}{destroy_map_str}{view_map_str}{topo_str}"

    label = f"{node.op}{gnode.output_idx}{id_str}{type_str}{shape_str}{var_name}{destroy_map_str}{view_map_str}{topo_str}{data}"

    if print_op_info and node not in op_information:
        op_information.update(op_debug_information(node.op, node))

    node_info = (
        gnode.parent_node and op_information.get(gnode.parent_node)
    ) or op_information.get(node)
    if node_info and var in node_info:
        label = f"{label} ({node_info[var]})"

    return label


def _iter_graph_nodes(
    var: Variable,
    depth: int = -1,
    done: dict | None = None,
    stop_on_name: bool = False,
    inner_graph_ops: list | None = None,
    inner_to_outer: dict | None = None,
    topo_order: Sequence | None = None,
    profile: "ProfileStats | None" = None,
    storage_map: "StorageMapType | None" = None,
    parent_node: Apply | None = None,
    inner_graph_node: Apply | None = None,
    is_inner_graph_header: bool = False,
    ancestor_is_last: list[bool] | None = None,
    is_last_child: bool = True,
    _current_depth: int = 0,
) -> Generator[GraphNode, None, None]:
    """Yield `GraphNode` records for a depth-first pre-order traversal of *var*.

    This generator encapsulates all graph-traversal logic that was previously
    embedded in the recursive `_debugprint` function.  Renderers consume the
    yielded records without needing to understand graph structure.

    Parameters
    ----------
    var
        The root `Variable` to start traversal from.
    depth
        Maximum traversal depth (``-1`` for unlimited).
    done
        Shared dict mapping `Apply` nodes to their id strings.  Nodes already
        present are not re-expanded — a repeat `GraphNode` is yielded instead.
    stop_on_name
        When ``True``, stop recursing into a node's inputs if that node's
        output variable has a non-``None`` name.
    inner_graph_ops
        Accumulator list — inner-graph op variables are appended here as they
        are discovered, for the caller's second-pass inner-graph printing.
    inner_to_outer, topo_order, profile, storage_map, parent_node,
    inner_graph_node, is_inner_graph_header
        Passed through to each yielded `GraphNode` unchanged.
    ancestor_is_last
        Tracks whether each ancestor level was itself a last child.
    is_last_child
        Whether this call is for the last input among its siblings.
    _current_depth
        Internal recursion counter.
    """
    if done is None:
        done = {}
    if inner_graph_ops is None:
        inner_graph_ops = []
    if ancestor_is_last is None:
        ancestor_is_last = []

    if depth != -1 and _current_depth >= depth:
        return

    gnode = GraphNode(
        var=var,
        is_repeat=False,
        is_last_child=is_last_child,
        ancestor_is_last=ancestor_is_last,
        parent_node=parent_node,
        inner_to_outer=inner_to_outer,
        inner_graph_node=inner_graph_node,
        is_inner_graph_header=is_inner_graph_header,
        topo_order=topo_order,
        profile=profile,
        storage_map=storage_map,
    )

    if var.owner is None:
        # Leaf node — input variable or constant
        yield gnode
        return

    node = var.owner
    already_done = node in done

    if already_done:
        gnode.is_repeat = True
        yield gnode
        return

    yield gnode

    if stop_on_name and var.name is not None:
        # Yield the node itself but don't recurse — text renderer adds "···"
        return

    # Mark as visited before recursing to handle DAG diamonds
    done[node] = ""

    child_ancestor = [*ancestor_is_last, is_last_child]

    for in_idx, in_var in enumerate(node.inputs):
        child_is_last = in_idx == len(node.inputs) - 1

        # Collect inner-graph ops for the second pass
        if hasattr(in_var, "owner") and hasattr(in_var.owner, "op"):
            if (
                isinstance(in_var.owner.op, HasInnerGraph)
                or (
                    hasattr(in_var.owner.op, "scalar_op")
                    and isinstance(in_var.owner.op.scalar_op, HasInnerGraph)
                )
            ) and in_var not in inner_graph_ops:
                inner_graph_ops.append(in_var)

        yield from _iter_graph_nodes(
            in_var,
            depth=depth,
            done=done,
            stop_on_name=stop_on_name,
            inner_graph_ops=inner_graph_ops,
            inner_to_outer=inner_to_outer,
            topo_order=topo_order,
            profile=profile,
            storage_map=storage_map,
            parent_node=node,
            inner_graph_node=inner_graph_node,
            is_inner_graph_header=False,
            ancestor_is_last=child_ancestor,
            is_last_child=child_is_last,
            _current_depth=_current_depth + 1,
        )


@singledispatch
def op_debug_information(op: Op, node: Apply) -> dict[Apply, dict[Variable, str]]:
    """Provide extra debug print information based on the type of `Op` and `Apply` node.

    Implementations of this dispatch function should return a ``dict`` keyed by
    the `Apply` node, `node`, associated with the given `op`.  The value
    associated with the `node` is another ``dict`` mapping `Variable` inputs
    and/or outputs of `node` to their debug information.

    The `node` key allows the information in the ``dict``'s values to be
    specific to the given `node`, so that--for instance--the provided debug
    information is only ever printed/associated with a given `Variable`
    input/output when that `Variable` is displayed as an input/output of `node`
    and not in every/any other place where said `Variable` is present in a
    graph.

    """
    return {}


def debugprint(
    graph_like: Variable
    | Apply
    | Function
    | FunctionGraph
    | Sequence[Variable | Apply | Function | FunctionGraph],
    depth: int = -1,
    print_type: bool = False,
    print_shape: bool = False,
    file: Literal["str", "rich"] | TextIO | None = None,
    id_type: IDTypesType = "CHAR",
    stop_on_name: bool = False,
    done: dict[Literal["output"] | Variable | Apply, str] | None = None,
    print_storage: bool = False,
    used_ids: dict[Literal["output"] | Variable | Apply, str] | None = None,
    print_op_info: bool = False,
    print_destroy_map: bool = False,
    print_view_map: bool = False,
    print_memory_map: bool = False,
    print_fgraph_inputs: bool = False,
) -> "str | TextIO | rich.tree.Tree":
    r"""Print a graph as text.

    Each line printed represents a `Variable` in a graph.
    The indentation of lines corresponds to its depth in the symbolic graph.
    The first part of the text identifies whether it is an input or the output
    of some `Apply` node.
    The second part of the text is an identifier of the `Variable`.

    If a `Variable` is encountered multiple times in the depth-first search,
    it is only printed recursively the first time. Later, just the `Variable`
    identifier is printed.

    If an `Apply` node has multiple outputs, then a ``.N`` suffix will be appended
    to the `Apply` node's identifier, indicating to which output a line corresponds.

    Parameters
    ----------
    graph_like
        The object(s) to be printed.
    depth
        Print graph to this depth (``-1`` for unlimited).
    print_type
        If ``True``, print the `Type`\s of each `Variable` in the graph.
    print_shape
        If ``True``, print the shape of each `Variable` in the graph.
    file
        When `file` extends `TextIO`, print to it; when `file` is
        equal to ``"str"``, return a string; when `file` is ``None``, print to
        `sys.stdout`; when `file` is ``"rich"``, return a ``rich.tree.Tree``
        that can be rendered with ``rich.print()``.
    id_type
        Determines the type of identifier used for `Variable`\s:
          - ``"id"``: print the python id value,
          - ``"int"``: print integer character,
          - ``"CHAR"``: print capital character,
          - ``"auto"``: print the `Variable.auto_name` values,
          - ``""``: don't print an identifier.
    stop_on_name
        When ``True``, if a node in the graph has a name, we don't print
        anything below it.
    done
        A ``dict`` where we store the ids of printed nodes.
        Useful to have multiple call to `debugprint` share the same ids.
    print_storage
        If ``True``, this will print the storage map for PyTensor functions. When
        combined with ``allow_gc=False``, after the execution of an PyTensor
        function, the output will show the intermediate results.
    used_ids
        A map between nodes and their printed ids.
    print_op_info
        Print extra information provided by the relevant `Op`\s.  For example,
        print the tap information for `Scan` inputs and outputs.
    print_destroy_map
        Whether to print the `destroy_map`\s of printed objects
    print_view_map
        Whether to print the `view_map`\s of printed objects
    print_memory_map
        Whether to set both `print_destroy_map` and `print_view_map` to ``True``.
    print_fgraph_inputs
        Print the inputs of `FunctionGraph`\s.

    Returns
    -------
    A string representing the printed graph if ``file="str"``, a
    ``rich.tree.Tree`` if ``file="rich"``, otherwise `file` (or ``None``
    when printing to stdout).

    """
    if not isinstance(depth, int):
        raise Exception("depth parameter must be an int")

    if file == "str":
        _file: TextIO | StringIO = StringIO()
    elif file == "rich":
        _file = sys.stdout  # placeholder; early return below will bypass text path
    elif file is None:
        _file = sys.stdout
    else:
        _file = file

    if done is None:
        done = dict()

    if used_ids is None:
        used_ids = dict()

    if print_memory_map:
        print_destroy_map = True
        print_view_map = True

    inputs_to_print = []
    outputs_to_print = []
    profile_list: list[Any | None] = []
    topo_orders: list[list[Apply] | None] = []
    storage_maps: list[StorageMapType | None] = []

    if isinstance(graph_like, list | tuple | set):
        graphs = graph_like
    else:
        graphs = (graph_like,)

    for obj in graphs:
        if isinstance(obj, Variable):
            outputs_to_print.append(obj)
            profile_list.append(None)
            storage_maps.append(None)
            topo_orders.append(None)
        elif isinstance(obj, Apply):
            outputs_to_print.extend(obj.outputs)
            profile_list.extend(None for item in obj.outputs)
            storage_maps.extend(None for item in obj.outputs)
            topo_orders.extend(None for item in obj.outputs)
        elif isinstance(obj, Function):
            if print_fgraph_inputs:
                inputs_to_print.extend(obj.maker.fgraph.inputs)
            outputs_to_print.extend(obj.maker.fgraph.outputs)
            profile_list.extend(obj.profile for item in obj.maker.fgraph.outputs)
            if print_storage:
                storage_maps.extend(
                    obj.vm.storage_map for item in obj.maker.fgraph.outputs
                )
            else:
                storage_maps.extend(None for item in obj.maker.fgraph.outputs)
            topo = obj.maker.fgraph.toposort()
            topo_orders.extend(topo for item in obj.maker.fgraph.outputs)
        elif isinstance(obj, FunctionGraph):
            if print_fgraph_inputs:
                inputs_to_print.extend(obj.inputs)
            outputs_to_print.extend(obj.outputs)
            profile_list.extend(getattr(obj, "profile", None) for item in obj.outputs)
            storage_maps.extend(
                getattr(obj, "storage_map", None) for item in obj.outputs
            )
            topo = obj.toposort()
            topo_orders.extend(topo for item in obj.outputs)
        elif isinstance(obj, int | float | np.ndarray):
            print(obj, file=_file)
        elif isinstance(obj, In | Out):
            outputs_to_print.append(obj.variable)
            profile_list.append(None)
            storage_maps.append(None)
            topo_orders.append(None)
        else:
            raise TypeError(f"debugprint cannot print an object type {type(obj)}")

    if file == "rich":
        return _build_rich_tree(
            outputs_to_print,
            depth=depth,
            id_type=id_type,
            print_type=print_type,
            print_shape=print_shape,
            print_destroy_map=print_destroy_map,
            print_view_map=print_view_map,
            print_op_info=print_op_info,
            stop_on_name=stop_on_name,
            topo_orders=topo_orders,
            profiles=profile_list,
            storage_maps=storage_maps,
        )

    inner_graph_vars: list[Variable] = []

    if any(p for p in profile_list if p is not None and p.fct_callcount > 0):
        print(
            """
Timing Info
-----------
--> <time> <% time> - <total time> <% total time>'

<time>         computation time for this node
<% time>       fraction of total computation time for this node
<total time>   time for this node + total times for this node's ancestors
<% total time> total time for this node over total computation time

N.B.:
* Times include the node time and the function overhead.
* <total time> and <% total time> may over-count computation times
  if inputs to a node share a common ancestor and should be viewed as a
  loose upper bound. Their intended use is to help rule out potential nodes
  to remove when optimizing a graph because their <total time> is very low.
""",
            file=_file,
        )

    op_information: dict[Apply, dict[Variable, str]] = {}

    for var in inputs_to_print:
        _debugprint(
            var,
            prefix="→ ",
            depth=depth,
            done=done,
            print_type=print_type,
            print_shape=print_shape,
            file=_file,
            id_type=id_type,
            inner_graph_ops=inner_graph_vars,
            stop_on_name=stop_on_name,
            used_ids=used_ids,
            op_information=op_information,
            parent_node=var.owner,
            print_op_info=print_op_info,
            print_destroy_map=print_destroy_map,
            print_view_map=print_view_map,
        )

    for var, profile, storage_map, topo_order in zip(
        outputs_to_print, profile_list, storage_maps, topo_orders, strict=True
    ):
        if hasattr(var.owner, "op"):
            if (
                isinstance(var.owner.op, HasInnerGraph)
                or (
                    hasattr(var.owner.op, "scalar_op")
                    and isinstance(var.owner.op.scalar_op, HasInnerGraph)
                )
            ) and var not in inner_graph_vars:
                inner_graph_vars.append(var)
            if print_op_info:
                op_information.update(op_debug_information(var.owner.op, var.owner))

        _debugprint(
            var,
            depth=depth,
            done=done,
            print_type=print_type,
            print_shape=print_shape,
            file=_file,
            topo_order=topo_order,
            id_type=id_type,
            inner_graph_ops=inner_graph_vars,
            stop_on_name=stop_on_name,
            profile=profile,
            storage_map=storage_map,
            used_ids=used_ids,
            op_information=op_information,
            parent_node=var.owner,
            print_op_info=print_op_info,
            print_destroy_map=print_destroy_map,
            print_view_map=print_view_map,
        )

    if len(inner_graph_vars) > 0:
        print("", file=_file)
        prefix = ""
        new_prefix = prefix + " ← "
        new_prefix_child = prefix + "   "
        print("Inner graphs:", file=_file)

        printed_inner_graphs_nodes = set()
        for ig_var in inner_graph_vars:
            if ig_var.owner in printed_inner_graphs_nodes:
                continue
            else:
                printed_inner_graphs_nodes.add(ig_var.owner)
            # This is a work-around to maintain backward compatibility
            # (e.g. to only print inner graphs that have been compiled through
            # a call to `Op.prepare_node`)
            inner_fn = getattr(ig_var.owner.op, "_fn", None)

            if inner_fn:
                # If the op was compiled, print the optimized version.
                inner_inputs = inner_fn.maker.fgraph.inputs
                inner_outputs = inner_fn.maker.fgraph.outputs
            else:
                if hasattr(ig_var.owner.op, "scalar_op"):
                    inner_inputs = ig_var.owner.op.scalar_op.inner_inputs
                    inner_outputs = ig_var.owner.op.scalar_op.inner_outputs
                else:
                    inner_inputs = ig_var.owner.op.inner_inputs
                    inner_outputs = ig_var.owner.op.inner_outputs

            outer_inputs = ig_var.owner.inputs

            if hasattr(ig_var.owner.op, "get_oinp_iinp_iout_oout_mappings"):
                inner_to_outer_inputs = {
                    inner_inputs[i]: outer_inputs[o]
                    for i, o in ig_var.owner.op.get_oinp_iinp_iout_oout_mappings()[
                        "outer_inp_from_inner_inp"
                    ].items()
                }
            else:
                inner_to_outer_inputs = None

            if print_op_info:
                op_information.update(
                    op_debug_information(ig_var.owner.op, ig_var.owner)
                )

            print("", file=_file)

            _debugprint(
                ig_var,
                prefix=prefix,
                depth=depth,
                done=done,
                print_type=print_type,
                print_shape=print_shape,
                file=_file,
                id_type=id_type,
                inner_graph_ops=inner_graph_vars,
                stop_on_name=stop_on_name,
                inner_to_outer_inputs=inner_to_outer_inputs,
                used_ids=used_ids,
                op_information=op_information,
                parent_node=ig_var.owner,
                print_op_info=print_op_info,
                print_destroy_map=print_destroy_map,
                print_view_map=print_view_map,
                is_inner_graph_header=True,
            )

            if print_fgraph_inputs:
                for inp in inner_inputs:
                    _debugprint(
                        inp,
                        prefix=" → ",
                        depth=depth,
                        done=done,
                        print_type=print_type,
                        print_shape=print_shape,
                        file=_file,
                        id_type=id_type,
                        stop_on_name=stop_on_name,
                        inner_graph_ops=inner_graph_vars,
                        inner_to_outer_inputs=inner_to_outer_inputs,
                        used_ids=used_ids,
                        op_information=op_information,
                        parent_node=ig_var.owner,
                        print_op_info=print_op_info,
                        print_destroy_map=print_destroy_map,
                        print_view_map=print_view_map,
                        inner_graph_node=ig_var.owner,
                    )
                inner_to_outer_inputs = None

            for out in inner_outputs:
                if (
                    out.owner is not None
                    and (
                        isinstance(out.owner.op, HasInnerGraph)
                        or isinstance(
                            getattr(out.owner.op, "scalar_op", None), HasInnerGraph
                        )
                    )
                    and out not in inner_graph_vars
                ):
                    inner_graph_vars.append(out)

                _debugprint(
                    out,
                    prefix=new_prefix,
                    depth=depth,
                    done=done,
                    print_type=print_type,
                    print_shape=print_shape,
                    file=_file,
                    id_type=id_type,
                    stop_on_name=stop_on_name,
                    prefix_child=new_prefix_child,
                    inner_graph_ops=inner_graph_vars,
                    inner_to_outer_inputs=inner_to_outer_inputs,
                    used_ids=used_ids,
                    op_information=op_information,
                    parent_node=ig_var.owner,
                    print_op_info=print_op_info,
                    print_destroy_map=print_destroy_map,
                    print_view_map=print_view_map,
                    inner_graph_node=ig_var.owner,
                )

    if file is _file:
        return file
    elif file == "str":
        assert isinstance(_file, StringIO)
        return _file.getvalue()
    else:
        _file.flush()
    return _file


def _debugprint(
    var: Variable,
    prefix: str = "",
    depth: int = -1,
    done: dict[Literal["output"] | Variable | Apply, str] | None = None,
    print_type: bool = False,
    print_shape: bool = False,
    file: TextIO = sys.stdout,
    print_destroy_map: bool = False,
    print_view_map: bool = False,
    topo_order: Sequence[Apply] | None = None,
    id_type: IDTypesType = "CHAR",
    stop_on_name: bool = False,
    prefix_child: str | None = None,
    inner_graph_ops: list[Variable] | None = None,
    profile: ProfileStats | None = None,
    inner_to_outer_inputs: dict[Variable, Variable] | None = None,
    storage_map: StorageMapType | None = None,
    used_ids: dict[Literal["output"] | Variable | Apply, str] | None = None,
    op_information: dict[Apply, dict[Variable, str]] | None = None,
    parent_node: Apply | None = None,
    print_op_info: bool = False,
    inner_graph_node: Apply | None = None,
    is_inner_graph_header: bool = False,
) -> TextIO:
    r"""Print the graph represented by `var`.

    Parameters
    ----------
    var
        A `Variable` instance.
    prefix
        Prefix to each line (typically some number of spaces).
    depth
        Print graph to this depth (``-1`` for unlimited).
    done
        See `debugprint`.
    print_type
        See `debugprint`.
    print_shape
        See `debugprint`.
    file
        File-like object to which to print.
    print_destroy_map
        Whether to print the `Variable`'s type.
    print_view_map
        Whether to print `Op` ``destroy_map``\s.
    topo_order
        If not empty will print the index in the toposort.
    id_type
        See `debugprint`.
    stop_on_name
        Whether to print `Op` ``view_map``\s.
    inner_graph_ops
        A list of `Op`\s with inner graphs.
    inner_to_outer_inputs
        A dictionary mapping an `Op`'s inner-inputs to its outer-inputs.
    storage_map
        ``None`` or the storage map (e.g. when printing an PyTensor function).
    used_ids
        See `debugprint`.
    op_information
        Extra `Op`-level information to be added to variable print-outs.
    parent_node
        The parent node of `var`.
    print_op_info
        See `debugprint`.
    inner_graph_node
        The inner-graph node in which `var` is contained.
    """
    if depth == 0:
        return file

    if done is None:
        done = {}
    if used_ids is None:
        used_ids = {}
    if inner_graph_ops is None:
        inner_graph_ops = []
    if op_information is None:
        op_information = {}

    # In the original recursive _debugprint, `prefix` is the string written
    # before the current node, and `prefix_child` is the base for children.
    # Normally prefix_child == prefix; the only exception is inner-graph outputs
    # which are called with prefix=" ← " and prefix_child="   ".
    # We replicate this by tracking an effective "child base" string and using
    # it for all nodes that are not the root of this _debugprint call.
    if prefix_child is None:
        prefix_child = prefix
    has_child_offset = prefix_child != prefix  # True only for inner-graph outputs

    for gnode in _iter_graph_nodes(
        var,
        depth=depth,
        done=done,
        stop_on_name=stop_on_name,
        inner_graph_ops=inner_graph_ops,
        inner_to_outer=inner_to_outer_inputs,
        topo_order=topo_order,
        profile=profile,
        storage_map=storage_map,
        parent_node=parent_node,
        inner_graph_node=inner_graph_node,
        is_inner_graph_header=is_inner_graph_header,
        ancestor_is_last=[],
        is_last_child=True,
    ):
        # Reconstruct the indentation prefix from ancestor_is_last.
        # The root node (depth 0) has ancestor_is_last=[] and no connector.
        # Deeper nodes build column bars from all but the last entry, then
        # append the connector selected by is_last_child.
        #
        # When prefix_child != prefix (inner-graph output case), the root uses
        # `prefix` and all children use `prefix_child` as their base.
        is_root = not gnode.ancestor_is_last
        if is_root:
            col_bars = ""
            connector = ""
            base = prefix
        else:
            col_bars = "".join(
                "   " if last else " │ " for last in gnode.ancestor_is_last[1:]
            )
            connector = " └─ " if gnode.is_last_child else " ├─ "
            base = prefix_child if has_child_offset else prefix

        full_prefix = base + col_bars + connector

        label = _build_label(
            gnode,
            done=done,
            used_ids=used_ids,
            id_type=id_type,
            print_type=print_type,
            print_shape=print_shape,
            print_destroy_map=print_destroy_map,
            print_view_map=print_view_map,
            print_op_info=print_op_info,
            op_information=op_information,
        )

        if gnode.is_repeat and not gnode.is_inner_graph_header:
            print(f"{full_prefix}···", file=file)
            continue

        if (
            gnode.profile
            and gnode.profile.apply_time
            and gnode.node in gnode.profile.apply_time
        ):
            node = gnode.node
            assert node is not None
            op_time = gnode.profile.apply_time[node]
            op_time_percent = (op_time / gnode.profile.fct_call_time) * 100
            tot_time_dict = gnode.profile.compute_total_times()
            tot_time = tot_time_dict[node]
            tot_time_percent = tot_time_dict[node] / gnode.profile.fct_call_time * 100
            print(
                f"{full_prefix}{label} --> {op_time:8.2e}s {op_time_percent:4.1f}%"
                f" {tot_time:8.2e}s {tot_time_percent:4.1f}%",
                file=file,
            )
        else:
            print(f"{full_prefix}{label}", file=file)

    return file


def _build_rich_tree(
    outputs: list[Variable],
    depth: int = -1,
    id_type: IDTypesType = "CHAR",
    print_type: bool = False,
    print_shape: bool = False,
    print_destroy_map: bool = False,
    print_view_map: bool = False,
    print_op_info: bool = False,
    stop_on_name: bool = False,
    topo_orders: list[Sequence[Apply] | None] | None = None,
    profiles: list | None = None,
    storage_maps: list | None = None,
) -> rich.tree.Tree:
    """Build a ``rich.Tree`` for one or more output `Variable`s.

    Returns a single ``rich.Tree`` with ``hide_root=True`` when there are
    multiple outputs (so the invisible root holds sibling output trees), or
    a plain tree when there is exactly one output.

    Parameters
    ----------
    outputs
        List of root `Variable`s to render.
    depth, id_type, print_type, print_shape, print_destroy_map,
    print_view_map, print_op_info, stop_on_name
        Same semantics as the identically-named parameters of `debugprint`.
    topo_orders, profiles, storage_maps
        Per-output metadata lists; ``None`` entries mean "not available".
    """
    import rich.markup

    if topo_orders is None:
        topo_orders = [None] * len(outputs)
    if profiles is None:
        profiles = [None] * len(outputs)
    if storage_maps is None:
        storage_maps = [None] * len(outputs)

    done: dict = {}
    used_ids: dict = {}
    op_information: dict[Apply, dict[Variable, str]] = {}
    inner_graph_vars: list[Variable] = []

    root = rich.tree.Tree("", hide_root=True)

    for var, topo_order, profile, storage_map in zip(
        outputs, topo_orders, profiles, storage_maps, strict=True
    ):
        # Stack maps traversal depth → rich.Tree node at that depth.
        # Depth 0 is the output variable itself (child of root).
        depth_stack: list[rich.tree.Tree] = [root]

        for gnode in _iter_graph_nodes(
            var,
            depth=depth,
            done=done,
            stop_on_name=stop_on_name,
            inner_graph_ops=inner_graph_vars,
            topo_order=topo_order,
            profile=profile,
            storage_map=storage_map,
            ancestor_is_last=[],
            is_last_child=True,
        ):
            current_depth = len(gnode.ancestor_is_last)

            label = _build_label(
                gnode,
                done=done,
                used_ids=used_ids,
                id_type=id_type,
                print_type=print_type,
                print_shape=print_shape,
                print_destroy_map=print_destroy_map,
                print_view_map=print_view_map,
                print_op_info=print_op_info,
                op_information=op_information,
            )

            label = rich.markup.escape(label)
            if gnode.is_repeat and not gnode.is_inner_graph_header:
                label = f"[dim]{label} ···[/dim]"

            parent_tree = depth_stack[current_depth]
            child_tree = parent_tree.add(label)
            # Trim the stack to this depth and push the new node.
            depth_stack = [*depth_stack[: current_depth + 1], child_tree]

    if inner_graph_vars:
        inner_root = root.add("[bold]Inner graphs:[/bold]")
        printed = set()
        for ig_var in inner_graph_vars:
            if ig_var.owner in printed:
                continue
            printed.add(ig_var.owner)

            inner_fn = getattr(ig_var.owner.op, "_fn", None)
            if inner_fn:
                inner_inputs = inner_fn.maker.fgraph.inputs
                inner_outputs = inner_fn.maker.fgraph.outputs
            else:
                if hasattr(ig_var.owner.op, "scalar_op"):
                    inner_inputs = ig_var.owner.op.scalar_op.inner_inputs
                    inner_outputs = ig_var.owner.op.scalar_op.inner_outputs
                else:
                    inner_inputs = ig_var.owner.op.inner_inputs
                    inner_outputs = ig_var.owner.op.inner_outputs

            outer_inputs = ig_var.owner.inputs
            inner_to_outer: dict[Variable, Variable] | None
            if hasattr(ig_var.owner.op, "get_oinp_iinp_iout_oout_mappings"):
                inner_to_outer = {
                    inner_inputs[i]: outer_inputs[o]
                    for i, o in ig_var.owner.op.get_oinp_iinp_iout_oout_mappings()[
                        "outer_inp_from_inner_inp"
                    ].items()
                }
            else:
                inner_to_outer = None

            if print_op_info:
                op_information.update(
                    op_debug_information(ig_var.owner.op, ig_var.owner)
                )

            # Header node for the inner graph op
            ig_depth_stack: list[rich.tree.Tree] = [inner_root]
            for gnode in _iter_graph_nodes(
                ig_var,
                depth=depth,
                done=done,
                stop_on_name=stop_on_name,
                inner_graph_ops=inner_graph_vars,
                inner_to_outer=inner_to_outer,
                parent_node=ig_var.owner,
                inner_graph_node=ig_var.owner,
                is_inner_graph_header=True,
                ancestor_is_last=[],
                is_last_child=True,
            ):
                current_depth = len(gnode.ancestor_is_last)
                label = _build_label(
                    gnode,
                    done=done,
                    used_ids=used_ids,
                    id_type=id_type,
                    print_type=print_type,
                    print_shape=print_shape,
                    print_destroy_map=print_destroy_map,
                    print_view_map=print_view_map,
                    print_op_info=print_op_info,
                    op_information=op_information,
                )
                label = rich.markup.escape(label)
                if gnode.is_repeat and not gnode.is_inner_graph_header:
                    label = f"[dim]{label} ···[/dim]"
                parent_tree = ig_depth_stack[current_depth]
                child_tree = parent_tree.add(label)
                ig_depth_stack = [*ig_depth_stack[: current_depth + 1], child_tree]

            # The header tree node is the first child of inner_root
            header_tree = inner_root.children[-1]

            for out in inner_outputs:
                if (
                    out.owner is not None
                    and (
                        isinstance(out.owner.op, HasInnerGraph)
                        or isinstance(
                            getattr(out.owner.op, "scalar_op", None), HasInnerGraph
                        )
                    )
                    and out not in inner_graph_vars
                ):
                    inner_graph_vars.append(out)

                out_stack: list[rich.tree.Tree] = [header_tree]
                for gnode in _iter_graph_nodes(
                    out,
                    depth=depth,
                    done=done,
                    stop_on_name=stop_on_name,
                    inner_graph_ops=inner_graph_vars,
                    inner_to_outer=inner_to_outer,
                    parent_node=ig_var.owner,
                    inner_graph_node=ig_var.owner,
                    ancestor_is_last=[],
                    is_last_child=True,
                ):
                    current_depth = len(gnode.ancestor_is_last)
                    label = _build_label(
                        gnode,
                        done=done,
                        used_ids=used_ids,
                        id_type=id_type,
                        print_type=print_type,
                        print_shape=print_shape,
                        print_destroy_map=print_destroy_map,
                        print_view_map=print_view_map,
                        print_op_info=print_op_info,
                        op_information=op_information,
                    )
                    label = rich.markup.escape(label)
                    if gnode.is_repeat and not gnode.is_inner_graph_header:
                        label = f"[dim]{label} ···[/dim]"
                    parent_tree = out_stack[current_depth]
                    child_tree = parent_tree.add(label)
                    out_stack = [*out_stack[: current_depth + 1], child_tree]

    return root


def _print_fn(op, xin):
    for attr in op.attrs:
        temp = getattr(xin, attr)
        if callable(temp):
            pmsg = temp()
        else:
            pmsg = temp
        print(op.message, attr, "=", pmsg)  # noqa: T201


class Print(Op):
    """This identity-like Op print as a side effect.

    This identity-like Op has the side effect of printing a message
    followed by its inputs when it runs. Default behaviour is to print
    the __str__ representation. Optionally, one can pass a list of the
    input member functions to execute, or attributes to print.

    @type message: String
    @param message: string to prepend to the output
    @type attrs: list of Strings
    @param attrs: list of input node attributes or member functions to print.
                  Functions are identified through callable(), executed and
                  their return value printed.

    :note: WARNING. This can disable some optimizations!
                    (speed and/or stabilization)

            Detailed explanation:
            As of 2012-06-21 the Print op is not known by any optimization.
            Setting a Print op in the middle of a pattern that is usually
            optimized out will block the optimization. for example, log(1+x)
            optimizes to log1p(x) but log(1+Print(x)) is unaffected by
            optimizations.

    """

    view_map = {0: [0]}

    __props__ = ("message", "attrs", "global_fn")

    def __init__(self, message="", attrs=("__str__",), global_fn=_print_fn):
        self.message = message
        self.attrs = tuple(attrs)  # attrs should be a hashable iterable
        self.global_fn = global_fn

    def make_node(self, xin):
        xout = xin.type()
        return Apply(op=self, inputs=[xin], outputs=[xout])

    def perform(self, node, inputs, output_storage):
        (xin,) = inputs
        (xout,) = output_storage
        xout[0] = xin
        self.global_fn(self, xin)

    def pullback(self, input, outputs, output_gradients):
        return output_gradients

    def pushforward(self, inputs, outputs, eval_points):
        return list(eval_points)

    def __setstate__(self, dct):
        dct.setdefault("global_fn", _print_fn)
        self.__dict__.update(dct)

    def c_code_cache_version(self):
        return (1,)

    def do_constant_folding(self, fgraph, node):
        return False


class PrinterState(Scratchpad):
    def __init__(self, props=None, **more_props):
        if props is None:
            props = {}
        elif isinstance(props, Scratchpad):
            self.__update__(props)
        else:
            self.__dict__.update(props)
        self.__dict__.update(more_props)
        # A dict from the object to print to its string
        # representation. If it is a dag and not a tree, it allow to
        # parse each node of the graph only once. They will still be
        # printed many times
        self.memo = {}


class Printer(ABC):
    @abstractmethod
    def process(self, var: Variable, pstate: PrinterState) -> str:
        """Construct a string representation for a `Variable`."""


@contextmanager
def set_precedence(pstate: PrinterState, precedence: int = -1000):
    """Temporarily set the precedence of a `PrinterState`."""
    old_precedence = getattr(pstate, "precedence", None)
    pstate.precedence = precedence
    try:
        yield
    finally:
        pstate.precedence = old_precedence


class OperatorPrinter(Printer):
    def __init__(self, operator, precedence, assoc="left"):
        self.operator = operator
        self.precedence = precedence
        self.assoc = assoc
        assert self.assoc in VALID_ASSOC

    def process(self, output, pstate):
        if output in pstate.memo:
            return pstate.memo[output]
        pprinter = pstate.pprinter
        node = output.owner
        if node is None:
            raise TypeError(
                f"operator {self.operator} cannot represent a variable that is "
                "not the result of an operation"
            )

        # Precedence seems to be buggy, see #249
        # So, in doubt, we parenthesize everything.
        # outer_precedence = getattr(pstate, 'precedence', -999999)
        # outer_assoc = getattr(pstate, 'assoc', 'none')
        # if outer_precedence > self.precedence:
        #    parenthesize = True
        # else:
        #    parenthesize = False
        parenthesize = True

        input_strings = []
        max_i = len(node.inputs) - 1
        for i, input in enumerate(node.inputs):
            new_precedence = self.precedence
            if (self.assoc == "left" and i != 0) or (
                self.assoc == "right" and i != max_i
            ):
                new_precedence += 1e-6

            with set_precedence(pstate, new_precedence):
                s = pprinter.process(input, pstate)

            input_strings.append(s)
        if len(input_strings) == 1:
            s = self.operator + input_strings[0]
        else:
            s = f" {self.operator} ".join(input_strings)
        if parenthesize:
            r = f"({s})"
        else:
            r = s
        pstate.memo[output] = r
        return r


class PatternPrinter(Printer):
    def __init__(self, *patterns):
        self.patterns = []
        for pattern in patterns:
            if isinstance(pattern, str):
                self.patterns.append((pattern, ()))
            else:
                self.patterns.append((pattern[0], pattern[1:]))

    def process(self, output, pstate):
        if output in pstate.memo:
            return pstate.memo[output]
        pprinter = pstate.pprinter
        node = output.owner
        if node is None:
            raise TypeError(
                f"Patterns {self.patterns} cannot represent a variable that is "
                "not the result of an operation"
            )
        idx = node.outputs.index(output)
        pattern, precedences = self.patterns[idx]
        precedences += (1000,) * (len(node.inputs) - len(precedences))

        def pp_process(input, new_precedence):
            with set_precedence(pstate, new_precedence):
                r = pprinter.process(input, pstate)
            return r

        d = {
            str(i): pp_process(input, precedence)
            for i, (input, precedence) in enumerate(
                zip(node.inputs, precedences, strict=True)
            )
        }
        r = pattern % d
        pstate.memo[output] = r
        return r


class FunctionPrinter(Printer):
    def __init__(self, names: list[str], keywords: list[str] | None = None):
        """
        Parameters
        ----------
        names
            The function names used for each output.
        keywords
            The `Op` keywords to include in the output.
        """
        self.names = names

        if keywords is None:
            keywords = []

        self.keywords = keywords

    def process(self, output, pstate):
        if output in pstate.memo:
            return pstate.memo[output]
        pprinter = pstate.pprinter
        node = output.owner
        if node is None:
            raise TypeError(
                f"function {self.names} cannot represent a variable that is "
                "not the result of an operation"
            )
        idx = node.outputs.index(output)
        name = self.names[idx]
        with set_precedence(pstate):
            inputs_str = ", ".join(
                pprinter.process(input, pstate) for input in node.inputs
            )
            keywords_str = ", ".join(
                f"{kw}={getattr(node.op, kw)}" for kw in self.keywords
            )

            if keywords_str and inputs_str:
                keywords_str = f", {keywords_str}"

            r = f"{name}({inputs_str}{keywords_str})"

        pstate.memo[output] = r
        return r


class IgnorePrinter(Printer):
    def process(self, output, pstate):
        if output in pstate.memo:
            return pstate.memo[output]
        pprinter = pstate.pprinter
        node = output.owner
        if node is None:
            raise TypeError(
                f"function {self.function} cannot represent a variable that is"
                " not the result of an operation"
            )
        input = node.inputs[0]
        r = f"{pprinter.process(input, pstate)}"
        pstate.memo[output] = r
        return r


class LeafPrinter(Printer):
    def process(self, output, pstate):
        if output in pstate.memo:
            return pstate.memo[output]
        if output.name in greek:
            r = greek[output.name]
        else:
            r = str(output)
        pstate.memo[output] = r
        return r


leaf_printer = LeafPrinter()


class ConstantPrinter(Printer):
    def process(self, output, pstate):
        if output in pstate.memo:
            return pstate.memo[output]
        r = str(output.data)
        pstate.memo[output] = r
        return r


constant_printer = ConstantPrinter()


class DefaultPrinter(Printer):
    def process(self, output, pstate):
        if output in pstate.memo:
            return pstate.memo[output]
        pprinter = pstate.pprinter
        node = output.owner
        if node is None:
            return leaf_printer.process(output, pstate)
        with set_precedence(pstate):
            args = ", ".join(pprinter.process(input, pstate) for input in node.inputs)
            r = f"{node.op}({args})"

        pstate.memo[output] = r
        return r


default_printer = DefaultPrinter()


class PPrinter(Printer):
    def __init__(self):
        self.printers: list[tuple[Op | type | Callable, Printer]] = []
        self.printers_dict: dict[Op | type | Callable, Printer] = {}

    def assign(self, condition: Op | type | Callable, printer: Printer):
        if isinstance(condition, Op | type):
            self.printers_dict[condition] = printer
        else:
            self.printers.insert(0, (condition, printer))

    def process(self, r: Variable, pstate: PrinterState | None = None) -> str:
        if pstate is None:
            pstate = PrinterState(pprinter=self)
        elif isinstance(pstate, dict):
            pstate = PrinterState(pprinter=self, **pstate)
        if getattr(r, "owner", None) is not None:
            if r.owner.op in self.printers_dict:
                return self.printers_dict[r.owner.op].process(r, pstate)
            if type(r.owner.op) in self.printers_dict:
                return self.printers_dict[type(r.owner.op)].process(r, pstate)
        for condition, printer in self.printers:
            if condition(pstate, r):
                return printer.process(r, pstate)
        return ""

    def clone(self):
        cp = copy(self)
        cp.printers = list(self.printers)
        cp.printers_dict = dict(self.printers_dict)
        return cp

    def clone_assign(self, condition, printer):
        cp = self.clone()
        cp.assign(condition, printer)
        return cp

    def process_graph(self, inputs, outputs, updates=None, display_inputs=False):
        if updates is None:
            updates = {}
        if not isinstance(inputs, list | tuple):
            inputs = [inputs]
        if not isinstance(outputs, list | tuple):
            outputs = [outputs]
        current = None
        if display_inputs:
            strings = [(0, "inputs: " + ", ".join(str(x) for x in [*inputs, *updates]))]
        else:
            strings = []
        pprinter = self.clone_assign(
            lambda pstate, r: r.name is not None and r is not current, leaf_printer
        )
        inv_updates = {b: a for (a, b) in updates.items()}
        i = 1
        for node in toposort([*outputs, *updates.values()], [*inputs, *updates]):
            for output in node.outputs:
                if output in inv_updates:
                    name = str(inv_updates[output])
                    strings.append((i + 1000, f"{name} <- {pprinter.process(output)}"))
                    i += 1
                if output.name is not None or output in outputs:
                    if output.name is None:
                        name = f"out[{outputs.index(output)}]"
                    else:
                        name = output.name
                    # backport
                    # name = 'out[%i]' % outputs.index(output) if output.name
                    #  is None else output.name
                    current = output
                    try:
                        idx = 2000 + outputs.index(output)
                    except ValueError:
                        idx = i
                    if len(outputs) == 1 and outputs[0] is output:
                        strings.append((idx, f"return {pprinter.process(output)}"))
                    else:
                        strings.append((idx, f"{name} = {pprinter.process(output)}"))
                    i += 1
        strings.sort()
        return "\n".join(s[1] for s in strings)

    def __call__(self, *args):
        if len(args) == 1:
            return self.process(*args)
        elif len(args) == 2 and isinstance(args[1], PrinterState | dict):
            return self.process(*args)
        elif len(args) > 2:
            return self.process_graph(*args)
        else:
            raise TypeError("Not enough arguments to call.")


use_ascii = True

if use_ascii:
    special = dict(middle_dot="\\dot", big_sigma="\\Sigma")

    greek = dict(
        alpha="\\alpha",
        beta="\\beta",
        gamma="\\gamma",
        delta="\\delta",
        epsilon="\\epsilon",
    )
else:
    special = dict(middle_dot="\u00b7", big_sigma="\u03a3")

    greek = dict(
        alpha="\u03b1",
        beta="\u03b2",
        gamma="\u03b3",
        delta="\u03b4",
        epsilon="\u03b5",
    )


pprint: PPrinter = PPrinter()
pprint.assign(lambda pstate, r: True, default_printer)
pprint.assign(lambda pstate, r: isinstance(r, Constant), constant_printer)


pp = pprint
"""
Print to the terminal a math-like expression.
"""

# colors not used: orange, amber#FFBF00, purple, pink,
# used by default: green, blue, grey, red
default_colorCodes = {
    "Scan": "yellow",
    "Shape": "brown",
    "IfElse": "magenta",
    "Elemwise": "#FFAABB",  # dark pink
    "Subtensor": "#FFAAFF",  # purple
    "Alloc": "#FFAA22",  # orange
    "Output": "blue",
}


def _try_pydot_import():
    try:
        import pydot as pd

        pd.Dot.create(pd.Dot())
        return pd
    except ImportError:
        # tests should not fail on optional dependency
        extra_msg = ""
    except Exception as e:
        extra_msg = f"\nAn error happened while importing/trying pydot: {e!r}"

    raise ImportError(
        "Failed to import pydot. You must install graphviz and pydot for "
        f"`pydotprint` to work.{extra_msg}",
    )


def pydotprint(
    fct,
    outfile: Path | str | None = None,
    compact: bool = True,
    format: str = "png",
    with_ids: bool = False,
    high_contrast: bool = True,
    cond_highlight=None,
    colorCodes: dict | None = None,
    max_label_size: int = 70,
    scan_graphs: bool = False,
    var_with_name_simple: bool = False,
    print_output_file: bool = True,
    return_image: bool = False,
):
    """Print to a file the graph of a compiled pytensor function's ops. Supports
    all pydot output formats, including png and svg.

    :param fct: a compiled PyTensor function, a Variable, an Apply or
                a list of Variable.
    :param outfile: the output file where to put the graph.
    :param compact: if True, will remove intermediate var that don't have name.
    :param format: the file format of the output.
    :param with_ids: Print the toposort index of the node in the node name.
                     and an index number in the variable ellipse.
    :param high_contrast: if true, the color that describes the respective
            node is filled with its corresponding color, instead of coloring
            the border
    :param colorCodes: dictionary with names of ops as keys and colors as
            values
    :param cond_highlight: Highlights a lazy if by surrounding each of the 3
                possible categories of ops with a border. The categories
                are: ops that are on the left branch, ops that are on the
                right branch, ops that are on both branches
                As an alternative you can provide the node that represents
                the lazy if
    :param scan_graphs: if true it will plot the inner graph of each scan op
                in files with the same name as the name given for the main
                file to which the name of the scan op is concatenated and
                the index in the toposort of the scan.
                This index can be printed with the option with_ids.
    :param var_with_name_simple: If true and a variable have a name,
                we will print only the variable name.
                Otherwise, we concatenate the type to the var name.
    :param return_image: If True, it will create the image and return it.
        Useful to display the image in ipython notebook.

        .. code-block:: python

            import pytensor

            v = pytensor.tensor.vector()
            from IPython.display import SVG

            SVG(pytensor.printing.pydotprint(v * 2, return_image=True, format="svg"))

    In the graph, ellipses are Apply Nodes (the execution of an op)
    and boxes are variables.  If variables have names they are used as
    text (if multiple vars have the same name, they will be merged in
    the graph).  Otherwise, if the variable is constant, we print its
    value and finally we print the type + a unique number to prevent
    multiple vars from being merged.  We print the op of the apply in
    the Apply box with a number that represents the toposort order of
    application of those Apply.  If an Apply has more than 1 input, we
    label each edge between an input and the Apply node with the
    input's index.

    Variable color code::
        - Cyan boxes are SharedVariable, inputs and/or outputs) of the graph,
        - Green boxes are inputs variables to the graph,
        - Blue boxes are outputs variables of the graph,
        - Grey boxes are variables that are not outputs and are not used,

    Default apply node code::
        - Red ellipses are transfers from/to the gpu
        - Yellow are scan node
        - Brown are shape node
        - Magenta are IfElse node
        - Dark pink are elemwise node
        - Purple are subtensor
        - Orange are alloc node

    For edges, they are black by default. If a node returns a view
    of an input, we put the corresponding input edge in blue. If it
    returns a destroyed input, we put the corresponding edge in red.

    .. note::

        Since October 20th, 2014, this print the inner function of all
        scan separately after the top level debugprint output.

    """
    pd = _try_pydot_import()

    from pytensor.scan.op import Scan

    if colorCodes is None:
        colorCodes = default_colorCodes

    if outfile is None:
        outfile = config.compiledir / f"pytensor.pydotprint.{config.device}.{format}"
    elif isinstance(outfile, str):
        outfile = Path(outfile)

    if isinstance(fct, Function):
        profile = getattr(fct, "profile", None)
        fgraph = fct.maker.fgraph
        outputs = fgraph.outputs
        topo = fgraph.toposort()
    elif isinstance(fct, FunctionGraph):
        profile = None
        outputs = fct.outputs
        topo = fct.toposort()
        fgraph = fct
    else:
        if isinstance(fct, Variable):
            fct = [fct]
        elif isinstance(fct, Apply):
            fct = fct.outputs
        assert isinstance(fct, list | tuple)
        assert all(isinstance(v, Variable) for v in fct)
        fct = FunctionGraph(inputs=list(graph_inputs(fct)), outputs=fct)
        profile = None
        outputs = fct.outputs
        topo = fct.toposort()
        fgraph = fct

    g = pd.Dot()

    if cond_highlight is not None:
        c1 = pd.Cluster("Left")
        c2 = pd.Cluster("Right")
        c3 = pd.Cluster("Middle")
        cond = None
        for node in topo:
            if (
                node.op.__class__.__name__ == "IfElse"
                and node.op.name == cond_highlight
            ):
                cond = node
        if cond is None:
            _logger.warning(
                "pydotprint: cond_highlight is set but there is no"
                " IfElse node in the graph"
            )
            cond_highlight = None

    if cond_highlight is not None:

        def recursive_pass(x, ls):
            if x.owner is None:
                return ls
            else:
                ls += [x.owner]
                for inp in x.inputs:
                    ls += recursive_pass(inp, ls)
                return ls

        left = set(recursive_pass(cond.inputs[1], []))
        right = set(recursive_pass(cond.inputs[2], []))
        middle = left.intersection(right)
        left = left.difference(middle)
        right = right.difference(middle)
        middle = list(middle)
        left = list(left)
        right = list(right)

    var_str = {}
    var_id = {}
    all_strings = set()

    def var_name(var):
        if var in var_str:
            return var_str[var], var_id[var]

        if var.name is not None:
            if var_with_name_simple:
                varstr = var.name
            else:
                varstr = "name=" + var.name + " " + str(var.type)
        elif isinstance(var, Constant):
            dstr = "val=" + str(np.asarray(var.data))
            if "\n" in dstr:
                dstr = dstr[: dstr.index("\n")]
            varstr = f"{dstr} {var.type}"
        elif var in input_update and input_update[var].name is not None:
            varstr = input_update[var].name
            if not var_with_name_simple:
                varstr += str(var.type)
        else:
            # a var id is needed as otherwise var with the same type will be
            # merged in the graph.
            varstr = str(var.type)
        if len(varstr) > max_label_size:
            varstr = varstr[: max_label_size - 3] + "..."
        var_str[var] = varstr
        var_id[var] = str(id(var))

        all_strings.add(varstr)

        return varstr, var_id[var]

    apply_name_cache = {}
    apply_name_id = {}

    def apply_name(node):
        if node in apply_name_cache:
            return apply_name_cache[node], apply_name_id[node]
        prof_str = ""
        if profile:
            time = profile.apply_time.get((fgraph, node), 0)
            # second, %fct time in profiler
            if profile.fct_callcount == 0 or profile.fct_call_time == 0:
                pf = 0
            else:
                pf = time * 100 / profile.fct_call_time
            prof_str = f"   ({time:.3f}s,{pf:.3f}%)"
        applystr = str(node.op).replace(":", "_")
        applystr += prof_str
        if (applystr in all_strings) or with_ids:
            idx = " id=" + str(topo.index(node))
            if len(applystr) + len(idx) > max_label_size:
                applystr = applystr[: max_label_size - 3 - len(idx)] + idx + "..."
            else:
                applystr = applystr + idx
        elif len(applystr) > max_label_size:
            applystr = applystr[: max_label_size - 3] + "..."
            idx = 1
            while applystr in all_strings:
                idx += 1
                suffix = " id=" + str(idx)
                applystr = applystr[: max_label_size - 3 - len(suffix)] + "..." + suffix

        all_strings.add(applystr)
        apply_name_cache[node] = applystr
        apply_name_id[node] = str(id(node))

        return applystr, apply_name_id[node]

    # Update the inputs that have an update function
    input_update = {}
    reverse_input_update = {}
    # Here outputs can be the original list, as we should not change
    # it, we must copy it.
    outputs = list(outputs)
    if isinstance(fct, Function):
        # TODO: Get rid of all this `expanded_inputs` nonsense and use
        # `fgraph.update_mapping`
        function_inputs = zip(fct.maker.expanded_inputs, fgraph.inputs, strict=True)
        for i, fg_ii in reversed(list(function_inputs)):
            if i.update is not None:
                k = outputs.pop()
                # Use the fgaph.inputs as it isn't the same as maker.inputs
                input_update[k] = fg_ii
                reverse_input_update[fg_ii] = k

    apply_shape = "ellipse"
    var_shape = "box"
    for node_idx, node in enumerate(topo):
        astr, aid = apply_name(node)

        use_color = None
        for opName, color in colorCodes.items():
            if opName in node.op.__class__.__name__:
                use_color = color

        if use_color is None:
            nw_node = pd.Node(aid, label=astr, shape=apply_shape)
        elif high_contrast:
            nw_node = pd.Node(
                aid, label=astr, style="filled", fillcolor=use_color, shape=apply_shape
            )
        else:
            nw_node = pd.Node(aid, label=astr, color=use_color, shape=apply_shape)
        g.add_node(nw_node)
        if cond_highlight:
            if node in middle:
                c3.add_node(nw_node)
            elif node in left:
                c1.add_node(nw_node)
            elif node in right:
                c2.add_node(nw_node)

        for idx, var in enumerate(node.inputs):
            varstr, varid = var_name(var)
            label = ""
            if len(node.inputs) > 1:
                label = str(idx)
            param = {}
            if label:
                param["label"] = label
            if node.op.view_map and idx in reduce(
                list.__add__, node.op.view_map.values(), []
            ):
                param["color"] = colorCodes["Output"]
            elif node.op.destroy_map and idx in reduce(
                list.__add__, node.op.destroy_map.values(), []
            ):
                param["color"] = "red"
            if var.owner is None:
                color = "green"
                if isinstance(var, SharedVariable):
                    # Input are green, output blue
                    # Mixing blue and green give cyan! (input and output var)
                    color = "cyan"
                if high_contrast:
                    g.add_node(
                        pd.Node(
                            varid,
                            style="filled",
                            fillcolor=color,
                            label=varstr,
                            shape=var_shape,
                        )
                    )
                else:
                    g.add_node(
                        pd.Node(varid, color=color, label=varstr, shape=var_shape)
                    )
                g.add_edge(pd.Edge(varid, aid, **param))
            elif var.name or not compact or var in outputs:
                g.add_edge(pd.Edge(varid, aid, **param))
            else:
                # no name, so we don't make a var ellipse
                if label:
                    label += " "
                label += str(var.type)
                if len(label) > max_label_size:
                    label = label[: max_label_size - 3] + "..."
                param["label"] = label
                g.add_edge(pd.Edge(apply_name(var.owner)[1], aid, **param))

        for idx, var in enumerate(node.outputs):
            varstr, varid = var_name(var)
            out = var in outputs
            label = ""
            if len(node.outputs) > 1:
                label = str(idx)
            if len(label) > max_label_size:
                label = label[: max_label_size - 3] + "..."
            param = {}
            if label:
                param["label"] = label
            if out or var in input_update:
                g.add_edge(pd.Edge(aid, varid, **param))
                if high_contrast:
                    g.add_node(
                        pd.Node(
                            varid,
                            style="filled",
                            label=varstr,
                            fillcolor=colorCodes["Output"],
                            shape=var_shape,
                        )
                    )
                else:
                    g.add_node(
                        pd.Node(
                            varid,
                            color=colorCodes["Output"],
                            label=varstr,
                            shape=var_shape,
                        )
                    )
            elif len(fgraph.clients[var]) == 0:
                g.add_edge(pd.Edge(aid, varid, **param))
                # grey mean that output var isn't used
                if high_contrast:
                    g.add_node(
                        pd.Node(
                            varid,
                            style="filled",
                            label=varstr,
                            fillcolor="grey",
                            shape=var_shape,
                        )
                    )
                else:
                    g.add_node(
                        pd.Node(varid, label=varstr, color="grey", shape=var_shape)
                    )
            elif var.name or not compact:
                if not (not compact):
                    if label:
                        label += " "
                    label += str(var.type)
                    if len(label) > max_label_size:
                        label = label[: max_label_size - 3] + "..."
                    param["label"] = label
                g.add_edge(pd.Edge(aid, varid, **param))
                g.add_node(pd.Node(varid, shape=var_shape, label=varstr))
    #            else:
    # don't add edge here as it is already added from the inputs.

    # The var that represent updates, must be linked to the input var.
    for sha, up in input_update.items():
        _, shaid = var_name(sha)
        _, upid = var_name(up)
        g.add_edge(pd.Edge(shaid, upid, label="UPDATE", color=colorCodes["Output"]))

    if cond_highlight:
        g.add_subgraph(c1)
        g.add_subgraph(c2)
        g.add_subgraph(c3)

    if outfile.suffix != f".{format}":
        outfile = outfile.with_suffix(f".{format}")

    if scan_graphs:
        scan_ops = [(idx, x) for idx, x in enumerate(topo) if isinstance(x.op, Scan)]

        for idx, scan_op in scan_ops:
            # is there a chance that name is not defined?
            if hasattr(scan_op.op, "name"):
                new_name = outfile.stem + "_" + scan_op.op.name + "_" + str(idx)
            else:
                new_name = outfile.stem + "_" + str(idx)
            new_name = outfile.with_stem(new_name)
            if hasattr(scan_op.op, "_fn"):
                to_print = scan_op.op.fn
            else:
                to_print = scan_op.op.inner_outputs
            pydotprint(
                to_print,
                new_name,
                compact,
                format,
                with_ids,
                high_contrast,
                cond_highlight,
                colorCodes,
                max_label_size,
                scan_graphs,
            )

    if return_image:
        return g.create(prog="dot", format=format)
    else:
        try:
            g.write(outfile, prog="dot", format=format)
        except pd.InvocationException:
            # based on https://github.com/Theano/Theano/issues/2988
            version = getattr(pd, "__version__", "")
            if version and [int(n) for n in version.split(".")] < [1, 0, 28]:
                raise Exception(
                    "Old version of pydot detected, which can "
                    "cause issues with pydot printing. Try "
                    "upgrading pydot version to a newer one"
                )
            raise

        if print_output_file:
            print("The output file is available at", outfile)  # noqa: T201


class _TagGenerator:
    """Class for giving abbreviated tags like to objects.
    Only really intended for internal use in order to
    implement min_informative_st"""

    def __init__(self):
        self.cur_tag_number = 0

    def get_tag(self):
        rval = char_from_number(self.cur_tag_number)

        self.cur_tag_number += 1

        return rval


def min_informative_str(
    obj, indent_level: int = 0, _prev_obs: dict | None = None, _tag_generator=None
) -> str:
    """
    Returns a string specifying to the user what obj is
    The string will print out as much of the graph as is needed
    for the whole thing to be specified in terms only of constants
    or named variables.


    Parameters
    ----------
    obj: the name to convert to a string
    indent_level: the number of tabs the tree should start printing at
                  (nested levels of the tree will get more tabs)
    _prev_obs: should only be used by min_informative_str
                    a dictionary mapping previously converted
                    objects to short tags


    Basic design philosophy
    -----------------------

    The idea behind this function is that it can be used as parts of
    command line tools for debugging or for error messages. The
    information displayed is intended to be concise and easily read by
    a human. In particular, it is intended to be informative when
    working with large graphs composed of subgraphs from several
    different people's code, as in pylearn2.

    Stopping expanding subtrees when named variables are encountered
    makes it easier to understand what is happening when a graph
    formed by composing several different graphs made by code written
    by different authors has a bug.

    An example output is:

    A. Elemwise{add_no_inplace}
        B. log_likelihood_v_given_h
        C. log_likelihood_h


    If the user is told they have a problem computing this value, it's
    obvious that either log_likelihood_h or log_likelihood_v_given_h
    has the wrong dimensionality. The variable's str object would only
    tell you that there was a problem with an
    Elemwise{add_no_inplace}. Since there are many such ops in a
    typical graph, such an error message is considerably less
    informative. Error messages based on this function should convey
    much more information about the location in the graph of the error
    while remaining succinct.

    One final note: the use of capital letters to uniquely identify
    nodes within the graph is motivated by legibility. I do not use
    numbers or lower case letters since these are pretty common as
    parts of names of ops, etc. I also don't use the object's id like
    in debugprint because it gives such a long string that takes time
    to visually diff.

    """

    if _prev_obs is None:
        _prev_obs = {}

    indent = " " * indent_level

    if id(obj) in _prev_obs:
        tag = _prev_obs[id(obj)]

        return indent + "<" + tag + ">"

    if _tag_generator is None:
        _tag_generator = _TagGenerator()

    cur_tag = _tag_generator.get_tag()

    _prev_obs[id(obj)] = cur_tag

    if hasattr(obj, "__array__"):
        name = "<ndarray>"
    elif hasattr(obj, "name") and obj.name is not None:
        name = obj.name
    elif hasattr(obj, "owner") and obj.owner is not None:
        name = str(obj.owner.op)
        for ipt in obj.owner.inputs:
            name += "\n"
            name += min_informative_str(
                ipt,
                indent_level=indent_level + 1,
                _prev_obs=_prev_obs,
                _tag_generator=_tag_generator,
            )
    else:
        name = str(obj)

    prefix = cur_tag + ". "

    rval = indent + prefix + name

    return rval


def var_descriptor(obj, _prev_obs: dict | None = None, _tag_generator=None) -> str:
    """
    Returns a string, with no endlines, fully specifying
    how a variable is computed. Does not include any memory
    location dependent information such as the id of a node.
    """
    if _prev_obs is None:
        _prev_obs = {}

    if id(obj) in _prev_obs:
        tag = _prev_obs[id(obj)]

        return "<" + tag + ">"

    if _tag_generator is None:
        _tag_generator = _TagGenerator()

    cur_tag = _tag_generator.get_tag()

    _prev_obs[id(obj)] = cur_tag

    if hasattr(obj, "__array__"):
        # hashlib hashes only the contents of the buffer, but
        # it can have different semantics depending on the strides
        # of the ndarray
        name = "<ndarray:"
        name += "strides=[" + ",".join(str(stride) for stride in obj.strides) + "]"
        name += ",digest=" + hashlib.sha256(obj).hexdigest() + ">"
    elif hasattr(obj, "owner") and obj.owner is not None:
        name = str(obj.owner.op) + "("
        name += ",".join(
            var_descriptor(ipt, _prev_obs=_prev_obs, _tag_generator=_tag_generator)
            for ipt in obj.owner.inputs
        )
        name += ")"
    elif hasattr(obj, "name") and obj.name is not None:
        # Only print the name if there is no owner.
        # This way adding a name to an intermediate node can't make
        # a deeper graph get the same descriptor as a shallower one
        name = obj.name
    else:
        name = str(obj)
        if " at 0x" in name:
            # The __str__ method is encoding the object's id in its str
            name = position_independent_str(obj)
            if " at 0x" in name:
                raise AssertionError(name)

    prefix = cur_tag + "="

    rval = prefix + name

    return rval


def position_independent_str(obj) -> str:
    if isinstance(obj, Variable):
        rval = "pytensor_var"
        rval += "{type=" + str(obj.type) + "}"
    else:
        raise NotImplementedError()

    return rval


def hex_digest(x: np.ndarray) -> str:
    """
    Returns a short, mostly hexadecimal hash of a numpy ndarray
    """
    assert isinstance(x, np.ndarray)
    rval = hashlib.sha256(x.tobytes()).hexdigest()
    # hex digest must be annotated with strides to avoid collisions
    # because the buffer interface only exposes the raw data, not
    # any info about the semantics of how that data should be arranged
    # into a tensor
    rval += "|strides=[" + ",".join(str(stride) for stride in x.strides) + "]"
    rval += "|shape=[" + ",".join(str(s) for s in x.shape) + "]"
    return rval


def get_node_by_id(
    graphs: Variable | Sequence[Variable] | Function | FunctionGraph,
    target_var_id: str,
    id_types: IDTypesType = "CHAR",
) -> Literal["output"] | Variable | Apply | None:
    r"""Get `Apply` nodes or `Variable`\s in a graph using their `debugprint` IDs.

    Parameters
    ----------
    graphs
        The graph, or graphs, to search.
    target_var_id
        The name to search for.
    id_types
        The ID scheme to use (see `debugprint`).

    Returns
    -------
    The `Apply`/`Variable` matching `target_var_id` or ``None``.

    """
    from pytensor.printing import debugprint

    used_ids: dict[Literal["output"] | Variable | Apply, str] = {}

    _ = debugprint(graphs, file="str", used_ids=used_ids, id_type=id_types)

    id_to_node = {v: k for k, v in used_ids.items()}

    id_str = f"[id {target_var_id}]"

    return id_to_node.get(id_str, None)
