"""Fuse indexed reads into Elemwise iteration loops.

Introduces ``IndexedElemwise``, an ``OpFromGraph`` that wraps
``AdvancedSubtensor1`` + ``Elemwise`` subgraphs so the Numba backend can
generate a single loop with indirect indexing, eliminating materialised
intermediate arrays.
"""

from pytensor.compile import optdb
from pytensor.compile.builders import OpFromGraph
from pytensor.graph.rewriting.basic import GraphRewriter
from pytensor.graph.rewriting.db import SequenceDB
from pytensor.printing import op_debug_information
from pytensor.tensor.elemwise import Elemwise
from pytensor.tensor.rewriting.elemwise import InplaceElemwiseOptimizer
from pytensor.tensor.subtensor import (
    AdvancedSubtensor1,
)


indexed_elemwise_optdb = SequenceDB()
optdb.register(
    "fuse_indexed_into_elemwise",
    indexed_elemwise_optdb,
    "numba",
    # After inplace_elemwise (position=50.5) so we see final inplace patterns,
    # same position as other numba-specific rewrites (BlockwiseWithCoreShape).
    position=100,
)


class IndexedElemwise(OpFromGraph):
    """Fuse indexed reads into a single Elemwise iteration loop.

    Absorbs ``AdvancedSubtensor1`` (indexed reads on inputs) into one loop,
    avoiding materialisation of intermediate arrays.

    Inner fgraph contains the unfused subgraph.
    Non-Numba backends run it as-is via ``OpFromGraph.perform``.
    The Numba backend generates a single loop with indirect indexing.

    Outer inputs are ordered as::

        [elemwise_inputs..., idx_0, idx_1, ...]

    Elemwise inputs whose values are read via an index have their source
    arrays substituted in place.

    Parameters
    ----------
    indexed_inputs : tuple of (tuple[int, ...], int, tuple[bool, ...])
        One entry per index array: ``(elemwise_input_positions, axis, idx_broadcastable)``.
    """

    def __init__(self, *args, indexed_inputs=(), **kwargs):
        self.indexed_inputs = indexed_inputs
        super().__init__(*args, on_unused_input="ignore", **kwargs)

    def __str__(self):
        for node in self.fgraph.apply_nodes:
            if isinstance(node.op, Elemwise):
                return f"IndexedElemwise{{{node.op!s}}}"
        return "IndexedElemwise"


@op_debug_information.register(IndexedElemwise)
def _op_debug_information_IndexedElemwise(op, node):
    info = {}

    n_idx = len(op.indexed_inputs)
    n_elemwise = len(node.inputs) - n_idx

    # Annotate indexed-read inputs
    for k, (positions, _axis, _bc) in enumerate(op.indexed_inputs):
        idx_label = f"idx_{k}"
        for pos in positions:
            if pos < len(node.inputs):
                info[node.inputs[pos]] = f"indexed read ({idx_label})"

    # Annotate index arrays (after elemwise inputs)
    for k in range(n_idx):
        idx_pos = n_elemwise + k
        if idx_pos < len(node.inputs):
            info[node.inputs[idx_pos]] = f"idx_{k}"

    return {node: info}


class FuseIndexedElemwise(GraphRewriter):
    """Fuse indexed reads into Elemwise loops.

    Absorbs single-client ``AdvancedSubtensor1`` on inputs (indexed reads)
    into the Elemwise iteration, avoiding intermediate arrays.

    Supports multiple index arrays: e.g. ``x[idx_a] + y[idx_b]`` produces
    two index groups.  Index arrays are shared between reads when they refer
    to the same variable.
    """

    def apply(self, fgraph):
        def _get_indexed_read_info(var):
            """Extract indexed-read info from a variable.

            Returns ``(source, [(idx_var, axis), ...])`` or ``None``.
            Handles:
            - ``AdvancedSubtensor1(source, idx)`` -> single index on axis 0
            """
            if var.owner is None:
                return None
            op = var.owner.op
            if isinstance(op, AdvancedSubtensor1):
                return (var.owner.inputs[0], [(var.owner.inputs[1], 0)])
            return None

        def find_indexed_input_groups(fgraph, node):
            """Find single-client indexed-read inputs grouped by (index, axis).

            Returns ``[(idx_var, axis, (pos, ...))]`` -- one entry per distinct
            ``(idx_var, axis)`` pair.
            """
            groups = {}  # (idx_var, axis) -> (idx_var, axis, list of positions)
            for i, inp in enumerate(node.inputs):
                info = _get_indexed_read_info(inp)
                if info is None:
                    continue
                if any(c is not node for c, _ in fgraph.clients[inp]):
                    continue
                _source, idx_axis_pairs = info
                for idx_var, axis in idx_axis_pairs:
                    key = (idx_var, axis)
                    if key not in groups:
                        groups[key] = (idx_var, axis, [])
                    groups[key][2].append(i)

            return [(var, axis, tuple(pos)) for var, axis, pos in groups.values()]

        for node in reversed(fgraph.toposort()):
            if not isinstance(node.op, Elemwise):
                continue

            read_groups = find_indexed_input_groups(fgraph, node)

            if not read_groups:
                continue

            indexed_positions = {
                p for _, _ax, positions in read_groups for p in positions
            }

            # If any inplace targets an indexed-read input, strip and re-run
            # inplace with those inputs protected
            if any(
                inp_idx in indexed_positions
                for inp_idx in node.op.inplace_pattern.values()
            ):
                stripped_node = Elemwise(node.op.scalar_op).make_node(*node.inputs)
                fgraph.replace_all_validate(
                    list(zip(node.outputs, stripped_node.outputs)),
                    reason="fuse_indexed_elemwise_strip_inplace",
                )

                protected = frozenset(
                    stripped_node.inputs[i] for i in indexed_positions
                )
                node = InplaceElemwiseOptimizer().try_inplace_on_node(
                    fgraph,
                    stripped_node,
                    reason="fuse_indexed_elemwise_reinplace",
                    extra_protected_inputs=protected,
                )

            # Merge read index arrays into a unified ordered list
            all_idx_groups = {}  # (idx_var, axis) -> (idx_var, position)
            for idx_var, _ax, _ in read_groups:
                key = (idx_var, _ax)
                if key not in all_idx_groups:
                    all_idx_groups[key] = (idx_var, len(all_idx_groups))

            n_indices = len(all_idx_groups)
            idx_vars = [None] * n_indices
            for idx_var, pos in all_idx_groups.values():
                idx_vars[pos] = idx_var

            # Build destroy_map
            outer_destroy_map = {}
            for out_idx, inp_idx in node.op.inplace_pattern.items():
                if inp_idx not in indexed_positions:
                    outer_destroy_map[out_idx] = [inp_idx]

            # Inner fgraph inputs:
            #   [elemwise_inputs (sources substituted)..., idx_0, ...]
            inner_inputs = [
                inp.owner.inputs[0] if i in indexed_positions else inp
                for i, inp in enumerate(node.inputs)
            ]
            inner_inputs = inner_inputs + idx_vars

            outer_inputs = list(inner_inputs)

            # Build indexed_inputs spec for the Op
            indexed_inputs_spec = [None] * n_indices
            for idx_var, _ax, positions in read_groups:
                key = (idx_var, _ax)
                _var, idx_pos = all_idx_groups[key]
                indexed_inputs_spec[idx_pos] = (
                    positions,
                    _ax,
                    idx_var.type.broadcastable,
                )
            # Fill entries for index arrays with no reads
            for k in range(n_indices):
                if indexed_inputs_spec[k] is None:
                    for (iv, ax), (v, p) in all_idx_groups.items():
                        if p == k:
                            indexed_inputs_spec[k] = ((), ax, iv.type.broadcastable)
                            break

            new_outs = IndexedElemwise(
                inner_inputs,
                node.outputs,
                destroy_map=outer_destroy_map,
                indexed_inputs=tuple(indexed_inputs_spec),
            )(*outer_inputs, return_list=True)

            fgraph.replace_all_validate(
                list(zip(node.outputs, new_outs)),
                reason="fuse_indexed_into_elemwise",
            )


indexed_elemwise_optdb.register(
    "fuse_indexed_elemwise",
    FuseIndexedElemwise(),
    "numba",
    position=1,
)
