"""Fuse indexed reads and updates into Elemwise iteration loops.

Introduces ``IndexedElemwise``, an ``OpFromGraph`` that wraps
``AdvancedSubtensor1`` + ``Elemwise`` + ``AdvancedIncSubtensor1`` subgraphs
so the Numba backend can generate a single loop with indirect indexing,
eliminating materialised intermediate arrays.
"""

from pytensor.compile import optdb
from pytensor.compile.builders import OpFromGraph
from pytensor.graph import node_rewriter
from pytensor.graph.rewriting.basic import GraphRewriter, dfs_rewriter
from pytensor.graph.rewriting.db import SequenceDB
from pytensor.printing import op_debug_information
from pytensor.scalar.basic import Composite, identity
from pytensor.tensor.elemwise import DimShuffle, Elemwise
from pytensor.tensor.rewriting.elemwise import InplaceElemwiseOptimizer
from pytensor.tensor.shape import Reshape
from pytensor.tensor.subtensor import (
    AdvancedIncSubtensor,
    AdvancedIncSubtensor1,
    AdvancedSubtensor,
    AdvancedSubtensor1,
    indices_from_subtensor,
)


@node_rewriter([DimShuffle])
def undo_take_dimshuffle_for_fusion(fgraph, node):
    """Undo ``DimShuffle(AdvancedSubtensor1(DimShuffle(x), idx))`` -> ``AdvancedSubtensor(x, :, ..., idx, :, ...)``.

    The ``local_replace_AdvancedSubtensor`` specialize rewrite converts
    ``x[:, idx]`` into ``x.T[idx].T`` (axis-swap + AdvancedSubtensor1 +
    axis-swap).  This rewrite undoes that when the result feeds a single
    Elemwise, so ``FuseIndexedElemwise`` can absorb the indexing directly
    on the correct axis.

    See also ``undo_take_reshape_for_fusion`` which handles the analogous
    Reshape+flatten pattern for ND indices.
    """
    # Outer DimShuffle must be an axis swap
    outer_ds = node.op
    if outer_ds.augment or outer_ds.drop:
        return None
    order = outer_ds.new_order
    ndim = len(order)
    if ndim < 2:
        return None

    # Find the swapped axis: exactly two positions differ from identity
    swapped = [i for i in range(ndim) if order[i] != i]
    if len(swapped) != 2:
        return None
    ax_a, ax_b = swapped
    if order[ax_a] != ax_b or order[ax_b] != ax_a:
        return None
    axis = max(ax_a, ax_b)  # the non-zero axis (0 was swapped to axis)

    # Inner must be AdvancedSubtensor1
    inner = node.inputs[0]
    if inner.owner is None or not isinstance(inner.owner.op, AdvancedSubtensor1):
        return None
    asub1_node = inner.owner

    # AdvancedSubtensor1's input must be the inverse DimShuffle.
    # For a pair swap, the inverse is the same permutation (swap is self-inverse).
    # A general permutation would need argsort(order) here.
    inner_ds_var = asub1_node.inputs[0]
    if inner_ds_var.owner is None or not isinstance(inner_ds_var.owner.op, DimShuffle):
        return None
    inner_ds = inner_ds_var.owner.op
    if inner_ds.new_order != tuple(order):
        return None

    # Both intermediates must be single-client
    if len(fgraph.clients[inner]) != 1:
        return None
    if len(fgraph.clients[inner_ds_var]) != 1:
        return None

    # Outer DimShuffle must be consumed only by a single Elemwise
    clients = fgraph.clients[node.outputs[0]]
    if len(clients) != 1:
        return None
    client_node, _client_idx = clients[0]
    if not isinstance(getattr(client_node, "op", None), Elemwise):
        return None

    # Build AdvancedSubtensor: x[:, ..., idx, :, ...]
    source = inner_ds_var.owner.inputs[0]
    idx_var = asub1_node.inputs[1]

    idx_list = [slice(None)] * ndim
    idx_list[axis] = 0  # pointer to the single index variable
    new_out = AdvancedSubtensor(idx_list=idx_list)(source, idx_var)
    return [new_out]


@node_rewriter([Reshape])
def undo_take_reshape_for_fusion(fgraph, node):
    """Undo ``Reshape(AdvancedSubtensor1(x, flatten(idx)), shape)`` for ND indices.

    ``transform_take`` rewrites ``x[mat_idx]`` (ND integer index) into
    ``AdvancedSubtensor1(x, mat_idx.ravel()).reshape(mat_idx.shape + ...)``,
    possibly with DimShuffle axis-swaps for non-zero axes.  This rewrite
    undoes that so ``FuseIndexedElemwise`` can absorb the ND index directly.
    """
    [reshape_out] = node.outputs

    # Must feed a single Elemwise (or chain to one via another pre-fusion rewrite)
    clients = fgraph.clients[reshape_out]
    if len(clients) != 1:
        return None
    client_node, _ = clients[0]
    if not isinstance(getattr(client_node, "op", None), Elemwise):
        return None

    inner = node.inputs[0]
    if inner.owner is None:
        return None

    # --- Detect axis-0 pattern: Reshape(AdvancedSubtensor1(src, flatten(idx)), shape)
    # --- Detect axis>0 pattern: Reshape(DimShuffle(AdvancedSubtensor1(DimShuffle(src), flatten(idx))), shape)
    axis = 0
    asub1_node = None

    if isinstance(inner.owner.op, AdvancedSubtensor1):
        asub1_node = inner.owner
    elif isinstance(inner.owner.op, DimShuffle):
        # Check for axis-swap DimShuffle wrapping AdvancedSubtensor1
        outer_ds = inner.owner.op
        if outer_ds.augment or outer_ds.drop:
            return None
        order = outer_ds.new_order
        ndim_ds = len(order)
        if ndim_ds < 2:
            return None
        swapped = [i for i in range(ndim_ds) if order[i] != i]
        if len(swapped) != 2:
            return None
        ax_a, ax_b = swapped
        if order[ax_a] != ax_b or order[ax_b] != ax_a:
            return None
        axis = max(ax_a, ax_b)

        ds_inner = inner.owner.inputs[0]
        if ds_inner.owner is None or not isinstance(
            ds_inner.owner.op, AdvancedSubtensor1
        ):
            return None
        asub1_node = ds_inner.owner

        # AdvancedSubtensor1's source must be the inverse DimShuffle
        src_var = asub1_node.inputs[0]
        if src_var.owner is None or not isinstance(src_var.owner.op, DimShuffle):
            return None
        if src_var.owner.op.new_order != tuple(order):
            return None

        # Intermediates must be single-client
        if len(fgraph.clients[ds_inner]) != 1:
            return None
        if len(fgraph.clients[src_var]) != 1:
            return None
    else:
        return None

    if asub1_node is None:
        return None

    # The index input to AdvancedSubtensor1 must be Reshape{1}(mat_idx, [-1]) (flatten)
    flat_idx = asub1_node.inputs[1]
    if flat_idx.owner is None or not isinstance(flat_idx.owner.op, Reshape):
        return None
    if flat_idx.owner.op.ndim != 1:
        return None
    mat_idx = flat_idx.owner.inputs[0]
    if mat_idx.ndim < 2:
        return None

    # AdvancedSubtensor1 output must be single-client
    if len(fgraph.clients[asub1_node.outputs[0]]) != 1:
        return None

    # Recover the original source (unwrap inner DimShuffle if axis > 0)
    if axis > 0:
        source = asub1_node.inputs[0].owner.inputs[0]
    else:
        source = asub1_node.inputs[0]

    # Build AdvancedSubtensor: source[:, ..., mat_idx, :, ...]
    src_ndim = source.type.ndim
    idx_list = [slice(None)] * src_ndim
    idx_list[axis] = 0  # pointer to the single index variable
    new_out = AdvancedSubtensor(idx_list=idx_list)(source, mat_idx)
    return [new_out]


indexed_elemwise_optdb = SequenceDB()
optdb.register(
    "fuse_indexed_into_elemwise",
    indexed_elemwise_optdb,
    "numba",
    # After inplace_elemwise (position=50.5) so we see final inplace patterns,
    # same position as other numba-specific rewrites (BlockwiseWithCoreShape).
    position=100,
)

indexed_elemwise_optdb.register(
    "undo_take_dimshuffle_for_fusion",
    dfs_rewriter(undo_take_dimshuffle_for_fusion),
    "numba",
    position=0,
)

indexed_elemwise_optdb.register(
    "undo_take_reshape_for_fusion",
    dfs_rewriter(undo_take_reshape_for_fusion),
    "numba",
    position=0.5,
)


class IndexedElemwise(OpFromGraph):
    """Fuse indexed reads and updates into a single Elemwise iteration loop.

    Absorbs ``AdvancedSubtensor1`` (indexed reads on inputs) and
    ``AdvancedIncSubtensor1`` (indexed updates on outputs) into one loop,
    avoiding materialisation of intermediate arrays.

    Inner fgraph contains the unfused subgraph.
    Non-Numba backends run it as-is via ``OpFromGraph.perform``.
    The Numba backend generates a single loop with indirect indexing.

    Outer inputs are ordered as::

        [elemwise_inputs..., idx_0, idx_1, ..., update_target_0, ...]

    Elemwise inputs whose values are read via an index have their source
    arrays substituted in place.

    Parameters
    ----------
    indexed_inputs : tuple of (tuple[int, ...], int, tuple[bool, ...])
        One entry per index array: ``(elemwise_input_positions, axis, idx_broadcastable)``.
    indexed_outputs : tuple of ((tuple[int, ...], str, int) | None)
        One entry per index array: ``(output_positions, mode, axis)`` or ``None``.
    """

    def __init__(self, *args, indexed_inputs=(), indexed_outputs=(), **kwargs):
        self.indexed_inputs = indexed_inputs
        self.indexed_outputs = indexed_outputs
        super().__init__(*args, on_unused_input="ignore", accept_inplace=True, **kwargs)

    def __str__(self):
        for node in self.fgraph.apply_nodes:
            if isinstance(node.op, Elemwise):
                return f"IndexedElemwise{{{node.op!s}}}"
        return "IndexedElemwise"


@op_debug_information.register(IndexedElemwise)
def _op_debug_information_IndexedElemwise(op, node):
    info = {}

    n_idx = len(op.indexed_inputs)
    n_update_targets = sum(1 for e in op.indexed_outputs if e is not None)
    n_elemwise = len(node.inputs) - n_idx - n_update_targets

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

    # Annotate update targets and outputs
    buf_counter = 0
    target_start = n_elemwise + n_idx
    target_offset = 0
    for k, entry in enumerate(op.indexed_outputs):
        if entry is None:
            continue
        out_positions, mode, _axis = entry
        buf_label = f"buf_{buf_counter}"
        buf_counter += 1
        idx_label = f"idx_{k}"

        target_pos = target_start + target_offset
        target_offset += 1
        if target_pos < len(node.inputs):
            info[node.inputs[target_pos]] = buf_label

        for out_idx in out_positions:
            if out_idx < len(node.outputs):
                info[node.outputs[out_idx]] = (
                    f"indexed {mode} ({buf_label}, {idx_label})"
                )

    return {node: info}


class FuseIndexedElemwise(GraphRewriter):
    """Fuse indexed reads and indexed updates into Elemwise loops.

    Absorbs single-client ``AdvancedSubtensor1`` on inputs (indexed reads)
    and single-client ``AdvancedIncSubtensor1`` on outputs (indexed updates)
    into the Elemwise iteration, avoiding intermediate arrays.

    Supports multiple index arrays: e.g. ``x[idx_a] + y[idx_b]`` produces
    two index groups.  Index arrays are shared between reads and updates
    when they refer to the same variable.
    """

    def apply(self, fgraph):
        def _get_indexed_read_info(var):
            """Extract indexed-read info from a variable.

            Returns ``(source, [(idx_var, axis), ...])`` or ``None``.
            Handles:
            - ``AdvancedSubtensor1(source, idx)`` -> single index on axis 0
            - ``AdvancedSubtensor(source, :, ..., idx, :, ...)`` -> single or
              multi-index with tensor indices on consecutive axes,
              followed only by full slices.
            """
            if var.owner is None:
                return None
            op = var.owner.op
            if isinstance(op, AdvancedSubtensor1):
                return (var.owner.inputs[0], [(var.owner.inputs[1], 0)])
            if isinstance(op, AdvancedSubtensor):
                indices = indices_from_subtensor(var.owner.inputs[1:], op.idx_list)
                # Collect consecutive advanced (tensor) indices
                adv = []
                for i, idx in enumerate(indices):
                    if idx == slice(None):
                        if adv:
                            break  # trailing slices after the advanced group
                    elif hasattr(idx, "ndim") and idx.dtype != "bool":
                        if adv and adv[-1][1] != i - 1:
                            return None  # non-consecutive
                        adv.append((idx, i))
                    else:
                        return None  # unsupported index type
                if not adv:
                    return None
                # Verify only full slices remain after the advanced group
                for idx in indices[adv[-1][1] + 1 :]:
                    if idx != slice(None):
                        return None
                return (var.owner.inputs[0], adv)
            return None

        def find_indexed_input_groups(fgraph, node):
            """Find single-client indexed-read inputs grouped by (index, axis).

            Returns ``[(idx_var, axis, (pos, ...))]`` -- one entry per distinct
            ``(idx_var, axis)`` pair.  A multi-index input contributes multiple
            entries (one per indexed axis).
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

        def _get_indexed_update_info(client_node):
            """Extract indexed-update info from an AdvancedInc node.

            Returns ``(target, [(idx_var, axis), ...], mode)`` or ``None``.
            """
            op = client_node.op
            if isinstance(op, AdvancedIncSubtensor1):
                target, _val, idx_var = client_node.inputs
                mode = "set" if op.set_instead_of_inc else "inc"
                return (target, [(idx_var, 0)], mode)
            if isinstance(op, AdvancedIncSubtensor):
                target = client_node.inputs[0]
                _val = client_node.inputs[1]
                index_vars = client_node.inputs[2:]
                indices = indices_from_subtensor(index_vars, op.idx_list)
                adv = []
                for i, idx in enumerate(indices):
                    if idx == slice(None):
                        if adv:
                            break
                    elif hasattr(idx, "ndim") and idx.dtype != "bool":
                        if adv and adv[-1][1] != i - 1:
                            return None
                        adv.append((idx, i))
                    else:
                        return None
                if not adv:
                    return None
                for idx in indices[adv[-1][1] + 1 :]:
                    if idx != slice(None):
                        return None
                mode = "set" if op.set_instead_of_inc else "inc"
                return (target, adv, mode)
            return None

        def find_indexed_update_consumers(fgraph, node):
            """Find indexed-update consumers of Elemwise outputs.

            Returns ``{out_idx: (update_node, target, [(idx_var, axis), ...], mode)}``.
            Only considers outputs that are the value input (position 1) of
            the indexed update.
            """
            update_info = {}
            for out_idx, out in enumerate(node.outputs):
                clients = fgraph.clients[out]
                inc_clients = [
                    (c, ci)
                    for c, ci in clients
                    if ci == 1
                    and isinstance(c.op, AdvancedIncSubtensor1 | AdvancedIncSubtensor)
                ]
                if len(inc_clients) != 1:
                    continue
                [(client_node, _)] = inc_clients
                info = _get_indexed_update_info(client_node)
                if info is None:
                    continue
                target, idx_axis_pairs, mode = info
                # Don't fuse if the value broadcasts on the index loop dim
                # (constant across index — recomputing per position is wasteful)
                # or against non-indexed target axes.
                val = client_node.inputs[1]
                n_idx_dims = max((idx.ndim for idx, _ in idx_axis_pairs), default=0)
                val_idx_bc = list(val.type.broadcastable)[:n_idx_dims]
                if any(val_idx_bc):
                    continue
                indexed_axes = {a for _, a in idx_axis_pairs}
                non_indexed_target_bc = [
                    bc
                    for i, bc in enumerate(target.type.broadcastable)
                    if i not in indexed_axes
                ]
                non_indexed_val_bc = list(val.type.broadcastable)
                non_indexed_val_bc = non_indexed_val_bc[n_idx_dims:]
                if len(non_indexed_val_bc) < len(non_indexed_target_bc) or any(
                    vbc and not tbc
                    for vbc, tbc in zip(
                        non_indexed_val_bc, non_indexed_target_bc, strict=False
                    )
                ):
                    continue
                update_info[out_idx] = (
                    client_node,
                    target,
                    idx_axis_pairs,
                    mode,
                )
            return update_info

        for node in reversed(fgraph.toposort()):
            if not isinstance(node.op, Elemwise):
                continue

            read_groups = find_indexed_input_groups(fgraph, node)
            update_consumers = find_indexed_update_consumers(fgraph, node)

            if not read_groups and not update_consumers:
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
                # Re-detect after inplace change
                update_consumers = find_indexed_update_consumers(fgraph, node)

            # Merge read and update index arrays into a unified ordered list
            all_idx_groups = {}  # (idx_var, axis) -> (idx_var, position)
            for idx_var, _ax, _ in read_groups:
                key = (idx_var, _ax)
                if key not in all_idx_groups:
                    all_idx_groups[key] = (idx_var, len(all_idx_groups))
            for _un, _target, idx_axis_pairs, _mode in update_consumers.values():
                for idx_var, axis in idx_axis_pairs:
                    key = (idx_var, axis)
                    if key not in all_idx_groups:
                        all_idx_groups[key] = (idx_var, len(all_idx_groups))

            n_indices = len(all_idx_groups)
            idx_vars = [None] * n_indices
            for idx_var, pos in all_idx_groups.values():
                idx_vars[pos] = idx_var

            # Build destroy_map
            outer_destroy_map = {}
            for out_idx, inp_idx in node.op.inplace_pattern.items():
                if inp_idx not in indexed_positions and out_idx not in update_consumers:
                    outer_destroy_map[out_idx] = [inp_idx]

            # Inner fgraph inputs:
            #   [elemwise_inputs (sources substituted)..., idx_0, ..., target_0, ...]
            inner_inputs = [
                inp.owner.inputs[0] if i in indexed_positions else inp
                for i, inp in enumerate(node.inputs)
            ]
            inner_inputs = inner_inputs + idx_vars

            # If any scatter output also has other consumers, duplicate
            # the elemwise output via Composite so the scatter can replace
            # the duplicate while the original stays available.
            multi_client_outs = set()
            for out_idx in update_consumers:
                update_node = update_consumers[out_idx][0]
                if any(
                    c is not update_node
                    for c, _ in fgraph.clients[node.outputs[out_idx]]
                ):
                    multi_client_outs.add(out_idx)

            if multi_client_outs:
                # Rebuild the Elemwise with duplicated outputs.
                # e.g. Exp(x) -> Composite([x], [exp(x), exp(x)])(x)
                # so the Elemwise produces [out0, out1] where out1 is
                # available for the scatter to replace.
                scalar_op = node.op.scalar_op
                if isinstance(scalar_op, Composite):
                    s_inputs = list(scalar_op.inputs)
                    s_outputs = list(scalar_op.outputs)
                else:
                    scalar_node = scalar_op.make_node(
                        *[inp.type.to_scalar_type()() for inp in node.inputs]
                    )
                    s_inputs = list(scalar_node.inputs)
                    s_outputs = list(scalar_node.outputs)

                # Map from original out_idx to the new duplicate out_idx.
                # Wrap duplicates with identity so Composite._cleanup_graph
                # doesn't clone the entire subgraph for repeated outputs.
                # TODO: _cleanup_graph should use identity instead of clone
                # for duplicate outputs.
                dup_map = {}
                for out_idx in sorted(multi_client_outs):
                    dup_map[out_idx] = len(s_outputs)
                    s_outputs.append(identity(s_outputs[out_idx]))

                new_scalar_op = Composite(s_inputs, s_outputs)
                new_elemwise = Elemwise(new_scalar_op)(*node.inputs, return_list=True)

                # Update node reference and remap outputs
                old_node = node
                node = new_elemwise[0].owner

                # Rebuild inner_inputs with the new node's inputs
                inner_inputs = [
                    inp.owner.inputs[0] if i in indexed_positions else inp
                    for i, inp in enumerate(node.inputs)
                ]
                inner_inputs = inner_inputs + idx_vars
            else:
                dup_map = {}

            # Inner fgraph outputs; add update targets
            inner_outputs = list(node.outputs)
            call_inputs = list(inner_inputs)
            for out_idx in sorted(update_consumers.keys()):
                update_node, target, idx_axis_pairs, _mode = update_consumers[out_idx]

                inner_inputs.append(target)

                target_pos = len(call_inputs)
                if update_node.op.inplace:
                    call_inputs.append(target)
                else:
                    call_inputs.append(target.copy())

                # Use the duplicate output for scatter if multi-client
                scatter_idx = dup_map.get(out_idx, out_idx)
                # The value to scatter: use the (possibly duplicated) Elemwise output
                scatter_value = node.outputs[scatter_idx]

                # Build the scatter output using the correct value
                if update_node.op.inplace:
                    # Rebuild with the new value
                    if isinstance(update_node.op, AdvancedIncSubtensor1):
                        scatter_out = update_node.op(
                            target, scatter_value, update_node.inputs[2]
                        )
                    else:
                        scatter_out = update_node.op(
                            target, scatter_value, *update_node.inputs[2:]
                        )
                elif isinstance(update_node.op, AdvancedIncSubtensor1):
                    inplace_op = AdvancedIncSubtensor1(
                        inplace=True,
                        set_instead_of_inc=update_node.op.set_instead_of_inc,
                    )
                    scatter_out = inplace_op(
                        target, scatter_value, update_node.inputs[2]
                    )
                else:
                    inplace_op = AdvancedIncSubtensor(
                        idx_list=update_node.op.idx_list,
                        inplace=True,
                        set_instead_of_inc=update_node.op.set_instead_of_inc,
                    )
                    scatter_out = inplace_op(
                        target, scatter_value, *update_node.inputs[2:]
                    )
                inner_outputs[scatter_idx] = scatter_out
                outer_destroy_map[scatter_idx] = [target_pos]

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
            # Fill entries for index arrays with no reads (write-only)
            for k in range(n_indices):
                if indexed_inputs_spec[k] is None:
                    for (iv, ax), (v, p) in all_idx_groups.items():
                        if p == k:
                            indexed_inputs_spec[k] = ((), ax, iv.type.broadcastable)
                            break

            # Build indexed_outputs spec for the Op
            indexed_outputs_spec = [None] * n_indices
            for out_idx in sorted(update_consumers.keys()):
                _update_node, _target, idx_axis_pairs, mode = update_consumers[out_idx]
                for idx_var, axis in idx_axis_pairs:
                    key = (idx_var, axis)
                    idx_pos = all_idx_groups[key][1]
                    scatter_idx = dup_map.get(out_idx, out_idx)
                    if indexed_outputs_spec[idx_pos] is None:
                        indexed_outputs_spec[idx_pos] = ([scatter_idx], mode, axis)
                    else:
                        indexed_outputs_spec[idx_pos][0].append(scatter_idx)
            indexed_outputs_spec = tuple(
                (tuple(e[0]), e[1], e[2]) if e is not None else None
                for e in indexed_outputs_spec
            )

            new_outs = IndexedElemwise(
                inner_inputs,
                inner_outputs,
                destroy_map=outer_destroy_map,
                indexed_inputs=tuple(indexed_inputs_spec),
                indexed_outputs=indexed_outputs_spec,
            )(*call_inputs, return_list=True)

            # The node whose outputs we need to replace in the outer graph
            orig_node = old_node if multi_client_outs else node
            replacements = []
            for out_idx in range(len(orig_node.outputs)):
                if out_idx in update_consumers:
                    update_node = update_consumers[out_idx][0]
                    scatter_idx = dup_map.get(out_idx, out_idx)
                    replacements.append((update_node.outputs[0], new_outs[scatter_idx]))
                    if out_idx in dup_map:
                        # Multi-client: also replace the raw elemwise output
                        replacements.append(
                            (orig_node.outputs[out_idx], new_outs[out_idx])
                        )
                else:
                    replacements.append((orig_node.outputs[out_idx], new_outs[out_idx]))

            fgraph.replace_all_validate(
                replacements,
                reason="fuse_indexed_into_elemwise",
            )


indexed_elemwise_optdb.register(
    "fuse_indexed_elemwise",
    FuseIndexedElemwise(),
    "numba",
    position=1,
)
