"""Fuse indexed reads/writes and reductions into Elemwise iteration loops.

Introduces ``FusedElemwise``, an ``OpFromGraph`` that wraps
``AdvancedSubtensor1`` + ``Elemwise`` + ``AdvancedIncSubtensor1`` / ``CAReduce``
subgraphs so the Numba backend can generate a single loop with indirect
indexing and inline accumulation, eliminating materialised intermediate arrays.
"""

from pytensor.compile import optdb
from pytensor.compile.builders import OpFromGraph
from pytensor.graph import node_rewriter
from pytensor.graph.rewriting.basic import GraphRewriter, dfs_rewriter
from pytensor.graph.rewriting.db import SequenceDB
from pytensor.graph.rewriting.unify import OpPattern
from pytensor.graph.utils import InconsistencyError
from pytensor.printing import op_debug_information
from pytensor.scalar.basic import AND, OR, XOR, Add, Composite, Maximum, Minimum, Mul
from pytensor.tensor.elemwise import CAReduce, DimShuffle, Elemwise
from pytensor.tensor.rewriting.elemwise import InplaceElemwiseOptimizer
from pytensor.tensor.shape import Reshape, shape_padright
from pytensor.tensor.subtensor import (
    AdvancedIncSubtensor,
    AdvancedIncSubtensor1,
    AdvancedSubtensor,
    AdvancedSubtensor1,
)
from pytensor.tensor.variable import TensorVariable


# CAReduce scalar ops whose reduction the Numba backend can fuse into the loop
# (those for which the codegen has an in-place accumulation: see
# ``accumulate_into_slice`` in the numba elemwise dispatch).
_REDUCE_SCALAR_OPS = (Add, Mul, Maximum, Minimum, AND, OR, XOR)


def _view_root(view_i, var):
    """Follow the destroy-handler view chain to the underlying buffer.

    ``view_i`` maps each view variable to the variable it directly views
    (e.g. ``SpecifyShape(x, ...) -> x``).  Two variables alias the same memory
    iff they resolve to the same root, so comparing roots — rather than the
    variables themselves — catches aliasing through intervening view ops.
    """
    while var in view_i:
        var = view_i[var]
    return var


def _unwrap_axis_swapped_subtensor1(fgraph, var):
    """Unwrap ``AdvancedSubtensor1`` with optional ``DimShuffle`` axis-swap.

    Detects two patterns (all intermediates must be single-client):

    - ``AdvancedSubtensor1(source, idx)`` → ``(source, idx, 0)``
    - ``DimShuffle{swap}(AdvancedSubtensor1(DimShuffle{swap}(source), idx))``
      → ``(source, idx, axis)`` where *axis* is the non-zero swapped axis.
    """
    if var.owner is None:
        return None

    # Bare AdvancedSubtensor1 on axis 0
    if isinstance(var.owner.op, AdvancedSubtensor1):
        if len(fgraph.clients[var]) != 1:
            return None
        return var.owner.inputs[0], var.owner.inputs[1], 0

    # Check for axis-swap DimShuffle wrapping AdvancedSubtensor1
    if not isinstance(var.owner.op, DimShuffle) or not var.owner.op.is_transpose:
        return None

    # Find the swapped axis: exactly two positions differ from identity
    order = var.owner.op.new_order
    swapped = [i for i, o in enumerate(order) if o != i]
    if len(swapped) != 2:
        return None
    ax_a, ax_b = swapped
    if order[ax_a] != ax_b or order[ax_b] != ax_a:
        return None
    axis = max(ax_a, ax_b)  # the non-zero axis (0 was swapped to axis)

    # Inner must be a single-client AdvancedSubtensor1
    asub1_out = var.owner.inputs[0]
    if len(fgraph.clients[asub1_out]) != 1:
        return None
    match asub1_out.owner_op_and_inputs:
        case AdvancedSubtensor1(), inner_ds_var, idx_var:
            pass
        case _:
            return None

    # AdvancedSubtensor1's input must be a single-client inverse DimShuffle (same swap)
    if len(fgraph.clients[inner_ds_var]) != 1:
        return None
    match inner_ds_var.owner_op_and_inputs:
        case DimShuffle(is_transpose=True, new_order=new_order), source:
            if new_order != tuple(order):
                return None
        case _:
            return None

    return source, idx_var, axis


@node_rewriter([OpPattern(DimShuffle, is_transpose=True)])
def undo_take_dimshuffle_for_fusion(fgraph, node):
    """Undo ``DimShuffle(AdvancedSubtensor1(DimShuffle(x), idx))`` -> ``AdvancedSubtensor(x, :, ..., idx, :, ...)``.

    The ``local_replace_AdvancedSubtensor`` specialize rewrite converts
    ``x[:, idx]`` into ``x.T[idx].T`` (axis-swap + AdvancedSubtensor1 +
    axis-swap).  This rewrite undoes that when the result feeds a single
    Elemwise, so ``FuseElemwise`` can absorb the indexing directly
    on the correct axis.

    See also ``undo_take_reshape_for_fusion`` which handles the analogous
    Reshape+flatten pattern for ND indices.
    """
    # Outer DimShuffle must be consumed only by a single Elemwise
    clients = fgraph.clients[node.outputs[0]]
    if len(clients) != 1:
        return None
    client_node, _client_idx = clients[0]
    if not isinstance(client_node.op, Elemwise):
        return None

    result = _unwrap_axis_swapped_subtensor1(fgraph, node.outputs[0])
    if result is None:
        return None
    source, idx_var, axis = result

    # Build AdvancedSubtensor: x[:, ..., idx, :, ...]
    idx_list = [slice(None)] * (axis + 1)
    idx_list[axis] = 0  # pointer to the single index variable
    new_out = AdvancedSubtensor(idx_list=idx_list)(source, idx_var)
    return [new_out]


@node_rewriter([Reshape])
def undo_take_reshape_for_fusion(fgraph, node):
    """Undo ``Reshape(AdvancedSubtensor1(x, flatten(idx)), shape)`` for ND indices.

    ``transform_take`` rewrites ``x[mat_idx]`` (ND integer index) into
    ``AdvancedSubtensor1(x, mat_idx.ravel()).reshape(mat_idx.shape + ...)``,
    possibly with DimShuffle axis-swaps for non-zero axes.  This rewrite
    undoes that so ``FuseElemwise`` can absorb the ND index directly.
    """
    [reshape_out] = node.outputs

    # Must feed a single Elemwise (or chain to one via another pre-fusion rewrite)
    clients = fgraph.clients[reshape_out]
    if len(clients) != 1:
        return None
    client_node, _ = clients[0]
    if not isinstance(client_node.op, Elemwise):
        return None

    result = _unwrap_axis_swapped_subtensor1(fgraph, node.inputs[0])
    if result is None:
        return None
    source, flat_idx, axis = result

    # The index input to AdvancedSubtensor1 must be Reshape{1}(mat_idx, [-1]) (flatten)
    if flat_idx.owner is None or not isinstance(flat_idx.owner.op, Reshape):
        return None
    if flat_idx.owner.op.ndim != 1:
        return None
    mat_idx = flat_idx.owner.inputs[0]
    if mat_idx.ndim < 2:
        return None

    # Build AdvancedSubtensor: source[:, ..., mat_idx, :, ...]
    src_ndim = source.type.ndim
    idx_list = [slice(None)] * src_ndim
    idx_list[axis] = 0  # pointer to the single index variable
    new_out = AdvancedSubtensor(idx_list=idx_list)(source, mat_idx)
    return [new_out]


fused_elemwise_optdb = SequenceDB()
optdb.register(
    # Predates the FusedElemwise rename; kept so existing .including/.excluding
    # calls targeting this name keep working.
    "fuse_indexed_into_elemwise",
    fused_elemwise_optdb,
    "numba",
    # symbolic_op_recognition is excluded from OpFromGraph inner-graph
    # compilation, preventing recursive fusion.
    "symbolic_op_recognition",
    # After inplace_elemwise (position=50.5) so we see final inplace patterns,
    # same position as other numba-specific rewrites (BlockwiseWithCoreShape).
    position=100,
)

fused_elemwise_optdb.register(
    "undo_take_dimshuffle_for_fusion",
    dfs_rewriter(undo_take_dimshuffle_for_fusion),
    "numba",
    position=0,
)

fused_elemwise_optdb.register(
    "undo_take_reshape_for_fusion",
    dfs_rewriter(undo_take_reshape_for_fusion),
    "numba",
    position=0.5,
)


class FusedElemwise(OpFromGraph):
    """Fuse indexed reads and updates into a single Elemwise iteration loop.

    Absorbs ``AdvancedSubtensor1`` (indexed reads on inputs) and
    ``AdvancedIncSubtensor1`` (indexed updates on outputs) into one loop,
    avoiding materialisation of intermediate arrays.

    Inner fgraph contains the unfused subgraph.
    Non-Numba backends run it as-is via ``OpFromGraph.perform``.
    The Numba backend generates a single loop with indirect indexing.

    Outer inputs are ordered as::

        [elemwise_inputs..., idx_0, idx_1, ..., update_target_0, ...]

    Parameters
    ----------
    indexed_inputs : tuple of ((tuple[int, ...], int) | None)
        One entry per index array k (at outer input position n_elemwise + k).
        ``None`` if index k has no read role (write-only).
        Otherwise ``(sources, source_axis)``:

        - ``sources``: which elemwise input positions read through this
          index.  Inputs that share the same ``(idx, axis)`` are grouped
          so the codegen loads the index once and reuses the indirect
          lookup for all of them.
          E.g. ``x[idx] + y[idx]`` sharing the same ``idx`` → ``(0, 1)``.
        - ``source_axis``: which axis of the source array is indexed
          (not the index array's own axes).
          E.g. ``x[:, idx]`` on a 3-D array → ``source_axis=1``.

        The grouping key is ``(idx_var, source_axis)``, so the same index variable
        on different axes produces separate entries.

        Examples::

            z = x[idx] + y[idx]          → [((0,1), 0)]
            z = x[idx_a] + y[idx_b]      → [((0,), 0), ((1,), 0)]
            write-only inc(tgt, v, idx)  → [None]

    indexed_outputs : tuple of ((tuple[int, ...], int, str) | None)
        One entry per index array k, parallel to ``indexed_inputs``.
        ``None`` if index k has no write role.
        Otherwise ``(sources, source_axis, mode)``:

        - ``sources``: which Elemwise output positions are written
          through this index into the update target buffer.
        - ``source_axis``: which target-array axis is indexed.
        - ``mode``: ``"inc"`` (accumulate) or ``"set"`` (overwrite).

        Examples::

            tgt[idx] += exp(x)   → indexed_outputs=[((0,), 0, "inc")]

    reduced_outputs : tuple of ((scalar_op, axes, identity, acc_dtype) | None)
        One entry per (inner Elemwise / outer op) output position.
        ``None`` if the output is not a reduction.
        Otherwise the output is the result of reducing the inner Elemwise output
        with ``CAReduce(scalar_op)`` over ``axes``:

        - ``scalar_op``: the commutative/associative binary scalar op of the
          reduction (e.g. ``add`` for sum, ``mul`` for prod, ``maximum`` for max).
        - ``axes``: tuple of reduced batch axes (in the inner Elemwise's dim space).
        - ``identity``: the reduction identity, used to seed the accumulator buffer.
        - ``acc_dtype``: dtype the accumulation is carried out in (the Numba loop
          accumulates in this dtype, then casts to the output dtype).

        The inner fgraph still holds a faithful ``CAReduce(Elemwise(...))`` so
        non-Numba backends evaluate correctly via ``OpFromGraph.perform``; only
        the Numba backend reads this spec to fuse the reduction into the loop.

        Examples::

            sum(exp(x))          → reduced_outputs=[(add, (0,), 0.0, "float64")]
    """

    def __init__(
        self,
        *args,
        indexed_inputs=(),
        indexed_outputs=(),
        reduced_outputs=(),
        **kwargs,
    ):
        self.indexed_inputs = indexed_inputs
        self.indexed_outputs = indexed_outputs
        self.reduced_outputs = reduced_outputs
        # A read buffer can occupy multiple input slots (e.g. read through
        # several indices); construct_nominal_fgraph dedupes those to one
        # nominal, leaving the extra slots as unused NominalVariables, which is
        # safe because reads don't destroy. Write targets always get their own
        # fresh inner input (see FuseElemwise) so a destroyed buffer is
        # never deduped onto a read source.
        super().__init__(*args, on_unused_input="ignore", accept_inplace=True, **kwargs)

    def __str__(self):
        elemwise_str = "Elemwise"
        for node in self.fgraph.apply_nodes:
            if isinstance(node.op, Elemwise):
                elemwise_str = str(node.op)
                break
        reductions = [
            f"{type(spec[0]).__name__.lower()}@{spec[1]}"
            for spec in self.reduced_outputs
            if spec is not None
        ]
        if reductions:
            return f"FusedElemwise{{{elemwise_str}, reduce[{', '.join(reductions)}]}}"
        return f"FusedElemwise{{{elemwise_str}}}"


@op_debug_information.register(FusedElemwise)
def _op_debug_information_FusedElemwise(op, node):
    info = {}

    n_idx = len(op.indexed_inputs)
    n_update_targets = sum(1 for e in op.indexed_outputs if e is not None)
    n_elemwise = len(node.inputs) - n_idx - n_update_targets

    # Annotate indexed-read inputs
    for k, entry in enumerate(op.indexed_inputs):
        if entry is None:
            continue
        sources, _source_axis = entry
        idx_label = f"idx_{k}"
        for src in sources:
            if src < len(node.inputs):
                info[node.inputs[src]] = f"indexed read ({idx_label})"

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
        sources, _source_axis, mode = entry
        buf_label = f"buf_{buf_counter}"
        buf_counter += 1
        idx_label = f"idx_{k}"

        target_pos = target_start + target_offset
        target_offset += 1
        if target_pos < len(node.inputs):
            info[node.inputs[target_pos]] = buf_label

        for out_idx in sources:
            if out_idx < len(node.outputs):
                info[node.outputs[out_idx]] = (
                    f"indexed {mode} ({buf_label}, {idx_label})"
                )

    # Annotate reduced outputs
    for out_idx, spec in enumerate(op.reduced_outputs):
        if spec is None or out_idx >= len(node.outputs):
            continue
        scalar_op, axes, _identity, acc_dtype = spec
        info[node.outputs[out_idx]] = (
            f"reduced[{type(scalar_op).__name__.lower()}] "
            f"over axes {axes} acc={acc_dtype}"
        )

    return {node: info}


class FuseElemwise(GraphRewriter):
    """Fuse indexed reads and indexed updates into Elemwise loops.

    Absorbs single-client ``AdvancedSubtensor1`` on inputs (indexed reads)
    and single-client ``AdvancedIncSubtensor1`` on outputs (indexed updates)
    into the Elemwise iteration, avoiding intermediate arrays.

    Supports multiple index arrays: e.g. ``x[idx_a] + y[idx_b]`` produces
    two index groups.  Index arrays are shared between reads and updates
    when they refer to the same variable.
    """

    @staticmethod
    def _extract_idx_axis_pairs(node, *, write=False):
        """Extract ``(idx_var, axis)`` pairs from an Advanced(Inc)Subtensor node.

        Returns a list of pairs, or ``None`` if the node uses non-consecutive
        advanced indexing, boolean indices, or mixed slice/integer patterns
        that we can't fuse.

        Parameters
        ----------
        write : bool
            If True, match write ops (AdvancedIncSubtensor variants).
            If False (default), match read ops (AdvancedSubtensor variants).
        """
        op = node.op
        if not write:
            if isinstance(op, AdvancedSubtensor1):
                return [(node.inputs[1], 0)]
            if isinstance(op, AdvancedSubtensor):
                n_skip = 1
            else:
                return None
        else:
            if isinstance(op, AdvancedIncSubtensor1):
                return [(node.inputs[2], 0)]
            if isinstance(op, AdvancedIncSubtensor):
                n_skip = 2
            else:
                return None

        if op.non_consecutive_adv_indexing(node):
            return None
        idx_vars = node.inputs[n_skip:]
        pairs = []
        for axis, entry in enumerate(op.idx_list):
            if isinstance(entry, slice):
                continue
            idx = idx_vars[entry]
            if not isinstance(idx, TensorVariable) or idx.type.dtype == "bool":
                return None
            pairs.append((idx, axis))
        return pairs or None

    @staticmethod
    def _duplicate_multi_client_outputs(node, multi_client_outs):
        """Add duplicate outputs for Elemwise results whose consumers must be split.

        Used when an output is both written/reduced and consumed directly, or is
        consumed by several reductions. Returns ``(new_node, dup_map)`` where
        *dup_map* maps each original output index to its duplicate position.
        """
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

        dup_map = {}
        for out_idx in sorted(multi_client_outs):
            dup_map[out_idx] = len(s_outputs)
            s_outputs.append(s_outputs[out_idx])

        new_scalar_op = Composite(s_inputs, s_outputs)
        # Duplicates are appended after the original outputs, so the node's
        # inplace pattern carries over unchanged (duplicates get no entries).
        new_node = Elemwise(new_scalar_op, dict(node.op.inplace_pattern)).make_node(
            *node.inputs
        )
        return new_node, dup_map

    @staticmethod
    def transpose_non_indexed_write_axes(node, write_targets):
        """Move excess leading non-indexed dims to the right of the write target.

        Only the leftmost non-indexed dims that aren't covered by the Elemwise
        loop are moved. Loop dims and indexed axes keep their relative order so
        the val's dim layout stays aligned with the write slice.

        Returns a list of ``(old_out, new_out)`` replacement pairs, or an
        empty list if no write target needed transposing.
        """
        replacements = []
        elemwise_batch_ndim = len(node.outputs[0].type.broadcastable)
        for update_node in write_targets.values():
            op = update_node.op
            target, val, *idx_vars = update_node.inputs

            idx_axes = [i for i, e in enumerate(op.idx_list) if e != slice(None)]
            n_indexed_axes = len(idx_axes)
            n_idx_dims = max(v.ndim for v in idx_vars)
            source_batch = elemwise_batch_ndim + n_indexed_axes - n_idx_dims
            if max(idx_axes) < source_batch:
                # Indexed axes already within batch dims, no transpose needed
                continue

            # Move excess leading non-indexed axes to the right
            non_idx_axes = [a for a in range(target.type.ndim) if a not in idx_axes]
            excess = target.type.ndim - source_batch
            excess_axes = non_idx_axes[:excess]
            non_excess_axes = [
                a for a in range(target.type.ndim) if a not in excess_axes
            ]
            perm = non_excess_axes + excess_axes
            target_t = target.dimshuffle(perm)

            # Pad val so it broadcasts with excess dims moved to the right.
            # Non-indexed axes can't be between indexed axes (_extract_idx_axis_pairs rejects non-consecutive indexing).
            val = shape_padright(val, excess)
            new_idx_list = [op.idx_list[perm[i]] for i in range(len(perm))]

            # Create new write node
            props = op._props_dict()
            props["idx_list"] = tuple(new_idx_list)
            new_inc = type(op)(**props)(target_t, val, *idx_vars)

            # Permute updated node so it behaves like original one
            inv_perm = [0] * len(perm)
            for i, p in enumerate(perm):
                inv_perm[p] = i
            new_update = new_inc.dimshuffle(inv_perm)

            replacements.append((update_node.outputs[0], new_update))
        return replacements

    def apply(self, fgraph):
        # Live reference to the destroy handler's view chain (mutated in place as
        # we rewrite); used to resolve read/write buffers to their roots for the
        # aliasing check below. replace_all_validate requires the destroy handler,
        # so it is present whenever a fusion could actually be rejected for aliasing.
        destroy_handler = getattr(fgraph, "destroy_handler", None)
        view_i = destroy_handler.view_i if destroy_handler is not None else {}

        worklist = list(reversed(fgraph.toposort()))
        while worklist:
            node = worklist.pop()
            if not isinstance(node.op, Elemwise):
                continue
            if node not in fgraph.apply_nodes:
                continue

            idx_groups = {}  # (idx_var, axis) -> (reads: list[int], writes: list[int])

            # Roots of the buffers we read through, used below to skip fusing a
            # write whose target buffer aliases one of them (see aliasing check).
            read_source_roots = set()

            # Find indexed reads to fuse: single client AdvancedSubtensor(1)
            for i, inp in enumerate(node.inputs):
                inp_node = inp.owner
                if inp_node is None or any(
                    c is not node for c, _ in fgraph.clients[inp]
                ):
                    continue
                idx_axis_pairs = self._extract_idx_axis_pairs(inp_node)
                if idx_axis_pairs is None:
                    continue
                read_source_roots.add(_view_root(view_i, inp_node.inputs[0]))
                for idx_axis_pair in idx_axis_pairs:
                    if idx_axis_pair not in idx_groups:
                        idx_groups[idx_axis_pair] = ([], [])
                    idx_groups[idx_axis_pair][0].append(i)

            # For indexed writes to fuse: single client AdvancedIncSubtensor(1)
            # The write may be separated from the output by a right expand_dims DimShuffle
            # All indexed write axes have to overlap and not broadcast the core Elemwise loop
            # Our current vectorize codegen can't produce write only loops that don't force
            # the recomputation of the core function in every step.
            write_targets = {}  # out_idx -> update_node
            must_transpose_write_axes = False
            for out_idx, out in enumerate(node.outputs):
                clients = fgraph.clients[out]
                # Allow right expand_dims between elemwise and write
                # buffer[idx].set(f(x)[..., None, None])
                # These will be absorbed by broadcasting the result at each iteration of the loop
                # into the sliced non-scalar indexed buffer, making use of vectorize_codegen with fake "core output dims"
                # We actually move left broadcasted dims to the right below
                right_pad = 0
                if (
                    len(clients) == 1
                    and isinstance((ds_op := clients[0][0].op), DimShuffle)
                    and ds_op.is_right_expand_dims
                    and len(fgraph.clients[(ds_out := clients[0][0].outputs[0])]) == 1
                ):
                    right_pad = len(ds_op.augment)
                    clients = fgraph.clients[ds_out]
                inc_clients = [
                    (c, ci)
                    for c, ci in clients
                    if ci == 1
                    and isinstance(c.op, AdvancedIncSubtensor1 | AdvancedIncSubtensor)
                ]
                if len(inc_clients) != 1:
                    # TODO: support multiple writes from the same Elemwise output via Composite duplication
                    continue
                [(client_node, _)] = inc_clients
                idx_axis_pairs = self._extract_idx_axis_pairs(client_node, write=True)
                if idx_axis_pairs is None:
                    continue

                target, _, *idx_vars = client_node.inputs

                # Fusing an in-place write whose target is a buffer we also read
                # through (same root) makes it two distinct inputs of one node, one
                # read and one destroyed: the destroy handler rejects that aliasing.
                # Leave such writes unfused (the reads still fuse, no added copy).
                # Non-in-place writes are copied below, so their copy breaks the alias.
                if client_node.op.inplace and _view_root(view_i, target) in (
                    read_source_roots
                ):
                    continue

                write_bcast = AdvancedSubtensor(idx_list=client_node.op.idx_list)(
                    target, *idx_vars
                ).type.broadcastable
                indexed_write_bcast = (
                    write_bcast[: len(write_bcast) - right_pad]
                    if right_pad
                    else write_bcast
                )
                left_pad = min(a for _, a in idx_axis_pairs)
                if out.type.ndim + left_pad < len(indexed_write_bcast):
                    # out does not cover all indexed write dims
                    continue

                if any(
                    ob and not iwb
                    for ob, iwb in zip(
                        reversed(out.type.broadcastable), reversed(indexed_write_bcast)
                    )
                ):
                    # TODO: support broadcast on non-indexed dims by squeezing them out of the Elemwise first
                    continue

                if len(indexed_write_bcast) > out.type.ndim:
                    must_transpose_write_axes = True

                for idx_axis_pair in idx_axis_pairs:
                    if idx_axis_pair not in idx_groups:
                        idx_groups[idx_axis_pair] = ([], [])
                    idx_groups[idx_axis_pair][1].append(out_idx)
                write_targets[out_idx] = client_node

            # Find reductions to fuse: an Elemwise output whose sole client is an
            # eligible CAReduce.  Outputs that are write targets are excluded (an
            # output can't be both an indexed write and a reduction).  Outputs
            # with extra (non-reduce) clients are handled by duplication below.
            reduced_outputs = {}  # out_idx -> car_node
            extra_reduction = (
                None  # (out_idx, car_node) when reductions share an output
            )
            for out_idx, out in enumerate(node.outputs):
                if out_idx in write_targets:
                    continue
                car_clients = [
                    c
                    for c, _ in fgraph.clients[out]
                    if isinstance(c.op, CAReduce)
                    and isinstance(c.op.scalar_op, _REDUCE_SCALAR_OPS)
                ]
                if not car_clients:
                    continue
                if len(car_clients) > 1:
                    extra_reduction = (out_idx, car_clients[1])
                    break
                reduced_outputs[out_idx] = car_clients[0]

            # An output consumed by several eligible reductions: peel one reduction
            # per pass onto a duplicate output, until each reduction has its own copy
            if extra_reduction is not None:
                out_idx, car_node = extra_reduction
                new_node, dup_map = self._duplicate_multi_client_outputs(
                    node, {out_idx}
                )
                replacements = list(
                    zip(node.outputs, new_node.outputs[: len(node.outputs)])
                )
                new_reduced = car_node.op(new_node.outputs[dup_map[out_idx]])
                replacements.append((car_node.outputs[0], new_reduced))
                fgraph.replace_all(
                    replacements, reason="fuse_elemwise_split_multi_reductions"
                )
                worklist.append(new_node)
                continue

            if not idx_groups and not reduced_outputs:
                continue

            if must_transpose_write_axes:
                replacements = self.transpose_non_indexed_write_axes(
                    node, write_targets
                )
                assert replacements
                fgraph.replace_all(
                    replacements,
                    reason="fuse_elemwise_move_write_axes",
                )
                worklist.append(node)
                continue

            # If a reduced output also feeds non-reduce consumers, duplicate it via
            # Composite so the reduction consumes the duplicate while the original
            # stays materialised for the other consumers. We still fuse the
            # reduction loop, even if the full output must also be produced.
            # This runs before the strip-inplace pass below, so any reduced output
            # reaching it is sole-client (truly de-materialized) and an inplace on a
            # materialized original survives the fusion.
            def _has_non_reduce_clients(out_idx):
                car_node = reduced_outputs[out_idx]
                return any(
                    c is not car_node for c, _ in fgraph.clients[node.outputs[out_idx]]
                )

            if reduce_and_direct_use_outs := {
                out_idx
                for out_idx in reduced_outputs
                if _has_non_reduce_clients(out_idx)
            }:
                new_node, dup_map = self._duplicate_multi_client_outputs(
                    node, reduce_and_direct_use_outs
                )
                replacements = list(
                    zip(node.outputs, new_node.outputs[: len(node.outputs)])
                )
                for out_idx, dup_idx in dup_map.items():
                    car_node = reduced_outputs[out_idx]
                    new_reduced = car_node.op(new_node.outputs[dup_idx])
                    replacements.append((car_node.outputs[0], new_reduced))
                fgraph.replace_all(
                    replacements,
                    reason="fuse_reduce_and_direct_outputs",
                )
                worklist.append(new_node)
                continue

            # If any indexed-write output also has other consumers,
            # duplicate it via Composite so the write replaces the duplicate
            # while the original stays available for non-write consumers.
            # We still avoid one extra write loop,
            # even if we can't skip the output materialization altogether.
            # Like the reduce duplication above, this runs before the strip-inplace
            # pass, so an inplace on the materialized original survives the fusion.
            def _has_non_write_clients(out_idx):
                update = write_targets[out_idx]
                for c, _ in fgraph.clients[node.outputs[out_idx]]:
                    if c is update:
                        continue
                    # Look through shape_padright from write normalization
                    if (
                        isinstance(c.op, DimShuffle)
                        and c.op.is_right_expand_dims
                        and len(ds_clients := fgraph.clients[c.outputs[0]]) == 1
                        and ds_clients[0][0] is update
                    ):
                        continue
                    return True
                return False

            if write_and_direct_use_outs := {
                out_idx for out_idx in write_targets if _has_non_write_clients(out_idx)
            }:
                new_node, dup_map = self._duplicate_multi_client_outputs(
                    node, write_and_direct_use_outs
                )
                replacements = list(
                    zip(node.outputs, new_node.outputs[: len(node.outputs)])
                )
                for out_idx, dup_idx in dup_map.items():
                    update_node = write_targets[out_idx]
                    new_update_out = update_node.op(
                        update_node.inputs[0],
                        new_node.outputs[dup_idx],
                        *update_node.inputs[2:],
                    )
                    replacements.append((update_node.outputs[0], new_update_out))
                fgraph.replace_all(
                    replacements,
                    reason="fuse_elemwise_write_and_direct_outputs",
                )
                worklist.append(new_node)
                continue

            indexed_reads = {i for reads, _ in idx_groups.values() for i in reads}

            # If any inplace targets an indexed-read input, or claims an output that is
            # not materialized as a plain output — a fused reduction (reusing the input
            # buffer as the reduce accumulator would skip the identity init and
            # accumulate on top of stale input values) or an indexed write (the loop
            # writes to the write buffer instead, so the input destruction would happen
            # only in the Python-mode fallback, undeclared by the outer destroy map) —
            # strip and re-run inplace with those inputs protected and outputs excluded.
            # The duplications above ran first, so such outputs here are sole-client.
            if any(
                inp_idx in indexed_reads for inp_idx in node.op.inplace_pattern.values()
            ) or any(
                out_idx in reduced_outputs or out_idx in write_targets
                for out_idx in node.op.inplace_pattern
            ):
                stripped_node = Elemwise(node.op.scalar_op).make_node(*node.inputs)
                fgraph.replace_all(
                    zip(node.outputs, stripped_node.outputs),
                    reason="fuse_elemwise_strip_inplace",
                )
                optimizer = InplaceElemwiseOptimizer()
                protected = optimizer._get_protected_inputs(fgraph)
                protected.update(stripped_node.inputs[i] for i in indexed_reads)
                # Candidates are plain-Elemwise (output, input) pairs; exclude
                # outputs the fusion is about to consume as indexed writes
                candidate_pairs = [
                    pair
                    for pair in optimizer.filter_candidate_pairs(
                        fgraph, stripped_node, protected
                    )
                    if pair[0][0] not in reduced_outputs
                    and pair[0][0] not in write_targets
                ]
                # try_inplace_on_node does its own fgraph.replace_all internally,
                # so the returned node is already in the fgraph
                new_inplace_node = optimizer.try_inplace_on_node(
                    fgraph,
                    stripped_node,
                    candidate_pairs=candidate_pairs,
                    reason="fuse_elemwise_inplace_read_buffers",
                )
                worklist.append(new_inplace_node)
                continue

            idx_vars = [idx for idx, _axis in idx_groups]

            # The strip-inplace pass above guarantees that reduced and indexed-write
            # outputs carry no inplace
            assert not any(
                out_idx in reduced_outputs or out_idx in write_targets
                for out_idx in node.op.inplace_pattern
            )
            fgraph_destroy_map = {
                out_idx: [inp_idx]
                for out_idx, inp_idx in node.op.inplace_pattern.items()
            }

            # Fgraph inputs: substitute indexed sources back to their
            # pre-subtensor arrays, append index arrays and update targets.
            fgraph_inputs = [
                inp.owner.inputs[0] if i in indexed_reads else inp
                for i, inp in enumerate(node.inputs)
            ] + idx_vars

            # Non-inplace write targets need a copy so the original isn't destroyed
            # Elemwise will always destroy the write buffers inplace afterwards.
            copy_positions = set()
            # Real outer buffer bound to each write-target input position. The inner
            # fgraph uses a fresh variable there (see below), so the actual buffer
            # (or its copy) is bound only at the outer call.
            outer_write_targets = {}

            # Inner fgraph outputs: Elemwise outputs, with write targets
            # replaced by their AdvancedIncSubtensor result
            fgraph_outputs = list(node.outputs)
            for out_idx, update_node in sorted(write_targets.items()):
                target = update_node.inputs[0]

                # Use a fresh inner input for the write target so it stays distinct
                # from a read source that is the same variable. Otherwise
                # construct_nominal_fgraph would dedupe the two into one nominal and
                # the copy bound below would be dead inside the inner graph, leaving
                # the inner write to destroy the read buffer instead of the copy.
                inner_target = target.type()
                fgraph_inputs.append(inner_target)
                target_pos = len(fgraph_inputs) - 1
                outer_write_targets[target_pos] = target

                if not update_node.op.inplace:
                    copy_positions.add(target_pos)

                # Build the indexed write for the inner fgraph, in-place on the fresh
                # inner target buffer (Elemwise destroys the write buffers anyway).
                if update_node.op.inplace:
                    write_out = update_node.op(inner_target, *update_node.inputs[1:])
                else:
                    props = update_node.op._props_dict()
                    props["inplace"] = True
                    inplace_op = type(update_node.op)(**props)
                    write_out = inplace_op(
                        inner_target, node.outputs[out_idx], *update_node.inputs[2:]
                    )

                fgraph_outputs[out_idx] = write_out
                fgraph_destroy_map[out_idx] = [target_pos]

            # Inner fgraph reduced outputs: wrap the Elemwise output in the real
            # CAReduce so non-Numba backends compute it faithfully via perform.
            # reduced_spec carries the (scalar_op, axes, identity, acc_dtype) the
            # Numba backend reads to fuse the reduction into the loop instead.
            reduced_spec_by_idx = {}
            for out_idx, car_node in sorted(reduced_outputs.items()):
                car_op = car_node.op
                ndim = node.outputs[out_idx].type.ndim
                axes = (
                    tuple(range(ndim))
                    if car_op.axis is None
                    else tuple(sorted(car_op.axis))
                )
                acc_dtype = (
                    car_op.acc_dtype
                    if car_op.acc_dtype is not None
                    else car_node.outputs[0].type.dtype
                )
                reduced_spec_by_idx[out_idx] = (
                    car_op.scalar_op,
                    axes,
                    car_op.scalar_op.identity,
                    acc_dtype,
                )
                fgraph_outputs[out_idx] = car_op(node.outputs[out_idx])

            reduced_spec = (
                tuple(
                    reduced_spec_by_idx.get(out_idx)
                    for out_idx in range(len(node.outputs))
                )
                if reduced_outputs
                else ()
            )

            # indexed_inputs_spec: ((read_positions, axis) | None, ...)
            # indexed_outputs_spec: ((write_positions, axis, "inc"|"set") | None, ...)
            indexed_inputs_spec = tuple(
                (tuple(reads), axis) if reads else None
                for (_, axis), (reads, _) in idx_groups.items()
            )
            indexed_outputs_spec = tuple(
                (
                    tuple(writes),
                    key[1],
                    "set" if write_targets[writes[0]].op.set_instead_of_inc else "inc",
                )
                if writes
                else None
                for key, (_, writes) in idx_groups.items()
            )

            outer_inputs = []
            for i, inp in enumerate(fgraph_inputs):
                val = outer_write_targets.get(i, inp)
                outer_inputs.append(val.copy() if i in copy_positions else val)

            new_outs = FusedElemwise(
                fgraph_inputs,
                fgraph_outputs,
                destroy_map=fgraph_destroy_map,
                indexed_inputs=indexed_inputs_spec,
                indexed_outputs=indexed_outputs_spec,
                reduced_outputs=reduced_spec,
            )(*outer_inputs, return_list=True)

            replacements = []
            for out_idx in range(len(node.outputs)):
                if out_idx in write_targets:
                    replacements.append(
                        (write_targets[out_idx].outputs[0], new_outs[out_idx])
                    )
                elif out_idx in reduced_outputs:
                    replacements.append(
                        (reduced_outputs[out_idx].outputs[0], new_outs[out_idx])
                    )
                else:
                    replacements.append((node.outputs[out_idx], new_outs[out_idx]))

            # Safety net for aliasing we didn't anticipate above (e.g. a write
            # target that is a view of another input): skip this node rather than
            # aborting the whole pass and leaving later nodes unfused.
            try:
                fgraph.replace_all_validate(
                    replacements,
                    reason="fuse_indexed_into_elemwise",
                )
            except InconsistencyError:
                continue


fused_elemwise_optdb.register(
    "fuse_elemwise",
    FuseElemwise(),
    "numba",
    position=1,
)
