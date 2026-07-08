"""Backend inner-graph rewrites for ``Scan``.

``rewrite_scan_inner_graph`` bakes a node's inner graph for the active backend;
the ``scan_inner_graph`` pass in ``pytensor.scan.rewriting.db`` drives it through
``pytensor.compile.rewriting.rewrite_inner_graph``.
"""

from functools import singledispatch

from pytensor.compile.aliasing import (
    add_supervisor_to_fgraph,
    alias_root,
    insert_deepcopy,
)
from pytensor.compile.io import In, Out
from pytensor.compile.mode import get_mode
from pytensor.configdefaults import config
from pytensor.graph.features import NoOutputFromInplace, NoOutputInplaceOnInput
from pytensor.link.basic import PerformLinker
from pytensor.link.c.basic import CLinker, OpWiseCLinker
from pytensor.link.jax.linker import JAXLinker
from pytensor.link.mlx.linker import MLXLinker
from pytensor.link.numba.linker import NumbaLinker
from pytensor.link.pytorch.linker import PytorchLinker
from pytensor.link.vm import VMLinker


@singledispatch
def rewrite_scan_inner_graph(linker, op, node, inner, *, mode):
    """Rewrite a ``Scan`` inner graph (in place) for ``linker``'s backend."""
    raise NotImplementedError(
        f"Linker {type(linker).__name__} has not registered a Scan inner-graph rewrite"
    )


def scan_inner_optimizer(op, mode):
    """Optimizer to run on a ``Scan`` inner graph.

    ``mode.optimizer`` unless the op carries a (deprecated) custom ``mode``, in
    which case that mode is combined with the active linker's required/incompatible
    rewrites so backend must-have ops still apply.
    """
    custom_mode = getattr(op, "mode", None)
    if custom_mode is None:
        return mode.optimizer
    linker = mode.linker
    return (
        get_mode(custom_mode)
        .including(*linker.required_rewrites)
        .excluding(*linker.incompatible_rewrites)
        .optimizer
    )


@rewrite_scan_inner_graph.register(VMLinker)
@rewrite_scan_inner_graph.register(PerformLinker)
@rewrite_scan_inner_graph.register(CLinker)
@rewrite_scan_inner_graph.register(OpWiseCLinker)
def cvm_rewrite_scan_inner_graph(linker, op, node, inner, *, mode):
    """Bake the tap inplace and protect the tap outputs ``Scan.prepare_fgraph`` does.

    Leaves the boundary deepcopies to ``Scan.fn`` (which re-optimizes the inner
    graph at ``make_thunk``).
    """
    # Grant aliasing/destruction of immediately-discarded taps; protect the same tap
    # outputs ``Scan.prepare_fgraph`` does so the baked inplace never makes a
    # protected output the result of a destroy-map node -- otherwise ``Scan.fn``,
    # which re-attaches ``NoOutputFromInplace`` and re-optimizes, would reject it.
    # ``prepare_fgraph`` only installs that protection with output preallocation.
    destroyable = op.inner_destroyable_inputs(node.inputs, inner.inputs)
    # Unlike numba, the C/VM backends don't copy an untraced state on the first
    # iteration, so they may destroy it only when the Scan owns the outer buffer;
    # otherwise the in-place step would corrupt the caller's input. Drop the unowned
    # untraced states from the destroyable set.
    destroyable -= set(op.inner_untraced_sit_sot(inner.inputs)) - (
        op.inner_owned_untraced_sit_sot(inner.inputs)
    )
    input_specs = [In(x, borrow=True, mutable=x in destroyable) for x in inner.inputs]
    add_supervisor_to_fgraph(fgraph=inner, input_specs=input_specs, accept_inplace=True)
    if config.scan__allow_output_prealloc:
        inner.attach_feature(NoOutputFromInplace(op.protected_inner_out_idxs()))
    scan_inner_optimizer(op, mode).rewrite(inner)


@rewrite_scan_inner_graph.register(JAXLinker)
@rewrite_scan_inner_graph.register(PytorchLinker)
@rewrite_scan_inner_graph.register(MLXLinker)
def functional_rewrite_scan_inner_graph(linker, op, node, inner, *, mode):
    """Structurally optimize the inner graph for the functional JIT backends."""
    scan_inner_optimizer(op, mode).rewrite(inner)


def find_overwritten_reads(op, outer_inputs, inner_inputs):
    """Inner traced reads a same-iteration write may overwrite, split by certainty.

    A traced read is "overwritten" when a same-iteration write reuses its physical
    buffer slot. With buffer length ``L``, output tap ``o_out`` writes the slot read
    by input tap ``o_in`` iff ``(o_out - o_in) % L == 0``. Since ``L`` is always
    ``>= reach`` (the minimal admissible length, the oldest lookback), only two gaps
    can trigger it:

    - gap 0 (``o_out == o_in``): the recurrence writes the very slot it just read
      (mit_mot accumulators: read ``g[k]``, write ``g[k] + delta`` back). Holds for
      ANY ``L``, so the read is *certainly* overwritten this iteration.
    - gap == reach: a write lands just past the oldest read; in a buffer truncated to
      its minimum (``L == reach``) it wraps onto that just-discarded oldest read. So
      the read is overwritten only at ``L == reach``: certain when the static length
      says so, merely possible when the length is statically unknown, impossible for
      any larger ``L``. (A write landing further ahead would store onto a slot still
      to be read -- an invalid recurrence we need not consider, just as we don't
      consider ``L < reach``.)

    A certainly-overwritten read is dead once consumed this iteration, so the inner
    function may destroy it in place. A possibly-overwritten read must not be destroyed
    (it may still be live at a larger length), but no output may alias it either, since
    a same-iteration overwrite would corrupt the alias before it is stored. Depends on
    the *outer* buffer's static length, hence a per-node property.
    """

    def find(grouped_inner, outers, in_slices_seq, out_slices_seq):
        certain, possible = [], []
        for inner_vars, outer, in_slices, out_slices in zip(
            grouped_inner, outers, in_slices_seq, out_slices_seq, strict=True
        ):
            reach = -min(0, min(in_slices))  # minimal admissible buffer length
            in_offsets = [reach + t for t in in_slices]
            out_offsets = [reach + t for t in out_slices]
            static_len = outer.type.shape[0]
            for v, o_in in zip(inner_vars, in_offsets, strict=True):
                gaps = {o_out - o_in for o_out in out_offsets}
                if 0 in gaps:
                    certain.append(v)
                elif reach in gaps:
                    if static_len == reach:
                        certain.append(v)
                    elif static_len is None:
                        possible.append(v)
        return certain, possible

    info = op.info
    mit_mot_certain, mit_mot_possible = find(
        op.inner_mitmot_grouped(inner_inputs),
        op.outer_mitmot(outer_inputs),
        info.mit_mot_in_slices,
        info.mit_mot_out_slices,
    )
    mit_sot_certain, mit_sot_possible = find(
        op.inner_mitsot_grouped(inner_inputs),
        op.outer_mitsot(outer_inputs),
        info.mit_sot_in_slices,
        [(0,)] * info.n_mit_sot,
    )
    sit_sot_certain, sit_sot_possible = find(
        [[v] for v in op.inner_sitsot(inner_inputs)],
        op.outer_sitsot(outer_inputs),
        info.sit_sot_in_slices,
        [(0,)] * info.n_sit_sot,
    )
    certain = {*mit_mot_certain, *mit_sot_certain, *sit_sot_certain}
    possible = {*mit_mot_possible, *mit_sot_possible, *sit_sot_possible}
    return certain, possible


@rewrite_scan_inner_graph.register(NumbaLinker)
def numba_rewrite_scan_inner_graph(linker, op, node, inner, *, mode):
    """Bake inplace, alias bans and boundary deepcopies for the numba backend.

    Numba is the only backend that consumes inner *tap* inplace, so the memory
    model that governs it lives here: the rewrite bakes the inplace, bans
    destroying reads the loop overwrites, and inserts the boundary deepcopies,
    leaving ``op.fgraph`` ready for ``numba_funcify_Scan`` to funcify with no
    further graph work.

    Memory-aliasing contract
    ------------------------
    Scan defines a loop over an inner function with signature:
        (*sequences[idx], *traced[idx], *untraced, *non_sequences)
        -> (*traced_updates[idx], *untraced_updates)

    Traced variables are read from an indexed circular buffer at every iteration,
    and the updates stored (copied) back to it immediately after. Untraced variables
    are carried by reference, with each update becoming the next iteration's input.

    Scan is sometimes allowed to destroy/alias the outer traced and untraced variables,
    but never sequences and non-sequences. Specifically, outer untraced variables can be
    destroyed (destroy_map, opt in) or aliased (view_map, default). Traced variables can be
    destroyed (destroy_map, opt in), but otherwise not alias (never in view_map).
    Note that destroy permission implies alias permission, but not the other way around.

    Scan is not allowed to return outputs that alias each other, unless they were already
    aliased from the outside, and it was itself allowed to alias/destroy them. This means
    PyTensor already gauged it was safe to destroy/alias them.

    Scan has some freedom in how this outer contract is respected. If needed, it can
    deepcopy the outer inputs once at the start, or make sure any aliased output
    the inner function returns is properly copied before the final return.

    Internally, Scan also has total control over the boundary memory management of the
    inner function: it grants the permissions to destroy or alias the inner loop inputs,
    and whether the inner outputs may alias each other. This inner boundary is distinct
    from the outer contract above, and it is Scan's responsibility to choose an inner
    strategy that produces correct results while still respecting the outer contract.

    Memory-aliasing strategy
    ------------------------
    Traced variables
    ~~~~~~~~~~~~~~~~
    Traced variables are deep-copied once at the start if they are not in the destroy_map.

    Because every inner trace update is copied back to the buffer immediately, the inner function
    is allowed to alias (but not destroy) the sequence reads, non_sequences, as well as the
    traced and untraced inputs or updates, when producing the traced updates.

    A special case occurs when the indexed reads will be immediately overwritten by the updates
    in the same loop iteration. For single output taps (mit-sot, sit-sot) this can only
    happen when the circular buffer is truncated to its minimum legal length.
    For mit-mot this can also happen without any buffer loop-around.
    In either case, traced updates are not allowed to alias those traced reads,
    as they may otherwise be corrupted if the reads are updated before they were copied to their own buffer.

    On the plus side, when this happens, the inner function is granted permission to destroy these
    immediately-to-be discarded reads, as long as the returned updates do not themselves alias them.

    The alias-restriction and destroy-permission caused by the loop-around behavior are derived from the
    buffer's static length:
        * known large enough: no overwrite is possible, neither alias restriction nor destroy permission applies;
        * length unknown: the loop-around overwrite can't be ruled out, alias restricted but not granted destroy permission;
        * known minimal: the overwrite is certain, alias restricted but granted destroy permission.

    Untraced variables
    ~~~~~~~~~~~~~~~~~~
    Untraced variables are deep-copied once at the start if they are not in destroy_map
    and the inner function destroys them.

    Untraced updates are allowed to alias their own untraced inputs (which happens when n_steps=0)
    or when the inner function update naturally alias the input (eg, o = i; o = i.T; o = i[::-1]).

    Because the last untraced updates are returned as is, the inner function is not allowed to
    alias sequences, non_sequences, or other untraced inputs and outputs (violates the outer alias restriction).
    Untraced updates are not allowed to alias traced reads (risks corruption by subsequent overwrites),
    but can alias traced updates, since the immediate copy to their buffer that follows, will break the alias.

    Because untraced inputs are immediately discarded (and protected from alias with other updates),
    the inner function is always granted permission to destroy them. It can do so from any computation,
    not only the one producing the matching untraced update.

    Controlling inner graph alias
    -----------------------------
    PyTensor allows initial graphs to contain arbitrary (non-destructive) aliasing.
    Alias at the boundary (output aliasing an input or another output) is controlled via
    targeted deepcopies at the end (the ``insert_deepcopy`` helper).

    In contrast, destruction is usually NOT allowed to be present in initial graphs.
    Destructive alias at the boundary is controlled here, with the following features:
        * Supervisor: Checks whether any protected input are destroyed
        * NoOutputInplaceOnInput: Checks whether an output is destroying a potentially
          overwritten read (protected inputs are already covered by Supervisor)
    Inside the boundary:
        * DestroyHandler: Checks whether a consistent ordering exists for the destruction/view chains,
          i.e., every read runs before its buffers' destruction and the chain has no cycle.
    These features can veto (undo) any rewrite that would violate their spec.
    They CANNOT fix violations that already existed in the initial graph.
    """
    certain_overwritten, possible_overwritten = find_overwritten_reads(
        op, node.inputs, inner.inputs
    )

    # Grant the inner function the right to alias (borrow=True) all inputs and to
    # destroy (mutable=True) reads that are immediately discarded: the always-discarded
    # reads from ``inner_destroyable_inputs`` (truncated sit_sot/mit_sot and every
    # untraced input) plus the certainly-overwritten reads (dead once consumed this
    # iteration). The certain set is what extends destruction to mit_mot, which the
    # shared ``inner_destroyable_inputs`` (also used by the C backend) leaves out.
    discarded = (
        op.inner_destroyable_inputs(node.inputs, inner.inputs) | certain_overwritten
    )
    input_specs = [In(x, borrow=True, mutable=x in discarded) for x in inner.inputs]
    add_supervisor_to_fgraph(fgraph=inner, input_specs=input_specs, accept_inplace=True)

    # No output may alias a (certainly or possibly) overwritten read: a same-iteration
    # overwrite could corrupt the alias before it is stored. Ban that inplace so the
    # inner graph allocates a fresh buffer instead; the deepcopy below still breaks any
    # remaining non-destructive alias of an overwritten read.
    overwritten_reads = certain_overwritten | possible_overwritten
    if overwritten_reads:
        in_pos = {v: i for i, v in enumerate(inner.inputs)}
        inner.attach_feature(
            NoOutputInplaceOnInput([in_pos[v] for v in overwritten_reads])
        )

    scan_inner_optimizer(op, mode).rewrite(inner)

    # Post-patch the alias contract with targeted deepcopies. A traced update may
    # borrow unless it aliases an overwritten read. An untraced update may borrow if
    # it is the FIRST viewer of a freshly produced buffer (or its own untraced
    # input) and not an alias of another untraced update; any other alias (sequence
    # reads, traced reads, non-sequences, other untraced inputs/updates) is copied.
    own_untraced_input = dict(
        zip(
            op.inner_untraced_sit_sot_outs(inner.outputs),
            op.inner_untraced_sit_sot(inner.inputs),
            strict=True,
        )
    )
    untraced_outs = set(own_untraced_input)
    seen_untraced_roots = set()
    output_specs = []
    for update in inner.outputs:
        root = alias_root(update)
        if update in untraced_outs:
            borrow = (
                root.owner is not None or root is own_untraced_input[update]
            ) and root not in seen_untraced_roots
            if borrow:
                seen_untraced_roots.add(root)
        else:
            borrow = root not in overwritten_reads
        output_specs.append(Out(update, borrow=borrow))
    insert_deepcopy(inner, wrapped_inputs=input_specs, wrapped_outputs=output_specs)
