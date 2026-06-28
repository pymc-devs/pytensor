from hashlib import sha256
from textwrap import dedent, indent

import numpy as np
from numba import types
from numba.extending import overload

from pytensor.compile.aliasing import (
    add_supervisor_to_fgraph,
    alias_root,
    insert_deepcopy,
)
from pytensor.compile.io import In, Out
from pytensor.compile.mode import NUMBA, get_mode
from pytensor.graph.features import NoOutputInplaceOnInput
from pytensor.link.numba.cache import compile_numba_function_src
from pytensor.link.numba.dispatch import basic as numba_basic
from pytensor.link.numba.dispatch.basic import (
    numba_funcify_and_cache_key,
    register_funcify_and_cache_key,
)
from pytensor.link.numba.dispatch.compile_ops import numba_deepcopy
from pytensor.link.numba.dispatch.string_codegen import create_tuple_string
from pytensor.scan.op import Scan


def idx_to_str(
    array_name: str,
    offset: int,
    size: str | None = None,
    idx_symbol: str = "i",
    allow_scalar=False,
) -> str:
    assert offset >= 0
    if offset > 0:
        indices = f"{idx_symbol} + {offset}"
    else:
        indices = idx_symbol

    if size:
        # TODO FIXME: The `Scan` `Op` should tell us which outputs are computed
        # in this way.  We shouldn't have to waste run-time efforts in order to
        # compensate for this poor `Op`/rewrite design and implementation.
        indices = f"({indices}) % {size}"

    if allow_scalar:
        return f"{array_name}[{indices}]"
    else:
        return f"np.asarray({array_name}[{indices}])"


@overload(range)
def array0d_range(x):
    if isinstance(x, types.Array) and x.ndim == 0:

        def range_arr(x):
            return range(x.item())

        return range_arr


@register_funcify_and_cache_key(Scan)
def numba_funcify_Scan(op: Scan, node, **kwargs):
    """Generate a Numba implementation of a `Scan` loop.

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
    Untraced updates are not allowed to alias traced reads (risks corruption by subsequent overwrittes),
    but can alias traced updates, since the immediate copy to their buffer that follows, will break the alias.

    Because untraced inputs are immediately discarded (and protected from alias with other updates),
    the inner function is always granted permission to destroy them. It can do so from any computation,
    not only the one producing the matching untraced update.

    Controlling inner graph alias
    -----------------------------
    PyTensor allows initial graphs to contain arbitrary (non-destructive) aliasing.
    Alias at the boundary (output aliasing an input or another output) is controlled via
    targeted deepcopies at the end (using the insert_deepcopy helper).

    In contrast, destruction is usually NOT allowed to be present in initial graphs.
    Destructive alias at the boundary is controlled during rewrites, with the following features:
        * Supervisor: Checks whether any protected input are destroyed
        * NoOutputInplaceOnInput: Checks whether an output is destroying a non-protected input
          (protected inputs are already covered by Supervisor)
    Inside the boundary:
        * DestroyHandler: Checks whether a consistent ordering exists for the destruction/view chains,
          i.e., every read runs before its buffers' destruction and the chain has no cycle.
    These features can veto (undo) any rewrite that would violate their spec.
    They CANNOT fix violations that already existed in the initial graph.

    """
    # Apply inner rewrites
    # TODO: Not sure this is the right place to do this, should we have a rewrite that
    #  explicitly triggers the optimization of the inner graphs of Scan?
    #  The C-code defers it to the make_thunk phase
    rewriter = (
        get_mode(op.mode)
        .including("numba")
        .excluding(*NUMBA._optimizer.exclude)
        .optimizer
    )
    fgraph = op.fgraph

    # A traced read is "overwritten" when a same-iteration write reuses its physical buffer
    # slot. With buffer length L, output tap o_out writes the slot read by input tap o_in iff
    # (o_out - o_in) % L == 0. Since L is always >= reach (the minimal admissible length, the
    # oldest lookback), only two gaps can trigger it:
    #  - gap 0 (o_out == o_in): the recurrence writes the very slot it just read (mit_mot
    #    accumulators: read g[k], write g[k] + delta back). Holds for ANY L, so the read is
    #    *certainly* overwritten this iteration.
    #  - gap == reach: a write lands just past the oldest read; in a buffer truncated to its
    #    minimum (L == reach) it wraps onto that just-discarded oldest read. So the read is
    #    overwritten only at L == reach: certain when the static length says so, merely possible
    #    when the length is statically unknown, impossible for any larger L. (A write landing
    #    further ahead would store onto a slot still to be read -- an invalid recurrence we
    #    need not consider, just as we don't consider L < reach.)
    # A certainly-overwritten read is dead once consumed this iteration, so the inner function
    # may destroy it in place. A possibly-overwritten read must not be destroyed (it may still
    # be live at a larger length), but no output may alias it either, since a same-iteration
    # overwrite would corrupt the alias before it is stored.
    def find_overwritten_reads(grouped_inner, outers, in_slices_seq, out_slices_seq):
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

    mit_mot_certain, mit_mot_possible = find_overwritten_reads(
        op.inner_mitmot_grouped(fgraph.inputs),
        op.outer_mitmot(node.inputs),
        op.info.mit_mot_in_slices,
        op.info.mit_mot_out_slices,
    )
    mit_sot_certain, mit_sot_possible = find_overwritten_reads(
        op.inner_mitsot_grouped(fgraph.inputs),
        op.outer_mitsot(node.inputs),
        op.info.mit_sot_in_slices,
        [(0,)] * op.info.n_mit_sot,
    )
    sit_sot_certain, sit_sot_possible = find_overwritten_reads(
        [[v] for v in op.inner_sitsot(fgraph.inputs)],
        op.outer_sitsot(node.inputs),
        op.info.sit_sot_in_slices,
        [(0,)] * op.info.n_sit_sot,
    )

    # Reads no output may alias (destructively or not): a same-iteration overwrite could corrupt
    # the alias before it is copied to its own buffer.
    potentially_overwritten_reads = [
        *mit_mot_certain,
        *mit_mot_possible,
        *mit_sot_certain,
        *mit_sot_possible,
        *sit_sot_certain,
        *sit_sot_possible,
    ]

    # Reads the inner function may destroy in place: the certainly-overwritten traced reads
    # (dead once consumed this iteration) and every untraced input (always immediately discarded).
    discarded = {
        *mit_mot_certain,
        *mit_sot_certain,
        *sit_sot_certain,
        *op.inner_untraced_sit_sot(fgraph.inputs),
    }
    # Grant the inner function the right to alias (borrow=True) all inputs and to destroy
    # (mutable=True) the reads known to be immediately discarded.
    input_specs = [In(x, borrow=True, mutable=x in discarded) for x in fgraph.inputs]
    add_supervisor_to_fgraph(
        fgraph=fgraph,
        input_specs=input_specs,
        accept_inplace=True,
    )

    if potentially_overwritten_reads:
        # Forbid any output from aliasing these reads. We could instead allow it and patch with a
        # deepcopy at the end (as we do for non-destructive boundary alias), but that is wasteful:
        # forbidding it makes the inner graph allocate a fresh buffer and write the result there.
        in_pos = {v: i for i, v in enumerate(fgraph.inputs)}
        fgraph.attach_feature(
            NoOutputInplaceOnInput([in_pos[t] for t in potentially_overwritten_reads])
        )

    # Rewrite graph
    rewriter(fgraph)

    # Post-patch alias contract via targeted deepcopies
    # Traced and untraced updates: copy an alias of a tap input the loop (may) overwrite
    #  this iteration.
    # Untraced updates: keep as is if it is the FIRST viewer of a freshly produced buffer
    # (including traced updates which will be copied immediately anyway), OR its own untraced input.
    # Any other alias is broken with a deepcopy: Sequences reads, traced reads, non-sequences,
    # other untraced inputs or updates.
    # Note: We could squeeze some more memory reuse by delaying the breaking of aliasing between untraced variables
    # by delaying the patched deepcopy until after the loop is over. This requires some care to handle alias transitions
    # between untraced updates that can happen over multiple iterations, and protect against cross-iteration destruction
    # that can corrupt such chains.
    own_untraced_input = dict(
        zip(
            op.inner_untraced_sit_sot_outs(fgraph.outputs),
            op.inner_untraced_sit_sot(fgraph.inputs),
            strict=True,
        )
    )
    untraced_outs = set(own_untraced_input)
    seen_untraced_roots = set()
    output_specs = []
    for update in fgraph.outputs:
        root = alias_root(update)
        if update in untraced_outs:
            borrow = (
                (
                    # freshly produced buffer
                    root.owner is not None
                    # or a self alias
                    or root is own_untraced_input[update]
                )
                # and not an alias of another untraced update
                and root not in seen_untraced_roots
            )
            if borrow:
                seen_untraced_roots.add(root)
        else:
            # traced update
            borrow = root not in potentially_overwritten_reads
        output_specs.append(Out(update, borrow=borrow))
    insert_deepcopy(fgraph, wrapped_inputs=input_specs, wrapped_outputs=output_specs)

    # Collect a set of untraced slots the inner function destroys in place.
    # These may demand an initial copy if the Scan is not granted permission to destroy them already.
    untraced_inputs_destroyed_by_inner_function = set()
    if hasattr(fgraph, "destroyers"):
        untraced_inputs_destroyed_by_inner_function = {
            outer_out_idx
            for inner_in, (outer_out_idx, _) in zip(
                op.inner_untraced_sit_sot(fgraph.inputs),
                op.outer_untraced_sit_sot_outs(node.outputs, with_idx=True),
                strict=True,
            )
            if fgraph.destroyers(inner_in)
        }

    scan_inner_func, inner_func_cache_key = numba_funcify_and_cache_key(
        op.fgraph, fgraph_name="numba_scan", ofg_memo=kwargs.get("ofg_memo")
    )

    outer_in_names_to_vars = {
        (f"outer_in_{i}" if i > 0 else "n_steps"): v for i, v in enumerate(node.inputs)
    }
    outer_in_names = list(outer_in_names_to_vars)
    outer_in_seqs_names = op.outer_seqs(outer_in_names)
    outer_in_mit_mot_names = op.outer_mitmot(outer_in_names)
    outer_in_mit_sot_names = op.outer_mitsot(outer_in_names)
    outer_in_sit_sot_names = op.outer_sitsot(outer_in_names)
    outer_in_nit_sot_names = op.outer_nitsot(outer_in_names)
    outer_in_untraced_sit_sot_names = op.outer_untraced_sit_sot(outer_in_names)
    outer_in_non_seqs_names = op.outer_non_seqs(outer_in_names)

    # These are all the outer-input names that have produce outputs/have output
    # taps (i.e. they have inner-outputs and corresponding outer-outputs).
    # Outer-outputs are ordered as follows:
    # mit-mot-outputs + mit-sot-outputs + sit-sot-outputs + nit-sots + untraced-sit-sot-outputs
    outer_in_outtap_names = (
        outer_in_mit_mot_names
        + outer_in_mit_sot_names
        + outer_in_sit_sot_names
        + outer_in_nit_sot_names
        + outer_in_untraced_sit_sot_names
    )

    # We create distinct variables for/references to the storage arrays for
    # each output.
    outer_in_to_storage_name: dict[str, str] = {}
    for outer_in_name in outer_in_mit_mot_names:
        outer_in_to_storage_name[outer_in_name] = f"{outer_in_name}_mitmot_storage"

    for outer_in_name in outer_in_mit_sot_names:
        outer_in_to_storage_name[outer_in_name] = f"{outer_in_name}_mitsot_storage"

    for outer_in_name in outer_in_sit_sot_names:
        outer_in_to_storage_name[outer_in_name] = f"{outer_in_name}_sitsot_storage"

    for outer_in_name in outer_in_nit_sot_names:
        outer_in_to_storage_name[outer_in_name] = f"{outer_in_name}_nitsot_storage"

    for outer_in_name in outer_in_untraced_sit_sot_names:
        outer_in_to_storage_name[outer_in_name] = (
            f"{outer_in_name}_untraced_sit_sot_storage"
        )

    outer_output_names = list(outer_in_to_storage_name.values())
    assert len(outer_output_names) == len(node.outputs)

    # Construct the inner-input expressions (e.g. indexed storage expressions)
    # Inner-inputs are ordered as follows:
    # sequences + mit-mot-inputs + mit-sot-inputs + sit-sot-inputs +
    # untraced-sit-sot-inputs + non-sequences.
    temp_0d_storage_alloc_stmts: list[str] = []
    inner_in_exprs_scalar: list[str] = []
    inner_in_exprs: list[str] = []

    def add_inner_in_expr(
        outer_in_name: str,
        tap_offset: int | None,
        storage_size_var: str | None,
        vector_slice_opt: bool,
    ):
        """Construct an inner-input expression."""
        storage_name = outer_in_to_storage_name.get(outer_in_name, outer_in_name)
        if vector_slice_opt:
            indexed_inner_in_str_scalar = idx_to_str(
                storage_name, tap_offset, size=storage_size_var, allow_scalar=True
            )
            temp_storage = f"{storage_name}_temp_scalar_{tap_offset}"
            storage_dtype = outer_in_var.type.numpy_dtype.name
            temp_0d_storage_alloc_stmts.append(
                f"{temp_storage} = np.empty((), dtype=np.{storage_dtype})"
            )
            inner_in_exprs_scalar.append(
                f"{temp_storage}[()] = {indexed_inner_in_str_scalar}"
            )
            indexed_inner_in_str = temp_storage
        else:
            indexed_inner_in_str = (
                storage_name
                if tap_offset is None
                else idx_to_str(
                    storage_name, tap_offset, size=storage_size_var, allow_scalar=True
                )
            )
        inner_in_exprs.append(indexed_inner_in_str)

    for outer_in_name in outer_in_seqs_names:
        # These outer-inputs are indexed without offsets or storage wrap-around
        outer_in_var = outer_in_names_to_vars[outer_in_name]
        is_vector = outer_in_var.ndim == 1
        add_inner_in_expr(outer_in_name, 0, None, vector_slice_opt=is_vector)

    inner_in_names_to_input_taps: dict[str, tuple[int, ...]] = dict(
        zip(
            outer_in_mit_mot_names + outer_in_mit_sot_names + outer_in_sit_sot_names,
            op.info.mit_mot_in_slices
            + op.info.mit_sot_in_slices
            + op.info.sit_sot_in_slices,
            strict=True,
        )
    )
    inner_in_names_to_output_taps: dict[str, tuple[int, ...] | None] = dict(
        zip(outer_in_mit_mot_names, op.info.mit_mot_out_slices, strict=True)
    )

    # Inner-outputs consist of:
    # mit-mot-outputs + mit-sot-outputs + sit-sot-outputs + nit-sots +
    # untraced-sit-sot-outputs [+ while-condition]
    inner_output_names = [f"inner_out_{i}" for i in range(len(op.inner_outputs))]

    # The assignment statements that copy inner-outputs into the outer-outputs
    # storage
    inner_out_to_outer_in_stmts: list[str] = []

    # Special statements that perform storage truncation for `while`-loops and
    # rotation for initially truncated storage.
    output_storage_post_proc_stmts: list[str] = []

    # In truncated storage situations (e.g. created by `scan_reduce_trace`),
    # the taps and output storage overlap, instead of the standard situation in
    # which the output storage is large enough to contain both the initial taps
    # values and the output storage.  In this truncated case, we use the
    # storage array like a circular buffer, and that's why we need to track the
    # storage size along with the taps length/indexing offset.
    def add_output_storage_post_proc_stmt(
        outer_in_name: str, max_offset: int, storage_size: str
    ):
        # Rotate the storage so that the last computed value is at the end of the storage array.
        # This is needed when the output storage array does not have a length
        # equal to the number of taps plus `n_steps`.
        output_storage_post_proc_stmts.append(
            dedent(
                f"""
                if 1 < {storage_size} < (i + {max_offset}):
                    {outer_in_name}_shift = (i + {max_offset}) % ({storage_size})
                    if {outer_in_name}_shift > 0:
                        {outer_in_name}_left = {outer_in_name}[:{outer_in_name}_shift]
                        {outer_in_name}_right = {outer_in_name}[{outer_in_name}_shift:]
                        {outer_in_name} = np.concatenate(({outer_in_name}_right, {outer_in_name}_left))
                """
            ).strip()
        )

        if op.info.as_while:
            # While loops need to truncate the output storage to a length given
            # by the number of iterations performed.
            output_storage_post_proc_stmts.append(
                dedent(
                    f"""
                    elif {storage_size} > (i + {max_offset}):
                        {outer_in_name} = {outer_in_name}[:i + {max_offset}]
                    """
                ).strip()
            )
        else:
            # And regular loops should zero out unused entries of the output buffer
            # These show up with truncated gradients of while loops
            output_storage_post_proc_stmts.append(
                dedent(
                    f"""
                    elif {storage_size} > (i + {max_offset}):
                        {outer_in_name}[i + {max_offset}:] = 0
                    """
                ).strip()
            )

    # Special in-loop statements that create (nit-sot) storage arrays after a
    # single iteration is performed.  This is necessary because we don't know
    # the exact shapes of the storage arrays that need to be allocated until
    # after an iteration is performed.
    inner_out_post_processing_stmts: list[str] = []

    # Storage allocation statements
    # For output storage allocated/provided by the inputs, these statements
    # will either construct aliases between the input names and the entries in
    # `outer_in_to_storage_name` or assign the latter to expressions that
    # create copies of those storage inputs.
    # In the nit-sot case, empty dummy arrays are assigned to the storage
    # variables and updated later by the statements in
    # `inner_out_post_processing_stmts`.
    storage_alloc_stmts: list[str] = []

    for outer_in_name in outer_in_outtap_names:
        outer_in_var = outer_in_names_to_vars[outer_in_name]

        if outer_in_name not in outer_in_nit_sot_names:
            storage_name = outer_in_to_storage_name[outer_in_name]

            is_tapped = outer_in_name not in outer_in_untraced_sit_sot_names
            if is_tapped:
                storage_size_name = f"{outer_in_name}_len"
                storage_size_stmt = f"{storage_size_name} = {outer_in_name}.shape[0]"
                input_taps = inner_in_names_to_input_taps[outer_in_name]
                max_lookback_inp_tap = -min(0, min(input_taps))
                assert max_lookback_inp_tap >= 0

                for in_tap in input_taps:
                    tap_offset = max_lookback_inp_tap + in_tap
                    is_vector = outer_in_var.ndim == 1
                    add_inner_in_expr(
                        outer_in_name,
                        tap_offset,
                        storage_size_name,
                        vector_slice_opt=is_vector,
                    )

                output_taps = inner_in_names_to_output_taps.get(outer_in_name, [0])
                for out_tap in output_taps:
                    tap_offset = max_lookback_inp_tap + out_tap
                    assert tap_offset >= 0
                    inner_out_to_outer_in_stmts.append(
                        idx_to_str(
                            storage_name,
                            tap_offset,
                            size=storage_size_name,
                            allow_scalar=True,
                        )
                    )

                if outer_in_name not in outer_in_mit_mot_names:
                    # MIT-SOT and SIT-SOT may require buffer rolling/truncation after the main loop
                    max_offset_out_tap = max(output_taps) + max_lookback_inp_tap
                    add_output_storage_post_proc_stmt(
                        storage_name, max_offset_out_tap, storage_size_name
                    )

            else:
                storage_size_stmt = ""
                add_inner_in_expr(outer_in_name, None, None, vector_slice_opt=False)
                inner_out_to_outer_in_stmts.append(storage_name)

            output_idx = outer_output_names.index(storage_name)
            # Copy the outer inputs when the loop mutates them and the destroy_map doesn't already grant permission
            needs_copy = output_idx not in node.op.destroy_map and (
                # Traced buffers are always mutated by the loop write-back procedure
                is_tapped
                # Untraced inputs are only mutated by the inner function,
                # so we make the copy conditional on that actually happening
                or output_idx in untraced_inputs_destroyed_by_inner_function
            )
            if needs_copy:
                storage_alloc_stmt = f"{storage_name} = numba_deepcopy({outer_in_name})"
            else:
                storage_alloc_stmt = f"{storage_name} = {outer_in_name}"

            storage_alloc_stmt = dedent(
                f"""
                {storage_size_stmt}
                {storage_alloc_stmt}
                """
            ).strip()

            storage_alloc_stmts.append(storage_alloc_stmt)

        else:
            assert outer_in_name in outer_in_nit_sot_names

            # This is a special case in which there are no outer-inputs used
            # for outer-output storage, so we need to create our own storage
            # from scratch.
            storage_name = outer_in_to_storage_name[outer_in_name]
            storage_size_name = f"{outer_in_name}_len"

            inner_out_to_outer_in_stmts.append(
                idx_to_str(storage_name, 0, size=storage_size_name, allow_scalar=True)
            )
            add_output_storage_post_proc_stmt(storage_name, 0, storage_size_name)

            # In case of nit-sots we are provided the length of the array in
            # the iteration dimension instead of actual arrays, hence we
            # allocate space for the results accordingly.
            curr_nit_sot_position = outer_in_nit_sot_names.index(outer_in_name)
            curr_nit_sot = op.inner_nitsot_outs(op.inner_outputs)[curr_nit_sot_position]

            known_static_shape = all(dim is not None for dim in curr_nit_sot.type.shape)
            if known_static_shape:
                storage_shape = create_tuple_string(
                    (storage_size_name, *(map(str, curr_nit_sot.type.shape)))
                )
            else:
                storage_shape = create_tuple_string(
                    (storage_size_name, *(["0"] * curr_nit_sot.ndim))
                )
            storage_dtype = curr_nit_sot.type.numpy_dtype.name

            storage_alloc_stmts.append(
                dedent(
                    f"""
                {storage_size_name} = {outer_in_name}.item()
                {storage_name} = np.empty({storage_shape}, dtype=np.{storage_dtype})
                """
                ).strip()
            )

            if not known_static_shape:
                # In this case, we don't know the shape of the output storage
                # array until we get some output from the inner-function.
                # With the following we add delayed output storage initialization:
                inner_out_name = op.inner_nitsot_outs(inner_output_names)[
                    curr_nit_sot_position
                ]
                inner_out_post_processing_stmts.append(
                    dedent(
                        f"""
                    if i == 0:
                        {storage_name} = np.empty(({storage_size_name},) + np.shape({inner_out_name}), dtype=np.{storage_dtype})
                    """
                    ).strip()
                )

    for name in outer_in_non_seqs_names:
        add_inner_in_expr(name, None, None, vector_slice_opt=False)

    if op.info.as_while:
        # The inner function will return a boolean as the last value
        inner_out_to_outer_in_stmts.append("cond")

    assert len(inner_in_exprs) == len(op.fgraph.inputs)

    inner_scalar_in_args_to_temp_storage = "\n".join(inner_in_exprs_scalar)
    # Break inputs in new lines, just for readability of the source code
    inner_in_args = f",\n{' ' * 12}".join(inner_in_exprs)
    inner_outputs = create_tuple_string(inner_output_names)
    input_storage_block = "\n".join(storage_alloc_stmts)
    input_temp_0d_storage_block = "\n".join(temp_0d_storage_alloc_stmts)
    output_storage_post_processing_block = "\n".join(output_storage_post_proc_stmts)
    inner_out_post_processing_block = "\n".join(inner_out_post_processing_stmts)

    inner_out_to_outer_out_stmts = "\n".join(
        f"{s} = {d}"
        for s, d in zip(inner_out_to_outer_in_stmts, inner_output_names, strict=True)
    )

    scan_op_src = f"""
def scan({", ".join(outer_in_names)}):

{indent(input_storage_block, " " * 4)}

{indent(input_temp_0d_storage_block, " " * 4)}

    i = 0
    cond = np.array(False)
    while i < n_steps and not cond.item():
{indent(inner_scalar_in_args_to_temp_storage, " " * 8)}

        {inner_outputs} = scan_inner_func(
            {inner_in_args}
        )
{indent(inner_out_post_processing_block, " " * 8)}
{indent(inner_out_to_outer_out_stmts, " " * 8)}
        i += 1

{indent(output_storage_post_processing_block, " " * 4)}

    return {", ".join(outer_output_names)}
    """

    scan_op_fn = compile_numba_function_src(
        scan_op_src,
        "scan",
        globals()
        | {
            "np": np,
            "scan_inner_func": scan_inner_func,
            "numba_deepcopy": numba_deepcopy,
        },
    )

    if inner_func_cache_key is None:
        # If we can't cache the inner function, we can't cache the Scan either
        scan_cache_key = None
    else:
        scan_cache_version = 1
        scan_cache_key = sha256(
            f"({scan_op_src}, {inner_func_cache_key}, {scan_cache_version})".encode()
        ).hexdigest()

    return numba_basic.numba_njit(scan_op_fn, boundscheck=False), scan_cache_key
