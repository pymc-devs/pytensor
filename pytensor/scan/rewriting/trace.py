"""Memory rewrites: shrink Scan storage and iteration counts.

These rewrites drop work and storage that nothing downstream actually
needs:

* :func:`scan_reduce_nsteps` — when every client of a Scan output reads a
  constant scalar index, shorten ``n_steps`` to the minimum that covers
  those reads and rewrite each client to a negative index against the
  trimmed trace.
* :func:`scan_reduce_trace_rewrite` (registered as ``scan_reduce_trace_prealloc``
  / ``scan_reduce_trace_no_prealloc``) — shorten outer buffers and the
  ``n_steps`` to the smallest range any client actually reads.
* :func:`scan_sit_sot_to_untraced` — convert sit_sot states whose
  history is unused into the cheaper ``untraced_sit_sot`` form.

Subtensor chains rooted at scan outputs (e.g. ``raw_out[init_l:][-k]``)
are folded by ``local_subtensor_merge_integer`` running in canonicalize
under the ``shape_unsafe`` tag, before this module's rewrites see them.
"""

import dataclasses
from itertools import chain
from typing import cast

import pytensor.tensor as pt
from pytensor.configdefaults import config
from pytensor.graph.basic import Constant, Variable
from pytensor.graph.rewriting.basic import node_rewriter
from pytensor.graph.traversal import apply_depends_on
from pytensor.scan.op import Scan
from pytensor.scan.utils import expand_empty
from pytensor.tensor.basic import (
    AllocEmpty,
    atleast_Nd,
)
from pytensor.tensor.basic import switch as pt_switch
from pytensor.tensor.math import ge, maximum, minimum
from pytensor.tensor.rewriting.basic import broadcasted_by
from pytensor.tensor.shape import Shape, Shape_i
from pytensor.tensor.subtensor import (
    IncSubtensor,
    Subtensor,
    as_index_constant,
    basic_subtensor,
    get_idx_list,
)
from pytensor.tensor.variable import TensorVariable


def _maybe_constant_int(v) -> int | None:
    """Return *v* as a Python int if it is a constant scalar, else ``None``.

    *v* must be a tensor variable (or Python/NumPy scalar), **not** ``None``.
    Callers dealing with slice components that may be ``None`` (= omitted)
    must check for that case themselves before calling.
    """
    if isinstance(v, Constant):
        return int(v.data)
    return None


def _init_l_per_output(op_info) -> list[int]:
    """Per-output strip length: ``abs(min(taps))`` for mit_sot/sit_sot, ``0``
    for mit_mot and nit_sot."""
    return (
        [0] * op_info.n_mit_mot
        + [
            abs(min(v))
            for v in chain(op_info.mit_sot_in_slices, op_info.sit_sot_in_slices)
        ]
        + [0] * op_info.n_nit_sot
    )


def _python_slice_from_idx(entry):
    """Convert an idx_list entry into a pure-Python ``slice(int|None, ...)`` or
    Python int. Returns ``None`` if any present component is non-constant.
    """
    if isinstance(entry, slice):
        a = None if entry.start is None else _maybe_constant_int(entry.start)
        b = None if entry.stop is None else _maybe_constant_int(entry.stop)
        s = None if entry.step is None else _maybe_constant_int(entry.step)
        if entry.start is not None and a is None:
            return None
        if entry.stop is not None and b is None:
            return None
        if entry.step is not None and s is None:
            return None
        return slice(a, b, s)
    return _maybe_constant_int(entry)


def _is_default_scan_buffer(final_buffer: TensorVariable, taps: int) -> bool:
    node = final_buffer.owner

    if node is None:
        return False

    op = node.op
    if not (
        isinstance(op, IncSubtensor)
        and op.set_instead_of_inc
        and op.idx_list == (slice(None, 0),)
    ):
        return False

    init_buffer, init_value, *_ = node.inputs
    if not (
        init_buffer.owner is not None and isinstance(init_buffer.owner.op, AllocEmpty)
    ):
        return False

    # The value may have been broadcast to fill in the initial taps.
    # If the user specified outputs as:
    #   x = scalar(); init = alloc(x, 2);
    #   outputs_info=[init, taps=(-2, -1)]
    # Scan will generate an initial buffer that looks like
    #   alloc_empty(2 + nsteps)[:2].set(alloc(x, 2))
    # PyTensor will then rewrite it as:
    #   alloc_empty(2 + nsteps)[:2].set(x)
    # When the initial value (x) is being broadcast by the set_subtensor
    # we can't recreate a newly sized buffer working with x alone
    # We want to check that:
    #   1. alloc_empty(2 + nsteps)[:2].broadcastable == x.broadcastable
    # But due to laziness we use the slightly more conservative check:
    #   2. alloc_empty(2 + nsteps).broadcastable == x.broadcastable
    if taps > 1:
        return not broadcasted_by(init_value, init_buffer)
    else:
        # In this case we know we have alloc_empty(1 + nsteps, ...)[:1].set(init_value)
        # The first dimension cannot possibly broadcast in the subtensor assignment,
        # so we exclude it from `broadcasted_by`. To exclude it we squeeze it out,
        # after adding any other implicit expand_dims. We select into the first entry of
        # the buffer, to check for potential broadcasting in other dimensions.
        init_value_ = atleast_Nd(init_value, n=init_buffer.ndim)
        return not broadcasted_by(init_value_.squeeze(0), init_buffer[0])


@node_rewriter([Scan])
def scan_reduce_nsteps(fgraph, node):
    """Reduce the number of scan iterations when clients don't need all steps.

    Analyzes constant indices on scan outputs to find the minimum n_steps
    needed. Adjusts client subtensors so negative indices are preserved,
    enabling the downstream buffer reduction rewrite to reason about them.

    Examples:
        scan(n)[-2]    -> scan(n-1)[-1]      (one fewer step)
        scan(10)[7]    -> scan(8)[-1]         (positive -> negative)
        scan(n)[-3:-1] -> scan(n-1)[-2:None]  (slice adjustment)
    """
    op = node.op
    op_info = op.info

    # Can't reduce n_steps for while-scans or scans with untraced sit_sot
    # outputs.
    if op_info.as_while or op_info.n_untraced_sit_sot:
        return None

    c_outs = (
        op_info.n_mit_mot + op_info.n_mit_sot + op_info.n_sit_sot + op_info.n_nit_sot
    )
    init_l = _init_l_per_output(op_info)

    n_steps = node.inputs[0]
    n_steps_const = _maybe_constant_int(n_steps)

    # Collect the maximum needed steps across all outputs and their clients.
    # Negative-index requirements are tracked as ``offset = idx + 1`` (each
    # ``<= 0``); if any equals 0 some client wants ``raw[-1]`` and we must
    # bail. Tracking the constant offset directly avoids producing
    # ``n_steps - minimum(n_steps, n_steps)`` style "no-op" deltas that
    # ``scan_reduce_trace`` can't read through.
    max_needed: int = 0  # max constant-position requirement (for non-negative idx)
    neg_offsets: list[int] = []  # collected ``idx + 1`` over negative-idx clients
    sym_constraints: list = []  # symbolic ``nw_steps_min`` contributions
    can_reduce = True

    for i, out in enumerate(node.outputs[:c_outs]):
        for cl, _ in fgraph.clients[out]:
            if not isinstance(cl.op, Subtensor):
                can_reduce = False
                break

            s0 = get_idx_list(cl.inputs, cl.op.idx_list)[0]

            if isinstance(s0, slice):
                # Slice step != 1 (e.g. reverse) needs different bookkeeping.
                if s0.step is not None and _maybe_constant_int(s0.step) != 1:
                    can_reduce = False
                    break
                # Open-ended [a:] or non-constant stop: can't reduce.
                stop_int = None if s0.stop is None else _maybe_constant_int(s0.stop)
                if stop_int is None:
                    can_reduce = False
                    break
                if stop_int < 0:
                    # ``[a:-k]``: max raw position read = raw_length - k - 1,
                    # so n_steps_min = n_steps - k. Track as offset = stop.
                    neg_offsets.append(stop_int)
                elif stop_int > 0:
                    # ``[a:b]``: max raw position = b - 1, so n_steps_min
                    # = b - init_l.
                    max_needed = max(max_needed, stop_int - init_l[i])
                # else: stop_int == 0: empty slice, contributes no constraint.
            else:
                if (idx_int := _maybe_constant_int(s0)) is not None:
                    if idx_int == -1:
                        # ``raw[-1]`` reads the last computed step -> all
                        # iterations needed, can't reduce.
                        can_reduce = False
                        break
                    if idx_int < 0:
                        neg_offsets.append(idx_int + 1)
                    else:
                        max_needed = max(max_needed, idx_int + 1 - init_l[i])
                else:
                    # Symbolic scalar idx: contribute
                    #   switch(s0 >= 0, s0 + 1 - init_l, n_steps + s0 + 1)
                    # to the running maximum. Sign-cancellation makes the new
                    # client idx collapse to ``-1`` in both branches when this
                    # client drives the reduction (and is well-defined when it
                    # doesn't). We can't bail on the symbolic ``raw[-1]`` case
                    # statically; ``minimum`` below caps at ``n_steps``, making
                    # the rewrite a no-op at runtime if it degenerates.
                    sym_constraints.append(
                        pt_switch(
                            ge(s0, 0),
                            s0 + (1 - init_l[i]),
                            n_steps + s0 + 1,
                        )
                    )
        if not can_reduce:
            break

    if not can_reduce:
        return None

    # Compute new n_steps: maximum over all collected contributions, capped
    # at the original ``n_steps``.
    contributions: list = []
    if max_needed > 0:
        contributions.append(max_needed)
    if neg_offsets:
        contributions.append(n_steps + max(neg_offsets))
    contributions.extend(sym_constraints)
    if not contributions:
        return None

    nw_steps: int | Variable = contributions[0]
    for c in contributions[1:]:
        nw_steps = maximum(nw_steps, c)
    # Cap ``nw_steps`` at the user's ``n_steps`` whenever we can't prove the
    # contributions stay below it. The all-positive-constants path (no
    # ``sym_constraints``, no ``neg_offsets``) only skips the cap when
    # ``n_steps`` itself is a known constant; otherwise an ``out[k]`` client
    # would silently extend a symbolic ``n_steps`` past what the user
    # requested.
    if not isinstance(nw_steps, int) or sym_constraints or n_steps_const is None:
        nw_steps = minimum(nw_steps, n_steps)

    # Static reduction check (only the all-constant path can decide here;
    # symbolic contributions are decided at runtime).
    if (
        isinstance(nw_steps, int)
        and n_steps_const is not None
        and nw_steps >= n_steps_const
    ):
        return None

    # Build delta for index adjustment
    delta = n_steps - nw_steps

    # Create new scan with reduced n_steps
    nw_inputs = list(node.inputs)
    nw_inputs[0] = pt.as_tensor_variable(nw_steps)

    # Shrink each recurrent (mit_sot/sit_sot) buffer to ``taps + nw_steps``
    # to match the reduced n_steps. Default empty buffers are re-allocated;
    # user-supplied buffers are sliced in place.
    offset = 1 + op_info.n_seqs + op_info.n_mit_mot
    for idx in range(op_info.n_mit_sot + op_info.n_sit_sot):
        i = idx + op_info.n_mit_mot
        taps = init_l[i]
        nw_input = nw_inputs[offset + idx]
        if _is_default_scan_buffer(nw_input, taps):
            nw_input = expand_empty(nw_input.owner.inputs[1], nw_steps)
        else:
            nw_input = nw_input[: (taps + nw_steps)]
        nw_inputs[offset + idx] = nw_input

    # Shrink each nit_sot's scalar buffer-size input to ``nw_steps``
    # (only when it was originally the scan's n_steps, i.e. "store all").
    nitsot_offset = (
        offset + op_info.n_mit_sot + op_info.n_sit_sot + op_info.n_untraced_sit_sot
    )
    for idx in range(op_info.n_nit_sot):
        pos = nitsot_offset + idx
        if nw_inputs[pos] == n_steps:
            nw_inputs[pos] = pt.as_tensor_variable(nw_steps)

    new_outs = cast(list[TensorVariable], op(*nw_inputs, return_list=True))

    # Adjust client subtensors
    old_new = []
    for i, out in enumerate(node.outputs[:c_outs]):
        old_raw_length = node.inputs[0] + init_l[i]  # symbolic in old n_steps
        new_length = nw_steps + init_l[i]
        for cl, _ in fgraph.clients[out]:
            if not isinstance(cl.op, Subtensor):
                continue
            this_slice = get_idx_list(cl.inputs, cl.op.idx_list)
            s0 = this_slice[0]
            rest = this_slice[1:]

            if isinstance(s0, slice):
                # Slice bounds: a negative ``b`` semantically refers to
                # position ``old_raw_length + b``. Pin each negative bound
                # to that absolute position so the slice still selects the
                # same elements against the shorter ``new_outs[i]``;
                # ``None`` and non-negative bounds carry over unchanged.
                def _maybe_to_positive(b):
                    if b is None:
                        return None
                    b_int = _maybe_constant_int(b)
                    if b_int is not None and b_int < 0:
                        return old_raw_length + b
                    return b

                new_s0 = slice(
                    _maybe_to_positive(s0.start),
                    _maybe_to_positive(s0.stop),
                    s0.step,
                )
            else:
                # Scalar idx: rewrite against the shorter ``new_outs[i]``
                # while pointing at the same absolute element.
                #   positive k -> ``k - new_length`` (negative form so
                #     save_mem can act on it).
                #   negative -j -> ``-j + delta`` (where delta =
                #     old_n_steps - nw_steps; same absolute position).
                # Symbolic idx picks the branch via switch; when this
                # client drives the reduction the formula collapses to -1.
                idx_int = _maybe_constant_int(s0)
                if idx_int is None:
                    new_s0 = pt_switch(ge(s0, 0), s0 - new_length, s0 + delta)
                elif idx_int >= 0:
                    new_s0 = s0 - new_length
                else:
                    new_s0 = s0 + delta
            nw_slice = (
                as_index_constant(new_s0),
                *(as_index_constant(s) for s in rest),
            )

            new_o = basic_subtensor(new_outs[i], *nw_slice)
            old_new.append((cl.outputs[0], new_o))

    if not old_new:
        return None

    # Check the new outputs don't depend on the old scan node
    if any(apply_depends_on(new.owner, node) for _, new in old_new):
        return False

    replacements = dict(old_new)
    replacements["remove"] = [node]
    return replacements


def _strip_chain_negative_start(cl, init_l_i, raw_length_const, fgraph):
    """Recognize ``out[init_l:][-k:]`` (initial-state strip + tail slice).

    ``out[init_l:]`` is the slice scan inserts internally so the user sees
    ``xs`` (the computed-only steps) rather than the raw buffer with the
    initial state prepended.

    Why save_mem-side rather than upstream pre-folding: pre-folding
    to a static ``out[-k:]`` would change semantics — when ``n_steps < k``
    the original chain clamps to ``n_steps`` elements, but ``out[-k:]``
    against the original buffer always returns ``k``.
    The semantics-preserving rewrite is the dynamic
    ``trimmed_out[len - minimum(k, n_steps):]`` against the *trimmed* buffer,
    whose length only exists once ``scan_reduce_trace`` has decided to trim —
    so recognition has to ride along with the buffer-reduction rebuild.

    Two shapes match:

    * **Un-merged chain** ``out[l:][-k:]`` (typical when ``n_steps`` is
      symbolic): the strip ``out[l:]`` survived ``local_subtensor_merge_slice``
      because it can't fold without shape info. The save_mem rebuild
      collapses this by replacing the *inner* ``[-k:]`` Subtensor's output
      with ``trimmed_out[len - minimum(k, n_steps):]``. The dynamic slice
      handles the ``n_steps < k`` regime (returns just ``n_steps`` elements
      via clamping at runtime).
    * **Merged single Subtensor** ``out[a:L]`` where ``L`` equals ``out``'s
      static length. When ``n_steps`` is constant, ``local_subtensor_merge_slice``
      flattens ``out[init_l:][-k:]`` to this form (e.g. ``init_l=2``,
      ``n_steps=5`` -> ``L=7``; user-written ``out[2:][-3:]`` becomes
      ``out[4:7]``). ``get_canonical_form_slice`` normalizes ``stop=None``
      to the concrete length, hiding the original ``-k``. We recover
      ``-k = a - L`` and rewrite to ``out[a-L:]`` (here ``out[-3:]``) so
      save_mem's buffer-trim logic picks it up.

    Returns ``(effective_negative_start, replace_slice, inner_cl_or_None)``
    or ``(None, None, None)``.

    * ``effective_negative_start`` is the recovered ``-k`` (a negative int).
      The caller turns it into the buffer size to keep via
      ``needed = abs(effective_negative_start)``.
    * ``replace_slice`` (merged shape only) is the rewritten slice
      ``(slice(-k, None, None),)`` — substituted into ``slices[i][k_cl]``
      so the rebuild emits the negative-start form. ``None`` for the
      un-merged shape.
    * ``inner_cl`` (un-merged shape only) is the deeper ``[-k:]`` Subtensor
      node; the rebuild replaces its output with the dynamic
      ``trimmed_out[len - minimum(k, n_steps):]``, orphaning the outer
      strip (DCE'd). ``None`` for the merged shape.
    """
    null = (None, None, None)

    match cl.op:
        case Subtensor(idx_list=(slice(),)):
            pass
        case _:
            # TODO: We could handle cl.op.idx_list > 1, trailing slices don't matter for save mem
            return null

    [outer_slice] = get_idx_list(cl.inputs, cl.op.idx_list)
    outer_slice = _python_slice_from_idx(outer_slice)
    if outer_slice is None or outer_slice.step not in (None, 1):
        return null
    start_const = 0 if outer_slice.start is None else outer_slice.start
    stop_const = outer_slice.stop  # int or None

    # Un-merged shape: ``cl`` is the strip ``[init_l:]`` (or ``[0:]`` on
    # nit_sot); the deeper client must be ``[-k:]``.
    if start_const in (0, init_l_i) and stop_const is None:
        inner_clients = fgraph.clients.get(cl.outputs[0], [])
        if len(inner_clients) != 1:
            return null
        [(inner_cl, _)] = inner_clients
        match inner_cl.op:
            case Subtensor(idx_list=(slice(),)):
                pass
            case _:
                return null

        [inner_slice] = get_idx_list(inner_cl.inputs, inner_cl.op.idx_list)
        inner_slice = _python_slice_from_idx(inner_slice)
        if (
            inner_slice is None
            or inner_slice.step not in (None, 1)
            or inner_slice.stop is not None
            or inner_slice.start is None
            or inner_slice.start >= 0
        ):
            return null
        return inner_slice.start, None, inner_cl

    # Merged shape: single ``out[a:L]`` with ``a > init_l``, equivalent to
    # ``out[a-L:]`` (negative start).
    if (
        raw_length_const is not None
        and start_const > init_l_i
        and stop_const == raw_length_const
    ):
        translated = start_const - raw_length_const  # negative
        return translated, (slice(translated, None, None),), None

    return null


def _translate_positive_to_negative(s0, raw_length_const, init_l_i):
    """Translate a constant positive-form client into negative-form.

    When the raw buffer length is a known constant ``raw_length_const`` we can
    rewrite a positive-form client into the negative form ``scan_reduce_trace``
    already understands, exposing buffer trims it would otherwise miss
    (e.g. ``out0[k]`` and ``out1[k:]`` siblings on a multi-output scan whose
    ``n_steps`` is symbolic — ``scan_reduce_nsteps`` bails for the open-ended
    sibling, so positive-form recognition has to live here).

    Recognized shapes:

    * scalar ``out[k]`` with ``0 < k < raw_length_const`` -> ``k - raw_length_const``.
    * open-ended ``out[k:]`` with ``init_l_i < k < raw_length_const``,
      step in {None, 1} -> ``slice(k - raw_length_const, None, None)``.
    * bounded ``out[k:m]`` with ``init_l_i < k < m <= raw_length_const``,
      step in {None, 1} -> ``slice(k - raw_length_const, m - raw_length_const,
      None)``, collapsing the stop to ``None`` when ``m == raw_length_const``
      (mirrors ``_strip_chain_negative_start``'s merged-shape recovery).

    Returns ``(new_first_axis, needed)`` on success and ``None`` otherwise.
    The caller splices ``new_first_axis`` into ``slices[i][k_cl]``'s leading
    dim (preserving any trailing axes) and uses ``needed = raw_length_const
    - k`` as the buffer size to keep.

    The strict ``k > 0`` (scalar) / ``k > init_l_i`` (slice) gates ensure
    ``needed < raw_length_const`` so the rewrite is non-trivial; without
    them a translated ``out[0]`` / ``out[init_l:]`` would yield no trim and
    the rewrite would re-fire on its own output.
    """
    if isinstance(s0, slice):
        if s0.step is not None and _maybe_constant_int(s0.step) != 1:
            return None
        start = None if s0.start is None else _maybe_constant_int(s0.start)
        if start is None or not (init_l_i < start < raw_length_const):
            return None
        if s0.stop is None:
            new_stop = None
        elif (
            stop := _maybe_constant_int(s0.stop)
        ) is not None and stop == raw_length_const:
            new_stop = None
        elif stop is not None and start < stop < raw_length_const:
            new_stop = stop - raw_length_const  # negative
        else:
            return None
        return slice(start - raw_length_const, new_stop, None), raw_length_const - start

    k = _maybe_constant_int(s0)
    if k is None or not (0 < k < raw_length_const):
        return None
    return k - raw_length_const, raw_length_const - k


def _n_steps_static_min(n_steps) -> int | None:
    """Largest statically-known lower bound on ``n_steps``, or ``None`` if no
    static info is available.

    Covers the constant case and the common shape-of-input cases where the
    source tensor's static type shape pins the dim length.
    """
    if (n_const := _maybe_constant_int(n_steps)) is not None:
        return n_const

    match n_steps.owner_op_and_inputs:
        case Shape_i(i=dim), src:
            pass
        case Subtensor(idx_list=(0,)), inner, dim_symbolic:
            match inner.owner_op_and_inputs:
                case Shape(), src:
                    dim = _maybe_constant_int(dim_symbolic)
                    if dim is None:
                        return None
                case _:
                    return None
        case _:
            return None

    try:
        static_dim: int | None = src.type.shape[dim]
    except IndexError:
        return None  # out of bounds index

    return static_dim


def scan_reduce_trace_rewrite(
    fgraph, node, backend_supports_output_pre_allocation: bool
):
    r"""Graph optimizer that reduces scan memory consumption.

    This optimizations attempts to determine if a `Scan` node, during its execution,
    for any of its outputs, can get away with allocating a memory buffer that is
    large enough to contain some of the computed timesteps of that output but not
    all of them.

    By default, during the execution of a `Scan` node, memory buffers will be
    allocated to store the values computed for every output at every iteration.
    However, in some cases, there are outputs for which there is only really a
    need to store the most recent ``N`` values, not all of them.

    For instance, if a `Scan` node has a SITSOT output (last computed value is
    fed back as an input at the next iteration) and only the last timestep of
    that output is ever used in the outer function, the `ScanSaveMem` optimization
    could determine that there is no need to store all computed timesteps for
    that SITSOT output. Only the most recently computed timestep ever needs to
    be kept in memory.

    There are two ways in which the Scan buffer size is controlled:
    1. Each recurring output is saved in an input empty tensor x with the initial
    state written at x[:abs(min(taps))]. The remaining x[abs(min(taps)):]
    positions determine how many intermediate results should be stored.
    This rewrite shortens x[abs(min(taps)):] to the smallest possible size.
    2. Each non-recurrent output (nit-sot) is associated with a scalar integer
    input that determines how many steps should be saved in the perform method.
    This rewrite reduces this number to the smallest possible.

    The scan perform implementation takes the output sizes into consideration,
    saving the newest results over the oldest ones whenever the buffer is filled.

    This rewrite must only run at compilation time, after grad() has already
    built the backward scan. The backward scan needs all intermediate forward
    states as sequence inputs (to evaluate f'(x[t])). If this rewrite truncates
    buffers before grad() is called, the gradient will be silently wrong.
    TODO: Use a subclass that raises explicitly on `L_op`

    Paramaters
    ----------
    backend_supports_output_pre_allocation: bool
        When the backend supports output pre-allocation Scan must keep buffers
        with a length of required_states + 1, because the inner function will
        attempt to write the inner function outputs directly into the provided
        position in the outer circular buffer. This would invalidate results,
        if the input is still needed for some other output computation.
    """
    # 1. Initialization of variables
    # Note 1) We do not actually care about outputs representing shared
    # variables (those have no intermediate values) so it is safer to
    # ignore them and not change them in any way. To simplify the
    # optimizations I construct the variable ``c_outs`` ( that counts
    # outputs up to those we care) and the list ``init_l`` which for any
    # output we care says the length of its initial state. Note that
    # defining ``init_l`` for mit_mot sequences is a bit trickier but
    # it is safe to set it to 0
    op = node.op
    op_info = op.info
    c_outs = (
        op_info.n_mit_mot + op_info.n_mit_sot + op_info.n_sit_sot + op_info.n_nit_sot
    )

    init_l = _init_l_per_output(op_info)

    # Constant raw length per output (init_l + n_steps), or None if symbolic.
    n_steps = node.inputs[0]
    n_steps_const = _maybe_constant_int(n_steps)
    raw_length_consts = (
        [il + n_steps_const for il in init_l]
        if isinstance(n_steps_const, int)
        else [None] * len(init_l)
    )
    # Per-output rebuild redirects for un-merged ``out[init_l:][-k:]`` chains
    # (recognised by ``_strip_chain_negative_start``):
    #   {client_index: (inner_cl_node, k_const)}.
    # At rebuild time we replace ``inner_cl.outputs[0]`` (the deeper
    # ``[-k:]``) with a slice on the trimmed buffer; the outer strip is left
    # orphaned (DCE'd).
    client_redirects: list[dict] = [{} for _ in node.outputs]
    # 2. Check the clients of each output and see for how many steps scan
    # needs to run.
    assert len(node.outputs) >= c_outs

    prealloc_outs = (
        backend_supports_output_pre_allocation and config.scan__allow_output_prealloc
    )
    last_sitsot_idx = op_info.n_mit_mot + op_info.n_mit_sot + op_info.n_sit_sot - 1

    # ``slices[i]`` records the per-client idx_list for each output, or None
    # if any client isn't a Subtensor (then the buffer can't be trimmed).
    # Used at rebuild time to redirect each client to the new (smaller)
    # output.
    slices: list[None | list] = [None] * len(node.outputs)
    # ``store_steps[i]`` is the buffer size to keep for output ``i``. ``0``
    # means "keep all" (mandatory for mit_mot and shared); ``-1`` is a
    # "no decision" sentinel flipped at the end.
    store_steps = [0] * op_info.n_mit_mot + [-1] * (c_outs - op_info.n_mit_mot)
    flag_store = False

    # Single pass over each non-mit_mot output's clients: collect slices and
    # decide store_steps together. mit_mot outputs always need full storage,
    # so we skip them entirely.
    for i, out in enumerate(
        node.outputs[op_info.n_mit_mot : c_outs], start=op_info.n_mit_mot
    ):
        slices[i] = []
        for k_cl, (cl, _) in enumerate(fgraph.clients[out]):
            if not isinstance(cl.op, Subtensor):
                # ``Output`` or any other consumer -> can't trim this output.
                slices[i] = None
                store_steps[i] = 0
                break

            this_slice = get_idx_list(cl.inputs, cl.op.idx_list)
            slices[i].append(this_slice)  # type: ignore[union-attr]
            s0 = this_slice[0]

            # Recognize the strip + ``[-k:]`` chain (or its merged ``[a:L]`` equivalent).
            # Translate to a negative-start slice so the buffer-trim logic below picks it up.
            neg_start, replace, inner_cl = _strip_chain_negative_start(
                cl, init_l[i], raw_length_consts[i], fgraph
            )
            if neg_start is not None:
                needed = abs(neg_start)
                if replace is not None:
                    # Merged form ``out[a:L]``: rewrite the recorded slice
                    # in place so the rebuild emits the negative-start form.
                    slices[i][k_cl] = replace  # type: ignore[index]
                if inner_cl is not None:
                    # Un-merged chain ``out[init_l:][-k:]``: defer to a
                    # rebuild-time redirect that replaces the *inner*
                    # ``[-k:]`` Subtensor's output with a dynamic slice on
                    # the trimmed buffer.
                    client_redirects[i][k_cl] = (inner_cl, needed)
            elif (
                raw_length_consts[i] is not None
                and (
                    translated := _translate_positive_to_negative(
                        s0, raw_length_consts[i], init_l[i]
                    )
                )
                is not None
            ):
                # Constant raw length lets us translate a positive-form
                # client (``out[k]``, ``out[k:]``, ``out[k:m]``) into the
                # negative-form the existing trim logic expects. Splice the
                # leading axis only; trailing axes ride along verbatim.
                new_first, needed = translated
                slices[i][k_cl] = (new_first, *this_slice[1:])  # type: ignore[index]
            elif isinstance(s0, slice):
                start = s0.start
                if isinstance(start, Constant):
                    start = start.data
                # ``[:]``, ``[0:]`` and ``[init_l:]`` access the buffer
                # from its natural start; trimming would require a roll
                # to realign the circular buffer, more expensive than
                # keeping everything.
                if start in (0, None, init_l[i]):
                    store_steps[i] = 0
                    break
                # Only forward ``[-k:...]`` has an obvious buffer-size
                # answer (``k``). Reverse / non-positive steps walk back
                # to index 0 and would need the full buffer.
                s_step = None if s0.step is None else _maybe_constant_int(s0.step)
                forward = s0.step is None or (s_step is not None and s_step > 0)
                s_start = None if s0.start is None else _maybe_constant_int(s0.start)
                needed = (
                    abs(s_start)
                    if forward and s_start is not None and s_start < 0
                    else None
                )
            else:
                idx = _maybe_constant_int(s0)
                needed = abs(idx) if (idx is not None and idx < 0) else None

            if needed is None:
                store_steps[i] = 0
                break

            preallocable = op_info.n_mit_mot <= i <= last_sitsot_idx
            needed = max(needed, init_l[i])
            pval = (
                max(needed, init_l[i] + 1)
                if (prealloc_outs and preallocable)
                else needed
            )
            if store_steps[i] != -1:
                pval = max(pval, store_steps[i])
            store_steps[i] = pval
            flag_store = True

    # A clientless mit_sot / sit_sot may still be read by the inner
    # recurrence; keep the minimum its taps need (plus one slot under prealloc).
    for i in range(
        op_info.n_mit_mot,
        op_info.n_mit_mot + op_info.n_mit_sot + op_info.n_sit_sot,
    ):
        if store_steps[i] == -1:
            store_steps[i] = init_l[i] + (1 if prealloc_outs else 0)
            flag_store = True
    # Remaining -1s are unused nit_sots; leave their buffers untouched (0 =
    # keep all) -- scan_remove_unused will drop them entirely.
    store_steps = [0 if x == -1 else x for x in store_steps]

    # 3. is there anything to change ?
    if flag_store:
        # 3.1 initialize inputs for the new scan
        old_outputs = []
        nw_inputs = list(node.inputs)

        # Precompute the static lower bound on ``n_steps`` once
        # the buffer rebuild loop below uses it to decide whether each prealloc cap is statically a no-op.
        n_steps_static_min = _n_steps_static_min(n_steps)

        # 3.2. compose replace pairs for those nodes that need not store everything in memory
        replaced_outs = []
        offset = 1 + op_info.n_seqs + op_info.n_mit_mot
        for idx, val in enumerate(store_steps[op_info.n_mit_mot :]):
            i = idx + op_info.n_mit_mot
            if not (isinstance(val, int) and val <= 0):
                # If the memory for this output has been pre-allocated
                # before going into the scan op (by an alloc node)
                if idx < op_info.n_mit_sot + op_info.n_sit_sot:
                    taps = init_l[i]
                    nw_input = nw_inputs[offset + idx]

                    # Recreate default buffers with new size
                    if _is_default_scan_buffer(nw_input, taps):
                        extra_size = val - taps
                        # Cap prealloc extra entries at n_steps to avoid uninitialized slots
                        # when n_steps < extra_size (#1878). Skip the symbolic ``minimum``
                        # when we can prove it's not needed
                        cap_provable = (
                            n_steps_static_min is not None
                            and n_steps_static_min >= extra_size
                        )
                        if extra_size > 0 and not cap_provable:
                            extra_size = minimum(extra_size, n_steps)
                        nw_input = expand_empty(nw_input.owner.inputs[1], extra_size)
                    # Otherwise, just trim with a slice
                    else:
                        nw_input = nw_input[:val]

                    nw_inputs[offset + idx] = nw_input
                    replaced_outs.append(op_info.n_mit_mot + idx)
                # If there is no memory pre-allocated for this output
                elif idx < op_info.n_mit_sot + op_info.n_sit_sot + op_info.n_nit_sot:
                    pos = (
                        op_info.n_mit_mot
                        + idx
                        + op_info.n_seqs
                        + 1
                        + op_info.n_untraced_sit_sot
                    )
                    if nw_inputs[pos] == node.inputs[0]:
                        nw_inputs[pos] = pt.as_tensor_variable(val)
                    replaced_outs.append(op_info.n_mit_mot + idx)
                else:
                    continue

                # Record the targets to rewire after the rebuild. For
                # redirected clients (un-merged ``[-k:]`` chain) the target
                # is the deeper ``[-k:]`` Subtensor's output and the strip is
                # then orphaned; for everything else it's the client's
                # output directly.
                odx = op_info.n_mit_mot + idx
                old_outputs.append(
                    (
                        odx,
                        [
                            (
                                client_redirects[odx][k][0]
                                if k in client_redirects[odx]
                                else cl
                            ).outputs[0]
                            for k, (cl, _) in enumerate(
                                fgraph.clients[node.outputs[odx]]
                            )
                        ],
                    )
                )
        # 3.3. Recreate the same scan with new outer inputs.
        new_outs = op(*nw_inputs, return_list=True)

        old_new = []
        # 3.4. Get replace pairs for outputs with reduced buffers.
        # Negative indices are preserved as-is since n_steps is unchanged
        # and the circular buffer correctly maps them.
        n_steps = node.inputs[0]
        for pos, old_outs in old_outputs:
            new_raw = new_outs[pos]
            for k, old in enumerate(old_outs):
                if k in client_redirects[pos]:
                    # Redirect rebuild for ``scan_out[init_l:][-k:]``: point
                    # the user's ``xs[-k:]`` at the last ``min(k, n_steps)``
                    # elements of the *trimmed* buffer. Use a clean Python
                    # negative slice when the cap is provably a no-op,
                    # otherwise compute the start in positive form (avoids
                    # the switch/min/max tree ``get_canonical_form_slice``
                    # would emit for a negative Variable start).
                    _, k_const = client_redirects[pos][k]
                    if n_steps_static_min is not None and n_steps_static_min >= k_const:
                        new_o = new_raw[-k_const:]
                    else:
                        new_o = new_raw[new_raw.shape[0] - minimum(k_const, n_steps) :]
                else:
                    new_o = new_raw[tuple(slices[pos][k])]
                old_new.append((old, new_o))

        # 3.5. Get replace pairs for all other nodes
        for idx, o in enumerate(node.outputs):
            if idx not in replaced_outs:
                old_new += [(o, new_outs[idx])]
        # Check if the new outputs depend on the old scan node
        old_scan_is_used = [apply_depends_on(new.owner, node) for old, new in old_new]
        if any(old_scan_is_used):
            return False

        replacements = dict(old_new)
        replacements["remove"] = [node]
        return replacements

    return False


@node_rewriter([Scan])
def scan_reduce_trace_prealloc(fgraph, node):
    return scan_reduce_trace_rewrite(
        fgraph, node, backend_supports_output_pre_allocation=True
    )


@node_rewriter([Scan])
def scan_reduce_trace_no_prealloc(fgraph, node):
    return scan_reduce_trace_rewrite(
        fgraph, node, backend_supports_output_pre_allocation=False
    )


@node_rewriter([Scan])
def scan_sit_sot_to_untraced(fgraph, node):
    """Convert sit_sot with buffer size=1 to untraced_sit_sot.

    After scan_reduce_trace has reduced buffer sizes, sit_sot outputs that only
    need one state stored (buffer size=1) can be converted to untraced_sit_sot,
    which avoids the overhead of reading/writing circular buffers each iteration.
    """
    op = node.op
    info = op.info

    if info.n_sit_sot == 0:
        return False

    outer_sitsot = op.outer_sitsot(node.inputs)
    convertible = [
        idx for idx in range(info.n_sit_sot) if outer_sitsot[idx].type.shape[0] == 1
    ]

    if not convertible:
        return False

    convertible_set = set(convertible)

    # Thaw the frozen inner graph; the reassembled category slices below are
    # handed to a new ``Scan`` and must be mutable.
    unfrozen_fgraph = op.fgraph.unfreeze()
    inner_inputs = list(unfrozen_fgraph.inputs)
    inner_outputs = list(unfrozen_fgraph.outputs)

    inner_sitsot_ins = op.inner_sitsot(inner_inputs)
    inner_sitsot_outs = op.inner_sitsot_outs(inner_outputs)
    inner_untraced_ins = op.inner_untraced_sit_sot(inner_inputs)
    inner_untraced_outs = op.inner_untraced_sit_sot_outs(inner_outputs)

    # Split sit_sot into remaining and converted
    new_sit_sot_in_slices = []
    remaining_inner_sitsot_ins = []
    remaining_inner_sitsot_outs = []
    remaining_outer_sitsot = []
    converted_inner_untraced_ins = []
    converted_inner_untraced_outs = []
    converted_outer_untraced = []

    for idx in range(info.n_sit_sot):
        if idx in convertible_set:
            converted_inner_untraced_ins.append(inner_sitsot_ins[idx])
            converted_inner_untraced_outs.append(inner_sitsot_outs[idx])
            converted_outer_untraced.append(outer_sitsot[idx][0])
        else:
            new_sit_sot_in_slices.append(info.sit_sot_in_slices[idx])
            remaining_inner_sitsot_ins.append(inner_sitsot_ins[idx])
            remaining_inner_sitsot_outs.append(inner_sitsot_outs[idx])
            remaining_outer_sitsot.append(outer_sitsot[idx])

    # Rebuild inner inputs:
    # seqs | mit_mot_taps | mit_sot_taps | sit_sot | untraced_sit_sot | non_seqs
    n_taps_before_sitsot = sum(
        len(x) for x in chain(info.mit_mot_in_slices, info.mit_sot_in_slices)
    )
    pre_sitsot_inner = inner_inputs[: info.n_seqs + n_taps_before_sitsot]
    inner_non_seqs = op.inner_non_seqs(inner_inputs)

    new_inner_inputs = (
        pre_sitsot_inner
        + remaining_inner_sitsot_ins
        + list(inner_untraced_ins)
        + converted_inner_untraced_ins
        + inner_non_seqs
    )

    # Rebuild inner outputs:
    # mit_mot_outs | mit_sot | sit_sot | nit_sot | untraced_sit_sot [| while_cond]
    n_mit_mot_outs = sum(len(x) for x in info.mit_mot_out_slices)
    pre_sitsot_inner_outs = inner_outputs[: n_mit_mot_outs + info.n_mit_sot]
    nitsot_outs = op.inner_nitsot_outs(inner_outputs)

    new_inner_outputs = (
        pre_sitsot_inner_outs
        + remaining_inner_sitsot_outs
        + nitsot_outs
        + list(inner_untraced_outs)
        + converted_inner_untraced_outs
    )
    if info.as_while:
        new_inner_outputs.append(inner_outputs[-1])

    # Rebuild outer inputs:
    # n_steps | seqs | mit_mot | mit_sot | sit_sot | untraced_sit_sot | nit_sot | non_seqs
    pre_sitsot_outer = list(
        node.inputs[: 1 + info.n_seqs + info.n_mit_mot + info.n_mit_sot]
    )
    outer_untraced = list(op.outer_untraced_sit_sot(node.inputs))
    outer_nitsot = list(op.outer_nitsot(node.inputs))
    outer_non_seqs = list(op.outer_non_seqs(node.inputs))

    new_outer_inputs = (
        pre_sitsot_outer
        + remaining_outer_sitsot
        + outer_untraced
        + converted_outer_untraced
        + outer_nitsot
        + outer_non_seqs
    )

    # Build new ScanInfo
    new_info = dataclasses.replace(
        info,
        sit_sot_in_slices=tuple(new_sit_sot_in_slices),
        n_untraced_sit_sot=info.n_untraced_sit_sot + len(convertible),
    )

    new_op = Scan(
        new_inner_inputs,
        new_inner_outputs,
        new_info,
        mode=op.mode,
        profile=op.profile,
        truncate_gradient=op.truncate_gradient,
        name=op.name,
        allow_gc=op.allow_gc,
    )
    new_outs = cast(list[TensorVariable], new_op(*new_outer_inputs, return_list=True))

    # Build replacement mapping
    # Old outer outputs: mit_mot | mit_sot | sit_sot | nit_sot | untraced_sit_sot
    # New outer outputs: mit_mot | mit_sot | remaining_sit_sot | nit_sot | old_untraced | converted_untraced
    old_outputs = node.outputs
    replacements: dict = {}

    # mit_mot + mit_sot: same relative positions
    n_pre = info.n_mit_mot + info.n_mit_sot
    for i in range(n_pre):
        replacements[old_outputs[i]] = new_outs[i]

    # sit_sot: remaining keep position, converted become untraced
    old_sitsot_offset = n_pre
    new_remaining_offset = new_info.n_mit_mot + new_info.n_mit_sot
    new_converted_offset = (
        new_info.n_mit_mot
        + new_info.n_mit_sot
        + new_info.n_sit_sot
        + new_info.n_nit_sot
        + info.n_untraced_sit_sot
    )
    remaining_count = 0
    converted_count = 0
    for idx in range(info.n_sit_sot):
        old_out = old_outputs[old_sitsot_offset + idx]
        if idx in convertible_set:
            new_untraced = new_outs[new_converted_offset + converted_count]
            replacements[old_out] = pt.expand_dims(new_untraced, 0)
            converted_count += 1
        else:
            replacements[old_out] = new_outs[new_remaining_offset + remaining_count]
            remaining_count += 1

    # nit_sot
    old_nitsot_offset = n_pre + info.n_sit_sot
    new_nitsot_offset = new_info.n_mit_mot + new_info.n_mit_sot + new_info.n_sit_sot
    for i in range(info.n_nit_sot):
        replacements[old_outputs[old_nitsot_offset + i]] = new_outs[
            new_nitsot_offset + i
        ]

    # Original untraced_sit_sot
    old_untraced_offset = n_pre + info.n_sit_sot + info.n_nit_sot
    new_untraced_offset = (
        new_info.n_mit_mot
        + new_info.n_mit_sot
        + new_info.n_sit_sot
        + new_info.n_nit_sot
    )
    for i in range(info.n_untraced_sit_sot):
        replacements[old_outputs[old_untraced_offset + i]] = new_outs[
            new_untraced_offset + i
        ]

    replacements["remove"] = [node]
    return replacements
