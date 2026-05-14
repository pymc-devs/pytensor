"""Rewrites that clean up the inputs / outputs of a Scan node.

These run under the ``scan_input_and_output_cleanup*`` registrations and
shrink the Scan signature without changing what it computes:

* :func:`scan_inline_invariant_constants` — inline outer ``Constant``
  non-sequences and uniform-value ``TensorConstant`` sequences.
* :func:`scan_merge_duplicate_inputs` — collapse outer seqs / non_seqs
  whose values are :func:`equal_computations`.
* :func:`scan_remove_unused` — drop inner-/outer-side state slots,
  sequences, and non-sequences that nobody downstream reads.
"""

from typing import NamedTuple

from pytensor.graph.basic import Constant, equal_computations
from pytensor.graph.rewriting.basic import node_rewriter
from pytensor.graph.traversal import explicit_graph_inputs
from pytensor.scan.op import Scan
from pytensor.scan.rewriting.utils import _rebuild_scan_with_new_signature
from pytensor.tensor.variable import TensorConstant


@node_rewriter([Scan])
def scan_inline_invariant_constants(fgraph, node):
    """Inline compile-time-constant, iteration-invariant Scan inputs.

    A non-sequence whose outer input is a ``Constant`` is replaced inside
    the inner graph by that constant. A sequence whose outer input is a
    ``TensorConstant`` with a uniform value is collapsed to a scalar
    constant. Once inlined, the inner graph can constant-fold through the
    value and the corresponding inner / outer input pair is dropped.
    """
    op = node.op
    drop_seqs: set = set()
    drop_non_seqs: set = set()
    substitutions: dict = {}

    for k, (inner, outer) in enumerate(
        zip(op.inner_seqs(op.inner_inputs), op.outer_seqs(node.inputs), strict=True)
    ):
        if isinstance(outer, TensorConstant) and outer.unique_value is not None:
            try:
                substitutions[inner] = outer[0]
                drop_seqs.add(k)
            except TypeError:
                pass

    for k, (inner, outer) in enumerate(
        zip(
            op.inner_non_seqs(op.inner_inputs),
            op.outer_non_seqs(node.inputs),
            strict=True,
        )
    ):
        if isinstance(outer, Constant):
            substitutions[inner] = outer
            drop_non_seqs.add(k)

    if not substitutions:
        return False
    return _rebuild_scan_with_new_signature(
        op,
        node,
        drop_seqs=drop_seqs,
        drop_non_seqs=drop_non_seqs,
        inner_substitutions=substitutions,
    )


@node_rewriter([Scan])
def scan_merge_duplicate_inputs(fgraph, node):
    """Merge outer seqs / non_seqs that are ``equal_computations``.

    When two outer inputs compute the same value, the later one's inner
    variable is rewired to the earlier one's, and the duplicate inner /
    outer input pair is dropped.
    """
    op = node.op

    def _duplicates(inner_list, outer_list):
        subs: dict = {}
        drop: set = set()
        canonical: list[tuple] = []
        for k, (inner, outer) in enumerate(zip(inner_list, outer_list, strict=True)):
            for canon_outer, canon_inner in canonical:
                if equal_computations([outer], [canon_outer]):
                    subs[inner] = canon_inner
                    drop.add(k)
                    break
            else:
                canonical.append((outer, inner))
        return subs, drop

    seq_subs, drop_seqs = _duplicates(
        op.inner_seqs(op.inner_inputs),
        op.outer_seqs(node.inputs),
    )
    non_seq_subs, drop_non_seqs = _duplicates(
        op.inner_non_seqs(op.inner_inputs),
        op.outer_non_seqs(node.inputs),
    )
    substitutions = {**seq_subs, **non_seq_subs}

    if not substitutions:
        return False
    return _rebuild_scan_with_new_signature(
        op,
        node,
        drop_seqs=drop_seqs,
        drop_non_seqs=drop_non_seqs,
        inner_substitutions=substitutions,
    )


class _RemoveUnusedCandidate(NamedTuple):
    category: str  # mit_mot / mit_sot / sit_sot / nit_sot / untraced_sit_sot
    category_idx: int  # within-category position
    taps: frozenset  # inner-input vars read by this state; empty for nit_sots
    out_positions: list[int]  # this state's slots in ``op.inner_outputs``


@node_rewriter([Scan])
def scan_remove_unused(fgraph, node):
    """Drop unused outputs and inputs from a Scan node.

    Drops:
      * State slots (mit_mot / mit_sot / sit_sot / nit_sot /
        untraced_sit_sot) whose outer output has no clients, provided none
        of their inner inputs is reached from any surviving inner output.
        Cross-dependent unused states are resolved together.
      * Sequences and non-sequences that the rebuilt inner graph no longer
        references.

    Partial-tap trimming of mit_mot / mit_sot is out of scope: a state is
    dropped as a whole or kept as a whole.
    """
    op = node.op
    info = op.info
    inner_inputs = op.inner_inputs
    inner_outputs = op.inner_outputs

    def _clientless(outer_idx):
        return not fgraph.clients.get(node.outputs[outer_idx])

    # Inner-output and outer-output positions by category -- needed because
    # the same Variable can occupy multiple inner-output slots, so dropping
    # must track slots positionally rather than by variable identity.
    mm_out_lens = [len(s) for s in info.mit_mot_out_slices]
    ms_out_pos_start = sum(mm_out_lens)
    ss_out_pos_start = ms_out_pos_start + info.n_mit_sot
    ns_out_pos_start = ss_out_pos_start + info.n_sit_sot
    us_out_pos_start = ns_out_pos_start + info.n_nit_sot

    outer_mm_start = 0
    outer_ms_start = info.n_mit_mot
    outer_ss_start = outer_ms_start + info.n_mit_sot
    outer_ns_start = outer_ss_start + info.n_sit_sot
    outer_us_start = outer_ns_start + info.n_nit_sot

    inner_mm_groups = op.inner_mitmot_grouped(inner_inputs)
    inner_ms_groups = op.inner_mitsot_grouped(inner_inputs)
    inner_ss = op.inner_sitsot(inner_inputs)
    inner_us = op.inner_untraced_sit_sot(inner_inputs)

    # Candidate state slots: those with no external outer clients.
    candidates: list[_RemoveUnusedCandidate] = []
    mm_group_starts = [sum(mm_out_lens[:k]) for k in range(info.n_mit_mot + 1)]
    candidates.extend(
        _RemoveUnusedCandidate(
            "mit_mot",
            k,
            frozenset(inner_mm_groups[k]),
            list(range(mm_group_starts[k], mm_group_starts[k + 1])),
        )
        for k in range(info.n_mit_mot)
        if _clientless(outer_mm_start + k)
    )
    candidates.extend(
        _RemoveUnusedCandidate(
            "mit_sot", k, frozenset(inner_ms_groups[k]), [ms_out_pos_start + k]
        )
        for k in range(info.n_mit_sot)
        if _clientless(outer_ms_start + k)
    )
    candidates.extend(
        _RemoveUnusedCandidate(
            "sit_sot", k, frozenset({inner_ss[k]}), [ss_out_pos_start + k]
        )
        for k in range(info.n_sit_sot)
        if _clientless(outer_ss_start + k)
    )
    candidates.extend(
        _RemoveUnusedCandidate("nit_sot", k, frozenset(), [ns_out_pos_start + k])
        for k in range(info.n_nit_sot)
        if _clientless(outer_ns_start + k)
    )
    candidates.extend(
        _RemoveUnusedCandidate(
            "untraced_sit_sot", k, frozenset({inner_us[k]}), [us_out_pos_start + k]
        )
        for k in range(info.n_untraced_sit_sot)
        if _clientless(outer_us_start + k)
    )

    # Fast path: nothing disconnected externally, so no state is droppable.
    # Only seq / non_seq staleness (from upstream input rewrites) could
    # remain -- one walk of the inner outputs covers that.
    if not candidates:
        final_outputs = inner_outputs
        droppable_state_idxs: set[int] = set()
    else:
        # A candidate is pinned (removed from ``droppable_idxs``) once any
        # of its taps is reached from a surviving inner output. On
        # pinning, we walk back from the candidate's own inner outputs
        # and fold any candidate taps they reach into ``reached_taps``,
        # so later candidates in the same pass see the update. The
        # fixpoint stays scoped to candidate taps -- seq / non_seq inputs
        # are handled in a single post-loop walk.
        all_candidate_taps: frozenset = frozenset().union(
            *(cand.taps for cand in candidates)
        )
        candidate_out_positions: set[int] = {
            pos for cand in candidates for pos in cand.out_positions
        }
        survivor_outputs = [
            out
            for pos, out in enumerate(inner_outputs)
            if pos not in candidate_out_positions
        ]
        reached_taps = all_candidate_taps & set(explicit_graph_inputs(survivor_outputs))
        droppable_state_idxs = set(range(len(candidates)))
        while True:
            changed = False
            for cand_idx in list(droppable_state_idxs):
                cand = candidates[cand_idx]
                if cand.taps & reached_taps:
                    droppable_state_idxs.discard(cand_idx)
                    reached_taps |= all_candidate_taps & set(
                        explicit_graph_inputs(
                            [inner_outputs[pos] for pos in cand.out_positions]
                        )
                    )
                    changed = True
            if not changed:
                break

        final_outputs = survivor_outputs + [
            inner_outputs[pos]
            for cand_idx, cand in enumerate(candidates)
            if cand_idx not in droppable_state_idxs
            for pos in cand.out_positions
        ]

    reached_inputs = set(explicit_graph_inputs(final_outputs))
    drop_seqs = {
        k
        for k, seq in enumerate(op.inner_seqs(inner_inputs))
        if seq not in reached_inputs
    }
    drop_non_seqs = {
        k
        for k, ns in enumerate(op.inner_non_seqs(inner_inputs))
        if ns not in reached_inputs
    }

    if not (droppable_state_idxs or drop_seqs or drop_non_seqs):
        return None

    drops_by_cat: dict[str, set[int]] = {
        "mit_mot": set(),
        "mit_sot": set(),
        "sit_sot": set(),
        "nit_sot": set(),
        "untraced_sit_sot": set(),
    }
    for cand_idx in droppable_state_idxs:
        cand = candidates[cand_idx]
        drops_by_cat[cand.category].add(cand.category_idx)

    return _rebuild_scan_with_new_signature(
        op,
        node,
        drop_seqs=drop_seqs,
        drop_mit_mot=drops_by_cat["mit_mot"],
        drop_mit_sot=drops_by_cat["mit_sot"],
        drop_sit_sot=drops_by_cat["sit_sot"],
        drop_nit_sot=drops_by_cat["nit_sot"],
        drop_untraced_sit_sot=drops_by_cat["untraced_sit_sot"],
        drop_non_seqs=drop_non_seqs,
    )
