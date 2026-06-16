import dataclasses
from collections.abc import Collection
from typing import cast

from pytensor.graph.basic import Apply, Variable
from pytensor.graph.replace import clone_replace
from pytensor.scan.op import Scan


def _rebuild_scan_with_new_signature(
    op: Scan,
    node: Apply,
    *,
    drop_seqs: Collection[int] = frozenset(),
    drop_mit_mot: Collection[int] = frozenset(),
    drop_mit_sot: Collection[int] = frozenset(),
    drop_sit_sot: Collection[int] = frozenset(),
    drop_nit_sot: Collection[int] = frozenset(),
    drop_untraced_sit_sot: Collection[int] = frozenset(),
    drop_non_seqs: Collection[int] = frozenset(),
    inner_substitutions: dict[Variable, Variable] | None = None,
) -> dict:
    """Build a replacement Scan node with a trimmed signature.

    Each ``drop_*`` argument is a set of indices into its category; the
    rebuilt op retains only the entries whose index is not listed.
    ``inner_substitutions``, when provided, is applied via ``clone_replace``
    on the inner outputs before the rebuild -- use it to inline constants
    or rewire duplicate inner inputs.

    Returns a ``replacements`` dict: kept outer outputs map to their
    counterparts on the new op, dropped outputs carry no mapping (they
    disappear along with the old node), and ``"remove"`` lists the old
    node.
    """
    info = op.info

    keep_seqs = [k for k in range(info.n_seqs) if k not in drop_seqs]
    keep_mm = [k for k in range(info.n_mit_mot) if k not in drop_mit_mot]
    keep_ms = [k for k in range(info.n_mit_sot) if k not in drop_mit_sot]
    keep_ss = [k for k in range(info.n_sit_sot) if k not in drop_sit_sot]
    keep_ns = [k for k in range(info.n_nit_sot) if k not in drop_nit_sot]
    keep_us = [
        k for k in range(info.n_untraced_sit_sot) if k not in drop_untraced_sit_sot
    ]
    keep_non_seqs = [k for k in range(info.n_non_seqs) if k not in drop_non_seqs]

    new_info = dataclasses.replace(
        info,
        n_seqs=len(keep_seqs),
        mit_mot_in_slices=tuple(info.mit_mot_in_slices[k] for k in keep_mm),
        mit_mot_out_slices=tuple(info.mit_mot_out_slices[k] for k in keep_mm),
        mit_sot_in_slices=tuple(info.mit_sot_in_slices[k] for k in keep_ms),
        sit_sot_in_slices=tuple(info.sit_sot_in_slices[k] for k in keep_ss),
        n_nit_sot=len(keep_ns),
        n_untraced_sit_sot=len(keep_us),
        n_non_seqs=len(keep_non_seqs),
    )

    inner_seqs = op.inner_seqs(op.inner_inputs)
    inner_mm_groups = op.inner_mitmot_grouped(op.inner_inputs)
    inner_ms_groups = op.inner_mitsot_grouped(op.inner_inputs)
    inner_ss = op.inner_sitsot(op.inner_inputs)
    inner_us = op.inner_untraced_sit_sot(op.inner_inputs)
    inner_non_seqs = op.inner_non_seqs(op.inner_inputs)

    new_inner_inputs = (
        [inner_seqs[k] for k in keep_seqs]
        + [v for k in keep_mm for v in inner_mm_groups[k]]
        + [v for k in keep_ms for v in inner_ms_groups[k]]
        + [inner_ss[k] for k in keep_ss]
        + [inner_us[k] for k in keep_us]
        + [inner_non_seqs[k] for k in keep_non_seqs]
    )

    inner_outputs = op.inner_outputs
    if inner_substitutions:
        inner_outputs = clone_replace(inner_outputs, replace=inner_substitutions)
    inner_mm_out_groups = op.inner_mitmot_outs_grouped(inner_outputs)
    inner_ms_outs = op.inner_mitsot_outs(inner_outputs)
    inner_ss_outs = op.inner_sitsot_outs(inner_outputs)
    inner_ns_outs = op.inner_nitsot_outs(inner_outputs)
    inner_us_outs = op.inner_untraced_sit_sot_outs(inner_outputs)
    # ``as_while`` appends the condition as the final inner output; preserve it.
    while_cond_tail = [inner_outputs[-1]] if info.as_while else []

    new_inner_outputs = (
        [v for k in keep_mm for v in inner_mm_out_groups[k]]
        + [inner_ms_outs[k] for k in keep_ms]
        + [inner_ss_outs[k] for k in keep_ss]
        + [inner_ns_outs[k] for k in keep_ns]
        + [inner_us_outs[k] for k in keep_us]
        + while_cond_tail
    )

    outer_seqs = op.outer_seqs(node.inputs)
    outer_mm = op.outer_mitmot(node.inputs)
    outer_ms = op.outer_mitsot(node.inputs)
    outer_ss = op.outer_sitsot(node.inputs)
    outer_us = op.outer_untraced_sit_sot(node.inputs)
    outer_ns = op.outer_nitsot(node.inputs)
    outer_non_seqs = op.outer_non_seqs(node.inputs)

    new_outer_inputs = (
        [node.inputs[0]]
        + [outer_seqs[k] for k in keep_seqs]
        + [outer_mm[k] for k in keep_mm]
        + [outer_ms[k] for k in keep_ms]
        + [outer_ss[k] for k in keep_ss]
        + [outer_us[k] for k in keep_us]
        + [outer_ns[k] for k in keep_ns]
        + [outer_non_seqs[k] for k in keep_non_seqs]
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
    new_outs = cast(list[Variable], new_op(*new_outer_inputs, return_list=True))

    # Outer outputs are laid out [mit_mot | mit_sot | sit_sot | nit_sot |
    # untraced_sit_sot]; walk each category and route each kept old output
    # to its new counterpart.
    replacements: dict = {}
    new_cursor = 0
    old_offset = 0
    for keep_list, n_old in (
        (keep_mm, info.n_mit_mot),
        (keep_ms, info.n_mit_sot),
        (keep_ss, info.n_sit_sot),
        (keep_ns, info.n_nit_sot),
        (keep_us, info.n_untraced_sit_sot),
    ):
        for k in keep_list:
            replacements[node.outputs[old_offset + k]] = new_outs[new_cursor]
            new_cursor += 1
        old_offset += n_old
    replacements["remove"] = [node]
    return replacements
