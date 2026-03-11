from collections.abc import Sequence
from dataclasses import dataclass
from functools import reduce
from typing import Any

import numpy as np

from pytensor.graph.basic import Constant, Variable
from pytensor.graph.replace import graph_replace
from pytensor.graph.traversal import graph_inputs, truncated_graph_inputs
from pytensor.scan.op import Scan, ScanInfo
from pytensor.scan.utils import expand_empty
from pytensor.tensor import TensorVariable, as_tensor, minimum


@dataclass(frozen=True)
class ShiftedArg:
    x: Any
    by: tuple[int, ...]


@dataclass(frozen=True)
class InnerShiftedArg:
    x: Any
    by: tuple[int, ...]
    readonly: bool
    update: Variable | None = None

    def push(self, x: Variable) -> "InnerShiftedArg":
        if self.readonly:
            raise ValueError(
                "Cannot push to a read-only ShiftedArg (xs shifts cannot be updated)"
            )
        if self.update is not None:
            raise ValueError("ShiftedArg can only have a value pushed once")
        return type(self)(x=self.x, by=self.by, update=x, readonly=self.readonly)

    def __getitem__(self, idx):
        if -len(self.by) <= idx < len(self.by):
            return self.x[idx]
        else:
            raise IndexError()

    def __len__(self):
        return len(self.by)


def shift(x: Any, by: int | Sequence[int] = -1):
    by = (by,) if isinstance(by, int) else tuple(by)
    if by != tuple(sorted(by)):
        raise ValueError(f"by entries must be sorted, got {by}")
    if min(by) < 0 and max(by) >= 0:
        raise ValueError(
            f"by cannot contain both negative and non-negative entries, got {by}"
        )
    # TODO: If shape is known statically, validate the input is as big as the min/max taps
    return ShiftedArg(x, by=by)


def flatten_tree(x, subsume_none: bool = False) -> tuple[tuple, Any]:
    def recurse(e, spec):
        match e:
            case tuple() | list():
                e_spec = []
                for e_i in e:
                    yield from recurse(e_i, e_spec)
                if isinstance(e, tuple):
                    e_spec = tuple(e_spec)
                    spec.append(e_spec)
            case None if subsume_none:
                spec.append(None)
            case x:
                spec.append("x")
                yield x

    spec: list[Any] = []
    flat_inputs = tuple(recurse(x, spec=spec))
    return flat_inputs, spec[0]


def unflatten_tree(x, spec):
    def recurse(x_iter, spec):
        match spec:
            case "x":
                return next(x_iter)
            case None:
                return None
            case tuple():
                return tuple(recurse(x_iter, e_spec) for e_spec in spec)
            case list():
                return [recurse(x_iter, e_spec) for e_spec in spec]
            case _:
                raise ValueError(f"Unrecognized spec: {spec}")

    iter_x = iter(x)
    res = recurse(iter_x, spec=spec)
    # Check we consumed the whole iterable
    try:
        next(iter_x)
    except StopIteration:
        return res
    else:
        raise ValueError(f"x {x} has more entries than expected from the spec: {spec}")


def loop(
    f,
    init,
    xs=None,
    length=None,
    truncate_gradient=False,
    **scan_kwargs,
):
    # Flatten and process user init and xs
    init_flat, init_tree = flatten_tree(init, subsume_none=True)
    init_flat = [
        i if isinstance(i, (Variable, ShiftedArg)) else as_tensor(i) for i in init_flat
    ]

    # Convert to inner inputs, (also learn about how they map to Scan outputs_info semantics)
    mit_sot_idxs = []
    sit_sot_idxs = []
    implicit_sit_sot_idxs = []
    untraced_sit_sot_idxs = []
    init_flat_inner_with_shifts = []
    for i, init_i in enumerate(init_flat):
        if isinstance(init_i, ShiftedArg):
            if max(init_i.by) >= 0:
                raise ValueError(f"Init shifts must be negative, got by={i.by}")
            elem_type = init_i.x.type.clone(shape=init_i.x.type.shape[1:])
            init_inner = InnerShiftedArg(
                x=[elem_type() for _ in init_i.by], by=init_i.by, readonly=False
            )
            if min(init_i.by) < -1:
                mit_sot_idxs.append(i)
            else:
                sit_sot_idxs.append(i)
            init_flat_inner_with_shifts.append(init_inner)
        elif isinstance(init_i, TensorVariable):
            implicit_sit_sot_idxs.append(i)
            init_flat_inner_with_shifts.append(init_i.type())
        else:
            untraced_sit_sot_idxs.append(i)
            init_flat_inner_with_shifts.append(init_i.type())

    # Do the same for sequences
    xs_flat, x_tree = flatten_tree(xs, subsume_none=True)
    xs_flat = [
        x if isinstance(x, (Variable, ShiftedArg)) else as_tensor(x) for x in xs_flat
    ]
    xs_flat_inner_with_shifts = []
    for x in xs_flat:
        if isinstance(x, ShiftedArg):
            if min(x.by) < 0:
                raise ValueError(f"Sequence shifts must be non-negative, got by={x.by}")
            elem_type = x.x.type.clone(shape=x.x.type.shape[1:])
            xs_flat_inner_with_shifts.append(
                InnerShiftedArg(x=[elem_type() for _ in x.by], by=x.by, readonly=True)
            )
        elif isinstance(x, TensorVariable):
            xs_flat_inner_with_shifts.append(x.type.clone(shape=x.type.shape[1:])())
        else:
            raise ValueError(f"xs must be TensorVariable got {x} of type {type(x)}")

    # Obtain inner outputs
    res = f(
        unflatten_tree(init_flat_inner_with_shifts, init_tree),
        unflatten_tree(xs_flat_inner_with_shifts, x_tree),
    )
    ys_inner, break_cond_inner = None, None
    match res:
        case (update_inner, ys_inner):
            pass
        case (update_inner, ys_inner, break_cond_inner):
            pass
        case _:
            raise ValueError("Scan f must return a tuple with 2 or 3 outputs")

    # Validate outputs
    update_flat_inner_with_shifts, update_tree = flatten_tree(
        update_inner, subsume_none=True
    )
    if init_tree != update_tree:
        raise ValueError(
            "The update expression (first output of f) does not match the init expression (first input of f), ",
            f"expected: {init_tree}, got: {update_tree}",
        )
    update_flat_inner = []
    for u in update_flat_inner_with_shifts:
        if isinstance(u, InnerShiftedArg):
            if u.update is None:
                raise ValueError(f"No update pushed for shifted argument {u}")
            update_flat_inner.append(u.update)
        else:
            update_flat_inner.append(u)

    ys_flat_inner, y_tree = flatten_tree(ys_inner, subsume_none=True)
    for y in ys_flat_inner:
        if not isinstance(y, TensorVariable):
            raise TypeError(
                f"ys outputs must be TensorVariables, got {y} of type {type(y)}. "
                "Non-traceable types like RNG states should be carried in init, not returned as ys."
            )

    if break_cond_inner is not None:
        # TODO: validate
        raise NotImplementedError

    # Get inputs aligned for Scan and unpack ShiftedArgs
    scan_inner_inputs, _ = flatten_tree(
        (
            [
                s.x if isinstance(s, InnerShiftedArg) else s
                for s in xs_flat_inner_with_shifts
            ],
            [init_flat_inner_with_shifts[idx].x for idx in mit_sot_idxs],
            [init_flat_inner_with_shifts[idx].x for idx in sit_sot_idxs],
            [init_flat_inner_with_shifts[idx] for idx in implicit_sit_sot_idxs],
            [init_flat_inner_with_shifts[idx] for idx in untraced_sit_sot_idxs],
        )
    )
    # Get outputs aligned for Scan and unpack ShiftedArgs.update
    scan_inner_outputs, _ = flatten_tree(
        (
            [update_flat_inner[idx] for idx in mit_sot_idxs],
            [update_flat_inner[idx] for idx in sit_sot_idxs],
            [update_flat_inner[idx] for idx in implicit_sit_sot_idxs],
            ys_flat_inner,
            [update_flat_inner[idx] for idx in untraced_sit_sot_idxs],
            break_cond_inner,
        ),
        subsume_none=True,
    )

    # TODO: if any of the ys, is the same as the update values, we could return a slice of the trace
    #  that discards the initial values and reduced the number of nit_sots
    #  (useful even if there's already a Scan rewrite for this)

    # Use graph analysis to get the smallest closure of loop-invariant constants
    # Expand ShiftedArgs into their individual tap variables for graph analysis
    def _find_scan_constants(inputs, outputs) -> list[Variable]:
        def _depends_only_on_constants(var: Variable) -> bool:
            if isinstance(var, Constant):
                return True
            if var.owner is None:
                return False
            return all(isinstance(v, Constant) for v in graph_inputs([var]))

        inputs_set = set(inputs)
        return [
            arg
            for arg in truncated_graph_inputs(outputs, inputs)
            if (arg not in inputs_set and not _depends_only_on_constants(arg))
        ]

    constants = _find_scan_constants(scan_inner_inputs, scan_inner_outputs)
    inner_constants = [c.type() for c in constants]
    if inner_constants:
        # These constants belong to the outer graph, we need to remake inner outputs using dummies
        scan_inner_inputs = (*scan_inner_inputs, *inner_constants)
        scan_inner_outputs = graph_replace(
            scan_inner_outputs,
            replace=tuple(zip(constants, inner_constants)),
            strict=True,
        )

    # Now build Scan Op
    info = ScanInfo(
        n_seqs=sum(
            len(x.by) if isinstance(x, InnerShiftedArg) else 1
            for x in xs_flat_inner_with_shifts
        ),
        mit_mot_in_slices=(),
        mit_mot_out_slices=(),
        mit_sot_in_slices=tuple(init_flat[idx].by for idx in mit_sot_idxs),
        sit_sot_in_slices=((-1,),) * (len(sit_sot_idxs) + len(implicit_sit_sot_idxs)),
        n_untraced_sit_sot=len(untraced_sit_sot_idxs),
        n_nit_sot=len(ys_flat_inner),
        n_non_seqs=len(inner_constants),
        as_while=break_cond_inner is not None,
    )

    scan_op = Scan(
        list(scan_inner_inputs),
        list(scan_inner_outputs),
        info,
        truncate_gradient=truncate_gradient,
        strict=True,
        **scan_kwargs,
    )

    # Create outer sequences (learning about their length as we go)
    outer_sequences = []
    sequences_lengths = []
    for x in xs_flat:
        if isinstance(x, ShiftedArg):
            maxtap = max(x.by)
            sequences_lengths.append(x.x.shape[0] - maxtap)
            for start in x.by:
                end = None if start == maxtap else -(maxtap - start)
                outer_sequences.append(x.x[start:end])
        else:
            sequences_lengths.append(x.shape[0])
            outer_sequences.append(x)

    if length is not None:
        n_steps = as_tensor(length)
    elif sequences_lengths:
        n_steps = reduce(minimum, sequences_lengths)
    else:
        raise ValueError("length must be provided when there are no xs")

    # Build outer input traces with as many entries as n_steps + lags
    mit_sot_outer_inputs = [
        expand_empty(init_flat[idx].x, n_steps) for idx in mit_sot_idxs
    ]
    sit_sot_outer_inputs = [
        expand_empty(init_flat[idx].x, n_steps) for idx in sit_sot_idxs
    ]
    implicit_sit_sot_outer_inputs = [
        expand_empty(init_flat[idx], n_steps, new_dim=True)
        for idx in implicit_sit_sot_idxs
    ]
    untraced_sit_sot_outer_inputs = [init_flat[idx] for idx in untraced_sit_sot_idxs]

    scan_outer_inputs, _ = flatten_tree(
        (
            n_steps,
            outer_sequences,
            mit_sot_outer_inputs,
            sit_sot_outer_inputs,
            implicit_sit_sot_outer_inputs,
            untraced_sit_sot_outer_inputs,
            ((n_steps,) * info.n_nit_sot),
            constants,
        )
    )
    scan_outputs = scan_op(*scan_outer_inputs, return_list=True)

    # Extract final values from traced/untraced_outputs
    final_values, _ = flatten_tree(
        (
            [mit_sot[-1] for mit_sot in scan_op.outer_mitsot_outs(scan_outputs)],
            [sit_sot[-1] for sit_sot in scan_op.outer_sitsot_outs(scan_outputs)],
            scan_op.outer_untraced_sit_sot_outs(scan_outputs),
        )
    )
    # These need to be reordered to the user order
    flat_idxs, _ = flatten_tree(
        (mit_sot_idxs, sit_sot_idxs, implicit_sit_sot_idxs, untraced_sit_sot_idxs)
    )
    final = unflatten_tree(
        [final_values[rev_idx] for rev_idx in np.argsort(flat_idxs)], init_tree
    )
    ys = unflatten_tree(scan_op.outer_nitsot_outs(scan_outputs), y_tree)

    return final, ys
