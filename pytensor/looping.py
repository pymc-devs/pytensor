"""Build symbolic loops with explicit carry state and structured inputs and outputs."""

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
    """A tensor and the time offsets to expose within a :func:`loop` step."""

    x: Any
    by: tuple[int, ...]


@dataclass(frozen=True)
class InnerShiftedArg:
    """Time slices of a shifted argument passed to a :func:`loop` step.

    Indexing and iteration follow the order of the offsets given to :func:`shift`.
    Return the value of :meth:`push` as the updated carry to advance its history.
    Sequence slices are read-only.
    """

    x: Any
    by: tuple[int, ...]
    readonly: bool
    update: Variable | None = None

    def push(self, x: Variable) -> "InnerShiftedArg":
        """Return a shifted carry with ``x`` as its next value."""
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
    """Expose selected time offsets of a tensor inside :func:`loop`.

    Parameters
    ----------
    x : tensor_like
        For a carry, the leading axis contains the initial history, from oldest
        to newest. For a sequence, it is the axis to iterate over.
    by : int or sequence of int, default -1
        Nonempty, strictly increasing offsets. Carry offsets must be negative;
        sequence offsets must be nonnegative. A carry with an oldest offset of
        ``-k`` needs at least ``k`` initial entries; extra entries are ignored.

    Returns
    -------
    ShiftedArg
        An argument to place in ``init`` or ``xs``. The step receives an
        :class:`InnerShiftedArg` exposing one slice per offset.
    """
    by = (by,) if isinstance(by, int | np.integer) else tuple(by)
    if not by or any(not isinstance(b, int | np.integer) for b in by):
        raise ValueError("by must contain one or more integer offsets")
    if by != tuple(sorted(by)):
        raise ValueError(f"by entries must be sorted, got {by}")
    if len(set(by)) != len(by):
        raise ValueError(f"by entries must be distinct, got {by}")
    if min(by) < 0 and max(by) >= 0:
        raise ValueError(
            f"by cannot contain both negative and non-negative entries, got {by}"
        )
    x = as_tensor(x)
    if x.ndim == 0:
        raise ValueError("shift requires a tensor with a leading time axis")
    if min(by) < 0 and x.type.shape[0] is not None and x.type.shape[0] < -min(by):
        raise ValueError(f"Shifted carry needs at least {-min(by)} initial entries")
    return ShiftedArg(x, by=by)


@dataclass(frozen=True)
class _NamedTupleSpec:
    tuple_type: type
    children: tuple


def flatten_tree(x, subsume_none: bool = False) -> tuple[tuple, Any]:
    """Flatten lists, tuples, namedtuples and dictionaries with sortable keys.

    Return the leaves and a specification accepted by :func:`unflatten_tree`.
    When ``subsume_none`` is true, ``None`` is preserved without becoming a leaf.
    Dictionary keys are sorted to align inputs and outputs independently of
    their insertion order.
    """

    def recurse(e, spec):
        match e:
            case tuple() | list():
                e_spec = []
                for e_i in e:
                    yield from recurse(e_i, e_spec)
                if isinstance(e, tuple):
                    e_spec = tuple(e_spec)
                    if hasattr(e, "_fields"):
                        e_spec = _NamedTupleSpec(type(e), e_spec)
                spec.append(e_spec)
            case dict():
                e_spec = {}
                for key in sorted(e):
                    child_spec = []
                    yield from recurse(e[key], child_spec)
                    e_spec[key] = child_spec[0]
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
    """Rebuild a tree from its leaves and a :func:`flatten_tree` specification."""

    def recurse(x_iter, spec):
        match spec:
            case "x":
                return next(x_iter)
            case None:
                return None
            case _NamedTupleSpec(tuple_type, children):
                return tuple_type(*[recurse(x_iter, e_spec) for e_spec in children])
            case tuple():
                return tuple([recurse(x_iter, e_spec) for e_spec in spec])
            case list():
                return [recurse(x_iter, e_spec) for e_spec in spec]
            case dict():
                return {key: recurse(x_iter, e_spec) for key, e_spec in spec.items()}
            case _:
                raise ValueError(f"Unrecognized spec: {spec}")

    iter_x = iter(x)
    try:
        res = recurse(iter_x, spec=spec)
    except StopIteration as exc:
        raise ValueError(
            f"x {x} has fewer entries than expected from the spec: {spec}"
        ) from exc
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
    truncate_gradient=-1,
    **scan_kwargs,
):
    """Apply a recurrence and return its final carry and stacked outputs.

    Parameters
    ----------
    f : callable
        Called once to build the symbolic step, as ``f(carry, x)``. Return
        ``(next_carry, y)`` or ``(next_carry, y, stop)``. ``next_carry`` must
        match the structure of ``init``; ``y`` contains tensors to stack along
        a new leading time axis, or ``None``. A scalar boolean ``stop`` ends
        the loop after the current step when true.
    init : tree
        Initial carry. Trees may contain tensors, RNG states, :func:`shift`
        arguments, ``None``, lists, tuples, namedtuples and dictionaries with
        sortable keys. Tensor carry shapes and dtypes must remain fixed.
    xs : tree, optional
        Tensors to iterate over along their leading axes, optionally wrapped
        in :func:`shift`. The step receives the corresponding tree of slices.
        ``None`` is passed to the step when there are no sequences.
    length : scalar integer, optional
        Number of steps, or maximum steps when ``f`` returns a stop condition.
        Defaults to the shortest sequence length after applying offsets.
        Required when there are no sequences. Must be nonnegative and cannot
        exceed any sequence's available length.
    truncate_gradient : int, default -1
        Number of steps to backpropagate through. ``-1`` differentiates through
        the entire loop.
    **scan_kwargs
        Additional :class:`pytensor.scan.op.Scan` options, such as ``name``,
        ``profile`` and ``allow_gc``. External symbolic variables referenced
        by ``f`` are captured automatically. Shared variable updates must be
        carried explicitly in ``init``.

    Returns
    -------
    final : tree
        Final carry with the structure of ``init``. Shifted histories are
        replaced by their last computed value.
    ys : tree
        Stacked step outputs with the structure of ``y``. The initial carry
        is excluded. RNG states can be carried but cannot be stacked.
    """
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
                raise ValueError(f"Init shifts must be negative, got by={init_i.by}")
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
            "The update expression (first output of f) does not match the init expression (first input of f), "
            f"expected: {init_tree}, got: {update_tree}"
        )
    update_flat_inner = []
    for u in update_flat_inner_with_shifts:
        if isinstance(u, InnerShiftedArg):
            if u.update is None:
                raise ValueError(f"No update pushed for shifted argument {u}")
            update_flat_inner.append(u.update)
        else:
            update_flat_inner.append(u if isinstance(u, Variable) else as_tensor(u))

    ys_flat_inner, y_tree = flatten_tree(ys_inner, subsume_none=True)
    for y in ys_flat_inner:
        if not isinstance(y, TensorVariable):
            raise TypeError(
                f"ys outputs must be TensorVariables, got {y} of type {type(y)}. "
                "Non-traceable types like RNG states should be carried in init, not returned as ys."
            )

    if break_cond_inner is not None:
        break_cond_inner = as_tensor(break_cond_inner)
        if break_cond_inner.ndim != 0 or break_cond_inner.dtype != "bool":
            raise TypeError("The stop condition must be a scalar boolean tensor")

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
    if n_steps.ndim != 0 or not n_steps.dtype.startswith(("int", "uint")):
        raise TypeError("length must be a scalar integer")

    # Build outer input traces with as many entries as n_steps + lags
    mit_sot_outer_inputs = [
        expand_empty(init_flat[idx].x[: -min(init_flat[idx].by)], n_steps)
        for idx in mit_sot_idxs
    ]
    sit_sot_outer_inputs = [
        expand_empty(init_flat[idx].x[:1], n_steps) for idx in sit_sot_idxs
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
