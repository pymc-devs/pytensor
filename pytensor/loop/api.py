from collections.abc import Sequence
from dataclasses import dataclass
from functools import reduce
from typing import Any

import numpy as np

from pytensor.graph import FunctionGraph, Variable, clone_replace
from pytensor.graph.basic import Constant
from pytensor.graph.replace import graph_replace
from pytensor.graph.traversal import truncated_graph_inputs
from pytensor.loop.op import Scan
from pytensor.tensor import as_tensor, constant, empty_like, invert, minimum
from pytensor.tensor.variable import TensorVariable


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


def loop(f, init, xs=None, length=None):
    """Repeatedly apply ``f`` to a carried state, optionally iterating over sequences.

    Roughly equivalent to::

        def loop(f, init, xs, length):
            carry = init
            ys = []
            for x in zip(*xs):  # or range(length)
                carry, y, *break_cond = f(carry, x)
                ys.append(y)
                if break_cond and break_cond[0]:
                    break
            return carry, stack(ys)

    Parameters
    ----------
    f
        Function from ``(carry, x)`` to ``(carry, y)`` or ``(carry, y, break_cond)``.
        ``carry`` and ``x`` have the tree structures of ``init`` and ``xs``.
        When ``break_cond`` is returned and evaluates to True, the loop stops
        after the current step.
    init
        Tree of initial states. Entries can be variables of any type (states
        that are not tensors, like RandomGenerators, can be carried but not traced),
        ``shift(x, by=[-k, ..., -1])`` for states with multiple taps (where ``x``
        stacks the ``k`` initial values along a new leading dimension), or None
        for no carry.
    xs
        Tree of sequences to iterate over, scalar-indexed along their leading
        dimension. Entries can be tensor variables or ``shift(x, by=[0, ..., k])``
        for reading multiple positions per step.
    length
        Number of steps. Required when there are no ``xs``, otherwise the minimum
        of the sequence lengths (adjusted for shifts) and ``length`` is used.

    Returns
    -------
    (final, ys)
        Trees with the structures of ``init`` and of the ``y`` returned by ``f``,
        holding the final states and the stacked per-step outputs.
    """
    init_flat, init_tree = flatten_tree(init, subsume_none=True)
    init_flat = [
        leaf if isinstance(leaf, Variable | ShiftedArg) else as_tensor(leaf)
        for leaf in init_flat
    ]

    xs_flat, xs_tree = flatten_tree(xs, subsume_none=True)
    xs_flat = [
        leaf if isinstance(leaf, Variable | ShiftedArg) else as_tensor(leaf)
        for leaf in xs_flat
    ]

    idx0 = constant(np.array(0, dtype="int64"), name="idx")
    symbolic_idx = idx0.type(name="idx")

    # Create the inner inputs of the carry. Shifted entries become chains of
    # states covering every position in the tap window, of which only the
    # requested taps are exposed to `f`
    state_dummies: list[Variable] = []  # All inner state inputs, except idx
    carry_entries = []  # (kind, first_slot, n_states) per init leaf
    init_flat_inner = []
    for leaf in init_flat:
        if isinstance(leaf, ShiftedArg):
            if max(leaf.by) >= 0:
                raise ValueError(f"Init shifts must be negative, got by={leaf.by}")
            if not isinstance(leaf.x, TensorVariable) or leaf.x.type.ndim < 1:
                raise ValueError(
                    "Shifted init must be a tensor stacking the initial values along a new leading dimension"
                )
            n_chain = -min(leaf.by)
            elem_type = leaf.x.type.clone(shape=leaf.x.type.shape[1:])
            chain_dummies = [elem_type() for _ in range(n_chain)]
            carry_entries.append(("chain", len(state_dummies), n_chain))
            state_dummies.extend(chain_dummies)
            init_flat_inner.append(
                InnerShiftedArg(
                    x=[chain_dummies[n_chain + k] for k in leaf.by],
                    by=leaf.by,
                    readonly=False,
                )
            )
        else:
            carry_entries.append(("plain", len(state_dummies), 1))
            dummy = leaf.type()
            state_dummies.append(dummy)
            init_flat_inner.append(dummy)

    # Create the inner inputs of the sequences: reads of the (outer) sequences
    # at the current iteration index. The sequences themselves become implicit
    # constants of the loop
    xs_flat_inner = []
    for leaf in xs_flat:
        if isinstance(leaf, ShiftedArg):
            if min(leaf.by) < 0:
                raise ValueError(
                    f"Sequence shifts must be non-negative, got by={leaf.by}"
                )
            xs_flat_inner.append(
                InnerShiftedArg(
                    x=[
                        leaf.x[symbolic_idx + k if k else symbolic_idx] for k in leaf.by
                    ],
                    by=leaf.by,
                    readonly=True,
                )
            )
        elif isinstance(leaf, TensorVariable):
            xs_flat_inner.append(leaf[symbolic_idx])
        else:
            raise ValueError(f"xs must be TensorVariables, got {leaf}")

    res = f(
        unflatten_tree(init_flat_inner, init_tree),
        unflatten_tree(xs_flat_inner, xs_tree),
    )
    match res:
        case (update_inner, ys_inner):
            break_cond_inner = None
        case (update_inner, ys_inner, break_cond_inner):
            pass
        case _:
            raise ValueError("loop f must return a tuple with 2 or 3 outputs")

    # Validate the updates and collect the next states, expanding the chains
    update_flat, update_tree = flatten_tree(update_inner, subsume_none=True)
    if update_tree != init_tree:
        raise ValueError(
            "The update (first output of f) does not match the structure of init "
            f"(first input of f), expected: {init_tree}, got: {update_tree}"
        )
    next_states: list[Variable] = []
    for (kind, slot, n_states), update in zip(carry_entries, update_flat):
        if kind == "chain":
            if not isinstance(update, InnerShiftedArg) or update.update is None:
                raise ValueError(f"No update pushed for shifted argument {update}")
            next_states.extend(state_dummies[slot + 1 : slot + n_states])
            next_states.append(update.update)
        else:
            if isinstance(update, InnerShiftedArg):
                raise ValueError(
                    f"Shifted update returned for a non-shifted init {update}"
                )
            next_states.append(update)

    # Per-step outputs become states that don't read their previous value.
    # Outputs that are updates of the carry simply reuse that state's trace
    ys_flat, ys_tree = flatten_tree(ys_inner, subsume_none=True)
    ys_entries = []  # state position (after idx) whose trace holds each y
    y_init_values = []
    for y in ys_flat:
        if not isinstance(y, TensorVariable):
            raise TypeError(
                f"ys outputs must be TensorVariables, got {y} of type {type(y)}. "
                "Non-traceable types like RNG states should be carried in init, not returned as ys."
            )
        if any(y is next_state for next_state in next_states):
            ys_entries.append(next(i for i, n in enumerate(next_states) if n is y))
        else:
            # y may reference idx. We replace it by the initial value, so that
            # the shape of the dummy init state does not depend on it
            [y_at_first_step] = clone_replace(output=[y], replace={symbolic_idx: idx0})
            y_init = empty_like(y_at_first_step)
            y_init.name = "empty_init_state"
            ys_entries.append(len(next_states))
            state_dummies.append(y_init.type(name="dummy_state"))
            next_states.append(y)
            y_init_values.append(y_init)

    if break_cond_inner is None:
        while_cond = as_tensor(np.array(True))
    else:
        if getattr(break_cond_inner.type, "dtype", None) != "bool":
            raise TypeError(
                f"break_cond must be a boolean scalar, got {break_cond_inner}"
            )
        while_cond = invert(break_cond_inner)

    # Number of steps: minimum of the sequence lengths and the requested length
    lengths = []
    for leaf in xs_flat:
        if isinstance(leaf, ShiftedArg):
            lengths.append(leaf.x.shape[0] - max(leaf.by))
        else:
            lengths.append(leaf.shape[0])
    if length is not None:
        lengths.append(as_tensor(length))
    if not lengths:
        raise ValueError("length must be provided when there are no xs")
    n_steps = reduce(minimum, lengths)

    # Assemble the initial values, in the order of the state dummies
    init_values = []
    for (kind, _, n_states), leaf in zip(carry_entries, init_flat):
        if kind == "chain":
            init_values.extend(leaf.x[j] for j in range(n_states))
        else:
            init_values.append(leaf)
    init_values.extend(y_init_values)

    fgraph_inputs = [symbolic_idx, *state_dummies]
    fgraph_outputs = [while_cond, symbolic_idx + 1, *next_states]

    # The remaining inputs of the loop (including the sequences read by their
    # index) are loop-invariant constants. They cannot be used directly as
    # inputs of the inner function graph, so they are replaced by dummies
    outer_constants = [
        leaf.x if isinstance(leaf, ShiftedArg) else leaf for leaf in xs_flat
    ]
    outer_constants.extend(
        inp
        for inp in truncated_graph_inputs(
            fgraph_outputs, ancestors_to_include=fgraph_inputs + outer_constants
        )
        if not isinstance(inp, Constant)
        and inp not in fgraph_inputs
        and inp not in outer_constants
    )
    inner_constants = [c.type() for c in outer_constants]
    if outer_constants:
        fgraph_outputs = graph_replace(
            fgraph_outputs,
            dict(zip(outer_constants, inner_constants)),
            strict=False,
        )

    # State shapes are loop-invariant, but shape inference may type the updated
    # states more weakly than the initial states. Coerce them back when needed
    while_cond_out, next_idx, *next_states = fgraph_outputs
    next_states = [
        next_state
        if dummy.type.is_super(next_state.type)
        else dummy.type.filter_variable(next_state)
        for dummy, next_state in zip(state_dummies, next_states)
    ]

    update_fg = FunctionGraph(
        inputs=[symbolic_idx, *state_dummies, *inner_constants],
        outputs=[while_cond_out, next_idx, *next_states],
    )

    scan_op = Scan(update_fg=update_fg, sequences=range(len(xs_flat)))
    scan_outs = scan_op(n_steps, idx0, *init_values, *outer_constants)
    n_states = scan_op.n_states
    finals = scan_outs[1:n_states]  # The idx final state is dropped
    traces = scan_outs[n_states + 1 :]

    # The final value of a shifted carry is its most recent state
    final_leaves = [
        finals[slot + n_states_entry - 1] for (_, slot, n_states_entry) in carry_entries
    ]
    ys_leaves = [traces[pos] for pos in ys_entries]

    return (
        unflatten_tree(final_leaves, init_tree),
        unflatten_tree(ys_leaves, ys_tree),
    )
