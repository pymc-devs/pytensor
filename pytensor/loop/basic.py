import functools

import numpy as np

from pytensor.basic import as_symbolic
from pytensor.graph import FunctionGraph, Variable
from pytensor.graph.basic import Constant
from pytensor.graph.replace import graph_replace
from pytensor.graph.traversal import truncated_graph_inputs
from pytensor.loop.op import Scan
from pytensor.scan.utils import until
from pytensor.tensor import as_tensor, constant, minimum


def scan(
    fn,
    init_states=None,
    sequences=None,
    non_sequences=None,
    n_steps=None,
    go_backwards=False,
) -> Variable | list[Variable]:
    if sequences is None and n_steps is None:
        raise ValueError("Must provide n_steps when scanning without sequences")

    if init_states is None:
        init_states = []
    else:
        if not isinstance(init_states, (tuple, list)):
            init_states = [init_states]
        init_states = [as_symbolic(i) if i is not None else None for i in init_states]

    if sequences is None:
        sequences = []
    else:
        if not isinstance(sequences, (tuple, list)):
            sequences = [sequences]
        sequences = [as_tensor(s) for s in sequences]

    if sequences:
        leading_dims = [seq.shape[0] for seq in sequences]
        shortest_dim = functools.reduce(minimum, leading_dims)
        if n_steps is None:
            n_steps = shortest_dim
        else:
            n_steps = minimum(n_steps, shortest_dim)

    if non_sequences is None:
        non_sequences = []
    else:
        if not isinstance(non_sequences, (tuple, list)):
            non_sequences = [non_sequences]
        non_sequences = [as_symbolic(n) for n in non_sequences]

    # Create dummy inputs for the init state. The user function should not
    # draw any relationship with the outer initial states, since these are only
    # valid in the first iteration
    inner_states = [i.type() if i is not None else None for i in init_states]

    # Create subsequence inputs for the inner function
    idx = constant(0, dtype="int64", name="idx")
    symbolic_idx = idx.type(name="idx")
    subsequences = [s[symbolic_idx] for s in sequences]

    # Call user function to retrieve inner outputs. We use the same order as the old Scan,
    # although inner_states + subsequences + non_sequences seems more intuitive,
    # since subsequences are just a fancy non_sequence
    # We don't pass the non-carried outputs [init is None] to the inner function
    fn_inputs = (
        subsequences + [i for i in inner_states if i is not None] + non_sequences
    )
    fn_outputs = fn(*fn_inputs)
    if not isinstance(fn_outputs, (tuple, list)):
        fn_outputs = [fn_outputs]
    next_states = [out for out in fn_outputs if not isinstance(out, until)]

    if len(next_states) > len(init_states):
        if not init_states:
            init_states = [None] * len(next_states)
            inner_states = init_states
        else:
            raise ValueError(
                "Please provide None as `init` for any output that is not carried over (i.e. it behaves like a map) "
            )

    # Partition the inner outputs into carries (with an init, fed back) and
    # map-like outputs (init is None): the latter become nit-sot outputs of the
    # Scan, with no input and no initial value.
    carry_prev_states = []  # outer init value per carry
    carry_inner_states = []  # inner input dummy per carry
    carry_next_states = []  # inner next value per carry
    output_values = []  # inner nit-sot output value per map output
    output_entries = []  # ("carry", pos) or ("output", pos), in fn-output order
    for init_state, inner_state, next_state in zip(
        init_states, inner_states, next_states
    ):
        if init_state is None:
            output_entries.append(("output", len(output_values)))
            output_values.append(next_state)
        else:
            # State shapes are loop-invariant, but shape inference may type the
            # updated state more weakly than the init; coerce it back when needed.
            if not inner_state.type.is_super(next_state.type):
                next_state = inner_state.type.filter_variable(next_state)
            output_entries.append(("carry", len(carry_next_states)))
            carry_prev_states.append(init_state)
            carry_inner_states.append(inner_state)
            carry_next_states.append(next_state)

    # Flip until to while condition
    while_condition = [~out.condition for out in fn_outputs if isinstance(out, until)]
    if not while_condition:
        while_condition = [as_tensor(np.array(True))]
    if len(while_condition) > 1:
        raise ValueError("Only one until condition can be returned")

    fgraph_inputs = [symbolic_idx, *carry_inner_states, *sequences, *non_sequences]
    fgraph_outputs = [
        *while_condition,
        symbolic_idx + 1,
        *carry_next_states,
        *output_values,
    ]

    all_fgraph_inputs = truncated_graph_inputs(
        fgraph_outputs, ancestors_to_include=fgraph_inputs
    )
    extra_fgraph_inputs = [
        inp
        for inp in all_fgraph_inputs
        if (not isinstance(inp, Constant) and inp not in fgraph_inputs)
    ]

    # The outer constants (which may include shared variables) cannot be used
    # directly as inputs of the inner function graph. Replace them by dummies.
    outer_constants = [*sequences, *non_sequences, *extra_fgraph_inputs]
    inner_constants = [c.type() for c in outer_constants]
    if outer_constants:
        fgraph_outputs = graph_replace(
            fgraph_outputs,
            dict(zip(outer_constants, inner_constants)),
            strict=False,
        )
    update_fg = FunctionGraph(
        inputs=[symbolic_idx, *carry_inner_states, *inner_constants],
        outputs=fgraph_outputs,
    )

    n_carries = 1 + len(carry_inner_states)
    scan_op = Scan(
        update_fg=update_fg, sequences=range(len(sequences)), n_carries=n_carries
    )
    scan_outs = scan_op(
        n_steps,
        idx,
        *carry_prev_states,
        *sequences,
        *non_sequences,
        *extra_fgraph_inputs,
    )
    assert isinstance(scan_outs, list)
    # Output layout: carry finals, carry traces, output traces; the idx is dropped.
    carry_traces = scan_outs[n_carries + 1 : 2 * n_carries]
    output_traces = scan_outs[2 * n_carries :]
    traces = [
        carry_traces[pos] if kind == "carry" else output_traces[pos]
        for kind, pos in output_entries
    ]
    if len(traces) == 1:
        return traces[0]
    return traces


def map(
    fn,
    sequences,
    non_sequences=None,
    go_backwards=False,
):
    traces = scan(
        fn=fn,
        sequences=sequences,
        non_sequences=non_sequences,
        go_backwards=go_backwards,
    )
    return traces


def reduce(
    fn,
    init_states,
    sequences,
    non_sequences=None,
    go_backwards=False,
):
    traces = scan(
        fn=fn,
        init_states=init_states,
        sequences=sequences,
        non_sequences=non_sequences,
        go_backwards=go_backwards,
    )
    if not isinstance(traces, list):
        return traces[-1]
    return [trace[-1] for trace in traces]


def filter(
    fn,
    sequences,
    non_sequences=None,
    go_backwards=False,
):
    if not isinstance(sequences, (tuple, list)):
        sequences = [sequences]

    masks = scan(
        fn=fn,
        sequences=sequences,
        non_sequences=non_sequences,
        go_backwards=go_backwards,
    )

    if not isinstance(masks, list):
        masks = [masks] * len(sequences)
    elif len(masks) != len(sequences):
        raise ValueError(
            "filter fn must return one variable or len(sequences), but it returned {len(masks)}"
        )
    if not all(mask.dtype == "bool" for mask in masks):
        raise TypeError("The output of filter fn should be a boolean variable")

    filtered_sequences = [seq[mask] for seq, mask in zip(sequences, masks)]

    if len(filtered_sequences) == 1:
        return filtered_sequences[0]
    return filtered_sequences
