import functools
from typing import List, Tuple

import numpy as np

from pytensor import Variable, as_symbolic, clone_replace
from pytensor.graph import FunctionGraph
from pytensor.graph.basic import Constant, truncated_graph_inputs
from pytensor.loop.op import Scan
from pytensor.scan.utils import until
from pytensor.tensor import as_tensor, constant, empty_like, minimum


def scan(
    fn,
    init_states=None,
    sequences=None,
    non_sequences=None,
    n_steps=None,
    go_backwards=False,
) -> Tuple[List[Variable], List[Variable]]:
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

    # Replace None init by dummy empty tensors
    prev_states = []
    prev_inner_states = []
    for i, (init_state, inner_state, next_state) in enumerate(
        zip(init_states, inner_states, next_states)
    ):
        if init_state is None:
            # next_state may reference idx. We replace that by the initial value,
            # so that the shape of the dummy init state does not depend on it.
            [next_state] = clone_replace(
                output=[next_state], replace={symbolic_idx: idx}
            )
            init_state = empty_like(next_state)
            init_state.name = "empty_init_state"
            inner_state = init_state.type(name="dummy_state")
        prev_states.append(init_state)
        prev_inner_states.append(inner_state)

    # Flip until to while condition
    while_condition = [~out.condition for out in fn_outputs if isinstance(out, until)]
    if not while_condition:
        while_condition = [as_tensor(np.array(True))]
    if len(while_condition) > 1:
        raise ValueError("Only one until condition can be returned")

    fgraph_inputs = [symbolic_idx] + prev_inner_states + sequences + non_sequences
    fgraph_outputs = while_condition + [symbolic_idx + 1] + next_states

    all_fgraph_inputs = truncated_graph_inputs(
        fgraph_outputs, ancestors_to_include=fgraph_inputs
    )
    extra_fgraph_inputs = [
        inp
        for inp in all_fgraph_inputs
        if (not isinstance(inp, Constant) and inp not in fgraph_inputs)
    ]
    fgraph_inputs = fgraph_inputs + extra_fgraph_inputs
    update_fg = FunctionGraph(inputs=fgraph_inputs, outputs=fgraph_outputs)

    scan_op = Scan(update_fg=update_fg)
    scan_outs = scan_op(
        n_steps, idx, *prev_states, *sequences, *non_sequences, *extra_fgraph_inputs
    )
    assert isinstance(scan_outs, list)
    last_states = scan_outs[: scan_op.n_states]
    traces = scan_outs[scan_op.n_states :]
    # Don't return the inner index state
    return last_states[1:], traces[1:]


def map(
    fn,
    sequences,
    non_sequences=None,
    go_backwards=False,
):
    _, traces = scan(
        fn=fn,
        sequences=sequences,
        non_sequences=non_sequences,
        go_backwards=go_backwards,
    )
    if len(traces) == 1:
        return traces[0]
    return traces


def reduce(
    fn,
    init_states,
    sequences,
    non_sequences=None,
    go_backwards=False,
):
    final_states, _ = scan(
        fn=fn,
        init_states=init_states,
        sequences=sequences,
        non_sequences=non_sequences,
        go_backwards=go_backwards,
    )
    if len(final_states) == 1:
        return final_states[0]
    return final_states


def filter(
    fn,
    sequences,
    non_sequences=None,
    go_backwards=False,
):
    if not isinstance(sequences, (tuple, list)):
        sequences = [sequences]

    _, masks = scan(
        fn=fn,
        sequences=sequences,
        non_sequences=non_sequences,
        go_backwards=go_backwards,
    )

    if not all(mask.dtype == "bool" for mask in masks):
        raise TypeError("The output of filter fn should be a boolean variable")
    if len(masks) == 1:
        masks = [masks[0]] * len(sequences)
    elif len(masks) != len(sequences):
        raise ValueError(
            "filter fn must return one variable or len(sequences), but it returned {len(masks)}"
        )

    filtered_sequences = [seq[mask] for seq, mask in zip(sequences, masks)]

    if len(filtered_sequences) == 1:
        return filtered_sequences[0]
    return filtered_sequences
