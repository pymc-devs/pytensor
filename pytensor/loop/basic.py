from typing import List, Tuple

import numpy as np

from pytensor import Variable, as_symbolic
from pytensor.graph import FunctionGraph
from pytensor.loop.op import Scan
from pytensor.scan.utils import until
from pytensor.tensor import as_tensor, empty_like


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
        init_states = [as_symbolic(i) for i in init_states]

    if sequences is None:
        sequences = []
    else:
        if not isinstance(sequences, (tuple, list)):
            sequences = [sequences]
        sequences = [as_tensor(s) for s in sequences]

    if non_sequences is None:
        non_sequences = []
    else:
        if not isinstance(non_sequences, (tuple, list)):
            non_sequences = [non_sequences]
        non_sequences = [as_symbolic(n) for n in non_sequences]

    # Note: Old scan order is sequences + init + non_sequences
    inner_sequences = [s[0] for s in sequences]
    inner_inputs = [i.type() for i in init_states + inner_sequences + non_sequences]
    inner_outputs = fn(*inner_inputs)
    if not isinstance(inner_outputs, (tuple, list)):
        inner_outputs = [inner_outputs]
    next_states = [out for out in inner_outputs if not isinstance(out, until)]

    if len(next_states) > len(init_states):
        if not init_states:
            init_states = [None] * len(next_states)
        else:
            raise ValueError(
                "Please provide None as `init` for any output that is not carried over (i.e. it behaves like a map) "
            )

    # Replace None init by dummy empty tensors
    prev_states = []
    for i, (init_state, next_state) in enumerate(zip(init_states, next_states)):
        if init_state is None:
            init_state = empty_like(next_state)
            init_state.name = "empty_init_state"
            inner_inputs.insert(i, init_state.type())
        prev_states.append(init_state)

    until_condition = [out.condition for out in inner_outputs if isinstance(out, until)]
    if not until_condition:
        until_condition = [as_tensor(np.array(True))]
    if len(until_condition) > 1:
        raise ValueError("Only one until condition can be returned")

    update_fg = FunctionGraph(
        inputs=inner_inputs, outputs=until_condition + next_states
    )
    scan_op = Scan(update_fg=update_fg, n_sequences=len(sequences))
    scan_outs = scan_op(n_steps, *prev_states, *sequences, *non_sequences)
    assert isinstance(scan_outs, list)
    last_states = scan_outs[: scan_op.n_states]
    traces = scan_outs[scan_op.n_states :]

    return last_states, traces


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
