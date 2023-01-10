import functools
from typing import Optional

import numpy as np

from pytensor import In, Out, get_scalar_constant_value
from pytensor.compile import optdb, pfunc
from pytensor.graph import Apply, FunctionGraph, Op, Type, node_rewriter
from pytensor.graph.rewriting.basic import in2out
from pytensor.scalar import constant
from pytensor.tensor import NoneConst, and_, empty, minimum, set_subtensor
from pytensor.tensor.exceptions import NotScalarConstantError
from pytensor.tensor.shape import Shape_i
from pytensor.tensor.type import DenseTensorType, TensorType
from pytensor.tensor.type_other import NoneTypeT


def validate_loop_update_types(update):
    assert update.outputs[0].type.dtype == "bool"
    for input_state, output_state in zip(update.inputs, update.outputs[1:]):
        assert input_state.type == output_state.type


class Loop(Op):
    """Represent a do-while loop.

    We represent the loop body as an inner FunctionGraph, which
    computes the next state and whether the loop should continue.

    Roughly equivalent to
    ```
    def loop(fn, initial_state, constants):
        state = initial_state
        while True:
            resume, state = fn(i, state, *constants)
            if not resume:
                break
        return state
    ```
    Multiple initial states and constants can be provided
    """

    def __init__(
        self,
        update_fg: FunctionGraph,  # (*state,  *consts) -> (bool, *state)
        reverse_fg: Optional[FunctionGraph] = None,
    ):
        validate_loop_update_types(update_fg)
        self.state_types = [out.type for out in update_fg.outputs[1:]]
        self.const_types = [
            inp.type for inp in update_fg.inputs[len(self.state_types) :]
        ]
        self.update_fg = update_fg
        self.reverse_fg = reverse_fg
        self._fn = None

    @property
    def fn(self):
        """Lazily compile the inner update function graph."""
        if self._fn is not None:
            return self._fn

        fgraph = self.update_fg
        wrapped_inputs = [In(x, borrow=True) for x in fgraph.inputs]
        wrapped_outputs = [Out(x, borrow=False) for x in fgraph.outputs]

        self._fn = pfunc(
            wrapped_inputs,
            wrapped_outputs,
            mode="FAST_RUN",  # TODO: Figure this out
            accept_inplace=False,
            on_unused_input="ignore",
            fgraph=fgraph,
        )
        return self._fn

    def make_node(self, *inputs):
        assert len(inputs) == len(self.state_types) + len(self.const_types)

        states = inputs[: len(self.state_types)]
        states = [
            inp_type.filter_variable(inp)
            for inp_type, inp in zip(self.state_types, states)
        ]

        consts = inputs[len(self.state_types) :]
        consts = [
            inp_type.filter_variable(inp)
            for inp_type, inp in zip(self.const_types, consts)
        ]

        return Apply(
            self,
            [*states, *consts],
            [state_type() for state_type in self.state_types],
        )

    def infer_shape(self, fgraph, node, input_shapes):
        return input_shapes[: len(self.state_types)]

    def perform(self, node, inputs, output_storage):
        update_fn = self.fn

        states = inputs[: len(self.state_types)]
        consts = inputs[len(self.state_types) :]
        resume = True
        while resume:
            resume, *states = update_fn(*states, *consts)

        for i, state in enumerate(states):
            output_storage[i][0] = state

    def L_Op(self, *args):
        if not self.reverse_fg:
            raise NotImplementedError()
        # Use L_Op of self.reverse_fg
        ...

    def R_Op(self, *args):
        # Use R_op of self.update_fg
        ...


class Scan(Op):
    """Represent a scan.

    This Op can be thought of as a loop that collects intermediate steps

    Roughly equivalent to
    ```
    def scan(fn, initial_states, sequences, constants, max_iters):
        traces = [[]*len(initial_states)]
        states = initial_states
        for (idx, *subsequences) in zip(*(range(max_iters), *sequences)):
            resume, states = fn(*states, *subsequences, *constants)
            for trace, state in zip(traces, states):
                trace.append(state)
            if not resume:
                break
        return states, traces
    ```
    Not all types of states can be collected, for instance RandomGenerator. For these
    `None` is returned in place of the respective traces

    The number of iterations is bounded by max_iters or the shortest of sequences.

    This Op must always be converted to a Loop during compilation.
    """

    def __init__(
        self,
        update_fg: FunctionGraph,  # (*state,  *consts) -> (bool, *state)
        n_sequences: int,
        reverse_fg: Optional[FunctionGraph] = None,
    ):
        validate_loop_update_types(update_fg)

        self.state_types = [out.type for out in update_fg.outputs[1:]]
        self.n_states = len(self.state_types)
        self.trace_types: list[Type] = []
        for state_type in self.state_types:
            # TODO: Accommodate SparseTensors and Scalars
            if isinstance(state_type, DenseTensorType):
                self.trace_types.append(
                    DenseTensorType(
                        shape=(None, *state_type.shape), dtype=state_type.dtype
                    )
                )
            else:
                # We can't concatenate all types of states, such as RandomTypes
                self.trace_types.append(NoneConst.type)

        self.n_sequences = n_sequences
        self.sequence_types = []
        for inner_seq in update_fg.inputs[
            self.n_states : self.n_states + self.n_sequences
        ]:
            # TODO: Accomodate other sequence types
            assert isinstance(inner_seq.type, DenseTensorType)
            self.sequence_types.append(
                DenseTensorType(
                    shape=(None, *inner_seq.type.shape), dtype=inner_seq.type.dtype
                )
            )

        self.non_sequence_types = [
            inp.type for inp in update_fg.inputs[self.n_states + self.n_sequences :]
        ]
        self.n_non_sequences = len(self.non_sequence_types)

        self.update_fg = update_fg.clone(check_integrity=False)
        self.reverse_fg = (
            reverse_fg.clone(check_integrity=False) if reverse_fg is not None else None
        )

    def make_node(self, max_iters, *inputs):
        assert len(inputs) == self.n_states + self.n_sequences + self.n_non_sequences

        if self.n_sequences == 0 and max_iters is None:
            raise ValueError("Must provide max_iters in Scans without sequences")

        if max_iters is not None:
            max_iters = TensorType(dtype="int64", shape=()).filter_variable(max_iters)

        states = inputs[: self.n_states]
        states = [
            inp_type.filter_variable(inp)
            for inp_type, inp in zip(self.state_types, states)
        ]

        sequences = inputs[self.n_states : self.n_states + self.n_sequences]
        sequences = [
            inp_type.filter_variable(inp)
            for inp_type, inp in zip(self.sequence_types, sequences)
        ]
        if sequences:
            leading_dims = [seq.shape[0] for seq in sequences]
            shortest_dim = functools.reduce(minimum, leading_dims)
            if max_iters is None:
                max_iters = shortest_dim
            else:
                max_iters = minimum(max_iters, shortest_dim)

        non_sequences = inputs[self.n_states + self.n_sequences :]
        non_sequences = [
            inp_type.filter_variable(inp)
            for inp_type, inp in zip(self.non_sequence_types, non_sequences)
        ]

        # If there is no loop condition, `max_iters` exclusively defines the number of iterations
        # If this value is constant, we can get static type shapes for the leading dimensions of traces
        try:
            if get_scalar_constant_value(self.update_fg.outputs[0]):
                n_iters = int(get_scalar_constant_value(max_iters))
        except NotScalarConstantError:
            trace_types = self.trace_types
        else:
            trace_types = []
            for trace_type in self.trace_types:
                if isinstance(trace_type, DenseTensorType):
                    trace_types.append(
                        DenseTensorType(
                            dtype=trace_type.dtype,
                            shape=(n_iters, *trace_type.shape[1:]),
                        )
                    )
                else:
                    trace_types.append(trace_type)

        return Apply(
            self,
            [max_iters, *states, *sequences, *non_sequences],
            [output_type() for output_type in self.state_types + trace_types],
        )

    def infer_shape(self, fgraph, node, input_shapes):
        # If there is a loop condition, `max_iters` provides only the upper bound for the number of iterations
        try:
            has_condition = not get_scalar_constant_value(self.update_fg.outputs[0])
        except NotScalarConstantError:
            has_condition = True
        if has_condition:
            # find the first non-None trace
            trace_out = next(
                trace
                for trace in node.outputs[self.n_states :]
                if not isinstance(trace.type, NoneTypeT)
            )
            n_iters = Shape_i(0)(trace_out)
        else:
            n_iters = node.inputs[0]  # max_iters

        state_shapes = input_shapes[1 : self.n_states + 1]
        trace_shapes = [
            (n_iters, *state_shape) if state_shape is not None else None
            for state_shape in state_shapes
        ]
        return state_shapes + trace_shapes

    def do_constant_folding(self, fgraph, node):
        return False

    def perform(self, node, inputs, output_storage):
        raise RuntimeError("Scan Op should not be present in compiled graph")

    def L_op(self, *args):
        # Use trace outputs
        ...

    def R_op(self, *args):
        # Use R_op of self.update
        ...


@node_rewriter([Scan])
def scan_to_loop(fgraph, node):
    """Rewrite a Scan Op into a Loop Op

    It roughly creates the following computational graph
    ```

    def scan(fn, initial_states, sequences, constants, max_iters):

        def update_fn(idx, states, traces, sequences, constants, max_iters)
            subsequences = [seq[idx] for seq in subsequences]
            resume, states = inner_fn(states, subsequences, constants)
            for trace, state in zip(traces, states):
                trace[idx] = state
            return (resume and (idx < max_iters)), idx + 1, states, traces

        idx = 0
        traces = [empty(max_iters, *initial_state.shape) for initial_state in initial_states]
        while True:
            resume, idx, states, traces = update_fn(idx, *states, *traces, *sequences, *constants, max_iters)
            if not resume:
                break
        traces = [trace[: idx] for trace in traces]
        return states, traces
    ```

    Traces that are not used anywhere in the graph are omitted from the final Loop

    """
    op: Scan = node.op  # type: ignore

    old_states = node.outputs[: op.n_states]
    old_traces = node.outputs[op.n_states :]

    # Only include the intermediate states that are used elsewhere
    used_traces_idxs = [
        i
        for i, trace in enumerate(node.outputs[op.n_states :])
        if fgraph.clients[trace]
    ]

    # Check that outputs that cannot be converted into sequences (such as RandomTypes) are not being referenced
    for trace_idx in used_traces_idxs:
        assert not isinstance(old_states[trace_idx].type, NoneTypeT)

    # Inputs to the new Loop
    max_iters = node.inputs[0]
    init_idx = constant(np.array(0, dtype="int64"), name="idx")
    init_states = node.inputs[1 : 1 + op.n_states]
    init_traces = [
        empty(
            (max_iters, *tuple(init_states[trace_idx].shape)),
            dtype=init_states[trace_idx].dtype,
        )
        for trace_idx in used_traces_idxs
    ]
    sequences = node.inputs[1 + op.n_states : 1 + op.n_states + op.n_sequences]
    non_sequences = node.inputs[1 + op.n_states + op.n_sequences :]

    new_fg = op.update_fg.clone(check_integrity=False)

    # Inner index
    inner_prev_idx = init_idx.type()
    inner_prev_idx.name = "prev_idx"

    # Inner traces
    inner_prev_states = new_fg.inputs[: op.n_states]
    inner_prev_traces = [init_trace.type() for init_trace in init_traces]
    for s, t in zip(inner_prev_states, inner_prev_traces):
        t.name = "prev_trace"
        if s.name:
            t.name = "_".join((t.name, s.name))

    inner_non_sequences = new_fg.inputs[op.n_states + op.n_sequences :]

    # Replace inner sub-sequences by sequence[idx]
    inner_seqs_news = []
    if op.n_sequences:
        inner_subseqs_old = new_fg.inputs[op.n_states : op.n_states + op.n_sequences]
        inner_subseqs_new = []
        for sequence in sequences:
            inner_seq_new = sequence.type()
            inner_seq_new.name = sequence.name or "sequence"
            inner_seqs_news.append(inner_seq_new)
            inner_subseq_new = inner_seq_new[inner_prev_idx]
            inner_subseq_new.name = inner_seq_new.name + "[prev_idx]"
            inner_subseqs_new.append(inner_subseq_new)

        # Replace inner_sequence input by sequence[idx]
        replacements = tuple(zip(inner_subseqs_old, inner_subseqs_new))
        new_fg.replace_all(replacements, import_missing=True)

    # Inner continue condition and index
    inner_continue_cond, *inner_next_states = new_fg.outputs
    inner_next_idx = inner_prev_idx + 1
    inner_next_idx.name = "next_idx"
    inner_next_traces = [
        set_subtensor(prev_trace[inner_prev_idx], inner_next_states[trace_idx])
        for trace_idx, prev_trace in zip(used_traces_idxs, inner_prev_traces)
    ]
    for t in inner_next_traces:
        t.name = "next_trace"
    inner_max_iters = max_iters.type()
    inner_continue_cond = and_(inner_continue_cond, inner_next_idx < inner_max_iters)
    inner_continue_cond.name = "continue(?)"

    new_fg = FunctionGraph(
        inputs=[
            inner_prev_idx,
            *inner_prev_states,
            *inner_prev_traces,
            *inner_seqs_news,
            *inner_non_sequences,
            inner_max_iters,
        ],
        outputs=[
            inner_continue_cond,
            inner_next_idx,
            *inner_next_states,
            *inner_next_traces,
        ],
    )

    # TODO: Implement Reverse?
    loop_op = Loop(update_fg=new_fg)

    final_idx, *new_outs = loop_op(
        init_idx, *init_states, *init_traces, *sequences, *non_sequences, max_iters
    )
    new_states = new_outs[: op.n_states]
    new_traces = new_outs[op.n_states :]

    try:
        has_condition = not get_scalar_constant_value(op.update_fg.outputs[0])
    except NotScalarConstantError:
        has_condition = True
    replacements = dict(zip(old_states, new_states))
    for trace_idx, new_trace in zip(used_traces_idxs, new_traces):
        # If there is no condition, the whole trace will be used
        if has_condition:
            new_trace = new_trace[:final_idx]
        replacements[old_traces[trace_idx]] = new_trace
    return replacements


# TODO: Create new Loop dataset
# Needs to be executed after `local_shape_to_shape_i`, otherwise shape graphs
# cannot be properly replaced
optdb.register(
    "scan_to_loop",
    in2out(scan_to_loop),
    "fast_compile",
    "fast_run",
    position=1.0,
)
