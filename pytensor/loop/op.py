from typing import Optional

import numpy as np

from pytensor import In, Out
from pytensor.compile import optdb, pfunc
from pytensor.graph import Apply, FunctionGraph, Op, Type, node_rewriter
from pytensor.graph.rewriting.basic import in2out
from pytensor.scalar import constant
from pytensor.tensor import (
    NoneConst,
    add,
    and_,
    empty,
    get_scalar_constant_value,
    set_subtensor,
)
from pytensor.tensor.exceptions import NotScalarConstantError
from pytensor.tensor.shape import Shape_i
from pytensor.tensor.type import DenseTensorType, TensorType
from pytensor.tensor.type_other import NoneTypeT


def validate_loop_update_types(update):
    assert update.outputs[0].type.dtype == "bool"
    for i, (input_state, output_state) in enumerate(
        zip(update.inputs, update.outputs[1:])
    ):
        if input_state.type != output_state.type:
            raise TypeError(
                f"The {i}-th input and output states of the inner loop function have different types: "
                f"{input_state.type} vs {output_state.type}."
            )


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
    def scan(fn, initial_states, constants, max_iters):
        traces = [[]*len(initial_states)]
        states = initial_states
        for i in range(max_iters):
            resume, states = fn(*states, *constants)
            for trace, state in zip(traces, states):
                trace.append(state)
            if not resume:
                break
        return states, traces
    ```
    Not all types of states can be collected, for instance RandomGenerator. For these
    `None` is returned in place of the respective traces

    This Op must always be converted to a Loop during compilation.
    """

    def __init__(
        self,
        update_fg: FunctionGraph,  # (*state,  *consts) -> (bool, *state)
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

        self.constant_types = [inp.type for inp in update_fg.inputs[self.n_states :]]
        self.n_constants = len(self.constant_types)

        self.update_fg = update_fg.clone(check_integrity=False)
        self.reverse_fg = (
            reverse_fg.clone(check_integrity=False) if reverse_fg is not None else None
        )

        # It's more conservative to assume the Op has a while condition
        self.has_while_condition = True
        try:
            self.has_while_condition = not get_scalar_constant_value(
                update_fg.outputs[0]
            )
        except NotScalarConstantError:
            pass

    def make_node(self, max_iters, *inputs):
        assert len(inputs) == self.n_states + self.n_constants

        max_iters = TensorType(dtype="int64", shape=()).filter_variable(max_iters)

        states = inputs[: self.n_states]
        states = [
            inp_type.filter_variable(inp)
            for inp_type, inp in zip(self.state_types, states)
        ]

        constants = inputs[self.n_states :]
        constants = [
            inp_type.filter_variable(inp)
            for inp_type, inp in zip(self.constant_types, constants)
        ]

        # If there is no while condition, `max_iters` exclusively defines the number of iterations
        # If this value is constant, we can get static type shapes for the leading dimensions of traces
        trace_types = self.trace_types
        if not self.has_while_condition:
            try:
                n_iters = int(get_scalar_constant_value(max_iters))
            except NotScalarConstantError:
                pass
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
            [max_iters, *states, *constants],
            [output_type() for output_type in self.state_types + trace_types],
        )

    def infer_shape(self, fgraph, node, input_shapes):
        # If there is a while condition, `max_iters` provides only the upper bound for the number of iterations
        if self.has_while_condition:
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

    def scan(fn, idx, initial_states, constants, max_iters):
        idx = 0
        states = initial_states
        traces = [empty(max_iters, *initial_state.shape) for initial_state in initial_states]
        while True:
            resume, states, fn(*states, *traces, *constants)
            for trace, state in zip(traces, states):
                trace[idx] = state
            idx += 1
            if not resume or idx >= max_iters:
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
    init_states = node.inputs[1 : 1 + op.n_states]
    init_traces = [
        empty(
            (max_iters, *tuple(init_states[trace_idx].shape)),
            dtype=init_states[trace_idx].dtype,
        )
        for trace_idx in used_traces_idxs
    ]
    constants = node.inputs[1 + op.n_states :]

    update_fg = op.update_fg.clone(check_integrity=False)

    # Check if inner_fg computes an index already, otherwise create a new one
    has_idx = False
    if len(node.inputs) > 1:
        try:
            outer_inp = node.inputs[1]
            outer_is_zero = get_scalar_constant_value(outer_inp) == 0
        except NotScalarConstantError:
            pass
        else:
            if (
                outer_is_zero
                and len(update_fg.inputs) > 0
                and len(update_fg.outputs) > 1
            ):
                inner_out = update_fg.outputs[1]
                if (
                    inner_out.owner is not None
                    and inner_out.owner.op == add
                    and len(inner_out.owner.inputs) == 2
                ):
                    left, right = inner_out.owner.inputs
                    if left is update_fg.inputs[0]:
                        try:
                            has_idx = (
                                get_scalar_constant_value(
                                    right, only_process_constants=True
                                )
                                == 1
                            )
                        except NotScalarConstantError:
                            pass

    if has_idx:
        init_idx = outer_inp
        inner_idx = inner_out.owner.inputs[0]
        inner_next_idx = inner_out
    if not has_idx:
        init_idx = constant(np.array(0, dtype="int64"), name="idx")
        inner_idx = init_idx.type()
        inner_idx.name = "idx"
        inner_next_idx = inner_idx + 1
        inner_next_idx.name = "next_idx"

    # Inner traces
    inner_states = update_fg.inputs[: op.n_states]
    inner_traces = [init_trace.type() for init_trace in init_traces]
    for s, t in zip(inner_states, inner_traces):
        t.name = "trace"
        if s.name:
            t.name = "_".join((t.name, s.name))

    inner_constants = update_fg.inputs[op.n_states :]

    # Inner while condition
    inner_while_cond, *inner_next_states = update_fg.outputs
    inner_next_traces = [
        set_subtensor(prev_trace[inner_idx], inner_next_states[trace_idx])
        for trace_idx, prev_trace in zip(used_traces_idxs, inner_traces)
    ]
    for t in inner_next_traces:
        t.name = "next_trace"
    inner_max_iters = max_iters.type()
    inner_while_cond = and_(inner_while_cond, inner_next_idx < inner_max_iters)
    inner_while_cond.name = "while(?)"

    if not has_idx:
        init_states = [init_idx] + init_states
        inner_states = [inner_idx] + inner_states
        inner_next_states = [inner_next_idx] + inner_next_states

    new_update_fg = FunctionGraph(
        inputs=[
            *inner_states,
            *inner_traces,
            *inner_constants,
            inner_max_iters,
        ],
        outputs=[
            inner_while_cond,
            *inner_next_states,
            *inner_next_traces,
        ],
    )

    # TODO: Implement Reverse?
    loop_op = Loop(update_fg=new_update_fg)

    new_outs = loop_op(*init_states, *init_traces, *constants, max_iters)
    if has_idx:
        # idx was part of the original scan, and therefore has a corresponding trace
        final_idx = new_outs[0]
    else:
        final_idx, *new_outs = new_outs
    new_states = new_outs[: op.n_states]
    new_traces = new_outs[op.n_states :]

    replacements = dict(zip(old_states, new_states))
    for trace_idx, new_trace in zip(used_traces_idxs, new_traces):
        # If there is no while condition, the whole trace will be used
        if op.has_while_condition:
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
    "not_jax",
    position=1.0,
)
