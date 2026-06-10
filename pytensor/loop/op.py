import numpy as np

from pytensor import In, Out, config
from pytensor.compile import function, optdb
from pytensor.compile.ops import view_op
from pytensor.gradient import DisconnectedType, NullType, grad_undefined, pullback
from pytensor.graph import Apply, FunctionGraph, Op, Type, node_rewriter
from pytensor.graph.replace import graph_replace
from pytensor.graph.rewriting.basic import in2out
from pytensor.graph.traversal import ancestors
from pytensor.scalar import constant
from pytensor.tensor import (
    add,
    and_,
    as_tensor,
    concatenate,
    empty,
    get_scalar_constant_value,
    invert,
    set_subtensor,
    shape_padleft,
    zeros_like,
)
from pytensor.tensor.exceptions import NotScalarConstantError
from pytensor.tensor.shape import Shape_i
from pytensor.tensor.subtensor import Subtensor, get_idx_list
from pytensor.tensor.type import (
    DenseTensorType,
    TensorType,
    continuous_dtypes,
    integer_dtypes,
)
from pytensor.tensor.type_other import NoneTypeT
from pytensor.typed_list import GetItem, TypedListType, append, make_empty_list


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


def compile_update_fn(fgraph):
    """Compile the inner update function graph of a Loop or Scan Op."""
    wrapped_inputs = [In(x, borrow=True) for x in fgraph.inputs]
    wrapped_outputs = [Out(x, borrow=False) for x in fgraph.outputs]

    # TODO: Figure this out
    # The numba linker cannot pass non-tensor types (TypedList, RandomGenerator)
    # as arguments of the compiled update function
    if all(
        isinstance(var.type, DenseTensorType)
        for var in (*fgraph.inputs, *fgraph.outputs[1:])
    ):
        mode = "FAST_RUN"
    else:
        mode = "CVM"

    return function(
        wrapped_inputs,
        wrapped_outputs,
        mode=mode,
        accept_inplace=False,
        on_unused_input="ignore",
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
    ):
        validate_loop_update_types(update_fg)
        self.state_types = [out.type for out in update_fg.outputs[1:]]
        self.const_types = [
            inp.type for inp in update_fg.inputs[len(self.state_types) :]
        ]
        self.update_fg = update_fg
        self._fn = None

    @property
    def fn(self):
        """Lazily compile the inner update function graph."""
        if self._fn is None:
            self._fn = compile_update_fn(self.update_fg)
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

    def pullback(self, inputs, outputs, cotangents):
        raise NotImplementedError(
            "Gradients are implemented by the Scan Op. "
            "The Loop Op should only be introduced during compilation, after gradients are computed."
        )


class Scan(Op):
    """Represent a scan.

    This Op can be thought of as a loop that collects intermediate steps

    Roughly equivalent to
    ```
    def scan(fn, initial_states, constants, max_iters):
        traces = [[] * len(initial_states)]
        states = initial_states
        for i in range(max_iters):
            resume, states = fn(*states, *constants)
            for trace, state in zip(traces, states):
                trace.append(state)
            if not resume:
                break
        return states, traces
    ```
    Tensor state traces are collected as tensors with a new leading dimension,
    other types (e.g. RandomGenerator) are collected in TypedLists.

    During compilation this Op is lowered to the legacy Scan Op
    (or to a Loop Op, when non-tensor traces are needed).
    """

    def __init__(
        self,
        update_fg: FunctionGraph,  # (*state,  *consts) -> (bool, *state)
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
                self.trace_types.append(TypedListType(state_type))

        self.constant_types = [inp.type for inp in update_fg.inputs[self.n_states :]]
        self.n_constants = len(self.constant_types)

        self.update_fg = update_fg.clone(check_integrity=False)

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

        max_iters = as_tensor(max_iters)
        if max_iters.type.dtype != "int64":
            if max_iters.type.dtype not in integer_dtypes:
                raise TypeError(
                    f"max_iters must be an integer scalar, got {max_iters.type}"
                )
            max_iters = max_iters.astype("int64")
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

    @property
    def fn(self):
        """Lazily compile the inner update function graph.

        Only used when the Op was not lowered away during compilation
        (e.g. in unoptimized modes).
        """
        if getattr(self, "_fn", None) is None:
            self._fn = compile_update_fn(self.update_fg)
        return self._fn

    def perform(self, node, inputs, output_storage):
        update_fn = self.fn

        max_iters = inputs[0]
        states = list(inputs[1 : 1 + self.n_states])
        constants = inputs[1 + self.n_states :]
        traces = [[] for _ in range(self.n_states)]
        for _ in range(max_iters):
            resume, *states = update_fn(*states, *constants)
            for trace, state in zip(traces, states):
                trace.append(state)
            if not resume:
                break

        for i, state in enumerate(states):
            output_storage[i][0] = state
        for i, (trace, trace_type) in enumerate(zip(traces, self.trace_types)):
            if isinstance(trace_type, DenseTensorType):
                if trace:
                    trace = np.stack(trace)
                else:
                    trace = np.empty(
                        (0, *np.shape(inputs[1 + i])), dtype=trace_type.dtype
                    )
            output_storage[self.n_states + i][0] = trace

    @staticmethod
    def _input_grad_kind(var) -> str:
        """Classify the gradient an input is entitled to.

        "grad": continuous tensors, a proper gradient is computed.
        "zeros": discrete tensors, which are structurally connected to the
        outputs, and by convention receive zero gradients.
        "undefined": non-tensor continuous inputs, for which gradients are not
        implemented.
        "disconnected": other types (RandomGenerator, TypedList, ...) through
        which gradients cannot flow.
        """
        if isinstance(var.type, DenseTensorType):
            return "grad" if var.type.dtype in continuous_dtypes else "zeros"
        if getattr(var.type, "dtype", None) in continuous_dtypes:
            return "undefined"
        return "disconnected"

    def connection_pattern(self, node):
        n_outputs = len(node.outputs)
        connected_inputs = [False]  # max_iters
        connected_inputs += [
            self._input_grad_kind(inp) != "disconnected" for inp in node.inputs[1:]
        ]
        return [[connected] * n_outputs for connected in connected_inputs]

    def pullback(self, inputs, outputs, cotangents):
        """Compute the gradient of a Scan as another Scan that iterates backwards over the traces.

        At the backward step corresponding to forward step t, the cotangent of the t-th
        states (the running cotangent plus the contribution of the respective trace entries)
        is pulled back through the update function evaluated at the inputs of forward step t.
        This yields the cotangents of the (t-1)-th states, and the contributions of the
        constants, which are accumulated across iterations.

        After the backward Scan, the running state cotangents correspond to the gradients
        with respect to the initial states, and the accumulated contributions to the
        gradients with respect to the constants.
        """
        # Avoid circular import
        from pytensor.loop.basic import scan

        max_iters = inputs[0]
        init_states = inputs[1 : 1 + self.n_states]
        constants = inputs[1 + self.n_states :]
        traces = outputs[self.n_states :]
        final_state_grads = cotangents[: self.n_states]
        trace_grads = cotangents[self.n_states :]

        def is_connected(g) -> bool:
            return not isinstance(g.type, DisconnectedType)

        disconnected = DisconnectedType()

        def default_grad(input_pos, var):
            """Gradient for inputs the backward Scan does not compute, consistent with connection_pattern."""
            kind = self._input_grad_kind(var)
            if kind == "grad":
                return var.zeros_like()
            if kind == "zeros":
                return var.zeros_like(dtype=config.floatX)
            if kind == "undefined":
                return grad_undefined(
                    self,
                    input_pos,
                    var,
                    f"Gradient of Scan not implemented for inputs of type {var.type}",
                )
            return disconnected()

        if any(isinstance(g.type, NullType) for g in cotangents if is_connected(g)):
            null = NullType("Gradient of Scan output is undefined")
            return [disconnected()] + [
                null()
                if self._input_grad_kind(var) != "disconnected"
                else disconnected()
                for var in (*init_states, *constants)
            ]

        diff_state_idxs = [
            i for i, s in enumerate(init_states) if self._input_grad_kind(s) == "grad"
        ]
        diff_const_idxs = [
            j for j, c in enumerate(constants) if self._input_grad_kind(c) == "grad"
        ]
        dense_state_idxs = [
            i for i, s in enumerate(init_states) if isinstance(s.type, DenseTensorType)
        ]

        if not diff_state_idxs or not any(is_connected(g) for g in cotangents):
            return [disconnected()] + [
                default_grad(pos, var)
                for pos, var in enumerate((*init_states, *constants), start=1)
            ]

        # If there is a while condition, the trace length tells how many iterations were actually run
        if self.has_while_condition:
            n_iters = traces[dense_state_idxs[0]].shape[0]
        else:
            n_iters = max_iters

        # Create the pullback (vector-Jacobian product) graph of the update function,
        # with dummy variables for the state cotangents
        update_fg = self.update_fg.clone(check_integrity=False)
        inner_prev_states = update_fg.inputs[: self.n_states]
        inner_constants = update_fg.inputs[self.n_states :]
        inner_next_states = update_fg.outputs[1:]

        state_cotangents = [inner_next_states[i].type() for i in diff_state_idxs]
        # The next states are wrapped in ViewOps so that no entry of `f` is an
        # ancestor of (or the same variable as) another. Otherwise the cotangents
        # seeded at the nested entry would override the gradient flowing into it.
        vjps = pullback(
            f=[view_op(inner_next_states[i]) for i in diff_state_idxs],
            wrt=(
                [inner_prev_states[i] for i in diff_state_idxs]
                + [inner_constants[j] for j in diff_const_idxs]
            ),
            cotangents=state_cotangents,
            disconnected_inputs="ignore",
            return_disconnected="zero",
        )
        non_dense_inner_states = [
            s for i, s in enumerate(inner_prev_states) if i not in dense_state_idxs
        ]

        # The inputs of forward step t are the states of step t-1: the initial states
        # followed by all trace entries but the last, reversed for the backward iteration
        prev_state_seqs = [
            concatenate([init_states[i][None], traces[i][:-1]])[::-1]
            for i in dense_state_idxs
        ]
        traced_grad_idxs = [i for i in diff_state_idxs if is_connected(trace_grads[i])]
        trace_grad_seqs = [trace_grads[i][::-1] for i in traced_grad_idxs]

        init_lambdas = [
            self.state_types[i].filter_variable(
                final_state_grads[i]
                if is_connected(final_state_grads[i])
                else zeros_like(init_states[i])
            )
            for i in diff_state_idxs
        ]
        init_accs = [zeros_like(constants[j]) for j in diff_const_idxs]

        n_prev = len(dense_state_idxs)
        n_traced = len(traced_grad_idxs)
        n_lambdas = len(diff_state_idxs)

        def backward_step(*args):
            prev_state_elems = args[:n_prev]
            trace_grad_elems = args[n_prev : n_prev + n_traced]
            lambdas = args[n_prev + n_traced : n_prev + n_traced + n_lambdas]
            accs = args[n_prev + n_traced + n_lambdas : -len(constants) or None]
            outer_constants = args[len(args) - len(constants) :]

            total_lambdas = list(lambdas)
            for k, i in enumerate(diff_state_idxs):
                if i in traced_grad_idxs:
                    total_lambdas[k] = (
                        total_lambdas[k] + trace_grad_elems[traced_grad_idxs.index(i)]
                    )

            replace = dict(
                zip(
                    (inner_prev_states[i] for i in dense_state_idxs),
                    prev_state_elems,
                )
            )
            replace.update(zip(inner_constants, outer_constants))
            replace.update(zip(state_cotangents, total_lambdas))
            results = graph_replace(vjps, replace, strict=False)

            if set(non_dense_inner_states) & set(ancestors(results)):
                raise NotImplementedError(
                    "Gradient of Scan requires the values of non-tensor states, which are not traced"
                )

            new_lambdas = [
                self.state_types[i].filter_variable(r)
                for i, r in zip(diff_state_idxs, results[:n_lambdas])
            ]
            new_accs = [
                self.constant_types[j].filter_variable(acc + r)
                for j, acc, r in zip(diff_const_idxs, accs, results[n_lambdas:])
            ]
            return [*new_lambdas, *new_accs]

        backward_outs = scan(
            backward_step,
            init_states=[*init_lambdas, *init_accs],
            sequences=[*prev_state_seqs, *trace_grad_seqs],
            non_sequences=list(constants),
            n_steps=n_iters,
        )
        if not isinstance(backward_outs, list):
            backward_outs = [backward_outs]
        final_lambdas = [trace[-1] for trace in backward_outs[:n_lambdas]]
        final_accs = [trace[-1] for trace in backward_outs[n_lambdas:]]

        input_grads = [disconnected()]  # max_iters
        for i, init_state in enumerate(init_states):
            if i in diff_state_idxs:
                input_grads.append(final_lambdas[diff_state_idxs.index(i)])
            else:
                input_grads.append(default_grad(1 + i, init_state))
        for j, constant_inp in enumerate(constants):
            if j in diff_const_idxs:
                input_grads.append(final_accs[diff_const_idxs.index(j)])
            else:
                input_grads.append(default_grad(1 + self.n_states + j, constant_inp))
        return input_grads


@node_rewriter([Scan])
def scan_to_loop(fgraph, node):
    """Rewrite a Scan Op into a Loop Op

    It roughly creates the following computational graph
    ```
    def scan(fn, idx, initial_states, constants, max_iters):
        idx = 0
        states = initial_states
        traces = [
            empty(max_iters, *initial_state.shape) for initial_state in initial_states
        ]
        while True:
            resume, states, fn(*states, *traces, *constants)
            for trace, state in zip(traces, states):
                trace[idx] = state
            idx += 1
            if not resume or idx >= max_iters:
                break
        traces = [trace[:idx] for trace in traces]
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

    # Inputs to the new Loop
    max_iters = node.inputs[0]
    init_states = node.inputs[1 : 1 + op.n_states]
    init_traces = [
        empty(
            (max_iters, *tuple(init_states[trace_idx].shape)),
            dtype=init_states[trace_idx].dtype,
        )
        if isinstance(init_states[trace_idx].type, DenseTensorType)
        else make_empty_list(init_states[trace_idx].type)
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
        if isinstance(prev_trace.type, DenseTensorType)
        else append(prev_trace, inner_next_states[trace_idx])
        for trace_idx, prev_trace in zip(used_traces_idxs, inner_traces)
    ]
    for t in inner_next_traces:
        t.name = "next_trace"
    inner_max_iters = max_iters.type()
    inner_while_cond = and_(inner_while_cond, inner_next_idx < inner_max_iters)
    inner_while_cond.name = "while(?)"

    if not has_idx:
        init_states = [init_idx, *init_states]
        inner_states = [inner_idx, *inner_states]
        inner_next_states = [inner_next_idx, *inner_next_states]

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
        if op.has_while_condition and isinstance(new_trace.type, DenseTensorType):
            new_trace = new_trace[:final_idx]
        replacements[old_traces[trace_idx]] = new_trace
    return replacements


# Fallback lowering for Scan nodes that `scan_to_legacy_scan` cannot handle
# (those whose non-tensor traces are used). Must come after that rewrite.
optdb.register(
    "scan_to_loop",
    in2out(scan_to_loop),
    "fast_compile",
    "fast_run",
    position=1.0,
)


@node_rewriter([Scan])
def scan_to_legacy_scan(fgraph, node):
    """Lower a Scan Op into the legacy Scan Op.

    Tensor states are mapped to sit-sots, other states to untraced sit-sots,
    and constants to non-sequences. This gives access to the legacy rewrites
    (buffer shortening, pushouts, merging, inplace) and backend dispatches.

    Traces of non-tensor states (TypedList) cannot be represented in the legacy
    Scan. Nodes that need them are left to be lowered to a Loop Op instead.
    """
    from pytensor.scan.op import Scan as LegacyScan
    from pytensor.scan.op import ScanInfo
    from pytensor.scan.utils import expand_empty

    op: Scan = node.op  # type: ignore

    max_iters = node.inputs[0]
    init_states = node.inputs[1 : 1 + op.n_states]
    constants = node.inputs[1 + op.n_states :]
    old_finals = node.outputs[: op.n_states]
    old_traces = node.outputs[op.n_states :]

    dense_idxs = [
        i
        for i, state_type in enumerate(op.state_types)
        if isinstance(state_type, DenseTensorType)
    ]
    other_idxs = [
        i
        for i, state_type in enumerate(op.state_types)
        if not isinstance(state_type, DenseTensorType)
    ]

    if any(fgraph.clients[old_traces[i]] for i in other_idxs):
        return None

    update_fg = op.update_fg.clone(check_integrity=False)
    inner_states = update_fg.inputs[: op.n_states]
    inner_constants = update_fg.inputs[op.n_states :]
    inner_cond, *inner_next_states = update_fg.outputs

    inner_inputs = [
        *(inner_states[i] for i in dense_idxs),
        *(inner_states[i] for i in other_idxs),
        *inner_constants,
    ]
    inner_outputs = [
        *(inner_next_states[i] for i in dense_idxs),
        *(inner_next_states[i] for i in other_idxs),
    ]
    if op.has_while_condition:
        # The legacy Scan stops when the condition output is True
        inner_outputs.append(invert(inner_cond))

    info = ScanInfo(
        n_seqs=0,
        mit_mot_in_slices=(),
        mit_mot_out_slices=(),
        mit_sot_in_slices=(),
        sit_sot_in_slices=((-1,),) * len(dense_idxs),
        n_nit_sot=0,
        n_untraced_sit_sot=len(other_idxs),
        n_non_seqs=op.n_constants,
        as_while=op.has_while_condition,
    )
    legacy_op = LegacyScan(inner_inputs, inner_outputs, info, strict=True)

    outer_inputs = [
        max_iters,
        *(expand_empty(shape_padleft(init_states[i]), max_iters) for i in dense_idxs),
        *(init_states[i] for i in other_idxs),
        *constants,
    ]
    new_outs = legacy_op(*outer_inputs, return_list=True)
    buffers = new_outs[: len(dense_idxs)]
    untraced_outs = new_outs[len(dense_idxs) :]

    # The legacy outputs lose the static length information of the trace types,
    # so the replacements must be filtered back into the original types
    replacements = {}
    for i, buffer in zip(dense_idxs, buffers):
        replacements[old_finals[i]] = old_finals[i].type.filter_variable(buffer[-1])
        replacements[old_traces[i]] = old_traces[i].type.filter_variable(buffer[1:])
    for i, untraced_out in zip(other_idxs, untraced_outs):
        replacements[old_finals[i]] = untraced_out
    return replacements


# Must run before the legacy scan rewrites (scan_eqopt1 at 0.05),
# so that pushouts, buffer shortening and merging can be applied
optdb.register(
    "scan_to_legacy_scan",
    in2out(scan_to_legacy_scan),
    "fast_compile",
    "fast_run",
    "scan",
    position=0.02,
)


@node_rewriter([Scan])
def scan_view_last_state(fgraph, node):
    """Replace trace[-1] by the last state output of a Scan node"""
    replacements = {}
    for final_state, trace in zip(
        node.outputs[: node.op.n_states], node.outputs[node.op.n_states :]
    ):
        clients = fgraph.clients[trace]
        for client, _ in clients:
            if client == "output":
                continue
            if isinstance(client.op, (Subtensor, GetItem)):
                if isinstance(client.op, Subtensor):
                    idxs = get_idx_list(client.inputs, client.op.idx_list)
                    if len(idxs) == 1:
                        idx = idxs[0]
                else:
                    idx = client.inputs[1]
                try:
                    last_index = get_scalar_constant_value(idx) == -1
                except NotScalarConstantError:
                    continue
                if last_index:
                    replacements[client.default_output()] = final_state
    return replacements


# Must run before the lowering rewrites
optdb.register(
    "scan_view_last_state",
    in2out(scan_view_last_state),
    "fast_compile",
    "fast_run",
    position=0.01,
)
