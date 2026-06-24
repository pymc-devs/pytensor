from collections.abc import Sequence

import numpy as np

from pytensor import In, Out, config
from pytensor.compile import function, optdb
from pytensor.compile.ops import view_op
from pytensor.gradient import (
    DisconnectedType,
    NullType,
    _is_zero,
    grad_undefined,
    pullback,
)
from pytensor.graph import Apply, FunctionGraph, Op, Type, Variable, node_rewriter
from pytensor.graph.replace import graph_replace
from pytensor.graph.rewriting.basic import in2out
from pytensor.graph.traversal import ancestors
from pytensor.scan.op import Scan as LegacyScan
from pytensor.scan.op import ScanInfo
from pytensor.scan.utils import expand_empty
from pytensor.tensor import (
    add,
    as_tensor,
    concatenate,
    get_scalar_constant_value,
    inc_subtensor,
    invert,
    shape_padleft,
    zeros_like,
)
from pytensor.tensor.basic import ScalarFromTensor
from pytensor.tensor.exceptions import NotScalarConstantError
from pytensor.tensor.shape import Shape_i
from pytensor.tensor.subtensor import (
    AdvancedIncSubtensor,
    AdvancedIncSubtensor1,
    IncSubtensor,
    Subtensor,
    get_idx_list,
)
from pytensor.tensor.type import (
    DenseTensorType,
    TensorType,
    continuous_dtypes,
    integer_dtypes,
)
from pytensor.tensor.type_other import NoneTypeT
from pytensor.typed_list import GetItem, TypedListType


def validate_loop_update_types(update, n_carries=None):
    assert update.outputs[0].type.dtype == "bool"
    # Only the carries are fed back into inputs; trailing outputs (nit-sots) have
    # no corresponding input.
    next_carries = (
        update.outputs[1:] if n_carries is None else update.outputs[1 : 1 + n_carries]
    )
    for i, (input_state, output_state) in enumerate(zip(update.inputs, next_carries)):
        # Outputs may be more specific than the inputs (e.g. have static shapes
        # the inputs lack), since they are fed back into them across iterations
        if not input_state.type.is_super(output_state.type):
            raise TypeError(
                f"The {i}-th input and output states of the inner loop function have incompatible types: "
                f"{input_state.type} vs {output_state.type}."
            )


def _scalar_constant(var):
    try:
        return int(get_scalar_constant_value(var, only_process_constants=True))
    except NotScalarConstantError:
        return None


def match_sequence_read(client, position, state_inputs):
    """Return (idx_state, tap) when the client is a `c[idx + tap]` read.

    Negative taps don't match: they would wrap around the sequence,
    which legacy Scan sequences cannot represent.
    """
    if client == "output" or not isinstance(client.op, Subtensor) or position != 0:
        return None
    indices = get_idx_list(client.inputs, client.op.idx_list)
    if len(indices) != 1:
        return None
    [index] = indices
    if (
        isinstance(index, Variable)
        and index.owner is not None
        and isinstance(index.owner.op, ScalarFromTensor)
    ):
        index = index.owner.inputs[0]
    if index in state_inputs:
        return index, 0
    if (
        isinstance(index, Variable)
        and index.owner is not None
        and index.owner.op == add
        and len(index.owner.inputs) == 2
    ):
        left, right = index.owner.inputs
        if (
            left in state_inputs
            and (tap := _scalar_constant(right)) is not None
            and tap >= 0
        ):
            return left, tap
        if (
            right in state_inputs
            and (tap := _scalar_constant(left)) is not None
            and tap >= 0
        ):
            return right, tap
    return None


def verify_sequence_reads(update_fg, n_states, sequences):
    """Verify which of the declared constants are only read along an index state.

    A constant `c` behaves as a sequence when its only uses are `c[idx + k]`
    reads, for constant offsets `k >= 0`, where `idx` is a state that advances
    by one in every iteration. Candidates that don't conform are simply not
    recorded: they behave as plain constants, which is always valid. The
    recorded reads only enable optimizations in the gradient and the lowering.

    Returns the verified read offsets per constant, and the position of the
    index state.
    """
    if not sequences:
        return {}, None

    state_inputs = set(update_fg.inputs[:n_states])

    def advances_by_one(some_state):
        next_state = update_fg.outputs[1 + update_fg.inputs.index(some_state)]
        return (
            next_state.owner is not None
            and next_state.owner.op == add
            and len(next_state.owner.inputs) == 2
            and any(inp is some_state for inp in next_state.owner.inputs)
            and any(
                inp is not some_state and _scalar_constant(inp) == 1
                for inp in next_state.owner.inputs
            )
        )

    idx_state = None
    sequence_reads = {}
    for j in sequences:
        const = update_fg.inputs[n_states + j]
        taps = set()
        for client, position in update_fg.clients[const]:
            read = match_sequence_read(client, position, state_inputs)
            if read is None:
                taps = None
                break
            read_idx, tap = read
            if read_idx is not idx_state and (
                idx_state is not None or not advances_by_one(read_idx)
            ):
                taps = None
                break
            idx_state = read_idx
            taps.add(tap)
        if taps:
            sequence_reads[j] = tuple(sorted(taps))

    idx_state_pos = update_fg.inputs.index(idx_state) if sequence_reads else None
    return sequence_reads, idx_state_pos


def peel_last_step_cotangent(trace_cotangent):
    """Extract g from trace cotangents of the form `inc_subtensor(zeros, g, -1)`.

    This is the cotangent produced when the cost is only connected to the last
    step of a trace (e.g. `cost = fn(trace[-1])`). Instead of streaming a one-hot
    sequence of cotangents through the backward Scan, g can seed its initial
    state cotangent directly.
    """
    node = trace_cotangent.owner
    if node is None or not isinstance(node.op, IncSubtensor):
        return None
    if len(node.op.idx_list) != 1:
        return None
    zeros, g, *index_inputs = node.inputs
    if _is_zero(zeros) != "yes":
        return None
    [index] = get_idx_list([zeros, *index_inputs], node.op.idx_list)
    try:
        if get_scalar_constant_value(index) != -1:
            return None
    except NotScalarConstantError:
        return None
    return g


def accumulate_grad(acc, contribution):
    """Accumulate a per-step gradient contribution into the running accumulator.

    The pullback computes constant contributions as IncSubtensor chains rooted
    in a zeros array. Grafting the accumulator as the root of the chain avoids
    adding two arrays (one of which has the size of the whole gradient) in
    every iteration of the backward Scan.
    """
    chain = []
    root = contribution
    while (
        root.owner is not None
        and isinstance(
            root.owner.op,
            IncSubtensor | AdvancedIncSubtensor | AdvancedIncSubtensor1,
        )
        and not root.owner.op.set_instead_of_inc
    ):
        chain.append(root.owner)
        root = root.owner.inputs[0]

    if root is contribution:
        if _is_zero(contribution) == "yes":
            return acc
        return acc + contribution

    # The graft is only valid if the zeros root is not also used in the
    # increment values or indices of the chain
    other_chain_inputs = [inp for node in chain for inp in node.inputs[1:]]
    if _is_zero(root) == "yes" and root not in ancestors(other_chain_inputs):
        return graph_replace([contribution], {root: acc})[0]
    return acc + contribution


def compile_update_fn(fgraph):
    """Compile the inner update function graph of a Scan Op."""
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

    During compilation this Op is lowered to the legacy Scan Op. Non-tensor
    states (e.g. RandomGenerators) can be carried but not traced.
    """

    def __init__(
        self,
        update_fg: FunctionGraph,  # (*carries, *consts) -> (bool, *next_carries, *outputs)
        sequences: Sequence[int] = (),
        n_carries: int | None = None,
    ):
        """
        Parameters
        ----------
        update_fg
            Inner function graph mapping (*carries, *constants) to
            (continue_flag, *next_carries, *outputs). Carries are fed back into
            the next iteration; trailing outputs are collected per step (nit-sot)
            without being fed back.
        sequences
            Indices of constants that are only read elementwise along an index
            state (`const[idx + k]`). This is validated, and recorded in
            `sequence_reads`/`idx_state` for the gradient and lowering to exploit.
        n_carries
            Number of carried states; the remaining inner outputs are nit-sot
            outputs. Defaults to every inner output being a carry.
        """
        n_inner_outputs = len(update_fg.outputs) - 1
        if n_carries is None:
            n_carries = n_inner_outputs
        validate_loop_update_types(update_fg, n_carries)

        # `state_types`/`n_states` refer to the carries throughout.
        self.n_states = n_carries
        self.state_types = [out.type for out in update_fg.outputs[1 : 1 + n_carries]]
        self.output_types = [out.type for out in update_fg.outputs[1 + n_carries :]]
        self.n_outputs = len(self.output_types)
        self.sequence_reads, self.idx_state = verify_sequence_reads(
            update_fg, self.n_states, tuple(sequences)
        )
        # Both carries and outputs are collected into traces.
        self.trace_types: list[Type] = []
        for traced_type in (*self.state_types, *self.output_types):
            # TODO: Accommodate SparseTensors and Scalars
            if isinstance(traced_type, DenseTensorType):
                self.trace_types.append(
                    DenseTensorType(
                        shape=(None, *traced_type.shape), dtype=traced_type.dtype
                    )
                )
            else:
                self.trace_types.append(TypedListType(traced_type))

        self.constant_types = [inp.type for inp in update_fg.inputs[self.n_states :]]
        self.n_constants = len(self.constant_types)

        # Stored immutable: the inner graph is frozen between construction and
        # lowering, so the sequence-read analysis stays exact and the Op is
        # hashable (structurally identical Scans can be merged).
        self.update_fg = update_fg.freeze()

        # It's more conservative to assume the Op has a while condition
        self.has_while_condition = True
        try:
            self.has_while_condition = not get_scalar_constant_value(
                update_fg.outputs[0]
            )
        except NotScalarConstantError:
            pass

    def __eq__(self, other):
        if self is other:
            return True
        if type(self) is not type(other):
            return False
        return (
            self.update_fg == other.update_fg
            and self.n_states == other.n_states
            and self.sequence_reads == other.sequence_reads
        )

    def __hash__(self):
        return hash(
            (
                type(self),
                self.update_fg,
                self.n_states,
                tuple(sorted(self.sequence_reads.items())),
            )
        )

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

        carry_shapes = input_shapes[1 : self.n_states + 1]
        carry_trace_shapes = [
            (n_iters, *carry_shape) if carry_shape is not None else None
            for carry_shape in carry_shapes
        ]
        # Outputs (nit-sots) have no input to take a per-step shape from; rely on
        # the static type shape when fully known, otherwise leave it unknown.
        output_trace_shapes = []
        for output_trace in node.outputs[2 * self.n_states :]:
            trailing = output_trace.type.shape[1:]
            if any(dim is None for dim in trailing):
                output_trace_shapes.append(None)
            else:
                output_trace_shapes.append((n_iters, *trailing))
        return carry_shapes + carry_trace_shapes + output_trace_shapes

    def do_constant_folding(self, fgraph, node):
        return False

    @property
    def fn(self):
        """Lazily compile the inner update function graph.

        Only used when the Op was not lowered away during compilation
        (e.g. in unoptimized modes).
        """
        if getattr(self, "_fn", None) is None:
            self._fn = compile_update_fn(self.update_fg.unfreeze())
        return self._fn

    def perform(self, node, inputs, output_storage):
        update_fn = self.fn

        max_iters = inputs[0]
        carries = list(inputs[1 : 1 + self.n_states])
        constants = inputs[1 + self.n_states :]
        # The inner function returns the next carries followed by the outputs;
        # both are traced, but only the carries are fed back.
        traces = [[] for _ in range(self.n_states + self.n_outputs)]
        for _ in range(max_iters):
            resume, *next_values = update_fn(*carries, *constants)
            for trace, value in zip(traces, next_values):
                trace.append(value)
            carries = next_values[: self.n_states]
            if not resume:
                break

        for i, carry in enumerate(carries):
            output_storage[i][0] = carry
        for i, (trace, trace_type) in enumerate(zip(traces, self.trace_types)):
            if isinstance(trace_type, DenseTensorType):
                if trace:
                    trace = np.stack(trace)
                else:
                    # No iterations ran: carries keep their input per-step shape;
                    # outputs fall back to their static (possibly partial) shape.
                    if i < self.n_states:
                        per_step_shape = np.shape(inputs[1 + i])
                    else:
                        per_step_shape = tuple(
                            dim if dim is not None else 0
                            for dim in trace_type.shape[1:]
                        )
                    trace = np.empty((0, *per_step_shape), dtype=trace_type.dtype)
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
        # Differentiable outputs (nit-sots) with a connected trace cotangent. They
        # feed cotangent sequences into the backward Scan but, unlike carries, are
        # not threaded as backward states.
        diff_output_idxs = [
            k
            for k, out_type in enumerate(self.output_types)
            if isinstance(out_type, DenseTensorType)
            and out_type.dtype in continuous_dtypes
            and is_connected(trace_grads[self.n_states + k])
        ]

        # Sequence constants (verified const[idx + k] reads) build their gradient
        # from per-tap backward traces placed once after the Scan, instead of a
        # per-step inc_subtensor accumulator. Only valid when the index starts at
        # 0 (so forward step t reads position t + k); otherwise the constant falls
        # back to the accumulator path.
        idx_pos = self.idx_state
        seq_eligible = (
            bool(self.sequence_reads)
            and idx_pos is not None
            and isinstance(init_states[idx_pos].type, DenseTensorType)
            and _scalar_constant(init_states[idx_pos]) == 0
        )
        diff_seq_idxs = [
            j for j in diff_const_idxs if seq_eligible and j in self.sequence_reads
        ]
        diff_acc_idxs = [j for j in diff_const_idxs if j not in diff_seq_idxs]

        # A gradient is only computable if there is something differentiable to
        # pull a connected cotangent through (a carry or an output) and something
        # to seed the backward Scan with (a carry or a constant).
        if (
            not (diff_state_idxs or diff_output_idxs)
            or not (diff_state_idxs or diff_const_idxs)
            or not any(is_connected(g) for g in cotangents)
        ):
            return [disconnected()] + [
                default_grad(pos, var)
                for pos, var in enumerate((*init_states, *constants), start=1)
            ]

        # When the cost is only connected to the last step of a trace, fold the
        # cotangent into the initial state cotangent of the backward Scan,
        # instead of streaming a one-hot sequence of cotangents through it
        final_state_grads = list(final_state_grads)
        trace_grads = list(trace_grads)
        for i in diff_state_idxs:
            if not is_connected(trace_grads[i]):
                continue
            last_step_cotangent = peel_last_step_cotangent(trace_grads[i])
            if last_step_cotangent is not None:
                if is_connected(final_state_grads[i]):
                    final_state_grads[i] = final_state_grads[i] + last_step_cotangent
                else:
                    final_state_grads[i] = last_step_cotangent
                trace_grads[i] = disconnected()

        # If there is a while condition, the trace length tells how many iterations were actually run
        if self.has_while_condition:
            dense_trace = next(
                traces[i]
                for i in (
                    *dense_state_idxs,
                    *(self.n_states + k for k in diff_output_idxs),
                )
            )
            n_iters = dense_trace.shape[0]
        else:
            n_iters = max_iters

        # Create the pullback (vector-Jacobian product) graph of the update function,
        # with dummy variables for the state cotangents
        update_fg = self.update_fg.unfreeze()
        inner_prev_states = update_fg.inputs[: self.n_states]
        inner_constants = update_fg.inputs[self.n_states :]
        inner_next_states = update_fg.outputs[1:]

        # Locate the inner read variables const[idx + k] of the sequence constants;
        # the backward Scan traces their cotangents (one trace per read).
        seq_state_inputs = set(inner_prev_states)
        seq_reads_info = []  # (const_idx, tap, inner_read_var)
        for j in diff_seq_idxs:
            for client, position in update_fg.clients[inner_constants[j]]:
                read = match_sequence_read(client, position, seq_state_inputs)
                assert read is not None  # Verified at op construction
                seq_reads_info.append((j, read[1], client.outputs[0]))

        state_cotangents = [inner_next_states[i].type() for i in diff_state_idxs]
        output_cotangents = [
            inner_next_states[self.n_states + k].type() for k in diff_output_idxs
        ]
        # The next states/outputs are wrapped in ViewOps so that no entry of `f` is
        # an ancestor of (or the same variable as) another. Otherwise the cotangents
        # seeded at the nested entry would override the gradient flowing into it.
        vjps = pullback(
            f=(
                [view_op(inner_next_states[i]) for i in diff_state_idxs]
                + [
                    view_op(inner_next_states[self.n_states + k])
                    for k in diff_output_idxs
                ]
            ),
            wrt=(
                [inner_prev_states[i] for i in diff_state_idxs]
                + [inner_constants[j] for j in diff_acc_idxs]
                + [read_var for (_j, _tap, read_var) in seq_reads_info]
            ),
            cotangents=state_cotangents + output_cotangents,
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
        output_grad_seqs = [
            trace_grads[self.n_states + k][::-1] for k in diff_output_idxs
        ]

        init_lambdas = [
            self.state_types[i].filter_variable(
                final_state_grads[i]
                if is_connected(final_state_grads[i])
                else zeros_like(init_states[i])
            )
            for i in diff_state_idxs
        ]
        init_accs = [zeros_like(constants[j]) for j in diff_acc_idxs]

        n_prev = len(dense_state_idxs)
        n_traced = len(traced_grad_idxs)
        n_outs = len(diff_output_idxs)
        n_lambdas = len(diff_state_idxs)
        n_acc = len(diff_acc_idxs)
        n_reads = len(seq_reads_info)

        def backward_step(*args):
            prev_state_elems = args[:n_prev]
            trace_grad_elems = args[n_prev : n_prev + n_traced]
            output_grad_elems = args[n_prev + n_traced : n_prev + n_traced + n_outs]
            base = n_prev + n_traced + n_outs
            lambdas = args[base : base + n_lambdas]
            accs = args[base + n_lambdas : base + n_lambdas + n_acc]
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
            replace.update(zip(output_cotangents, output_grad_elems))
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
                self.constant_types[j].filter_variable(accumulate_grad(acc, r))
                for j, acc, r in zip(
                    diff_acc_idxs, accs, results[n_lambdas : n_lambdas + n_acc]
                )
            ]
            # Per-step cotangents of the sequence reads, traced (not accumulated)
            read_cotangents = [
                read_var.type.filter_variable(r)
                for (_j, _tap, read_var), r in zip(
                    seq_reads_info, results[n_lambdas + n_acc :]
                )
            ]
            return [*new_lambdas, *new_accs, *read_cotangents]

        backward_outs = scan(
            backward_step,
            # `None` inits make the read cotangents nit-sot outputs (traced, not fed back)
            init_states=[*init_lambdas, *init_accs, *([None] * n_reads)],
            sequences=[*prev_state_seqs, *trace_grad_seqs, *output_grad_seqs],
            non_sequences=list(constants),
            n_steps=n_iters,
        )
        if not isinstance(backward_outs, list):
            backward_outs = [backward_outs]
        final_lambdas = [trace[-1] for trace in backward_outs[:n_lambdas]]
        final_accs = [
            trace[-1] for trace in backward_outs[n_lambdas : n_lambdas + n_acc]
        ]
        read_traces = backward_outs[n_lambdas + n_acc :]

        # A sequence gradient is its per-tap cotangent traces (reversed back to
        # forward order) placed at the slice each tap read, summed over taps.
        seq_grads = {j: zeros_like(constants[j]) for j in diff_seq_idxs}
        for (j, tap, _read_var), trace in zip(seq_reads_info, read_traces):
            seq_grads[j] = inc_subtensor(seq_grads[j][tap : tap + n_iters], trace[::-1])

        input_grads = [disconnected()]  # max_iters
        for i, init_state in enumerate(init_states):
            if i in diff_state_idxs:
                input_grads.append(final_lambdas[diff_state_idxs.index(i)])
            else:
                input_grads.append(default_grad(1 + i, init_state))
        for j, constant_inp in enumerate(constants):
            if j in diff_seq_idxs:
                input_grads.append(self.constant_types[j].filter_variable(seq_grads[j]))
            elif j in diff_acc_idxs:
                input_grads.append(final_accs[diff_acc_idxs.index(j)])
            else:
                input_grads.append(default_grad(1 + self.n_states + j, constant_inp))
        return input_grads


@node_rewriter([Scan])
def scan_to_legacy_scan(fgraph, node):
    """Lower a Scan Op into the legacy Scan Op.

    Verified sequence constants are mapped to legacy sequences (one per read
    offset), and the index state is elided when feeding those reads was its
    only job. Tensor states whose traces are used are mapped to sit-sots, all
    other states to untraced sit-sots, and the remaining constants to
    non-sequences. This gives access to the legacy rewrites (pushouts, merging,
    inplace) and backend dispatches. Untraced states need no buffers at all,
    and their inner updates are allowed to operate inplace.

    Traces of non-tensor states (TypedList) cannot be represented in the legacy
    Scan, so tracing such a state is not supported.
    """
    op: Scan = node.op  # type: ignore

    max_iters = node.inputs[0]
    init_states = node.inputs[1 : 1 + op.n_states]
    constants = node.inputs[1 + op.n_states :]
    old_finals = node.outputs[: op.n_states]
    old_traces = node.outputs[op.n_states :]

    traced_idxs = []
    untraced_idxs = []
    for i, state_type in enumerate(op.state_types):
        if fgraph.clients[old_traces[i]]:
            if not isinstance(state_type, DenseTensorType):
                raise NotImplementedError(
                    "Non-tensor states (e.g. RandomGenerators) can be carried "
                    "but not traced: their trace cannot be represented in the legacy Scan."
                )
            traced_idxs.append(i)
        else:
            untraced_idxs.append(i)

    # Outputs (nit-sots) are emitted only when their trace is used.
    output_traces = old_traces[op.n_states :]
    used_output_idxs = [
        k for k in range(op.n_outputs) if fgraph.clients[output_traces[k]]
    ]

    update_fg = op.update_fg.unfreeze()
    inner_states = update_fg.inputs[: op.n_states]
    inner_constants = update_fg.inputs[op.n_states :]
    inner_cond, *inner_next_states = update_fg.outputs

    # The sequence reads align with the legacy sequence slices only when the
    # index starts at zero
    sequence_reads = op.sequence_reads
    idx_pos = op.idx_state
    if sequence_reads and (
        _scalar_constant(node.inputs[1 + idx_pos]) != 0
        or not isinstance(init_states[idx_pos].type, DenseTensorType)
    ):
        sequence_reads = {}

    # Replace the sequence reads by legacy sequence inputs (one per read offset),
    # with the outer sequences sliced so that all offsets align at each step
    inner_seqs = []
    outer_seqs = []
    read_replacements = {}
    state_input_set = set(inner_states)
    for j, taps in sequence_reads.items():
        inner_const = inner_constants[j]
        outer_const = constants[j]
        max_tap = max(taps)
        elem_by_tap = {}
        for tap in taps:
            elem = inner_const.type.clone(shape=inner_const.type.shape[1:])()
            elem_by_tap[tap] = elem
            inner_seqs.append(elem)
            if tap == max_tap == 0:
                outer_seqs.append(outer_const)
            elif tap == max_tap:
                outer_seqs.append(outer_const[tap:])
            else:
                outer_seqs.append(outer_const[tap : tap - max_tap])
        for client, position in update_fg.clients[inner_const]:
            read = match_sequence_read(client, position, state_input_set)
            assert read is not None  # Verified at op construction
            read_replacements[client.outputs[0]] = elem_by_tap[read[1]]
    if read_replacements:
        inner_cond, *inner_next_states = graph_replace(
            [inner_cond, *inner_next_states], read_replacements, strict=False
        )

    # Drop the index state when its only remaining job was feeding the reads
    if (
        sequence_reads
        and not fgraph.clients[old_finals[idx_pos]]
        and not fgraph.clients[old_traces[idx_pos]]
        and inner_states[idx_pos]
        not in set(
            ancestors(
                [
                    inner_cond,
                    *(s for i, s in enumerate(inner_next_states) if i != idx_pos),
                ]
            )
        )
    ):
        untraced_idxs = [i for i in untraced_idxs if i != idx_pos]

    remaining_const_idxs = [j for j in range(op.n_constants) if j not in sequence_reads]

    inner_inputs = [
        *inner_seqs,
        *(inner_states[i] for i in traced_idxs),
        *(inner_states[i] for i in untraced_idxs),
        *(inner_constants[j] for j in remaining_const_idxs),
    ]
    inner_outputs = [
        *(inner_next_states[i] for i in traced_idxs),
        *(inner_next_states[op.n_states + k] for k in used_output_idxs),
        *(inner_next_states[i] for i in untraced_idxs),
    ]
    if op.has_while_condition:
        # The legacy Scan stops when the condition output is True
        inner_outputs.append(invert(inner_cond))

    info = ScanInfo(
        n_seqs=len(inner_seqs),
        mit_mot_in_slices=(),
        mit_mot_out_slices=(),
        mit_sot_in_slices=(),
        sit_sot_in_slices=((-1,),) * len(traced_idxs),
        n_nit_sot=len(used_output_idxs),
        n_untraced_sit_sot=len(untraced_idxs),
        n_non_seqs=len(remaining_const_idxs),
        as_while=op.has_while_condition,
    )
    legacy_op = LegacyScan(inner_inputs, inner_outputs, info, strict=True)

    outer_inputs = [
        max_iters,
        *outer_seqs,
        *(expand_empty(shape_padleft(init_states[i]), max_iters) for i in traced_idxs),
        *(init_states[i] for i in untraced_idxs),
        *(max_iters for _ in used_output_idxs),  # nit-sot buffer lengths
        *(constants[j] for j in remaining_const_idxs),
    ]
    new_outs = legacy_op(*outer_inputs, return_list=True)
    n_traced = len(traced_idxs)
    n_nit = len(used_output_idxs)
    buffers = new_outs[:n_traced]
    nit_outs = new_outs[n_traced : n_traced + n_nit]
    untraced_outs = new_outs[n_traced + n_nit :]

    # The legacy outputs lose the static length information of the trace types,
    # so the replacements must be filtered back into the original types
    replacements = {}
    for i, buffer in zip(traced_idxs, buffers):
        replacements[old_finals[i]] = old_finals[i].type.filter_variable(buffer[-1])
        replacements[old_traces[i]] = old_traces[i].type.filter_variable(buffer[1:])
    # nit-sot buffers are the output traces directly (no leading init row)
    for k, nit_out in zip(used_output_idxs, nit_outs):
        replacements[output_traces[k]] = output_traces[k].type.filter_variable(nit_out)
    for i, untraced_out in zip(untraced_idxs, untraced_outs):
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
