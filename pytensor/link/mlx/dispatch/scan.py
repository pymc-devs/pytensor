from itertools import chain

import mlx.core as mx
import numpy as np

from pytensor.link.mlx.dispatch.basic import convert_dtype_to_mlx, mlx_funcify
from pytensor.scan.op import Scan
from pytensor.tensor.basic import get_scalar_constant_value
from pytensor.tensor.exceptions import NotScalarConstantError


@mlx_funcify.register(Scan)
def mlx_funcify_Scan(op: Scan, node, **kwargs):
    # Mirrors the JAX dispatch (`link/jax/dispatch/scan.py`): the loop recreates
    # the concatenated trace from the per-step values and prepends the initial
    # state / truncates as needed, instead of writing into the PyTensor buffers.
    #
    # Workaround until MLX exposes a native loop primitive (ml-explore/mlx#1441,
    # an `mx` equivalent of `jax.lax.scan`/`while_loop`): MLX has none today, so
    # JAX's `lax.scan` is replaced by a plain Python carry loop that is unrolled
    # into the graph at trace time, which `mx.compile` then compiles. The cost
    # is structural -- it needs a statically known number of steps and the graph
    # grows as O(n_steps * inner_ops) (a real primitive would keep it
    # O(inner_ops)). When the primitive lands, lower to it instead of unrolling.
    # `n_steps` is read from the (full-sized) recurring buffers since
    # `scan_reduce_trace_prealloc` is excluded for MLX.
    info = op.info

    if info.as_while:
        raise NotImplementedError("While Scan cannot yet be converted to MLX")

    # NIT-SOT output lengths are runtime scalars under `mx.compile`; take the
    # static output shape when known and fall back to ``n_steps`` otherwise.
    nitsot_static_sizes = [
        out.type.shape[0] for out in op.outer_nitsot_outs(node.outputs)
    ]
    traced_outputs = (
        op.outer_mitsot_outs(node.outputs)
        + op.outer_sitsot_outs(node.outputs)
        + op.outer_nitsot_outs(node.outputs)
    )
    traced_dtypes = [convert_dtype_to_mlx(out.type.dtype) for out in traced_outputs]
    traced_trailing_shapes = [out.type.shape[1:] for out in traced_outputs]

    # A constant ``n_steps`` is authoritative (and the only inference source for
    # scans without recurring buffers or sequences, e.g. a pure NIT-SOT map).
    try:
        static_n_steps = int(get_scalar_constant_value(node.inputs[0]))
    except NotScalarConstantError:
        static_n_steps = None

    scan_inner_func = mlx_funcify(op.fgraph, **kwargs)

    def scan(*outer_inputs):
        outer_inputs = list(outer_inputs)
        n_steps = _infer_n_steps(op, outer_inputs, nitsot_static_sizes, static_n_steps)
        seqs = [seq[:n_steps] for seq in op.outer_seqs(outer_inputs)]

        # MIT-MOT have no "initial state"; the whole buffer is meaningful.
        # MIT-SOT and SIT-SOT initial states are extracted from the buffers.
        # The ``_init`` states are kept untouched (they prepend the final traces),
        # while ``carry`` copies evolve through the loop.
        mit_sot_init = [
            buff[: -min(tap)]
            for buff, tap in zip(
                op.outer_mitsot(outer_inputs), info.mit_sot_in_slices, strict=True
            )
        ]
        sit_sot_init = [buff[0] for buff in op.outer_sitsot(outer_inputs)]

        mit_mot = list(op.outer_mitmot(outer_inputs))
        mit_sot = list(mit_sot_init)
        sit_sot = list(sit_sot_init)
        untraced_sit_sot = list(op.outer_untraced_sit_sot(outer_inputs))
        non_seqs = op.outer_non_seqs(outer_inputs)

        n_traced = info.n_mit_sot + info.n_sit_sot + info.n_nit_sot
        traces: list[list] = [[] for _ in range(n_traced)]
        for i in range(n_steps):
            inner_seqs = [seq[i] for seq in seqs]
            mit_mot_flatten = list(
                chain.from_iterable(
                    buffer[[i + tap for tap in taps]]
                    for buffer, taps in zip(
                        mit_mot, info.normalized_mit_mot_in_slices, strict=True
                    )
                )
            )
            mit_sot_flatten = list(
                chain.from_iterable(
                    buffer[list(taps)]
                    for buffer, taps in zip(
                        mit_sot, info.mit_sot_in_slices, strict=True
                    )
                )
            )

            inner_outs = list(
                scan_inner_func(
                    *inner_seqs,
                    *mit_mot_flatten,
                    *mit_sot_flatten,
                    *sit_sot,
                    *untraced_sit_sot,
                    *non_seqs,
                )
            )

            new_mit_mot_vals = op.inner_mitmot_outs_grouped(inner_outs)
            new_mit_sot_vals = op.inner_mitsot_outs(inner_outs)
            new_sit_sot = op.inner_sitsot_outs(inner_outs)
            new_nit_sot = op.inner_nitsot_outs(inner_outs)
            new_untraced_sit_sot = op.inner_untraced_sit_sot_outs(inner_outs)

            # Write the new MIT-MOT values at the output-tap positions.
            mit_mot = [
                _functional_set(
                    buffer, [i + tap for tap in taps], mx.stack(new_vals, axis=0)
                )
                for buffer, new_vals, taps in zip(
                    mit_mot,
                    new_mit_mot_vals,
                    info.normalized_mit_mot_out_slices,
                    strict=True,
                )
            ]
            # Discard oldest MIT-SOT tap and append the newest value.
            mit_sot = [
                mx.concatenate([buffer[1:], new_val[None, ...]], axis=0)
                for buffer, new_val in zip(mit_sot, new_mit_sot_vals, strict=True)
            ]
            sit_sot = new_sit_sot
            untraced_sit_sot = new_untraced_sit_sot

            step_traced = [*new_mit_sot_vals, *new_sit_sot, *new_nit_sot]
            for trace, val in zip(traces, step_traced, strict=True):
                trace.append(val)

        init_states = mit_sot_init + sit_sot_init + [None] * info.n_nit_sot
        stacked_traces = []
        for trace, trailing, dtype, init_state in zip(
            traces, traced_trailing_shapes, traced_dtypes, init_states, strict=True
        ):
            if trace:
                stacked_traces.append(mx.stack(trace, axis=0))
            else:
                if None in trailing:
                    if init_state is None:
                        raise NotImplementedError(
                            "MLX Scan requires a static trailing shape for zero-step "
                            "NIT-SOT outputs."
                        )
                    # Unknown recurrent dimensions come from the initial buffer.
                    init_shape = (
                        init_state.shape[1:]
                        if init_state.ndim > len(trailing)
                        else init_state.shape
                    )
                    trailing = tuple(
                        dim if dim is not None else init_dim
                        for dim, init_dim in zip(trailing, init_shape, strict=True)
                    )
                stacked_traces.append(mx.zeros((0, *trailing), dtype=dtype))

        def get_partial_traces(traces):
            """Prepend initial states and slice traces down to buffer sizes."""
            buffer_sizes = (
                [buff.shape[0] for buff in op.outer_mitsot(outer_inputs)]
                + [buff.shape[0] for buff in op.outer_sitsot(outer_inputs)]
                + [
                    size if size is not None else n_steps
                    for size in nitsot_static_sizes
                ]
            )
            partial_traces = []
            for init_state, trace, buffer_size in zip(
                init_states, traces, buffer_sizes, strict=True
            ):
                if init_state is not None:
                    if trace.shape[0] >= buffer_size:
                        # Trace at least as long as the buffer: keep the tail.
                        partial_trace = trace[-buffer_size:]
                    else:
                        # Trace shorter than the buffer: prepend (part of) init.
                        if init_state.ndim < trace.ndim:
                            init_state = init_state[None]
                        if (
                            n_init_needed := buffer_size - trace.shape[0]
                        ) < init_state.shape[0]:
                            init_state = init_state[-n_init_needed:]
                        partial_trace = mx.concatenate([init_state, trace], axis=0)
                else:
                    partial_trace = (
                        trace[-buffer_size:] if trace.shape[0] > buffer_size else trace
                    )

                assert partial_trace.shape[0] == buffer_size
                partial_traces.append(partial_trace)

            return partial_traces

        scan_outs_final = [
            *mit_mot,
            *get_partial_traces(stacked_traces),
            *untraced_sit_sot,
        ]

        if len(scan_outs_final) == 1:
            return scan_outs_final[0]
        return scan_outs_final

    return scan


def _infer_n_steps(op, outer_inputs, nitsot_static_sizes, static_n_steps):
    """Derive the number of steps for the unrolled loop.

    Scalar input values are not readable while ``mx.compile`` traces, but array
    shapes are concrete. A constant ``n_steps`` is used directly; otherwise the
    count comes from a recurring buffer (which stays full-sized because the
    trace-prealloc reduction is disabled for MLX, so each encodes ``n_steps``
    plus its initial taps) or a sequence. A non-constant ``n_steps`` with no
    such buffer (e.g. a dynamic-length pure ``map``) is an MLX static-shape
    limitation, like dynamic ``arange``.
    """
    info = op.info
    if static_n_steps is not None:
        return static_n_steps
    for buff in op.outer_sitsot(outer_inputs):
        return buff.shape[0] - 1
    for buff, taps in zip(
        op.outer_mitsot(outer_inputs), info.mit_sot_in_slices, strict=True
    ):
        return buff.shape[0] + min(taps)
    for seq in op.outer_seqs(outer_inputs):
        return seq.shape[0]
    for size in nitsot_static_sizes:
        if size is not None:
            return size
    raise NotImplementedError(
        "MLX Scan requires a statically known number of steps when there are no "
        "recurring buffers or sequences to infer it from."
    )


def _functional_set(buffer, idx, vals):
    """Return ``buffer`` with distinct rows ``idx`` set to ``vals``.

    MLX has no ``.at[].set`` and in-place item assignment aliases buffers under
    ``mx.compile``. A static row mask selects replacements without arithmetic.
    """
    if not idx:
        return buffer
    mask = np.zeros(buffer.shape[0], dtype=bool)
    mask[idx] = True
    row_selector = np.zeros(buffer.shape[0], dtype=np.int32)
    row_selector[idx] = np.arange(len(idx), dtype=np.int32)
    mask = mx.array(mask).reshape((buffer.shape[0],) + (1,) * (buffer.ndim - 1))
    vals = vals.astype(buffer.dtype)
    return mx.where(mask, vals[mx.array(row_selector)], buffer)
