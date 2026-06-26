from itertools import chain

import mlx.core as mx

from pytensor.compile.mode import MLX, get_mode
from pytensor.link.mlx.dispatch.basic import mlx_funcify
from pytensor.scan.op import Scan
from pytensor.tensor.basic import get_scalar_constant_value
from pytensor.tensor.exceptions import NotScalarConstantError


@mlx_funcify.register(Scan)
def mlx_funcify_Scan(op: Scan, node, **kwargs):
    # Mirrors the JAX dispatch (`link/jax/dispatch/scan.py`): the loop recreates
    # the concatenated trace from the per-step values and prepends the initial
    # state / truncates as needed, instead of writing into the PyTensor buffers.
    # MLX has no native general-scan primitive, so JAX's `lax.scan` is replaced
    # by a Python carry loop that `mx.compile` unrolls. This needs a statically
    # known number of steps, which is read from the (full-sized) recurring
    # buffers since `scan_reduce_trace_prealloc` is excluded for MLX.
    info = op.info

    if info.as_while:
        raise NotImplementedError("While Scan cannot yet be converted to MLX")

    # NIT-SOT output lengths are runtime scalars under `mx.compile`; take the
    # static output shape when known and fall back to ``n_steps`` otherwise.
    nitsot_static_sizes = [
        out.type.shape[0] for out in op.outer_nitsot_outs(node.outputs)
    ]

    # A constant ``n_steps`` is authoritative (and the only inference source for
    # scans without recurring buffers or sequences, e.g. a pure NIT-SOT map).
    try:
        static_n_steps = int(get_scalar_constant_value(node.inputs[0]))
    except NotScalarConstantError:
        static_n_steps = None

    rewriter = (
        get_mode(op.mode)
        .including("mlx")
        .excluding("numba", *MLX._optimizer.exclude)
        .optimizer
    )
    rewriter(op.fgraph)
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

        # Per-step shape of each traced output (for synthesizing empty traces
        # when ``n_steps == 0``): MIT-SOT/SIT-SOT match their state shape.
        traced_trailing = (
            [tuple(init.shape[1:]) for init in mit_sot_init]
            + [tuple(init.shape) for init in sit_sot_init]
            + [() for _ in range(info.n_nit_sot)]
        )
        stacked_traces = [
            mx.stack(trace, axis=0) if trace else mx.zeros((0, *trailing))
            for trace, trailing in zip(traces, traced_trailing, strict=True)
        ]

        def get_partial_traces(traces):
            """Prepend initial states and slice traces down to buffer sizes."""
            init_states = mit_sot_init + sit_sot_init + [None] * info.n_nit_sot
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
    for buff, in_taps, out_taps in zip(
        op.outer_mitmot(outer_inputs),
        info.normalized_mit_mot_in_slices,
        info.normalized_mit_mot_out_slices,
        strict=True,
    ):
        return buff.shape[0] - (max(*in_taps, *out_taps) - min(*in_taps, *out_taps))
    for size in nitsot_static_sizes:
        if size is not None:
            return size
    raise NotImplementedError(
        "MLX Scan requires a statically known number of steps when there are no "
        "recurring buffers or sequences to infer it from."
    )


def _functional_set(buffer, idx, vals):
    """Return ``buffer`` with rows ``idx`` set to ``vals``.

    MLX has no ``.at[].set`` and in-place item assignment aliases buffers under
    ``mx.compile``, so a functional scatter-add of the delta is used instead.
    """
    idx = mx.array(idx)
    return buffer.at[idx].add(vals - buffer[idx])
