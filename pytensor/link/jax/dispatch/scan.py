from itertools import chain

import jax.numpy as jnp
import numpy as np
from jax._src.lax.control_flow import scan as jax_scan

from pytensor.compile.mode import JAX, get_mode
from pytensor.link.jax.dispatch.basic import jax_funcify
from pytensor.scan.op import Scan


@jax_funcify.register(Scan)
def jax_funcify_Scan(op: Scan, node, **kwargs):
    # Note: This implementation is different from the internal PyTensor Scan op.
    # In particular, we don't make use of the provided buffers for recurring outputs (MIT-SOT, SIT-SOT)
    # These buffers include the initial state and enough space to store as many intermediate results as needed.
    # Instead, we let JAX scan recreate the concatenated buffer itself from the values computed in each iteration,
    # and then prepend the initial_state and/or truncate results we don't need at the end.
    # Likewise, we allow JAX to stack NIT-SOT outputs itself, instead of writing to an empty buffer with the final size.
    # In contrast, MIT-MOT behave like PyTensor Scan. We read from and write to the original buffer as we iterate.
    # Hopefully, JAX can do the same sort of memory optimizations as PyTensor does.
    # Performance-wise, the benchmarks show this approach is better, specially when auto-diffing through JAX.
    # For an implementation that is closer to the internal PyTensor Scan, check intermediate commit in
    # https://github.com/pymc-devs/pytensor/pull/1651
    info = op.info

    if info.as_while:
        raise NotImplementedError("While Scan cannot yet be converted to JAX")

    # Optimize inner graph (exclude any defalut rewrites that are incompatible with JAX mode)
    rewriter = (
        get_mode(op.mode)
        .including("jax")
        .excluding("numba", *JAX._optimizer.exclude)
        .optimizer
    )
    rewriter(op.fgraph)
    scan_inner_func = jax_funcify(op.fgraph, **kwargs)

    def scan(*outer_inputs):
        # Extract JAX scan inputs
        # JAX doesn't want some inputs to be tuple, but later lists (e.g., from list-comprehensions).
        # We convert everything to list, so that it remains a list after slicing.
        outer_inputs = list(outer_inputs)
        n_steps = outer_inputs[0]  # JAX `length`
        seqs = [seq[:n_steps] for seq in op.outer_seqs(outer_inputs)]  # JAX `xs`

        # MIT-MOT don't have a concept of "initial state"
        # The whole buffer is meaningful at the start of the Scan
        mit_mot_init = op.outer_mitmot(outer_inputs)

        # For MIT-SOT and SIT-SOT, extract the initial states from the outer input buffers
        mit_sot_init = [
            buff[: -min(tap)]
            for buff, tap in zip(
                op.outer_mitsot(outer_inputs), op.info.mit_sot_in_slices, strict=True
            )
        ]
        sit_sot_init = [buff[0] for buff in op.outer_sitsot(outer_inputs)]

        init_carry = (
            0,  # loop counter, needed for indexing MIT-MOT
            mit_mot_init,
            mit_sot_init,
            sit_sot_init,
            op.outer_untraced_sit_sot(outer_inputs),
            op.outer_non_seqs(outer_inputs),
        )  # JAX `init`

        def jax_args_to_inner_func_args(carry, x):
            """Convert JAX scan arguments into format expected by scan_inner_func.

            scan(carry, x) -> scan_inner_func(seqs, MIT-SOT, SIT-SOT, untraced SIT-SOT, non_seqs)
            """

            # `carry` contains all inner taps and non_seqs
            (
                i,
                inner_mit_mot,
                inner_mit_sot,
                inner_sit_sot,
                inner_untraced_sit_sot,
                inner_non_seqs,
            ) = carry

            # `x` contains the inner sequences
            inner_seqs = x

            # chain.from_iterable is used to flatten the first dimension of each indexed buffer
            # [buf1[[idx0, idx1]], buf2[[idx0, idx1]]] -> [buf1[idx0], buf1[idx1], buf2[idx0], buf2[idx1]]
            # Benchmarking suggests unpacking advanced indexing on all taps is faster than basic index one tap at a time
            mit_mot_flatten = list(
                chain.from_iterable(
                    buffer[(i + np.array(taps))]
                    for buffer, taps in zip(
                        inner_mit_mot, info.normalized_mit_mot_in_slices, strict=True
                    )
                )
            )
            mit_sot_flatten = list(
                chain.from_iterable(
                    buffer[np.array(taps)]
                    for buffer, taps in zip(
                        inner_mit_sot, info.mit_sot_in_slices, strict=True
                    )
                )
            )

            return (
                *inner_seqs,
                *mit_mot_flatten,
                *mit_sot_flatten,
                *inner_sit_sot,
                *inner_untraced_sit_sot,
                *inner_non_seqs,
            )

        def inner_func_outs_to_jax_outs(
            old_carry,
            inner_scan_outs,
        ):
            """Convert inner_scan_func outputs into format expected by JAX scan.

            old_carry + (MIT-SOT_outs, SIT-SOT_outs, NIT-SOT_outs, untraced_SIT-SOT_outs) -> (new_carry, ys)
            """
            (
                i,
                old_mit_mot,
                old_mit_sot,
                _old_sit_sot,
                _old_untraced_sit_sot,
                inner_non_seqs,
            ) = old_carry

            new_mit_mot_vals = op.inner_mitmot_outs_grouped(inner_scan_outs)
            new_mit_sot_vals = op.inner_mitsot_outs(inner_scan_outs)
            new_sit_sot = op.inner_sitsot_outs(inner_scan_outs)
            new_nit_sot = op.inner_nitsot_outs(inner_scan_outs)
            new_untraced_sit_sot = op.inner_untraced_sit_sot_outs(inner_scan_outs)

            # New carry for next step
            # Update MIT-MOT buffer at positions indicated by output taps
            new_mit_mot = [
                buffer.at[i + np.array(taps)].set(new_vals)
                for buffer, new_vals, taps in zip(
                    old_mit_mot,
                    new_mit_mot_vals,
                    info.normalized_mit_mot_out_slices,
                    strict=True,
                )
            ]
            # Discard oldest MIT-SOT and append newest value
            new_mit_sot = [
                jnp.concatenate([old_buffer[1:], new_val[None, ...]], axis=0)
                for old_buffer, new_val in zip(
                    old_mit_sot, new_mit_sot_vals, strict=True
                )
            ]
            # For SIT-SOT just pass along the new value
            # Non-sequences remain unchanged
            new_carry = (
                i + 1,
                new_mit_mot,
                new_mit_sot,
                new_sit_sot,
                new_untraced_sit_sot,
                inner_non_seqs,
            )

            # Select new MIT-SOT, SIT-SOT, and NIT-SOT for tracing
            traced_outs = [
                *new_mit_sot_vals,
                *new_sit_sot,
                *new_nit_sot,
            ]

            return new_carry, traced_outs

        def jax_inner_func(carry, x):
            inner_args = jax_args_to_inner_func_args(carry, x)
            inner_scan_outs = list(scan_inner_func(*inner_args))
            new_carry, traced_outs = inner_func_outs_to_jax_outs(carry, inner_scan_outs)
            return new_carry, traced_outs

        # Extract PyTensor scan outputs
        (
            (
                _final_i,
                final_mit_mot,
                _final_mit_sot,
                _final_sit_sot,
                final_untraced_sit_sot,
                _final_non_seqs,
            ),
            traces,
        ) = jax_scan(jax_inner_func, init_carry, seqs, length=n_steps)

        def get_partial_traces(traces):
            """Convert JAX scan traces to PyTensor traces.

            We need to:
                1. Prepend initial states to JAX output traces
                2. Slice final traces if Scan was instructed to only keep a portion
            """

            init_states = mit_sot_init + sit_sot_init + [None] * op.info.n_nit_sot
            buffers = (
                op.outer_mitsot(outer_inputs)
                + op.outer_sitsot(outer_inputs)
                + op.outer_nitsot(outer_inputs)
            )
            partial_traces = []
            for init_state, trace, buffer in zip(
                init_states, traces, buffers, strict=True
            ):
                if init_state is not None:
                    # MIT-SOT and SIT-SOT: The final output should be as long as the input buffer
                    buffer_size = buffer.shape[0]
                    if trace.shape[0] > buffer_size:
                        # Trace is longer than buffer, keep just the last `buffer.shape[0]` entries
                        partial_trace = trace[-buffer_size:]
                    elif trace.shape[0] == buffer_size:
                        partial_trace = trace
                    else:
                        # Trace is shorter than buffer, this happens when we keep the initial_state
                        if init_state.ndim < buffer.ndim:
                            init_state = init_state[None]
                        if (
                            n_init_needed := buffer_size - trace.shape[0]
                        ) < init_state.shape[0]:
                            # We may not need to keep all the initial states
                            init_state = init_state[-n_init_needed:]
                        partial_trace = jnp.concatenate([init_state, trace], axis=0)
                else:
                    # NIT-SOT: Buffer is just the number of entries that should be returned
                    buffer_size = buffer
                    partial_trace = (
                        trace[-buffer_size:] if trace.shape[0] > buffer else trace
                    )

                assert partial_trace.shape[0] == buffer_size
                partial_traces.append(partial_trace)

            return partial_traces

        scan_outs_final = [
            *final_mit_mot,
            *get_partial_traces(traces),
            *final_untraced_sit_sot,
        ]

        if len(scan_outs_final) == 1:
            scan_outs_final = scan_outs_final[0]
        return scan_outs_final

    return scan
