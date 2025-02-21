import jax
import jax.numpy as jnp

from pytensor.compile.mode import JAX
from pytensor.link.jax.dispatch.basic import jax_funcify
from pytensor.scan.op import Scan


@jax_funcify.register(Scan)
def jax_funcify_Scan(op: Scan, **kwargs):
    info = op.info

    if info.as_while:
        raise NotImplementedError("While Scan cannot yet be converted to JAX")

    # Optimize inner graph (exclude any defalut rewrites that are incompatible with JAX mode)
    rewriter = op.mode_instance.excluding(*JAX._optimizer.exclude).optimizer
    rewriter(op.fgraph)
    scan_inner_func = jax_funcify(op.fgraph, **kwargs)

    def scan(*outer_inputs):
        # Extract JAX scan inputs
        outer_inputs = list(outer_inputs)
        n_steps = outer_inputs[0]  # JAX `length`
        seqs = op.outer_seqs(outer_inputs)  # JAX `xs`

        # MIT-MOT and MIT-SOT are provided from outside as a tape long enough to store the initial values and intermediate outputs
        # To bootstrap the inner function we need to slice the initial values
        mit_mot_inits = []
        for taps, seq in zip(
            op.info.mit_mot_in_slices, op.outer_mitmot(outer_inputs), strict=True
        ):
            # mit-mot taps are non-negative
            init_slice = seq[: max(taps) + 1]
            mit_mot_inits.append(init_slice)

        mit_sot_inits = []
        for taps, seq in zip(
            op.info.mit_sot_in_slices, op.outer_mitsot(outer_inputs), strict=True
        ):
            # mit-sot taps are negative
            init_slice = seq[: abs(min(taps))]
            mit_sot_inits.append(init_slice)

        sit_sot_inits = [seq[0] for seq in op.outer_sitsot(outer_inputs)]

        init_carry = (
            mit_mot_inits,
            mit_sot_inits,
            sit_sot_inits,
            op.outer_shared(outer_inputs),
            op.outer_non_seqs(outer_inputs),
        )  # JAX `init`

        def jax_args_to_inner_func_args(carry, x):
            """Convert JAX scan arguments into format expected by scan_inner_func.

            scan(carry, x) -> scan_inner_func(seqs, mit_mot, mit_sot, sit_sot, shared, non_seqs)
            """

            # `carry` contains all inner taps, shared terms, and non_seqs
            (
                inner_mit_mots,
                inner_mit_sots,
                inner_sit_sots,
                inner_shareds,
                inner_non_seqs,
            ) = carry

            # `x` contains the inner sequences
            inner_seqs = x

            # MIT-MOT and MIT-SOT are provided as unified tensors and should be split
            # into distinct entries for the inner function
            split_mit_mots = []
            for taps, seq in zip(
                op.info.mit_mot_in_slices, inner_mit_mots, strict=True
            ):
                for tap in taps:
                    split_mit_mots.append(seq[tap])

            split_mit_sots = []
            for taps, seq in zip(
                op.info.mit_sot_in_slices, inner_mit_sots, strict=True
            ):
                for tap in taps:
                    split_mit_sots.append(seq[tap])

            inner_scan_inputs = [
                *inner_seqs,
                *split_mit_mots,  # TODO: Confirm oreding
                *split_mit_sots,
                *inner_sit_sots,
                *inner_shareds,
                *inner_non_seqs,
            ]

            return inner_scan_inputs

        def inner_func_outs_to_jax_outs(
            old_carry,
            inner_scan_outs,
        ):
            """Convert inner_scan_func outputs into format expected by JAX scan.

            old_carry + (mit_mot_outs, mit_sot_outs, sit_sot_outs, nit_sot_outs, shared_outs) -> (new_carry, ys)
            """
            (
                inner_mit_mots,
                inner_mit_sots,
                inner_sit_sots,
                inner_shareds,
                inner_non_seqs,
            ) = old_carry

            inner_mit_mot_outs = op.inner_mitmot_outs(inner_scan_outs)
            inner_mit_sot_outs = op.inner_mitsot_outs(inner_scan_outs)
            inner_sit_sot_outs = op.inner_sitsot_outs(inner_scan_outs)
            inner_nit_sot_outs = op.inner_nitsot_outs(inner_scan_outs)
            inner_shared_outs = op.inner_shared_outs(inner_scan_outs)

            # Group split mit_mot_outs into the respective groups
            start = 0
            grouped_inner_mit_mot_outs = []
            for mit_mot_out_slice in op.info.mit_mot_out_slices:
                end = start + len(mit_mot_out_slice)
                elements = inner_mit_mot_outs[start:end]
                group = jnp.concatenate([e[None] for e in elements], axis=0)
                grouped_inner_mit_mot_outs.append(group)
                start = end

            # Replace the oldest mit-mot taps (last entries) and prepend the newest values
            new_inner_mit_mots = []
            for old_mit_mot, new_outs in zip(
                inner_mit_mots, grouped_inner_mit_mot_outs, strict=True
            ):
                n_outs = len(new_outs)
                inner_mit_mot_new = jnp.concatenate(
                    [old_mit_mot[n_outs:], group], axis=0
                )
                new_inner_mit_mots.append(inner_mit_mot_new)

            # Drop the oldest mit-sot tap (first entry) and append the newest value at end
            new_inner_mit_sots = []
            for old_mit_sot, new_out in zip(
                inner_mit_sots, inner_mit_sot_outs, strict=True
            ):
                inner_mit_sot_new = jnp.concatenate(
                    [old_mit_sot[1:], new_out[None, ...]], axis=0
                )
                new_inner_mit_mots.append(inner_mit_sot_new)

            # Nothing needs to be done with sit_sot
            new_inner_sit_sots = inner_sit_sot_outs

            new_inner_shareds = inner_shareds
            # Replace old shared inputs by new shared outputs
            new_inner_shareds[: len(inner_shared_outs)] = inner_shared_outs

            new_carry = (
                new_inner_mit_mots,
                new_inner_mit_sots,
                new_inner_sit_sots,
                new_inner_shareds,
                inner_non_seqs,
            )

            # Shared variables and non_seqs are not traced
            traced_outs = [
                *grouped_inner_mit_mot_outs,
                *inner_mit_sot_outs,
                *inner_sit_sot_outs,
                *inner_nit_sot_outs,
            ]

            return new_carry, traced_outs

        def jax_inner_func(carry, x):
            inner_args = jax_args_to_inner_func_args(carry, x)
            inner_scan_outs = list(scan_inner_func(*inner_args))
            new_carry, traced_outs = inner_func_outs_to_jax_outs(carry, inner_scan_outs)
            return new_carry, traced_outs

        # Extract PyTensor scan outputs
        final_carry, traces = jax.lax.scan(
            jax_inner_func, init_carry, seqs, length=n_steps
        )

        def get_partial_traces(traces):
            """Convert JAX scan traces to PyTensor traces.

            We need to:
                1. Prepend initial states to JAX output traces
                2. Slice final traces if Scan was instructed to only keep a portion
            """

            init_states = (
                mit_mot_inits
                + mit_sot_inits
                + sit_sot_inits
                + [None] * op.info.n_nit_sot
            )
            buffers = (
                op.outer_mitmot(outer_inputs)
                + op.outer_mitsot(outer_inputs)
                + op.outer_sitsot(outer_inputs)
                + op.outer_nitsot(outer_inputs)
            )
            partial_traces = []
            for init_state, trace, buffer in zip(
                init_states, traces, buffers, strict=True
            ):
                if init_state is not None:
                    # MIT-MOT, MIT-SOT and SIT-SOT: The final output should be as long as the input buffer
                    trace = jnp.atleast_1d(trace)
                    init_state = jnp.expand_dims(init_state, 1)
                    # TODO: delete this, shouldn't be needed?
                    full_trace = jnp.concatenate([init_state, trace], axis=0)
                    buffer_size = buffer.shape[0]
                else:
                    # NIT-SOT: Buffer is just the number of entries that should be returned
                    full_trace = jnp.atleast_1d(trace)
                    buffer_size = buffer

                partial_trace = full_trace[-buffer_size:]
                partial_traces.append(partial_trace)

            return partial_traces

        def get_shared_outs(final_carry):
            """Retrive last state of shared_outs from final_carry.

            These outputs cannot be traced in PyTensor Scan
            """
            (
                inner_out_mit_sot,
                inner_out_sit_sot,
                inner_out_shared,
                inner_in_non_seqs,
            ) = final_carry

            shared_outs = inner_out_shared[: info.n_shared_outs]
            return list(shared_outs)

        scan_outs_final = get_partial_traces(traces) + get_shared_outs(final_carry)

        if len(scan_outs_final) == 1:
            scan_outs_final = scan_outs_final[0]
        return scan_outs_final

    return scan
