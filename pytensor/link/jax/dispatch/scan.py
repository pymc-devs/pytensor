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

    if info.n_mit_mot:
        raise NotImplementedError(
            "Scan with MIT-MOT (gradients of scan) cannot yet be converted to JAX"
        )

    # Optimize inner graph (exclude any defalut rewrites that are incompatible with JAX mode)
    rewriter = op.mode_instance.excluding(*JAX._optimizer.exclude).optimizer
    rewriter(op.fgraph)
    scan_inner_func = jax_funcify(op.fgraph, **kwargs)

    def scan(*outer_inputs):
        # Extract JAX scan inputs
        outer_inputs = list(outer_inputs)
        n_steps = outer_inputs[0]  # JAX `length`
        seqs = op.outer_seqs(outer_inputs)  # JAX `xs`

        mit_sot_init = []
        for tap, seq in zip(op.info.mit_sot_in_slices, op.outer_mitsot(outer_inputs)):
            init_slice = seq[: abs(min(tap))]
            mit_sot_init.append(init_slice)

        sit_sot_init = [seq[0] for seq in op.outer_sitsot(outer_inputs)]

        init_carry = (
            mit_sot_init,
            sit_sot_init,
            op.outer_shared(outer_inputs),
            op.outer_non_seqs(outer_inputs),
        )  # JAX `init`

        def jax_args_to_inner_func_args(carry, x):
            """Convert JAX scan arguments into format expected by scan_inner_func.

            scan(carry, x) -> scan_inner_func(seqs, mit_sot, sit_sot, shared, non_seqs)
            """

            # `carry` contains all inner taps, shared terms, and non_seqs
            (
                inner_mit_sot,
                inner_sit_sot,
                inner_shared,
                inner_non_seqs,
            ) = carry

            # `x` contains the inner sequences
            inner_seqs = x

            mit_sot_flatten = []
            for array, index in zip(inner_mit_sot, op.info.mit_sot_in_slices):
                mit_sot_flatten.extend(array[jnp.array(index)])

            inner_scan_inputs = [
                *inner_seqs,
                *mit_sot_flatten,
                *inner_sit_sot,
                *inner_shared,
                *inner_non_seqs,
            ]

            return inner_scan_inputs

        def inner_func_outs_to_jax_outs(
            old_carry,
            inner_scan_outs,
        ):
            """Convert inner_scan_func outputs into format expected by JAX scan.

            old_carry + (mit_sot_outs, sit_sot_outs, nit_sot_outs, shared_outs) -> (new_carry, ys)
            """
            (
                inner_mit_sot,
                inner_sit_sot,
                inner_shared,
                inner_non_seqs,
            ) = old_carry

            inner_mit_sot_outs = op.inner_mitsot_outs(inner_scan_outs)
            inner_sit_sot_outs = op.inner_sitsot_outs(inner_scan_outs)
            inner_nit_sot_outs = op.inner_nitsot_outs(inner_scan_outs)
            inner_shared_outs = op.inner_shared_outs(inner_scan_outs)

            # Replace the oldest mit_sot tap by the newest value
            inner_mit_sot_new = [
                jnp.concatenate([old_mit_sot[1:], new_val[None, ...]], axis=0)
                for old_mit_sot, new_val in zip(
                    inner_mit_sot,
                    inner_mit_sot_outs,
                )
            ]

            # Nothing needs to be done with sit_sot
            inner_sit_sot_new = inner_sit_sot_outs

            inner_shared_new = inner_shared
            # Replace old shared inputs by new shared outputs
            inner_shared_new[: len(inner_shared_outs)] = inner_shared_outs

            new_carry = (
                inner_mit_sot_new,
                inner_sit_sot_new,
                inner_shared_new,
                inner_non_seqs,
            )

            # Shared variables and non_seqs are not traced
            traced_outs = [
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

            init_states = mit_sot_init + sit_sot_init + [None] * op.info.n_nit_sot
            buffers = (
                op.outer_mitsot(outer_inputs)
                + op.outer_sitsot(outer_inputs)
                + op.outer_nitsot(outer_inputs)
            )
            partial_traces = []
            for init_state, trace, buffer in zip(init_states, traces, buffers):
                if init_state is not None:
                    # MIT-SOT and SIT-SOT: The final output should be as long as the input buffer
                    trace = jnp.atleast_1d(trace)
                    init_state = jnp.expand_dims(
                        init_state, range(trace.ndim - init_state.ndim)
                    )
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
