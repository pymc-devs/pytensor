from itertools import chain

import jax.numpy as jnp
import numpy as np
from jax._src.lax.control_flow import fori_loop

from pytensor.compile.mode import JAX, get_mode
from pytensor.link.jax.dispatch.basic import jax_funcify
from pytensor.scan.op import Scan


def call_inner_func_with_indexed_buffers(
    info,
    scan_inner_func,
    i,
    sequences,
    mit_mot_buffers,
    mit_sot_buffers,
    sit_sot_buffers,
    shareds,
    non_sequences,
):
    sequence_vals = [seq[i] for seq in sequences]

    # chain.from_iterable is used flatten the first dimension of each indexed buffer
    # [buf1[[idx0, idx1]], buf2[[idx0, idx1]]] -> [buf1[idx0], buf1[idx1], buf2[idx0], buf2[idx1]]
    # Benchmarking suggests unpacking advanced indexing on all taps is faster than basic index one tap at a time
    mit_mot_vals = list(
        chain.from_iterable(
            buffer[(i + np.array(in_taps))]
            for buffer, in_taps in zip(
                mit_mot_buffers, info.mit_mot_in_slices, strict=True
            )
        )
    )
    mit_sot_vals = list(
        chain.from_iterable(
            # Convert negative taps (-2, -1) to positive indices (0, 1)
            buffer[((i + (np.array(in_taps) - min(in_taps))) % buffer.shape[0])]
            for buffer, in_taps in zip(
                mit_sot_buffers, info.mit_sot_in_slices, strict=True
            )
        )
    )
    sit_sot_vals = [buffer[i % buffer.shape[0]] for buffer in sit_sot_buffers]

    return scan_inner_func(
        *sequence_vals,
        *mit_mot_vals,
        *mit_sot_vals,
        *sit_sot_vals,
        *shareds,
        *non_sequences,
    )


def update_buffers(buffers, update_vals, indices, may_roll: bool = True):
    return tuple(
        buffer.at[(index % buffer.shape[0]) if may_roll else index].set(update_val)
        for buffer, update_val, index in zip(buffers, update_vals, indices, strict=True)
    )


def align_buffers(buffers, n_steps, max_taps):
    return [
        jnp.roll(
            buffer,
            shift=jnp.where(
                # Only needs rolling if last write position is beyond the buffer length
                (n_steps + max_tap) > buffer.shape[0],
                # Roll left by the amount of overflow
                -((n_steps + max_tap + 1) % buffer.shape[0]),
                0,
            ),
            axis=0,
        )
        for buffer, max_tap in zip(buffers, max_taps, strict=True)
    ]


@jax_funcify.register(Scan)
def jax_funcify_Scan(op: Scan, node, **kwargs):
    op = op  # Need to bind to a local variable
    info = op.info

    if info.as_while:
        raise NotImplementedError("While Scan cannot yet be converted to JAX")

    # Optimize inner graph (exclude any defalut rewrites that are incompatible with JAX mode)
    rewriter = (
        get_mode(op.mode).including("jax").excluding(*JAX._optimizer.exclude).optimizer
    )
    rewriter(op.fgraph)
    # TODO: Use scan name from Op when available
    scan_inner_func = jax_funcify(op.fgraph, fgraph_name="scan_inner_func", **kwargs)

    def scan(*outer_inputs, op=op, node=node):
        n_steps = outer_inputs[0]
        sequences = op.outer_seqs(outer_inputs)
        has_empty_sequences = any(seq.shape[0] == 0 for seq in sequences)
        init_mit_mot_buffers = op.outer_mitmot(outer_inputs)
        init_mit_sot_buffers = op.outer_mitsot(outer_inputs)
        init_sit_sot_buffers = op.outer_sitsot(outer_inputs)
        nit_sot_buffer_lens = op.outer_nitsot(outer_inputs)
        # Shareds are special-cased SIT-SOTs that are not traced, but updated at each step.
        # Only last value is returned. It's a hack for special types (like RNG) that can't be "concatenated" over time.
        init_shareds = op.outer_shared(outer_inputs)
        non_sequences = op.outer_non_seqs(outer_inputs)
        assert (
            1
            + len(sequences)
            + len(init_mit_mot_buffers)
            + len(init_mit_sot_buffers)
            + len(init_sit_sot_buffers)
            + len(nit_sot_buffer_lens)
            + len(init_shareds)
            + len(non_sequences)
        ) == len(outer_inputs)

        # Initialize NIT-SOT buffers
        if nit_sot_buffer_lens:
            if has_empty_sequences:
                # In this case we cannot call the inner function to infer the shapes of the nit_sot outputs
                # So we must rely on static shapes of the outputs (if available)
                nit_sot_core_shapes = [
                    n.type.shape for n in op.inner_nitsot_outs(op.fgraph.outputs)
                ]
                if any(d is None for shape in nit_sot_core_shapes for d in shape):
                    raise ValueError(
                        "Scan with NIT-SOT outputs (None in outputs_info) cannot have 0 steps unless the output shapes are statically known)\n"
                        f"The static shapes of the NIT-SOT outputs for this Scan {node.op} are: {nit_sot_core_shapes}."
                    )

            else:
                # Otherwise, call the function once to get the shapes and dtypes of the nit_sot outputs
                buffer_vals = call_inner_func_with_indexed_buffers(
                    info,
                    scan_inner_func,
                    0,
                    sequences,
                    init_mit_mot_buffers,
                    init_mit_sot_buffers,
                    init_sit_sot_buffers,
                    init_shareds,
                    non_sequences,
                )
                nit_sot_core_shapes = [
                    n.shape for n in op.inner_nitsot_outs(buffer_vals)
                ]
            nit_sot_dtypes = [
                n.type.dtype for n in op.inner_nitsot_outs(op.fgraph.outputs)
            ]
            init_nit_sot_buffers = tuple(
                jnp.empty(
                    (nit_sot_buffer_len, *nit_sot_core_shape),
                    dtype=nit_sot_dtype,
                )
                for nit_sot_buffer_len, nit_sot_core_shape, nit_sot_dtype in zip(
                    nit_sot_buffer_lens,
                    nit_sot_core_shapes,
                    nit_sot_dtypes,
                    strict=True,
                )
            )
        else:
            init_nit_sot_buffers = ()

        if has_empty_sequences:
            # fori_loop still gets called with n_steps=0, which would raise an IndexError, we return early here
            init_vals = (
                *init_mit_mot_buffers,
                *init_mit_sot_buffers,
                *init_sit_sot_buffers,
                *init_nit_sot_buffers,
                *init_shareds,
            )
            return init_vals[0] if len(init_vals) == 1 else init_vals

        def body_fun(i, prev_vals):
            (
                mit_mot_buffers,
                mit_sot_buffers,
                sit_sot_buffers,
                nit_sot_buffers,
                shareds,
            ) = prev_vals

            next_vals = call_inner_func_with_indexed_buffers(
                info,
                scan_inner_func,
                i,
                sequences,
                mit_mot_buffers,
                mit_sot_buffers,
                sit_sot_buffers,
                shareds,
                non_sequences,
            )
            # For MIT-MOT buffers, we want to store at the positions indicated by the output taps
            mit_mot_updated_buffers = update_buffers(
                mit_mot_buffers,
                op.inner_mitmot_outs_grouped(next_vals),
                # Taps are positive, we stack them to obtain advanced indices
                indices=[i + jnp.stack(taps) for taps in info.mit_mot_out_slices],
                # MIT-MOT buffers never roll, as they are never truncated
                may_roll=False,
            )
            # For regular buffers, we want to store at the position after the last reading
            mit_sot_updated_buffers = update_buffers(
                mit_sot_buffers,
                op.inner_mitsot_outs(next_vals),
                indices=[i - min(taps) for taps in info.mit_sot_in_slices],
            )
            sit_sot_updated_buffers = update_buffers(
                sit_sot_buffers,
                op.inner_sitsot_outs(next_vals),
                # Taps are always -1 for SIT-SOT, so we just use i + 1
                indices=[i + 1 for _ in sit_sot_buffers],
            )
            nit_sot_updated_buffers = update_buffers(
                nit_sot_buffers,
                op.inner_nitsot_outs(next_vals),
                # Taps are always 0 for NIT-SOT, so we just use i
                indices=[i for _ in nit_sot_buffers],
            )
            shareds_update_vals = op.inner_shared_outs(next_vals)

            return (
                mit_mot_updated_buffers,
                mit_sot_updated_buffers,
                sit_sot_updated_buffers,
                nit_sot_updated_buffers,
                shareds_update_vals,
            )

        (
            updated_mit_mot_buffers,
            updated_mit_sot_buffers,
            updated_sit_sot_buffers,
            updated_nit_sot_buffers,
            updated_shareds,
        ) = fori_loop(
            0,
            n_steps,
            body_fun,
            init_val=(
                init_mit_mot_buffers,
                init_mit_sot_buffers,
                init_sit_sot_buffers,
                init_nit_sot_buffers,
                init_shareds,
            ),
        )

        # Roll the output buffers to match PyTensor Scan semantics
        # MIT-MOT buffers are never truncated, so no rolling is needed
        aligned_mit_mot_buffers = updated_mit_mot_buffers
        aligned_mit_sot_buffers = align_buffers(
            updated_mit_sot_buffers,
            n_steps,
            # (-3, -1) -> max is 2
            max_taps=[-min(taps) - 1 for taps in info.mit_sot_in_slices],
        )

        aligned_sit_sot_buffers = align_buffers(
            updated_sit_sot_buffers,
            n_steps,
            max_taps=[0 for _ in updated_sit_sot_buffers],
        )
        aligned_nit_sot_buffers = align_buffers(
            updated_nit_sot_buffers,
            n_steps,
            max_taps=[0 for _ in updated_nit_sot_buffers],
        )

        all_outputs = tuple(
            chain.from_iterable(
                (
                    aligned_mit_mot_buffers,
                    aligned_mit_sot_buffers,
                    aligned_sit_sot_buffers,
                    aligned_nit_sot_buffers,
                    updated_shareds,
                )
            )
        )
        return all_outputs[0] if len(all_outputs) == 1 else all_outputs

    return scan
