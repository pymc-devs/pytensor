import mlx.core as mx

from pytensor.link.mlx.dispatch import mlx_funcify
from pytensor.tensor.blockwise import Blockwise, _check_runtime_broadcast_core


def _reshape_stream(dtype):
    # The default (GPU) stream does not support float64, so pin the squeeze and
    # expand_dims of a float64 array to the CPU stream where it survives. Other
    # dtypes stay on the default stream.
    return mx.cpu if dtype == "float64" else None


@mlx_funcify.register(Blockwise)
def funcify_Blockwise(op: Blockwise, node, **kwargs):
    core_node = op._create_dummy_core_node(node.inputs)
    core_f = mlx_funcify(op.core_op, node=core_node)

    batch_ndim = op.batch_ndim(node)
    if batch_ndim == 0:
        return core_f

    multi_output = len(op.outputs_sig) > 1
    in_core = [len(sig) for sig in op.inputs_sig]

    # Decide batching purely from static shapes so a graph batches identically
    # here and in every other backend: a batch axis broadcasts (is never mapped)
    # only when its static size is exactly 1, or the input lacks it entirely.
    batch_bcast = [inp.type.broadcastable[:batch_ndim] for inp in node.inputs]
    squeeze_axes, padded_batch, squeeze_stream = [], [], []
    for inp, n_core in zip(node.inputs, in_core):
        batch_shape = inp.type.shape[: inp.type.ndim - n_core]
        squeeze_axes.append(tuple(i for i, s in enumerate(batch_shape) if s == 1))
        padded_batch.append((1,) * (batch_ndim - len(batch_shape)) + tuple(batch_shape))
        squeeze_stream.append(_reshape_stream(inp.type.dtype))

    # Nest one mx.vmap per mapped batch axis (innermost first, so array axis 0
    # tracks the outermost batch dim). All-broadcast axes are squeezed out of
    # every input above and re-inserted as size-1 dims after the mapped call.
    fn, expand_axes = core_f, []
    for axis in reversed(range(batch_ndim)):
        in_axes = tuple(None if batch[axis] == 1 else 0 for batch in padded_batch)
        if all(ax is None for ax in in_axes):
            expand_axes.append(axis)
        else:
            fn = mx.vmap(fn, in_axes=in_axes)

    expand_axes.sort()
    expand_stream = [_reshape_stream(out.type.dtype) for out in node.outputs]

    def blockwise(*args):
        # Verify the static broadcast pattern holds: a runtime size-1 batch dim
        # that is not statically broadcastable must not silently broadcast here
        # when every other backend would reject it.
        _check_runtime_broadcast_core(args, batch_bcast, batch_ndim)

        squeezed = [
            mx.squeeze(arg, axes, stream=stream) if axes else arg
            for arg, axes, stream in zip(args, squeeze_axes, squeeze_stream)
        ]
        out = fn(*squeezed)
        if not expand_axes:
            return out

        # Re-insert the never-mapped all-broadcast axes as size-1 dims, in
        # ascending order so each insertion's index stays valid for the next.
        outs = out if multi_output else (out,)
        for ax in expand_axes:
            outs = [
                mx.expand_dims(o, ax, stream=stream)
                for o, stream in zip(outs, expand_stream)
            ]
        return tuple(outs) if multi_output else outs[0]

    return blockwise
