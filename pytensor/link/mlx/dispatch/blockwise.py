import mlx.core as mx

from pytensor.link.mlx.dispatch import mlx_funcify
from pytensor.tensor.blockwise import Blockwise, _check_runtime_broadcast_core


@mlx_funcify.register(Blockwise)
def funcify_Blockwise(op: Blockwise, node, **kwargs):
    core_node = op._create_dummy_core_node(node.inputs)
    core_f = mlx_funcify(op.core_op, node=core_node)

    batch_ndim = op.batch_ndim(node)
    if batch_ndim == 0:
        return core_f

    multi_output = len(node.outputs) > 1
    core_ndims = [len(sig) for sig in op.inputs_sig]

    # Hoisted out of the per-call path, unlike Blockwise._check_runtime_broadcast.
    batch_bcast = [inp.type.broadcastable[:batch_ndim] for inp in node.inputs]

    # Decide batching purely from static shapes so a graph batches identically
    # here and in every other backend: a batch axis broadcasts (is never mapped)
    # only when its static size is exactly 1, or the input lacks it entirely.
    squeeze_axes, padded_batch = [], []
    for inp, n_core_dims in zip(node.inputs, core_ndims):
        batch_shape = inp.type.shape[: inp.type.ndim - n_core_dims]
        squeeze_axes.append(tuple(i for i, s in enumerate(batch_shape) if s == 1))
        padded_batch.append((1,) * (batch_ndim - len(batch_shape)) + tuple(batch_shape))

    # Nest one mx.vmap per mapped batch axis (innermost first, so array axis 0
    # tracks the outermost batch dim). All-broadcast axes are squeezed out of
    # every input above and re-inserted as size-1 dims after the mapped call.
    fn, expand_axes = core_f, []
    for axis in reversed(range(batch_ndim)):
        in_axes = tuple(None if shape[axis] == 1 else 0 for shape in padded_batch)
        if all(ax is None for ax in in_axes):
            expand_axes.append(axis)
        else:
            fn = mx.vmap(fn, in_axes=in_axes)

    expand_axes.sort()

    def blockwise(*args):
        # Other backends reject a runtime size-1 batch dim that is not statically
        # broadcastable; match them rather than silently broadcasting here.
        _check_runtime_broadcast_core(args, batch_bcast, batch_ndim)

        squeezed = [
            mx.squeeze(arg, axes) if axes else arg
            for arg, axes in zip(args, squeeze_axes)
        ]
        out = fn(*squeezed)
        if not expand_axes:
            return tuple(out) if multi_output else out

        # Re-insert the never-mapped all-broadcast axes as size-1 dims, in
        # ascending order so each insertion's index stays valid for the next.
        outs = out if multi_output else (out,)
        for ax in expand_axes:
            outs = [mx.expand_dims(o, ax) for o in outs]
        return tuple(outs) if multi_output else outs[0]

    return blockwise
