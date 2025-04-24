from pytensor.graph.basic import Constant
from pytensor.graph.rewriting.basic import copy_stack_trace, node_rewriter
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.rewriting.basic import register_specialize, register_stabilize
from pytensor.tensor.signal import convolve1d
from pytensor.tensor.signal.conv import Convolve1d
from pytensor.tensor.subtensor import Subtensor, indices_from_subtensor


@register_stabilize
@register_specialize
@node_rewriter([Subtensor])
def local_sliced_full_conv_to_valid_conv(fgraph, node):
    """Rewrite sliced full conv that are equivalent to valid.

    The gradient of a valid Conv1d always implements the worst case scenario - full convolution -
    because it would need to know which input is larger to do something smarter.
    If we find out (through rewrites or static shape) we provide the direct implementation
    which can be orders of magnitude faster.

    # if x.shape[-1] > y.shape[-1]
    # z = convolve1d(x, y, mode="full")
    # z[..., y.shape[-1] - 1: z.shape[-1] - y.shape[-1] - 1] -> convolve1d(x, y, mode="valid")
    """
    conv, *other_idx_vars = node.inputs

    if not (
        conv.owner is not None
        and isinstance(conv.owner.op, Blockwise)
        and isinstance(conv.owner.op.core_op, Convolve1d)
        and conv.owner.op.core_op.mode == "full"
    ):
        return None

    # Check we have an (a:b) constant slice at the last axis of the input
    idx_list = node.op.idx_list
    if not (len(idx_list) == conv.type.ndim and isinstance(idx_list[-1], slice)):
        return None

    last_slice = idx_list[-1]
    if not (
        last_slice.start is not None
        and last_slice.stop is not None
        and last_slice.step is None
    ):
        return None

    *other_idx_vars, start, stop = other_idx_vars
    if not (isinstance(start, Constant) and isinstance(stop, Constant)):
        return None

    x, y = conv.owner.inputs
    len_x = x.type.shape[-1]
    len_y = y.type.shape[-1]
    if len_x is None or len_y is None:
        return None

    start, stop = start.data, stop.data
    if len_x < len_y:
        # Convolution is symmetric with input order
        x, y = y, x
        len_x, len_y = len_y, len_x

    if (
        start == len_y - 1
        # equivalent to stop = conv.shape[-1] - len_y - 1
        and stop == start + (len_x - len_y) + 1
    ):
        new_conv = convolve1d(x, y, mode="valid")
        copy_stack_trace(conv, new_conv)

        if other_idx_vars:
            # If there were more than just empty slices besides the last one
            new_indices = indices_from_subtensor(idx_list[:-1], other_idx_vars)
            new_conv = new_conv[new_indices]
            copy_stack_trace(node.out, new_conv)

        return [new_conv]
