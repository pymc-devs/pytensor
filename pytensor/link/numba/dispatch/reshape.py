import numpy as np

from pytensor.link.numba.dispatch import basic as numba_basic
from pytensor.link.numba.dispatch.basic import register_funcify_default_op_cache_key
from pytensor.tensor.reshape import JoinDims, SplitDims


@register_funcify_default_op_cache_key(JoinDims)
def numba_funcify_JoinDims(op, node, **kwargs):
    start_axis = op.start_axis
    n_axes = op.n_axes

    @numba_basic.numba_njit
    def join_dims(x):
        output_shape = (*x.shape[:start_axis], -1, *x.shape[start_axis + n_axes :])
        return np.reshape(x, output_shape)

    return join_dims


@register_funcify_default_op_cache_key(SplitDims)
def numba_funcify_SplitDims(op, node, **kwargs):
    axis = op.axis

    @numba_basic.numba_njit
    def split_dims(x, shape):
        output_shape = (*x.shape[:axis], *shape, *x.shape[axis + 1 :])
        return np.reshape(x, output_shape)

    return split_dims
