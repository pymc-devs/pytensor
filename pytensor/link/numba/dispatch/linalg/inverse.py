import numpy as np

from pytensor import config
from pytensor.link.numba.dispatch import basic as numba_basic
from pytensor.link.numba.dispatch.basic import register_funcify_default_op_cache_key
from pytensor.tensor._linalg.inverse import MatrixInverse, MatrixPinv


@register_funcify_default_op_cache_key(MatrixInverse)
def numba_funcify_MatrixInverse(op, node, **kwargs):
    out_dtype = node.outputs[0].type.numpy_dtype
    discrete_input = node.inputs[0].type.numpy_dtype.kind in "ibu"
    if discrete_input and config.compiler_verbose:
        print("MatrixInverse requires casting discrete input to float")  # noqa: T201

    @numba_basic.numba_njit
    def matrix_inverse(x):
        if discrete_input:
            x = x.astype(out_dtype)
        return np.linalg.inv(x)

    cache_version = 1
    return matrix_inverse, cache_version


@register_funcify_default_op_cache_key(MatrixPinv)
def numba_funcify_MatrixPinv(op, node, **kwargs):
    out_dtype = node.outputs[0].type.numpy_dtype
    discrete_input = node.inputs[0].type.numpy_dtype.kind in "ibu"
    if discrete_input and config.compiler_verbose:
        print("MatrixPinv requires casting discrete input to float")  # noqa: T201

    @numba_basic.numba_njit
    def matrix_pinv(x):
        if discrete_input:
            x = x.astype(out_dtype)
        return np.linalg.pinv(x)

    cache_version = 1
    return matrix_pinv, cache_version
