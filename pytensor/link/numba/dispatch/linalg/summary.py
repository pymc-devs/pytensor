import numpy as np

from pytensor import config
from pytensor.link.numba.dispatch import basic as numba_basic
from pytensor.link.numba.dispatch.basic import register_funcify_default_op_cache_key
from pytensor.tensor.linalg.summary import Det, SLogDet


@register_funcify_default_op_cache_key(Det)
def numba_funcify_Det(op, node, **kwargs):
    out_dtype = node.outputs[0].type.numpy_dtype
    discrete_input = node.inputs[0].type.numpy_dtype.kind in "ibu"
    if discrete_input and config.compiler_verbose:
        print("Det requires casting discrete input to float")  # noqa: T201

    @numba_basic.numba_njit
    def det(x):
        if discrete_input:
            x = x.astype(out_dtype)
        return np.array(np.linalg.det(x), dtype=out_dtype)

    cache_version = 1
    return det, cache_version


@register_funcify_default_op_cache_key(SLogDet)
def numba_funcify_SLogDet(op, node, **kwargs):
    out_dtype_sign = node.outputs[0].type.numpy_dtype
    out_dtype_det = node.outputs[1].type.numpy_dtype

    discrete_input = node.inputs[0].type.numpy_dtype.kind in "ibu"
    if discrete_input and config.compiler_verbose:
        print("SLogDet requires casting discrete input to float")  # noqa: T201

    @numba_basic.numba_njit
    def slogdet(x):
        if discrete_input:
            x = x.astype(out_dtype_det)
        sign, det = np.linalg.slogdet(x)
        return (
            np.array(sign, dtype=out_dtype_sign),
            np.array(det, dtype=out_dtype_det),
        )

    cache_version = 1
    return slogdet, cache_version
