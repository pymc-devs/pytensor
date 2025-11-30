import warnings

import numba
import numpy as np

import pytensor.link.numba.dispatch.basic as numba_basic
from pytensor import config
from pytensor.link.numba.dispatch.basic import (
    get_numba_type,
    register_funcify_default_op_cache_key,
)
from pytensor.tensor.nlinalg import (
    SVD,
    Det,
    Eig,
    Eigh,
    MatrixInverse,
    MatrixPinv,
    SLogDet,
)


@register_funcify_default_op_cache_key(SVD)
def numba_funcify_SVD(op, node, **kwargs):
    full_matrices = op.full_matrices
    compute_uv = op.compute_uv
    out_dtype = np.dtype(node.outputs[0].dtype)

    discrete_input = node.inputs[0].type.numpy_dtype.kind in "ibu"
    if discrete_input and config.compiler_verbose:
        print("SVD requires casting discrete input to float")  # noqa: T201

    if not compute_uv:

        @numba_basic.numba_njit
        def svd(x):
            if discrete_input:
                x = x.astype(out_dtype)
            _, ret, _ = np.linalg.svd(x, full_matrices)
            return ret

    else:

        @numba_basic.numba_njit
        def svd(x):
            if discrete_input:
                x = x.astype(out_dtype)
            return np.linalg.svd(x, full_matrices)

    cache_version = 1
    return svd, cache_version


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


@register_funcify_default_op_cache_key(Eig)
def numba_funcify_Eig(op, node, **kwargs):
    w_dtype = node.outputs[0].type.numpy_dtype
    non_complex_input = node.inputs[0].type.numpy_dtype.kind != "c"
    if non_complex_input and config.compiler_verbose:
        print("Eig requires casting input to complex")  # noqa: T201

    @numba_basic.numba_njit
    def eig(x):
        if non_complex_input:
            # Even floats are better cast to complex, otherwise numba may raise
            # ValueError: eig() argument must not cause a domain change.
            x = x.astype(w_dtype)
        w, v = np.linalg.eig(x)
        return w.astype(w_dtype), v.astype(w_dtype)

    cache_version = 2
    return eig, cache_version


@register_funcify_default_op_cache_key(Eigh)
def numba_funcify_Eigh(op, node, **kwargs):
    uplo = op.UPLO

    if uplo != "L":
        warnings.warn(
            (
                "Numba will use object mode to allow the "
                "`UPLO` argument to `numpy.linalg.eigh`."
            ),
            UserWarning,
        )

        out_dtypes = tuple(o.type.numpy_dtype for o in node.outputs)
        ret_sig = numba.types.Tuple(
            [get_numba_type(node.outputs[0].type), get_numba_type(node.outputs[1].type)]
        )

        @numba_basic.numba_njit
        def eigh(x):
            with numba.objmode(ret=ret_sig):
                out = np.linalg.eigh(x, UPLO=uplo)
                ret = (out[0].astype(out_dtypes[0]), out[1].astype(out_dtypes[1]))
            return ret

    else:

        @numba_basic.numba_njit
        def eigh(x):
            return np.linalg.eigh(x)

    return eigh


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
