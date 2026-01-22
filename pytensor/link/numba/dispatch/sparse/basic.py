import numpy as np
import scipy as sp
from numba.extending import overload

from pytensor import config
from pytensor.link.numba.dispatch import basic as numba_basic
from pytensor.link.numba.dispatch.basic import (
    generate_fallback_impl,
    register_funcify_default_op_cache_key,
)
from pytensor.link.numba.dispatch.compile_ops import numba_deepcopy
from pytensor.link.numba.dispatch.sparse.variable import CSMatrixType
from pytensor.sparse import CSM, Cast, CSMProperties, DenseFromSparse, Transpose


@overload(numba_deepcopy)
def numba_deepcopy_sparse(x):
    if isinstance(x, CSMatrixType):

        def sparse_deepcopy(x):
            return x.copy()

        return sparse_deepcopy


@register_funcify_default_op_cache_key(CSMProperties)
def numba_funcify_CSMProperties(op, node, **kwargs):
    @numba_basic.numba_njit
    def csm_properties(x):
        # Reconsider this int32/int64. Scipy/base PyTensor use int32 for indices/indptr.
        # But this seems to be legacy mistake and devs would choose int64 nowadays, and may move there.
        return x.data, x.indices, x.indptr, np.asarray(x.shape, dtype="int32")

    return csm_properties


@register_funcify_default_op_cache_key(CSM)
def numba_funcify_CSM(op, node, **kwargs):
    format = op.format

    @numba_basic.numba_njit
    def csm_constructor(data, indices, indptr, shape):
        constructor_arg = (data, indices, indptr)
        shape_arg = (shape[0], shape[1])
        if format == "csr":
            return sp.sparse.csr_matrix(constructor_arg, shape=shape_arg)
        else:
            return sp.sparse.csc_matrix(constructor_arg, shape=shape_arg)

    return csm_constructor


@register_funcify_default_op_cache_key(Cast)
def numba_funcify_Cast(op, node, **kwargs):
    inp_dtype = node.inputs[0].type.dtype
    out_dtype = np.dtype(op.out_type)
    if not np.can_cast(inp_dtype, out_dtype):
        if config.compiler_verbose:
            print(  # noqa: T201
                f"Sparse Cast fallback to obj mode due to unsafe casting from {inp_dtype} to {out_dtype}"
            )
        return generate_fallback_impl(op, node, **kwargs)

    @numba_basic.numba_njit
    def cast(x):
        return x.astype(out_dtype)

    return cast


@register_funcify_default_op_cache_key(Transpose)
def numba_funcify_Transpose(op, node, **kwargs):
    @numba_basic.numba_njit
    def transpose(x):
        return x.T

    return transpose


@register_funcify_default_op_cache_key(DenseFromSparse)
def numba_funcify_DenseFromSparse(op, node, **kwargs):
    @numba_basic.numba_njit
    def to_array(x):
        return x.toarray()

    return to_array
