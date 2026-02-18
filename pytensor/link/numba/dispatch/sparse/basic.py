import numpy as np
import scipy as sp
from numba import literal_unroll
from numba.extending import overload

from pytensor import config
from pytensor.link.numba.dispatch import basic as numba_basic
from pytensor.link.numba.dispatch.basic import (
    generate_fallback_impl,
    register_funcify_default_op_cache_key,
)
from pytensor.link.numba.dispatch.compile_ops import numba_deepcopy
from pytensor.link.numba.dispatch.sparse.variable import CSMatrixType
from pytensor.sparse import (
    CSM,
    Cast,
    ColScaleCSC,
    CSMProperties,
    DenseFromSparse,
    HStack,
    RowScaleCSC,
    SparseFromDense,
    Transpose,
    VStack,
)


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


@register_funcify_default_op_cache_key(SparseFromDense)
def numba_funcify_SparseFromDense(op, node, **kwargs):
    if op.format == "csr":

        @numba_basic.numba_njit
        def dense_to_csr(matrix):
            return sp.sparse.csr_matrix(matrix)

        return dense_to_csr
    else:

        @numba_basic.numba_njit
        def dense_to_csc(matrix):
            return sp.sparse.csc_matrix(matrix)

        return dense_to_csc


@register_funcify_default_op_cache_key(HStack)
def numba_funcify_HStack(op, node, **kwargs):
    output_format = op.format
    out_dtype = np.dtype(op.dtype)

    @numba_basic.numba_njit
    def hstack_csc(*blocks):
        n_rows = blocks[0].shape[0]
        total_n_cols = 0
        total_nnz = 0

        blocks_csc = []
        for block in literal_unroll(blocks):
            if block.shape[0] != n_rows:
                raise ValueError("Mismatching dimensions along axis 0")

            # `hstack` operates on CSC inputs, so we convert each block to CSC.
            # This allocates memory for CSR inputs, but not for inputs already in CSC format.
            block_csc = block.tocsc()
            blocks_csc.append(block_csc)

            # Count number of columns and non-zeros for the output matrix.
            total_nnz += block_csc.indptr[block_csc.shape[1]]
            total_n_cols += block_csc.shape[1]

        data = np.empty(total_nnz, dtype=out_dtype)
        indices = np.empty(total_nnz, dtype=np.int32)
        indptr = np.empty(total_n_cols + 1, dtype=np.int32)
        indptr[0] = 0

        # Append each CSC block into the preallocated output by
        # tracking global offsets for columns (`col_offset`) and nonzeros (`nnz_offset`).
        col_offset = 0
        nnz_offset = 0
        for block in blocks_csc:
            block_n_cols = block.shape[1]
            block_nnz = block.indptr[block_n_cols]

            data[nnz_offset : nnz_offset + block_nnz] = block.data
            indices[nnz_offset : nnz_offset + block_nnz] = block.indices

            for col_idx in range(block_n_cols):
                indptr[col_offset + col_idx + 1] = (
                    nnz_offset + block.indptr[col_idx + 1]
                )

            nnz_offset += block_nnz
            col_offset += block_n_cols

        return sp.sparse.csc_matrix(
            (data, indices, indptr), shape=(n_rows, total_n_cols)
        )

    if output_format == "csc":
        return hstack_csc

    @numba_basic.numba_njit
    def hstack_csr(*blocks):
        return hstack_csc(*blocks).tocsr()

    return hstack_csr


@register_funcify_default_op_cache_key(VStack)
def numba_funcify_VStack(op, node, **kwargs):
    output_format = op.format
    out_dtype = np.dtype(op.dtype)

    @numba_basic.numba_njit
    def vstack_csr(*blocks):
        n_cols = blocks[0].shape[1]
        total_n_rows = 0
        total_nnz = 0

        blocks_csr = []
        for block in literal_unroll(blocks):
            if block.shape[1] != n_cols:
                raise ValueError("Mismatching dimensions along axis 1")

            # `vstack` operates on CSR inputs, so we convert each block to CSR.
            # This allocates memory for CSC inputs, but not for inputs already in CSR format.
            block_csr = block.tocsr()
            blocks_csr.append(block_csr)

            # Count number of rows and non-zeros for the output matrix.
            total_nnz += block_csr.indptr[block_csr.shape[0]]
            total_n_rows += block_csr.shape[0]

        data = np.empty(total_nnz, dtype=out_dtype)
        indices = np.empty(total_nnz, dtype=np.int32)
        indptr = np.empty(total_n_rows + 1, dtype=np.int32)
        indptr[0] = 0

        # Append each CSR block into the preallocated output by
        # tracking global offsets for rows (`row_offset`) and nonzeros (`nnz_offset`).
        row_offset = 0
        nnz_offset = 0
        for block in blocks_csr:
            block_n_rows = block.shape[0]
            block_nnz = block.indptr[block_n_rows]

            data[nnz_offset : nnz_offset + block_nnz] = block.data
            indices[nnz_offset : nnz_offset + block_nnz] = block.indices

            for row_idx in range(block_n_rows):
                indptr[row_offset + row_idx + 1] = (
                    nnz_offset + block.indptr[row_idx + 1]
                )

            nnz_offset += block_nnz
            row_offset += block_n_rows

        return sp.sparse.csr_matrix(
            (data, indices, indptr), shape=(total_n_rows, n_cols)
        )

    if output_format == "csr":
        return vstack_csr

    @numba_basic.numba_njit
    def vstack_csc(*blocks):
        return vstack_csr(*blocks).tocsc()

    return vstack_csc


@register_funcify_default_op_cache_key(ColScaleCSC)
def numba_funcify_ColScaleCSC(op, node, **kwargs):
    @numba_basic.numba_njit
    def col_scale_csc(x, v):
        n_cols = x.shape[1]
        assert v.shape == (n_cols,)

        z = x.copy()
        z_data = z.data
        z_indptr = z.indptr.view(np.uint32)

        for col_idx in range(n_cols):
            scale = v[col_idx]
            # Could use slicing, but numba is usually faster with explicit loops.
            for idx in range(z_indptr[col_idx], z_indptr[col_idx + 1]):
                z_data[idx] *= scale
        return z

    return col_scale_csc


@register_funcify_default_op_cache_key(RowScaleCSC)
def numba_funcify_RowScaleCSC(op, node, **kwargs):
    @numba_basic.numba_njit
    def row_scale_csc(x, v):
        n_rows, n_cols = x.shape
        assert v.shape == (n_rows,)

        indices = x.indices.view(np.uint32)
        indptr = x.indptr.view(np.uint32)
        z_data = x.data.copy()

        for col_idx in range(n_cols):
            for idx in range(indptr[col_idx], indptr[col_idx + 1]):
                z_data[idx] *= v[indices[idx]]

        return sp.sparse.csc_matrix(
            (z_data, x.indices.copy(), x.indptr.copy()), shape=x.shape
        )

    return row_scale_csc
