from hashlib import sha256

import numpy as np
import scipy.sparse as sp

import pytensor.sparse.basic as psb
from pytensor.link.numba.dispatch import basic as numba_basic
from pytensor.link.numba.dispatch.basic import (
    register_funcify_and_cache_key,
    register_funcify_default_op_cache_key,
)
from pytensor.sparse import (
    Dot,
    SparseDenseMultiply,
    SparseDenseVectorMultiply,
    StructuredDot,
)


@register_funcify_default_op_cache_key(SparseDenseMultiply)
@register_funcify_default_op_cache_key(SparseDenseVectorMultiply)
def numba_funcify_SparseDenseMultiply(op, node, **kwargs):
    x, y = node.inputs
    [z] = node.outputs
    out_dtype = z.type.dtype
    format = z.type.format
    same_dtype = x.type.dtype == out_dtype

    if y.ndim == 0:

        @numba_basic.numba_njit
        def sparse_multiply_scalar(x, y):
            if same_dtype:
                z = x.copy()
            else:
                z = x.astype(out_dtype)
            # Numba doesn't know how to handle in-place mutation / assignment of fields
            # z.data *= y
            z_data = z.data
            z_data *= y
            return z

        return sparse_multiply_scalar

    elif y.ndim == 1:

        @numba_basic.numba_njit
        def sparse_dense_multiply(x, y):
            assert x.shape[1] == y.shape[0]
            if same_dtype:
                z = x.copy()
            else:
                z = x.astype(out_dtype)

            M, N = x.shape
            indices = x.indices
            indptr = x.indptr
            z_data = z.data
            if format == "csc":
                for j in range(0, N):
                    for i_idx in range(indptr[j], indptr[j + 1]):
                        z_data[i_idx] *= y[j]
                return z

            else:
                for i in range(0, M):
                    for j_idx in range(indptr[i], indptr[i + 1]):
                        j = indices[j_idx]
                        z_data[j_idx] *= y[j]

            return z

        return sparse_dense_multiply

    else:  # y.ndim == 2

        @numba_basic.numba_njit
        def sparse_dense_multiply(x, y):
            assert x.shape == y.shape
            if same_dtype:
                z = x.copy()
            else:
                z = x.astype(out_dtype)

            M, N = x.shape
            indices = x.indices
            indptr = x.indptr
            z_data = z.data
            if format == "csc":
                for j in range(0, N):
                    for i_idx in range(indptr[j], indptr[j + 1]):
                        i = indices[i_idx]
                        z_data[i_idx] *= y[i, j]
                return z

            else:
                for i in range(0, M):
                    for j_idx in range(indptr[i], indptr[i + 1]):
                        j = indices[j_idx]
                        z_data[j_idx] *= y[i, j]

            return z

        return sparse_dense_multiply


@register_funcify_and_cache_key(Dot)
@register_funcify_and_cache_key(StructuredDot)
def numba_funcify_SparseDot(op, node, **kwargs):
    # Inputs can be of types: (sparse, dense), (dense, sparse), (sparse, sparse).
    # Dot always returns a dense result.
    # StructuredDot returns a sparse object when all entries are sparse, otherwise dense.
    x, y = node.inputs
    [z] = node.outputs
    out_dtype = z.type.dtype

    x_is_sparse = psb._is_sparse_variable(x)
    y_is_sparse = psb._is_sparse_variable(y)
    z_is_sparse = psb._is_sparse_variable(z)

    x_format = x.type.format if x_is_sparse else None
    y_format = y.type.format if y_is_sparse else None

    cache_key = sha256(
        str(
            (
                type(op),
                x_format,
                y_format,
                z_is_sparse,
                y.type.ndim,
                y.type.broadcastable,
            )
        ).encode()
    ).hexdigest()

    if x_is_sparse and y_is_sparse:
        # General spmspm algorithm in CSR format
        @numba_basic.numba_njit
        def _spmspm(n_row, n_col, x_ptr, x_ind, x_data, y_ptr, y_ind, y_data):
            # Pass 1
            x_ind = x_ind.view(np.uint32)
            y_ind = y_ind.view(np.uint32)
            x_ptr = x_ptr.view(np.uint32)
            y_ptr = y_ptr.view(np.uint32)

            output_nnz = 0
            mask = np.full(n_col, -1, dtype=np.int32)
            for i in range(n_row):
                row_nnz = 0
                for jj in range(x_ptr[i], x_ptr[i + 1]):
                    j = x_ind[jj]
                    for kk in range(y_ptr[j], y_ptr[j + 1]):
                        k = y_ind[kk]
                        if mask[k] != i:
                            mask[k] = i
                            row_nnz += 1
                output_nnz += row_nnz

            # Pass 2
            z_ptr = np.empty(n_row + 1, dtype=np.uint32)
            z_ind = np.empty(output_nnz, dtype=np.uint32)
            z_data = np.empty(output_nnz, dtype=out_dtype)

            # Refill original mask for reuse
            mask.fill(-1)
            sums = np.zeros(n_col, dtype=out_dtype)

            nnz = 0
            z_ptr[0] = 0

            for i in range(n_row):
                head = -2
                length = 0

                for jj in range(x_ptr[i], x_ptr[i + 1]):
                    j = x_ind[jj]
                    v = x_data[jj]

                    for kk in range(y_ptr[j], y_ptr[j + 1]):
                        k = y_ind[kk]
                        sums[k] += v * y_data[kk]

                        if mask[k] == -1:
                            mask[k] = head
                            head = k
                            length += 1

                for _ in range(length):
                    if sums[head] != 0:
                        z_ind[nnz] = head
                        z_data[nnz] = sums[head]
                        nnz += 1

                    temp = head
                    head = mask[head]
                    mask[temp] = -1
                    sums[temp] = 0

                z_ptr[i + 1] = nnz

            return z_ptr.view(np.int32), z_ind.view(np.int32), z_data

        @numba_basic.numba_njit
        def spmspm(x, y):
            if x_format != "csr":
                x = x.tocsr()

            if y_format != "csr":
                y = y.tocsr()

            x_ptr, x_ind, x_data = x.indptr, x.indices, x.data
            y_ptr, y_ind, y_data = y.indptr, y.indices, y.data
            n_row, n_col = x.shape[0], y.shape[1]

            z_ptr, z_ind, z_data = _spmspm(
                n_row, n_col, x_ptr, x_ind, x_data, y_ptr, y_ind, y_data
            )

            output = sp.csr_matrix((z_data, z_ind, z_ptr), shape=(n_row, n_col))

            # Dot returns a dense result even in spMspM
            if not z_is_sparse:
                return output.toarray()

            # StructuredDot returns in the format of 'x'
            if x_format == "csc":
                return output.tocsc()

            return output

        return spmspm, cache_key

    # Only one of 'x' or 'y' is sparse, not both.
    # Before using a general dot(sparse-matrix, dense-matrix) algorithm,
    # we check if we can rely on the less intensive (sparse-matrix, dense-vector) algorithm (spmv).
    y_is_1d_like = y.type.ndim == 1 or (y.type.ndim == 2 and y.type.shape[1] == 1)
    x_is_1d = x.type.ndim == 1

    if (x_is_sparse and y_is_1d_like) or (y_is_sparse and x_is_1d):
        # We can use spmv
        @numba_basic.numba_njit
        def _spmdv_csr(x_ptr, x_ind, x_data, x_shape, y):
            n_row = x_shape[0]
            x_ptr = x_ptr.view(np.uint32)
            x_ind = x_ind.view(np.uint32)
            output = np.zeros(n_row, dtype=out_dtype)

            for row_idx in range(n_row):
                acc = 0.0
                for k in range(x_ptr[row_idx], x_ptr[row_idx + 1]):
                    acc += x_data[k] * y[x_ind[k]]
                output[row_idx] = acc

            return output

        @numba_basic.numba_njit
        def _spmdv_csc(x_ptr, x_ind, x_data, x_shape, y):
            n_row, n_col = x_shape
            x_ptr = x_ptr.view(np.uint32)
            x_ind = x_ind.view(np.uint32)
            output = np.zeros(n_row, dtype=out_dtype)

            for col_idx in range(n_col):
                yj = y[col_idx]
                for k in range(x_ptr[col_idx], x_ptr[col_idx + 1]):
                    output[x_ind[k]] += x_data[k] * yj

            return output

        if x_is_sparse:
            if x_format == "csr":
                _spmdv = _spmdv_csr
            else:
                _spmdv = _spmdv_csc

            if y.type.ndim == 1:

                @numba_basic.numba_njit
                def spmdv(x, y):
                    assert x.shape[1] == y.shape[0]
                    return _spmdv(x.indptr, x.indices, x.data, x.shape, y)
            else:

                @numba_basic.numba_njit
                def spmdv(x, y):
                    # Output must be 2d.
                    assert x.shape[1] == y.shape[0]
                    return _spmdv(x.indptr, x.indices, x.data, x.shape, y[:, 0])[
                        :, None
                    ]

            return spmdv, cache_key
        else:  # y_is_sparse
            # Rely on: z = dot(x, y) -> z^T = dot(x, y)^T -> z^T = dot(y^T, x^T)
            if y_format == "csr":
                _spmdv = _spmdv_csc
            else:  # csc
                _spmdv = _spmdv_csr

            @numba_basic.numba_njit
            def spmdv(x, y):
                # SciPy treats (p, ) * (p, k) as (1, p) @ (p, k),
                # but returns the result as of shape (k, ).
                assert x.shape[0] == y.shape[0]
                yT = y.T  # (k, p)
                return _spmdv(yT.indptr, yT.indices, yT.data, yT.shape, x)

            return spmdv, cache_key

    # Only one of 'x' or 'y' is sparse, and we can't use spmdv.
    # We know we have to rely on the general (sparse-matrix, dense-matrix) dot product (spmdm).
    @numba_basic.numba_njit
    def spmdm_csr(x, y):
        assert x.shape[1] == y.shape[0]
        n = x.shape[0]
        k = y.shape[1]
        z = np.zeros((n, k), dtype=out_dtype)

        x_ind = x.indices.view(np.uint32)
        x_ptr = x.indptr.view(np.uint32)
        x_data = x.data

        for row_idx in range(n):
            for idx in range(x_ptr[row_idx], x_ptr[row_idx + 1]):
                col_idx = x_ind[idx]
                value = x_data[idx]
                z[row_idx] += value * y[col_idx]
        return z

    @numba_basic.numba_njit
    def spmdm_csc(x, y):
        assert x.shape[1] == y.shape[0]
        k = y.shape[1]
        n = x.shape[0]
        p = x.shape[1]
        z = np.zeros((n, k), dtype=out_dtype)

        x_ind = x.indices.view(np.uint32)
        x_ptr = x.indptr.view(np.uint32)
        x_data = x.data

        for col_idx in range(p):
            for idx in range(x_ptr[col_idx], x_ptr[col_idx + 1]):
                row_idx = x_ind[idx]
                value = x_data[idx]
                z[row_idx] += value * y[col_idx]
        return z

    if x_is_sparse:
        if x_format == "csr":
            return spmdm_csr, cache_key
        else:
            return spmdm_csc, cache_key

    if y_is_sparse:
        # We don't implement a dense-sparse dot product.
        # Instead, we use properties of transpose:
        #     z = dot(x, y) -> z^T = dot(x, y)^T -> z^T = dot(y^T, x^T)
        # which allows us to reuse sparse-dense dot.
        if y_format == "csr":
            # y.T will be CSC
            @numba_basic.numba_njit
            def dmspm(x, y):
                return spmdm_csc(y.T, x.T).T
        else:
            # y.T will be CSR
            @numba_basic.numba_njit
            def dmspm(x, y):
                return spmdm_csr(y.T, x.T).T

        return dmspm, cache_key
