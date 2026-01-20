import numpy as np
import scipy.sparse as sp

import pytensor.sparse.basic as psb
from pytensor.link.numba.dispatch import basic as numba_basic
from pytensor.link.numba.dispatch.basic import register_funcify_default_op_cache_key
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


@register_funcify_default_op_cache_key(Dot)
@register_funcify_default_op_cache_key(StructuredDot)
def numba_funcify_SparseDot(op, node, **kwargs):
    # (n, p) @ (p, k) -> (n, k)
    # Inputs can be of types: (sparse, dense), (dense, sparse), (sparse, sparse)
    # Output is sparse when all entries are sparse, otherwise dense.

    x, y = node.inputs
    [z] = node.outputs
    out_dtype = z.type.dtype

    x_is_sparse = psb._is_sparse_variable(x)
    y_is_sparse = psb._is_sparse_variable(y)

    x_format = x.type.format if x_is_sparse else None
    y_format = y.type.format if y_is_sparse else None

    if x_is_sparse and y_is_sparse:
        # General spmspm algorithm in CSR format
        @numba_basic.numba_njit
        def _spmspm(n_row, n_col, x_ptr, x_ind, x_data, y_ptr, y_ind, y_data):
            # Pass 1
            x_ind = x_ind.view(np.uint32)
            y_ind = y_ind.view(np.uint32)
            x_ptr = x_ptr.view(np.uint32)
            y_ptr = y_ptr.view(np.uint32)

            nnz = 0
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
                nnz += row_nnz

            # Pass 2
            z_ptr = np.empty(n_row + 1, dtype=np.int32)  # NOTE: consider int64?
            z_ind = np.empty(nnz, dtype=np.int32)
            z_data = np.empty(nnz, dtype=out_dtype)

            mask2 = np.full(n_col, -1, dtype=np.int32)
            sums = np.zeros(n_col, dtype=x_data.dtype)

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

                        if mask2[k] == -1:
                            mask2[k] = head
                            head = k
                            length += 1

                for _ in range(length):
                    if sums[head] != 0:
                        z_ind[nnz] = head
                        z_data[nnz] = sums[head]
                        nnz += 1

                    temp = head
                    head = mask2[head]
                    mask2[temp] = -1
                    sums[temp] = 0

                z_ptr[i + 1] = nnz

            return z_ptr, z_ind, z_data

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
            return sp.csr_matrix((z_data, z_ind, z_ptr), shape=(n_row, n_col))

        return spmspm

    if x_is_sparse and y.type.shape[1] == 1:  # NOTE: y.ndim is always 2 within dot

        @numba_basic.numba_njit
        def _spmdv(x_ptr, x_ind, x_data, y):
            if x_format == "csr":
                n_rows = x_ptr.shape[0] - 1
                output = np.zeros(n_rows, dtype=out_dtype)

                for i in range(n_rows):
                    start = x_ptr[i]
                    end = x_ptr[i + 1]
                    acc = 0.0
                    for k in range(start, end):
                        acc += x_data[k] * y[x_ind[k]]
                    output[i] = acc
                return output

            else:  # "csc"
                n_rows = np.max(x_ind) + 1
                output = np.zeros(n_rows, dtype=out_dtype)
                n_cols = x_ptr.shape[0] - 1

                for j in range(n_cols):
                    start = x_ptr[j]
                    end = x_ptr[j + 1]
                    yj = y[j]
                    for k in range(start, end):
                        output[x_ind[k]] += x_data[k] * yj

                return output

        def spmdv(x, y):
            # Output _has to be_ 2d.
            return _spmdv(x.indptr, x.indices, x.data, np.squeeze(y))[:, np.newaxis]

        return spmdv

    @numba_basic.numba_njit
    def spmdm_csr(x, y):
        assert x.shape[1] == y.shape[0]
        n = x.shape[0]
        k = y.shape[1]
        z = np.zeros((n, k), dtype=out_dtype)

        x_ind = x.indices
        x_ptr = x.indptr
        x_data = x.data

        for row_idx in range(n):
            row_start = x_ptr[row_idx]
            row_end = x_ptr[row_idx + 1]
            for idx in range(row_start, row_end):
                col_idx = x_ind[idx]
                value = x_data[idx]
                z[row_idx, :] += value * y[col_idx, :]

        return z

    @numba_basic.numba_njit
    def spmdm_csc(x, y):
        assert x.shape[1] == y.shape[0]
        n = x.shape[0]
        k = y.shape[1]
        z = np.zeros((n, k), dtype=out_dtype)

        x_ind = x.indices
        x_ptr = x.indptr
        x_data = x.data

        p = x.shape[1]
        for col_idx in range(p):
            col_start = x_ptr[col_idx]
            col_end = x_ptr[col_idx + 1]

            for idx in range(col_start, col_end):
                row_idx = x_ind[idx]
                value = x_data[idx]
                z[row_idx, :] += value * y[col_idx, :]

        return z

    if x_is_sparse:
        if x_format == "csr":
            return spmdm_csr
        else:
            return spmdm_csc

    if y_is_sparse:
        # We don't implement a dense-sparse dot product.
        # Instead, we use properties of transpose:
        #     z = dot(x, y) -> z^T = dot(x, y)^T -> z^T = dot(y^T, x^T)
        # which allows us to reuse sparse-dense dot.
        if y_format == "csr":
            # y.T will be CSC
            @numba_basic.numba_njit
            def dmspm_1(x, y):
                return spmdm_csc(y.T, x.T).T

            return dmspm_1
        else:
            # y.T will be CSR
            @numba_basic.numba_njit
            def dmspm_2(x, y):
                return spmdm_csr(y.T, x.T).T

            return dmspm_2
