from hashlib import sha256

import numpy as np
import scipy.sparse as sp

import pytensor.sparse.basic as psb
from pytensor import config
from pytensor.link.numba.dispatch import basic as numba_basic
from pytensor.link.numba.dispatch.basic import (
    register_funcify_and_cache_key,
    register_funcify_default_op_cache_key,
)
from pytensor.link.numba.dispatch.sparse.basic import make_sum_duplicates
from pytensor.sparse import (
    AddSparseDense,
    AddSparseSparse,
    AddSparseSparseData,
    Dot,
    MultiplySparseDense,
    MultiplySparseDenseVector,
    MultiplySparseSparse,
    SpSum,
    StructuredAddSparseDenseVector,
    StructuredDot,
    StructuredDotGradCSC,
    StructuredDotGradCSR,
    Usmm,
)
from pytensor.sparse.rewriting import UsmmCscDense


@register_funcify_default_op_cache_key(SpSum)
def numba_funcify_SpSum(op, node, **kwargs):
    axis = op.axis
    cache_version = 2

    @numba_basic.numba_njit
    def perform(x):
        return x.sum(axis)

    return perform, cache_version


@register_funcify_default_op_cache_key(MultiplySparseDense)
@register_funcify_default_op_cache_key(MultiplySparseDenseVector)
def numba_funcify_MultiplySparseDense(op, node, **kwargs):
    x, y = node.inputs
    [z] = node.outputs
    out_dtype = z.type.dtype
    out_format = z.type.format
    same_dtype = x.type.dtype == out_dtype

    cache_version = 2
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

        return sparse_multiply_scalar, cache_version

    elif y.ndim == 1:

        @numba_basic.numba_njit
        def sparse_dense_multiply(x, y):
            assert x.shape[1] == y.shape[0]
            if same_dtype:
                z = x.copy()
            else:
                z = x.astype(out_dtype)

            M, N = x.shape
            indices = x.indices.view(np.uint32)
            indptr = x.indptr.view(np.uint32)
            z_data = z.data
            if out_format == "csc":
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

        return sparse_dense_multiply, cache_version

    else:  # y.ndim == 2

        @numba_basic.numba_njit
        def sparse_dense_multiply(x, y):
            assert x.shape == y.shape
            if same_dtype:
                z = x.copy()
            else:
                z = x.astype(out_dtype)

            M, N = x.shape
            indices = x.indices.view(np.uint32)
            indptr = x.indptr.view(np.uint32)
            z_data = z.data
            if out_format == "csc":
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

        return sparse_dense_multiply, cache_version


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

    out_type = np.dtype(out_dtype).type

    cache_version = 4
    cache_key = sha256(
        str(
            (
                type(op),
                x_format,
                y_format,
                z_is_sparse,
                y.type.ndim,
                y.type.broadcastable,
                cache_version,
            )
        ).encode()
    ).hexdigest()

    if x_is_sparse and y_is_sparse:
        # General spmspm algorithm in CSR format
        @numba_basic.numba_njit
        def _spmspm_csr(x, y, n_row, n_col):
            # Pass 1
            x_ind = x.indices.view(np.uint32)
            y_ind = y.indices.view(np.uint32)
            x_ptr = x.indptr.view(np.uint32)
            y_ptr = y.indptr.view(np.uint32)
            x_data = x.data
            y_data = y.data

            # Sparse indices are int32, so the top two uint32 values are free sentinels.
            not_visited = np.uint32(2**32 - 1)
            end_of_list = np.uint32(2**32 - 2)

            # Unsigned increments prevent signed promotion in index arithmetic.
            output_nnz = np.uint64(0)

            # Reuse the next-index buffer as row markers during the counting pass.
            next_index = np.full(n_col, not_visited, dtype=np.uint32)
            for i in range(n_row):
                row_nnz = np.uint64(0)
                for jj in range(x_ptr[i], x_ptr[i + 1]):
                    j = x_ind[jj]
                    for kk in range(y_ptr[j], y_ptr[j + 1]):
                        k = y_ind[kk]
                        if next_index[k] != i:
                            next_index[k] = i
                            row_nnz += np.uint64(1)
                output_nnz += row_nnz

            # Pass 2
            z_ptr = np.empty(n_row + 1, dtype=np.uint32)
            z_ind = np.empty(output_nnz, dtype=np.uint32)
            z_data = np.empty(output_nnz, dtype=out_dtype)

            next_index.fill(not_visited)
            sums = np.zeros(n_col, dtype=out_dtype)

            nnz = np.uint64(0)
            z_ptr[0] = 0

            for i in range(n_row):
                head = end_of_list
                length = np.uint64(0)

                for jj in range(x_ptr[i], x_ptr[i + 1]):
                    j = x_ind[jj]
                    v = out_type(x_data[jj])

                    for kk in range(y_ptr[j], y_ptr[j + 1]):
                        k = y_ind[kk]
                        sums[k] += v * out_type(y_data[kk])

                        if next_index[k] == not_visited:
                            next_index[k] = head
                            head = k
                            length += np.uint64(1)

                for _ in range(length):
                    if sums[head] != 0:
                        z_ind[nnz] = head
                        z_data[nnz] = sums[head]
                        nnz += np.uint64(1)

                    next_head = next_index[head]
                    next_index[head] = not_visited
                    sums[head] = 0
                    head = next_head

                z_ptr[i + 1] = nnz

            return z_ptr.view(np.int32), z_ind.view(np.int32), z_data

        formats = (x_format, y_format)
        if formats == ("csc", "csc"):
            # In all cases, the output is dense when the op is Dot.
            @numba_basic.numba_njit
            def spmspm_csc_csc(x, y):
                # Swap inputs
                n_row, n_col = x.shape[0], y.shape[1]
                z_ptr, z_ind, z_data = _spmspm_csr(x=y, y=x, n_row=n_col, n_col=n_row)
                output = sp.csc_matrix((z_data, z_ind, z_ptr), shape=(n_row, n_col))
                if not z_is_sparse:
                    return output.toarray()
                return output

            return spmspm_csc_csc, cache_key
        elif formats == ("csc", "csr"):

            @numba_basic.numba_njit
            def spmspm_csc_csr(x, y):
                # Convert csr to csc and swap
                n_row, n_col = x.shape[0], y.shape[1]
                z_ptr, z_ind, z_data = _spmspm_csr(
                    x=y.tocsc(), y=x, n_row=n_col, n_col=n_row
                )
                output = sp.csc_matrix((z_data, z_ind, z_ptr), shape=(n_row, n_col))
                if not z_is_sparse:
                    return output.toarray()
                return output

            return spmspm_csc_csr, cache_key
        elif formats == ("csr", "csc"):

            @numba_basic.numba_njit
            def spmspm_csr_csc(x, y):
                # Convert csc to csr, no swap
                n_row, n_col = x.shape[0], y.shape[1]
                z_ptr, z_ind, z_data = _spmspm_csr(
                    x=x, y=y.tocsr(), n_row=n_row, n_col=n_col
                )
                output = sp.csr_matrix((z_data, z_ind, z_ptr), shape=(n_row, n_col))
                if not z_is_sparse:
                    return output.toarray()
                return output

            return spmspm_csr_csc, cache_key
        else:

            @numba_basic.numba_njit
            def spmspm_csr_csr(x, y):
                # No conversion, no swap
                n_row, n_col = x.shape[0], y.shape[1]
                z_ptr, z_ind, z_data = _spmspm_csr(x=x, y=y, n_row=n_row, n_col=n_col)
                output = sp.csr_matrix((z_data, z_ind, z_ptr), shape=(n_row, n_col))
                if not z_is_sparse:
                    return output.toarray()
                return output

            return spmspm_csr_csr, cache_key

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
            output = np.empty(n_row, dtype=out_dtype)

            for row_idx in range(n_row):
                acc = out_type(0)
                for k in range(x_ptr[row_idx], x_ptr[row_idx + 1]):
                    acc += out_type(x_data[k]) * out_type(y[x_ind[k]])
                output[row_idx] = acc

            return output

        @numba_basic.numba_njit
        def _spmdv_csc(x_ptr, x_ind, x_data, x_shape, y):
            n_row, n_col = x_shape
            x_ptr = x_ptr.view(np.uint32)
            x_ind = x_ind.view(np.uint32)
            output = np.zeros(n_row, dtype=out_dtype)

            for col_idx in range(n_col):
                yj = out_type(y[col_idx])
                for k in range(x_ptr[col_idx], x_ptr[col_idx + 1]):
                    output[x_ind[k]] += out_type(x_data[k]) * yj

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
            # The transpose of Y is not materialized to avoid unnecessary copies..
            if y_format == "csr":
                _spmdv = _spmdv_csc
            else:  # csc
                _spmdv = _spmdv_csr

            @numba_basic.numba_njit
            def spmdv(x, y):
                # SciPy treats (p, ) * (p, k) as (1, p) @ (p, k),
                # but returns the result as of shape (k, ).
                assert x.shape[0] == y.shape[0]
                return _spmdv(y.indptr, y.indices, y.data, (y.shape[1], y.shape[0]), x)

            return spmdv, cache_key

    # Only one of 'x' or 'y' is sparse, and we can't use spmdv.
    # We know we have to rely on the general (sparse-matrix, dense-matrix) dot product (spmdm).
    @numba_basic.numba_njit
    def _spmdm_csr(x_ptr, x_ind, x_data, x_shape, y):
        n = x_shape[0]
        k = y.shape[1]
        z = np.zeros((n, k), dtype=out_dtype)

        x_ind = x_ind.view(np.uint32)
        x_ptr = x_ptr.view(np.uint32)

        for row_idx in range(n):
            for idx in range(x_ptr[row_idx], x_ptr[row_idx + 1]):
                col_idx = x_ind[idx]
                value = out_type(x_data[idx])
                for j in range(k):
                    z[row_idx, j] += value * y[col_idx, j]
        return z

    @numba_basic.numba_njit
    def _spmdm_csc(x_ptr, x_ind, x_data, x_shape, y):
        k = y.shape[1]
        n, p = x_shape
        z = np.zeros((n, k), dtype=out_dtype)

        x_ind = x_ind.view(np.uint32)
        x_ptr = x_ptr.view(np.uint32)

        for col_idx in range(p):
            for idx in range(x_ptr[col_idx], x_ptr[col_idx + 1]):
                row_idx = x_ind[idx]
                value = out_type(x_data[idx])
                for j in range(k):
                    z[row_idx, j] += value * y[col_idx, j]
        return z

    if x_is_sparse:
        if x_format == "csr":
            _spmdm = _spmdm_csr
        else:
            _spmdm = _spmdm_csc

        @numba_basic.numba_njit
        def spmdm(x, y):
            assert x.shape[1] == y.shape[0]
            return _spmdm(x.indptr, x.indices, x.data, x.shape, y)

        return spmdm, cache_key

    if y_is_sparse:
        # We don't implement a dense-sparse dot product.
        # Instead, we use properties of transpose:
        #     z = dot(x, y) -> z^T = dot(x, y)^T -> z^T = dot(y^T, x^T)
        # which allows us to reuse sparse-dense dot.
        # Note we don't need to actually transpose Y, which would create unnecessary copies.
        if y_format == "csr":
            # y.T will be CSC.
            _spmdm = _spmdm_csc
        else:
            # y.T will be CSR
            _spmdm = _spmdm_csr

        @numba_basic.numba_njit
        def dmspm(x, y):
            assert x.shape[1] == y.shape[0]
            return _spmdm(y.indptr, y.indices, y.data, (y.shape[1], y.shape[0]), x.T).T

        return dmspm, cache_key


@register_funcify_and_cache_key(StructuredDotGradCSR)
@register_funcify_and_cache_key(StructuredDotGradCSC)
def numba_funcify_StructuredDotGrad(op, node, **kwargs):
    """Overload StructuredDotGrad in Numba.

    Let:
      Z = structured_dot(X, Y)
      L = L(Z), a scalar loss depending on Z.

    This function computes the gradient of the loss with respect to X:

      dL/dX = dot(dL/dZ, Y^T)

    where G = dL/dZ is the accumulated (upstream) gradient.

    The returned gradient is structured, preserving the sparsity pattern of X,
    and only the `.data` component of the sparse matrix is computed.
    If Y is sparse, the sparsity pattern of the result is not recomputed.
    The output may contain explicit zeros at positions that would be structural zeros
    if the sparsity structure were updated.

    The core of the algorithm is:

     dot(g_xy[i], y[j])

    where g_xy[i] (row of G) and y[j] (column of Y^T) are vectors of length 'k'

    Reminder:
    x.shape        (n, p)
    y.shape        (p, k)
    g_xy.shape     (n, k)
    """
    _, _, y, g_xy = node.inputs

    y_dtype = y.type.dtype
    y_is_sparse = psb._is_sparse_variable(y)
    y_format = y.type.format if y_is_sparse else None

    g_xy_dtype = g_xy.type.dtype
    g_xy_is_sparse = psb._is_sparse_variable(g_xy)
    g_xy_format = g_xy.type.format if g_xy_is_sparse else None

    x_format = "csc" if isinstance(op, StructuredDotGradCSC) else "csr"
    out_dtype = node.outputs[0].type.dtype
    out_type = np.result_type(y_dtype, g_xy_dtype).type

    cache_version = 2
    cache_key = sha256(
        str(
            (
                type(op),
                x_format,
                y_format,
                y_dtype,
                g_xy_format,
                g_xy_dtype,
                out_dtype,
                y.type.shape,
                cache_version,
            )
        ).encode()
    ).hexdigest()

    if not g_xy_is_sparse:
        # X is sparse, Y and G_xy are dense.
        if x_format == "csr":
            if y.type.shape[1] == 1:
                # If Y is actually 1D, use more performant specialized algorithm
                # Inputs with ndims > 2 will never appear in the StructuredDot Op
                @numba_basic.numba_njit
                def _grad_spmdv_csr(x_indices, x_ptr, y, g_xy):
                    output = np.empty(len(x_indices), dtype=out_dtype)
                    size = len(x_ptr) - 1
                    x_indices = x_indices.view(np.uint32)
                    x_ptr = x_ptr.view(np.uint32)
                    for row_idx in range(size):
                        for value_idx in range(x_ptr[row_idx], x_ptr[row_idx + 1]):
                            output[value_idx] = out_type(g_xy[row_idx]) * out_type(
                                y[x_indices[value_idx]]
                            )
                    return output

                @numba_basic.numba_njit
                def grad_spmdv_csr(x_indices, x_ptr, y, g_xy):
                    return _grad_spmdv_csr(x_indices, x_ptr, y[:, 0], g_xy[:, 0])

                return grad_spmdv_csr, cache_key
            else:
                if config.compiler_verbose and y_dtype != g_xy_dtype:
                    print(  # noqa: T201
                        "Numba StructuredDotGrad uses scalar input casts: "
                        f"{y_dtype=}, {g_xy_dtype=}."
                    )

                @numba_basic.numba_njit
                def grad_spmdm_csr(x_indices, x_ptr, y, g_xy):
                    if y.shape[1] != g_xy.shape[1]:
                        raise ValueError("shapes not aligned")

                    size = len(x_ptr) - 1
                    x_indices = x_indices.view(np.uint32)
                    x_ptr = x_ptr.view(np.uint32)

                    output = np.empty(len(x_indices), dtype=out_dtype)

                    for row_idx in range(size):
                        for value_idx in range(x_ptr[row_idx], x_ptr[row_idx + 1]):
                            col_idx = x_indices[value_idx]
                            acc = out_type(0)
                            for j in range(y.shape[1]):
                                acc += out_type(g_xy[row_idx, j]) * out_type(
                                    y[col_idx, j]
                                )
                            output[value_idx] = acc
                    return output

                return grad_spmdm_csr, cache_key
        else:
            # X is CSC
            @numba_basic.numba_njit
            def grad_spmdm_csc(x_indices, x_ptr, y, g_xy):
                if y.shape[1] != g_xy.shape[1]:
                    raise ValueError("shapes not aligned")
                # len(x_indices) gives the number of non-zero elements in X.
                output = np.empty(len(x_indices), dtype=out_dtype)
                size = len(x_ptr) - 1
                x_indices = x_indices.view(np.uint32)
                x_ptr = x_ptr.view(np.uint32)

                for col_idx in range(size):
                    for value_idx in range(x_ptr[col_idx], x_ptr[col_idx + 1]):
                        row_idx = x_indices[value_idx]
                        acc = out_type(0)
                        for j in range(y.shape[1]):
                            acc += out_type(g_xy[row_idx, j]) * out_type(y[col_idx, j])
                        output[value_idx] = acc
                return output

            return grad_spmdm_csc, cache_key

    # Y is sparse. In either case we need 'dot_csr_rows'
    @numba_basic.numba_njit
    def dot_csr_rows(x_ptr, x_indices, x_data, x_row, y_ptr, y_indices, y_data, y_row):
        x_row = np.uint64(x_row)
        y_row = np.uint64(y_row)
        x_p = np.uint64(x_ptr[x_row])
        x_end = np.uint64(x_ptr[x_row + np.uint64(1)])
        y_p = np.uint64(y_ptr[y_row])
        y_end = np.uint64(y_ptr[y_row + np.uint64(1)])

        acc = out_type(0)
        while x_p < x_end and y_p < y_end:
            x_col = x_indices[x_p]
            y_col = y_indices[y_p]
            if x_col == y_col:
                acc += out_type(x_data[x_p]) * out_type(y_data[y_p])
                x_p += np.uint64(1)
                y_p += np.uint64(1)
            elif x_col < y_col:
                x_p += np.uint64(1)
            else:
                y_p += np.uint64(1)

        return acc

    if x_format == "csr":
        assert g_xy_format == "csr"
        assert psb._is_sparse_variable(y)

        @numba_basic.numba_njit
        def grad_spmspm_csr(x_indices, x_ptr, y, g_xy):
            x_indices = x_indices.view(np.uint32)
            x_ptr = x_ptr.view(np.uint32)
            if y_format == "csc":
                y = y.tocsr()

            g_xy_data = g_xy.data
            g_xy_indices = g_xy.indices.view(np.uint32)
            g_xy_ptr = g_xy.indptr.view(np.uint32)

            y_data = y.data
            y_indices = y.indices.view(np.uint32)
            y_ptr = y.indptr.view(np.uint32)

            n_row = len(x_ptr) - 1
            output = np.empty(len(x_indices), dtype=out_dtype)

            for x_row in range(n_row):
                for data_idx in range(x_ptr[x_row], x_ptr[x_row + 1]):
                    x_col = x_indices[data_idx]
                    output[data_idx] = dot_csr_rows(
                        g_xy_ptr,
                        g_xy_indices,
                        g_xy_data,
                        x_row,
                        y_ptr,
                        y_indices,
                        y_data,
                        x_col,
                    )
            return output

        return grad_spmspm_csr, cache_key
    else:
        assert g_xy_format == "csc"
        assert psb._is_sparse_variable(y)

        @numba_basic.numba_njit
        def grad_spmspm_csc(x_indices, x_ptr, y, g_xy):
            x_indices = x_indices.view(np.uint32)
            x_ptr = x_ptr.view(np.uint32)
            if y_format == "csc":
                y = y.tocsr()

            # Looping a CSC matrix rowwise is too painful, slow, and cryptic.
            g_xy = g_xy.tocsr()

            g_xy_data = g_xy.data
            g_xy_indices = g_xy.indices.view(np.uint32)
            g_xy_ptr = g_xy.indptr.view(np.uint32)

            y_data = y.data
            y_indices = y.indices.view(np.uint32)
            y_ptr = y.indptr.view(np.uint32)

            n_cols = len(x_ptr) - 1
            output = np.empty(len(x_indices), dtype=out_dtype)

            for x_col in range(n_cols):
                for data_idx in range(x_ptr[x_col], x_ptr[x_col + 1]):
                    x_row = x_indices[data_idx]
                    output[data_idx] = dot_csr_rows(
                        g_xy_ptr,
                        g_xy_indices,
                        g_xy_data,
                        x_row,
                        y_ptr,
                        y_indices,
                        y_data,
                        x_col,
                    )
            return output

        return grad_spmspm_csc, cache_key


@register_funcify_and_cache_key(AddSparseSparse)
def numba_funcify_AddSparseSparse(op, node, **kwargs):
    # NOTE: There are promotions to uint64 in a few places.
    # The reason is numba promotes uint32 + uint32 to uint64,
    # while uint64 + uint64 is just as fast.
    # The benefit comes from using np.uint64(1) instead of 1 in increments:
    # uint32/uint64 + 1 becomes signed, adding negative-index handling to array access,
    # while uint64 + uint64 remains unsigned.
    # Also, there are scalar promotions to the output type. This is more efficient that
    # promoting the whole array at once, which requires additional memory allocation.

    x, y = node.inputs
    [z] = node.outputs
    out_dtype = z.type.dtype
    out_type = np.dtype(out_dtype).type

    x_format = x.type.format
    y_format = y.type.format

    # The output `z` has the same format as `x`.
    if x_format == "csr":
        constructor = sp.csr_matrix
    else:
        constructor = sp.csc_matrix

    @numba_basic.numba_njit
    def add_sparse_sparse_canonical(x, y):
        x_data = x.data
        y_data = y.data
        x_indices = x.indices.view(np.uint32)
        y_indices = y.indices.view(np.uint32)
        x_indptr = x.indptr.view(np.uint32)
        y_indptr = y.indptr.view(np.uint32)

        n_major = len(x_indptr) - 1
        capacity = len(x_data) + len(y_data)
        z_indptr = np.empty(n_major + 1, dtype=np.uint32)
        z_indices = np.empty(capacity, dtype=np.uint32)
        z_data = np.empty(capacity, dtype=out_dtype)

        nnz = np.uint64(0)
        z_indptr[0] = 0
        for major in range(n_major):
            x_ptr = np.uint64(x_indptr[major])
            x_end = np.uint64(x_indptr[major + 1])
            y_ptr = np.uint64(y_indptr[major])
            y_end = np.uint64(y_indptr[major + 1])

            while x_ptr < x_end and y_ptr < y_end:
                if x_indices[x_ptr] == y_indices[y_ptr]:
                    index = x_indices[x_ptr]
                    # Numba promotes narrow integer types.
                    # The outer cast ensures the result remains in the desired output dtype.
                    value = out_type(out_type(x_data[x_ptr]) + out_type(y_data[y_ptr]))
                    x_ptr += np.uint64(1)
                    y_ptr += np.uint64(1)
                elif x_indices[x_ptr] < y_indices[y_ptr]:
                    index = x_indices[x_ptr]
                    value = out_type(x_data[x_ptr])
                    x_ptr += np.uint64(1)
                else:
                    index = y_indices[y_ptr]
                    value = out_type(y_data[y_ptr])
                    y_ptr += np.uint64(1)
                z_indices[nnz] = index
                z_data[nnz] = value
                nnz += np.uint64(value != 0)

            while x_ptr < x_end:
                value = out_type(x_data[x_ptr])
                z_indices[nnz] = x_indices[x_ptr]
                z_data[nnz] = value
                nnz += np.uint64(value != 0)
                x_ptr += np.uint64(1)

            while y_ptr < y_end:
                value = out_type(y_data[y_ptr])
                z_indices[nnz] = y_indices[y_ptr]
                z_data[nnz] = value
                nnz += np.uint64(value != 0)
                y_ptr += np.uint64(1)

            z_indptr[major + 1] = nnz

        components = (
            z_data[:nnz],
            z_indices[:nnz].view(np.int32),
            z_indptr.view(np.int32),
        )
        return constructor(components, shape=x.shape)

    # Sparse indices are int32, so the top two uint32 values are free sentinels.
    # This approach allows us to have unsigned `next_index` and unsigned `head`
    not_visited = np.uint32(2**32 - 1)
    end_of_list = np.uint32(2**32 - 2)

    @numba_basic.numba_njit
    def add_sparse_sparse_general(x, y):
        x_data = x.data
        y_data = y.data
        x_indices = x.indices.view(np.uint32)
        y_indices = y.indices.view(np.uint32)
        x_indptr = x.indptr.view(np.uint32)
        y_indptr = y.indptr.view(np.uint32)
        n_major = len(x_indptr) - 1

        if x_format == "csr":
            n_minor = x.shape[1]
        else:
            n_minor = x.shape[0]

        next_index = np.full(n_minor, not_visited, dtype=np.uint32)
        x_row = np.empty(n_minor, dtype=out_dtype)
        y_row = np.empty(n_minor, dtype=out_dtype)
        capacity = len(x_data) + len(y_data)
        z_data = np.empty(capacity, dtype=out_dtype)
        z_indices = np.empty(capacity, dtype=np.int32)
        z_indptr = np.empty(n_major + 1, dtype=np.int32)
        nnz = np.uint64(0)
        z_indptr[0] = 0

        for major in range(n_major):
            head = end_of_list
            length = np.uint64(0)

            for ptr in range(x_indptr[major], x_indptr[major + 1]):
                index = x_indices[ptr]
                if next_index[index] == not_visited:
                    # Initialize accumulators only for indices visited in this row.
                    x_row[index] = out_type(x_data[ptr])
                    y_row[index] = 0
                    next_index[index] = head
                    head = index
                    length += np.uint64(1)
                else:
                    x_row[index] += out_type(x_data[ptr])

            for ptr in range(y_indptr[major], y_indptr[major + 1]):
                index = y_indices[ptr]
                if next_index[index] == not_visited:
                    x_row[index] = 0
                    y_row[index] = out_type(y_data[ptr])
                    next_index[index] = head
                    head = index
                    length += np.uint64(1)
                else:
                    y_row[index] += out_type(y_data[ptr])

            for _ in range(length):
                value = out_type(x_row[head] + y_row[head])
                if value != 0:
                    z_indices[nnz] = head
                    z_data[nnz] = value
                    nnz += np.uint64(1)
                index = head
                head = next_index[index]
                next_index[index] = not_visited

            z_indptr[major + 1] = nnz
        return constructor((z_data[:nnz], z_indices[:nnz], z_indptr), shape=x.shape)

    cache_version = 1
    cache_key = sha256(
        str((AddSparseSparse, x_format, y_format, out_dtype, cache_version)).encode()
    ).hexdigest()

    @numba_basic.numba_njit
    def add_sparse_sparse(x, y):
        assert x.shape == y.shape

        if x_format != y_format:
            # Addition is CSR-CSR or CSC-CSC
            if x_format == "csr":
                y = y.tocsr()
            else:
                y = y.tocsc()

        if x.has_canonical_format and y.has_canonical_format:
            return add_sparse_sparse_canonical(x, y)

        return add_sparse_sparse_general(x, y)

    return add_sparse_sparse, cache_key


@register_funcify_and_cache_key(MultiplySparseSparse)
def numba_funcify_MultiplySparseSparse(op, node, **kwargs):
    # NOTE: There are promotions to uint64 in a few places.
    # The reason is numba promotes uint32 + uint32 to uint64,
    # while uint64 + uint64 is just as fast.
    # The benefit comes from using np.uint64(1) instead of 1 in increments:
    # uint32/uint64 + 1 becomes signed, adding negative-index handling to array access,
    # while uint64 + uint64 remains unsigned.
    # Also, there are scalar promotions to the output type. This is more efficient that
    # promoting the whole array at once, which requires additional memory allocation.

    x, y = node.inputs
    [z] = node.outputs
    out_dtype = z.type.dtype
    out_type = np.dtype(out_dtype).type

    x_format = x.type.format
    y_format = y.type.format

    # The output `z` has the same format as `x`.
    if x_format == "csr":
        constructor = sp.csr_matrix
    else:
        constructor = sp.csc_matrix

    @numba_basic.numba_njit
    def multiply_sparse_sparse_canonical(x, y):
        x_data = x.data
        y_data = y.data
        x_indices = x.indices.view(np.uint32)
        y_indices = y.indices.view(np.uint32)
        x_indptr = x.indptr.view(np.uint32)
        y_indptr = y.indptr.view(np.uint32)
        n_major = len(x_indptr) - 1
        capacity = len(x_data) + len(y_data)
        z_indptr = np.empty(n_major + 1, dtype=np.uint32)
        z_indices = np.empty(capacity, dtype=np.uint32)
        z_data = np.empty(capacity, dtype=out_dtype)

        nnz = np.uint64(0)
        z_indptr[0] = 0
        for major in range(n_major):
            x_ptr = np.uint64(x_indptr[major])
            x_end = np.uint64(x_indptr[major + 1])
            y_ptr = np.uint64(y_indptr[major])
            y_end = np.uint64(y_indptr[major + 1])

            while x_ptr < x_end and y_ptr < y_end:
                # Numba promotes narrow integer types.
                # The outer casts ensure the result remains in the desired output dtype.
                if x_indices[x_ptr] == y_indices[y_ptr]:
                    index = x_indices[x_ptr]
                    value = out_type(out_type(x_data[x_ptr]) * out_type(y_data[y_ptr]))
                    x_ptr += np.uint64(1)
                    y_ptr += np.uint64(1)
                elif x_indices[x_ptr] < y_indices[y_ptr]:
                    index = x_indices[x_ptr]
                    value = out_type(out_type(x_data[x_ptr]) * out_type(0))
                    x_ptr += np.uint64(1)
                else:
                    index = y_indices[y_ptr]
                    value = out_type(out_type(0) * out_type(y_data[y_ptr]))
                    y_ptr += np.uint64(1)

                if value != 0:
                    z_indices[nnz] = index
                    z_data[nnz] = value
                    nnz += np.uint64(1)

            while x_ptr < x_end:
                value = out_type(out_type(x_data[x_ptr]) * out_type(0))
                if value != 0:
                    z_indices[nnz] = x_indices[x_ptr]
                    z_data[nnz] = value
                    nnz += np.uint64(1)
                x_ptr += np.uint64(1)

            while y_ptr < y_end:
                value = out_type(out_type(0) * out_type(y_data[y_ptr]))
                if value != 0:
                    z_indices[nnz] = y_indices[y_ptr]
                    z_data[nnz] = value
                    nnz += np.uint64(1)
                y_ptr += np.uint64(1)

            z_indptr[major + 1] = nnz

        components = (
            z_data[:nnz],
            z_indices[:nnz].view(np.int32),
            z_indptr.view(np.int32),
        )

        return constructor(components, shape=x.shape)

    # Sparse indices are int32, so the top two uint32 values are free sentinels.
    not_visited = np.uint32(2**32 - 1)
    end_of_list = np.uint32(2**32 - 2)

    @numba_basic.numba_njit
    def multiply_sparse_sparse_general(x, y):
        x_data = x.data
        y_data = y.data
        x_indices = x.indices.view(np.uint32)
        y_indices = y.indices.view(np.uint32)
        x_indptr = x.indptr.view(np.uint32)
        y_indptr = y.indptr.view(np.uint32)
        n_major = len(x_indptr) - 1

        if x_format == "csr":
            n_minor = x.shape[1]
        else:
            n_minor = x.shape[0]

        next_index = np.full(n_minor, not_visited, dtype=np.uint32)
        x_row = np.empty(n_minor, dtype=out_dtype)
        y_row = np.empty(n_minor, dtype=out_dtype)
        capacity = len(x_data) + len(y_data)
        z_data = np.empty(capacity, dtype=out_dtype)
        z_indices = np.empty(capacity, dtype=np.int32)
        z_indptr = np.empty(n_major + 1, dtype=np.int32)

        nnz = np.uint64(0)
        z_indptr[0] = 0
        for major in range(n_major):
            head = end_of_list
            length = np.uint64(0)

            for ptr in range(x_indptr[major], x_indptr[major + 1]):
                index = x_indices[ptr]
                if next_index[index] == not_visited:
                    # Initialize accumulators only for indices visited in this row.
                    x_row[index] = out_type(x_data[ptr])
                    y_row[index] = 0
                    next_index[index] = head
                    head = index
                    length += np.uint64(1)
                else:
                    x_row[index] += out_type(x_data[ptr])

            for ptr in range(y_indptr[major], y_indptr[major + 1]):
                index = y_indices[ptr]
                if next_index[index] == not_visited:
                    x_row[index] = 0
                    y_row[index] = out_type(y_data[ptr])
                    next_index[index] = head
                    head = index
                    length += np.uint64(1)
                else:
                    y_row[index] += out_type(y_data[ptr])

            for _ in range(length):
                value = out_type(x_row[head] * y_row[head])
                if value != 0:
                    z_indices[nnz] = head
                    z_data[nnz] = value
                    nnz += np.uint64(1)
                index = head
                head = next_index[index]
                next_index[index] = not_visited
            z_indptr[major + 1] = nnz

        return constructor((z_data[:nnz], z_indices[:nnz], z_indptr), shape=x.shape)

    cache_version = 1
    cache_key = sha256(
        str(
            (MultiplySparseSparse, x_format, y_format, out_dtype, cache_version)
        ).encode()
    ).hexdigest()

    @numba_basic.numba_njit
    def multiply_sparse_sparse(x, y):
        assert x.shape == y.shape

        if x_format != y_format:
            # Multiplication is CSR-CSR or CSC-CSC
            if x_format == "csr":
                y = y.tocsr()
            else:
                y = y.tocsc()

        if x.has_canonical_format and y.has_canonical_format:
            return multiply_sparse_sparse_canonical(x, y)

        return multiply_sparse_sparse_general(x, y)

    return multiply_sparse_sparse, cache_key


@register_funcify_default_op_cache_key(AddSparseDense)
def numba_funcify_AddSparseDense(op, node, **kwargs):
    x, y = node.inputs
    [z] = node.outputs
    out_dtype = z.type.dtype
    x_format = x.type.format
    same_dtype = y.type.dtype == out_dtype

    @numba_basic.numba_njit
    def add_sparse_dense(x, y):
        assert x.shape == y.shape
        if same_dtype:
            z = y.copy()
        else:
            z = y.astype(out_dtype)  # Also generates a copy

        M = x.shape[0]
        N = x.shape[1]
        data = x.data
        indices = x.indices.view(np.uint32)
        indptr = x.indptr.view(np.uint32)

        # Since `z` already has the output type, and we use `+=`, we don't need to cast `data`.
        # Numba emits LLVM that automatically handles the cast.
        if x_format == "csc":
            for j in range(N):
                for i_idx in range(indptr[j], indptr[j + 1]):
                    i = indices[i_idx]
                    z[i, j] += data[i_idx]
        else:
            for i in range(M):
                for j_idx in range(indptr[i], indptr[i + 1]):
                    j = indices[j_idx]
                    z[i, j] += data[j_idx]

        return z

    return add_sparse_dense


@register_funcify_default_op_cache_key(AddSparseSparseData)
def numba_funcify_AddSparseSparseData(op, node, **kwargs):
    if node.outputs[0].type.format == "csr":
        constructor = sp.csr_matrix
    else:
        constructor = sp.csc_matrix

    @numba_basic.numba_njit
    def add_sparse_sparse_data(x, y):
        assert x.shape == y.shape
        assert x.data.shape == y.data.shape

        # The Op forces x.data and y.data to be of the same type, no need to cast.
        data = np.empty_like(x.data)
        for ptr in range(len(data)):
            data[ptr] = x.data[ptr] + y.data[ptr]

        return constructor((data, x.indices.copy(), x.indptr.copy()), shape=x.shape)

    return add_sparse_sparse_data


@register_funcify_and_cache_key(StructuredAddSparseDenseVector)
def numba_funcify_StructuredAddSparseDenseVector(op, node, **kwargs):
    # NOTE: There are promotions to uint64 in a few places.
    # The reason is numba promotes uint32 + uint32 to uint64,
    # while uint64 + uint64 is just as fast.
    # The benefit comes from using np.uint64(1) instead of 1 in increments:
    # uint32/uint64 + 1 becomes signed, adding negative-index handling to array access,
    # while uint64 + uint64 remains unsigned.

    [z] = node.outputs
    out_dtype = z.type.dtype
    out_type = np.dtype(out_dtype).type
    complex_dtype = np.issubdtype(np.dtype(out_dtype), np.complexfloating)
    out_format = z.type.format

    cache_version = 1
    cache_key = sha256(
        str((type(op), out_format, out_dtype, cache_version)).encode()
    ).hexdigest()

    if out_format == "csr":
        constructor = sp.csr_matrix
    else:
        constructor = sp.csc_matrix

    sum_duplicates = make_sum_duplicates(out_format)

    @numba_basic.numba_njit(fastmath=False)
    def add_nonfinite(value, other):
        # Preserve NaN propagation from 0 * NaN and 0 * inf.
        return out_type(value + (value != 0) * other)

    @numba_basic.numba_njit
    def finite_kernel(x, y):
        assert x.shape[1] == y.shape[0]

        # Can't reasign like in `x = sum_duplicates(x)` because Numba cannot infer a common
        # type when the original indices are readonly.
        if x.has_canonical_format:
            x_data = x.data
            x_indices = x.indices.view(np.uint32)
            x_indptr = x.indptr.view(np.uint32)
        else:
            canonical_x = sum_duplicates(x)
            x_data = canonical_x.data
            x_indices = canonical_x.indices.view(np.uint32)
            x_indptr = canonical_x.indptr.view(np.uint32)

        n_major = len(x_indptr) - 1
        capacity = len(x_data)
        z_indptr = np.empty(n_major + 1, dtype=np.uint32)
        z_indices = np.empty(capacity, dtype=np.uint32)
        z_data = np.empty(capacity, dtype=out_dtype)

        nnz = np.uint64(0)
        z_indptr[0] = 0

        for major in range(n_major):
            start = x_indptr[major]
            end = x_indptr[major + 1]
            for ptr in range(start, end):
                index = x_indices[ptr]
                if out_format == "csr":
                    column = index
                else:
                    column = major

                value = x_data[ptr]

                # Avoid a complex multiply for the structured mask.
                if complex_dtype:
                    if value != 0:
                        value = out_type(value + y[column])
                        if value != 0:
                            z_indices[nnz] = index
                            z_data[nnz] = value
                            nnz += np.uint64(1)
                else:
                    value = out_type(value + (value != 0) * y[column])
                    if value != 0:
                        z_indices[nnz] = index
                        z_data[nnz] = value
                        nnz += np.uint64(1)

            z_indptr[major + 1] = nnz

        components = (
            z_data[:nnz],
            z_indices[:nnz].view(np.int32),
            z_indptr.view(np.int32),
        )

        return constructor(components, shape=x.shape)

    @numba_basic.numba_njit
    def nonfinite_kernel(x, y, nonfinite_columns):
        assert x.shape[1] == y.shape[0]

        # Can't reasign like in `x = sum_duplicates(x)` because Numba cannot infer a common
        # type when the original indices are readonly.
        if x.has_canonical_format:
            x_data = x.data
            x_indices = x.indices.view(np.uint32)
            x_indptr = x.indptr.view(np.uint32)
        else:
            canonical_x = sum_duplicates(x)
            x_data = canonical_x.data
            x_indices = canonical_x.indices.view(np.uint32)
            x_indptr = canonical_x.indptr.view(np.uint32)

        n_major = len(x_indptr) - 1
        if out_format == "csr":
            n_minor = np.uint64(x.shape[1])
        else:
            n_minor = np.uint64(x.shape[0])

        nonfinite_columns_n = len(nonfinite_columns)
        capacity = len(x_data) + x.shape[0] * nonfinite_columns_n

        z_indptr = np.empty(n_major + 1, dtype=np.uint32)
        z_indices = np.empty(capacity, dtype=np.uint32)
        z_data = np.empty(capacity, dtype=out_dtype)

        nnz = np.uint64(0)
        z_indptr[0] = 0
        for major in range(n_major):
            ptr = np.uint64(x_indptr[major])
            end = np.uint64(x_indptr[major + 1])
            extra_ptr = np.uint64(0)

            if out_format == "csr":
                extra_count = np.uint64(nonfinite_columns_n)
            elif not np.isfinite(y[major]):
                extra_count = n_minor
            else:
                extra_count = np.uint64(0)

            # CSC columns with finite y need no implicit entries, even in this kernel.
            if out_format == "csc" and extra_count == 0:
                for data_ptr in range(ptr, end):
                    value = x_data[data_ptr]
                    value = out_type(value + (value != 0) * y[major])
                    if value != 0:
                        z_indices[nnz] = x_indices[data_ptr]
                        z_data[nnz] = value
                        nnz += np.uint64(1)
                z_indptr[major + 1] = nnz
                continue

            while ptr < end or extra_ptr < extra_count:
                if ptr < end:
                    stored_index = x_indices[ptr]
                else:
                    stored_index = n_minor

                extra_index = n_minor
                if extra_ptr < extra_count:
                    if out_format == "csr":
                        extra_index = np.uint64(nonfinite_columns[extra_ptr])
                    else:
                        extra_index = extra_ptr

                index = min(stored_index, extra_index)
                value = out_type(0)

                if stored_index == index:
                    value = x_data[ptr]
                    ptr += np.uint64(1)

                if extra_index == index:
                    extra_ptr += np.uint64(1)

                if out_format == "csr":
                    column = index
                else:
                    column = major

                value = add_nonfinite(value, y[column])
                if value != 0:
                    z_indices[nnz] = index
                    z_data[nnz] = value
                    nnz += np.uint64(1)

            z_indptr[major + 1] = nnz

        components = (
            z_data[:nnz],
            z_indices[:nnz].view(np.int32),
            z_indptr.view(np.int32),
        )

        return constructor(components, shape=x.shape)

    @numba_basic.numba_njit
    def structured_add_sparse_dense_vector(x, y):
        all_finite = True

        # This is faster in numba, it's SIMD-friendly.
        for y_i in y:
            all_finite &= np.isfinite(y_i)

        if all_finite:
            return finite_kernel(x, y)

        nonfinite_columns = np.flatnonzero(~np.isfinite(y))
        return nonfinite_kernel(x, y, nonfinite_columns)

    return structured_add_sparse_dense_vector, cache_key


@register_funcify_and_cache_key(Usmm)
def numba_funcify_Usmm(op, node, **kwargs):
    """Computes the dense matrix resulting from `alpha * x @ y + z`.

    `alpha` is scalar, at least one of `x` and `y` is a sparse matrix, and `z` is a dense matrix.
    """
    _, x, y, z = node.inputs
    [out] = node.outputs
    out_dtype = out.type.dtype
    out_type = np.dtype(out_dtype).type

    x_is_sparse = psb._is_sparse_variable(x)
    y_is_sparse = psb._is_sparse_variable(y)
    x_format = x.type.format if x_is_sparse else None
    y_format = y.type.format if y_is_sparse else None
    z_same_dtype = z.type.dtype == out_dtype

    # Used in the wrapper's fallback, when alpha is nonfinite.
    dot_node = Dot().make_node(x, y)
    dot, dot_cache_key = numba_funcify_SparseDot(dot_node.op, dot_node, **kwargs)
    cache_version = 5

    cache_key = sha256(
        str(
            (
                type(op),
                tuple(inp.type for inp in node.inputs),
                tuple(out.type for out in node.outputs),
                dot_cache_key,
                cache_version,
            )
        ).encode()
    ).hexdigest()

    def wrap_fused_kernel(kernel):
        # `fastmath=False` to preserve nonfinite results in the fallback expression.
        @numba_basic.numba_njit(fastmath=False)
        def usmm(alpha, x, y, z):
            n_row, n_inner = x.shape
            y_n_row, n_col = y.shape
            assert n_inner == y_n_row
            shape = (n_row, n_col)

            if (
                not np.isfinite(alpha.item())
                or (z.shape[0] != 1 and z.shape[0] != n_row)
                or (z.shape[1] != 1 and z.shape[1] != n_col)
            ):
                # Alpha is nonfinite or z can't broadcast to x @ y.
                # The original expression will resolve the broadcasting.
                return (dot(x, y) * alpha.item() + z).astype(out_dtype)

            # Initialize `out` from `z`, casting or broadcasting as needed, before accumulating.
            if z.shape == shape:
                if z_same_dtype:
                    out = z.copy()
                else:
                    out = z.astype(out_dtype)
            else:
                out = np.empty(shape, dtype=out_dtype)
                out[:, :] = z

            return kernel(alpha, x, y, out, n_row, n_inner, n_col)

        return usmm, cache_key

    # NOTE: It's more performant to apply `out_type` at the element level rather than converting
    # the entire array at once. When the actual type the same than the output type, numba
    # does an optimization that eliminates unnecessary type conversions.
    if x_is_sparse and not y_is_sparse:

        @numba_basic.numba_njit
        def usmm_sparse_dense(alpha, x, y, out, n_row, n_inner, n_col):
            alpha_val = alpha.item()

            x_indices = x.indices.view(np.uint32)
            x_indptr = x.indptr.view(np.uint32)

            x_data = x.data

            # CSR completes each row with one write using a scalar accumulator.
            # CSC revisits output rows across columns, so it accumulates directly in out.
            if n_col == 1:
                # Y is a dense vector.
                if x_format == "csr":
                    for i in range(n_row):
                        acc = out[i, 0]
                        for x_idx in range(x_indptr[i], x_indptr[i + 1]):
                            k = x_indices[x_idx]
                            x_val = alpha_val * out_type(x_data[x_idx])
                            acc += x_val * out_type(y[k, 0])
                        out[i, 0] = acc
                else:
                    for k in range(n_inner):
                        y_val = out_type(y[k, 0])
                        for x_idx in range(x_indptr[k], x_indptr[k + 1]):
                            i = x_indices[x_idx]
                            x_val = alpha_val * out_type(x_data[x_idx])
                            out[i, 0] += x_val * y_val
                return out

            if x_format == "csr":
                for i in range(n_row):
                    for x_idx in range(x_indptr[i], x_indptr[i + 1]):
                        k = x_indices[x_idx]
                        x_val = alpha_val * out_type(x_data[x_idx])
                        for j in range(n_col):
                            out[i, j] += x_val * out_type(y[k, j])
            else:
                for k in range(n_inner):
                    for x_idx in range(x_indptr[k], x_indptr[k + 1]):
                        i = x_indices[x_idx]
                        x_val = alpha_val * out_type(x_data[x_idx])
                        for j in range(n_col):
                            out[i, j] += x_val * out_type(y[k, j])

            return out

        return wrap_fused_kernel(usmm_sparse_dense)

    if not x_is_sparse and y_is_sparse:

        @numba_basic.numba_njit
        def usmm_dense_sparse(alpha, x, y, out, n_row, n_inner, n_col):
            alpha_val = alpha.item()
            indices = y.indices.view(np.uint32)
            indptr = y.indptr.view(np.uint32)
            y_data = y.data

            # CSC completes each column with one write using a scalar accumulator.
            # CSR revisits output columns across rows, so it accumulates directly in out.
            if n_row == 1:
                # X is a dense vector.
                if y_format == "csc":
                    for j in range(n_col):
                        acc = out[0, j]
                        for pos in range(indptr[j], indptr[j + 1]):
                            k = indices[pos]
                            value = alpha_val * out_type(y_data[pos])
                            acc += out_type(x[0, k]) * value
                        out[0, j] = acc
                else:
                    for k in range(n_inner):
                        x_val = out_type(x[0, k])
                        for pos in range(indptr[k], indptr[k + 1]):
                            j = indices[pos]
                            value = alpha_val * out_type(y_data[pos])
                            out[0, j] += x_val * value
                return out

            if y_format == "csc":
                for j in range(n_col):
                    for pos in range(indptr[j], indptr[j + 1]):
                        k = indices[pos]
                        value = alpha_val * out_type(y_data[pos])
                        for i in range(n_row):
                            out[i, j] += out_type(x[i, k]) * value
            else:
                for k in range(n_inner):
                    for pos in range(indptr[k], indptr[k + 1]):
                        j = indices[pos]
                        value = alpha_val * out_type(y_data[pos])
                        for i in range(n_row):
                            out[i, j] += out_type(x[i, k]) * value
            return out

        return wrap_fused_kernel(usmm_dense_sparse)

    @numba_basic.numba_njit
    def usmm_sparse_sparse(alpha, x, y, out, n_row, n_inner, n_col):
        alpha_val = alpha.item()

        if x_format == "csr" and y_format == "csc":
            y = y.tocsr()

        x_indices = x.indices.view(np.uint32)
        x_indptr = x.indptr.view(np.uint32)
        y_indices = y.indices.view(np.uint32)
        y_indptr = y.indptr.view(np.uint32)

        x_data = x.data
        y_data = y.data

        if x_format == "csr":
            for i in range(n_row):
                for x_idx in range(x_indptr[i], x_indptr[i + 1]):
                    k = x_indices[x_idx]
                    x_val = alpha_val * out_type(x_data[x_idx])
                    for y_idx in range(y_indptr[k], y_indptr[k + 1]):
                        out[i, y_indices[y_idx]] += x_val * out_type(y_data[y_idx])
        elif y_format == "csc":
            for j in range(n_col):
                for y_idx in range(y_indptr[j], y_indptr[j + 1]):
                    k = y_indices[y_idx]
                    y_val = alpha_val * out_type(y_data[y_idx])
                    for x_idx in range(x_indptr[k], x_indptr[k + 1]):
                        out[x_indices[x_idx], j] += out_type(x_data[x_idx]) * y_val
        else:
            for k in range(n_inner):
                for x_idx in range(x_indptr[k], x_indptr[k + 1]):
                    i = x_indices[x_idx]
                    x_val = alpha_val * out_type(x_data[x_idx])
                    for y_idx in range(y_indptr[k], y_indptr[k + 1]):
                        out[i, y_indices[y_idx]] += x_val * out_type(y_data[y_idx])

        return out

    return wrap_fused_kernel(usmm_sparse_sparse)


@register_funcify_and_cache_key(UsmmCscDense)
def numba_funcify_UsmmCscDense(op, node, **kwargs):
    inplace = op.inplace
    out_dtype = node.outputs[0].dtype
    out_type = np.dtype(out_dtype).type

    cache_version = 4
    cache_key = sha256(
        str(
            (
                type(op),
                inplace,
                tuple(inp.type for inp in node.inputs),
                tuple(out.type for out in node.outputs),
                cache_version,
            )
        ).encode()
    ).hexdigest()

    @numba_basic.numba_njit
    def accumulate(scale, values, indices, indptr, y, out, n_cols):
        indices = indices.view(np.uint32)
        indptr = indptr.view(np.uint32)

        if n_cols == 1:
            for k in range(len(indptr) - 1):
                y_value = y[k, 0]
                for pos in range(indptr[k], indptr[k + 1]):
                    row = indices[pos]
                    value = scale * values[pos]
                    out[row, 0] += value * y_value
            return out

        for k in range(len(indptr) - 1):
            for pos in range(indptr[k], indptr[k + 1]):
                row = indices[pos]
                value = scale * values[pos]
                for col in range(n_cols):
                    out[row, col] += value * y[k, col]

        return out

    # Preserve nonfinite results in the final scaling and addition.
    @numba_basic.numba_njit(fastmath=False)
    def usmm_csc_dense(alpha, values, indices, indptr, n_rows, y, z):
        assert len(indptr) - 1 == y.shape[0]
        n_cols = y.shape[1]
        shape = (n_rows.item(), n_cols)
        if (z.shape[0] != 1 and z.shape[0] != shape[0]) or (
            z.shape[1] != 1 and z.shape[1] != shape[1]
        ):
            raise ValueError("z must broadcast to the shape of x @ y")

        reuse_z = inplace and z.shape == shape
        scale = alpha.item()
        nonfinite_alpha = not np.isfinite(scale)

        if nonfinite_alpha:
            out = np.zeros(shape, dtype=out_dtype)
            scale = out_type(1)
        elif reuse_z:
            out = z
        elif z.shape == shape:
            out = z.copy()
        else:
            out = np.empty(shape, dtype=out_dtype)
            out[:, :] = z

        out = accumulate(scale, values, indices, indptr, y, out, n_cols)

        if nonfinite_alpha:
            out *= alpha.item()
            out += z
            if reuse_z:
                z[:, :] = out
                return z
        return out

    return usmm_csc_dense, cache_key
