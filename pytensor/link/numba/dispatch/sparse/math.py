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
from pytensor.sparse import (
    Dot,
    SparseDenseMultiply,
    SparseDenseVectorMultiply,
    SparseSparseMultiply,
    SpSum,
    StructuredDot,
    StructuredDotGradCSC,
    StructuredDotGradCSR,
)


@register_funcify_default_op_cache_key(SpSum)
def numba_funcify_SpSum(op, node, **kwargs):
    axis = op.axis

    @numba_basic.numba_njit
    def perform(x):
        return x.sum(axis)

    return perform


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

    cache_version = 2
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
    out_dtype = g_xy_dtype

    cache_key = sha256(
        str(
            (
                type(op),
                x_format,
                y_format,
                y_dtype,
                g_xy_format,
                out_dtype,
                y.type.shape,
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
                            output[value_idx] = g_xy[row_idx] * y[x_indices[value_idx]]
                    return output

                @numba_basic.numba_njit
                def grad_spmdv_csr(x_indices, x_ptr, y, g_xy):
                    return _grad_spmdv_csr(x_indices, x_ptr, y[:, 0], g_xy[:, 0])

                return grad_spmdv_csr, cache_key
            else:
                # Y is a matrix
                if config.compiler_verbose and y_dtype != out_dtype:
                    print(  # noqa: T201
                        "Numba StructuredDotGrad requires a type casting of inputs: "
                        f"{y_dtype=}, {g_xy_dtype=}."
                    )

                @numba_basic.numba_njit
                def grad_spmdm_csr(x_indices, x_ptr, y, g_xy):
                    size = len(x_ptr) - 1
                    x_indices = x_indices.view(np.uint32)
                    x_ptr = x_ptr.view(np.uint32)

                    if y_dtype != out_dtype:
                        new_out_dtype = np.result_type(y, g_xy)
                        output = np.zeros(len(x_indices), dtype=new_out_dtype)
                        y = y.astype(out_dtype)
                        g_xy = g_xy.astype(out_dtype)
                    else:
                        output = np.zeros(len(x_indices), dtype=out_dtype)

                    for row_idx in range(size):
                        for value_idx in range(x_ptr[row_idx], x_ptr[row_idx + 1]):
                            output[value_idx] = np.dot(
                                g_xy[row_idx], y[x_indices[value_idx]]
                            )
                    return output

                return grad_spmdm_csr, cache_key
        else:
            # X is CSC
            @numba_basic.numba_njit
            def grad_spmdm_csc(x_indices, x_ptr, y, g_xy):
                # len(x_indices) gives the number of non-zero elements in X.
                output = np.zeros(len(x_indices), dtype=out_dtype)
                size = len(x_ptr) - 1
                x_indices = x_indices.view(np.uint32)
                x_ptr = x_ptr.view(np.uint32)

                for col_idx in range(size):
                    for value_idx in range(x_ptr[col_idx], x_ptr[col_idx + 1]):
                        output[value_idx] = np.dot(
                            g_xy[x_indices[value_idx]], y[col_idx]
                        )
                return output

            return grad_spmdm_csc, cache_key

    # Y is sparse. In either case we need 'dot_csr_rows'
    @numba_basic.numba_njit
    def dot_csr_rows(x_ptr, x_indices, x_data, x_row, y_ptr, y_indices, y_data, y_row):
        x_p = x_ptr[x_row]
        x_end = x_ptr[x_row + 1]
        y_p = y_ptr[y_row]
        y_end = y_ptr[y_row + 1]

        acc = 0.0
        while x_p < x_end and y_p < y_end:
            x_col = x_indices[x_p]
            y_col = y_indices[y_p]
            if x_col == y_col:
                acc += x_data[x_p] * y_data[y_p]
                x_p += 1
                y_p += 1
            elif x_col < y_col:
                x_p += 1
            else:
                y_p += 1

        return acc

    if x_format == "csr":
        assert g_xy_format == "csr"
        assert psb._is_sparse_variable(y)

        @numba_basic.numba_njit
        def grad_spmspm_csr(x_indices, x_ptr, y, g_xy):
            if y_format == "csc":
                y = y.tocsr()

            g_xy_data = g_xy.data
            g_xy_indices = g_xy.indices.view(np.uint32)
            g_xy_ptr = g_xy.indptr.view(np.uint32)

            y_data = y.data
            y_indices = y.indices.view(np.uint32)
            y_ptr = y.indptr.view(np.uint32)

            n_row = len(x_ptr) - 1
            output = np.zeros(len(x_indices), dtype=out_dtype)

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


@register_funcify_default_op_cache_key(SparseSparseMultiply)
def numba_funcify_SparseSparseMultiply(op, node, **kwargs):
    _, y = node.inputs
    [z] = node.outputs
    out_dtype = z.type.dtype
    out_format = z.type.format
    y_format = y.type.format

    # `out_format` is equal to `x.format`. We only convert `y`, if needed.
    # This is a single pass implementation. We may trade memory for fewer passes.
    # The number of nonzeros is the "intersection" of nonzeros
    if out_format == "csr":

        @numba_basic.numba_njit
        def multiply_s_s_csr(x, y):
            assert x.shape == y.shape
            if y_format != "csr":
                y = y.tocsr()

            n_row = x.shape[0]
            x_indices = x.indices.view(np.uint32)
            y_indices = y.indices.view(np.uint32)
            x_indptr = x.indptr.view(np.uint32)
            y_indptr = y.indptr.view(np.uint32)

            output_capacity = min(len(x_indices), len(y_indices))
            z_indptr = np.empty(n_row + 1, dtype=np.int32)
            z_indices = np.empty(output_capacity, dtype=np.int32)
            z_data = np.empty(output_capacity, dtype=out_dtype)

            nnz = 0
            z_indptr[0] = 0
            for i in range(n_row):
                x_ptr = x_indptr[i]
                y_ptr = y_indptr[i]
                while x_ptr < x_indptr[i + 1] and y_ptr < y_indptr[i + 1]:
                    x_j = x_indices[x_ptr]
                    y_j = y_indices[y_ptr]
                    if x_j == y_j:
                        z_indices[nnz] = x_j
                        z_data[nnz] = x.data[x_ptr] * y.data[y_ptr]
                        nnz += 1
                        x_ptr += 1
                        y_ptr += 1
                    elif x_j < y_j:
                        x_ptr += 1
                    else:
                        y_ptr += 1
                z_indptr[i + 1] = nnz

            return sp.csr_matrix(
                (z_data[:nnz], z_indices[:nnz], z_indptr), shape=x.shape
            )

        return multiply_s_s_csr
    else:

        @numba_basic.numba_njit
        def multiply_s_s_csc(x, y):
            assert x.shape == y.shape
            if y_format != "csc":
                y = y.tocsc()

            n_col = x.shape[1]
            x_indices = x.indices.view(np.uint32)
            y_indices = y.indices.view(np.uint32)
            x_indptr = x.indptr.view(np.uint32)
            y_indptr = y.indptr.view(np.uint32)

            output_capacity = min(len(x_indices), len(y_indices))
            z_indptr = np.empty(n_col + 1, dtype=np.int32)
            z_indices = np.empty(output_capacity, dtype=np.int32)
            z_data = np.empty(output_capacity, dtype=out_dtype)

            nnz = 0
            z_indptr[0] = 0
            for j in range(n_col):
                x_ptr = x_indptr[j]
                y_ptr = y_indptr[j]
                while x_ptr < x_indptr[j + 1] and y_ptr < y_indptr[j + 1]:
                    x_i = x_indices[x_ptr]
                    y_i = y_indices[y_ptr]
                    if x_i == y_i:
                        z_indices[nnz] = x_i
                        z_data[nnz] = x.data[x_ptr] * y.data[y_ptr]
                        nnz += 1
                        x_ptr += 1
                        y_ptr += 1
                    elif x_i < y_i:
                        x_ptr += 1
                    else:
                        y_ptr += 1
                z_indptr[j + 1] = nnz

            return sp.csc_matrix(
                (z_data[:nnz], z_indices[:nnz], z_indptr), shape=x.shape
            )

        return multiply_s_s_csc
