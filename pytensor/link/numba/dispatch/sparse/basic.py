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
    GetItem2d,
    GetItem2Lists,
    GetItem2ListsGrad,
    GetItemList,
    GetItemListGrad,
    GetItemScalar,
    HStack,
    Neg,
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


@register_funcify_default_op_cache_key(GetItemList)
def numba_funcify_GetItemList(op, node, **kwargs):
    output_format = node.outputs[0].type.format

    @numba_basic.numba_njit
    def get_item_list_csr(x, idxs):
        # Reproduces SciPy when running:
        # x_sparse[idxs]
        x_csr = x.tocsr()
        n_rows, n_cols = x_csr.shape
        n_out_rows = idxs.shape[0]

        x_data = x_csr.data
        x_indices = x_csr.indices.view(np.uint32)
        x_indptr = x_csr.indptr.view(np.uint32)

        out_indptr = np.empty(n_out_rows + 1, dtype=np.int32)
        out_indptr[0] = 0
        norm_idx = np.empty(n_out_rows, dtype=np.int32)

        # Normalize (negative) indices and compute output indptr in the same pass.
        total_nnz = 0
        for out_row_idx in range(n_out_rows):
            row_idx = idxs[out_row_idx]
            if row_idx < 0:
                row_idx += n_rows
            if row_idx < 0 or row_idx >= n_rows:
                raise IndexError("row index out of bounds")

            norm_row_idx = row_idx
            norm_idx[out_row_idx] = norm_row_idx
            total_nnz += x_indptr[norm_row_idx + 1] - x_indptr[norm_row_idx]
            out_indptr[out_row_idx + 1] = total_nnz

        # Once the number of non-zero elements is known, allocate data and indices vectors.
        out_data = np.empty(total_nnz, dtype=x_data.dtype)
        out_indices = np.empty(total_nnz, dtype=np.int32)

        # For the selected rows, copy data and indices from source to destination.
        # Duplicated entries will lead to duplicated rows.
        for out_row_idx in range(n_out_rows):
            row_idx = norm_idx[out_row_idx]
            src_start = x_indptr[row_idx]
            src_stop = x_indptr[row_idx + 1]
            dst_start = out_indptr[out_row_idx]

            # We could have used slicing, but numba is faster with explicit loops.
            dst_idx = dst_start
            for src_i in range(src_start, src_stop):
                out_data[dst_idx] = x_data[src_i]
                out_indices[dst_idx] = x_indices[src_i]
                dst_idx += 1

        return sp.sparse.csr_matrix(
            (out_data, out_indices, out_indptr), shape=(n_out_rows, n_cols)
        )

    if output_format == "csr":
        return get_item_list_csr

    @numba_basic.numba_njit
    def get_item_list_csc(x, idx):
        return get_item_list_csr(x, idx).tocsc()

    return get_item_list_csc


@register_funcify_default_op_cache_key(GetItemListGrad)
def numba_funcify_GetItemListGrad(op, node, **kwargs):
    output_format = node.outputs[0].type.format
    out_dtype = node.outputs[0].type.dtype

    @numba_basic.numba_njit
    def get_item_list_grad_csr(x, idxs, gz):
        # Reproduces SciPy when running:
        # y = [csc|csr]_matrix(x.shape)
        # for i in range(len(idxs)):
        #     y[idxs[i]] = gz[i]

        n_rows, n_cols = x.shape
        n_out_rows = idxs.shape[0]
        gz_n_rows = gz.shape[0]

        # Normalize (negative) indices and build row_to_pos mapping.
        norm_idx = np.empty(n_out_rows, dtype=np.int32)
        row_to_pos = np.full(n_rows, -1, dtype=np.int32)
        touched_n_rows = 0
        for src_row in range(n_out_rows):
            row_idx = idxs[src_row]
            if row_idx < 0:
                row_idx += n_rows
            if row_idx < 0 or row_idx >= n_rows:
                raise IndexError("row index out of bounds")
            if src_row >= gz_n_rows:
                raise IndexError("gradient row index out of bounds")
            norm_idx[src_row] = row_idx

            if row_to_pos[row_idx] == -1:
                row_to_pos[row_idx] = touched_n_rows
                touched_n_rows += 1

        # Process gz in CSR format.
        gz_csr = gz.tocsr()
        gz_data = gz_csr.data
        gz_indices = gz_csr.indices.view(np.uint32)
        gz_indptr = gz_csr.indptr.view(np.uint32)

        # Row-wise buffers that reproduce SciPy row-assignment behavior:
        # repeated assignments keep the union of touched columns and turn
        # missing entries into explicit zeros.
        row_data = np.zeros((touched_n_rows, n_cols), dtype=out_dtype)
        row_mask = np.zeros((touched_n_rows, n_cols), dtype=np.bool_)
        row_seen = np.zeros(touched_n_rows, dtype=np.bool_)

        for src_row in range(n_out_rows):
            row_idx = norm_idx[src_row]
            row_pos = row_to_pos[row_idx]

            if row_seen[row_pos]:
                for col_idx in range(n_cols):
                    if row_mask[row_pos, col_idx]:
                        row_data[row_pos, col_idx] = 0
            else:
                row_seen[row_pos] = True

            for i in range(gz_indptr[src_row], gz_indptr[src_row + 1]):
                col_idx = gz_indices[i]
                row_data[row_pos, col_idx] = gz_data[i]
                row_mask[row_pos, col_idx] = True

        # Compute out_indptr by counting True entries in row_mask row-by-row.
        out_indptr = np.empty(n_rows + 1, dtype=np.int32)
        out_indptr[0] = 0

        total_nnz = 0
        for row_idx in range(n_rows):
            row_pos = row_to_pos[row_idx]
            if row_pos >= 0 and row_seen[row_pos]:
                row_nnz = 0
                for col_idx in range(n_cols):
                    if row_mask[row_pos, col_idx]:
                        row_nnz += 1
                total_nnz += row_nnz
            out_indptr[row_idx + 1] = total_nnz

        # Once the number of non-zero elements is known, allocate data and indices vectors.
        out_data = np.empty(total_nnz, dtype=out_dtype)
        out_indices = np.empty(total_nnz, dtype=np.int32)

        # Populate output indices and data, by row, scanning columns in ascending order.
        out_pos = 0
        for row_idx in range(n_rows):
            row_pos = row_to_pos[row_idx]
            if row_pos < 0 or not row_seen[row_pos]:
                continue

            for col_idx in range(n_cols):
                if row_mask[row_pos, col_idx]:
                    out_indices[out_pos] = col_idx
                    out_data[out_pos] = row_data[row_pos, col_idx]
                    out_pos += 1

        return sp.sparse.csr_matrix(
            (out_data, out_indices, out_indptr), shape=(n_rows, n_cols)
        )

    if output_format == "csr":
        return get_item_list_grad_csr

    @numba_basic.numba_njit
    def get_item_list_grad_csc(x, idx, gz):
        return get_item_list_grad_csr(x, idx, gz).tocsc()

    return get_item_list_grad_csc


@register_funcify_default_op_cache_key(GetItem2Lists)
def numba_funcify_GetItem2Lists(op, node, **kwargs):
    out_dtype = node.outputs[0].type.dtype

    @numba_basic.numba_njit
    def get_item_2lists(x, ind1, ind2):
        # Reproduces SciPy and NumPy when running:
        # np.asarray(x[ind1, ind2]).flatten()

        if ind1.shape != ind2.shape:
            raise ValueError("shape mismatch in row/column indices")

        # Output vector contains as many elements as the length of the index lists.
        out_size = ind1.shape[0]
        out = np.zeros(out_size, dtype=out_dtype)

        x_csr = x.tocsr()
        x_indices = x_csr.indices.view(np.uint32)
        x_indptr = x_csr.indptr.view(np.uint32)
        n_rows, n_cols = x_csr.shape

        for i in range(out_size):
            # Normalize row index
            row_idx = ind1[i]
            if row_idx < 0:
                row_idx += n_rows
            if row_idx < 0 or row_idx >= n_rows:
                raise IndexError("row index out of bounds")

            # Normalize column index
            col_idx = ind2[i]
            if col_idx < 0:
                col_idx += n_cols
            if col_idx < 0 or col_idx >= n_cols:
                raise IndexError("column index out of bounds")

            row_idx = np.uint32(row_idx)
            col_idx = np.uint32(col_idx)
            for data_idx in range(x_indptr[row_idx], x_indptr[row_idx + 1]):
                if x_indices[data_idx] == col_idx:
                    # Duplicate sparse entries must accumulate like scipy indexing.
                    out[i] += x_csr.data[data_idx]

        return out

    return get_item_2lists


@register_funcify_default_op_cache_key(GetItem2ListsGrad)
def numba_funcify_GetItem2ListsGrad(op, node, **kwargs):
    output_format = node.outputs[0].type.format
    out_dtype = node.outputs[0].type.dtype

    @numba_basic.numba_njit
    def get_item_2lists_grad_csr(x, ind1, ind2, gz):
        # Reproduces SciPy when running:
        # y = [csc|csr]_matrix(x.shape)
        # for i in range(len(ind1)):
        #     y[(ind1[i], ind2[i])] = gz[i]
        #
        # Note that gz is a dense vector.

        if ind1.shape != ind2.shape:
            raise ValueError("shape mismatch in row/column indices")

        n_assignments = ind1.shape[0]
        if gz.shape[0] < n_assignments:
            raise IndexError("gradient index out of bounds")

        # Vectors with normalized (non-negative) row and column indices
        norm_row = np.empty(n_assignments, dtype=np.uint32)
        norm_col = np.empty(n_assignments, dtype=np.uint32)

        n_rows, n_cols = x.shape
        # Maps original rows to values in [0, ..., touched_n_rows - 1]
        row_to_pos = np.full(n_rows, -1, dtype=np.int32)
        touched_n_rows = 0

        for i in range(n_assignments):
            # Normalize row idx
            row_idx = ind1[i]
            if row_idx < 0:
                row_idx += n_rows
            if row_idx < 0 or row_idx >= n_rows:
                raise IndexError("row index out of bounds")

            # Normalize column idx
            col_idx = ind2[i]
            if col_idx < 0:
                col_idx += n_cols
            if col_idx < 0 or col_idx >= n_cols:
                raise IndexError("column index out of bounds")

            norm_row[i] = row_idx
            norm_col[i] = col_idx

            if row_to_pos[row_idx] == -1:
                row_to_pos[row_idx] = touched_n_rows
                touched_n_rows += 1

        # Build row-wise buffers for touched rows. Repeated writes overwrite values.
        row_data = np.zeros((touched_n_rows, n_cols), dtype=out_dtype)
        row_mask = np.zeros((touched_n_rows, n_cols), dtype=np.bool_)
        row_nnz = np.zeros(touched_n_rows, dtype=np.int32)

        for i in range(n_assignments):
            row_pos = row_to_pos[norm_row[i]]
            col_idx = norm_col[i]
            if not row_mask[row_pos, col_idx]:
                row_nnz[row_pos] += 1
                row_mask[row_pos, col_idx] = True
            row_data[row_pos, col_idx] = gz[i]

        # Build output indptr.
        # For touched rows add row_nnz[row_pos] to total_nnz.
        # For untouched rows, carry forward the previous total_nnz count.
        out_indptr = np.empty(n_rows + 1, dtype=np.int32)
        out_indptr[0] = 0

        total_nnz = 0
        for row_idx in range(n_rows):
            row_pos = row_to_pos[row_idx]
            if row_pos >= 0:
                total_nnz += row_nnz[row_pos]
            out_indptr[row_idx + 1] = total_nnz

        # Build output data and indices, which need the total number of non-zero elements.
        out_data = np.empty(total_nnz, dtype=out_dtype)
        out_indices = np.empty(total_nnz, dtype=np.int32)

        # Populate indices and data by storing col_idx and value (row_data[row_pos, col_idx])
        # for touched rows/columns.
        for row_idx in range(n_rows):
            row_pos = row_to_pos[row_idx]
            if row_pos < 0:
                continue

            dst = out_indptr[row_idx]
            for col_idx in range(n_cols):
                if row_mask[row_pos, col_idx]:
                    out_indices[dst] = col_idx
                    out_data[dst] = row_data[row_pos, col_idx]
                    dst += 1

        return sp.sparse.csr_matrix(
            (out_data, out_indices, out_indptr), shape=(n_rows, n_cols)
        )

    if output_format == "csr":
        return get_item_2lists_grad_csr

    @numba_basic.numba_njit
    def get_item_2lists_grad_csc(x, ind1, ind2, gz):
        return get_item_2lists_grad_csr(x, ind1, ind2, gz).tocsc()

    return get_item_2lists_grad_csc


@register_funcify_default_op_cache_key(GetItem2d)
def numba_funcify_GetItem2d(op, node, **kwargs):
    input_format = node.inputs[0].type.format

    @numba_basic.numba_njit
    def normalize_index(idx):
        # Slice construction requires scalars or None, but we may receive a 0d array.
        return np.asarray(idx).item() if idx is not None else None

    @numba_basic.numba_njit
    def slice_indices(size, start, stop, step):
        start, stop, step = slice(
            normalize_index(start), normalize_index(stop), normalize_index(step)
        ).indices(size)
        return np.arange(start, stop, step, dtype=np.int32)

    if input_format == "csr":

        @numba_basic.numba_njit
        def get_item_2d_csr(x, start1, stop1, step1, start2, stop2, step2):
            # Reproduces SciPy when running:
            # x[start1:stop1:step1, start2:stop2:step2]
            n_rows, n_cols = x.shape

            selected_rows = slice_indices(n_rows, start1, stop1, step1)
            selected_cols = slice_indices(n_cols, start2, stop2, step2)
            out_n_rows = len(selected_rows)
            out_n_cols = len(selected_cols)

            col_map = np.full(n_cols, -1, dtype=np.int32)
            for out_col in range(out_n_cols):
                col_map[selected_cols[out_col]] = out_col

            x_indptr = x.indptr.view(np.uint32)
            x_indices = x.indices.view(np.uint32)

            out_indptr = np.empty(out_n_rows + 1, dtype=np.int32)
            out_indptr[0] = 0
            total_nnz = 0
            for out_row in range(out_n_rows):
                src_row = selected_rows[out_row]
                row_nnz = 0
                for data_idx in range(x_indptr[src_row], x_indptr[src_row + 1]):
                    src_col = x_indices[data_idx]
                    if col_map[src_col] != -1:
                        row_nnz += 1
                total_nnz += row_nnz
                out_indptr[out_row + 1] = total_nnz

            out_data = np.empty(total_nnz, dtype=x.data.dtype)
            out_indices = np.empty(total_nnz, dtype=np.int32)
            for out_row in range(out_n_rows):
                src_row = selected_rows[out_row]
                dst = out_indptr[out_row]
                for data_idx in range(x_indptr[src_row], x_indptr[src_row + 1]):
                    src_col = x_indices[data_idx]
                    out_col = col_map[src_col]
                    if out_col != -1:
                        out_data[dst] = x.data[data_idx]
                        out_indices[dst] = out_col
                        dst += 1

            return sp.sparse.csr_matrix(
                (out_data, out_indices, out_indptr), shape=(out_n_rows, out_n_cols)
            )

        return get_item_2d_csr

    @numba_basic.numba_njit
    def get_item_2d_csc(x, start1, stop1, step1, start2, stop2, step2):
        # Reproduces SciPy when running:
        # x[start1:stop1:step1, start2:stop2:step2]
        n_rows, n_cols = x.shape

        selected_rows = slice_indices(n_rows, start1, stop1, step1)
        selected_cols = slice_indices(n_cols, start2, stop2, step2)
        out_n_rows = selected_rows.shape[0]
        out_n_cols = selected_cols.shape[0]

        row_map = np.full(n_rows, -1, dtype=np.int32)
        for out_row in range(out_n_rows):
            row_map[selected_rows[out_row]] = out_row

        x_indptr = x.indptr.view(np.uint32)
        x_indices = x.indices.view(np.uint32)

        out_indptr = np.empty(out_n_cols + 1, dtype=np.int32)
        out_indptr[0] = 0
        total_nnz = 0
        for out_col in range(out_n_cols):
            src_col = selected_cols[out_col]
            col_nnz = 0
            for data_idx in range(x_indptr[src_col], x_indptr[src_col + 1]):
                src_row = x_indices[data_idx]
                if row_map[src_row] != -1:
                    col_nnz += 1
            total_nnz += col_nnz
            out_indptr[out_col + 1] = total_nnz

        out_data = np.empty(total_nnz, dtype=x.data.dtype)
        out_indices = np.empty(total_nnz, dtype=np.int32)
        for out_col in range(out_n_cols):
            src_col = selected_cols[out_col]
            dst = out_indptr[out_col]
            for data_idx in range(x_indptr[src_col], x_indptr[src_col + 1]):
                src_row = x_indices[data_idx]
                out_row = row_map[src_row]
                if out_row != -1:
                    out_data[dst] = x.data[data_idx]
                    out_indices[dst] = out_row
                    dst += 1

        return sp.sparse.csc_matrix(
            (out_data, out_indices, out_indptr), shape=(out_n_rows, out_n_cols)
        )

    return get_item_2d_csc


@register_funcify_default_op_cache_key(GetItemScalar)
def numba_funcify_GetItemScalar(op, node, **kwargs):
    input_format = node.inputs[0].type.format
    out_dtype = np.dtype(node.outputs[0].type.dtype)

    if input_format == "csr":

        @numba_basic.numba_njit
        def get_item_scalar_csr(x, ind1, ind2):
            n_rows, n_cols = x.shape

            row_idx = np.asarray(ind1).item()
            if row_idx < 0:
                row_idx += n_rows
            if row_idx < 0 or row_idx >= n_rows:
                raise IndexError("row index out of bounds")

            col_idx = np.asarray(ind2).item()
            if col_idx < 0:
                col_idx += n_cols
            if col_idx < 0 or col_idx >= n_cols:
                raise IndexError("column index out of bounds")

            row_idx = np.uint32(row_idx)
            col_idx = np.uint32(col_idx)

            indptr = x.indptr.view(np.uint32)
            indices = x.indices.view(np.uint32)

            out = 0
            for data_idx in range(indptr[row_idx], indptr[row_idx + 1]):
                # Duplicate sparse entries must accumulate like scipy indexing.
                if indices[data_idx] == col_idx:
                    out += x.data[data_idx]
            return np.asarray(out, dtype=out_dtype)

        return get_item_scalar_csr

    @numba_basic.numba_njit
    def get_item_scalar_csc(x, ind1, ind2):
        n_rows, n_cols = x.shape

        row_idx = np.asarray(ind1).item()
        if row_idx < 0:
            row_idx += n_rows
        if row_idx < 0 or row_idx >= n_rows:
            raise IndexError("row index out of bounds")

        col_idx = np.asarray(ind2).item()
        if col_idx < 0:
            col_idx += n_cols
        if col_idx < 0 or col_idx >= n_cols:
            raise IndexError("column index out of bounds")

        row_idx = np.uint32(row_idx)
        col_idx = np.uint32(col_idx)

        indptr = x.indptr.view(np.uint32)
        indices = x.indices.view(np.uint32)

        out = 0
        for data_idx in range(indptr[col_idx], indptr[col_idx + 1]):
            # Duplicate sparse entries must accumulate like scipy indexing.
            if indices[data_idx] == row_idx:
                out += x.data[data_idx]
        return np.asarray(out, dtype=out_dtype)

    return get_item_scalar_csc


@register_funcify_default_op_cache_key(Neg)
def numba_funcify_Neg(op, node, **kwargs):
    @numba_basic.numba_njit
    def neg(x):
        z = x.copy()
        z_data = z.data
        z_data *= -1
        return z

    return neg
