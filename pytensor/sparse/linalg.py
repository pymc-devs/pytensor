from typing import Literal

import scipy.sparse

from pytensor.graph import Apply
from pytensor.sparse import as_sparse_or_tensor_variable, matrix
from pytensor.tensor import TensorVariable
from pytensor.tensor.slinalg import BaseBlockDiagonal, _largest_common_dtype


class SparseBlockDiagonal(BaseBlockDiagonal):
    __props__ = (
        "n_inputs",
        "format",
    )

    def __init__(self, n_inputs: int, format: Literal["csc", "csr"] = "csc"):
        super().__init__(n_inputs)
        self.format = format

    def make_node(self, *matrices):
        matrices = self._validate_and_prepare_inputs(
            matrices, as_sparse_or_tensor_variable
        )
        dtype = _largest_common_dtype(matrices)
        out_type = matrix(format=self.format, dtype=dtype)

        return Apply(self, matrices, [out_type])

    def perform(self, node, inputs, output_storage, params=None):
        dtype = node.outputs[0].type.dtype
        output_storage[0][0] = scipy.sparse.block_diag(
            inputs, format=self.format
        ).astype(dtype)


def block_diag(*matrices: TensorVariable, format: Literal["csc", "csr"] = "csc"):
    r"""
    Construct a block diagonal matrix from a sequence of input matrices.

    Given the inputs `A`, `B` and `C`, the output will have these arrays arranged on the diagonal:

    [[A, 0, 0],
     [0, B, 0],
     [0, 0, C]]

    Parameters
    ----------
    A, B, C ... : tensors
        Input tensors to form the block diagonal matrix. last two dimensions of the inputs will be used, and all
        inputs should have at least 2 dimensins.

        Note that the input matrices need not be sparse themselves, and will be automatically converted to the
        requested format if they are not.

    format: str, optional
        The format of the output sparse matrix. One of 'csr' or 'csc'. Default is 'csr'. Ignored if sparse=False.

    Returns
    -------
    out: sparse matrix tensor
        Symbolic sparse matrix in the specified format.

    Examples
    --------
    Create a sparse block diagonal matrix from two sparse 2x2 matrices:

    .. testcode::
        import numpy as np
        from pytensor.sparse.linalg import block_diag
        from scipy.sparse import csr_matrix

        A = csr_matrix([[1, 2], [3, 4]])
        B = csr_matrix([[5, 6], [7, 8]])
        result_sparse = block_diag(A, B, format='csr')

        print(result_sparse)
        print(result_sparse.toarray().eval())

    .. testoutput::

        SparseVariable{csr,int64}
        [[1 2 0 0]
         [3 4 0 0]
         [0 0 5 6]
         [0 0 7 8]]

    """
    if len(matrices) == 1:
        return matrices

    _sparse_block_diagonal = SparseBlockDiagonal(n_inputs=len(matrices), format=format)
    return _sparse_block_diagonal(*matrices)
