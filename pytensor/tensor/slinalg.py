import warnings
from collections.abc import Sequence
from functools import reduce

import numpy as np
import scipy.linalg as scipy_linalg

import pytensor
from pytensor import tensor as pt
from pytensor.graph.basic import Apply
from pytensor.graph.op import Op
from pytensor.tensor import basic as ptb
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.variable import TensorVariable
from pytensor.utils import unzip


def _largest_common_dtype(tensors: Sequence[TensorVariable]) -> np.dtype:
    return reduce(lambda l, r: np.promote_types(l, r), [x.dtype for x in tensors])


class BaseBlockDiagonal(Op):
    __props__: tuple[str, ...] = ("n_inputs",)

    def __init__(self, n_inputs):
        input_sig = ",".join(f"(m{i},n{i})" for i in range(n_inputs))
        self.gufunc_signature = f"{input_sig}->(m,n)"

        if n_inputs <= 1:
            raise ValueError("n_inputs must be greater than 1")
        self.n_inputs = n_inputs

    def pullback(self, inputs, outputs, gout):
        shapes = pt.stack([i.shape for i in inputs])
        index_end = shapes.cumsum(0)
        index_begin = index_end - shapes
        slices = [
            ptb.ix_(
                pt.arange(index_begin[i, 0], index_end[i, 0]),
                pt.arange(index_begin[i, 1], index_end[i, 1]),
            )
            for i in range(len(inputs))
        ]
        return [gout[0][slc] for slc in slices]

    def infer_shape(self, fgraph, nodes, shapes):
        first, second = unzip(shapes, n=2, strict=True)
        return [(pt.add(*first), pt.add(*second))]

    def _validate_and_prepare_inputs(self, matrices, as_tensor_func):
        if len(matrices) != self.n_inputs:
            raise ValueError(
                f"Expected {self.n_inputs} matri{'ces' if self.n_inputs > 1 else 'x'}, got {len(matrices)}"
            )
        matrices = list(map(as_tensor_func, matrices))
        if any(mat.type.ndim != 2 for mat in matrices):
            raise TypeError("All inputs must have dimension 2")
        return matrices


class BlockDiagonal(BaseBlockDiagonal):
    __props__ = ("n_inputs",)

    def make_node(self, *matrices):
        matrices = self._validate_and_prepare_inputs(matrices, pt.as_tensor)
        dtype = _largest_common_dtype(matrices)

        shapes_by_dim = tuple(zip(*(m.type.shape for m in matrices)))
        out_shape = tuple(
            [
                sum(dim_shapes)
                if not any(shape is None for shape in dim_shapes)
                else None
                for dim_shapes in shapes_by_dim
            ]
        )

        out_type = pytensor.tensor.matrix(shape=out_shape, dtype=dtype)
        return Apply(self, matrices, [out_type])

    def perform(self, node, inputs, output_storage, params=None):
        dtype = node.outputs[0].type.dtype
        output_storage[0][0] = scipy_linalg.block_diag(*inputs).astype(dtype)


def block_diag(*matrices: TensorVariable):
    """
    Construct a block diagonal matrix from a sequence of input tensors.

    Given the inputs `A`, `B` and `C`, the output will have these arrays arranged on the diagonal:

    [[A, 0, 0],
     [0, B, 0],
     [0, 0, C]]

    Parameters
    ----------
    A, B, C ... : tensors
        Input tensors to form the block diagonal matrix. last two dimensions of the inputs will be used, and all
        inputs should have at least 2 dimensins.

    Returns
    -------
    out: tensor
        The block diagonal matrix formed from the input matrices.

    Examples
    --------
    Create a block diagonal matrix from two 2x2 matrices:

    ..code-block:: python

        import numpy as np
        from pytensor.tensor.linalg import block_diag

        A = pt.as_tensor_variable(np.array([[1, 2], [3, 4]]))
        B = pt.as_tensor_variable(np.array([[5, 6], [7, 8]]))

        result = block_diagonal(A, B, name='X')
        print(result.eval())
        Out: array([[1, 2, 0, 0],
                     [3, 4, 0, 0],
                     [0, 0, 5, 6],
                     [0, 0, 7, 8]])
    """
    if len(matrices) == 1:
        return matrices[0]

    _block_diagonal_matrix = Blockwise(BlockDiagonal(n_inputs=len(matrices)))
    return _block_diagonal_matrix(*matrices)


_deprecated_names = {
    "solve_continuous_lyapunov",
    "solve_discrete_are",
    "solve_discrete_lyapunov",
}


def __getattr__(name):
    if name in _deprecated_names:
        warnings.warn(
            f"{name} has been moved from tensor/slinalg.py as part of a reorganization "
            "of linear algebra routines in Pytensor. Imports from slinalg.py will fail in Pytensor 3.0.\n"
            f"Please use the stable user-facing linalg API: from pytensor.tensor.linalg import {name}",
            DeprecationWarning,
            stacklevel=2,
        )
        from pytensor.tensor._linalg.solve import linear_control

        return getattr(linear_control, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")


# Re-exports: decomposition ops that were moved to _linalg/decomposition/
# These are kept here for backwards compatibility.
from pytensor.tensor._linalg.decomposition.cholesky import (  # noqa: E402, F401
    Cholesky,
    cholesky,
)
from pytensor.tensor._linalg.decomposition.eigen import (  # noqa: E402, F401
    Eigvalsh,
    EigvalshGrad,
    eigvalsh,
)
from pytensor.tensor._linalg.decomposition.lu import (  # noqa: E402, F401
    LU,
    LUFactor,
    PivotToPermutations,
    lu,
    lu_factor,
    pivot_to_permutation,
)
from pytensor.tensor._linalg.decomposition.qr import QR, qr  # noqa: E402, F401
from pytensor.tensor._linalg.decomposition.schur import (  # noqa: E402, F401
    QZ,
    Schur,
    ordqz,
    qz,
    schur,
)

# Re-exports: product ops that were moved to _linalg/products.py
from pytensor.tensor._linalg.products import Expm, expm  # noqa: E402, F401

# Re-exports: solve ops that were moved to _linalg/solve/
from pytensor.tensor._linalg.solve.core import (  # noqa: E402, F401
    SolveBase,
    _default_b_ndim,
)
from pytensor.tensor._linalg.solve.general import (  # noqa: E402, F401
    Solve,
    lu_solve,
    solve,
)
from pytensor.tensor._linalg.solve.psd import (  # noqa: E402, F401
    CholeskySolve,
    cho_solve,
)
from pytensor.tensor._linalg.solve.triangular import (  # noqa: E402, F401
    SolveTriangular,
    solve_triangular,
)


__all__ = [
    "block_diag",
    "cho_solve",
    "cholesky",
    "eigvalsh",
    "expm",
    "lu",
    "lu_factor",
    "lu_solve",
    "ordqz",
    "pivot_to_permutation",
    "qr",
    "qz",
    "schur",
    "solve",
    "solve_triangular",
]
